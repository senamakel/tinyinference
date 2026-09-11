//! Anthropic Messages API provider with explicit prompt-cache breakpoints.
//!
//! Unlike OpenAI-compatible APIs, Anthropic enables prompt caching by attaching
//! `{"type":"ephemeral"}` as `cache_control` to a tool, system, or content
//! block. This adapter turns TinyInference's cacheable prompt segments into that
//! wire shape and maps the provider's cache usage counters back into [`Usage`].
//!
//! # Why the native adapter matters for caching
//!
//! Anthropic's OpenAI-compatible endpoint documents prompt caching as
//! **unsupported** — a Claude model reached through `OpenAiModel` pays full
//! price for its whole prefix on every call. So any host that wants cache hits
//! on Claude has to speak the Messages API, which means this adapter has to be a
//! complete one: tools, tool results, images, signed thinking blocks, and
//! streaming, not text-in/text-out.
//!
//! # Breakpoint placement
//!
//! Anthropic caches the prefix up to each `cache_control` marker and, for the
//! last marker, also looks back roughly twenty blocks for an earlier hit. The
//! request therefore carries up to three of the four allowed markers, in wire
//! order:
//!
//! 1. the last **tool** declaration — tools precede the system prompt on the
//!    wire, so a stable tool set is its own reusable prefix;
//! 2. the last **system** block;
//! 3. the last content block of the **final message** — this is what makes a
//!    growing conversation cache *incrementally*: iteration `n+1` of an agent
//!    loop reuses everything iteration `n` wrote, and only the newest tool
//!    result is billed as a cache write.
//!
//! Without the third marker only the system prefix is ever reused, and in a
//! long tool loop the transcript — not the system prompt — is most of the bill.
//!
//! Breakpoints are emitted when the request declares at least one cacheable
//! [`PromptSegment`](crate::model::PromptSegment) and the request's
//! [`CachePolicy`](crate::cache::CachePolicy), when it carries one, does not
//! opt out via `protect_prompt_prefix = false`. A request without a policy
//! follows its segments: declaring a cacheable prefix *is* the opt-in.

mod request;
mod response;
mod stream;

#[cfg(test)]
mod test;

use async_trait::async_trait;
use serde_json::Value;
use std::time::Duration;

use crate::model::{
    ChatModel, Modalities, ModelProfile, ModelRequest, ModelResponse, ModelStream, ProviderError,
};
use crate::{Error, Result};

pub(crate) use request::request_body;
pub(crate) use response::parse_response;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const DEFAULT_MODEL: &str = "claude-sonnet-4-6";
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Anthropic requires `max_tokens`; this is the ceiling used when the request
/// does not set one. Sized for an agent turn that has to fit a tool call plus
/// prose — the previous 1,024 truncated real tool calls mid-argument.
const DEFAULT_MAX_TOKENS: u32 = 4096;
const PROVIDER: &str = "anthropic";

/// A chat model backed by Anthropic's native Messages API.
pub struct AnthropicModel {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    profile: ModelProfile,
}

impl std::fmt::Debug for AnthropicModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnthropicModel")
            .field("client", &self.client)
            .field("api_key", &"[redacted]")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("profile", &self.profile)
            .finish()
    }
}

impl AnthropicModel {
    /// Creates an Anthropic model using the default Messages API endpoint.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Creates an Anthropic model targeting a Messages-API-compatible endpoint.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let model = DEFAULT_MODEL.to_string();
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            profile: ModelProfile {
                provider: Some(PROVIDER.to_string()),
                model: Some(model.clone()),
                modalities: Modalities {
                    text_in: true,
                    text_out: true,
                    image_in: true,
                    ..Modalities::default()
                },
                tool_calling: true,
                parallel_tool_calls: true,
                streaming: true,
                streaming_tool_chunks: true,
                reasoning: true,
                ..ModelProfile::default()
            },
            model,
        }
    }

    /// Overrides the default model id used when a request does not specify one.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self.profile.model = Some(self.model.clone());
        self
    }

    /// Replaces the HTTP client, so a host can supply its own transport
    /// (platform TLS, proxies, default headers, timeouts).
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Reads `ANTHROPIC_API_KEY`, plus optional `ANTHROPIC_BASE_URL` and
    /// `ANTHROPIC_MODEL`, from the environment.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Model`] when `ANTHROPIC_API_KEY` is not set.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| Error::Model("ANTHROPIC_API_KEY is not set".to_string()))?;
        let base_url =
            std::env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
        Ok(Self::with_base_url(key, base_url).with_model(model))
    }

    fn endpoint(&self) -> String {
        if self.base_url.ends_with("/messages") {
            self.base_url.clone()
        } else {
            format!("{}/messages", self.base_url)
        }
    }

    fn request_model<'a>(&'a self, request: &'a ModelRequest) -> &'a str {
        request.model.as_deref().unwrap_or(&self.model)
    }

    async fn post(&self, request: &ModelRequest, streaming: bool) -> Result<reqwest::Response> {
        let mut body = request_body(request, &self.model);
        if streaming {
            body["stream"] = Value::Bool(true);
        }
        let request_builder = self
            .client
            .post(self.endpoint())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body);
        let request_builder = match request.timeout_ms {
            Some(timeout_ms) => request_builder.timeout(Duration::from_millis(timeout_ms)),
            None => request_builder,
        };
        let response = request_builder.send().await.map_err(|error| {
            Error::Provider(Box::new(self.provider_error(
                request,
                format!("anthropic request failed: {error}"),
                None,
                None,
                None,
                None,
            )))
        })?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let text = response.text().await.unwrap_or_default();
        Err(Error::Provider(Box::new(self.parse_error_body(
            request,
            status.as_u16(),
            &text,
            retry_after.as_deref(),
        ))))
    }

    fn provider_error(
        &self,
        request: &ModelRequest,
        message: String,
        status: Option<u16>,
        code: Option<String>,
        raw: Option<Value>,
        retry_after: Option<&str>,
    ) -> ProviderError {
        let retryable =
            crate::failure::classify_provider_failure(status, code.as_deref(), &message)
                .is_retryable();
        ProviderError {
            provider: PROVIDER.to_string(),
            model: Some(self.request_model(request).to_string()),
            status,
            code,
            message,
            retryable,
            retry_after_ms: crate::embeddings::parse_retry_after_ms(retry_after),
            raw,
        }
    }

    fn parse_error_body(
        &self,
        request: &ModelRequest,
        status: u16,
        text: &str,
        retry_after: Option<&str>,
    ) -> ProviderError {
        let raw = serde_json::from_str::<Value>(text).ok();
        let error = raw.as_ref().and_then(|value| value.get("error"));
        let message = error
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .filter(|message| !message.trim().is_empty())
            .unwrap_or(text)
            .to_string();
        let code = error
            .and_then(|error| error.get("type"))
            .and_then(Value::as_str)
            .map(str::to_string);
        self.provider_error(
            request,
            format!("anthropic returned HTTP {status}: {message}"),
            Some(status),
            code,
            raw,
            retry_after,
        )
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for AnthropicModel {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    fn cache_identity(&self) -> Option<String> {
        Some(format!("anthropic:{}:{}", self.base_url, self.model))
    }

    async fn invoke(&self, _state: &State, request: ModelRequest) -> Result<ModelResponse> {
        let response = self.post(&request, false).await?;
        let body: Value = response
            .json()
            .await
            .map_err(|error| Error::Model(format!("anthropic response was not JSON: {error}")))?;
        parse_response(body)
    }

    async fn stream(&self, _state: &State, request: ModelRequest) -> Result<ModelStream> {
        let response = self.post(&request, true).await?;
        let model = self.request_model(&request).to_string();
        Ok(stream::into_model_stream(response, model))
    }
}
