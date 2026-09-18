//! Complete construction policy for OpenAI-compatible chat models.

use std::sync::Arc;

use serde_json::json;

use crate::model::ChatModel;

use super::{AuthStyle, OpenAiModel};

/// Resolved configuration for an OpenAI-compatible provider.
#[derive(Clone)]
pub struct OpenAiConfig<'a> {
    /// Provider family identifier used in profiles and normalized errors.
    pub provider_name: &'a str,
    /// Base URL for the provider.
    pub endpoint: &'a str,
    /// API credential; it may be empty when no authentication is used.
    pub api_key: &'a str,
    /// How the credential is attached to requests.
    pub auth_style: AuthStyle,
    /// Default model identifier.
    pub model: &'a str,
    /// Model-id glob patterns whose targets reject temperature.
    pub temperature_unsupported_models: &'a [String],
    /// Fixed temperature override for every call.
    pub temperature_override: Option<f64>,
    /// Whether system messages must be folded into the first user message.
    pub merge_system_into_user: bool,
    /// Static headers attached to every request.
    pub extra_headers: &'a [(String, String)],
    /// Override the advertised native tool-calling capability.
    pub native_tool_calling: Option<bool>,
    /// Override the advertised vision capability.
    pub vision: Option<bool>,
    /// Provider options merged underneath per-request options.
    pub default_provider_options: Option<serde_json::Value>,
    /// Route calls through the Responses API instead of Chat Completions.
    pub responses_api_primary: bool,
    /// Omit max_output_tokens on Responses API requests.
    pub responses_omit_max_output_tokens: bool,
    /// Static query parameters appended to every request URL.
    pub extra_query_params: &'a [(String, String)],
    /// Optional User-Agent override.
    pub user_agent: Option<&'a str>,
    /// Emit explicit prompt-cache breakpoints.
    pub explicit_cache_control: bool,
}

impl std::fmt::Debug for OpenAiConfig<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header_names = self
            .extra_headers
            .iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        formatter
            .debug_struct("OpenAiConfig")
            .field("provider_name", &self.provider_name)
            .field("endpoint", &self.endpoint)
            .field("api_key", &"[REDACTED]")
            .field("auth_style", &self.auth_style)
            .field("model", &self.model)
            .field(
                "temperature_unsupported_models",
                &self.temperature_unsupported_models,
            )
            .field("temperature_override", &self.temperature_override)
            .field("merge_system_into_user", &self.merge_system_into_user)
            .field("extra_header_names", &header_names)
            .field("native_tool_calling", &self.native_tool_calling)
            .field("vision", &self.vision)
            .field("responses_api_primary", &self.responses_api_primary)
            .field(
                "responses_omit_max_output_tokens",
                &self.responses_omit_max_output_tokens,
            )
            .field("extra_query_params", &self.extra_query_params)
            .field("user_agent", &self.user_agent)
            .field("explicit_cache_control", &self.explicit_cache_control)
            .finish_non_exhaustive()
    }
}

/// Returns whether an endpoint is OpenRouter's first-party API.
pub fn endpoint_is_openrouter(endpoint: &str) -> bool {
    reqwest::Url::parse(endpoint.trim())
        .ok()
        .and_then(|url| {
            url.host_str()
                .map(|host| host.eq_ignore_ascii_case("openrouter.ai"))
        })
        .unwrap_or(false)
}

/// Builds an OpenAI-compatible chat model from fully resolved configuration.
pub fn build_openai_model(config: OpenAiConfig<'_>) -> Arc<dyn ChatModel<()>> {
    let mut model = OpenAiModel::compatible_provider(
        config.provider_name,
        config.api_key,
        config.endpoint,
        config.model,
    )
    .with_auth_style(config.auth_style);

    if !config.temperature_unsupported_models.is_empty() {
        model = model
            .with_temperature_unsupported_models(config.temperature_unsupported_models.to_vec());
    }
    if config.temperature_override.is_some() {
        model = model.with_temperature_override(config.temperature_override);
    }
    if config.merge_system_into_user {
        model = model.with_merge_system_into_user();
    }
    for (name, value) in config.extra_headers {
        model = model.with_header(name.clone(), value.clone());
    }
    if let Some(enabled) = config.native_tool_calling {
        model = model.with_native_tool_calling(enabled);
    }
    if let Some(enabled) = config.vision {
        model = model.with_vision(enabled);
    }
    if let Some(options) = config.default_provider_options {
        model = model.with_default_provider_options(options);
    }
    for (name, value) in config.extra_query_params {
        model = model.with_extra_query_param(name.clone(), value.clone());
    }
    if let Some(user_agent) = config.user_agent {
        model = model.with_user_agent(user_agent);
    }
    if config.explicit_cache_control {
        model = model.with_explicit_cache_control(true);
    }
    if config.responses_api_primary {
        model = model.with_responses_api_primary();
    }
    if config.responses_omit_max_output_tokens {
        model = model.with_responses_omit_max_output_tokens();
    }

    Arc::new(model)
}

/// Builds a conventional hosted OpenAI-compatible model.
#[allow(clippy::too_many_arguments)]
pub fn build_openai_chat_model(
    provider_name: &str,
    endpoint: &str,
    api_key: &str,
    auth_style: AuthStyle,
    model: &str,
    temperature_unsupported_models: &[String],
    temperature_override: Option<f64>,
    merge_system_into_user: bool,
) -> Arc<dyn ChatModel<()>> {
    build_openai_model(OpenAiConfig {
        provider_name,
        endpoint,
        api_key,
        auth_style,
        model,
        temperature_unsupported_models,
        temperature_override,
        merge_system_into_user,
        extra_headers: &[],
        native_tool_calling: None,
        vision: None,
        default_provider_options: None,
        responses_api_primary: false,
        responses_omit_max_output_tokens: false,
        extra_query_params: &[],
        user_agent: None,
        explicit_cache_control: false,
    })
}

/// Builds a local OpenAI-compatible model with conservative capabilities.
#[allow(clippy::too_many_arguments)]
pub fn build_local_runtime_chat_model(
    provider_name: &str,
    endpoint: &str,
    api_key: &str,
    auth_style: AuthStyle,
    model: &str,
    temperature_unsupported_models: &[String],
    temperature_override: Option<f64>,
    num_ctx: Option<u32>,
) -> Arc<dyn ChatModel<()>> {
    let default_provider_options = num_ctx.map(|value| {
        json!({
            "options": { "num_ctx": value }
        })
    });
    build_openai_model(OpenAiConfig {
        provider_name,
        endpoint,
        api_key,
        auth_style,
        model,
        temperature_unsupported_models,
        temperature_override,
        merge_system_into_user: false,
        extra_headers: &[],
        native_tool_calling: Some(false),
        vision: Some(false),
        default_provider_options,
        responses_api_primary: false,
        responses_omit_max_output_tokens: false,
        extra_query_params: &[],
        user_agent: None,
        explicit_cache_control: false,
    })
}
