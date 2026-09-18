//! LM Studio OpenAI-compatible HTTP types and helpers.
//!
//! LM Studio exposes an OpenAI-compatible API under `http://localhost:1234/v1`
//! by default. This module keeps the wire contract separate from the Ollama
//! native API structs so the two providers can evolve independently.

#![allow(missing_docs, reason = "wire DTO fields mirror LM Studio payloads")]

use serde::{Deserialize, Serialize};

pub const DEFAULT_LM_STUDIO_BASE_URL: &str = "http://localhost:1234/v1";

/// Resolves the effective LM Studio base URL.
pub fn lm_studio_base_url(configured: Option<&str>) -> String {
    for (source, candidate) in [
        (
            "OPENHUMAN_LM_STUDIO_BASE_URL",
            std::env::var("OPENHUMAN_LM_STUDIO_BASE_URL").ok(),
        ),
        (
            "LM_STUDIO_BASE_URL",
            std::env::var("LM_STUDIO_BASE_URL").ok(),
        ),
        ("configured", configured.map(str::to_owned)),
    ] {
        let Some(raw) = candidate else {
            tracing::trace!(source, "[lm-studio] base URL candidate missing");
            continue;
        };
        tracing::trace!(
            source,
            raw = %redact_url(&raw),
            "[lm-studio] inspecting base URL candidate"
        );
        if let Some(normalized) = normalize_lm_studio_base_url(&raw) {
            tracing::debug!(
                source,
                base_url = %redact_url(&normalized),
                "[lm-studio] selected normalized base URL"
            );
            return normalized;
        }
        tracing::trace!(source, "[lm-studio] rejected blank base URL candidate");
    }

    tracing::debug!(
        base_url = %DEFAULT_LM_STUDIO_BASE_URL,
        "[lm-studio] using default base URL"
    );
    DEFAULT_LM_STUDIO_BASE_URL.to_string()
}

pub fn normalize_lm_studio_base_url(raw: &str) -> Option<String> {
    let trimmed = raw.trim().trim_end_matches('/');
    tracing::trace!(
        raw = %redact_url(raw),
        trimmed = %redact_url(trimmed),
        "[lm-studio] normalizing base URL"
    );
    if trimmed.is_empty() {
        tracing::trace!("[lm-studio] base URL normalization rejected blank input");
        return None;
    }

    let with_scheme = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    tracing::trace!(
        with_scheme = %redact_url(&with_scheme),
        "[lm-studio] base URL scheme normalized"
    );

    let without_known_endpoint = with_scheme
        .trim_end_matches("/chat/completions")
        .trim_end_matches("/models")
        .trim_end_matches('/')
        .to_string();
    tracing::trace!(
        without_known_endpoint = %redact_url(&without_known_endpoint),
        "[lm-studio] base URL endpoint suffix normalized"
    );

    if without_known_endpoint.ends_with("/v1") {
        tracing::trace!(
            appended_v1 = false,
            base_url = %redact_url(&without_known_endpoint),
            "[lm-studio] base URL normalization complete"
        );
        Some(without_known_endpoint)
    } else {
        let normalized = format!("{without_known_endpoint}/v1");
        tracing::trace!(
            appended_v1 = true,
            base_url = %redact_url(&normalized),
            "[lm-studio] base URL normalization complete"
        );
        Some(normalized)
    }
}

pub fn apply_lm_studio_auth(
    request: reqwest::RequestBuilder,
    api_key: Option<&str>,
) -> reqwest::RequestBuilder {
    match api_key.map(str::trim) {
        Some(key) if !key.is_empty() => {
            tracing::trace!(
                api_key_present = true,
                api_key_len = key.len(),
                "[lm-studio] auth applied"
            );
            request.bearer_auth(key)
        }
        _ => {
            tracing::trace!(api_key_present = false, "[lm-studio] auth skipped");
            request
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct LmStudioModelsResponse {
    #[serde(default)]
    pub data: Vec<LmStudioModel>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LmStudioModel {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub owned_by: Option<String>,
}

/// LM Studio **native** REST (`GET /api/v0/models`) model entry.
///
/// Unlike the OpenAI-compatible `/v1/models` (which returns only
/// `id`/`object`/`owned_by`), the native API reports the model's context
/// window — the value the agent harness must budget against to avoid an
/// `n_ctx` overflow when the user loaded the model with a small context
/// (issue #3550 / Sentry TAURI-RUST-6V0).
#[derive(Debug, Clone, Deserialize)]
pub struct LmStudioNativeModel {
    pub id: String,
    /// Context window the model is *currently loaded* with — the runtime's
    /// hard limit. Authoritative for budgeting. (LM Studio also returns a
    /// `state` field, which we ignore — we prefer the loaded window whenever
    /// present regardless of load state.)
    #[serde(default)]
    pub loaded_context_length: Option<u64>,
    /// Model's declared maximum context. Fallback when not currently loaded.
    #[serde(default)]
    pub max_context_length: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct LmStudioNativeModelsResponse {
    #[serde(default)]
    pub data: Vec<LmStudioNativeModel>,
}

/// Map a normalized `…/v1` base URL to the LM Studio native models endpoint
/// `…/api/v0/models` (a sibling of `/v1`, served at the host root).
pub fn lm_studio_native_models_url(v1_base_url: &str) -> String {
    format!("{}/api/v0/models", host_root_of(v1_base_url))
}

/// Strip a trailing `/v1` so sibling endpoints served at the host root can be
/// derived from an OpenAI-compatible base URL.
pub fn host_root_of(v1_base_url: &str) -> &str {
    v1_base_url
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .trim_end_matches('/')
}

/// Ollama-native `GET /api/tags` URL derived from an OpenAI-compatible base.
///
/// Only used as the one-shot 404 fallback in
/// [`LocalAiService::list_lm_studio_models`](crate::inference::local::service::LocalAiService):
/// some runtimes are reachable on an OpenAI-shaped base URL but expose only the
/// Ollama listing (e.g. plain Ollama configured with a `/v1` base). Discovery is
/// still chosen by provider type first — this is a recovery path, not a probe
/// order (GH #5055).
pub fn ollama_tags_fallback_url(v1_base_url: &str) -> String {
    format!("{}/api/tags", host_root_of(v1_base_url))
}

/// Resolve the context window LM Studio reports for `model_id` from a native
/// `/api/v0/models` payload: prefer the *loaded* context (the limit the
/// runtime actually enforces), else the model's declared maximum. Zero/absent
/// values are treated as unknown. Returns `None` when the model isn't present
/// or reports no usable window.
pub fn lm_studio_context_window_for(
    resp: &LmStudioNativeModelsResponse,
    model_id: &str,
) -> Option<u64> {
    resp.data.iter().find(|m| m.id == model_id).and_then(|m| {
        m.loaded_context_length
            .filter(|&v| v > 0)
            .or(m.max_context_length.filter(|&v| v > 0))
    })
}

#[cfg(test)]
#[path = "lm_studio_test.rs"]
mod tests;

fn redact_url(raw: &str) -> String {
    reqwest::Url::parse(raw)
        .map(|mut url| {
            let _ = url.set_username("");
            let _ = url.set_password(None);
            url.set_query(None);
            url.set_fragment(None);
            url.to_string()
        })
        .unwrap_or_else(|_| "<invalid-endpoint>".to_string())
}
