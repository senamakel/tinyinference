//! Model limits reported by the provider.
//!
//! A context window guessed from a model id is wrong as soon as a provider
//! ships a bigger (or smaller) variant under a familiar name:
//! `deepseek/deepseek-v4.1-flash` serves about 1M tokens, while the generic
//! `deepseek` hint in [`super::context_window_for_model_id`] says 128k. This
//! module asks the provider instead:
//!
//! - [`discover_model_limits`] reads the provider's model listing (OpenRouter
//!   `/api/v1/models`, OpenAI-compatible `/models`, vLLM's `max_model_len`, and
//!   OpenRouter's per-endpoint limits when routing pins a provider), bounded by
//!   a timeout and cached per `(endpoint, model)` with a TTL.
//! - [`record_overflow_error`] learns the window a provider states in a
//!   context-overflow error ("maximum context length is 131072 tokens") and
//!   stores it as a correction for the same key. The OpenAI-compatible and
//!   Anthropic adapters call it on every overflow.
//!
//! Hosts resolve a window in this order: their explicit override, then the
//! provider-sourced value from here, and only then a static table. The static
//! id table never overrides a value from this module.

mod cache;
mod fetch;
mod parse;
mod types;

pub use cache::{
    CachedLimits, DEFAULT_DISCOVERED_TTL, DEFAULT_LEARNED_TTL, DEFAULT_NEGATIVE_TTL,
    ModelLimitsCache, model_limits_cache,
};
pub use fetch::{
    ModelListingFetcher, ReqwestListingFetcher, discover_model_limits, discover_model_limits_with,
};
pub use parse::{
    input_modalities_from_entry, limits_from_entry, model_ids_match, parse_listing_limits,
    parse_model_limits, parse_ollama_show, parse_openrouter_endpoint_limits,
    pinned_openrouter_providers,
};
pub use types::{DiscoveryRequest, LimitSource, ModelLimits, looks_like_ollama};

/// The provider-sourced limits already cached for `(endpoint, model)`, without
/// any request: a fresh default-variant discovery, lowered by a learned overflow
/// window (discoveries made under a non-default listing URL or pinned-provider
/// set are cached per request variant and read through `get_variant`).
#[must_use]
pub fn cached_model_limits(endpoint: &str, model: &str) -> Option<ModelLimits> {
    model_limits_cache().get(endpoint, model).effective()
}

/// Learns the context window from an overflow error into `cache`.
///
/// Returns the window recorded, or `None` when `message` states no limit.
pub fn record_overflow_error_in(
    cache: &ModelLimitsCache,
    endpoint: &str,
    model: &str,
    message: &str,
) -> Option<u64> {
    let window = crate::failure::parse_context_limit_from_error(message)?;
    cache.insert_learned(endpoint, model, window);
    tracing::info!(
        endpoint,
        model,
        context_window = window,
        "[model_limits] learned context window from provider overflow error"
    );
    Some(window)
}

/// Learns the context window a provider stated in an overflow error for the
/// model served from `endpoint`, into the process-wide cache.
///
/// `endpoint` must be the same base URL discovery uses for this model (the
/// adapter's base URL), so the correction lands on the same key.
pub fn record_overflow_error(endpoint: &str, model: &str, message: &str) -> Option<u64> {
    record_overflow_error_in(model_limits_cache(), endpoint, model, message)
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
