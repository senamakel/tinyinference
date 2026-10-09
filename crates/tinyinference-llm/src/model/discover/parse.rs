//! Pure parsers for provider model-listing payloads.
//!
//! Shapes covered:
//!
//! - **OpenRouter** `GET /api/v1/models`: `data[]` with `id`, `context_length`,
//!   `top_provider.context_length` and `top_provider.max_completion_tokens`.
//! - **OpenRouter** `GET /api/v1/models/{id}/endpoints`: `data.endpoints[]`, one
//!   per serving provider, each with its own `context_length`,
//!   `max_completion_tokens` and `provider_name`.
//! - **OpenAI-compatible** `GET /models` and `GET /models/{id}`: `data[]` (or a
//!   bare object) with whichever of `context_window`, `context_length`,
//!   `max_context_length`, `max_model_len` (vLLM) or `max_input_tokens` the
//!   server publishes.

use serde_json::Value;

use super::types::{LimitSource, ModelLimits};

/// Keys a listing entry may use for the context window, most specific first.
const CONTEXT_KEYS: &[&str] = &[
    "context_length",
    "context_window",
    "max_context_length",
    "max_context_window",
    "max_model_len",
    "max_input_tokens",
];

/// Keys a listing entry may use for the per-response output cap.
const OUTPUT_KEYS: &[&str] = &[
    "max_completion_tokens",
    "max_output_tokens",
    "max_output_length",
];

/// Reads a positive integer, accepting numeric strings (some proxies quote
/// numbers).
fn positive_u64(value: &Value) -> Option<u64> {
    let number = match value {
        Value::Number(number) => number
            .as_u64()
            .or_else(|| number.as_f64().filter(|f| f.is_finite()).map(|f| f as u64)),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }?;
    (number > 0).then_some(number)
}

fn first_key(item: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| item.get(*key).and_then(positive_u64))
}

/// The id a listing entry advertises.
fn entry_id(item: &Value) -> Option<&str> {
    ["id", "slug", "name", "model"]
        .into_iter()
        .filter_map(|key| item.get(key).and_then(Value::as_str))
        .map(str::trim)
        .find(|id| !id.is_empty())
}

/// Reads advertised input modalities from OpenRouter, models.dev, or managed catalogs.
/// Missing or malformed metadata remains unknown rather than asserting support.
#[must_use]
pub fn input_modalities_from_entry(item: &Value) -> Option<Vec<String>> {
    let values = item
        .get("input_modalities")
        .or_else(|| item.pointer("/architecture/input_modalities"))
        .or_else(|| item.pointer("/modalities/input"))?
        .as_array()?;
    values
        .iter()
        .map(|value| value.as_str().map(|s| s.trim().to_ascii_lowercase()))
        .collect()
}

/// Limits one listing entry advertises, or `None` when it advertises neither a
/// window nor an output cap.
///
/// The model-level `context_length` wins over `top_provider.context_length`:
/// with fallbacks enabled OpenRouter only routes a request to endpoints that
/// fit it, so the model maximum is the usable window.
#[must_use]
pub fn limits_from_entry(item: &Value) -> Option<ModelLimits> {
    let top_provider = item.get("top_provider");
    let context_window = first_key(item, CONTEXT_KEYS)
        .or_else(|| item.pointer("/limit/context").and_then(positive_u64))
        .or_else(|| {
            top_provider.and_then(|top| first_key(top, &["context_length", "context_window"]))
        });
    let max_output_tokens = top_provider
        .and_then(|top| first_key(top, OUTPUT_KEYS))
        .or_else(|| first_key(item, OUTPUT_KEYS))
        .or_else(|| item.pointer("/limit/output").and_then(positive_u64));
    let limits = ModelLimits {
        context_window,
        max_output_tokens,
        input_modalities: input_modalities_from_entry(item),
        source: LimitSource::ProviderListing,
    };
    (!limits.is_empty()).then_some(limits)
}

/// The window an Ollama `POST /api/show` body reports.
///
/// A `num_ctx` in the model's `parameters` text (the Modelfile setting the
/// server loads it with) is the real ceiling for a request, so it wins; else
/// the architecture's `*.context_length` from `model_info`. Overstating the
/// window would stop compaction from firing, so with several candidates the
/// smallest is used.
#[must_use]
pub fn parse_ollama_show(body: &Value) -> Option<ModelLimits> {
    let num_ctx = body
        .get("parameters")
        .and_then(Value::as_str)
        .and_then(|parameters| {
            parameters.lines().find_map(|line| {
                let mut parts = line.split_whitespace();
                (parts.next() == Some("num_ctx"))
                    .then(|| parts.next().and_then(|value| value.parse::<u64>().ok()))
                    .flatten()
            })
        })
        .filter(|value| *value > 0);
    let architecture = body
        .get("model_info")
        .and_then(Value::as_object)
        .and_then(|info| {
            info.iter()
                .filter(|(key, _)| {
                    key.ends_with(".context_length") || key.as_str() == "context_length"
                })
                .filter_map(|(_, value)| positive_u64(value))
                .min()
        });
    let context_window = num_ctx.or(architecture)?;
    Some(ModelLimits {
        context_window: Some(context_window),
        max_output_tokens: None,
        input_modalities: None,
        source: LimitSource::NativeApi,
    })
}

/// Strips one leading routing segment (`openrouter/deepseek/x` → `deepseek/x`).
fn without_route_prefix(id: &str) -> Option<&str> {
    id.split_once('/')
        .map(|(_, rest)| rest)
        .filter(|rest| !rest.is_empty())
}

/// Whether a listed model id names the requested one.
///
/// Matches case-insensitively, and tolerates one leading routing segment on
/// either side, so `openrouter/deepseek/deepseek-v4-flash` matches the listed
/// `deepseek/deepseek-v4-flash`.
#[must_use]
pub fn model_ids_match(listed: &str, requested: &str) -> bool {
    let listed = listed.trim().to_ascii_lowercase();
    let requested = requested.trim().to_ascii_lowercase();
    if listed.is_empty() || requested.is_empty() {
        return false;
    }
    listed == requested
        || without_route_prefix(&requested) == Some(listed.as_str())
        || without_route_prefix(&listed) == Some(requested.as_str())
}

/// The entries array of a listing envelope (`data` or `models`).
fn listing_entries(body: &Value) -> Option<&Vec<Value>> {
    body.get("data")
        .and_then(Value::as_array)
        .or_else(|| body.get("models").and_then(Value::as_array))
        .or_else(|| body.as_array())
}

/// Every `(id, limits)` pair a listing advertises. Entries without an id or
/// without any limit are skipped.
#[must_use]
pub fn parse_listing_limits(body: &Value) -> Vec<(String, ModelLimits)> {
    listing_entries(body)
        .map(|entries| {
            entries
                .iter()
                .filter_map(|item| {
                    let id = entry_id(item)?;
                    Some((id.to_string(), limits_from_entry(item)?))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The limits `model` has in a listing (`{"data": [...]}`) or a single-model
/// record (`{"id": ..., ...}` or `{"data": {...}}`).
///
/// An exact id match wins over a prefix-tolerant one, so a listing that
/// carries both `deepseek/x` and `x` resolves each to itself.
#[must_use]
pub fn parse_model_limits(body: &Value, model: &str) -> Option<ModelLimits> {
    if let Some(entries) = listing_entries(body) {
        let exact = entries
            .iter()
            .find(|item| entry_id(item).is_some_and(|id| id.eq_ignore_ascii_case(model.trim())));
        let item = exact.or_else(|| {
            entries
                .iter()
                .find(|item| entry_id(item).is_some_and(|id| model_ids_match(id, model)))
        })?;
        return limits_from_entry(item);
    }
    let item = body
        .get("data")
        .filter(|data| data.is_object())
        .unwrap_or(body);
    if !item.is_object() {
        return None;
    }
    match entry_id(item) {
        Some(id) if !model_ids_match(id, model) => None,
        _ => limits_from_entry(item),
    }
}

fn normalize_provider(name: &str) -> String {
    name.chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_ascii_lowercase()
}

/// Whether an OpenRouter endpoint record belongs to `provider` (matched against
/// `provider_name`, `name` and the `tag` prefix, ignoring case and punctuation).
fn endpoint_matches_provider(endpoint: &Value, provider: &str) -> bool {
    let wanted = normalize_provider(provider);
    if wanted.is_empty() {
        return false;
    }
    let by_name = ["provider_name", "name"]
        .into_iter()
        .filter_map(|key| endpoint.get(key).and_then(Value::as_str))
        .any(|name| normalize_provider(name) == wanted);
    let by_tag = endpoint
        .get("tag")
        .and_then(Value::as_str)
        .map(|tag| tag.split('/').next().unwrap_or(tag))
        .is_some_and(|tag| normalize_provider(tag) == wanted);
    by_name || by_tag
}

/// The limits of the OpenRouter endpoints serving `providers`, from a
/// `/models/{id}/endpoints` payload.
///
/// When several pinned providers match, the result is the smallest window and
/// smallest output cap among them: routing may land on any of them, so only
/// the minimum is safe. Returns `None` when no endpoint matches.
#[must_use]
pub fn parse_openrouter_endpoint_limits(body: &Value, providers: &[String]) -> Option<ModelLimits> {
    let endpoints = body
        .get("data")
        .and_then(|data| data.get("endpoints"))
        .or_else(|| body.get("endpoints"))
        .and_then(Value::as_array)?;
    let matched: Vec<&Value> = endpoints
        .iter()
        .filter(|endpoint| {
            providers
                .iter()
                .any(|provider| endpoint_matches_provider(endpoint, provider))
        })
        .collect();
    let first = matched.first()?;
    let provider = first
        .get("provider_name")
        .or_else(|| first.get("name"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let context_window = matched
        .iter()
        .filter_map(|endpoint| first_key(endpoint, &["context_length", "context_window"]))
        .min();
    let max_output_tokens = matched
        .iter()
        .filter_map(|endpoint| first_key(endpoint, OUTPUT_KEYS))
        .min();
    let limits = ModelLimits {
        context_window,
        max_output_tokens,
        input_modalities: matched
            .iter()
            .try_fold(None::<Vec<String>>, |common, endpoint| {
                let reported = input_modalities_from_entry(endpoint)?;
                Some(Some(match common {
                    None => reported,
                    Some(mut values) => {
                        values.retain(|m| reported.contains(m));
                        values
                    }
                }))
            })
            .flatten(),
        source: LimitSource::ProviderEndpoint { provider },
    };
    (!limits.is_empty()).then_some(limits)
}

/// The OpenRouter providers a request's routing options pin it to.
///
/// `provider.only` pins outright. `provider.order` pins only when
/// `provider.allow_fallbacks` is `false`; with fallbacks on, OpenRouter may
/// route past the list, so the model-level limit applies and this returns an
/// empty list. Accepts either the full options object (`{"provider": {...}}`)
/// or the `provider` object itself.
#[must_use]
pub fn pinned_openrouter_providers(options: &Value) -> Vec<String> {
    let provider = options.get("provider").unwrap_or(options);
    let names = |key: &str| -> Vec<String> {
        provider
            .get(key)
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    };
    let only = names("only");
    if !only.is_empty() {
        return only;
    }
    let fallbacks_off = provider
        .get("allow_fallbacks")
        .and_then(Value::as_bool)
        .is_some_and(|allowed| !allowed);
    if fallbacks_off {
        return names("order");
    }
    Vec::new()
}

#[cfg(test)]
#[path = "parse_tests.rs"]
mod tests;
