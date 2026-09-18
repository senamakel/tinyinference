//! The setup-time embed probe: sending one request to a custom endpoint and
//! classifying what came back into an accept-or-reject verdict.

use crate::model_supports_dimensions;

/// A portable rejection returned by embedding endpoint validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingProbeRejection {
    /// Stable machine-readable error code.
    pub error: &'static str,
    /// User-facing remediation message.
    pub message: &'static str,
    /// Concise diagnostic summary.
    pub summary: &'static str,
    /// Sanitized provider detail, when available.
    pub detail: Option<String>,
}

/// Send one OpenAI-compatible embedding request. Models with configurable
/// widths are probed at the requested width; other models discover their native
/// width. This remains separate from the live provider so setup can validate
/// configuration before it is persisted.
pub async fn probe_custom_embeddings(
    endpoint: &str,
    api_key: &str,
    model: &str,
    configured_dimensions: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let url = embeddings_probe_url(endpoint)?;
    let safe_url = tinyinference_core::sanitize::redact_url(url.as_str());
    let mut body = serde_json::json!({ "model": model, "input": ["connection test"] });
    if model_supports_dimensions(model) && configured_dimensions > 0 {
        body["dimensions"] = serde_json::json!(configured_dimensions);
    }
    let mut request = reqwest::Client::new().post(url).json(&body);
    if !api_key.trim().is_empty() {
        request = request.bearer_auth(api_key.trim());
    }
    let response = request
        .send()
        .await
        .map_err(|e| format!("custom embeddings request to {safe_url} failed: {e}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| format!("custom embeddings response read failed: {e}"))?;
    if !status.is_success() {
        return Err(format!("custom embeddings returned HTTP {status}: {body}"));
    }
    let data = serde_json::from_str::<serde_json::Value>(&body)
        .map_err(|e| format!("custom embeddings response was not JSON: {e}"))?
        .get("data")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .ok_or_else(|| "custom embeddings response missing data array".to_string())?;
    let vectors = data
        .into_iter()
        .map(|item| {
            item.get("embedding")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| "custom embeddings response missing embedding array".to_string())?
                .iter()
                .map(|value| {
                    value.as_f64().map(|value| value as f32).ok_or_else(|| {
                        "custom embeddings response contains a non-numeric vector".to_string()
                    })
                })
                .collect()
        })
        .collect::<Result<Vec<Vec<f32>>, String>>()?;
    validate_probe_vectors(model, configured_dimensions, &vectors)?;
    Ok(vectors)
}

fn validate_probe_vectors(
    model: &str,
    configured_dimensions: usize,
    vectors: &[Vec<f32>],
) -> Result<(), String> {
    if vectors.len() != 1 {
        return Err(format!(
            "custom embeddings vector count mismatch: expected 1, got {}",
            vectors.len()
        ));
    }
    if model_supports_dimensions(model)
        && configured_dimensions > 0
        && vectors[0].len() != configured_dimensions
    {
        let actual = vectors.first().map(Vec::len).unwrap_or(0);
        return Err(format!(
            "custom embeddings dimension mismatch: expected {configured_dimensions}, got {actual}"
        ));
    }
    Ok(())
}

fn embeddings_probe_url(endpoint: &str) -> Result<url::Url, String> {
    let mut url = url::Url::parse(endpoint.trim())
        .map_err(|e| format!("invalid custom embeddings endpoint: {e}"))?;
    let path = url.path().trim_end_matches('/');
    let path = if path.ends_with("/embeddings") {
        path.to_string()
    } else if path.ends_with("/v1") {
        format!("{path}/embeddings")
    } else {
        format!("{path}/v1/embeddings")
    };
    url.set_path(&path);
    Ok(url)
}

/// Dimension to persist after a successful Custom verification probe.
///
/// The probe sends and verifies a configurable width when the model supports
/// it. Every successful probe can therefore persist the actual returned length
/// (`actual`) without assuming that a server honoured a request parameter.
/// Falls back to `configured` if the probe somehow reported a zero-length
/// vector (defensive — `classify_embed_probe` already rejects empty vectors).
pub fn final_probe_dims(_model: &str, configured: usize, actual: usize) -> usize {
    if actual == 0 { configured } else { actual }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_url_appends_path_before_query_parameters() {
        let url = embeddings_probe_url("https://host.example/v1?api-version=2026").unwrap();
        assert_eq!(url.path(), "/v1/embeddings");
        assert_eq!(url.query(), Some("api-version=2026"));
    }

    #[test]
    fn final_dimensions_use_the_proven_response_width() {
        assert_eq!(final_probe_dims("text-embedding-3-large", 1024, 3072), 3072);
        assert_eq!(final_probe_dims("bge-m3", 1024, 768), 768);
        assert_eq!(final_probe_dims("bge-m3", 1024, 0), 1024);
    }

    #[test]
    fn probe_requires_one_vector_for_its_one_input() {
        let error = validate_probe_vectors("bge-m3", 1024, &[vec![0.0], vec![1.0]])
            .expect_err("two vectors must be rejected");
        assert!(error.contains("expected 1, got 2"));
    }
}

/// Normalized result of a setup-time test embed.
/// Collapses the `Result<Result<_, _>, Elapsed>` timeout shape into one enum so
/// the verification policy can be expressed (and unit-tested) as a pure
/// function over it.
#[derive(Debug)]
pub enum EmbedProbe {
    /// The endpoint returned vectors (may still be empty/zero-dim — checked).
    Returned(Vec<Vec<f32>>),
    /// The embed call returned an error; the string is the provider detail.
    Failed(String),
    /// The probe didn't complete within the time box.
    TimedOut,
}

/// Setup-time embeddings verification policy. Returns `None` when the endpoint
/// is verified (accept + persist the config) or `Some(reject)` — the
/// "not saved" RPC payload — otherwise.
///
/// The endpoint must prove it can embed before we accept it: only a non-empty
/// vector passes; every failure mode (no model loaded, no `/embeddings` route,
/// 5xx/auth/network, timeout, empty vector) rejects the save. We do NOT try to
/// classify-and-suppress the resulting embed flood in code — residual floods
/// (e.g. the user unloads the model after a good save) are handled Sentry-side.
/// The known shapes only get a friendlier remediation message.
pub fn classify_embed_probe(outcome: EmbedProbe) -> Option<EmbeddingProbeRejection> {
    let reject = |error: &'static str,
                  message: &'static str,
                  summary: &'static str,
                  detail: Option<&str>| {
        Some(EmbeddingProbeRejection {
            error,
            message,
            summary,
            detail: detail.map(redact_secrets),
        })
    };

    match outcome {
        // Pass only when the endpoint returns a usable vector.
        EmbedProbe::Returned(vectors)
            if vectors.first().map(|v| !v.is_empty()).unwrap_or(false) =>
        {
            None
        }
        // Reachable but produced no usable vector — not a valid embedder.
        EmbedProbe::Returned(_) => reject(
            "EMBEDDINGS_VERIFICATION_FAILED",
            "The embeddings endpoint responded but returned no vector. Choose an \
             embeddings-capable provider or endpoint, then save again.",
            "test embed returned no vectors — not saved",
            None,
        ),
        EmbedProbe::Failed(detail) => {
            let lower = detail.to_ascii_lowercase();
            // The endpoint IS reachable and correctly shaped (POST /v1/embeddings
            // with the user's model + key — verified conformant by the mock-endpoint
            // regression test). The failures below are all *distinct causes*; issue
            // #5017 was that they collapsed into one generic "test embed failed"
            // message, so a user whose endpoint works for chat couldn't tell that
            // (e.g.) their chosen model isn't an embeddings model, their key was
            // rejected, or the host was unreachable. Order matters: check the
            // specific shapes before the generic fallback.
            if lower.contains("no models loaded") {
                // Reachable but no model loaded (e.g. LM Studio idle).
                reject(
                    "EMBEDDINGS_NO_MODEL_LOADED",
                    "Your local embeddings server (e.g. LM Studio) is running but has no \
                     model loaded. Load an embedding model — in LM Studio use the developer \
                     page or the `lms load` command — then save again.",
                    "embeddings server has no model loaded — not saved",
                    Some(&detail),
                )
            } else if is_embedding_endpoint_absent(&lower) {
                // Endpoint exposes no embeddings API (404/405).
                reject(
                    "EMBEDDINGS_ENDPOINT_NO_API",
                    "This endpoint has no embeddings API. Choose an embeddings-capable \
                     provider (Managed, Voyage, OpenAI, Cohere, Ollama) or a different \
                     custom endpoint.",
                    "embeddings endpoint has no embeddings API — not saved",
                    Some(&detail),
                )
            } else if is_embedding_dimension_mismatch(&lower) {
                // Endpoint embedded fine but returned a different vector length than
                // the (Matryoshka) size we requested — a `text-embedding-3-*` model
                // name pointed at a host that ignores the `dimensions` param.
                reject(
                    "EMBEDDINGS_DIMENSION_MISMATCH",
                    "The endpoint returned a vector with a different length than the \
                     dimensions you entered. Set dimensions to match the model's native \
                     output, then save again.",
                    "embeddings endpoint returned mismatched dimensions — not saved",
                    Some(&detail),
                )
            } else if is_embedding_model_incompatible(&lower) {
                // Reachable, authenticated embeddings API that rejected the model —
                // the user pasted a chat/reasoning model (e.g. `gpt-5-mini`) into the
                // embeddings model field. This is the #5017 reporter's exact case:
                // the same model works for chat but is not an embeddings model.
                reject(
                    "EMBEDDINGS_MODEL_INCOMPATIBLE",
                    "That model isn't an embeddings model on this endpoint. A chat model \
                     (the one that works in Chat settings) can't produce embeddings — \
                     enter an embeddings model id (e.g. text-embedding-3-small, bge-m3), \
                     then save again.",
                    "embeddings model is not an embeddings model — not saved",
                    Some(&detail),
                )
            } else if embed_error_mentions_status(&lower, 401)
                || embed_error_mentions_status(&lower, 403)
            {
                // Auth failure — key missing/wrong/lacking embeddings scope. The
                // embeddings key is stored separately from the chat BYOK key, so
                // "works for chat" does not imply the embeddings key is set.
                reject(
                    "EMBEDDINGS_AUTH_FAILED",
                    "The endpoint rejected the API key (401/403). Enter a valid key for \
                     this endpoint — note the embeddings key is stored separately from the \
                     Chat provider key — then save again.",
                    "embeddings endpoint rejected the API key — not saved",
                    Some(&detail),
                )
            } else if is_embedding_endpoint_unreachable(&lower) {
                // Transport-level failure — DNS, refused connection, TLS. The base
                // URL is wrong or the host is down.
                reject(
                    "EMBEDDINGS_ENDPOINT_UNREACHABLE",
                    "Couldn't reach the embeddings endpoint (network/DNS/connection \
                     error). Check the base URL and that the host is reachable, then save \
                     again.",
                    "embeddings endpoint unreachable — not saved",
                    Some(&detail),
                )
            } else {
                // Any other failure (5xx, unclassified) — didn't pass verification.
                reject(
                    "EMBEDDINGS_VERIFICATION_FAILED",
                    "Couldn't verify the embeddings endpoint — the test embed failed. Make \
                     sure the endpoint is reachable and serving an embedding model, then \
                     save again.",
                    "embeddings endpoint failed verification — not saved",
                    Some(&detail),
                )
            }
        }
        EmbedProbe::TimedOut => reject(
            "EMBEDDINGS_ENDPOINT_UNREACHABLE",
            "Couldn't verify the embeddings endpoint — the test embed timed out. Make sure \
             the endpoint is running and reachable, then save again.",
            "embeddings endpoint timed out during verification — not saved",
            None,
        ),
    }
}

/// Whether a lowercased embed-error detail names the given HTTP status, tolerant
/// of the wire shapes the embeddings stack emits:
///   `openai embeddings returned HTTP 401 Unauthorized: …` (tinyagents adapter)
///   `Embedding API error (401 Unauthorized): …`           (parenthesized host shape)
///   `Embedding API error 401 Unauthorized: …`             (bare-status host shape)
/// The bare-status `Embedding API error {code}` form is the one the observability
/// classifier in `crates/openhuman-core/src/core/observability.rs` covers; without it, setup-time
/// verification for those hosts fell through to the generic failure code (#5017).
fn embed_error_mentions_status(lower: &str, code: u16) -> bool {
    let code = code.to_string();
    lower.contains(&format!("http {code}"))
        || lower.contains(&format!("({code}"))
        || lower.contains(&format!("embedding api error {code}"))
}

/// A reachable, authenticated embeddings API that **rejected the model id** — the
/// user pointed the embeddings model field at a chat/reasoning model.
///
/// Two tiers of phrasing:
///
/// - **Strong, status-independent phrasings** unambiguously name a model that
///   can't embed. OpenAI returns *HTTP 403* "You are not allowed to generate
///   embeddings from this model" when a chat model (e.g. `gpt-4o-mini`) is used
///   as the embeddings model — a MODEL problem, not an auth problem. Because
///   `classify_embed_probe` checks this **before** the 401/403 auth branch, that
///   403 must be caught here or it falls through and misreports "enter a valid
///   key" (issue #5116). None of these phrases appear in a genuine auth rejection
///   (`Incorrect API key provided …`), so matching them ahead of auth is safe.
/// - **Weak phrasings** (a stray "does not exist" / odd model-name format) are
///   only unambiguous alongside a 400/422 bad-request, so a genuine 5xx or an
///   oversized-input 400 still falls through to the generic failure (issue #5017).
fn is_embedding_model_incompatible(lower: &str) -> bool {
    let strong_model_rejection = lower.contains("not allowed to generate embeddings")
        || lower.contains("does not support embeddings")
        || lower.contains("not an embedding model")
        || lower.contains("is not an embedding")
        || lower.contains("not supported for embeddings")
        || (lower.contains("unsupported") && lower.contains("embedding"));
    if strong_model_rejection {
        return true;
    }
    let bad_request =
        embed_error_mentions_status(lower, 400) || embed_error_mentions_status(lower, 422);
    bad_request
        && (lower.contains("does not exist") || lower.contains("unexpected model name format"))
}

/// Strip API-key / bearer-token material from any text before it reaches the UI
/// or logs. Matches OpenAI-style keys (`sk-…`, including the modern `sk-proj-…`
/// form with embedded hyphens/underscores) and `Bearer <token>` headers, and
/// replaces each **whole** match — the replacements deliberately contain no `sk-`
/// substring, so not even a key *prefix* can surface (#5116).
pub fn redact_secrets(input: &str) -> String {
    use once_cell::sync::Lazy;
    use regex::Regex;
    static SK_KEY_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\bsk-[A-Za-z0-9_-]+").unwrap());
    static BEARER_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+").unwrap());
    let redacted = SK_KEY_RE.replace_all(input, "[redacted-key]");
    BEARER_RE
        .replace_all(&redacted, "Bearer [redacted]")
        .into_owned()
}

/// Whether an error identifies an endpoint without an embeddings route.
pub fn is_embedding_endpoint_absent(lower: &str) -> bool {
    (embed_error_mentions_status(lower, 404) || embed_error_mentions_status(lower, 405))
        && (lower.contains("embedding") || lower.contains("/v1/embeddings"))
}

/// Whether a provider error is a managed embedding backend token failure.
pub fn is_embedding_backend_auth_failure(lower: &str) -> bool {
    lower.contains("embedding api error")
        && lower.contains("401")
        && lower.contains("invalid token")
}

/// Whether an embeddings endpoint rejected the selected model identifier.
pub fn is_embedding_model_rejected(lower: &str) -> bool {
    lower.contains("embedding api error")
        && lower.contains("(400")
        && (lower.contains("does not exist")
            || lower.contains("does not support embeddings")
            || lower.contains("unexpected model name format")
            || (lower.contains("invalid_argument")
                && lower.contains("batchembedcontentsrequest.model")))
}

/// Whether an error indicates that a local Ollama runtime or model is absent.
pub fn is_local_embedding_unavailable(lower: &str) -> bool {
    lower.contains("is ollama running at")
        || lower.contains("ollama embed failed with status 404")
        || (lower.contains("is not installed at") && lower.contains("ollama"))
}

/// The post-response length guard fired: the endpoint embedded but returned a
/// vector whose length differs from the requested (Matryoshka) `dimensions`.
/// Canonical shape from the tinyagents adapter:
/// `openai embed dimension mismatch: expected 1024, got 3072`.
fn is_embedding_dimension_mismatch(lower: &str) -> bool {
    lower.contains("dimension mismatch")
}

/// A transport-level failure (DNS, refused connection, TLS, connect timeout) —
/// the endpoint was never reached, so the base URL is wrong or the host is down.
/// The tinyagents adapter wraps these as
/// `openai embeddings request to <url> failed: <reqwest error>`.
fn is_embedding_endpoint_unreachable(lower: &str) -> bool {
    lower.contains("request to") && lower.contains("failed")
        || lower.contains("connection refused")
        || lower.contains("error sending request")
        || lower.contains("error trying to connect")
        || lower.contains("dns error")
        || lower.contains("failed to lookup address")
        || lower.contains("tcp connect error")
}
