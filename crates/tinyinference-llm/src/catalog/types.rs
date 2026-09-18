//! `ModelInfo`: the typed representation of one `/models` catalog entry.

use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
/// Normalized metadata for one model advertised by a provider catalog.
pub struct ModelInfo {
    /// Provider model identifier used in inference requests.
    pub id: String,
    /// Provider or organization that owns the model, when advertised.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
    /// Maximum context length in tokens, when advertised.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// Human-readable name when the listing supplies one (the managed
    /// `?catalog=` listing does; a bare OpenAI-compatible `/models` does not).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Charged price in USD per 1M tokens, when the listing publishes it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_per_1m: Option<f64>,
    /// Charged output price in USD per million tokens, when advertised.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_per_1m: Option<f64>,
}
