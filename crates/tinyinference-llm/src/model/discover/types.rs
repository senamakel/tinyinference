//! Types for provider-reported model limits.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Where a [`ModelLimits`] value came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LimitSource {
    /// The model's entry in the provider's `/models` listing (or its
    /// `/models/{id}` record).
    ProviderListing,
    /// One routed endpoint's own limit, from OpenRouter's
    /// `/models/{id}/endpoints` (used when routing pins the provider).
    ProviderEndpoint {
        /// The provider name the endpoint reported (for example `"DeepInfra"`).
        provider: String,
    },
    /// The limit the provider stated in a context-overflow error.
    LearnedFromOverflow,
    /// The server's native API (Ollama `POST /api/show`), used when its
    /// OpenAI-compatible listing carries no window.
    NativeApi,
}

impl LimitSource {
    /// Stable label for logs and telemetry.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::ProviderListing => "provider_listing",
            Self::ProviderEndpoint { .. } => "provider_endpoint",
            Self::LearnedFromOverflow => "learned_from_overflow",
            Self::NativeApi => "native_api",
        }
    }
}

/// The token limits a provider reports for one model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelLimits {
    /// Maximum context window (prompt plus completion) in tokens.
    pub context_window: Option<u64>,
    /// Maximum completion tokens the provider allows per response.
    pub max_output_tokens: Option<u64>,
    /// Advertised input modalities; `None` means unknown, never assumed vision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_modalities: Option<Vec<String>>,
    /// Where the values came from.
    pub source: LimitSource,
}

impl ModelLimits {
    /// Whether the record carries neither usable limits nor modality metadata.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.context_window.is_none()
            && self.max_output_tokens.is_none()
            && self.input_modalities.is_none()
    }
}

/// One discovery request: which endpoint and model to ask about, and how.
///
/// `endpoint` is the OpenAI-compatible base the model is served from (for
/// example `https://openrouter.ai/api/v1`). It is also the cache key, together
/// with `model`, so it must match the base the chat adapter sends to; that is
/// what lets an overflow learned by the adapter
/// ([`super::record_overflow_error`]) correct the discovered value.
#[derive(Clone, PartialEq, Eq)]
pub struct DiscoveryRequest {
    /// OpenAI-compatible base URL the model is served from (cache key).
    pub endpoint: String,
    /// The model id as sent on the wire.
    pub model: String,
    /// The listing URL to read instead of `{endpoint}/models`, when the
    /// provider publishes its catalogue elsewhere.
    pub listing_url: Option<String>,
    /// Whether to fall back to `GET {endpoint}/models/{id}` when the listing
    /// does not contain the model or either token limit is missing.
    pub probe_single_model: bool,
    /// Headers to send (for example `Authorization`). Never logged.
    pub headers: Vec<(String, String)>,
    /// OpenRouter providers routing is pinned to. When non-empty the limit is
    /// the smallest of those endpoints' own limits
    /// (`/models/{id}/endpoints`), not the model-level maximum.
    pub pinned_providers: Vec<String>,
    /// Whether to ask Ollama's native `POST {root}/api/show` when the listing
    /// carries no context window. `None` auto-detects from the endpoint
    /// ([`looks_like_ollama`]); `Some(_)` forces it on or off.
    pub ollama_native: Option<bool>,
    /// Upper bound on the whole discovery (all requests together).
    pub timeout: Duration,
}

/// Whether `endpoint` is plausibly an Ollama server: the default port
/// (11434) or an `ollama` host name. Ollama's OpenAI-compatible `/v1/models`
/// never states a window, so those endpoints get the native probe.
#[must_use]
pub fn looks_like_ollama(endpoint: &str) -> bool {
    let rest = endpoint
        .split_once("://")
        .map_or(endpoint, |(_, rest)| rest);
    let authority = rest.split('/').next().unwrap_or(rest);
    let authority = authority.rsplit('@').next().unwrap_or(authority);
    let authority = authority.to_ascii_lowercase();
    authority.ends_with(":11434") || authority.contains("ollama")
}

/// The server root (`http://host:11434`) an OpenAI-compatible base hangs off.
fn native_root(endpoint: &str) -> String {
    endpoint
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .trim_end_matches('/')
        .to_string()
}

impl DiscoveryRequest {
    /// Default bound on one discovery, all requests included.
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(4);

    /// A request for `model` served from `endpoint`, with no auth, the default
    /// listing URL, and the default timeout.
    #[must_use]
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into().trim().trim_end_matches('/').to_string(),
            model: model.into().trim().to_string(),
            listing_url: None,
            probe_single_model: true,
            headers: Vec::new(),
            pinned_providers: Vec::new(),
            ollama_native: None,
            timeout: Self::DEFAULT_TIMEOUT,
        }
    }

    /// Adds a header sent on every discovery request.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Reads the catalogue from `url` instead of `{endpoint}/models`.
    #[must_use]
    pub fn with_listing_url(mut self, url: impl Into<String>) -> Self {
        self.listing_url = Some(url.into());
        self
    }

    /// Enables or disables the `GET {endpoint}/models/{id}` fallback.
    #[must_use]
    pub fn with_single_model_probe(mut self, enabled: bool) -> Self {
        self.probe_single_model = enabled;
        self
    }

    /// Restricts the limit to the endpoints of these OpenRouter providers.
    #[must_use]
    pub fn with_pinned_providers(mut self, providers: Vec<String>) -> Self {
        self.pinned_providers = providers;
        self
    }

    /// Forces the Ollama native probe on or off (default: auto-detect).
    #[must_use]
    pub fn with_ollama_native(mut self, enabled: bool) -> Self {
        self.ollama_native = Some(enabled);
        self
    }

    /// Whether the Ollama native probe applies to this request.
    #[must_use]
    pub fn ollama_native_enabled(&self) -> bool {
        self.ollama_native
            .unwrap_or_else(|| looks_like_ollama(&self.endpoint))
    }

    /// `POST` URL of Ollama's `/api/show` for this endpoint.
    #[must_use]
    pub fn ollama_show_url(&self) -> String {
        format!("{}/api/show", native_root(&self.endpoint))
    }

    /// Sets the bound on the whole discovery.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Identity of the inputs that change what discovery returns for the same
    /// `(endpoint, model)`: the effective listing URL, the single-model probe
    /// mode, and the sorted, lowercased pinned-provider set. Discovered
    /// results are cached per variant so a pinned lookup never reuses an
    /// unpinned (broader) window, or the reverse.
    #[must_use]
    pub fn cache_variant(&self) -> String {
        let mut providers: Vec<String> = self
            .pinned_providers
            .iter()
            .map(|provider| provider.trim().to_ascii_lowercase())
            .filter(|provider| !provider.is_empty())
            .collect();
        providers.sort();
        providers.dedup();
        format!(
            "{}|{}|{}|{}|{:016x}",
            self.effective_listing_url(),
            self.probe_single_model,
            self.ollama_native_enabled(),
            providers.join(","),
            self.credential_scope()
        )
    }

    /// Non-reversible digest of the request headers, so a gateway that
    /// answers differently per credential keeps its results (and negative
    /// results) separate per tenant without the secret ever being stored.
    fn credential_scope(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut headers: Vec<(String, &str)> = self
            .headers
            .iter()
            .map(|(name, value)| (name.to_ascii_lowercase(), value.as_str()))
            .collect();
        headers.sort();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        headers.hash(&mut hasher);
        hasher.finish()
    }

    /// The listing URL this request reads.
    #[must_use]
    pub fn effective_listing_url(&self) -> String {
        self.listing_url
            .clone()
            .unwrap_or_else(|| format!("{}/models", self.endpoint))
    }
}

impl std::fmt::Debug for DiscoveryRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Header values carry credentials: print only their names.
        let header_names: Vec<&str> = self.headers.iter().map(|(name, _)| name.as_str()).collect();
        formatter
            .debug_struct("DiscoveryRequest")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .field("listing_url", &self.listing_url)
            .field("probe_single_model", &self.probe_single_model)
            .field("headers", &header_names)
            .field("pinned_providers", &self.pinned_providers)
            .field("ollama_native", &self.ollama_native)
            .field("timeout", &self.timeout)
            .finish()
    }
}
