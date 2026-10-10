//! Bounded, cached discovery of model limits over HTTP.

use async_trait::async_trait;
use serde_json::Value;

use super::cache::{ModelLimitsCache, model_limits_cache};
use super::parse::{
    model_ids_match, parse_listing_limits, parse_model_limits, parse_ollama_show,
    parse_openrouter_endpoint_limits,
};
use super::types::{DiscoveryRequest, LimitSource, ModelLimits};

/// Fetches a JSON document for discovery. Injectable so hosts can reuse their
/// own HTTP client (proxies, TLS policy) and tests never touch the network.
#[async_trait]
pub trait ModelListingFetcher: Send + Sync {
    /// `GET url` with `headers`, decoded as JSON.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Catalog`] on a transport failure, a non-2xx
    /// status, or a body that is not JSON. Discovery treats any error as
    /// "nothing found here" and moves on.
    async fn get_json(&self, url: &str, headers: &[(String, String)]) -> crate::Result<Value>;

    /// `POST url` with a JSON `body` and `headers`, decoded as JSON. Used for
    /// native APIs that have no GET form (Ollama's `/api/show`).
    ///
    /// The default reports "unsupported", which discovery treats like any
    /// other failed lookup.
    ///
    /// # Errors
    ///
    /// Same contract as [`Self::get_json`].
    async fn post_json(
        &self,
        url: &str,
        _headers: &[(String, String)],
        _body: &Value,
    ) -> crate::Result<Value> {
        Err(crate::Error::Catalog(format!(
            "POST {} unsupported by this fetcher",
            redact_url(url)
        )))
    }
}

/// `url` without userinfo, query and fragment, safe for errors and logs (a
/// custom listing URL may carry an API key in either).
fn redact_url(url: &str) -> String {
    let without_tail = url.split(['?', '#']).next().unwrap_or(url);
    match without_tail.split_once("://") {
        Some((scheme, rest)) => {
            let (authority, path) = rest.split_once('/').map_or((rest, ""), |(a, p)| (a, p));
            let host = authority.rsplit('@').next().unwrap_or(authority);
            if path.is_empty() {
                format!("{scheme}://{host}")
            } else {
                format!("{scheme}://{host}/{path}")
            }
        }
        None => without_tail.to_string(),
    }
}

/// The default [`ModelListingFetcher`], over `reqwest`.
#[derive(Clone, Debug)]
pub struct ReqwestListingFetcher {
    client: reqwest::Client,
}

impl Default for ReqwestListingFetcher {
    /// A client that never follows redirects: `get_json` forwards credential
    /// headers, which must not be replayed to a redirect target.
    fn default() -> Self {
        Self {
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_default(),
        }
    }
}

impl ReqwestListingFetcher {
    /// A fetcher over the given client.
    #[must_use]
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }

    async fn send(&self, builder: reqwest::RequestBuilder, shown: &str) -> crate::Result<Value> {
        let response = builder.send().await.map_err(|error| {
            // reqwest errors embed the full URL; strip it, we add a redacted one.
            crate::Error::Catalog(format!("{shown} failed: {}", error.without_url()))
        })?;
        let status = response.status();
        if !status.is_success() {
            return Err(crate::Error::Catalog(format!(
                "{shown} returned {}",
                status.as_u16()
            )));
        }
        response.json::<Value>().await.map_err(|error| {
            crate::Error::Catalog(format!("{shown} body: {}", error.without_url()))
        })
    }
}

#[async_trait]
impl ModelListingFetcher for ReqwestListingFetcher {
    async fn get_json(&self, url: &str, headers: &[(String, String)]) -> crate::Result<Value> {
        if crate::network_models_denied() {
            return Err(crate::Error::Catalog(
                "network-backed model calls are denied for this process".to_string(),
            ));
        }
        let shown = format!("GET {}", redact_url(url));
        let mut builder = self.client.get(url);
        for (name, value) in headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        self.send(builder, &shown).await
    }

    async fn post_json(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
    ) -> crate::Result<Value> {
        if crate::network_models_denied() {
            return Err(crate::Error::Catalog(
                "network-backed model calls are denied for this process".to_string(),
            ));
        }
        let shown = format!("POST {}", redact_url(url));
        let mut builder = self.client.post(url).json(body);
        for (name, value) in headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        self.send(builder, &shown).await
    }
}

/// Discovers the limits of `request.model` served from `request.endpoint`,
/// through `cache`.
///
/// - A fresh cache entry (found or not found) answers without any request.
/// - Otherwise the listing is read (`{endpoint}/models` or
///   `request.listing_url`); every entry it lists is cached, so one fetch
///   serves every model on the endpoint. When the model or either token limit
///   is missing and `probe_single_model` is set, `{endpoint}/models/{id}` is
///   tried; missing fields are filled without discarding listing modalities. When
///   `pinned_providers` is non-empty, OpenRouter's
///   `{endpoint}/models/{id}/endpoints` narrows the window to those providers.
/// - The whole probe is bounded by `request.timeout`; a timeout or any error
///   is cached as "nothing found" for the cache's negative TTL.
/// - [`ReqwestListingFetcher`] fetches nothing while network models are denied
///   ([`crate::deny_network_models`]).
///
/// The result folds in any window learned from an overflow error
/// ([`super::record_overflow_error`]): it is the smaller of the two.
pub async fn discover_model_limits_with(
    fetcher: &dyn ModelListingFetcher,
    cache: &ModelLimitsCache,
    request: &DiscoveryRequest,
) -> Option<ModelLimits> {
    let variant = request.cache_variant();
    let cached = cache.get_variant(&request.endpoint, &request.model, &variant);
    if cached.discovered.is_some() {
        tracing::debug!(
            endpoint = %request.endpoint,
            model = %request.model,
            found = cached.discovered.as_ref().is_some_and(Option::is_some),
            "[model_limits] discovery cache hit"
        );
        return cached.effective();
    }
    let started = std::time::Instant::now();
    let limits =
        match tokio::time::timeout(request.timeout, fetch_limits(fetcher, cache, request)).await {
            Ok(limits) => limits,
            Err(_) => {
                tracing::warn!(
                    endpoint = %request.endpoint,
                    model = %request.model,
                    timeout_ms = request.timeout.as_millis() as u64,
                    "[model_limits] discovery timed out"
                );
                None
            }
        };
    tracing::info!(
        endpoint = %request.endpoint,
        model = %request.model,
        context_window = ?limits.as_ref().and_then(|limits| limits.context_window),
        max_output_tokens = ?limits.as_ref().and_then(|limits| limits.max_output_tokens),
        source = limits.as_ref().map_or("none", |limits| limits.source.label()),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "[model_limits] discovery finished"
    );
    cache.insert_discovered_variant(&request.endpoint, &request.model, &variant, limits);
    cache
        .get_variant(&request.endpoint, &request.model, &variant)
        .effective()
}

/// [`discover_model_limits_with`] through the process-wide cache.
pub async fn discover_model_limits(
    fetcher: &dyn ModelListingFetcher,
    request: &DiscoveryRequest,
) -> Option<ModelLimits> {
    discover_model_limits_with(fetcher, model_limits_cache(), request).await
}

async fn fetch_limits(
    fetcher: &dyn ModelListingFetcher,
    cache: &ModelLimitsCache,
    request: &DiscoveryRequest,
) -> Option<ModelLimits> {
    let listing_url = request.effective_listing_url();
    let mut found = match fetcher.get_json(&listing_url, &request.headers).await {
        Ok(body) => {
            let listed = parse_listing_limits(&body);
            tracing::debug!(
                endpoint = %request.endpoint,
                listed = listed.len(),
                "[model_limits] read model listing"
            );
            // Listing-level limits are the broad (unpinned) answer; never
            // prime them for a pinned request, which needs endpoint limits.
            if request.pinned_providers.is_empty() {
                let variant = request.cache_variant();
                for (id, limits) in listed {
                    // A partial entry (no window) must not become a finished
                    // discovery for its model when the native probe could still
                    // supply the window on that model's own request.
                    if request.ollama_native_enabled() && limits.context_window.is_none() {
                        continue;
                    }
                    if !model_ids_match(&id, &request.model) {
                        cache.insert_discovered_variant(
                            &request.endpoint,
                            &id,
                            &variant,
                            Some(limits),
                        );
                    }
                }
            }
            parse_model_limits(&body, &request.model)
        }
        Err(error) => {
            tracing::debug!(
                endpoint = %request.endpoint,
                error = %error,
                "[model_limits] model listing unavailable"
            );
            None
        }
    };

    if found
        .as_ref()
        .is_none_or(|limits| limits.context_window.is_none() || limits.max_output_tokens.is_none())
        && request.probe_single_model
    {
        let url = format!("{}/models/{}", request.endpoint, request.model);
        match fetcher.get_json(&url, &request.headers).await {
            Ok(body) => {
                if let Some(record) = parse_model_limits(&body, &request.model) {
                    if let Some(listed) = &mut found {
                        listed.context_window = listed.context_window.or(record.context_window);
                        listed.max_output_tokens =
                            listed.max_output_tokens.or(record.max_output_tokens);
                        if listed.input_modalities.is_none() {
                            listed.input_modalities = record.input_modalities;
                        }
                    } else {
                        found = Some(record);
                    }
                }
            }
            Err(error) => tracing::debug!(
                endpoint = %request.endpoint,
                model = %request.model,
                error = %error,
                "[model_limits] single-model record unavailable"
            ),
        }
    }

    if found
        .as_ref()
        .is_none_or(|limits| limits.context_window.is_none())
        && request.ollama_native_enabled()
    {
        let url = request.ollama_show_url();
        let body = serde_json::json!({ "model": request.model });
        match fetcher.post_json(&url, &request.headers, &body).await {
            Ok(reply) => {
                if let Some(native) = parse_ollama_show(&reply) {
                    tracing::debug!(
                        endpoint = %redact_url(&request.endpoint),
                        model = %request.model,
                        context_window = ?native.context_window,
                        "[model_limits] context window from Ollama /api/show"
                    );
                    found = Some(match found {
                        Some(mut listed) => {
                            // The window now comes from the native API; label it
                            // so, keeping the listing's other fields.
                            listed.context_window = native.context_window;
                            listed.source = LimitSource::NativeApi;
                            listed
                        }
                        None => native,
                    });
                }
            }
            Err(error) => tracing::debug!(
                endpoint = %redact_url(&request.endpoint),
                model = %request.model,
                error = %error,
                "[model_limits] Ollama /api/show unavailable"
            ),
        }
    }

    if !request.pinned_providers.is_empty() {
        let url = format!("{}/models/{}/endpoints", request.endpoint, request.model);
        match fetcher.get_json(&url, &request.headers).await {
            Ok(body) => {
                if let Some(mut pinned) =
                    parse_openrouter_endpoint_limits(&body, &request.pinned_providers)
                {
                    if pinned.max_output_tokens.is_none() {
                        pinned.max_output_tokens =
                            found.as_ref().and_then(|limits| limits.max_output_tokens);
                    }
                    found = Some(pinned);
                } else {
                    // The model-level value may be the maximum across
                    // providers; it does not describe the pinned route.
                    tracing::debug!(
                        endpoint = %request.endpoint,
                        model = %request.model,
                        "[model_limits] no pinned provider endpoint matched; dropping model-level limit"
                    );
                    found = None;
                }
            }
            Err(error) => {
                tracing::debug!(
                    endpoint = %request.endpoint,
                    model = %request.model,
                    error = %error,
                    "[model_limits] pinned-provider endpoints unavailable; dropping model-level limit"
                );
                found = None;
            }
        }
    }

    found
}

#[cfg(test)]
#[path = "fetch_tests.rs"]
mod tests;
