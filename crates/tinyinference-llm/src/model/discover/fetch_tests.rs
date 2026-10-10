use super::*;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use serde_json::json;

use crate::model::discover::LimitSource;

const ENDPOINT: &str = "https://openrouter.ai/api/v1";
const MODEL: &str = "deepseek/deepseek-v4.1-flash";

/// Serves canned JSON per URL and records every request; never touches the
/// network.
#[derive(Default)]
struct FakeFetcher {
    routes: HashMap<String, Value>,
    calls: Mutex<Vec<(String, Vec<String>)>>,
    hang: bool,
    /// Canned `POST` replies per URL (`Err` carries a status to report).
    post_routes: HashMap<String, std::result::Result<Value, u16>>,
    post_hang: bool,
    post_bodies: Mutex<Vec<Value>>,
}

impl FakeFetcher {
    fn with(mut self, url: &str, body: Value) -> Self {
        self.routes.insert(url.to_string(), body);
        self
    }

    fn with_post(mut self, url: &str, reply: std::result::Result<Value, u16>) -> Self {
        self.post_routes.insert(url.to_string(), reply);
        self
    }

    fn post_hits(&self) -> usize {
        self.post_bodies.lock().unwrap().len()
    }

    fn calls(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|(url, _)| url.clone())
            .collect()
    }
}

#[async_trait]
impl ModelListingFetcher for FakeFetcher {
    async fn get_json(&self, url: &str, headers: &[(String, String)]) -> crate::Result<Value> {
        self.calls.lock().unwrap().push((
            url.to_string(),
            headers.iter().map(|(name, _)| name.clone()).collect(),
        ));
        if self.hang {
            futures::future::pending::<()>().await;
        }
        self.routes
            .get(url)
            .cloned()
            .ok_or_else(|| crate::Error::Catalog(format!("GET {url} returned 404")))
    }

    async fn post_json(
        &self,
        url: &str,
        _headers: &[(String, String)],
        body: &Value,
    ) -> crate::Result<Value> {
        self.post_bodies.lock().unwrap().push(body.clone());
        if self.post_hang {
            futures::future::pending::<()>().await;
        }
        match self.post_routes.get(url) {
            Some(Ok(reply)) => Ok(reply.clone()),
            Some(Err(status)) => Err(crate::Error::Catalog(format!(
                "POST {url} returned {status}"
            ))),
            None => Err(crate::Error::Catalog(format!("POST {url} returned 404"))),
        }
    }
}

fn openrouter_listing() -> Value {
    json!({ "data": [
        {
            "id": MODEL,
            "context_length": 1_048_576,
            "top_provider": { "context_length": 1_048_576, "max_completion_tokens": 65_536 }
        },
        { "id": "z-ai/glm-5", "context_length": 202_752 }
    ]})
}

fn cache() -> ModelLimitsCache {
    ModelLimitsCache::default()
}

#[tokio::test]
async fn discovers_deepseek_v41_flash_window_from_openrouter_listing() {
    let fetcher = FakeFetcher::default().with(&format!("{ENDPOINT}/models"), openrouter_listing());
    let cache = cache();
    let request =
        DiscoveryRequest::new(ENDPOINT, MODEL).with_header("Authorization", "Bearer sk-secret");

    let limits = discover_model_limits_with(&fetcher, &cache, &request)
        .await
        .expect("listing reports the model");
    assert_eq!(limits.context_window, Some(1_048_576));
    assert_ne!(limits.context_window, Some(128_000));
    assert_eq!(limits.max_output_tokens, Some(65_536));
    assert_eq!(limits.source, LimitSource::ProviderListing);

    // The auth header is forwarded.
    let calls = fetcher.calls.lock().unwrap().clone();
    assert_eq!(calls[0].1, ["Authorization"]);
}

#[tokio::test]
async fn cache_hit_skips_the_network_and_listing_serves_other_models() {
    let fetcher = FakeFetcher::default().with(&format!("{ENDPOINT}/models"), openrouter_listing());
    let cache = cache();
    let request = DiscoveryRequest::new(ENDPOINT, MODEL);
    discover_model_limits_with(&fetcher, &cache, &request).await;
    discover_model_limits_with(&fetcher, &cache, &request).await;
    let other = discover_model_limits_with(
        &fetcher,
        &cache,
        &DiscoveryRequest::new(ENDPOINT, "z-ai/glm-5"),
    )
    .await
    .unwrap();
    assert_eq!(other.context_window, Some(202_752));
    assert_eq!(
        fetcher.calls().len(),
        1,
        "one listing fetch serves all three"
    );
}

#[tokio::test]
async fn probes_single_model_record_when_listing_lacks_it() {
    let endpoint = "http://vllm.local:8000/v1";
    let fetcher = FakeFetcher::default()
        .with(&format!("{endpoint}/models"), json!({ "data": [] }))
        .with(
            &format!("{endpoint}/models/qwen3"),
            json!({ "id": "qwen3", "max_model_len": 40_960 }),
        );
    let limits = discover_model_limits_with(
        &fetcher,
        &cache(),
        &DiscoveryRequest::new(endpoint, "qwen3"),
    )
    .await
    .unwrap();
    assert_eq!(limits.context_window, Some(40_960));
    assert_eq!(
        fetcher.calls(),
        [
            format!("{endpoint}/models"),
            format!("{endpoint}/models/qwen3")
        ]
    );
}

#[tokio::test]
async fn single_model_probe_can_be_disabled_and_listing_url_overridden() {
    let fetcher = FakeFetcher::default().with(
        "https://api.example/openai/v1/models?catalog=openrouter",
        json!({ "data": [] }),
    );
    let request = DiscoveryRequest::new("https://api.example/openai/v1", "m")
        .with_listing_url("https://api.example/openai/v1/models?catalog=openrouter")
        .with_single_model_probe(false);
    assert!(
        discover_model_limits_with(&fetcher, &cache(), &request)
            .await
            .is_none()
    );
    assert_eq!(fetcher.calls().len(), 1);
}

#[tokio::test]
async fn pinned_provider_reads_endpoint_limits() {
    let fetcher = FakeFetcher::default()
        .with(&format!("{ENDPOINT}/models"), openrouter_listing())
        .with(
            &format!("{ENDPOINT}/models/{MODEL}/endpoints"),
            json!({ "data": { "endpoints": [
                { "provider_name": "DeepInfra", "context_length": 163_840 },
                { "provider_name": "DeepSeek", "context_length": 1_048_576 }
            ]}}),
        );
    let request =
        DiscoveryRequest::new(ENDPOINT, MODEL).with_pinned_providers(vec!["DeepInfra".into()]);
    let limits = discover_model_limits_with(&fetcher, &cache(), &request)
        .await
        .unwrap();
    assert_eq!(limits.context_window, Some(163_840));
    // The endpoint omitted its output cap, so the listing's is kept.
    assert_eq!(limits.max_output_tokens, Some(65_536));
    assert!(matches!(
        limits.source,
        LimitSource::ProviderEndpoint { .. }
    ));
}

#[tokio::test]
async fn failure_is_cached_as_not_found() {
    let fetcher = FakeFetcher::default();
    let cache = cache();
    let request = DiscoveryRequest::new("https://nowhere.example/v1", "m");
    assert!(
        discover_model_limits_with(&fetcher, &cache, &request)
            .await
            .is_none()
    );
    assert!(
        discover_model_limits_with(&fetcher, &cache, &request)
            .await
            .is_none()
    );
    // Listing + single-model probe once; the second call is a negative hit.
    assert_eq!(fetcher.calls().len(), 2);
    assert_eq!(
        cache
            .get_variant("https://nowhere.example/v1", "m", &request.cache_variant())
            .discovered,
        Some(None)
    );
}

#[tokio::test]
async fn hung_provider_is_bounded_by_timeout() {
    let fetcher = FakeFetcher {
        hang: true,
        ..FakeFetcher::default()
    };
    let request = DiscoveryRequest::new(ENDPOINT, MODEL).with_timeout(Duration::from_millis(20));
    assert!(
        discover_model_limits_with(&fetcher, &cache(), &request)
            .await
            .is_none()
    );
}

#[tokio::test]
async fn learned_overflow_lowers_discovered_window() {
    let fetcher = FakeFetcher::default().with(&format!("{ENDPOINT}/models"), openrouter_listing());
    let cache = cache();
    crate::model::discover::record_overflow_error_in(
        &cache,
        ENDPOINT,
        MODEL,
        "This endpoint's maximum context length is 163840 tokens. However, you requested about 200000 tokens",
    );
    let limits =
        discover_model_limits_with(&fetcher, &cache, &DiscoveryRequest::new(ENDPOINT, MODEL))
            .await
            .unwrap();
    assert_eq!(limits.context_window, Some(163_840));
    assert_eq!(limits.source, LimitSource::LearnedFromOverflow);
}

#[test]
fn request_debug_redacts_header_values() {
    let request =
        DiscoveryRequest::new(ENDPOINT, MODEL).with_header("Authorization", "Bearer sk-secret");
    let rendered = format!("{request:?}");
    assert!(rendered.contains("Authorization"));
    assert!(!rendered.contains("sk-secret"));
}

#[tokio::test]
async fn pinned_and_unpinned_requests_do_not_share_cached_limits() {
    let fetcher = FakeFetcher::default()
        .with(&format!("{ENDPOINT}/models"), openrouter_listing())
        .with(
            &format!("{ENDPOINT}/models/{MODEL}/endpoints"),
            json!({ "data": { "endpoints": [
                { "provider_name": "DeepInfra", "context_length": 163_840 }
            ]}}),
        );
    let cache = cache();
    let unpinned = DiscoveryRequest::new(ENDPOINT, MODEL);
    let pinned = unpinned
        .clone()
        .with_pinned_providers(vec!["DeepInfra".into()]);
    let broad = discover_model_limits_with(&fetcher, &cache, &unpinned)
        .await
        .unwrap();
    assert_eq!(broad.context_window, Some(1_048_576));
    let narrow = discover_model_limits_with(&fetcher, &cache, &pinned)
        .await
        .unwrap();
    assert_eq!(narrow.context_window, Some(163_840));
    let again = discover_model_limits_with(&fetcher, &cache, &unpinned)
        .await
        .unwrap();
    assert_eq!(again.context_window, Some(1_048_576));
}

#[tokio::test]
async fn pinned_lookup_failure_does_not_fall_back_to_model_level_limit() {
    // Endpoints route missing: the listing's 1M window must not be reported.
    let fetcher = FakeFetcher::default().with(&format!("{ENDPOINT}/models"), openrouter_listing());
    let request =
        DiscoveryRequest::new(ENDPOINT, MODEL).with_pinned_providers(vec!["DeepInfra".into()]);
    assert!(
        discover_model_limits_with(&fetcher, &cache(), &request)
            .await
            .is_none()
    );

    // Endpoints present but no pinned provider matches.
    let fetcher = FakeFetcher::default()
        .with(&format!("{ENDPOINT}/models"), openrouter_listing())
        .with(
            &format!("{ENDPOINT}/models/{MODEL}/endpoints"),
            json!({ "data": { "endpoints": [
                { "provider_name": "Other", "context_length": 1_048_576 }
            ]}}),
        );
    assert!(
        discover_model_limits_with(&fetcher, &cache(), &request)
            .await
            .is_none()
    );
}

#[test]
fn cache_variant_is_order_and_case_insensitive_over_providers() {
    let a =
        DiscoveryRequest::new(ENDPOINT, MODEL).with_pinned_providers(vec!["B".into(), "a".into()]);
    let b =
        DiscoveryRequest::new(ENDPOINT, MODEL).with_pinned_providers(vec!["A".into(), "b".into()]);
    assert_eq!(a.cache_variant(), b.cache_variant());
    assert_ne!(
        a.cache_variant(),
        DiscoveryRequest::new(ENDPOINT, MODEL).cache_variant()
    );
    assert_ne!(
        DiscoveryRequest::new(ENDPOINT, MODEL).cache_variant(),
        DiscoveryRequest::new(ENDPOINT, MODEL)
            .with_single_model_probe(false)
            .cache_variant()
    );
}

#[test]
fn default_fetcher_builds_without_following_redirects() {
    // Construction must not panic; redirects are disabled so credential
    // headers are never replayed to a redirect target.
    let _ = ReqwestListingFetcher::default();
}

#[test]
fn redact_url_strips_userinfo_query_and_fragment() {
    assert_eq!(
        redact_url("https://user:pw@api.example/v1/models?key=sk-secret#frag"),
        "https://api.example/v1/models"
    );
    assert_eq!(redact_url("https://api.example"), "https://api.example");
    assert_eq!(redact_url("not a url?x=1"), "not a url");
}

#[tokio::test]
async fn different_credentials_do_not_share_cached_limits() {
    let fetcher = FakeFetcher::default().with(&format!("{ENDPOINT}/models"), openrouter_listing());
    let cache = cache();
    let a = DiscoveryRequest::new(ENDPOINT, MODEL).with_header("Authorization", "Bearer a");
    let b = DiscoveryRequest::new(ENDPOINT, MODEL).with_header("Authorization", "Bearer b");
    assert_ne!(a.cache_variant(), b.cache_variant());
    assert!(!a.cache_variant().contains("Bearer"));
    discover_model_limits_with(&fetcher, &cache, &a).await;
    discover_model_limits_with(&fetcher, &cache, &b).await;
    assert_eq!(fetcher.calls().len(), 2, "each tenant probes for itself");
}

#[tokio::test]
async fn cached_accessor_sees_a_default_discovery() {
    let fetcher = FakeFetcher::default().with(&format!("{ENDPOINT}/models"), openrouter_listing());
    let cache = cache();
    discover_model_limits_with(&fetcher, &cache, &DiscoveryRequest::new(ENDPOINT, MODEL)).await;
    assert_eq!(
        cache
            .get(ENDPOINT, MODEL)
            .effective()
            .unwrap()
            .context_window,
        Some(1_048_576)
    );
}

#[tokio::test]
async fn modality_only_listing_merges_single_model_token_limits() {
    let fetcher = FakeFetcher::default()
        .with(
            &format!("{ENDPOINT}/models"),
            json!({"data":[{"id":MODEL,"input_modalities":["text","image"]}]}),
        )
        .with(
            &format!("{ENDPOINT}/models/{MODEL}"),
            json!({"id":MODEL,"context_window":64000,"max_output_tokens":8192}),
        );
    let cache = cache();
    let request = DiscoveryRequest::new(ENDPOINT, MODEL);
    let result = discover_model_limits_with(&fetcher, &cache, &request)
        .await
        .unwrap();
    assert_eq!(result.context_window, Some(64000));
    assert_eq!(result.max_output_tokens, Some(8192));
    assert_eq!(
        result.input_modalities,
        Some(vec!["text".into(), "image".into()])
    );
    assert_eq!(fetcher.calls().len(), 2);
    assert_eq!(
        discover_model_limits_with(&fetcher, &cache, &request).await,
        Some(result)
    );
    assert_eq!(fetcher.calls().len(), 2);
}

#[tokio::test]
async fn partial_listing_fills_missing_limits_and_retains_facts_if_probe_unavailable() {
    for probe in [
        None,
        Some(json!({"id":MODEL,"context_window":96000,"max_output_tokens":4096})),
    ] {
        let mut fetcher = FakeFetcher::default().with(
            &format!("{ENDPOINT}/models"),
            json!({"data":[{"id":MODEL,"context_window":32000,"input_modalities":[]}]}),
        );
        if let Some(record) = probe {
            fetcher = fetcher.with(&format!("{ENDPOINT}/models/{MODEL}"), record);
        }
        let result =
            discover_model_limits_with(&fetcher, &cache(), &DiscoveryRequest::new(ENDPOINT, MODEL))
                .await
                .unwrap();
        assert_eq!(result.context_window, Some(32000));
        assert_eq!(result.input_modalities, Some(vec![]));
        assert_eq!(
            result.max_output_tokens,
            fetcher
                .routes
                .get(&format!("{ENDPOINT}/models/{MODEL}"))
                .map(|_| 4096)
        );
        assert_eq!(fetcher.calls().len(), 2);
    }
    let fetcher = FakeFetcher::default().with(
        &format!("{ENDPOINT}/models"),
        json!({"data":[{"id":MODEL,"input_modalities":["text"]}]}),
    );
    let result = discover_model_limits_with(
        &fetcher,
        &cache(),
        &DiscoveryRequest::new(ENDPOINT, MODEL).with_single_model_probe(false),
    )
    .await
    .unwrap();
    assert_eq!(result.input_modalities, Some(vec!["text".into()]));
    assert_eq!(fetcher.calls().len(), 1);
}

// ── Ollama native `/api/show` discovery, against an in-memory fetcher ────────

const OLLAMA_ROOT: &str = "http://ollama.test";
const OLLAMA_SHOW: &str = "http://ollama.test/api/show";

/// An Ollama-style `/v1` listing: ids only, no window.
fn ollama_fetcher(show: std::result::Result<Value, u16>) -> FakeFetcher {
    FakeFetcher::default()
        .with(
            &format!("{OLLAMA_ROOT}/v1/models"),
            json!({"object":"list","data":[{"id":"qwen3:14b","object":"model","owned_by":"library"}]}),
        )
        .with_post(OLLAMA_SHOW, show)
}

fn ollama_request() -> DiscoveryRequest {
    // Forced on because the host name is not an Ollama one.
    DiscoveryRequest::new(format!("{OLLAMA_ROOT}/v1"), "qwen3:14b").with_ollama_native(true)
}

fn show_body() -> Value {
    json!({
        "model_info": { "general.architecture": "qwen3", "qwen3.context_length": 40960 },
        "capabilities": ["completion", "tools"]
    })
}

#[tokio::test]
async fn ollama_show_supplies_the_window_the_v1_listing_lacks() {
    let fetcher = ollama_fetcher(Ok(show_body()));
    let limits = discover_model_limits_with(&fetcher, &cache(), &ollama_request())
        .await
        .expect("limits");
    assert_eq!(limits.context_window, Some(40_960));
    assert_eq!(limits.source, LimitSource::NativeApi);
    assert_eq!(
        fetcher.post_bodies.lock().unwrap().as_slice(),
        [json!({"model":"qwen3:14b"})]
    );
}

#[tokio::test]
async fn ollama_num_ctx_below_the_architecture_window_wins() {
    let mut body = show_body();
    body["parameters"] = json!("num_ctx                        16384\nstop \"<|im_end|>\"");
    let fetcher = ollama_fetcher(Ok(body));
    let limits = discover_model_limits_with(&fetcher, &cache(), &ollama_request())
        .await
        .unwrap();
    assert_eq!(limits.context_window, Some(16_384));
}

#[tokio::test]
async fn ollama_num_ctx_above_the_architecture_window_is_capped() {
    let mut body = show_body();
    body["parameters"] = json!("num_ctx 100000");
    let fetcher = ollama_fetcher(Ok(body));
    let limits = discover_model_limits_with(&fetcher, &cache(), &ollama_request())
        .await
        .unwrap();
    assert_eq!(limits.context_window, Some(40_960));
}

#[tokio::test]
async fn ollama_native_window_relabels_a_partial_listing_entry() {
    let fetcher = FakeFetcher::default()
        .with(
            &format!("{OLLAMA_ROOT}/v1/models"),
            json!({"data":[{"id":"qwen3:14b","top_provider":{"max_completion_tokens":2048}}]}),
        )
        .with_post(OLLAMA_SHOW, Ok(show_body()));
    let limits = discover_model_limits_with(&fetcher, &cache(), &ollama_request())
        .await
        .unwrap();
    assert_eq!(limits.context_window, Some(40_960));
    assert_eq!(limits.max_output_tokens, Some(2048));
    assert_eq!(limits.source, LimitSource::NativeApi);
}

#[tokio::test]
async fn partial_listing_entry_of_another_model_still_gets_its_own_native_probe() {
    let fetcher = FakeFetcher::default()
        .with(
            &format!("{OLLAMA_ROOT}/v1/models"),
            json!({"data":[
                {"id":"qwen3:14b"},
                {"id":"other:7b","top_provider":{"max_completion_tokens":512}}
            ]}),
        )
        .with_post(OLLAMA_SHOW, Ok(show_body()));
    let cache = cache();
    discover_model_limits_with(&fetcher, &cache, &ollama_request()).await;
    let other =
        DiscoveryRequest::new(format!("{OLLAMA_ROOT}/v1"), "other:7b").with_ollama_native(true);
    let limits = discover_model_limits_with(&fetcher, &cache, &other)
        .await
        .unwrap();
    assert_eq!(limits.context_window, Some(40_960));
    assert_eq!(fetcher.post_hits(), 2);
}

#[tokio::test]
async fn ollama_probe_is_off_when_disabled() {
    let fetcher = ollama_fetcher(Ok(show_body()));
    let request = ollama_request().with_ollama_native(false);
    let limits = discover_model_limits_with(&fetcher, &cache(), &request).await;
    assert!(limits.and_then(|l| l.context_window).is_none());
    assert_eq!(fetcher.post_hits(), 0);
}

#[tokio::test]
async fn failed_ollama_probe_is_cached_and_not_retried() {
    let fetcher = ollama_fetcher(Err(500));
    let cache = cache();
    for _ in 0..3 {
        let limits = discover_model_limits_with(&fetcher, &cache, &ollama_request()).await;
        assert!(limits.and_then(|l| l.context_window).is_none());
    }
    assert_eq!(fetcher.post_hits(), 1);
}

// Paused time: the runtime advances its clock itself once everything is idle,
// so the timeout fires deterministically without sleeping.
#[tokio::test(start_paused = true)]
async fn hanging_ollama_probe_is_bounded_by_the_timeout() {
    let mut fetcher = ollama_fetcher(Ok(show_body()));
    fetcher.post_hang = true;
    let request = ollama_request().with_timeout(Duration::from_millis(50));
    let cache = cache();
    let limits = discover_model_limits_with(&fetcher, &cache, &request).await;
    assert!(limits.and_then(|l| l.context_window).is_none());
    // The timeout is remembered: no second hit.
    discover_model_limits_with(&fetcher, &cache, &request).await;
    assert_eq!(fetcher.post_hits(), 1);
}

#[test]
fn ollama_endpoints_are_detected_from_the_url() {
    for yes in [
        "http://127.0.0.1:11434/v1",
        "http://localhost:11434",
        "https://ollama.internal/v1",
    ] {
        assert!(crate::model::discover::looks_like_ollama(yes), "{yes}");
    }
    for no in ["https://openrouter.ai/api/v1", "http://localhost:1234/v1"] {
        assert!(!crate::model::discover::looks_like_ollama(no), "{no}");
    }
}
