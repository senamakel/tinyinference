use super::*;

fn unsigned_jwt(payload: serde_json::Value) -> String {
    let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"none"}"#);
    let payload = URL_SAFE_NO_PAD.encode(payload.to_string());
    format!("{header}.{payload}.")
}

#[test]
fn codex_config_contains_the_public_client_contract() {
    let config = openai_codex_config("http://127.0.0.1:1455/auth/callback");
    assert_eq!(config.client_id, OPENAI_CODEX_CLIENT_ID);
    assert_eq!(config.authorize_url, OPENAI_AUTHORIZE_URL);
    assert_eq!(config.token_url, OPENAI_TOKEN_URL);
    assert!(config.scopes.iter().any(|scope| scope == "offline_access"));
}

#[test]
fn codex_cli_parser_normalizes_tokens_and_reads_jwt_metadata() {
    let access_token = unsigned_jwt(serde_json::json!({
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
        "exp": 2_000_000_000_i64,
    }));
    let bytes = serde_json::to_vec(&serde_json::json!({
        "tokens": {
            "access_token": access_token,
            "refresh_token": " refresh ",
            "id_token": " id "
        }
    }))
    .expect("fixture");
    let imported = parse_openai_codex_auth_json(&bytes).expect("parse");
    assert_eq!(imported.account_id.as_deref(), Some("acct_123"));
    assert_eq!(imported.expires_at_unix, Some(2_000_000_000));
    assert_eq!(imported.token.refresh_token.as_deref(), Some("refresh"));
    assert_eq!(imported.token.id_token.as_deref(), Some("id"));
}

#[test]
fn codex_cli_parser_rejects_missing_access_token() {
    let error = parse_openai_codex_auth_json(br#"{"tokens":{}}"#).unwrap_err();
    assert!(error.contains("access token"));
}

#[test]
fn debug_output_redacts_oauth_credentials() {
    let token = OAuthTokenSet {
        access_token: "access-secret".to_string(),
        refresh_token: Some("refresh-secret".to_string()),
        id_token: Some("id-secret".to_string()),
        expires_in: 3600,
        issued_at: 1,
    };
    let debug = format!("{token:?}");
    assert!(!debug.contains("access-secret"));
    assert!(!debug.contains("refresh-secret"));
    assert!(!debug.contains("id-secret"));
    assert!(debug.contains("[REDACTED]"));
}

fn flow_config(token_url: &str) -> OAuthConfig {
    OAuthConfig {
        client_id: "client".to_string(),
        client_secret: None,
        authorize_url: "https://auth.example/authorize".to_string(),
        token_url: token_url.to_string(),
        redirect_uri: "http://127.0.0.1:1455/cb".to_string(),
        scopes: vec!["openid".to_string()],
        extra_authorize_params: Vec::new(),
        state_equals_verifier: false,
        pending_secret: "test-oauth-pending".to_string(),
    }
}

fn memory_flow(token_url: &str) -> (OAuthFlow, Arc<dyn tinystoragedrivers_api::SecretStore>) {
    let scope = tinystoragedrivers_types::Scope::new("oauth-test").unwrap();
    let secrets = tinystoragedrivers_testkit::memory_storage(&scope)
        .secrets()
        .clone();
    (
        OAuthFlow::new(flow_config(token_url), secrets.clone()),
        secrets,
    )
}

async fn token_server(hits: Arc<std::sync::atomic::AtomicUsize>) -> String {
    let app = axum::Router::new().route(
        "/token",
        axum::routing::post(move |body: String| {
            let hits = hits.clone();
            async move {
                hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                assert!(body.contains("code_verifier="));
                axum::Json(serde_json::json!({
                    "access_token": "at", "refresh_token": "rt", "expires_in": 60
                }))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}/token")
}

#[tokio::test]
async fn start_keeps_pending_state_in_the_secret_store_not_on_disk() {
    let (flow, secrets) = memory_flow("http://127.0.0.1:9/token");
    let start = flow.start().await.unwrap();
    assert!(start.auth_url.contains("code_challenge="));
    let stored = secrets.get("test-oauth-pending").await.unwrap().unwrap();
    let pending: PendingOAuth = serde_json::from_slice(&stored).unwrap();
    assert_eq!(pending.state, start.state);
    assert!(!pending.verifier.is_empty());
}

#[tokio::test]
async fn state_equals_verifier_uses_the_stored_verifier_as_state() {
    let scope = tinystoragedrivers_types::Scope::new("oauth-state-test").unwrap();
    let secrets = tinystoragedrivers_testkit::memory_storage(&scope)
        .secrets()
        .clone();
    let mut config = flow_config("http://127.0.0.1:9/token");
    config.state_equals_verifier = true;
    let flow = OAuthFlow::new(config, secrets.clone());
    let start = flow.start().await.unwrap();
    let stored = secrets.get("test-oauth-pending").await.unwrap().unwrap();
    let pending: PendingOAuth = serde_json::from_slice(&stored).unwrap();
    assert_eq!(pending.verifier, start.state);
}

#[tokio::test]
async fn reads_pending_secret_bytes_written_by_v041() {
    // This byte fixture was emitted by the v0.4.1 PendingOAuth serializer; its
    // provenance and regeneration command live beside the fixture.
    let (flow, secrets) = memory_flow("http://127.0.0.1:9/token");
    let mut released_v041_bytes =
        include_bytes!("../tests/fixtures/oauth-pending-v041.json").to_vec();
    let marker = b"\"created_at\":";
    let timestamp_start = released_v041_bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .expect("released fixture has the created_at field")
        + marker.len();
    let timestamp_end = timestamp_start
        + released_v041_bytes[timestamp_start..]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
    released_v041_bytes.splice(
        timestamp_start..timestamp_end,
        unix_now_secs().to_string().bytes(),
    );
    let released_pending: PendingOAuth = serde_json::from_slice(&released_v041_bytes).unwrap();
    assert_eq!(released_pending.state, "expected");
    assert_eq!(released_pending.verifier, "released-verifier");
    assert_eq!(released_pending.redirect_uri, "http://127.0.0.1:1455/cb");
    secrets
        .set("test-oauth-pending", &released_v041_bytes)
        .await
        .unwrap();

    let error = flow.complete("code=abc&state=other").await.unwrap_err();
    assert!(error.contains("state mismatch"));
    assert!(secrets.get("test-oauth-pending").await.unwrap().is_none());
}

#[tokio::test]
async fn complete_without_a_pending_session_is_refused() {
    let (flow, _secrets) = memory_flow("http://127.0.0.1:9/token");
    let error = flow.complete("code=a&state=b").await.unwrap_err();
    assert!(error.contains("no pending OAuth session"));
}

#[tokio::test]
async fn state_mismatch_clears_the_pending_session() {
    let (flow, secrets) = memory_flow("http://127.0.0.1:9/token");
    flow.start().await.unwrap();
    let error = flow.complete("code=abc&state=forged").await.unwrap_err();
    assert!(error.contains("state mismatch"));
    assert!(secrets.get("test-oauth-pending").await.unwrap().is_none());
}

#[tokio::test]
async fn expired_pending_session_is_dropped_and_not_exchanged() {
    let (flow, secrets) = memory_flow("http://127.0.0.1:9/token");
    let stale = PendingOAuth {
        state: "s".to_string(),
        verifier: "v".to_string(),
        redirect_uri: "http://127.0.0.1:1455/cb".to_string(),
        created_at: unix_now_secs() - PENDING_TTL_SECS - 5,
    };
    secrets
        .set("test-oauth-pending", &serde_json::to_vec(&stale).unwrap())
        .await
        .unwrap();
    let error = flow.complete("code=abc&state=s").await.unwrap_err();
    assert!(error.contains("no pending OAuth session"));
    assert!(secrets.get("test-oauth-pending").await.unwrap().is_none());
}

#[tokio::test]
async fn corrupt_pending_state_is_an_error_not_a_panic() {
    let (flow, secrets) = memory_flow("http://127.0.0.1:9/token");
    secrets
        .set("test-oauth-pending", b"{not json")
        .await
        .unwrap();
    assert!(flow.complete("code=abc&state=s").await.is_err());
}

#[tokio::test]
async fn complete_exchanges_the_code_once_and_clears_pending() {
    let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let token_url = token_server(hits.clone()).await;
    let (flow, secrets) = memory_flow(&token_url);
    let start = flow.start().await.unwrap();
    let token = flow
        .complete(&format!(
            "http://127.0.0.1:1455/cb?code=abc&state={}",
            start.state
        ))
        .await
        .unwrap();
    assert_eq!(token.refresh_token.as_deref(), Some("rt"));
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(secrets.get("test-oauth-pending").await.unwrap().is_none());
    assert!(flow.complete("code=abc&state=x").await.is_err());
}

#[test]
fn callback_input_requires_code_and_state() {
    assert_eq!(
        parse_callback_input("code=a&state=b").unwrap(),
        ("a".to_string(), "b".to_string())
    );
    assert!(parse_callback_input("").is_err());
    assert!(parse_callback_input("nonsense").is_err());
    assert!(parse_callback_input("state=b").is_err());
}
