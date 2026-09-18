use super::*;
use axum::{Json, Router, routing::post};
use serde_json::json;

async fn spawn_mock(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://127.0.0.1:{}", addr.port())
}

fn enabled_config() -> Config {
    let mut config = Config::default();
    config.local_ai.runtime_enabled = true;
    config
}

fn openai_response(content: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl-test",
        "choices": [{
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8 }
    })
}

fn lm_studio_config(base: &str) -> Config {
    let mut config = enabled_config();
    config.local_ai.provider = "lm_studio".to_string();
    config.local_ai.base_url = Some(format!("{base}/v1"));
    config.local_ai.model_id = "local-model".to_string();
    config.local_ai.chat_model_id = "local-model".to_string();
    config.local_ai.opt_in_confirmed = true;
    config
}

/// Build a LocalAiService pre-seeded to `ready` so inference calls skip
/// `bootstrap()` and hit the HTTP path directly.
fn ready_service(config: &Config) -> LocalAiService {
    let service = LocalAiService::new(config);
    {
        let mut guard = service.status.lock();
        guard.state = "ready".to_string();
    }
    service
}

#[tokio::test]
async fn inference_hits_ollama_chat_completions_and_returns_response() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/v1/chat/completions",
        post(|Json(_body): Json<serde_json::Value>| async move {
            let mut response = openai_response("hello from mock");
            response["prompt_eval_count"] = json!(5);
            response["prompt_eval_duration"] = json!(100_000u64);
            response["eval_count"] = json!(3);
            response["eval_duration"] = json!(500_000u64);
            Json(response)
        }),
    );
    let base = spawn_mock(app).await;
    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", &base);
    }

    let config = enabled_config();
    let service = ready_service(&config);
    let reply = service
        .prompt(&config, "hi", Some(16), true)
        .await
        .expect("ollama prompt");
    assert_eq!(reply, "hello from mock");

    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }
}

#[tokio::test]
async fn inference_errors_on_non_success_status() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async { (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
    );
    let base = spawn_mock(app).await;
    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", &base);
    }

    let config = enabled_config();
    let service = ready_service(&config);
    let err = service.prompt(&config, "hi", None, true).await.unwrap_err();
    assert!(err.contains("500"));

    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }
}

#[tokio::test]
async fn inference_connection_failure_mentions_external_ollama_runtime() {
    let _guard = crate::service::inference_test_guard();

    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", "http://127.0.0.1:1");
    }

    let config = enabled_config();
    let service = ready_service(&config);
    let err = service.prompt(&config, "hi", None, true).await.unwrap_err();

    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }

    assert!(
        err.contains("external Ollama endpoint"),
        "unexpected error: {err}"
    );
    assert!(err.contains("already running"), "unexpected error: {err}");
}

#[tokio::test]
async fn inference_errors_on_empty_response_when_allow_empty_false() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async { Json(openai_response("   ")) }),
    );
    let base = spawn_mock(app).await;
    unsafe {
        std::env::set_var("OPENHUMAN_OLLAMA_BASE_URL", &base);
    }

    let config = enabled_config();
    let service = ready_service(&config);
    // `inference()` is the lower-level entry that hard-codes
    // allow_empty=false, so a whitespace-only mock response must
    // surface as the "empty content" error.
    let res = service.inference(&config, "", "hi", None, false).await;

    unsafe {
        std::env::remove_var("OPENHUMAN_OLLAMA_BASE_URL");
    }

    let err = res.expect_err("whitespace response must be rejected when allow_empty=false");
    assert!(
        err.contains("empty"),
        "expected an empty-content error, got: {err}"
    );
}

#[tokio::test]
async fn lm_studio_prompt_hits_openai_chat_completions() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/v1/chat/completions",
        post(|Json(body): Json<serde_json::Value>| async move {
            assert_eq!(body["model"], "local-model");
            assert!(body.get("stream").is_none());
            assert_eq!(body["max_tokens"], 16);
            assert_eq!(body["messages"][0]["role"], "system");
            assert_eq!(body["messages"][1]["role"], "user");
            Json(json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "hello from lm studio" },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11 }
            }))
        }),
    );
    let base = spawn_mock(app).await;
    let config = lm_studio_config(&base);
    let service = ready_service(&config);

    let reply = service
        .prompt(&config, "hi", Some(16), true)
        .await
        .expect("lm studio prompt");

    assert_eq!(reply, "hello from lm studio");
    let status = service.status();
    assert_eq!(status.provider, "lm_studio");
    assert_eq!(status.state, "ready");
}

#[tokio::test]
async fn lm_studio_prompt_errors_on_non_success_status() {
    let _guard = crate::service::inference_test_guard();

    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async { (axum::http::StatusCode::BAD_GATEWAY, "not ready") }),
    );
    let base = spawn_mock(app).await;
    let config = lm_studio_config(&base);
    let service = ready_service(&config);

    let err = service.prompt(&config, "hi", None, true).await.unwrap_err();

    assert!(err.contains("502"));
}

#[tokio::test]
async fn summarize_disabled_returns_error() {
    // When local_ai is disabled the summarize fn should short-circuit.
    let mut config = Config::default();
    config.local_ai.runtime_enabled = false;
    let service = LocalAiService::new(&config);
    let err = service.summarize(&config, "text", None).await.unwrap_err();
    assert!(err.contains("local ai is disabled"));
}

#[tokio::test]
async fn prompt_disabled_returns_error() {
    let mut config = Config::default();
    config.local_ai.runtime_enabled = false;
    let service = LocalAiService::new(&config);
    let err = service
        .prompt(&config, "text", None, false)
        .await
        .unwrap_err();
    assert!(err.contains("local ai is disabled"));
}

#[tokio::test]
async fn direct_interactive_inference_honors_runtime_disablement() {
    let mut config = Config::default();
    config.local_ai.runtime_enabled = false;
    let service = LocalAiService::new(&config);
    let error = service
        .inference_interactive(&config, "system", "prompt", None, true)
        .await
        .unwrap_err();
    assert!(error.contains("local ai is disabled"));
}

#[test]
fn runtime_debug_output_redacts_api_key() {
    let mut config = Config::default();
    config.local_ai.api_key = Some("local-secret".to_string());
    let debug = format!("{config:?}");
    assert!(!debug.contains("local-secret"));
    assert!(debug.contains("[REDACTED]"));
}

#[tokio::test]
async fn inline_complete_disabled_returns_empty_string() {
    let mut config = Config::default();
    config.local_ai.runtime_enabled = false;
    let service = LocalAiService::new(&config);
    let out = service
        .inline_complete(&config, "ctx", "casual", None, &[], None)
        .await
        .unwrap();
    assert!(out.is_empty());
}

#[tokio::test]
async fn inline_complete_interactive_disabled_returns_empty_string() {
    // Interactive variant must match the standard variant on the
    // disabled short-circuit so the autocomplete UX is identical.
    let mut config = Config::default();
    config.local_ai.runtime_enabled = false;
    let service = LocalAiService::new(&config);
    let out = service
        .inline_complete_interactive(&config, "ctx", "casual", None, &[], None)
        .await
        .unwrap();
    assert!(out.is_empty());
}
