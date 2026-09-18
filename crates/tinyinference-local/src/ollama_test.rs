use super::*;

#[test]
fn chat_capability_classifies_embedding_completion_and_unknown() {
    let owned = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();

    // Embedding-only model (bge-m3) → not chat-capable. TAURI-RUST-4P6.
    assert_eq!(ollama_chat_capability(&owned(&["embedding"])), Some(false));
    // Chat/completion models → chat-capable.
    assert_eq!(ollama_chat_capability(&owned(&["completion"])), Some(true));
    assert_eq!(
        ollama_chat_capability(&owned(&["completion", "tools", "vision"])),
        Some(true)
    );
    assert_eq!(ollama_chat_capability(&owned(&["chat"])), Some(true));
    // A model exposing BOTH stays chat-capable (completion wins).
    assert_eq!(
        ollama_chat_capability(&owned(&["embedding", "completion"])),
        Some(true)
    );
    // Unknown / fail-open: empty, or a tag set we don't recognise → None
    // (caller keeps the model visible).
    assert_eq!(ollama_chat_capability(&[]), None);
    assert_eq!(ollama_chat_capability(&owned(&["insert"])), None);
    // Case / whitespace tolerant.
    assert_eq!(
        ollama_chat_capability(&owned(&[" Embedding "])),
        Some(false)
    );
    assert_eq!(ollama_chat_capability(&owned(&["COMPLETION"])), Some(true));
}

#[test]
fn pull_progress_aggregates_layered_download_events() {
    let mut progress = OllamaPullProgress::default();

    progress.observe(&OllamaPullEvent {
        status: Some("pulling".to_string()),
        digest: Some("sha256:layer-a".to_string()),
        total: Some(100),
        completed: Some(20),
        error: None,
    });
    progress.observe(&OllamaPullEvent {
        status: Some("pulling".to_string()),
        digest: Some("sha256:layer-b".to_string()),
        total: Some(200),
        completed: Some(50),
        error: None,
    });
    progress.observe(&OllamaPullEvent {
        status: Some("pulling".to_string()),
        digest: Some("sha256:layer-a".to_string()),
        total: Some(100),
        completed: Some(100),
        error: None,
    });

    assert_eq!(progress.aggregate_downloaded(), 150);
    assert_eq!(progress.aggregate_total(), Some(300));
}

#[test]
fn pull_progress_falls_back_when_digest_is_missing() {
    let mut progress = OllamaPullProgress::default();

    progress.observe(&OllamaPullEvent {
        status: Some("pulling manifest".to_string()),
        digest: None,
        total: Some(120),
        completed: Some(30),
        error: None,
    });
    progress.observe(&OllamaPullEvent {
        status: Some("pulling manifest".to_string()),
        digest: None,
        total: Some(120),
        completed: Some(80),
        error: None,
    });

    assert_eq!(progress.aggregate_downloaded(), 80);
    assert_eq!(progress.aggregate_total(), Some(120));
}

// ── /api/show context-length extraction ──────────────────────────

fn show_response(json: serde_json::Value) -> OllamaShowResponse {
    serde_json::from_value(json).expect("OllamaShowResponse")
}

#[test]
fn context_length_uses_general_architecture_prefix() {
    let resp = show_response(serde_json::json!({
        "model_info": {
            "general.architecture": "bert",
            "bert.context_length": 8192,
            "bert.embedding_length": 1024
        }
    }));
    assert_eq!(resp.context_length(), Some(8192));
}

#[test]
fn context_length_falls_back_when_architecture_missing() {
    let resp = show_response(serde_json::json!({
        "model_info": { "llama.context_length": 4096 }
    }));
    assert_eq!(resp.context_length(), Some(4096));
}

#[test]
fn context_length_handles_float_and_string_encodings() {
    // Some servers serialize the metadata number as a float.
    let float = show_response(serde_json::json!({
        "model_info": { "general.architecture": "qwen2", "qwen2.context_length": 32768.0 }
    }));
    assert_eq!(float.context_length(), Some(32768));

    // Non-numeric / missing → None (caller treats as Unknown, not a hard fail).
    let missing = show_response(serde_json::json!({ "model_info": {} }));
    assert_eq!(missing.context_length(), None);
    let absent_field = show_response(serde_json::json!({}));
    assert_eq!(absent_field.context_length(), None);
}

#[test]
fn context_length_prefers_architecture_key_over_unrelated_match() {
    let resp = show_response(serde_json::json!({
        "model_info": {
            "general.architecture": "llama",
            "llama.context_length": 8192,
            "clip.context_length": 77
        }
    }));
    assert_eq!(resp.context_length(), Some(8192));
}

#[test]
fn context_length_fallback_returns_max_not_first() {
    // Without `general.architecture`, the fallback must pick the *largest*
    // `.context_length` value, not the first one encountered. Multimodal
    // models can carry a low secondary value (e.g. `clip.context_length:77`)
    // which, if chosen first, would incorrectly mark the model below minimum.
    let resp = show_response(serde_json::json!({
        "model_info": {
            "clip.context_length": 77,
            "llama.context_length": 32768
        }
    }));
    assert_eq!(resp.context_length(), Some(32768));
}

// ── Ollama base URL resolution ─────────────────────────────────────

#[test]
fn base_url_resolution_honors_priority_and_normalization() {
    assert_eq!(resolve_ollama_base_url(None, None), DEFAULT_OLLAMA_BASE_URL);
    assert_eq!(
        resolve_ollama_base_url(Some("  http://127.0.0.1:55555///  "), None),
        "http://127.0.0.1:55555"
    );
    assert_eq!(
        resolve_ollama_base_url(None, Some("myhost:11434/")),
        "http://myhost:11434"
    );
    assert_eq!(
        resolve_ollama_base_url(Some("http://127.0.0.1:55555"), Some("192.168.1.5:11434")),
        "http://127.0.0.1:55555"
    );
    assert_eq!(
        resolve_ollama_base_url(None, Some("http://0.0.0.0:11434")),
        "http://localhost:11434"
    );
}

#[test]
fn configured_override_wins_without_process_environment_mutation() {
    assert_eq!(
        ollama_base_url_from_override(Some("http://192.168.1.5:11434")),
        "http://192.168.1.5:11434"
    );
}

// ── normalize_unspecified_host ──────────────────────────────────────

#[test]
fn normalize_rewrites_ipv4_unspecified() {
    assert_eq!(
        normalize_unspecified_host("http://0.0.0.0:11434"),
        "http://localhost:11434"
    );
}

#[test]
fn normalize_rewrites_ipv6_unspecified() {
    assert_eq!(
        normalize_unspecified_host("http://[::]:11434"),
        "http://[::1]:11434"
    );
}

#[test]
fn normalize_preserves_loopback() {
    assert_eq!(
        normalize_unspecified_host("http://127.0.0.1:11434"),
        "http://127.0.0.1:11434"
    );
    assert_eq!(
        normalize_unspecified_host("http://[::1]:11434"),
        "http://[::1]:11434"
    );
}

#[test]
fn normalize_preserves_named_host() {
    assert_eq!(
        normalize_unspecified_host("http://localhost:11434"),
        "http://localhost:11434"
    );
    assert_eq!(
        normalize_unspecified_host("http://my-ollama.lan:11434"),
        "http://my-ollama.lan:11434"
    );
}

#[test]
fn normalize_preserves_private_ip() {
    assert_eq!(
        normalize_unspecified_host("http://192.168.1.5:11434"),
        "http://192.168.1.5:11434"
    );
}

#[test]
fn normalize_handles_invalid_url() {
    assert_eq!(normalize_unspecified_host("not a url"), "not a url");
}

// ── ollama_base_url_from_override: 0.0.0.0 normalization ────────────

#[test]
fn ollama_base_url_from_override_normalizes_unspecified() {
    assert_eq!(
        ollama_base_url_from_override(Some("http://0.0.0.0:11434")),
        "http://localhost:11434"
    );
}

#[test]
fn ollama_base_url_from_override_normalizes_ipv6_unspecified() {
    assert_eq!(
        ollama_base_url_from_override(Some("http://[::]:11434")),
        "http://[::1]:11434"
    );
}

// ── validate_ollama_url ───────────────────────────────────────────

#[test]
fn validate_ollama_url_accepts_http() {
    assert_eq!(
        validate_ollama_url("http://localhost:11434"),
        Ok("http://localhost:11434".to_string())
    );
}

#[test]
fn validate_ollama_url_accepts_https() {
    assert_eq!(
        validate_ollama_url("https://remote-ollama.example.com:11434"),
        Ok("https://remote-ollama.example.com:11434".to_string())
    );
}

#[test]
fn validate_ollama_url_rejects_no_scheme() {
    assert!(validate_ollama_url("localhost:11434").is_err());
    assert!(validate_ollama_url("ftp://localhost:11434").is_err());
}

#[test]
fn validate_ollama_url_rejects_credentials() {
    assert!(validate_ollama_url("http://user:pass@localhost:11434").is_err());
}

#[test]
fn validate_ollama_url_strips_path_and_normalizes() {
    assert_eq!(
        validate_ollama_url("http://192.168.1.5:11434/api/tags"),
        Ok("http://192.168.1.5:11434".to_string())
    );
}

#[test]
fn validate_ollama_url_rejects_empty() {
    assert!(validate_ollama_url("").is_err());
    assert!(validate_ollama_url("   ").is_err());
}

#[test]
fn validate_ollama_url_handles_ipv6() {
    assert_eq!(
        validate_ollama_url("http://[::1]:11434"),
        Ok("http://[::1]:11434".to_string())
    );
}

#[test]
fn validate_ollama_url_rewrites_ipv4_unspecified_to_localhost() {
    assert_eq!(
        validate_ollama_url("http://0.0.0.0:11434"),
        Ok("http://localhost:11434".to_string())
    );
}

#[test]
fn validate_ollama_url_rewrites_ipv6_unspecified_to_loopback() {
    assert_eq!(
        validate_ollama_url("http://[::]:11434"),
        Ok("http://[::1]:11434".to_string())
    );
}

// ── public Ollama registry-host rejection (TAURI-RUST-A3T) ─────────

#[test]
fn is_ollama_registry_host_matches_public_hosts_case_insensitively() {
    for host in [
        "ollama.com",
        "OLLAMA.COM",
        "www.ollama.com",
        "ollama.ai",
        "www.ollama.ai",
        "registry.ollama.ai",
        "ollama.com.", // trailing FQDN dot
    ] {
        assert!(is_ollama_registry_host(host), "expected reject: {host}");
    }
}

#[test]
fn is_ollama_registry_host_allows_self_hosted_and_loopback() {
    // No suffix match: a user's own subdomain must NOT be blocked.
    for host in [
        "localhost",
        "127.0.0.1",
        "ollama.mycompany.com",
        "my-ollama.lan",
        "remote-ollama.example.com",
        "notollama.com",
    ] {
        assert!(!is_ollama_registry_host(host), "expected allow: {host}");
    }
}

#[test]
fn validate_ollama_url_rejects_public_registry_hosts() {
    // The bug: ollama.com is the website/registry, not a local server.
    // `/api/tags` against it returns HTTP 429 and floods Sentry.
    for url in [
        "https://ollama.com",
        "http://ollama.com",
        "https://ollama.com/",
        "https://www.ollama.com:443",
        "https://ollama.ai",
        "https://registry.ollama.ai",
    ] {
        let err = validate_ollama_url(url).unwrap_err();
        assert!(
            err.contains("ollama.com is the Ollama website"),
            "expected website rejection for {url}, got: {err}"
        );
    }
}

#[test]
fn validate_ollama_url_still_accepts_self_hosted_subdomain() {
    assert_eq!(
        validate_ollama_url("https://ollama.mycompany.com:11434"),
        Ok("https://ollama.mycompany.com:11434".to_string())
    );
}

#[test]
fn ollama_base_url_from_override_heals_registry_host_to_default() {
    // Existing users who saved ollama.com before validation rejected it
    // must stop hitting the website on the next launch.
    assert_eq!(
        ollama_base_url_from_override(Some("https://ollama.com")),
        DEFAULT_OLLAMA_BASE_URL
    );
}

#[test]
fn ollama_base_url_heals_registry_host_from_env_override() {
    assert_eq!(
        resolve_ollama_base_url(Some("https://ollama.com"), None),
        DEFAULT_OLLAMA_BASE_URL
    );
}

#[test]
fn ollama_base_url_heals_registry_host_from_ollama_host() {
    assert_eq!(
        resolve_ollama_base_url(None, Some("ollama.com")),
        DEFAULT_OLLAMA_BASE_URL
    );
}

// ── redact_ollama_base_url ────────────────────────────────────────

#[test]
fn redact_strips_userinfo_query_and_fragment() {
    assert_eq!(
        redact_ollama_base_url("http://user:pass@host:11434/api?token=abc#frag"),
        "http://host:11434/api"
    );
}

#[test]
fn redact_keeps_plain_url() {
    assert_eq!(
        redact_ollama_base_url("http://127.0.0.1:11434/"),
        "http://127.0.0.1:11434/"
    );
}

#[test]
fn redact_handles_invalid_url() {
    assert_eq!(redact_ollama_base_url("not a url"), "<invalid-endpoint>");
}
