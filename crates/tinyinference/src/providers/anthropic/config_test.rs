use super::config::*;

#[test]
fn only_first_party_anthropic_endpoint_selects_messages_api() {
    assert!(endpoint_is_anthropic_messages(
        "https://api.anthropic.com/v1"
    ));
    assert!(!endpoint_is_anthropic_messages(
        "https://anthropic-proxy.example/v1"
    ));
}

#[test]
fn builds_a_native_anthropic_model_with_the_configured_profile() {
    let model = build_anthropic_model(AnthropicConfig {
        endpoint: "https://api.anthropic.com/v1",
        api_key: "sk-ant-secret",
        model: "claude-sonnet-4-6",
        temperature_override: Some(0.2),
        temperature_unsupported_models: &[],
    });
    let profile = model.profile().expect("anthropic models expose a profile");
    assert_eq!(profile.provider.as_deref(), Some("anthropic"));
    assert_eq!(profile.model.as_deref(), Some("claude-sonnet-4-6"));
    assert!(profile.tool_calling);
    assert!(profile.streaming);
    let identity = model.cache_identity().expect("model identity");
    assert!(identity.contains("api.anthropic.com"));
    assert!(!identity.contains("sk-ant-secret"));
}
