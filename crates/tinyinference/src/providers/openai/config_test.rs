use super::AuthStyle;
use super::config::*;

#[test]
fn builds_a_chat_model_with_the_configured_profile() {
    let model = build_openai_model(OpenAiConfig {
        provider_name: "deepseek",
        endpoint: "https://api.deepseek.com/v1",
        api_key: "secret",
        auth_style: AuthStyle::Bearer,
        model: "deepseek-chat",
        temperature_unsupported_models: &[],
        temperature_override: None,
        merge_system_into_user: false,
        extra_headers: &[],
        native_tool_calling: None,
        vision: None,
        default_provider_options: None,
        responses_api_primary: false,
        responses_omit_max_output_tokens: false,
        extra_query_params: &[],
        user_agent: None,
        explicit_cache_control: false,
    });
    let profile = model.profile().expect("openai models expose a profile");
    assert_eq!(profile.provider.as_deref(), Some("deepseek"));
    assert_eq!(profile.model.as_deref(), Some("deepseek-chat"));
    assert!(profile.tool_calling);
}

#[test]
fn local_runtime_builder_disables_native_tools_and_vision() {
    let model = build_local_runtime_chat_model(
        "ollama",
        "http://localhost:11434/v1",
        "",
        AuthStyle::None,
        "qwen2.5",
        &[],
        None,
        Some(8192),
    );
    let profile = model.profile().expect("openai models expose a profile");
    assert!(!profile.tool_calling);
    assert!(!profile.modalities.image_in);
}

#[test]
fn openrouter_endpoints_are_recognized() {
    assert!(endpoint_is_openrouter("https://openrouter.ai/api/v1"));
    assert!(endpoint_is_openrouter("HTTPS://OpenRouter.ai:443/api/v1/"));
    assert!(!endpoint_is_openrouter("https://notopenrouter.ai/api/v1"));
}
