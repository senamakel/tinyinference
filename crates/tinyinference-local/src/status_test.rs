use super::*;

#[derive(Default)]
struct Config {
    provider: String,
    chat: String,
    legacy: String,
    vision: String,
    embedding: String,
    stt: String,
    tts: String,
    quantization: String,
}

impl crate::models::LocalModelConfig for Config {
    fn local_provider_name(&self) -> &str {
        &self.provider
    }
    fn local_chat_model_id(&self) -> &str {
        &self.chat
    }
    fn local_legacy_model_id(&self) -> &str {
        &self.legacy
    }
    fn local_vision_model_id(&self) -> &str {
        &self.vision
    }
    fn local_embedding_model_id(&self) -> &str {
        &self.embedding
    }
    fn local_stt_model_id(&self) -> &str {
        &self.stt
    }
    fn local_tts_voice_id(&self) -> &str {
        &self.tts
    }
    fn local_quantization(&self) -> &str {
        &self.quantization
    }
}

#[test]
fn disabled_status_marks_all_capabilities_disabled() {
    let config = Config::default();
    let status = LocalAiStatus::disabled(&config, "disabled");

    assert_eq!(status.state, "disabled");
    assert_eq!(status.vision_state, "disabled");
    assert_eq!(status.embedding_state, "disabled");
    assert_eq!(status.stt_state, "disabled");
    assert_eq!(status.tts_state, "disabled");
    assert_eq!(status.provider, "ollama");
    assert_eq!(status.active_backend, "ollama");
}

#[test]
fn disabled_status_reflects_lm_studio_provider() {
    use crate::provider::LocalAiProvider;

    let mut config = Config::default();
    config.provider = LocalAiProvider::LmStudio.as_str().to_string();
    let status = LocalAiStatus::disabled(&config, "disabled");

    assert_eq!(status.provider, "lm_studio");
    assert_eq!(status.active_backend, "lm_studio");
}

#[test]
fn disabled_status_uses_config_vision_mode() {
    let mut config = Config::default();
    config.chat = "gemma3:1b-it-qat".to_string();
    config.vision.clear();
    config.embedding = "all-minilm:latest".to_string();

    let status = LocalAiStatus::disabled(&config, "disabled");
    assert_eq!(status.vision_mode, "disabled");
}
