//! Local Ollama / piper stack — implementation split across submodules.

#![allow(
    dead_code,
    missing_docs,
    clippy::await_holding_lock,
    clippy::field_reassign_with_default,
    reason = "runtime internals and wire-shaped settings are exercised across feature-specific hosts"
)]

mod assets;
mod bootstrap;
mod lm_studio;
mod model_rpc;
mod ollama_admin;
mod public_infer;
pub use ollama_admin::test_ollama_connection;
pub mod paths;
mod vision_embed;

use crate::status::LocalAiStatus;
use parking_lot::Mutex;
use std::path::PathBuf;

#[cfg(test)]
static INFERENCE_TEST_MUTEX: once_cell::sync::Lazy<std::sync::Mutex<()>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(()));

#[cfg(test)]
pub(crate) fn inference_test_guard() -> std::sync::MutexGuard<'static, ()> {
    INFERENCE_TEST_MUTEX
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

pub(crate) mod presets_adapter {
    use super::LocalRuntimeSettings;
    use crate::presets::{ModelTier, VisionMode, all_presets, preset_for_tier};

    pub(crate) fn vision_mode_for_config(config: &LocalRuntimeSettings) -> VisionMode {
        match current_tier_from_config(config) {
            ModelTier::Custom if config.vision_model_id.trim().is_empty() => VisionMode::Disabled,
            ModelTier::Custom if config.preload_vision_model => VisionMode::Bundled,
            ModelTier::Custom => VisionMode::Ondemand,
            tier => crate::presets::vision_mode_for_tier(tier),
        }
    }

    pub(crate) fn apply_preset_to_config(config: &mut LocalRuntimeSettings, tier: ModelTier) {
        let Some(preset) = preset_for_tier(tier) else {
            return;
        };
        config.model_id = preset.chat_model_id.to_string();
        config.chat_model_id = preset.chat_model_id.to_string();
        config.vision_model_id = preset.vision_model_id.to_string();
        config.embedding_model_id = preset.embedding_model_id.to_string();
        config.quantization = preset.quantization.to_string();
        config.preload_vision_model = matches!(preset.vision_mode, VisionMode::Bundled);
        config.preload_embedding_model = true;
        config.selected_tier = Some(tier.as_str().to_string());
        config.runtime_enabled = true;
    }

    pub(crate) fn current_tier_from_config(config: &LocalRuntimeSettings) -> ModelTier {
        if let Some(tier) = config
            .selected_tier
            .as_deref()
            .and_then(ModelTier::from_str_opt)
            && (tier == ModelTier::Custom
                || preset_for_tier(tier).is_some_and(|preset| preset_matches(&preset, config)))
        {
            return tier;
        }
        all_presets()
            .into_iter()
            .find(|preset| preset_matches(preset, config))
            .map_or(ModelTier::Custom, |preset| preset.tier)
    }

    fn preset_matches(preset: &crate::presets::ModelPreset, config: &LocalRuntimeSettings) -> bool {
        let vision_matches = if matches!(preset.vision_mode, VisionMode::Disabled) {
            config.vision_model_id.trim().is_empty()
        } else {
            config.vision_model_id == preset.vision_model_id
        };
        config.chat_model_id == preset.chat_model_id
            && vision_matches
            && config.embedding_model_id == preset.embedding_model_id
    }
}

/// Host-selected local runtime settings consumed by the runtime service.
#[derive(Clone, Debug, Default)]
pub struct LocalRuntimeSettings {
    pub runtime_enabled: bool,
    pub provider: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model_id: String,
    pub chat_model_id: String,
    pub vision_model_id: String,
    pub embedding_model_id: String,
    pub stt_model_id: String,
    pub stt_download_url: Option<String>,
    pub tts_voice_id: String,
    pub tts_download_url: Option<String>,
    pub tts_config_download_url: Option<String>,
    pub quantization: String,
    pub preload_vision_model: bool,
    pub preload_embedding_model: bool,
    pub preload_stt_model: bool,
    pub preload_tts_voice: bool,
    pub download_url: Option<String>,
    pub autosummary_debounce_ms: u64,
    pub selected_tier: Option<String>,
    pub opt_in_confirmed: bool,
    pub ollama_binary_path: Option<String>,
    pub num_ctx: Option<u32>,
}

/// Complete host snapshot needed by local inference execution.
#[derive(Clone, Debug, Default)]
pub struct RuntimeConfig {
    pub local_ai: LocalRuntimeSettings,
    pub workspace_dir: PathBuf,
    pub config_path: PathBuf,
    pub shared_root_dir: PathBuf,
    pub default_temperature: f64,
}

impl crate::models::LocalModelConfig for RuntimeConfig {
    fn local_provider_name(&self) -> &str {
        &self.local_ai.provider
    }
    fn local_chat_model_id(&self) -> &str {
        &self.local_ai.chat_model_id
    }
    fn local_legacy_model_id(&self) -> &str {
        &self.local_ai.model_id
    }
    fn local_vision_model_id(&self) -> &str {
        &self.local_ai.vision_model_id
    }
    fn local_embedding_model_id(&self) -> &str {
        &self.local_ai.embedding_model_id
    }
    fn local_stt_model_id(&self) -> &str {
        &self.local_ai.stt_model_id
    }
    fn local_tts_voice_id(&self) -> &str {
        &self.local_ai.tts_voice_id
    }
    fn local_quantization(&self) -> &str {
        &self.local_ai.quantization
    }
}

pub struct LocalAiService {
    pub(crate) status: Mutex<LocalAiStatus>,
    pub(crate) bootstrap_lock: tokio::sync::Mutex<()>,
    pub(crate) last_memory_summary_at: Mutex<Option<std::time::Instant>>,
    pub(crate) http: reqwest::Client,
    /// Handle to any `ollama serve` openhuman itself spawned. `None` when
    /// the daemon currently on `:11434` was started outside openhuman (and
    /// adopted via the health probe) — those are never killed on exit.
    pub(crate) owned_ollama: Mutex<Option<tokio::process::Child>>,
}

impl std::fmt::Debug for LocalAiService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalAiService")
            .field("status", &self.status.lock())
            .field("has_owned_ollama", &self.owned_ollama.lock().is_some())
            .finish_non_exhaustive()
    }
}

impl LocalAiService {
    /// Replaces the observable runtime state and returns its previous value.
    ///
    /// Hosts may use this when an external capability changes readiness.
    pub fn replace_status_state(&self, state: impl Into<String>) -> String {
        std::mem::replace(&mut self.status.lock().state, state.into())
    }

    /// Marks hosted speech recognition ready after a host-owned STT call.
    pub fn mark_stt_ready(&self) {
        self.status.lock().stt_state = "ready".to_string();
    }

    /// Marks local speech synthesis ready after a host-owned TTS call.
    pub fn mark_tts_ready(&self) {
        self.status.lock().tts_state = "ready".to_string();
    }
    /// Returns `true` iff openhuman currently holds an owned Ollama child handle.
    ///
    /// Intended for tests and health-check callers that need to inspect the
    /// ownership state without going through the full bootstrap path.
    pub fn has_owned_ollama(&self) -> bool {
        self.owned_ollama.lock().is_some()
    }

    /// Inject a pre-spawned child as the owned Ollama handle.
    ///
    /// This allows integration tests to set up the ownership state without
    /// running the full `start_and_wait_for_server` path (which requires a
    /// real Ollama binary). Production code uses the internal field directly
    /// inside `ollama_admin.rs`; this method is the public bridge for the
    /// `tests/` integration test crate.
    pub fn inject_owned_ollama(&self, child: tokio::process::Child) {
        *self.owned_ollama.lock() = Some(child);
    }
}
