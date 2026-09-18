//! Tiered model presets and recommendation logic for local AI.
//!
//! Text generation is always the primary summarizer. Vision is a secondary
//! scene-description sidecar whose output can be merged with OCR by the text
//! model when a tier supports it.

use serde::{Deserialize, Serialize};

use crate::device::DeviceProfile;

/// Performance tier for local AI model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelTier {
    /// Approximately one gigabyte of model memory.
    #[serde(rename = "ram_1gb")]
    Ram1Gb,
    /// Approximately two to four gigabytes of model memory.
    #[serde(rename = "ram_2_4gb")]
    Ram2To4Gb,
    /// Approximately four to eight gigabytes of model memory.
    #[serde(rename = "ram_4_8gb")]
    Ram4To8Gb,
    /// Approximately eight to sixteen gigabytes of model memory.
    #[serde(rename = "ram_8_16gb")]
    Ram8To16Gb,
    /// Sixteen gigabytes or more of model memory.
    #[serde(rename = "ram_16_plus_gb")]
    Ram16PlusGb,
    /// A caller-defined model selection.
    #[serde(rename = "custom")]
    Custom,
}

/// Single local model tier exposed in the current build. Larger local models
/// stay blocked to keep summarization lightweight and battery-friendly.
pub const MVP_MAX_TIER: ModelTier = ModelTier::Ram2To4Gb;

/// Minimum host RAM (in whole GB) below which the **default** is to skip
/// local inference and use the cloud summarizer instead.  The user can still
/// override this and opt into local AI via settings.
pub const MIN_RAM_GB_FOR_LOCAL_AI: u64 = 8;

/// Returns `true` when the device has enough RAM that local AI should be
/// enabled by default. Below the floor we recommend cloud fallback instead.
pub fn device_supports_local_ai(device: &DeviceProfile) -> bool {
    device.total_ram_gb() >= MIN_RAM_GB_FOR_LOCAL_AI
}

/// Returns `true` when the device is below the RAM floor and local AI should
/// default to disabled (cloud fallback). This is a **recommendation**, not a
/// hard gate — the user can still opt in.
pub fn should_default_to_cloud_fallback(device: &DeviceProfile) -> bool {
    !device_supports_local_ai(device)
}

impl ModelTier {
    /// Returns the stable serialized tier identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ram1Gb => "ram_1gb",
            Self::Ram2To4Gb => "ram_2_4gb",
            Self::Ram4To8Gb => "ram_4_8gb",
            Self::Ram8To16Gb => "ram_8_16gb",
            Self::Ram16PlusGb => "ram_16_plus_gb",
            Self::Custom => "custom",
        }
    }

    /// Whether this tier is allowed in the current MVP build.
    pub fn is_mvp_allowed(self) -> bool {
        matches!(self, Self::Ram2To4Gb)
    }

    /// Parses a canonical tier identifier or supported legacy alias.
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "ram_1gb" | "tier_1gb" | "1gb" => Some(Self::Ram1Gb),
            "ram_2_4gb" | "tier_2_4gb" | "2_4gb" | "low" => Some(Self::Ram2To4Gb),
            "ram_4_8gb" | "tier_4_8gb" | "4_8gb" => Some(Self::Ram4To8Gb),
            "ram_8_16gb" | "tier_8_16gb" | "8_16gb" | "medium" => Some(Self::Ram8To16Gb),
            "ram_16_plus_gb" | "tier_16_plus_gb" | "16_plus_gb" | "high" => Some(Self::Ram16PlusGb),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// How a local preset makes vision inference available.
pub enum VisionMode {
    /// Vision inference is unavailable.
    Disabled,
    /// The vision model is loaded only when requested.
    Ondemand,
    /// The vision model is bundled or preloaded with the text model.
    Bundled,
}

/// A concrete model preset tied to a performance tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPreset {
    /// Performance tier represented by this preset.
    pub tier: ModelTier,
    /// Human-readable label.
    pub label: &'static str,
    /// Human-readable description.
    pub description: &'static str,
    /// Chat model identifier.
    pub chat_model_id: &'static str,
    /// Vision model identifier, or an empty string when disabled.
    pub vision_model_id: &'static str,
    /// Embedding model identifier.
    pub embedding_model_id: &'static str,
    /// Quantization label.
    pub quantization: &'static str,
    /// Vision loading mode.
    pub vision_mode: VisionMode,
    /// Whether the preset can summarize screen images.
    pub supports_screen_summary: bool,
    /// Intended device memory tier in gigabytes.
    pub target_ram_gb: u64,
    /// Minimum memory required in gigabytes.
    pub min_ram_gb: u64,
    /// Approximate total download size in gigabytes.
    pub approx_download_gb: f32,
}

/// Return all built-in presets.
pub fn all_presets() -> Vec<ModelPreset> {
    vec![
        ModelPreset {
            tier: ModelTier::Ram1Gb,
            label: "1 GB",
            description: "Fastest chat-only tier for ultra-low-memory devices. OCR + text only.",
            chat_model_id: "gemma3:270m-it-qat",
            vision_model_id: "",
            embedding_model_id: "all-minilm:latest",
            quantization: "qat",
            vision_mode: VisionMode::Disabled,
            supports_screen_summary: false,
            target_ram_gb: 1,
            min_ram_gb: 1,
            approx_download_gb: 0.3,
        },
        ModelPreset {
            tier: ModelTier::Ram2To4Gb,
            label: "Gemma 3 1B",
            description: "Battery-friendly local summarization preset. Uses the 1B Gemma model with vision disabled.",
            chat_model_id: "gemma3:1b-it-qat",
            vision_model_id: "",
            // bge-m3 — 1024 dims required by memory tree's on-disk format
            // and 8192-token context for long-chunk embeds.
            embedding_model_id: "bge-m3",
            quantization: "qat",
            vision_mode: VisionMode::Disabled,
            supports_screen_summary: false,
            target_ram_gb: 2,
            min_ram_gb: 2,
            approx_download_gb: 2.3,
        },
        ModelPreset {
            tier: ModelTier::Ram4To8Gb,
            label: "4-8 GB",
            description: "Light Gemma chat with on-demand vision summaries via Moondream.",
            chat_model_id: "gemma3:1b-it-qat",
            vision_model_id: "moondream:1.8b-v2-q4_K_S",
            embedding_model_id: "all-minilm:latest",
            quantization: "qat",
            vision_mode: VisionMode::Ondemand,
            supports_screen_summary: true,
            target_ram_gb: 4,
            min_ram_gb: 4,
            approx_download_gb: 2.8,
        },
        ModelPreset {
            tier: ModelTier::Ram8To16Gb,
            label: "8-16 GB",
            description: "Balanced Gemma multimodal preset with bundled vision support.",
            chat_model_id: "gemma3:4b-it-qat",
            // Gemma 3 is multimodal from 4B upward, so one model covers chat
            // and vision here. (The 270M and 1B builds are text-only.)
            vision_model_id: "gemma3:4b-it-qat",
            // bge-m3 (1024 dims). nomic-embed-text is 768 dims and fails the
            // memory tree's post-call dimension validator; the embedding
            // allowlist already rewrote it to bge-m3 at resolution time, so
            // naming bge-m3 here only makes the preset honest about what is
            // actually pulled.
            embedding_model_id: "bge-m3",
            quantization: "qat",
            vision_mode: VisionMode::Bundled,
            supports_screen_summary: true,
            target_ram_gb: 8,
            min_ram_gb: 8,
            // gemma3:4b-it-qat (4.0 GB) + bge-m3 (1.2 GB)
            approx_download_gb: 5.2,
        },
        ModelPreset {
            tier: ModelTier::Ram16PlusGb,
            label: "16 GB+",
            description: "Best local quality with Gemma 4 on higher-end devices.",
            // GH #5055 moved this tier off `gemma4:e4b` because no `gemma4`
            // namespace existed on the Ollama library at the time, and landed
            // on `gemma3n:e4b-it-q8_0`. Two things have changed (#5146 §1.3):
            //
            //  1. Gemma 4 has since been published, and `gemma4:e4b-it-q8_0`
            //     resolves (11.6 GB, 128K context).
            //  2. Gemma 3n is **text-only** on Ollama, so using it as the
            //     `vision_model_id` of a `Bundled` vision tier pointed every
            //     vision request at a model with no vision encoder. Ollama
            //     accepts the `images` array against such a model, discards
            //     it, and answers from the prompt text — a hallucinated
            //     description rather than an error.
            //
            // Gemma 4 is multimodal at every published size, so this tier is
            // back to one model serving both chat and vision, matching how the
            // 8-16 GB tier uses `gemma3:4b-it-qat`.
            chat_model_id: "gemma4:e4b-it-q8_0",
            vision_model_id: "gemma4:e4b-it-q8_0",
            // bge-m3 (1024 dims) — see the 8-16 GB tier note above.
            embedding_model_id: "bge-m3",
            // The other tiers ship QAT builds; this one is q8_0. The field is a
            // display label (`effective_quantization`) and does not take part in
            // resolving the model tag, but it should still match what is pulled.
            quantization: "q8_0",
            vision_mode: VisionMode::Bundled,
            supports_screen_summary: true,
            target_ram_gb: 16,
            min_ram_gb: 16,
            // gemma4:e4b-it-q8_0 (11.6 GB) + bge-m3 (1.2 GB)
            approx_download_gb: 12.8,
        },
    ]
}

/// Return only the presets allowed under the current MVP ceiling.
pub fn mvp_presets() -> Vec<ModelPreset> {
    all_presets()
        .into_iter()
        .filter(|preset| preset.tier.is_mvp_allowed())
        .collect()
}

/// Return the preset for a specific tier, or `None` for `Custom`.
pub fn preset_for_tier(tier: ModelTier) -> Option<ModelPreset> {
    all_presets().into_iter().find(|preset| preset.tier == tier)
}

/// Recommend a tier based on device capabilities.
pub fn recommend_tier(device: &DeviceProfile) -> ModelTier {
    let ram_gb = device.total_ram_gb();
    let tier = match ram_gb {
        0..=1 => ModelTier::Ram1Gb,
        2..=3 => ModelTier::Ram2To4Gb,
        4..=7 => ModelTier::Ram4To8Gb,
        8..=15 => ModelTier::Ram8To16Gb,
        _ => ModelTier::Ram16PlusGb,
    };
    tracing::debug!(ram_gb, ?tier, "[local_ai] recommended model tier");
    tier
}

/// Returns the vision loading mode associated with a tier.
pub fn vision_mode_for_tier(tier: ModelTier) -> VisionMode {
    match tier {
        ModelTier::Ram1Gb | ModelTier::Ram2To4Gb => VisionMode::Disabled,
        ModelTier::Ram4To8Gb => VisionMode::Ondemand,
        ModelTier::Ram8To16Gb | ModelTier::Ram16PlusGb => VisionMode::Bundled,
        ModelTier::Custom => VisionMode::Bundled,
    }
}

#[cfg(test)]
#[path = "presets_test.rs"]
mod presets_tests;

#[cfg(test)]
#[path = "presets_device_test.rs"]
mod device_tests;
