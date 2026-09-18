use serde::{Deserialize, Serialize};

/// One frame on an Oculus-compatible viseme timeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VisemeFrame {
    /// Viseme identifier such as `sil` or `aa`.
    pub viseme: String,
    /// Inclusive frame start in milliseconds.
    pub start_ms: u64,
    /// Exclusive frame end in milliseconds.
    pub end_ms: u64,
}

/// Normalized result of local Piper synthesis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PiperSpeech {
    /// Base64-encoded WAV bytes.
    pub audio_base64: String,
    /// MIME type of the generated audio.
    pub audio_mime: String,
    /// Synthetic fallback viseme timeline.
    pub visemes: Vec<VisemeFrame>,
}
