//! OpenAI-compatible multipart speech-to-text transport.

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use reqwest::header::AUTHORIZATION;
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};

/// Maximum accepted base64 audio input length.
pub const MAX_AUDIO_BASE64_LEN: usize = 33_554_432;
/// Default hosted transcription model.
pub const DEFAULT_MODEL: &str = "whisper-v1";

/// Caller-tunable transcription fields.
#[derive(Debug, Default, Clone)]
pub struct CloudTranscribeOptions {
    /// Optional model override.
    pub model: Option<String>,
    /// Optional language hint.
    pub language: Option<String>,
    /// Audio MIME type.
    pub mime_type: Option<String>,
    /// Original file-name hint.
    pub file_name: Option<String>,
}

/// Normalized transcription response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CloudTranscribeResult {
    /// Transcribed text.
    pub text: String,
}

/// Upload base64 audio to an OpenAI-compatible transcription endpoint.
pub async fn transcribe(
    client: &reqwest::Client,
    url: reqwest::Url,
    bearer_token: &str,
    audio_base64: &str,
    options: &CloudTranscribeOptions,
) -> Result<CloudTranscribeResult, String> {
    let trimmed = audio_base64.trim();
    if trimmed.is_empty() {
        return Err("audio_base64 is required".to_string());
    }
    if trimmed.len() > MAX_AUDIO_BASE64_LEN {
        return Err(format!(
            "audio_base64 exceeds maximum size ({}MB)",
            MAX_AUDIO_BASE64_LEN / 1_048_576
        ));
    }
    let bytes = BASE64
        .decode(trimmed)
        .map_err(|error| format!("invalid base64 audio: {error}"))?;
    if bytes.is_empty() {
        return Err("decoded audio is empty".to_string());
    }
    let mime = nonempty(options.mime_type.as_deref()).unwrap_or("audio/webm");
    let file_name = nonempty(options.file_name.as_deref()).unwrap_or("audio.webm");
    let model = nonempty(options.model.as_deref()).unwrap_or(DEFAULT_MODEL);
    let part = Part::bytes(bytes)
        .file_name(file_name.to_string())
        .mime_str(mime)
        .map_err(|error| format!("invalid mime '{mime}': {error}"))?;
    let mut form = Form::new()
        .part("file", part)
        .text("model", model.to_string());
    if let Some(language) = nonempty(options.language.as_deref()) {
        form = form.text("language", language.to_string());
    }
    let response = client
        .post(url)
        .header(AUTHORIZATION, format!("Bearer {bearer_token}"))
        .multipart(form)
        .send()
        .await
        .map_err(|error| format!("transcription request failed: {error}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("read transcription response failed: {error}"))?;
    if !status.is_success() {
        return Err(format!("transcription request failed ({status}): {body}"));
    }
    let value: serde_json::Value = serde_json::from_str(&body)
        .map_err(|error| format!("parse transcription response failed: {error}; body={body}"))?;
    let text = value
        .get("text")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("transcription response missing string `text`: {body}"))?
        .trim()
        .to_string();
    Ok(CloudTranscribeResult { text })
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}
