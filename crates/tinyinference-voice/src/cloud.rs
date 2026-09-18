//! OpenAI-compatible multipart speech-to-text transport.

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use reqwest::header::AUTHORIZATION;
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};

/// Maximum accepted base64 audio input length.
pub const MAX_AUDIO_BASE64_LEN: usize = 33_554_432;
const MAX_RESPONSE_BYTES: usize = 64 * 1024;
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
    let body = bounded_response_body(response).await?;
    let safe_body = sanitize_response_detail(&body, bearer_token);
    if !status.is_success() {
        return Err(format!(
            "transcription request failed ({status}): {safe_body}"
        ));
    }
    let value: serde_json::Value = serde_json::from_str(&body).map_err(|error| {
        format!("parse transcription response failed: {error}; body={safe_body}")
    })?;
    let text = value
        .get("text")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("transcription response missing string `text`: {safe_body}"))?
        .trim()
        .to_string();
    Ok(CloudTranscribeResult { text })
}

async fn bounded_response_body(mut response: reqwest::Response) -> Result<String, String> {
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| format!("read transcription response failed: {error}"))?
    {
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err(format!(
                "transcription response exceeds {MAX_RESPONSE_BYTES} bytes"
            ));
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body)
        .map_err(|error| format!("transcription response was not UTF-8: {error}"))
}

fn sanitize_response_detail(body: &str, bearer_token: &str) -> String {
    let redacted = if bearer_token.trim().is_empty() {
        body.to_string()
    } else {
        body.replace(bearer_token.trim(), "[REDACTED]")
    };
    tinyinference_core::sanitize::sanitize_api_error(&redacted)
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_sanitization_removes_exact_and_prefixed_secrets() {
        let detail = sanitize_response_detail(
            "authorization=opaque-token and sk-provider-secret",
            "opaque-token",
        );
        assert!(!detail.contains("opaque-token"));
        assert!(!detail.contains("sk-provider-secret"));
        assert!(detail.contains("[REDACTED]"));
    }
}
