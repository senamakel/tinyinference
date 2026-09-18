//! Local Piper text-to-speech execution.

use std::path::{Path, PathBuf};

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use tokio::io::AsyncWriteExt;

use crate::{PiperSpeech, VisemeFrame};

/// Run Piper with explicit binary and voice-model paths.
pub async fn synthesize(
    piper_binary: &Path,
    voice_model: &Path,
    text: &str,
) -> Result<PiperSpeech, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("text is required".to_string());
    }
    let output_path = temporary_output_path();
    let mut child = tokio::process::Command::new(piper_binary)
        .args([
            "--model",
            &voice_model.to_string_lossy(),
            "--output_file",
            &output_path.to_string_lossy(),
            "--length-scale",
            "1.15",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to launch piper: {error}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(text.as_bytes())
            .await
            .map_err(|error| format!("failed to write text to piper stdin: {error}"))?;
    }
    let output = child
        .wait_with_output()
        .await
        .map_err(|error| format!("failed to wait on piper: {error}"))?;
    if !output.status.success() {
        let _ = tokio::fs::remove_file(&output_path).await;
        return Err(format!(
            "piper failed (exit={:?}): {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let bytes = tokio::fs::read(&output_path)
        .await
        .map_err(|error| format!("failed to read piper output: {error}"))?;
    let _ = tokio::fs::remove_file(&output_path).await;
    Ok(PiperSpeech {
        audio_base64: BASE64.encode(bytes),
        audio_mime: "audio/wav".to_string(),
        visemes: synthetic_viseme_timeline(text),
    })
}

fn temporary_output_path() -> PathBuf {
    std::env::temp_dir().join(format!("tinyinference-piper-{}.wav", uuid::Uuid::new_v4()))
}

/// Build a neutral fallback viseme timeline from text length.
#[must_use]
pub fn synthetic_viseme_timeline(text: &str) -> Vec<VisemeFrame> {
    let duration = (text
        .chars()
        .filter(|character| !character.is_whitespace())
        .count()
        .max(1) as u64)
        * 80;
    vec![
        VisemeFrame {
            viseme: "sil".to_string(),
            start_ms: 0,
            end_ms: 40,
        },
        VisemeFrame {
            viseme: "aa".to_string(),
            start_ms: 40,
            end_ms: duration.max(80),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeline_is_nonempty_and_scales() {
        let short = synthetic_viseme_timeline("hi");
        let long = synthetic_viseme_timeline("the quick brown fox");
        assert_eq!(short[0].viseme, "sil");
        assert!(long.last().unwrap().end_ms > short.last().unwrap().end_ms);
    }
}
