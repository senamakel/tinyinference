//! OpenAI Codex subscription-backend routing metadata and client identity.

use std::path::{Path, PathBuf};

/// ChatGPT account header used by the Codex backend.
pub const OPENAI_CODEX_ACCOUNT_HEADER: &str = "ChatGPT-Account-ID";
/// Codex subscription backend base URL.
pub const OPENAI_CODEX_BACKEND_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
/// Originator header used by Codex clients.
pub const OPENAI_CODEX_ORIGINATOR_HEADER: &str = "originator";
/// Originator value used by this adapter.
pub const OPENAI_CODEX_ORIGINATOR: &str = "codex_cli_rs";
/// Model hints advertised by the Codex backend.
pub const OPENAI_CODEX_MODEL_HINTS: &[&str] =
    &["gpt-5.5", "gpt-5.4", "gpt-5.3-codex-spark", "gpt-5.3-codex"];
// Conservative Codex CLI release known to work with the ChatGPT Codex backend.
// Bump this when field reports show the backend rejecting older client versions.
/// Conservative Codex client version fallback.
pub const OPENAI_CODEX_DEFAULT_CLIENT_VERSION: &str = "0.130.0";

/// Builds the Codex-compatible user-agent for an embedding host.
pub fn openai_codex_user_agent(product_name: &str, product_version: &str) -> String {
    format!("codex_cli_rs/0.0.0 ({product_name} {product_version})")
}

/// Resolved routing metadata for an OpenAI-compatible request.
#[derive(Debug, Clone)]
pub struct OpenAiCodexRouting {
    /// Effective endpoint.
    pub endpoint: String,
    /// Whether ChatGPT OAuth routing is active.
    pub using_oauth: bool,
    /// Optional ChatGPT account identifier.
    pub account_id: Option<String>,
}

impl OpenAiCodexRouting {
    /// Creates standard non-OAuth routing metadata.
    pub fn standard(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            using_oauth: false,
            account_id: None,
        }
    }
}

/// Resolves the client version from the environment and Codex metadata files.
pub fn openai_codex_client_version() -> String {
    tracing::trace!("[providers][openai-codex] client_version resolve start");
    let (version, source) = resolve_openai_codex_client_version();
    tracing::debug!(
        "[providers][openai-codex] resolved client_version source={source} value={version}"
    );
    version
}

fn resolve_openai_codex_client_version() -> (String, &'static str) {
    resolve_openai_codex_client_version_from(
        std::env::var("OPENAI_CODEX_CLIENT_VERSION").ok(),
        codex_home_dir(),
    )
}

fn resolve_openai_codex_client_version_from(
    version_override: Option<String>,
    codex_home: Option<PathBuf>,
) -> (String, &'static str) {
    if let Some(version) = version_override.and_then(non_empty_trimmed) {
        tracing::trace!("[providers][openai-codex] client_version source=env");
        return (version, "env");
    }

    if let Some(home) = codex_home {
        tracing::trace!(
            "[providers][openai-codex] client_version probing codex home={}",
            home.display()
        );
        if let Some(version) =
            read_json_string_field(&home.join("models_cache.json"), "client_version")
        {
            tracing::trace!("[providers][openai-codex] client_version source=models_cache");
            return (version, "models_cache");
        }
        if let Some(version) = read_json_string_field(&home.join("version.json"), "latest_version")
        {
            tracing::trace!("[providers][openai-codex] client_version source=version_json");
            return (version, "version_json");
        }
    } else {
        tracing::trace!("[providers][openai-codex] client_version codex home unresolved");
    }

    tracing::trace!("[providers][openai-codex] client_version source=default");
    (OPENAI_CODEX_DEFAULT_CLIENT_VERSION.to_string(), "default")
}

fn non_empty_trimmed(value: String) -> Option<String> {
    let trimmed = value.trim().to_string();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn codex_home_dir() -> Option<PathBuf> {
    if let Some(codex_home) = std::env::var_os("CODEX_HOME") {
        let path = PathBuf::from(codex_home);
        if !path.as_os_str().is_empty() {
            return Some(path);
        }
    }

    home_dir_from_env().map(|home| home.join(".codex"))
}

fn home_dir_from_env() -> Option<PathBuf> {
    for key in ["HOME", "USERPROFILE"] {
        if let Some(value) = std::env::var_os(key) {
            let path = PathBuf::from(value);
            if !path.as_os_str().is_empty() {
                return Some(path);
            }
        }
    }

    match (std::env::var_os("HOMEDRIVE"), std::env::var_os("HOMEPATH")) {
        (Some(drive), Some(path))
            if !drive.as_os_str().is_empty() && !path.as_os_str().is_empty() =>
        {
            Some(PathBuf::from(drive).join(path))
        }
        _ => None,
    }
}

fn read_json_string_field(path: &Path, field: &str) -> Option<String> {
    let file = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("<unknown>");
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) => {
            tracing::trace!(
                "[providers][openai-codex] client_version read miss file={file} err={err}"
            );
            return None;
        }
    };
    let json: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(json) => json,
        Err(err) => {
            tracing::trace!(
                "[providers][openai-codex] client_version parse miss file={file} err={err}"
            );
            return None;
        }
    };
    json.get(field)
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .and_then(non_empty_trimmed)
}

#[cfg(test)]
#[path = "codex_test.rs"]
mod tests;
