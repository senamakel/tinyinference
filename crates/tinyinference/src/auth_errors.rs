//! Process-lived registry of BYO provider auth failures (invalid / revoked
//! API key, HTTP 401 / 403).
//!
//! Hosts can use the registry to feed both one-shot notifications and live
//! provider-error status surfaces without duplicating failure state.
//!
//! Entries are recorded where a provider auth failure is classified and
//! cleared when the host updates or removes that provider key. The [`record`]
//! latch is what makes a notification
//! fire **once per failure episode** rather than once per retry: the
//! triggering 401 can repeat across retries and background work.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// A recorded BYO provider auth failure. Serialized verbatim onto the
/// `inference_provider_auth_errors` RPC snapshot the AI settings panel reads.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProviderAuthError {
    /// Provider slug as used by the chat factory / classifier, e.g.
    /// `"openrouter"`.
    pub provider: String,
    /// The rejecting HTTP status — always `401` or `403` (the classifier
    /// gate), kept for fidelity in the surfaced copy.
    pub status: u16,
    /// Pre-formatted, user-facing, actionable message — safe to show in a
    /// notification or settings notice as-is. See [`auth_error_message`].
    pub message: String,
    /// Wall-clock milliseconds since the unix epoch when last recorded.
    pub timestamp_ms: u64,
}

fn registry() -> &'static RwLock<HashMap<String, ProviderAuthError>> {
    static REG: OnceLock<RwLock<HashMap<String, ProviderAuthError>>> = OnceLock::new();
    REG.get_or_init(|| RwLock::new(HashMap::new()))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build the actionable, user-facing message for a BYO key the provider
/// rejected. Phrased so it reads the same in the notification center and the
/// AI-settings notice, and points the user at the one fix that resolves it
/// (the third-party key is invalid / revoked — OpenHuman has no other lever).
pub fn auth_error_message(provider: &str, status: u16) -> String {
    format!(
        "{provider} rejected the API key (HTTP {status}). Update your {provider} \
         API key in Connections → API keys → LLM to restore it."
    )
}

/// Record a BYO provider auth failure.
///
/// Returns `true` when this is a **new** failure episode for `provider` (no
/// prior entry) — callers publish the one-shot notification only on `true` so
/// a 401 that repeats per-retry doesn't re-notify. An existing entry is
/// refreshed (status / timestamp) and returns `false`.
pub fn record(provider: &str, status: u16) -> bool {
    let entry = ProviderAuthError {
        provider: provider.to_string(),
        status,
        message: auth_error_message(provider, status),
        timestamp_ms: now_ms(),
    };
    let mut map = registry().write().unwrap_or_else(|e| e.into_inner());
    let is_new = !map.contains_key(provider);
    map.insert(provider.to_string(), entry);
    is_new
}

/// Clear any recorded auth error for `provider` (the user updated / removed
/// the key, or a call newly succeeded). Returns `true` if an entry was
/// removed. Re-arms the [`record`] latch so a fresh failure re-notifies.
pub fn clear(provider: &str) -> bool {
    registry()
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .remove(provider)
        .is_some()
}

/// Snapshot all currently-recorded provider auth errors, sorted by provider
/// slug for a stable RPC payload.
pub fn snapshot() -> Vec<ProviderAuthError> {
    let map = registry().read().unwrap_or_else(|e| e.into_inner());
    let mut out: Vec<ProviderAuthError> = map.values().cloned().collect();
    out.sort_by(|a, b| a.provider.cmp(&b.provider));
    out
}

/// Test-only: wipe the registry so a test can't be flaked by an entry an
/// earlier test left behind (the map is process-global).
#[cfg(any(test, debug_assertions))]
pub fn reset_for_tests() {
    registry()
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
}

#[cfg(test)]
#[path = "auth_errors_test.rs"]
mod tests;
