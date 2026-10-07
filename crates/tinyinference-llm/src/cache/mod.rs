//! Per-request cache policy and JSON canonicalization.
//!
//! [`CachePolicy`] is the per-request knob a [`crate::model::ModelRequest`]
//! carries (`cache_policy`): whether a host may serve the call from its local
//! response cache, whether the provider prompt-cache prefix must be protected,
//! and the entry TTL and key namespace. This crate only carries the policy and
//! honours `protect_prompt_prefix` in the providers that map it onto wire
//! cache markers; the response cache itself, its keys and the prompt-layout
//! guard live in the host harness (`tinyagents-harness`'s `cache` module).
//!
//! [`canonical_value`] sorts JSON object keys so a host's cache keys hash
//! equal requests equally regardless of field insertion order.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Policy knobs controlling both response caching and provider prompt-cache
/// layout protection.
///
/// Both flags default to `false` (no caching / no protection) so a host is
/// safe-by-default and opts must be explicit.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePolicy {
    /// When `true`, the host may look up (and write) local response cache
    /// entries before calling the provider.
    pub response_cache_enabled: bool,
    /// When `true`, middleware must preserve the order and content of cacheable
    /// prefix segments, and providers that support it mark the stable prefix
    /// for their own prompt cache.
    pub protect_prompt_prefix: bool,
    /// Entry time-to-live in milliseconds; `None` means no expiry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_ms: Option<u64>,
    /// Optional cache-key namespace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

impl CachePolicy {
    /// Creates a policy with response caching enabled and no expiry.
    pub fn enabled() -> Self {
        Self {
            response_cache_enabled: true,
            ..Self::default()
        }
    }

    /// Returns the configured entry TTL.
    pub fn ttl(&self) -> Option<std::time::Duration> {
        self.ttl_ms.map(std::time::Duration::from_millis)
    }

    /// Sets the entry TTL.
    pub fn with_ttl(mut self, ttl: std::time::Duration) -> Self {
        self.ttl_ms = Some(ttl.as_millis() as u64);
        self
    }

    /// Sets the cache-key namespace.
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }
}

/// Recursively sorts the keys of every JSON object so that the serialized form
/// is canonical regardless of insertion order.
///
/// Public so every cache key derived from JSON — this crate's and a host's —
/// canonicalizes the same way; two copies that drifted would make equal
/// requests hash apart.
#[must_use]
pub fn canonical_value(v: Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut pairs: Vec<(String, Value)> = map.into_iter().collect();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Object(
                pairs
                    .into_iter()
                    .map(|(k, val)| (k, canonical_value(val)))
                    .collect(),
            )
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(canonical_value).collect()),
        other => other,
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
