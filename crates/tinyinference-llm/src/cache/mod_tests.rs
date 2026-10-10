//! Tests for [`CachePolicy`] and [`canonical_value`].

use super::*;
use serde_json::json;
use std::time::Duration;

#[test]
fn cache_policy_defaults_to_no_caching() {
    let policy = CachePolicy::default();
    assert!(!policy.response_cache_enabled);
    assert!(!policy.protect_prompt_prefix);
    assert_eq!(policy.ttl(), None);
    assert_eq!(policy.namespace, None);
}

#[test]
fn cache_policy_builders_set_fields() {
    let policy = CachePolicy::enabled()
        .with_ttl(Duration::from_secs(90))
        .with_namespace("tenant-7");
    assert!(policy.response_cache_enabled);
    assert!(!policy.protect_prompt_prefix);
    assert_eq!(policy.ttl(), Some(Duration::from_secs(90)));
    assert_eq!(policy.namespace.as_deref(), Some("tenant-7"));
}

#[test]
fn cache_policy_serde_omits_unset_options() {
    let json = serde_json::to_value(CachePolicy::enabled()).unwrap();
    assert_eq!(
        json,
        json!({"response_cache_enabled": true, "protect_prompt_prefix": false})
    );
    let back: CachePolicy = serde_json::from_value(json).unwrap();
    assert_eq!(back, CachePolicy::enabled());
}

#[test]
fn canonical_value_sorts_keys_at_every_depth() {
    let value = json!({"b": {"z": 1, "a": [{"y": 2, "x": 3}]}, "a": null});
    let bytes = serde_json::to_string(&canonical_value(value)).unwrap();
    assert_eq!(bytes, r#"{"a":null,"b":{"a":[{"x":3,"y":2}],"z":1}}"#);
}
