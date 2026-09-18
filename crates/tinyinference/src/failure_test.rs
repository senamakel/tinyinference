use super::*;
use crate::model::ProviderError;

#[test]
fn structured_status_ignores_unanchored_numbers() {
    assert_eq!(
        structured_http_status("API error (403 Forbidden): nope"),
        Some(403)
    );
    assert_eq!(structured_http_status("HTTP 404 Not Found"), Some(404));
    assert_eq!(structured_http_status("status: 401"), Some(401));
    assert_eq!(structured_http_status("408 Request Timeout"), Some(408));
    assert_eq!(structured_http_status("upstream took 450ms"), None);
    assert_eq!(structured_http_status("gpt-4-0409 returned nothing"), None);
}

#[test]
fn provider_failure_taxonomy_is_complete() {
    assert_eq!(
        classify_provider_failure(Some(401), None, "invalid api key"),
        ProviderFailureClass::NonRetryable
    );
    assert_eq!(
        classify_provider_failure(Some(429), None, "too many requests"),
        ProviderFailureClass::RateLimited
    );
    assert_eq!(
        classify_provider_failure(Some(429), None, "insufficient_balance"),
        ProviderFailureClass::NonRetryableRateLimit
    );
    assert_eq!(
        classify_provider_failure(None, None, "503 Service Unavailable"),
        ProviderFailureClass::UpstreamUnhealthy
    );
    assert_eq!(
        classify_provider_failure(None, None, "api key not set"),
        ProviderFailureClass::NonRetryable
    );
}

#[test]
fn structured_provider_error_uses_the_same_classifier() {
    let error = ProviderError {
        status: Some(429),
        code: Some("insufficient_quota".into()),
        message: "quota exhausted".into(),
        ..ProviderError::default()
    };
    assert_eq!(
        classify_provider_error(&error),
        ProviderFailureClass::NonRetryableRateLimit
    );
    assert!(!provider_error_is_retryable(&error));
}

#[test]
fn retry_after_accepts_integer_and_fractional_seconds() {
    assert_eq!(parse_retry_after_ms("Retry-After: 5"), Some(5_000));
    assert_eq!(
        parse_retry_after_ms("retry_after: 2.5 seconds"),
        Some(2_500)
    );
    assert_eq!(parse_retry_after_ms("Retry-After 7"), Some(7_000));
    assert_eq!(parse_retry_after_ms("no retry hint"), None);
}
