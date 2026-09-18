//! Provider authentication failure classification.

/// Returns whether an error body indicates an expired OpenAI OAuth session.
///
/// This distinguishes ChatGPT/Codex subscription token expiry from ordinary
/// API-key rejection. HTTP status and host-provider exclusions remain the
/// responsibility of the caller because they are routing policy.
#[must_use]
pub fn is_openai_oauth_session_expired_message(message: &str) -> bool {
    const OAUTH_EXPIRY_MARKERS: &[&str] = &[
        "token_expired",
        "authentication token is expired",
        "please try signing in again",
    ];
    let lower = message.to_ascii_lowercase();
    OAUTH_EXPIRY_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::is_openai_oauth_session_expired_message;

    #[test]
    fn detects_oauth_expiry_markers_without_matching_api_key_failures() {
        assert!(is_openai_oauth_session_expired_message(
            r#"{"error":{"code":"token_expired"}}"#
        ));
        assert!(is_openai_oauth_session_expired_message(
            "Provided authentication token is expired"
        ));
        assert!(!is_openai_oauth_session_expired_message(
            "Incorrect API key provided"
        ));
    }
}
