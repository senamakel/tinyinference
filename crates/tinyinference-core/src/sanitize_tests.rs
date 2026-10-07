use super::*;

#[test]
fn truncation_limit_includes_the_suffix() {
    let sanitized = sanitize_api_error(&"x".repeat(MAX_API_ERROR_CHARS + 50));
    assert_eq!(sanitized.chars().count(), MAX_API_ERROR_CHARS);
    assert!(sanitized.ends_with("..."));
}

#[test]
fn secret_scrubbing_preserves_unicode_boundaries() {
    assert_eq!(
        scrub_secret_patterns("é before sk-secret after 🚀"),
        "é before [REDACTED] after 🚀"
    );
}

#[test]
fn test_scrub_credentials_utf8() {
    // Regex requires at least 8 chars for the value
    // The [a-zA-Z0-9_\-\.]{8,} part of the regex does NOT match emoji
    // So we must use quotes to hit the "([^"]{8,})" part
    let input = "api_key: \"🦀🦀🦀🦀🦀🦀🦀🦀\"";
    let output = scrub_credentials(input);
    // Should preserve 4 crabs and then redact
    assert!(output.contains("🦀🦀🦀🦀*[REDACTED]"));
}

#[test]
fn test_scrub_credentials_short_val() {
    let input = "api_key: 12345678";
    let output = scrub_credentials(input);
    assert!(output.contains("api_key: 1234*[REDACTED]"));
}

#[test]
fn scrubbed_json_keeps_its_quoted_key_and_parses() {
    let input = r#"{"token":"example-secret-value","status":"ready"}"#;
    let output = scrub_credentials(input);
    let parsed: serde_json::Value = serde_json::from_str(&output).expect("valid JSON after scrub");
    assert_eq!(parsed["status"], "ready");
    assert!(!output.contains("example-secret-value"));
    assert!(output.contains("*[REDACTED]"));
}

// #4453: bare, unlabelled secrets that show up in env dumps / API responses.

#[test]
fn scrubs_bare_aws_access_key() {
    let out = scrub_credentials("config dump AKIAIOSFODNN7EXAMPLE trailing text");
    assert!(
        !out.contains("AKIAIOSFODNN7EXAMPLE"),
        "bare AWS access key must be redacted: {out}"
    );
    assert!(
        out.contains("[REDACTED]"),
        "redaction marker present: {out}"
    );
}

#[test]
fn scrubs_bare_openai_key() {
    let out = scrub_credentials("response body sk-abcdefghij1234567890ABCDEFGHIJ end");
    assert!(
        !out.contains("abcdefghij1234567890ABCDEFGHIJ"),
        "openai secret body must be redacted: {out}"
    );
    assert!(
        out.contains("sk-"),
        "the sk- scheme is kept for context: {out}"
    );
    assert!(
        out.contains("[REDACTED]"),
        "redaction marker present: {out}"
    );
}

#[test]
fn scrubs_space_separated_bearer_token() {
    let out = scrub_credentials("Authorization: Bearer abcDEF1234567890ghijklmnop done");
    assert!(
        !out.contains("abcDEF1234567890ghijklmnop"),
        "space-separated bearer token must be redacted: {out}"
    );
    assert!(
        out.contains("Bearer"),
        "the scheme word is kept for context: {out}"
    );
    assert!(
        out.contains("[REDACTED]"),
        "redaction marker present: {out}"
    );
}

#[test]
fn scrubs_complete_basic_authorization_credentials() {
    let out = scrub_credentials("Authorization: Basic dXNlcjpzdXBlclNlY3JldA==");
    assert!(!out.contains("dXNlcjpzdXBlclNlY3JldA=="));
    assert!(out.contains("Basic *[REDACTED]"));
}

#[test]
fn scrubs_plus_and_escaped_quotes_in_labelled_values() {
    let plus = scrub_credentials("api_key=abcd+efgh/ijklmnop==");
    assert!(!plus.contains("efgh/ijklmnop"));

    let escaped = scrub_credentials(r#"token="abcd\"efghijklmnop""#);
    assert!(!escaped.contains("efghijklmnop"));
}

#[test]
fn redacts_short_labelled_values_and_keeps_json_escape_pairs_valid() {
    let short = scrub_credentials(r#"{"api_key":"short"}"#);
    assert!(!short.contains("short"));

    let escaped = scrub_credentials(r#"{"token":"abc\\secret-value"}"#);
    let parsed: serde_json::Value =
        serde_json::from_str(&escaped).expect("escaped JSON remains valid");
    assert_eq!(parsed["token"], "abc\\*[REDACTED]");
    assert!(!escaped.contains("secret-value"));
}

#[test]
fn test_scrub_credentials() {
    let input = "API_KEY=sk-1234567890abcdef; token: 1234567890; password=\"secret123456\"";
    let scrubbed = scrub_credentials(input);
    assert!(scrubbed.contains("API_KEY=sk-1*[REDACTED]"));
    assert!(scrubbed.contains("token: 1234*[REDACTED]"));
    assert!(scrubbed.contains("password=\"secr*[REDACTED]\""));
    assert!(!scrubbed.contains("abcdef"));
    assert!(!scrubbed.contains("secret123456"));
}

#[test]
fn scrub_credentials_is_idempotent() {
    let once = scrub_credentials("token=aB3dEfGh1234 password: hunter2hunter2");
    assert_eq!(scrub_credentials(&once), once);
}

#[test]
fn scrub_credentials_empty_input() {
    assert_eq!(scrub_credentials(""), "");
}

#[test]
fn scrub_credentials_no_sensitive_data() {
    let input = "normal text without any secrets";
    assert_eq!(scrub_credentials(input), input);
}

#[test]
fn scrub_credentials_short_values_are_redacted() {
    let input = r#"api_key="short""#;
    let output = scrub_credentials(input);
    assert_eq!(output, r#"api_key="shor*[REDACTED]""#);
    assert!(!output.contains("short"));
}

// #6954: ordinary source code that mentions tokens must pass through intact.

#[track_caller]
fn assert_unchanged(input: &str) {
    assert_eq!(scrub_credentials(input), input, "source code was redacted");
}

#[test]
fn placeholder_values_in_angle_brackets_are_not_secrets() {
    assert_unchanged(r#"self.eos_token = "<eos>""#);
}

#[test]
fn values_followed_by_an_index_or_call_are_expressions() {
    assert_unchanged("token = self.vocab[int(token_id)]");
    assert_unchanged("token = yyToknames[0]");
    assert_unchanged("token = get_token(request)");
}

#[test]
fn equality_comparison_is_not_an_assignment() {
    assert_unchanged("if token == self.unk_token:");
    assert_unchanged("if (token === previous) {");
}

#[test]
fn type_annotation_with_none_default_is_not_a_secret() {
    assert_unchanged("previous_token: Optional[int] = None");
}

#[test]
fn identifiers_that_merely_start_with_a_keyword_are_not_keys() {
    assert_unchanged("return score / max(token_count, 1)");
}

#[test]
fn signed_and_float_numbers_are_not_secrets() {
    assert_unchanged(r#""mean_logprob_per_token": -4.73,"#);
    assert_unchanged("temperature_token = 0.7");
    assert_unchanged("token_bias = +12");
}

#[test]
fn key_and_value_never_pair_across_a_newline() {
    let source = "    if token == self.unk_token:\n        return score / max(token_count, 1)\n";
    assert_unchanged(source);

    let setext_heading = "Access Token\n============\n";
    assert_unchanged(setext_heading);

    let split_assignment = "total = token\n= 12345678";
    assert_unchanged(split_assignment);
}

#[test]
fn whole_source_snippet_from_the_issue_is_unchanged() {
    let source = r#"class Tokenizer:
    def __init__(self):
        self.eos_token = "<eos>"
        previous_token: Optional[int] = None

    def decode(self, token_id):
        token = self.vocab[int(token_id)]
        if token == self.unk_token:
            return score / max(token_count, 1)
        return {"mean_logprob_per_token": -4.73}
"#;
    assert_unchanged(source);
}

/// Assemble a provider-style fixture at runtime so no credential-shaped
/// literal sits in the source for secret scanners to flag.
fn fixture(prefix: &str, body: &str) -> String {
    format!("{prefix}{body}")
}

#[test]
fn known_secret_prefixes_are_always_redacted() {
    let openai_key = fixture("sk-", "abc123def456ghi789jkl");
    let openai = scrub_credentials(&format!(r#"api_key = "{openai_key}""#));
    assert!(!openai.contains("abc123def456ghi789jkl"), "{openai}");
    assert!(openai.contains("*[REDACTED]"), "{openai}");

    // Letters only: the digit heuristic alone would not catch it.
    let github_token = fixture("gh", "p_abcdefghijklmnopqrstuvwxyzABCD");
    let github = scrub_credentials(&format!(r#"token: "{github_token}""#));
    assert!(
        !github.contains("abcdefghijklmnopqrstuvwxyzABCD"),
        "{github}"
    );
    assert!(github.contains("*[REDACTED]"), "{github}");

    let slack_token = fixture("xo", "xb-abcdefghij-klmnopqrst");
    let slack = scrub_credentials(&format!("SLACK_TOKEN={slack_token}"));
    assert!(!slack.contains("abcdefghij-klmnopqrst"), "{slack}");

    let jwt_token = fixture("ey", "JhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.c2lnbmF0dXJl");
    let jwt = scrub_credentials(&format!("bearer: {jwt_token}"));
    assert!(!jwt.contains("eyJzdWIiOiJ4In0"), "{jwt}");
}

#[test]
fn prefixed_keys_still_redact_secret_looking_values() {
    let github_token = fixture("gh", "p_abcdefghijklmnopqrstuvwxyz0123");
    let github = scrub_credentials(&format!("GITHUB_TOKEN={github_token}"));
    assert!(
        !github.contains("abcdefghijklmnopqrstuvwxyz0123"),
        "{github}"
    );

    let access = scrub_credentials(r#"{"access_token": "a8f3k2m9q7x1z5"}"#);
    assert!(!access.contains("a8f3k2m9q7x1z5"), "{access}");
    assert!(access.contains("*[REDACTED]"), "{access}");
}

#[test]
fn plain_password_assignment_is_redacted() {
    let out = scrub_credentials("password=hunter2secret");
    assert_eq!(out, "password=hunt*[REDACTED]");
}

#[test]
fn alphabetic_password_assignment_is_redacted() {
    let out = scrub_credentials("password=correcthorse");
    assert_eq!(out, "password=corr*[REDACTED]");
    assert!(!out.contains("correcthorse"));
}

#[test]
fn member_access_on_a_literal_is_code_but_a_full_stop_is_not() {
    assert_unchanged(r#"token = "<pad>".strip()"#);
    assert_unchanged(r#"secret = "abc123def".encode("utf-8")"#);

    let prose = scrub_credentials(r#"Set password="hunter2secret". Then restart."#);
    assert!(!prose.contains("hunter2secret"), "{prose}");
}

#[test]
fn references_and_type_annotations_are_not_secrets() {
    assert_unchanged("client = Client(api_key=api_key)");
    assert_unchanged("    api_key: String,");
    assert_unchanged(r#"token = os.environ["GITHUB_TOKEN"]"#);
}
