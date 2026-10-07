//! Secret scrubbing and bounded provider-error formatting.

use regex::Regex;
use std::sync::LazyLock;

/// Maximum number of characters retained from a provider API error.
pub const MAX_API_ERROR_CHARS: usize = 200;
const TRANSPORT_ERROR_MAX_CHARS: usize = 1200;

/// Redact credentials carried by a URL while retaining its routing shape.
///
/// Userinfo and fragments are removed. Query parameter names remain visible for
/// diagnostics, but every value is replaced so presigned URLs and API keys can
/// never reach logs or error strings.
pub fn redact_url(input: &str) -> String {
    let Ok(mut url) = url::Url::parse(input.trim()) else {
        return "[REDACTED INVALID URL]".to_string();
    };
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_fragment(None);
    let names = url
        .query_pairs()
        .map(|(name, _)| name.into_owned())
        .collect::<Vec<_>>();
    url.set_query(None);
    if !names.is_empty() {
        let mut query = url.query_pairs_mut();
        for name in names {
            query.append_pair(&name, "[REDACTED]");
        }
    }
    url.to_string()
}

fn truncate_with_suffix(input: &str, max_chars: usize, suffix: &str) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let suffix_chars = suffix.chars().count();
    if suffix_chars >= max_chars {
        return suffix.chars().take(max_chars).collect();
    }
    let mut truncated: String = input.chars().take(max_chars - suffix_chars).collect();
    truncated.push_str(suffix);
    truncated
}

fn is_secret_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':')
}

fn token_end(input: &str, from: usize) -> usize {
    let mut end = from;
    for (i, c) in input[from..].char_indices() {
        if is_secret_char(c) {
            end = from + i + c.len_utf8();
        } else {
            break;
        }
    }
    end
}

/// Scrub known secret-like token prefixes from provider error strings.
pub fn scrub_secret_patterns(input: &str) -> String {
    const PREFIXES: [&str; 7] = [
        "sk-",
        "xoxb-",
        "xoxp-",
        "ghp_",
        "gho_",
        "ghu_",
        "github_pat_",
    ];

    let mut scrubbed = input.to_string();

    for prefix in PREFIXES {
        let mut search_from = 0;
        while let Some(rel) = scrubbed[search_from..].find(prefix) {
            let start = search_from + rel;
            let content_start = start + prefix.len();
            let end = token_end(&scrubbed, content_start);

            if end == content_start {
                search_from = content_start;
                continue;
            }

            scrubbed.replace_range(start..end, "[REDACTED]");
            search_from = start + "[REDACTED]".len();
        }
    }

    scrubbed
}

/// Key/value credential shapes: `token: "…"`, `api_key=…`, `bearer: …`, etc.
///
/// The key is a whole identifier ending in a sensitive word (`token`,
/// `access_token`, `GITHUB_TOKEN`), so `token_count` is never a key. Key,
/// operator and value stay on one line (`[ \t]*`, not `\s*`). The operator is
/// captured rather than excluded because the regex crate has no lookaround;
/// [`scrub_credentials`] drops `==` comparisons and asks [`looks_like_secret`]
/// whether the value is a credential or ordinary code.
///
/// Groups: `key` = sensitive key, 2 = operator, 3 = double-quoted value,
/// 4 = single-quoted value, 5 = bare value.
static SENSITIVE_KV_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)\b(?P<key>[a-z0-9_\-]*(?:token|api[_-]?key|password|secret|user[_-]?key|bearer|credential))["']?[ \t]*(:=|==|[:=])[ \t]*(?:"((?:\\.|[^"\\])*)"|'((?:\\.|[^'\\])*)'|([a-zA-Z0-9_+./=\-]+))"#).unwrap()
});

/// Value prefixes issued by credential providers. A labelled value starting
/// with one of these is redacted whatever else it looks like.
const KNOWN_SECRET_PREFIXES: [&str; 9] = [
    "sk-",
    "ghp_",
    "gho_",
    "ghu_",
    "github_pat_",
    "AKIA",
    "ASIA",
    // JWT: base64url of `{"`.
    "eyJ",
    "xox",
];

/// Signed or fractional numbers (`-4.73`, `+12`, `0.7`, `1e-5`). Plain
/// unsigned digit runs are not matched: they can be PINs or numeric keys.
static NON_SECRET_NUMBER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(?:[+-]\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\d*\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)$",
    )
    .unwrap()
});

/// Identifiers and dotted member paths with no digit (`None`, `self.vocab`,
/// `os.environ`): references to a value in code, not the value itself.
static DIGITLESS_IDENTIFIER_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z_][A-Za-z_]*(?:\.[A-Za-z_]+)*$").unwrap());

fn has_known_secret_prefix(value: &str) -> bool {
    KNOWN_SECRET_PREFIXES.iter().any(|prefix| {
        value.starts_with(prefix)
            // Slack tokens: `xoxb-`, `xoxp-`, `xoxa-`, … (`xox?-`).
            && (*prefix != "xox" || value.as_bytes().get(4) == Some(&b'-'))
    })
}

/// Decide whether a labelled value is a credential or ordinary source code.
///
/// `quoted` says whether the value was a string literal; `rest` is the text
/// right after the whole match (closing quote included).
fn looks_like_secret(value: &str, quoted: bool, rest: &str, key: &str) -> bool {
    if has_known_secret_prefix(value) {
        return true;
    }
    // `"<eos>"`, `"<your key here>"`: placeholders and markup.
    if value
        .find('<')
        .is_some_and(|open| value[open..].contains('>'))
    {
        return false;
    }
    if value.chars().any(char::is_whitespace) {
        return false;
    }
    if NON_SECRET_NUMBER_REGEX.is_match(value) {
        return false;
    }
    // Indexing, calls and member access: `self.vocab[i]`, `get_token()`,
    // `"x".join(…)`. A bare `.` at the end of a sentence is not member access.
    let mut after = rest.chars();
    match after.next() {
        Some('(' | '[') => return false,
        Some('.') if after.next().is_some_and(|c| c.is_alphabetic() || c == '_') => {
            return false;
        }
        _ => {}
    }
    // A string literal is data, so `"short"` stays redacted. A bare word that
    // is exactly the sensitive key is a common source-code reference
    // (`api_key=api_key`); other identifier-shaped values under a sensitive
    // key, including `password=correcthorse`, are data and must be redacted.
    quoted || !DIGITLESS_IDENTIFIER_REGEX.is_match(value) || !value.eq_ignore_ascii_case(key)
}

/// Bare AWS access-key IDs — `AKIA…`/`ASIA…` followed by 16 base32 chars — which
/// appear naked in env dumps and config reads with no surrounding key name.
static AWS_ACCESS_KEY_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b((?:AKIA|ASIA)[0-9A-Z]{16})\b").unwrap());

/// Bare OpenAI-style secret keys — `sk-…` (incl. `sk-proj-…`) with a long token
/// body. Not necessarily attached to a `key:` label in raw API responses.
static OPENAI_KEY_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(sk-[A-Za-z0-9_\-]{16,})\b").unwrap());

/// Space-separated bearer tokens as they appear in HTTP auth headers
/// (`Authorization: Bearer <token>`) — the KV regex only catches `bearer:`/`=`.
static BEARER_SPACE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(Bearer)\s+([A-Za-z0-9_\-\.=+/]{16,})").unwrap());

/// Authorization headers can carry Basic credentials, which are encoded but
/// remain recoverable and must be removed in full.
static AUTHORIZATION_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(authorization["']?\s*[:=]\s*["']?)(basic|bearer)\s+([A-Za-z0-9_\-\.=+/]{8,})"#,
    )
    .unwrap()
});

/// Preserve the first 4 chars of `val` for context, returning the redacted
/// prefix (empty when the value is too short to safely reveal any of it).
fn redact_prefix(val: &str) -> &str {
    if val.chars().count() <= 4 {
        return "";
    }
    let mut chars = val.char_indices().peekable();
    let mut logical_chars = 0;
    let mut end = 0;
    while let Some((idx, ch)) = chars.next() {
        end = idx + ch.len_utf8();
        if ch == '\\'
            && let Some((escape_idx, escape)) = chars.next()
        {
            end = escape_idx + escape.len_utf8();
            if escape == 'u' {
                for _ in 0..4 {
                    if let Some((hex_idx, hex)) = chars.next() {
                        end = hex_idx + hex.len_utf8();
                    }
                }
            }
        }
        logical_chars += 1;
        if logical_chars == 4 {
            break;
        }
    }
    &val[..end]
}

/// Scrub credentials from tool output to prevent accidental exfiltration.
///
/// Complements [`scrub_secret_patterns`], which redacts known token *prefixes*
/// from provider errors; this one covers labelled key/value pairs and a few
/// bare secret shapes.
///
/// Replaces known credential patterns with a redacted placeholder while preserving
/// a small prefix for context.
///
/// Covers labelled key/value pairs plus bare secrets that show up unlabelled in
/// env dumps, config reads and API responses: AWS access-key IDs (`AKIA…`/
/// `ASIA…`), OpenAI-style `sk-…` keys, and space-separated `Bearer <token>`
/// auth headers.
pub fn scrub_credentials(input: &str) -> String {
    let stage_kv = SENSITIVE_KV_REGEX.replace_all(input, |caps: &regex::Captures<'_>| {
        let full_match = &caps[0];
        // `token == other` compares; it does not assign.
        if &caps[2] == "==" {
            return full_match.to_string();
        }
        let quoted = caps.get(5).is_none();
        let value = caps
            .get(3)
            .or(caps.get(4))
            .or(caps.get(5))
            .expect("sensitive key-value match has a value");
        let rest = &input[caps.get(0).expect("full match").end()..];
        let key = caps.name("key").expect("sensitive key capture");
        // `api_key: String` is a type annotation, not a credential
        // assignment. Uppercase type names make this unambiguous while
        // lowercase alphabetic values such as `password: correcthorse`
        // remain protected.
        if &caps[2] == ":"
            && value
                .as_str()
                .chars()
                .next()
                .is_some_and(char::is_uppercase)
            && DIGITLESS_IDENTIFIER_REGEX.is_match(value.as_str())
        {
            return full_match.to_string();
        }
        if !looks_like_secret(value.as_str(), quoted, rest, key.as_str()) {
            return full_match.to_string();
        }
        // Already redacted: an unquoted value stops at `*`, so a second pass
        // would match the kept prefix and stack another marker. Leave it.
        if rest.starts_with("*[REDACTED]") {
            return full_match.to_string();
        }
        // Replace only the value span. Rebuilding the key from captures used
        // to add a second opening quote to JSON (`""token": ...`), making
        // tool results impossible to parse after redaction.
        let start = value.start() - caps.get(0).expect("full match").start();
        let end = value.end() - caps.get(0).expect("full match").start();
        format!(
            "{}{}*[REDACTED]{}",
            &full_match[..start],
            redact_prefix(value.as_str()),
            &full_match[end..]
        )
    });

    // Bare AWS access-key IDs: keep the 4-char `AKIA`/`ASIA` prefix for context.
    let stage_aws = AWS_ACCESS_KEY_REGEX.replace_all(&stage_kv, |caps: &regex::Captures<'_>| {
        format!("{}*[REDACTED]", redact_prefix(&caps[1]))
    });

    // Bare `sk-…` keys: keep the `sk-` scheme, redact the secret body.
    let stage_openai = OPENAI_KEY_REGEX.replace_all(&stage_aws, |_caps: &regex::Captures<'_>| {
        "sk-*[REDACTED]".to_string()
    });

    // Space-separated `Bearer <token>`: keep the scheme word, redact the token.
    let stage_authorization = AUTHORIZATION_REGEX
        .replace_all(&stage_openai, |caps: &regex::Captures<'_>| {
            format!("{}{} *[REDACTED]", &caps[1], &caps[2])
        });

    BEARER_SPACE_REGEX
        .replace_all(&stage_authorization, |caps: &regex::Captures<'_>| {
            format!("{} *[REDACTED]", &caps[1])
        })
        .to_string()
}

/// Sanitize API error text by scrubbing secrets and truncating length.
pub fn sanitize_api_error(input: &str) -> String {
    let scrubbed = scrub_secret_patterns(input);
    truncate_with_suffix(&scrubbed, MAX_API_ERROR_CHARS, "...")
}

/// Full `source()` chain for connection / TLS failures (scrubbed, longer than API body snippets).
pub fn format_error_chain(err: &dyn std::error::Error) -> String {
    let mut parts: Vec<String> = vec![err.to_string()];
    let mut src = std::error::Error::source(err);
    while let Some(e) = src {
        parts.push(e.to_string());
        src = std::error::Error::source(e);
    }
    let joined = parts.join(" | ");
    let scrubbed = scrub_secret_patterns(&joined);
    truncate_with_suffix(&scrubbed, TRANSPORT_ERROR_MAX_CHARS, "…")
}

/// Cause chain from [`anyhow::Error`] (e.g. responses fallback), scrubbed and length-limited.
pub fn format_anyhow_chain(err: &anyhow::Error) -> String {
    let joined = err
        .chain()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(" | ");
    let scrubbed = scrub_secret_patterns(&joined);
    truncate_with_suffix(&scrubbed, TRANSPORT_ERROR_MAX_CHARS, "…")
}

#[cfg(test)]
#[path = "sanitize_tests.rs"]
mod tests;
