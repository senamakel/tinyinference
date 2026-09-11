//! Messages API response parsing shared by the unary and streaming paths.

use serde_json::Value;

use crate::Result;
use crate::message::{AssistantMessage, ContentBlock};
use crate::model::ModelResponse;
use crate::tool::ToolCall;
use crate::usage::Usage;

/// Maps a `usage` object onto [`Usage`]. Anthropic reports the three input
/// classes separately; `input_tokens` here is their sum so it stays the "size
/// of the prompt" every other adapter reports, with the cache split preserved
/// alongside.
pub(super) fn parse_usage(usage: &Value) -> Usage {
    let uncached_input_tokens = usage["input_tokens"].as_u64().unwrap_or(0);
    let cache_read_tokens = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);
    let cache_creation_tokens = usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
    let input_tokens = uncached_input_tokens + cache_read_tokens + cache_creation_tokens;
    let output_tokens = usage["output_tokens"].as_u64().unwrap_or(0);
    Usage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
        cache_read_tokens,
        cache_creation_tokens,
        ..Usage::default()
    }
}

/// Converts one response content block into the neutral message vocabulary.
/// Returns `(content block, tool call)`; exactly one side is populated for
/// the block kinds this adapter understands, and unknown kinds yield neither.
pub(super) fn parse_content_block(block: &Value) -> (Option<ContentBlock>, Option<ToolCall>) {
    match block["type"].as_str() {
        Some("text") => (
            block["text"]
                .as_str()
                .map(|text| ContentBlock::Text(text.to_string())),
            None,
        ),
        Some("tool_use") => (
            None,
            Some(ToolCall::new(
                block["id"].as_str().unwrap_or_default(),
                block["name"].as_str().unwrap_or_default(),
                block.get("input").cloned().unwrap_or(Value::Null),
            )),
        ),
        Some("thinking") => (
            Some(ContentBlock::Thinking {
                text: block["thinking"].as_str().unwrap_or_default().to_string(),
                signature: block["signature"].as_str().map(str::to_string),
            }),
            None,
        ),
        Some("redacted_thinking") => (
            Some(ContentBlock::RedactedThinking {
                data: block["data"].as_str().unwrap_or_default().to_string(),
            }),
            None,
        ),
        _ => (None, None),
    }
}

pub(crate) fn parse_response(body: Value) -> Result<ModelResponse> {
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();
    for block in body["content"].as_array().into_iter().flatten() {
        let (block, call) = parse_content_block(block);
        content.extend(block);
        tool_calls.extend(call);
    }
    let usage = body.get("usage").map(parse_usage);
    Ok(ModelResponse {
        message: AssistantMessage {
            id: body["id"].as_str().map(str::to_string),
            content,
            tool_calls,
            usage,
        },
        usage,
        finish_reason: body["stop_reason"].as_str().map(str::to_string),
        raw: Some(body),
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
    })
}
