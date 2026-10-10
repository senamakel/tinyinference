use super::*;
use serde_json::json;

/// Runs a synthetic SSE byte stream through the parser and returns its
/// terminal [`ModelStreamItem::Completed`] response.
async fn completed(raw: Vec<&'static [u8]>) -> ModelResponse {
    use futures::StreamExt;

    let bytes = futures::stream::iter(
        raw.into_iter()
            .map(|v| Ok::<bytes::Bytes, crate::Error>(bytes::Bytes::from_static(v))),
    );
    let state = SseState {
        bytes: Box::pin(bytes),
        buf: Vec::new(),
        pending: VecDeque::new(),
        acc: OpenAiStreamAcc::default(),
        provider: "openai".to_string(),
        model: "gpt-4.1-mini".to_string(),
        started: false,
        finished: false,
        completion_seen: false,
        terminal_emitted: false,
    };
    let items: Vec<ModelStreamItem> = futures::stream::unfold(state, sse_next).collect().await;
    items
        .into_iter()
        .find_map(|item| match item {
            ModelStreamItem::Completed(response) => Some(response),
            _ => None,
        })
        .expect("stream ends with Completed")
}

/// A gateway that bills the call sends its charge on its own SSE frame after
/// the usage chunk (`event: openhuman-metadata`). The streamed response must
/// carry it on `raw`, as the non-streaming response does, or a streaming
/// caller can never see what it was charged.
#[tokio::test]
async fn sse_stream_keeps_gateway_extension_frame_on_raw() {
    let response = completed(vec![
        b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
        b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5,\"total_tokens\":105}}\n\n",
        b"event: openhuman-metadata\ndata: {\"openhuman\":{\"usage\":{\"input_tokens\":100,\"output_tokens\":5,\"total_tokens\":105,\"cached_input_tokens\":80},\"billing\":{\"charged_amount_usd\":0.0000183}}}\n\n",
        b"data: [DONE]\n\n",
    ])
    .await;

    assert_eq!(response.text(), "hi");
    assert_eq!(response.usage.unwrap().total_tokens, 105);
    let raw = response.raw.as_ref().expect("extension frame kept on raw");
    assert_eq!(
        raw["openhuman"]["billing"]["charged_amount_usd"],
        json!(0.0000183)
    );
    assert_eq!(raw["openhuman"]["usage"]["cached_input_tokens"], json!(80));
    // Chunk-schema fields belong to the normalized response, not `raw`.
    assert!(raw.get("choices").is_none());
    assert!(raw.get("usage").is_none());
}

/// Extension fields riding on ordinary chunks are kept too (last value per
/// key).
#[tokio::test]
async fn sse_stream_keeps_chunk_extension_fields_last_value_wins() {
    let response = completed(vec![
        b"data: {\"provider\":\"BaseTen\",\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        b"data: {\"provider\":\"Modal\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        b"data: [DONE]\n\n",
    ])
    .await;
    let raw = response.raw.as_ref().expect("extension field kept");
    assert_eq!(raw["provider"], json!("Modal"));
}

/// A plain OpenAI stream carries only chunk-schema fields, so its response
/// keeps `raw: None` exactly as before.
#[tokio::test]
async fn sse_stream_without_extensions_has_no_raw() {
    let response = completed(vec![
        b"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"system_fingerprint\":\"fp\",\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":\"stop\"}]}\n\n",
        b"data: [DONE]\n\n",
    ])
    .await;
    assert!(response.raw.is_none());
}
