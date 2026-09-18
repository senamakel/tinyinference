# TinyInference

TinyInference is the provider-facing Rust layer shared by TinyHumans AI agent
runtimes. It owns model and embedding API concerns without owning an agent loop,
graph runtime, middleware system, capability registry, or workspace policy.

The workspace provides:

- provider-neutral messages, tool-call shapes, requests, responses, usage, and
  capability profiles;
- a `ChatModel<State>` abstraction with real asynchronous streaming;
- OpenAI Chat Completions, OpenAI Responses, and OpenAI-compatible provider
  adapters for Anthropic, Ollama, DeepSeek, Groq, xAI, OpenRouter, Together,
  and Mistral;
- OpenAI, Cohere, Ollama, Voyage, cloud, no-op, and deterministic mock
  embeddings;
- request caching, stream accumulation, normalized provider failures,
  provider-neutral retry classification and `Retry-After` parsing;
- conservative context-window and vision-capability hints for raw model ids
  when a provider cannot supply an authoritative model profile;
- normalized provider model-catalog parsing and local runtime model, vision,
  embedding, speech model, voice, and quantization resolution;
- embedding-provider catalogs, local model-tier presets, Ollama installation,
  and Piper binary/voice installation with atomic downloads and status tracking.

## Use

```rust
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ChatModel, ModelRequest};
use tinyinference_llm::providers::MockModel;

tokio::runtime::Runtime::new().unwrap().block_on(async {
let model = MockModel::echo();
let response = model
    .invoke(&(), ModelRequest::new(vec![Message::user("hello")]))
    .await
    .unwrap();
assert_eq!(response.text(), "hello");
});
```

TinyAgents vendors this repository at `vendor/tinyinference` and re-exports the
public modules through its historical `tinyagents::harness::*` paths. New code
can depend on `tinyinference-llm` for language models,
`tinyinference-embeddings` for vector generation and retrieval,
`tinyinference-local` for local runtimes and installers, and
`tinyinference-core` only for shared infrastructure.

## Layout

```text
Cargo.toml
crates/tinyinference-core/
└── src/
    ├── retry_after.rs shared Retry-After parsing and bounded backoff
    └── sanitize.rs    credential-safe diagnostic formatting
crates/tinyinference-llm/
└── src/
    ├── cache/       request fingerprints and response-cache contracts
    ├── catalog/     provider model-catalog types and response parsing
    ├── message/    provider-neutral message and content blocks
    ├── model/      ChatModel, request/response, profiles, and streaming
    ├── providers/  mock and OpenAI-compatible transports
    ├── error.rs    crate-wide Error and Result
    ├── failure.rs  provider-failure classification and retry hints
    ├── tool.rs     model-visible tool schemas and call/delta shapes
    └── usage/      normalized token accounting
crates/tinyinference-embeddings/
└── src/            embedding clients, vector store, and retriever
crates/tinyinference-local/
└── src/            device profiling, local runtimes, model selection, and installers
```

## Development

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-targets --all-features
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

Tests are deterministic and offline. Provider integration tests operate on
wire payloads and synthetic byte streams; constructing a hosted provider does
not make a network call.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
