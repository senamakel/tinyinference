//! Provider-neutral language-model inference for Rust.
//!
//! TinyInference owns the reusable API boundary between agent runtimes and
//! model vendors: typed messages, model requests and responses, asynchronous
//! streaming, tool-call wire shapes, normalized usage, hosted providers, and
//! embedding clients. It deliberately contains no agent loop, graph runtime,
//! middleware stack, registry, or workspace policy.

pub mod cache;
pub mod catalog;
pub mod classification;
pub mod completion;
pub mod device;
pub mod embeddings;
pub mod error;
pub mod failure;
pub mod local;
pub mod message;
pub mod model;
pub mod providers;
pub mod sanitize;
pub mod sentiment;
pub mod tool;
pub mod usage;

pub use error::{Error, Result};
pub use failure::{
    ProviderFailureClass, classify_provider_error, classify_provider_failure, parse_retry_after_ms,
    provider_error_is_retryable, structured_http_status,
};

pub use embeddings::{
    EmbeddingModel, InMemoryVectorStore, MockEmbeddingModel, Retriever, ScoredDoc, VectorStore,
    cosine_similarity,
};
pub use local::models::LocalModelConfig;
pub use message::{AssistantMessage, ContentBlock, Message, MessageDelta};
pub use model::{
    ChatModel, ModelRequest, ModelResponse, ModelStream, ModelStreamItem,
    context_window_for_model_id, model_id_supports_vision,
};
pub use providers::{MockModel, ProviderKind, ProviderSpec};
pub use tool::{ToolCall, ToolDelta, ToolFormat, ToolSchema};
pub use usage::{Usage, UsageTotals};
