//! Error types for inference, provider transport, and embeddings.

use thiserror::Error;

use crate::model::ProviderError;

/// Result returned by TinyInference APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// A normalized inference failure.
#[derive(Debug, Error)]
pub enum Error {
    /// A model transport or response failed without structured provider detail.
    #[error("model error: {0}")]
    Model(String),
    /// A provider returned structured failure detail.
    #[error("model error: {0}")]
    Provider(Box<ProviderError>),
    /// Caller input or configuration was invalid.
    #[error("validation error: {0}")]
    Validation(String),
    /// A provider payload could not be encoded or decoded.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    /// A provider model catalog used an invalid response envelope.
    #[error("catalog error: {0}")]
    Catalog(String),
    /// Embedding generation or vector-store behavior failed.
    #[error("embedding error: {0}")]
    Embedding(String),
    /// A local artifact could not be read or written.
    #[error("download I/O error: {0}")]
    DownloadIo(String),
    /// A local artifact request failed at the HTTP layer.
    #[error("download HTTP error: {0}")]
    DownloadHttp(String),
    /// A local artifact request exceeded a bounded deadline.
    #[error("download timeout: {0}")]
    DownloadTimeout(String),
    /// A downloaded artifact failed size or digest validation.
    #[error("download integrity error: {0}")]
    DownloadIntegrity(String),
}
