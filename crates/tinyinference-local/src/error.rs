//! Errors produced by local runtime and artifact operations.

use thiserror::Error;

/// Result returned by local inference APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// A normalized local inference failure.
#[derive(Debug, Error)]
pub enum Error {
    /// A caller supplied an invalid local-runtime identifier or option.
    #[error("invalid local inference input: {0}")]
    InvalidInput(String),
    /// No local vision model was configured for a vision request.
    #[error("no local vision model is configured: {0}")]
    VisionModelNotConfigured(String),
    /// The configured model cannot accept image input.
    #[error("configured model is not vision-capable: {0}")]
    VisionModelUnsupported(String),
    /// A local-runtime ownership marker could not be serialized.
    #[error("spawn marker serialization error: {0}")]
    MarkerSerialization(String),
    /// A local-runtime ownership marker could not be written atomically.
    #[error("spawn marker I/O error: {0}")]
    MarkerIo(String),
    /// Another install for the same engine is already running.
    #[error("local inference install already in progress: {0}")]
    InstallInProgress(String),
    /// A local runtime installation failed outside the download transport.
    #[error("local inference install error: {0}")]
    Install(String),
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
