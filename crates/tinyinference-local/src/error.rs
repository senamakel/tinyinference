//! Errors produced by local runtime and artifact operations.

use thiserror::Error;

/// Result returned by local inference APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// A normalized local inference failure.
#[derive(Debug, Error)]
pub enum Error {
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
