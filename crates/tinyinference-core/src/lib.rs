//! Shared inference infrastructure.
//!
//! This crate contains only protocol-agnostic utilities shared by the focused
//! `tinyinference-llm`, `tinyinference-embeddings`, and `tinyinference-local`
//! crates. Model and embedding APIs live in their owning crates.

pub mod retry_after;
pub mod sanitize;

pub use retry_after::{
    BASE_BACKOFF_MS, MAX_BACKOFF_MS, MAX_RETRIES, backoff_ms_for_attempt, parse_retry_after_ms,
};
