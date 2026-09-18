//! Local inference runtimes, device profiling, model selection, and installers.
//!
//! This crate builds on `tinyinference-core` and owns integrations that touch
//! local processes, hardware, files, and runtime-specific HTTP APIs.

#![cfg_attr(not(test), forbid(unsafe_code))]

pub mod device;
pub mod download;
pub mod install;
pub mod lm_studio;
pub mod model_requirements;
pub mod models;
pub mod ollama;
pub mod piper;
pub mod presets;
pub mod process;
pub mod profile;
pub mod provider;
pub mod service;
pub mod spawn_marker;
pub mod status;

pub mod error;

pub use error::{Error, Result};
pub use models::LocalModelConfig;
pub use status::{
    LocalAiAssetStatus, LocalAiAssetsStatus, LocalAiDownloadProgressItem, LocalAiDownloadsProgress,
    LocalAiEmbeddingResult, LocalAiSpeechResult, LocalAiStatus, LocalAiTtsResult,
};
