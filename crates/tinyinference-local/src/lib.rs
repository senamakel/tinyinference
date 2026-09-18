//! Local inference runtimes, device profiling, model selection, and installers.
//!
//! This crate builds on `tinyinference-core` and owns integrations that touch
//! local processes, hardware, files, and runtime-specific HTTP APIs.

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
pub mod spawn_marker;

pub mod error;

pub use error::{Error, Result};
pub use models::LocalModelConfig;
