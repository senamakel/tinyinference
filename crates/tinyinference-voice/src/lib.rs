//! Provider-neutral voice-inference building blocks.
//!
//! Hosts retain authentication, configuration persistence, RPC, and provider
//! policy. This crate owns reusable speech transport, local Piper execution,
//! transcription cleanup, and PCM streaming mechanics.

pub mod cloud;
pub mod piper;
pub mod postprocess;
pub mod streaming;

mod types;

pub use types::{PiperSpeech, VisemeFrame};
