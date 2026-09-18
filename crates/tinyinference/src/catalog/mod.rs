//! Provider model-catalog types and response parsing.

mod parsing;
mod types;

use crate::providers::openai::codex::OPENAI_CODEX_MODEL_HINTS;

pub use parsing::{merge_openai_codex_model_hints, model_items_from_body, parse_models_response};
pub use types::ModelInfo;

#[cfg(test)]
mod test;
