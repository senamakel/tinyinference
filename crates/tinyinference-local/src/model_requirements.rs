//! Reusable evaluation of local-model context requirements.

use serde::Serialize;

/// Verdict for a single model's context window against
/// a caller-supplied minimum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ContextEligibility {
    /// Context window ≥ the minimum. Accepted.
    Ok {
        /// Advertised context length.
        context_length: u64,
    },
    /// Context window below the minimum. Rejected for memory-layer use.
    BelowMinimum {
        /// Advertised context length.
        context_length: u64,
        /// Minimum required context length.
        required: u64,
    },
    /// Context window could not be determined (`/api/show` error or the
    /// metadata key is absent). Not hard-rejected — surfaced as unknown so
    /// the UI can warn without blocking a model that may still be fine.
    Unknown {
        /// Minimum required context length.
        required: u64,
    },
}

impl ContextEligibility {
    /// `true` only when the model is positively accepted.
    pub fn is_accepted(&self) -> bool {
        matches!(self, ContextEligibility::Ok { .. })
    }

    /// `true` when the model is conclusively rejected (reported a context
    /// window below the floor). `Unknown` is **not** a rejection.
    pub fn is_rejected(&self) -> bool {
        matches!(self, ContextEligibility::BelowMinimum { .. })
    }
}

/// Classify a model's optional reported context length against `required`.
pub fn evaluate_context(context_length: Option<u64>, required: u64) -> ContextEligibility {
    match context_length {
        Some(ctx) if ctx >= required => ContextEligibility::Ok {
            context_length: ctx,
        },
        Some(ctx) => ContextEligibility::BelowMinimum {
            context_length: ctx,
            required,
        },
        None => ContextEligibility::Unknown { required },
    }
}

#[cfg(test)]
#[path = "model_requirements_test.rs"]
mod tests;
