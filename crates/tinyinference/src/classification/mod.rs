//! Provider-neutral error classification and fallback diagnostics.

mod billing;
pub mod fallback;

pub use billing::is_budget_exhausted_message;
