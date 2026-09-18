//! Provider and managed-backend error classification.

mod backend_error;
mod billing;
mod config_rejection;
pub mod fallback;

pub use backend_error::{
    BackendErrorCode, backend_error_code_skips_sentry, body_flags_malformed,
    extract_backend_error_code, extract_backend_error_code_token, is_backend_client_guard_leak,
    is_backend_malformed_bad_request, is_managed_backend_envelope, managed_error_skips_sentry,
};
pub use billing::is_budget_exhausted_message;
pub use config_rejection::{
    NO_MODEL_CONFIGURED_ANCHOR, is_openai_compatible_unknown_model_message,
    is_provider_config_rejection_message,
};
