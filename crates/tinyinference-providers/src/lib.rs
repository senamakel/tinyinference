//! Provider routing, authentication protocols, and error classification.

pub mod auth;
pub mod config_rejection;
pub mod oauth;

pub use auth::is_openai_oauth_session_expired_message;
pub use config_rejection::{
    NO_MODEL_CONFIGURED_ANCHOR, is_openai_compatible_unknown_model_message,
    is_provider_config_rejection_message,
};
