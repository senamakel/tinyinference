//! A decorator that preserves the most recent provider failure of a model call.
//!
//! Harness error channels commonly stringify model errors. Hosts that classify
//! failures (user-facing copy, telemetry suppression) need the typed original,
//! so [`ErrorSlotModel`] stashes it in a shared [`ModelErrorSlot`] before the
//! error crosses that boundary.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{
    ChatModel, InputModality, InputSource, ModelProfile, ModelRequest, ModelResponse, ModelStream,
    ModelStreamItem, ProviderError,
};
use crate::{Error, Result};

/// Shared slot holding the most recent (sanitized) failure of a model call.
pub type ModelErrorSlot = Arc<Mutex<Option<Error>>>;

/// Fills a [`ModelErrorSlot`] with the provider failure that ended a model call.
///
/// The slot is emptied when each attempt *starts*, not only when one succeeds:
/// an attempt that is dropped (call timeout), cancelled, or ends without a
/// terminal item must not leave the previous attempt's error behind.
///
/// Covers both an `Err` from `invoke`/`stream` and a failure reported *inside*
/// a stream ([`ModelStreamItem::ProviderFailed`], e.g. an HTTP 200 SSE
/// `{"error":...}` payload). Recorded errors are secret-scrubbed and truncated
/// ([`sanitize_api_error`](tinyinference_core::sanitize::sanitize_api_error))
/// and carry no raw body, because they flow into logs and telemetry.
pub struct ErrorSlotModel<State: Send + Sync> {
    inner: Arc<dyn ChatModel<State>>,
    slot: ModelErrorSlot,
}

impl<State: Send + Sync> std::fmt::Debug for ErrorSlotModel<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ErrorSlotModel")
            .finish_non_exhaustive()
    }
}

impl<State: Send + Sync> ErrorSlotModel<State> {
    /// Wraps `inner`, recording its failures into `slot`.
    #[must_use]
    pub fn new(inner: Arc<dyn ChatModel<State>>, slot: ModelErrorSlot) -> Self {
        Self { inner, slot }
    }
}

fn store_model_error(slot: &ModelErrorSlot, error: Option<Error>) {
    *slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = error;
}

fn slot_provider_error(error: &ProviderError) -> Error {
    let mut error = error.clone();
    error.message = tinyinference_core::sanitize::sanitize_api_error(&error.message);
    error.raw = None;
    Error::Provider(Box::new(error))
}

fn slot_error(error: &Error) -> Error {
    match error {
        Error::Provider(provider_error) => slot_provider_error(provider_error),
        other => Error::Model(tinyinference_core::sanitize::sanitize_api_error(
            &other.to_string(),
        )),
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for ErrorSlotModel<State> {
    fn profile(&self) -> Option<&ModelProfile> {
        self.inner.profile()
    }

    fn supports_input(
        &self,
        modality: InputModality,
        mime_type: &str,
        source: InputSource,
    ) -> bool {
        self.inner.supports_input(modality, mime_type, source)
    }

    fn cache_identity(&self) -> Option<String> {
        self.inner.cache_identity()
    }

    fn tracks_physical_attempts(&self) -> bool {
        self.inner.tracks_physical_attempts()
    }

    fn has_pre_invocation_admission_gate(&self) -> bool {
        self.inner.has_pre_invocation_admission_gate()
    }

    async fn invoke(&self, state: &State, request: ModelRequest) -> Result<ModelResponse> {
        store_model_error(&self.slot, None);
        let result = self.inner.invoke(state, request).await;
        store_model_error(&self.slot, result.as_ref().err().map(slot_error));
        result
    }

    async fn stream(&self, state: &State, request: ModelRequest) -> Result<ModelStream> {
        store_model_error(&self.slot, None);
        match self.inner.stream(state, request).await {
            Ok(stream) => {
                let slot = self.slot.clone();
                Ok(stream.map_items(move |item| {
                    match &item {
                        ModelStreamItem::ProviderFailed(error) => {
                            store_model_error(&slot, Some(slot_provider_error(error)));
                        }
                        ModelStreamItem::Completed(_) => store_model_error(&slot, None),
                        _ => {}
                    }
                    item
                }))
            }
            Err(error) => {
                store_model_error(&self.slot, Some(slot_error(&error)));
                Err(error)
            }
        }
    }
}

#[cfg(test)]
#[path = "error_slot_tests.rs"]
mod tests;
