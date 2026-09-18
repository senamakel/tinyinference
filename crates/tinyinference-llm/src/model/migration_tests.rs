//! Migration-contract tests for provider-neutral model boundary metadata.

use super::*;
use crate::providers::MockModel;
use crate::usage::ChargedAmount;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;

struct RecordingModel {
    requests: Mutex<Vec<ModelRequest>>,
    response: ModelResponse,
}

impl RecordingModel {
    fn new(response: ModelResponse) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            response,
        }
    }
}

#[async_trait]
impl ChatModel<()> for RecordingModel {
    async fn invoke(&self, _state: &(), request: ModelRequest) -> crate::Result<ModelResponse> {
        self.requests.lock().unwrap().push(request);
        Ok(self.response.clone())
    }
}

struct FailingModel;

#[async_trait]
impl ChatModel<()> for FailingModel {
    async fn invoke(&self, _state: &(), _request: ModelRequest) -> crate::Result<ModelResponse> {
        Err(crate::Error::Model("synthetic failure".to_string()))
    }
}

#[derive(Default)]
struct RecordingObserver(Mutex<Vec<ModelCallObservation>>);

impl ModelObserver for RecordingObserver {
    fn observe(&self, observation: ModelCallObservation) {
        self.0.lock().unwrap().push(observation);
    }
}

#[test]
fn usage_round_trips_billing_and_context_metadata_without_raw_json() {
    let usage = Usage {
        cache_read_tokens: 3,
        cache_creation_tokens: 5,
        reasoning_tokens: 7,
        charged_amount: Some(ChargedAmount::usd_micros(42)),
        context_window_tokens: Some(128_000),
        ..Usage::new(11, 13)
    };

    let encoded = serde_json::to_value(usage).expect("usage serializes");
    assert!(encoded.get("charged_amount").is_some());
    assert!(encoded.get("context_window_tokens").is_some());
    assert!(!encoded.to_string().contains("raw"));
    assert_eq!(serde_json::from_value::<Usage>(encoded).unwrap(), usage);
}

#[test]
fn request_correlation_and_resolved_route_survive_response_serialization() {
    let correlation = ModelCallCorrelation::new("run-7", "call-9");
    let route = ResolvedModelRoute::new("openai", "gpt-5", "chat-v1");
    let response = ModelResponse::assistant("ok")
        .with_correlation(correlation.clone())
        .with_resolved_route(route.clone());

    let decoded: ModelResponse =
        serde_json::from_value(serde_json::to_value(response).unwrap()).unwrap();
    assert_eq!(decoded.correlation.as_ref(), Some(&correlation));
    assert_eq!(decoded.resolved_route.as_ref(), Some(&route));
}

#[tokio::test]
async fn direct_provider_propagates_request_correlation_to_sync_and_stream_results() {
    let model = MockModel::constant("ok");
    let correlation = ModelCallCorrelation::new("run-direct", "call-direct");
    let request = ModelRequest::default().with_correlation(correlation.clone());

    let response = model.invoke(&(), request.clone()).await.unwrap();
    assert_eq!(response.correlation.as_ref(), Some(&correlation));

    let stream = model.stream(&(), request).await.unwrap();
    assert_eq!(stream.metadata().correlation.as_ref(), Some(&correlation));
    let items = stream.collect::<Vec<_>>().await;
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("mock stream must complete");
    };
    assert_eq!(response.correlation.as_ref(), Some(&correlation));
}

#[tokio::test]
async fn default_chat_model_stream_propagates_request_correlation_to_terminal_response() {
    let model = RecordingModel::new(ModelResponse::assistant("ok"));
    let correlation = ModelCallCorrelation::new("run-default", "call-default");
    let stream = model
        .stream(
            &(),
            ModelRequest::default().with_correlation(correlation.clone()),
        )
        .await
        .unwrap();
    assert_eq!(stream.metadata().correlation.as_ref(), Some(&correlation));
    let items = stream.collect::<Vec<_>>().await;
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("default stream must complete");
    };
    assert_eq!(response.correlation.as_ref(), Some(&correlation));
}

#[tokio::test]
async fn decorators_apply_defaults_clamp_tokens_and_stamp_sync_and_stream_results() {
    let inner = Arc::new(RecordingModel::new(ModelResponse::assistant("ok")));
    let profile = ModelProfile {
        max_input_tokens: Some(128_000),
        ..ModelProfile::default()
    };
    let decorated: Arc<dyn ChatModel<()>> = Arc::new(MaxTokensModel::new(
        Arc::new(
            ProfileOverrideModel::new(inner.clone(), profile)
                .with_request_model("chat-v1")
                .with_request_temperature(0.3),
        ),
        64,
    ));
    let route = ResolvedModelRoute::new("openai", "gpt-5", "chat-v1");
    let model = RouteRecordingModel::new(decorated, route.clone());
    let request = ModelRequest::default()
        .with_max_tokens(100)
        .with_correlation(ModelCallCorrelation::new("run", "call"));

    let response = model.invoke(&(), request.clone()).await.unwrap();
    assert_eq!(response.correlation.as_ref().unwrap().call_id, "call");
    assert_eq!(response.resolved_route.as_ref(), Some(&route));

    let stream_items = model
        .stream(&(), request)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let ModelStreamItem::Completed(stream_response) = stream_items.last().unwrap() else {
        panic!("default stream must complete");
    };
    assert_eq!(stream_response.resolved_route.as_ref(), Some(&route));

    let requests = inner.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.model.as_deref() == Some("chat-v1"))
    );
    assert!(
        requests
            .iter()
            .all(|request| request.temperature == Some(0.3))
    );
    assert!(
        requests
            .iter()
            .all(|request| request.max_tokens == Some(64))
    );
}

#[tokio::test]
async fn stream_metadata_and_abort_guard_follow_the_consumer_lifetime() {
    let producer = tokio::spawn(std::future::pending::<()>());
    let stream = ModelStream::new(Box::pin(futures::stream::pending()))
        .with_correlation(ModelCallCorrelation::new("run", "call"))
        .with_resolved_route(ResolvedModelRoute::new("mock", "m", "route"))
        .abort_on_drop(AbortOnDrop::from_join_handle(&producer));
    assert_eq!(
        stream.metadata().correlation.as_ref().unwrap().run_id,
        "run"
    );
    assert_eq!(
        stream.metadata().resolved_route.as_ref().unwrap().provider,
        "mock"
    );
    drop(stream);
    assert!(producer.await.unwrap_err().is_cancelled());
}

#[tokio::test]
async fn completed_stream_disarms_its_abort_guard() {
    let producer = tokio::spawn(std::future::pending::<()>());
    let stream = ModelStream::new(Box::pin(futures::stream::iter(vec![
        ModelStreamItem::Completed(ModelResponse::assistant("done")),
    ])))
    .abort_on_drop(AbortOnDrop::from_join_handle(&producer));
    let _ = stream.collect::<Vec<_>>().await;
    tokio::task::yield_now().await;
    assert!(
        !producer.is_finished(),
        "terminal streams must not abort producers"
    );
    producer.abort();
    assert!(producer.await.unwrap_err().is_cancelled());
}

#[tokio::test]
async fn observer_reports_each_terminal_outcome_once() {
    let observer = Arc::new(RecordingObserver::default());
    let success = ObservingModel::new(
        Arc::new(RecordingModel::new(ModelResponse::assistant("ok"))),
        observer.clone(),
    );
    success.invoke(&(), ModelRequest::default()).await.unwrap();

    let cached = ObservingModel::new(
        Arc::new(RecordingModel::new(ModelResponse {
            served_from_cache: true,
            ..ModelResponse::assistant("cached")
        })),
        observer.clone(),
    );
    cached.invoke(&(), ModelRequest::default()).await.unwrap();

    let fallback = ObservingModel::new(
        Arc::new(RouteRecordingModel::new(
            Arc::new(RecordingModel::new(ModelResponse::assistant("fallback"))),
            ResolvedModelRoute::new("mock", "fallback", "fallback-route"),
        )),
        observer.clone(),
    );
    fallback
        .invoke(
            &(),
            ModelRequest::default()
                .with_model("provider-model-id")
                .with_requested_route("primary-route"),
        )
        .await
        .unwrap();

    let same_route = ObservingModel::new(
        Arc::new(RouteRecordingModel::new(
            Arc::new(RecordingModel::new(ModelResponse::assistant("same route"))),
            ResolvedModelRoute::new("mock", "other-provider-model", "same-route"),
        )),
        observer.clone(),
    );
    same_route
        .invoke(
            &(),
            ModelRequest::default()
                .with_model("different-provider-model-id")
                .with_requested_route("same-route"),
        )
        .await
        .unwrap();

    let failure = ObservingModel::new(Arc::new(FailingModel), observer.clone());
    assert!(failure.invoke(&(), ModelRequest::default()).await.is_err());

    let observations = observer.0.lock().unwrap();
    assert!(matches!(
        observations[0],
        ModelCallObservation::Succeeded { .. }
    ));
    assert!(matches!(
        observations[1],
        ModelCallObservation::CacheHit { .. }
    ));
    assert!(matches!(
        observations[2],
        ModelCallObservation::Fallback { .. }
    ));
    assert!(matches!(
        observations[3],
        ModelCallObservation::Succeeded { .. }
    ));
    assert!(matches!(
        observations[4],
        ModelCallObservation::Failed { .. }
    ));
    assert_eq!(observations.len(), 5);
}
