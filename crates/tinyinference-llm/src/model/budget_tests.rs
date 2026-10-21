use super::*;

#[test]
fn concurrent_reservations_debit_parent_atomically() {
    let parent = Budget::new(SpendLimits {
        tokens: Some(100),
        cost_micros: Some(100),
    });
    let first = parent.child(SpendLimits::default());
    let second = parent.child(SpendLimits::default());
    let reservation = first
        .reserve(Spend {
            tokens: 60,
            cost_micros: 60,
        })
        .unwrap();
    let refused = second
        .reserve(Spend {
            tokens: 60,
            cost_micros: 60,
        })
        .unwrap_err();
    assert_eq!(refused.snapshot.reserved.tokens, 60);
    reservation.settle(Spend {
        tokens: 20,
        cost_micros: 20,
    });
    let reservation = second
        .reserve(Spend {
            tokens: 60,
            cost_micros: 60,
        })
        .unwrap();
    drop(reservation);
    assert_eq!(parent.snapshot().spent.tokens, 80);
    assert_eq!(first.snapshot().spent.tokens, 20);
    assert_eq!(second.snapshot().spent.tokens, 60);
}

#[test]
fn unknown_or_cancelled_calls_retain_reservation() {
    let budget = Budget::new(SpendLimits {
        tokens: None,
        cost_micros: Some(1),
    });
    drop(
        budget
            .reserve(Spend {
                tokens: 50,
                cost_micros: 1,
            })
            .unwrap(),
    );
    assert_eq!(budget.snapshot().spent.cost_micros, 1);
    assert!(budget.reserve(Spend::default()).is_err());
}

#[test]
fn per_turn_refusal_does_not_mutate_parent() {
    let parent = Budget::new(SpendLimits {
        tokens: Some(100),
        cost_micros: None,
    });
    let turn = parent.child(SpendLimits {
        tokens: Some(10),
        cost_micros: None,
    });
    assert!(
        turn.reserve(Spend {
            tokens: 11,
            cost_micros: 0
        })
        .is_err()
    );
    assert_eq!(parent.snapshot(), BudgetSnapshot::default());
}

#[derive(Debug)]
struct BlockingModel {
    started: tokio::sync::Notify,
    release: tokio::sync::Notify,
    calls: std::sync::atomic::AtomicUsize,
}
#[async_trait]
impl ChatModel<()> for BlockingModel {
    async fn invoke(&self, _: &(), request: ModelRequest) -> crate::Result<ModelResponse> {
        assert_eq!(request.max_tokens, Some(20));
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.started.notify_one();
        self.release.notified().await;
        let mut response = ModelResponse::assistant("ok");
        response.usage = Some(crate::usage::Usage {
            input_tokens: 10,
            output_tokens: 5,
            charged_amount: Some(crate::usage::ChargedAmount::usd_micros(5)),
            ..Default::default()
        });
        Ok(response)
    }
}
fn call_policy() -> CallBudget {
    CallBudget {
        input_tokens: 1_000,
        output_tokens: 20,
        cost_micros: 100,
    }
}
fn request() -> ModelRequest {
    let mut request = ModelRequest::new(vec![crate::message::Message::user("hello")]);
    request.max_tokens = Some(30);
    request
}
#[tokio::test]
async fn physical_provider_calls_reserve_and_reconcile_atomically() {
    let budget = Budget::new(SpendLimits {
        tokens: None,
        cost_micros: Some(100),
    });
    let inner = Arc::new(BlockingModel {
        started: tokio::sync::Notify::new(),
        release: tokio::sync::Notify::new(),
        calls: Default::default(),
    });
    let model = Arc::new(BudgetedModel::new(
        inner.clone(),
        budget.clone(),
        call_policy(),
    ));
    let first = tokio::spawn({
        let model = model.clone();
        async move { model.invoke(&(), request()).await }
    });
    inner.started.notified().await;
    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::BudgetExceeded(_))
    ));
    assert_eq!(inner.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    inner.release.notify_one();
    first.await.unwrap().unwrap();
    assert_eq!(
        budget.snapshot(),
        BudgetSnapshot {
            spent: Spend {
                tokens: 15,
                cost_micros: 5
            },
            reserved: Spend::default()
        }
    );
}
#[tokio::test]
async fn aborted_provider_future_charges_reserved_upper_bound() {
    let budget = Budget::new(SpendLimits {
        tokens: None,
        cost_micros: Some(100),
    });
    let inner = Arc::new(BlockingModel {
        started: tokio::sync::Notify::new(),
        release: tokio::sync::Notify::new(),
        calls: Default::default(),
    });
    let model = Arc::new(BudgetedModel::new(
        inner.clone(),
        budget.clone(),
        call_policy(),
    ));
    let task = tokio::spawn(async move { model.invoke(&(), request()).await });
    inner.started.notified().await;
    task.abort();
    let _ = task.await;
    assert_eq!(budget.snapshot().spent.cost_micros, 100);
    assert_eq!(budget.snapshot().reserved, Spend::default());
}
#[tokio::test]
async fn oversized_input_is_refused_without_provider_call() {
    let inner = Arc::new(BlockingModel {
        started: tokio::sync::Notify::new(),
        release: tokio::sync::Notify::new(),
        calls: Default::default(),
    });
    let model = BudgetedModel::new(
        inner.clone(),
        Budget::new(SpendLimits::default()),
        CallBudget {
            input_tokens: 1,
            ..call_policy()
        },
    );
    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::Validation(_))
    ));
    assert_eq!(inner.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[test]
fn enormous_reservations_cannot_wrap_past_a_ceiling() {
    let budget = Budget::new(SpendLimits {
        tokens: Some(u64::MAX),
        cost_micros: None,
    });
    let _first = budget
        .reserve(Spend {
            tokens: u64::MAX - 1,
            cost_micros: 0,
        })
        .unwrap();
    assert!(
        budget
            .reserve(Spend {
                tokens: 2,
                cost_micros: 0
            })
            .is_err()
    );
}

#[derive(Debug)]
struct StreamingModel;
#[async_trait]
impl ChatModel<()> for StreamingModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        unreachable!()
    }
    async fn stream(&self, _: &(), _: ModelRequest) -> crate::Result<ModelStream> {
        Ok(ModelStream::new(Box::pin(futures::stream::pending())))
    }
}
#[tokio::test]
async fn dropping_a_stream_retains_the_live_reservation_as_spend() {
    let budget = Budget::new(SpendLimits {
        tokens: None,
        cost_micros: Some(100),
    });
    let model = BudgetedModel::new(Arc::new(StreamingModel), budget.clone(), call_policy());
    let stream = model.stream(&(), request()).await.unwrap();
    assert_eq!(budget.snapshot().reserved.cost_micros, 100);
    assert!(matches!(
        model.stream(&(), request()).await,
        Err(crate::Error::BudgetExceeded(_))
    ));
    drop(stream);
    assert_eq!(budget.snapshot().spent.cost_micros, 100);
    assert_eq!(budget.snapshot().reserved.cost_micros, 0);
}
#[tokio::test]
async fn passthrough_output_cap_cannot_override_the_budgeted_cap() {
    let model = BudgetedModel::new(
        Arc::new(StreamingModel),
        Budget::new(SpendLimits::default()),
        call_policy(),
    );
    let mut request = request();
    request.provider_options = serde_json::json!({"max_completion_tokens":500});
    assert!(matches!(
        model.invoke(&(), request).await,
        Err(crate::Error::Validation(_))
    ));
}

#[derive(Debug)]
struct InternalRetryModel;
#[async_trait]
impl ChatModel<()> for InternalRetryModel {
    fn tracks_physical_attempts(&self) -> bool {
        true
    }

    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        before_physical_attempt()?;
        before_physical_attempt()?;
        Ok(ModelResponse::assistant("ok"))
    }
}
#[derive(Default)]
struct CountPhysicalAdmission(std::sync::atomic::AtomicUsize);
impl super::BeforeDispatch for CountPhysicalAdmission {
    fn check(&self) -> crate::Result<()> {
        Ok(())
    }

    fn physical_attempt(&self) -> crate::Result<()> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}
#[tokio::test]
async fn provider_internal_retry_cannot_reuse_the_first_attempt_reservation() {
    let budget = Budget::new(SpendLimits {
        tokens: None,
        cost_micros: Some(100),
    });
    let admission = Arc::new(CountPhysicalAdmission::default());
    let model = BudgetedModel::new(Arc::new(InternalRetryModel), budget.clone(), call_policy())
        .before_dispatch(admission.clone());
    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::BudgetExceeded(_))
    ));
    assert_eq!(budget.snapshot().spent.cost_micros, 100);
    assert_eq!(budget.snapshot().reserved.cost_micros, 0);
    assert_eq!(
        admission.0.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "a retry refused by the money ledger must not consume a host call slot"
    );
}

#[tokio::test]
async fn temporary_wait_mode_retry_refusal_does_not_mark_budget_exhausted() {
    let budget = Budget::new(SpendLimits {
        tokens: None,
        cost_micros: Some(250),
    })
    .wait_for_capacity();
    let blocker = budget
        .reserve(Spend {
            tokens: 0,
            cost_micros: 100,
        })
        .expect("live reservation leaves capacity for the first send");
    let admission = Arc::new(CountPhysicalAdmission::default());
    let model = BudgetedModel::new(Arc::new(InternalRetryModel), budget.clone(), call_policy())
        .before_dispatch(admission.clone());

    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::BudgetExceeded(_))
    ));
    assert_eq!(
        budget.refusal(),
        None,
        "a retry denied only by live capacity must not publish terminal exhaustion"
    );
    assert_eq!(
        budget.snapshot().spent.cost_micros,
        call_policy().cost_micros
    );
    assert_eq!(budget.snapshot().reserved.cost_micros, 100);
    assert_eq!(
        admission.0.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "retry denied by live money reservations must not consume another host slot"
    );
    blocker.settle(Spend::default());
}

#[derive(Debug)]
struct CompletedStreamModel;
#[async_trait]
impl ChatModel<()> for CompletedStreamModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        unreachable!()
    }
    async fn stream(&self, _: &(), _: ModelRequest) -> crate::Result<ModelStream> {
        let response = ModelResponse::assistant("ok").with_usage(crate::usage::Usage {
            input_tokens: 10,
            output_tokens: 5,
            charged_amount: Some(crate::usage::ChargedAmount::usd_micros(5)),
            ..Default::default()
        });
        Ok(ModelStream::new(Box::pin(futures::stream::iter(vec![
            ModelStreamItem::Started,
            ModelStreamItem::Completed(response),
        ]))))
    }
}
#[tokio::test]
async fn completed_stream_reconciles_once_and_preserves_response() {
    let budget = Budget::new(SpendLimits {
        tokens: None,
        cost_micros: Some(100),
    });
    let model = BudgetedModel::new(
        Arc::new(CompletedStreamModel),
        budget.clone(),
        call_policy(),
    );
    let response = super::super::collect_model_stream(model.stream(&(), request()).await.unwrap())
        .await
        .unwrap();
    assert_eq!(response.text(), "ok");
    assert_eq!(
        budget.snapshot(),
        BudgetSnapshot {
            spent: Spend {
                tokens: 15,
                cost_micros: 5
            },
            reserved: Spend::default()
        }
    );
}
#[test]
fn cache_replay_spends_nothing_and_unknown_charge_keeps_its_bound() {
    let budget = Budget::new(SpendLimits::default());
    let amount = Spend {
        tokens: 100,
        cost_micros: 50,
    };
    let mut replay = ModelResponse::assistant("cached");
    replay.served_from_cache = true;
    BudgetedModel::<()>::settle(budget.reserve(amount).unwrap(), &replay);
    assert_eq!(budget.snapshot(), BudgetSnapshot::default());
    let response = ModelResponse::assistant("unknown").with_usage(crate::usage::Usage {
        input_tokens: 5,
        output_tokens: 2,
        charged_amount: Some(crate::usage::ChargedAmount::usd_micros(-1)),
        ..Default::default()
    });
    BudgetedModel::<()>::settle(budget.reserve(amount).unwrap(), &response);
    assert_eq!(
        budget.snapshot().spent,
        Spend {
            tokens: 7,
            cost_micros: 50
        }
    );
}

#[tokio::test]
async fn multimodal_inputs_are_refused_without_guessing_their_token_cost() {
    let model = BudgetedModel::new(
        Arc::new(StreamingModel),
        Budget::new(SpendLimits::default()),
        call_policy(),
    );
    let request = ModelRequest::new(vec![crate::message::Message::User(
        crate::message::UserMessage {
            content: vec![crate::message::ContentBlock::Image(
                crate::message::ImageRef {
                    url: "https://example.com/image.png".into(),
                    mime_type: None,
                },
            )],
        },
    )]);
    assert!(matches!(
        model.invoke(&(), request).await,
        Err(crate::Error::Validation(_))
    ));
    assert!(!model.supports_input(
        super::super::InputModality::Image,
        "image/png",
        super::super::InputSource::Url
    ));
}

#[tokio::test]
async fn oversized_schema_and_provider_prompt_are_refused_before_http() {
    for schema in [true, false] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let provider = crate::providers::openai::OpenAiModel::new("fixture")
            .with_base_url(format!("http://{}/v1", listener.local_addr().unwrap()));
        let budget = Budget::new(SpendLimits::default());
        let model = BudgetedModel::new(Arc::new(provider), budget.clone(), call_policy());
        let mut request = request();
        if schema {
            request.response_format = Some(crate::model::ResponseFormat::JsonSchema {
                name: "answer".into(),
                schema: serde_json::json!({"type":"object", "description":"x".repeat(2_000)}),
            });
        } else {
            request.provider_options = serde_json::json!({"instructions":"x".repeat(2_000)});
        }
        tokio::select! {
            outcome = model.invoke(&(), request) => assert!(matches!(outcome, Err(crate::Error::Validation(_)))),
            connection = listener.accept() => panic!("validation opened an HTTP connection: {connection:?}"),
        }
        assert_eq!(budget.snapshot(), BudgetSnapshot::default());
    }
}

#[derive(Debug)]
struct ProbeModel {
    invokes: std::sync::atomic::AtomicUsize,
    streams: std::sync::atomic::AtomicUsize,
}
#[async_trait]
impl ChatModel<()> for ProbeModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        self.invokes
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(
            ModelResponse::assistant("ok").with_usage(crate::usage::Usage {
                input_tokens: 2,
                output_tokens: 3,
                charged_amount: Some(crate::usage::ChargedAmount::usd_micros(5)),
                ..Default::default()
            }),
        )
    }

    async fn stream(&self, _: &(), _: ModelRequest) -> crate::Result<ModelStream> {
        self.streams
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(ModelStream::new(Box::pin(futures::stream::iter(vec![
            ModelStreamItem::Completed(ModelResponse::assistant("ok").with_usage(
                crate::usage::Usage {
                    input_tokens: 2,
                    output_tokens: 3,
                    charged_amount: Some(crate::usage::ChargedAmount::usd_micros(5)),
                    ..Default::default()
                },
            )),
        ]))))
    }
}

#[derive(Debug)]
struct DispatchCheck {
    allowed: bool,
    checks: std::sync::atomic::AtomicUsize,
}
impl super::BeforeDispatch for DispatchCheck {
    fn check(&self) -> crate::Result<()> {
        self.checks
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.allowed {
            Ok(())
        } else {
            Err(crate::Error::Validation("dispatch refused".into()))
        }
    }
}

fn probe_model() -> Arc<ProbeModel> {
    Arc::new(ProbeModel {
        invokes: Default::default(),
        streams: Default::default(),
    })
}

#[tokio::test]
async fn before_dispatch_success_keeps_existing_budget_settlement() {
    let budget = Budget::new(SpendLimits::default());
    let inner = probe_model();
    let hook = Arc::new(DispatchCheck {
        allowed: true,
        checks: Default::default(),
    });
    let model = BudgetedModel::new(inner.clone(), budget.clone(), call_policy())
        .before_dispatch(hook.clone());

    let response = model.invoke(&(), request()).await.unwrap();

    assert_eq!(response.text(), "ok");
    assert_eq!(inner.invokes.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(hook.checks.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(budget.snapshot().spent.cost_micros, 5);
    assert_eq!(budget.snapshot().reserved, Spend::default());
}

#[tokio::test]
async fn before_dispatch_refusal_refunds_reservation_without_invoking_provider() {
    let budget = Budget::new(SpendLimits::default());
    let inner = probe_model();
    let hook = Arc::new(DispatchCheck {
        allowed: false,
        checks: Default::default(),
    });
    let model = BudgetedModel::new(inner.clone(), budget.clone(), call_policy())
        .before_dispatch(hook.clone());

    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::Validation(message)) if message == "dispatch refused"
    ));

    assert_eq!(hook.checks.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(inner.invokes.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(budget.snapshot(), BudgetSnapshot::default());
}

#[tokio::test]
async fn before_dispatch_refusal_prevents_stream_construction_and_refunds() {
    let budget = Budget::new(SpendLimits::default());
    let inner = probe_model();
    let hook = Arc::new(DispatchCheck {
        allowed: false,
        checks: Default::default(),
    });
    let model = BudgetedModel::new(inner.clone(), budget.clone(), call_policy())
        .before_dispatch(hook.clone());

    assert!(matches!(
        model.stream(&(), request()).await,
        Err(crate::Error::Validation(message)) if message == "dispatch refused"
    ));

    assert_eq!(hook.checks.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(inner.streams.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(budget.snapshot(), BudgetSnapshot::default());
}

#[tokio::test]
async fn before_dispatch_success_checks_once_and_keeps_stream_settlement() {
    let budget = Budget::new(SpendLimits::default());
    let inner = probe_model();
    let hook = Arc::new(DispatchCheck {
        allowed: true,
        checks: Default::default(),
    });
    let model = BudgetedModel::new(inner.clone(), budget.clone(), call_policy())
        .before_dispatch(hook.clone());

    let response = super::super::collect_model_stream(model.stream(&(), request()).await.unwrap())
        .await
        .unwrap();

    assert_eq!(response.text(), "ok");
    assert_eq!(inner.streams.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(hook.checks.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(budget.snapshot().spent.cost_micros, 5);
    assert_eq!(budget.snapshot().reserved, Spend::default());
}

#[derive(Debug)]
struct RevocableDispatchCheck {
    allowed: std::sync::atomic::AtomicBool,
    checks: std::sync::atomic::AtomicUsize,
}
impl super::BeforeDispatch for RevocableDispatchCheck {
    fn check(&self) -> crate::Result<()> {
        self.checks
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.allowed.load(std::sync::atomic::Ordering::SeqCst) {
            Ok(())
        } else {
            Err(crate::Error::Validation("dispatch revoked".into()))
        }
    }
}

#[derive(Debug)]
struct RefuseAtPhysicalSend(std::sync::atomic::AtomicUsize);
impl super::BeforeDispatch for RefuseAtPhysicalSend {
    fn check(&self) -> crate::Result<()> {
        if self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            Ok(())
        } else {
            Err(crate::Error::Validation("dispatch revoked".into()))
        }
    }
}

#[derive(Debug)]
struct RefuseRetrySend(std::sync::atomic::AtomicUsize);
impl super::BeforeDispatch for RefuseRetrySend {
    fn check(&self) -> crate::Result<()> {
        if self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
            Ok(())
        } else {
            Err(crate::Error::Validation("dispatch revoked".into()))
        }
    }
}

fn assert_no_connection(listener: tokio::net::TcpListener) {
    let listener = listener.into_std().unwrap();
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock,
        "a refused dispatch must not open a provider connection"
    );
}

#[tokio::test]
async fn openai_unary_refusal_before_first_send_refunds_budget_reservation() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider = crate::providers::openai::OpenAiModel::new("fixture")
        .with_base_url(format!("http://{}/v1", listener.local_addr().unwrap()));
    let budget = Budget::new(SpendLimits::default());
    let hook = Arc::new(RefuseAtPhysicalSend(Default::default()));
    let model = BudgetedModel::new(Arc::new(provider), budget.clone(), call_policy())
        .before_dispatch(hook.clone());

    let outcome = tokio::select! {
        result = model.invoke(&(), request()) => result,
        connection = listener.accept() => panic!("refused OpenAI call opened a connection: {connection:?}"),
    };
    assert!(matches!(
        outcome,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    assert_eq!(hook.0.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(budget.snapshot(), BudgetSnapshot::default());
    assert_no_connection(listener);
}

#[tokio::test]
async fn openai_stream_refusal_before_first_send_refunds_budget_reservation() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider = crate::providers::openai::OpenAiModel::new("fixture")
        .with_base_url(format!("http://{}/v1", listener.local_addr().unwrap()));
    let budget = Budget::new(SpendLimits::default());
    let hook = Arc::new(RefuseAtPhysicalSend(Default::default()));
    let model = BudgetedModel::new(Arc::new(provider), budget.clone(), call_policy())
        .before_dispatch(hook.clone());

    let outcome = tokio::select! {
        result = model.stream(&(), request()) => result,
        connection = listener.accept() => panic!("refused OpenAI stream opened a connection: {connection:?}"),
    };
    assert!(matches!(
        outcome,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    assert_eq!(hook.0.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(budget.snapshot(), BudgetSnapshot::default());
    assert_no_connection(listener);
}

#[tokio::test]
async fn nested_dispatch_checked_refusal_refunds_outer_budget_reservation() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider = crate::providers::openai::OpenAiModel::new("fixture")
        .with_base_url(format!("http://{}/v1", listener.local_addr().unwrap()));
    let budget = Budget::new(SpendLimits::default());
    let hook = Arc::new(RefuseAtPhysicalSend(Default::default()));
    let budget_hook = Arc::new(DispatchCheck {
        allowed: true,
        checks: Default::default(),
    });
    let dispatch_checked = Arc::new(super::super::DispatchCheckedModel::new(
        Arc::new(provider),
        hook.clone(),
    ));
    let model = BudgetedModel::new(dispatch_checked, budget.clone(), call_policy())
        .before_dispatch(budget_hook.clone());

    let outcome = tokio::select! {
        result = model.invoke(&(), request()) => result,
        connection = listener.accept() => panic!("nested refusal opened an OpenAI connection: {connection:?}"),
    };
    assert!(matches!(
        outcome,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    assert_eq!(hook.0.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(
        budget_hook.checks.load(std::sync::atomic::Ordering::SeqCst),
        2
    );
    assert_eq!(budget.snapshot(), BudgetSnapshot::default());
    assert_no_connection(listener);
}

#[tokio::test]
async fn openai_retry_refusal_keeps_charge_for_the_first_physical_send() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (invoke_finished, finished) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) =
            tokio::time::timeout(std::time::Duration::from_secs(5), listener.accept())
                .await
                .expect("first OpenAI physical send reaches the local server")
                .unwrap();
        let mut request = vec![0; 16 * 1024];
        let read = socket.read(&mut request).await.unwrap();
        assert!(request[..read].starts_with(b"POST /v1/chat/completions"));
        let body = r#"{"error":{"message":"response_format json_object unsupported"}}"#;
        let response = format!(
            "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        drop(socket);
        finished.await.unwrap();
        assert_no_connection(listener);
    });

    let provider = crate::providers::openai::OpenAiModel::new("fixture")
        .with_base_url(format!("http://{addr}/v1"));
    let budget = Budget::new(SpendLimits::default());
    let hook = Arc::new(RefuseRetrySend(Default::default()));
    let model = BudgetedModel::new(Arc::new(provider), budget.clone(), call_policy())
        .before_dispatch(hook.clone());
    let mut request = request();
    request.response_format = Some(crate::model::ResponseFormat::JsonObject);

    assert!(matches!(
        model.invoke(&(), request).await,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    invoke_finished.send(()).unwrap();
    server.await.unwrap();
    assert_eq!(hook.0.load(std::sync::atomic::Ordering::SeqCst), 3);
    assert_eq!(budget.snapshot().reserved, Spend::default());
    assert_eq!(
        budget.snapshot().spent.cost_micros,
        call_policy().cost_micros,
        "a refused retry cannot refund a call after the first request was sent"
    );
}

#[tokio::test]
async fn nested_hooks_both_recheck_a_real_openai_retry() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let outer_hook = Arc::new(RevocableDispatchCheck {
        allowed: std::sync::atomic::AtomicBool::new(true),
        checks: Default::default(),
    });
    let server_hook = Arc::clone(&outer_hook);
    let (invoke_finished, finished) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) =
            tokio::time::timeout(std::time::Duration::from_secs(5), listener.accept())
                .await
                .expect("first OpenAI physical send reaches the local server")
                .unwrap();
        let mut request = vec![0; 16 * 1024];
        let read = socket.read(&mut request).await.unwrap();
        assert!(request[..read].starts_with(b"POST /v1/chat/completions"));
        server_hook
            .allowed
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let body = r#"{"error":{"message":"response_format json_object unsupported"}}"#;
        let response = format!(
            "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        drop(socket);
        let retried = tokio::select! {
            biased;
            connection = listener.accept() => {
                let (mut socket, _) = connection.unwrap();
                let mut request = vec![0; 16 * 1024];
                let read = socket.read(&mut request).await.unwrap();
                assert!(request[..read].starts_with(b"POST /v1/chat/completions"));
                let body = r#"{"id":"chatcmpl-test","object":"chat.completion","created":1,"model":"fixture","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
                true
            }
            _ = finished => false,
        };
        if !retried {
            assert_no_connection(listener);
        }
        retried
    });

    let provider = crate::providers::openai::OpenAiModel::new("fixture")
        .with_base_url(format!("http://{addr}/v1"));
    let budget = Budget::new(SpendLimits::default());
    let budget_hook = Arc::new(DispatchCheck {
        allowed: true,
        checks: Default::default(),
    });
    let budgeted = BudgetedModel::new(Arc::new(provider), budget.clone(), call_policy())
        .before_dispatch(budget_hook.clone());
    let model = super::super::DispatchCheckedModel::new(Arc::new(budgeted), outer_hook.clone());
    let mut request = request();
    request.response_format = Some(crate::model::ResponseFormat::JsonObject);

    let outcome = model.invoke(&(), request).await;
    let _ = invoke_finished.send(());
    assert!(matches!(
        outcome,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    assert!(
        !server.await.unwrap(),
        "outer admission allowed the retry through"
    );
    assert_eq!(
        outer_hook.checks.load(std::sync::atomic::Ordering::SeqCst),
        3,
        "the outer gate is checked before the call, first send, and retry"
    );
    assert_eq!(
        budget_hook.checks.load(std::sync::atomic::Ordering::SeqCst),
        3
    );
    assert_eq!(budget.snapshot().reserved, Spend::default());
    assert_eq!(
        budget.snapshot().spent.cost_micros,
        call_policy().cost_micros
    );
}

#[tokio::test]
async fn nested_budgeted_models_recheck_every_policy_on_a_real_retry() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let outer_hook = Arc::new(RevocableDispatchCheck {
        allowed: std::sync::atomic::AtomicBool::new(true),
        checks: Default::default(),
    });
    let server_hook = Arc::clone(&outer_hook);
    let (invoke_finished, finished) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) =
            tokio::time::timeout(std::time::Duration::from_secs(5), listener.accept())
                .await
                .expect("first OpenAI send reaches the local server")
                .unwrap();
        let mut request = vec![0; 16 * 1024];
        let read = socket.read(&mut request).await.unwrap();
        assert!(request[..read].starts_with(b"POST /v1/chat/completions"));
        server_hook
            .allowed
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let body = r#"{"error":{"message":"response_format json_object unsupported"}}"#;
        socket
            .write_all(
                format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        drop(socket);
        tokio::select! {
            biased;
            connection = listener.accept() => {
                let (mut socket, _) = connection.unwrap();
                let mut request = vec![0; 16 * 1024];
                let read = socket.read(&mut request).await.unwrap();
                assert!(request[..read].starts_with(b"POST /v1/chat/completions"));
                true
            }
            _ = finished => false,
        }
    });

    let provider = crate::providers::openai::OpenAiModel::new("fixture")
        .with_base_url(format!("http://{addr}/v1"));
    let inner_budget = Budget::new(SpendLimits::default());
    let inner_hook = Arc::new(DispatchCheck {
        allowed: true,
        checks: Default::default(),
    });
    let inner = BudgetedModel::new(Arc::new(provider), inner_budget.clone(), call_policy())
        .before_dispatch(inner_hook.clone());
    let outer_budget = Budget::new(SpendLimits::default());
    let model = BudgetedModel::new(Arc::new(inner), outer_budget.clone(), call_policy())
        .before_dispatch(outer_hook.clone());
    let mut request = request();
    request.response_format = Some(crate::model::ResponseFormat::JsonObject);

    let outcome = model.invoke(&(), request).await;
    let _ = invoke_finished.send(());
    assert!(matches!(
        outcome,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    assert!(!server.await.unwrap(), "outer hook refused the real retry");
    assert_eq!(
        outer_hook.checks.load(std::sync::atomic::Ordering::SeqCst),
        3
    );
    assert_eq!(
        inner_hook.checks.load(std::sync::atomic::Ordering::SeqCst),
        2
    );
    assert_eq!(outer_budget.snapshot().reserved, Spend::default());
    assert_eq!(inner_budget.snapshot().reserved, Spend::default());
    assert_eq!(
        outer_budget.snapshot().spent.cost_micros,
        call_policy().cost_micros
    );
    assert_eq!(
        inner_budget.snapshot().spent.cost_micros,
        call_policy().cost_micros
    );
}

#[derive(Debug)]
struct PhysicalAttemptModel;

#[async_trait]
impl ChatModel<()> for PhysicalAttemptModel {
    fn tracks_physical_attempts(&self) -> bool {
        true
    }

    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        super::before_physical_attempt()?;
        Ok(ModelResponse::assistant("ok"))
    }

    async fn stream(&self, _: &(), _: ModelRequest) -> crate::Result<ModelStream> {
        unreachable!()
    }
}

struct AttemptThenInvoke {
    inner: Arc<dyn ChatModel<()>>,
}

impl std::fmt::Debug for AttemptThenInvoke {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AttemptThenInvoke")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ChatModel<()> for AttemptThenInvoke {
    fn tracks_physical_attempts(&self) -> bool {
        true
    }

    async fn invoke(&self, state: &(), request: ModelRequest) -> crate::Result<ModelResponse> {
        super::before_physical_attempt()?;
        self.inner.invoke(state, request).await
    }

    async fn stream(&self, _: &(), _: ModelRequest) -> crate::Result<ModelStream> {
        unreachable!()
    }
}

#[tokio::test]
async fn denied_outer_retry_reservation_refunds_unsent_inner_budget() {
    let outer_budget = Budget::new(SpendLimits {
        tokens: None,
        cost_micros: Some(call_policy().cost_micros),
    });
    let inner_budget = Budget::new(SpendLimits::default());
    let inner = Arc::new(BudgetedModel::new(
        Arc::new(PhysicalAttemptModel),
        inner_budget.clone(),
        call_policy(),
    ));
    let model = BudgetedModel::new(
        Arc::new(AttemptThenInvoke { inner }),
        outer_budget.clone(),
        call_policy(),
    );

    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::BudgetExceeded(_))
    ));
    assert_eq!(
        outer_budget.snapshot().spent.cost_micros,
        call_policy().cost_micros
    );
    assert_eq!(inner_budget.snapshot(), BudgetSnapshot::default());
}

#[tokio::test]
async fn budget_refunds_when_a_gated_fallback_refuses_before_any_child_invocation() {
    let primary = probe_model();
    let fallback = probe_model();
    let chain = Arc::new(super::super::FallbackModel::new(
        Arc::new(super::super::DispatchCheckedModel::new(
            primary.clone(),
            Arc::new(DispatchCheck {
                allowed: false,
                checks: Default::default(),
            }),
        )),
        vec![Arc::new(super::super::DispatchCheckedModel::new(
            fallback.clone(),
            Arc::new(DispatchCheck {
                allowed: true,
                checks: Default::default(),
            }),
        ))],
    ));
    let budget = Budget::new(SpendLimits::default());
    let model = BudgetedModel::new(chain, budget.clone(), call_policy());

    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::Validation(message)) if message == "dispatch refused"
    ));
    assert_eq!(primary.invokes.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(
        fallback.invokes.load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    assert_eq!(budget.snapshot(), BudgetSnapshot::default());
}

#[tokio::test]
async fn mixed_fallback_refusal_before_opaque_child_entry_refunds_budget() {
    let primary = probe_model();
    let opaque_fallback = probe_model();
    let chain = Arc::new(super::super::FallbackModel::new(
        Arc::new(super::super::DispatchCheckedModel::new(
            primary.clone(),
            Arc::new(DispatchCheck {
                allowed: false,
                checks: Default::default(),
            }),
        )),
        vec![opaque_fallback.clone()],
    ));
    let budget = Budget::new(SpendLimits::default());
    let model = BudgetedModel::new(chain, budget.clone(), call_policy());

    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::Validation(message)) if message == "dispatch refused"
    ));
    assert_eq!(primary.invokes.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(
        opaque_fallback
            .invokes
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    assert_eq!(budget.snapshot(), BudgetSnapshot::default());
}

#[tokio::test]
async fn mixed_fallback_stream_refusal_before_opaque_child_entry_refunds_budget() {
    let primary = probe_model();
    let opaque_fallback = probe_model();
    let chain = Arc::new(super::super::FallbackModel::new(
        Arc::new(super::super::DispatchCheckedModel::new(
            primary.clone(),
            Arc::new(DispatchCheck {
                allowed: false,
                checks: Default::default(),
            }),
        )),
        vec![opaque_fallback.clone()],
    ));
    let budget = Budget::new(SpendLimits::default());
    let model = BudgetedModel::new(chain, budget.clone(), call_policy());

    assert!(matches!(
        model.stream(&(), request()).await,
        Err(crate::Error::Validation(message)) if message == "dispatch refused"
    ));
    assert_eq!(primary.streams.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(
        opaque_fallback
            .streams
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    assert_eq!(budget.snapshot(), BudgetSnapshot::default());
}

#[tokio::test]
async fn refused_dispatch_wrapper_does_not_charge_an_untracked_inner_model() {
    let inner = probe_model();
    let dispatch = Arc::new(super::super::DispatchCheckedModel::new(
        inner.clone(),
        Arc::new(DispatchCheck {
            allowed: false,
            checks: Default::default(),
        }),
    ));
    let budget = Budget::new(SpendLimits::default());
    let model = BudgetedModel::new(dispatch, budget.clone(), call_policy());

    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::Validation(message)) if message == "dispatch refused"
    ));
    assert_eq!(inner.invokes.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(budget.snapshot(), BudgetSnapshot::default());
}

#[derive(Debug)]
struct DeferredModel;

#[async_trait]
impl ChatModel<()> for DeferredModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        unreachable!()
    }

    async fn stream(&self, _: &(), _: ModelRequest) -> crate::Result<ModelStream> {
        Ok(ModelStream::new(Box::pin(futures::stream::iter([
            ModelStreamItem::Deferred(crate::model::DeferredHandle::new("fixture", "job-1")),
        ]))))
    }

    async fn fetch_deferred(
        &self,
        _: &crate::model::DeferredHandle,
    ) -> crate::Result<crate::model::DeferredStatus> {
        Ok(crate::model::DeferredStatus::Pending)
    }
}

#[tokio::test]
async fn dispatch_checked_model_delegates_deferred_fetches() {
    let model = super::super::DispatchCheckedModel::new(
        Arc::new(DeferredModel),
        Arc::new(DispatchCheck {
            allowed: true,
            checks: Default::default(),
        }),
    );

    assert!(matches!(
        model
            .fetch_deferred(&crate::model::DeferredHandle::new("fixture", "job-1"))
            .await
            .unwrap(),
        crate::model::DeferredStatus::Pending
    ));
}

#[derive(Debug)]
struct PhysicalDeferredModel {
    fetches: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl ChatModel<()> for PhysicalDeferredModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        unreachable!()
    }

    async fn stream(&self, _: &(), _: ModelRequest) -> crate::Result<ModelStream> {
        Ok(ModelStream::new(Box::pin(futures::stream::iter([
            ModelStreamItem::Deferred(crate::model::DeferredHandle::new("fixture", "job-1")),
        ]))))
    }

    async fn fetch_deferred(
        &self,
        _: &crate::model::DeferredHandle,
    ) -> crate::Result<crate::model::DeferredStatus> {
        super::before_physical_attempt()?;
        self.fetches
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(crate::model::DeferredStatus::Pending)
    }
}

#[tokio::test]
async fn dispatch_checked_model_rechecks_policy_for_deferred_provider_polls() {
    use futures::StreamExt;

    let inner = Arc::new(PhysicalDeferredModel {
        fetches: Default::default(),
    });
    let hook = Arc::new(RevocableDispatchCheck {
        allowed: std::sync::atomic::AtomicBool::new(true),
        checks: Default::default(),
    });
    let model = super::super::DispatchCheckedModel::new(inner.clone(), hook.clone());
    let mut stream = model.stream(&(), request()).await.unwrap();
    let handle = match stream.next().await.unwrap() {
        ModelStreamItem::Deferred(handle) => handle,
        item => panic!("expected deferred item, got {item:?}"),
    };
    hook.allowed
        .store(false, std::sync::atomic::Ordering::SeqCst);

    assert!(matches!(
        model.fetch_deferred(&handle).await,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    assert_eq!(inner.fetches.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[derive(Debug)]
struct UntrackedPaidModel {
    endpoint: String,
    hook: Arc<RevocableDispatchCheck>,
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl ChatModel<()> for UntrackedPaidModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        let response = reqwest::Client::new()
            .post(&self.endpoint)
            .send()
            .await
            .map_err(|error| crate::Error::Model(error.to_string()))?;
        assert!(response.status().is_success());
        let _ = response.bytes().await;
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.hook
            .allowed
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Err(crate::Error::Model("primary exhausted after send".into()))
    }
}

#[tokio::test]
async fn unknown_adapter_send_is_not_refunded_after_later_nested_refusal() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/paid", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0; 4096];
        let read = socket.read(&mut request).await.unwrap();
        assert!(request[..read].starts_with(b"POST /paid "));
        let body = "ok";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });
    let hook = Arc::new(RevocableDispatchCheck {
        allowed: std::sync::atomic::AtomicBool::new(true),
        checks: Default::default(),
    });
    let paid = Arc::new(UntrackedPaidModel {
        endpoint,
        hook: Arc::clone(&hook),
        calls: Default::default(),
    });
    let fallback = super::super::FallbackModel::new(
        paid.clone(),
        vec![Arc::new(super::super::DispatchCheckedModel::new(
            probe_model(),
            hook,
        ))],
    );
    let budget = Budget::new(SpendLimits::default());
    let model = BudgetedModel::new(Arc::new(fallback), budget.clone(), call_policy());

    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    server.await.unwrap();
    assert_eq!(paid.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(budget.snapshot().reserved, Spend::default());
    assert_eq!(
        budget.snapshot().spent.cost_micros,
        call_policy().cost_micros,
        "an untracked adapter may have sent before a nested gate refuses"
    );
}

#[tokio::test]
async fn opaque_stream_fallback_send_remains_charged_after_later_gated_refusal() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/paid", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0; 4096];
        let read = socket.read(&mut request).await.unwrap();
        assert!(request[..read].starts_with(b"POST /paid "));
        let body = "ok";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });
    let hook = Arc::new(RevocableDispatchCheck {
        allowed: std::sync::atomic::AtomicBool::new(true),
        checks: Default::default(),
    });
    let paid = Arc::new(UntrackedPaidModel {
        endpoint,
        hook: Arc::clone(&hook),
        calls: Default::default(),
    });
    let fallback = super::super::FallbackModel::new(
        paid.clone(),
        vec![Arc::new(super::super::DispatchCheckedModel::new(
            probe_model(),
            hook,
        ))],
    );
    let budget = Budget::new(SpendLimits::default());
    let model = BudgetedModel::new(Arc::new(fallback), budget.clone(), call_policy());

    assert!(matches!(
        model.stream(&(), request()).await,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    server.await.unwrap();
    assert_eq!(paid.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(budget.snapshot().reserved, Spend::default());
    assert_eq!(
        budget.snapshot().spent.cost_micros,
        call_policy().cost_micros,
        "a possibly-paid opaque fallback stream must remain charged"
    );
}

#[derive(Debug)]
struct RevokeAfterWarmup {
    checks: std::sync::atomic::AtomicUsize,
    revoked: std::sync::atomic::AtomicBool,
}

impl super::BeforeDispatch for RevokeAfterWarmup {
    fn check(&self) -> crate::Result<()> {
        self.checks
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if !self.revoked.load(std::sync::atomic::Ordering::SeqCst) {
            Ok(())
        } else {
            Err(crate::Error::Validation("dispatch revoked".into()))
        }
    }
}

#[tokio::test]
async fn ollama_native_warmup_is_a_tracked_and_gated_physical_attempt() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
    let hook = Arc::new(RevokeAfterWarmup {
        checks: Default::default(),
        revoked: Default::default(),
    });
    let server_hook = Arc::clone(&hook);
    let server = tokio::spawn(async move {
        let (mut warmup, _) =
            tokio::time::timeout(std::time::Duration::from_secs(5), listener.accept())
                .await
                .expect("Ollama warm-up reaches the local server")
                .unwrap();
        let mut bytes = vec![0; 8192];
        let read = warmup.read(&mut bytes).await.unwrap();
        let warmup_request = String::from_utf8_lossy(&bytes[..read]);
        assert!(warmup_request.starts_with("POST /api/chat "));
        server_hook
            .revoked
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let warmup_response = "{}";
        warmup
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    warmup_response.len(),
                    warmup_response
                )
                .as_bytes(),
            )
            .await
            .unwrap();

        let mut requests = vec!["/api/chat".to_string()];
        let second = tokio::select! {
            biased;
            accepted = listener.accept() => Some(accepted.unwrap()),
            _ = finished_rx => None,
        };
        if let Some((mut chat, _)) = second {
            let mut bytes = vec![0; 8192];
            let read = chat.read(&mut bytes).await.unwrap();
            let request = String::from_utf8_lossy(&bytes[..read]);
            assert!(request.starts_with("POST /v1/chat/completions "));
            requests.push("/v1/chat/completions".to_string());
            let body = r#"{"id":"test","object":"chat.completion","created":1,"model":"llama3.2","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#;
            chat.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        } else {
            assert_no_connection(listener);
        }
        requests
    });

    let mut spec = crate::providers::ProviderSpec::for_kind(crate::providers::ProviderKind::Ollama);
    spec.base_url = endpoint;
    let provider = crate::providers::registry::ProviderRegistry::new(
        |_: &crate::providers::ProviderSpec| -> crate::Result<String> { Ok(String::new()) },
    )
    .build_with(
        &spec,
        crate::providers::registry::ProviderBuildOptions {
            num_ctx: Some(4096),
            ..Default::default()
        },
    )
    .unwrap();
    let budget = Budget::new(SpendLimits::default());
    let model =
        BudgetedModel::new(provider, budget.clone(), call_policy()).before_dispatch(hook.clone());

    let outcome = model.invoke(&(), request()).await;
    let _ = finished_tx.send(());

    assert!(matches!(
        outcome,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    assert_eq!(
        budget.snapshot().spent.cost_micros,
        call_policy().cost_micros
    );
    assert_eq!(budget.snapshot().reserved, Spend::default());
    assert_eq!(hook.checks.load(std::sync::atomic::Ordering::SeqCst), 3);
    assert_eq!(server.await.unwrap(), vec!["/api/chat"]);
}

#[derive(Debug, Default)]
struct LimitOnePhysicalAttempt(std::sync::atomic::AtomicUsize);

impl super::BeforeDispatch for LimitOnePhysicalAttempt {
    fn check(&self) -> crate::Result<()> {
        Ok(())
    }

    fn physical_attempt(&self) -> crate::Result<()> {
        if self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            Ok(())
        } else {
            Err(crate::Error::Validation("physical call limit".into()))
        }
    }
}

#[tokio::test]
async fn ollama_warmup_consumes_the_only_physical_call_slot() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut warmup, _) =
            tokio::time::timeout(std::time::Duration::from_secs(5), listener.accept())
                .await
                .expect("warm-up request reaches the local server")
                .unwrap();
        let mut bytes = vec![0; 8192];
        let read = warmup.read(&mut bytes).await.unwrap();
        assert!(String::from_utf8_lossy(&bytes[..read]).starts_with("POST /api/chat "));
        let body = "{}";
        warmup
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();

        let second = tokio::select! {
            biased;
            accepted = listener.accept() => Some(accepted.unwrap()),
            _ = finished_rx => None,
        };
        let Some((mut chat, _)) = second else {
            assert_no_connection(listener);
            return vec!["/api/chat"];
        };
        let mut bytes = vec![0; 8192];
        let read = chat.read(&mut bytes).await.unwrap();
        assert!(String::from_utf8_lossy(&bytes[..read]).starts_with("POST /v1/chat/completions "));
        let body = r#"{"id":"test","object":"chat.completion","created":1,"model":"llama3.2","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#;
        chat.write_all(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .unwrap();
        vec!["/api/chat", "/v1/chat/completions"]
    });

    let mut spec = crate::providers::ProviderSpec::for_kind(crate::providers::ProviderKind::Ollama);
    spec.base_url = endpoint;
    let provider = crate::providers::registry::ProviderRegistry::new(
        |_: &crate::providers::ProviderSpec| -> crate::Result<String> { Ok(String::new()) },
    )
    .build_with(
        &spec,
        crate::providers::registry::ProviderBuildOptions {
            num_ctx: Some(4096),
            ..Default::default()
        },
    )
    .unwrap();
    let budget = Budget::new(SpendLimits::default());
    let admission = Arc::new(LimitOnePhysicalAttempt::default());
    let model = BudgetedModel::new(provider, budget.clone(), call_policy())
        .before_dispatch(admission.clone());

    let outcome = model.invoke(&(), request()).await;
    let _ = finished_tx.send(());
    let requests = server.await.unwrap();

    assert!(matches!(
        outcome,
        Err(crate::Error::Validation(message)) if message == "physical call limit"
    ));
    assert_eq!(requests, vec!["/api/chat"]);
    assert_eq!(admission.0.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(budget.snapshot().reserved, Spend::default());
    assert_eq!(
        budget.snapshot().spent.cost_micros,
        call_policy().cost_micros
    );
}

#[tokio::test]
async fn anthropic_fallback_rechecks_dispatch_before_its_second_send() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let first_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let first_addr = first_listener.local_addr().unwrap();
    let second_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let second_addr = second_listener.local_addr().unwrap();
    let hook = Arc::new(RevocableDispatchCheck {
        allowed: std::sync::atomic::AtomicBool::new(true),
        checks: Default::default(),
    });
    let first_hook = Arc::clone(&hook);
    let first_server = tokio::spawn(async move {
        let (mut socket, _) = first_listener.accept().await.unwrap();
        let mut request = vec![0; 8192];
        let read = socket.read(&mut request).await.unwrap();
        assert!(request[..read].starts_with(b"POST /v1/messages "));
        first_hook
            .allowed
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let body = r#"{"type":"error","error":{"type":"api_error","message":"retryable"}}"#;
        let response = format!(
            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });
    let (invoke_finished, finished) = tokio::sync::oneshot::channel();
    let second_server = tokio::spawn(async move {
        tokio::select! {
            biased;
            connection = second_listener.accept() => {
                let (mut socket, _) = connection.unwrap();
                let mut request = vec![0; 8192];
                let read = socket.read(&mut request).await.unwrap();
                assert!(request[..read].starts_with(b"POST /v1/messages "));
                let body = r#"{"id":"msg-test","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"unexpected fallback"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
                true
            }
            _ = finished => {
                assert_no_connection(second_listener);
                false
            }
        }
    });

    let first = crate::providers::anthropic::AnthropicModel::with_base_url(
        "fixture",
        format!("http://{first_addr}/v1"),
    )
    .with_insecure_http(true);
    let second = crate::providers::anthropic::AnthropicModel::with_base_url(
        "fixture",
        format!("http://{second_addr}/v1"),
    )
    .with_insecure_http(true);
    let fallback = super::super::FallbackModel::new(Arc::new(first), vec![Arc::new(second)]);
    assert!(fallback.tracks_physical_attempts());
    let model = super::super::DispatchCheckedModel::new(Arc::new(fallback), hook.clone());
    let outcome = model.invoke(&(), request()).await;
    let _ = invoke_finished.send(());
    let second_send = second_server.await.unwrap();
    first_server.await.unwrap();

    assert!(!second_send, "Anthropic fallback bypassed dispatch");
    assert!(matches!(
        outcome,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    assert_eq!(hook.checks.load(std::sync::atomic::Ordering::SeqCst), 3);
}

#[derive(Debug)]
struct RevocableRetryModel {
    calls: Arc<std::sync::atomic::AtomicUsize>,
    hook: Arc<RevocableDispatchCheck>,
}
#[async_trait]
impl ChatModel<()> for RevocableRetryModel {
    fn tracks_physical_attempts(&self) -> bool {
        true
    }

    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        before_physical_attempt()?;
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.hook
            .allowed
            .store(false, std::sync::atomic::Ordering::SeqCst);
        before_physical_attempt()?;
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(ModelResponse::assistant("unsafe retry"))
    }
}

#[tokio::test]
async fn before_dispatch_is_rechecked_before_each_physical_retry() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let hook = Arc::new(RevocableDispatchCheck {
        allowed: std::sync::atomic::AtomicBool::new(true),
        checks: Default::default(),
    });
    let budget = Budget::new(SpendLimits::default());
    let model = BudgetedModel::new(
        Arc::new(RevocableRetryModel {
            calls: Arc::clone(&calls),
            hook: Arc::clone(&hook),
        }),
        budget.clone(),
        call_policy(),
    )
    .before_dispatch(hook.clone());

    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(hook.checks.load(std::sync::atomic::Ordering::SeqCst), 3);
    assert_eq!(budget.snapshot().reserved, Spend::default());
    assert_eq!(
        budget.snapshot().spent.cost_micros,
        call_policy().cost_micros
    );
}

#[tokio::test]
async fn nested_dispatch_and_budget_wrappers_recheck_revoked_retry_policy() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let hook = Arc::new(RevocableDispatchCheck {
        allowed: std::sync::atomic::AtomicBool::new(true),
        checks: Default::default(),
    });
    let budget = Budget::new(SpendLimits::default());
    let budgeted = Arc::new(
        BudgetedModel::new(
            Arc::new(RevocableRetryModel {
                calls: Arc::clone(&calls),
                hook: Arc::clone(&hook),
            }),
            budget.clone(),
            call_policy(),
        )
        .before_dispatch(hook.clone()),
    );
    let model = super::super::DispatchCheckedModel::new(budgeted, hook.clone());

    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::Validation(message)) if message == "dispatch revoked"
    ));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(hook.checks.load(std::sync::atomic::Ordering::SeqCst), 5);
    assert_eq!(budget.snapshot().reserved, Spend::default());
    assert_eq!(
        budget.snapshot().spent.cost_micros,
        call_policy().cost_micros
    );
}

#[derive(Debug)]
struct RevocableLazyStreamModel(Arc<std::sync::atomic::AtomicUsize>);
#[async_trait]
impl ChatModel<()> for RevocableLazyStreamModel {
    fn tracks_physical_attempts(&self) -> bool {
        true
    }

    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        unreachable!()
    }
    async fn stream(&self, _: &(), _: ModelRequest) -> crate::Result<ModelStream> {
        let calls = Arc::clone(&self.0);
        Ok(ModelStream::new(Box::pin(futures::stream::poll_fn(
            move |_| {
                if let Err(error) = before_physical_attempt() {
                    return std::task::Poll::Ready(Some(ModelStreamItem::Failed(
                        error.to_string(),
                    )));
                }
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::task::Poll::Ready(None)
            },
        ))))
    }
}

#[tokio::test]
async fn before_dispatch_is_rechecked_when_a_lazy_stream_is_first_polled() {
    use futures::StreamExt;
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let hook = Arc::new(RevocableDispatchCheck {
        allowed: std::sync::atomic::AtomicBool::new(true),
        checks: Default::default(),
    });
    let budget = Budget::new(SpendLimits::default());
    let model = BudgetedModel::new(
        Arc::new(RevocableLazyStreamModel(Arc::clone(&calls))),
        budget.clone(),
        call_policy(),
    )
    .before_dispatch(hook.clone());

    let mut stream = model.stream(&(), request()).await.unwrap();
    hook.allowed
        .store(false, std::sync::atomic::Ordering::SeqCst);
    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::Failed(_))
    ));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(hook.checks.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(budget.snapshot(), BudgetSnapshot::default());
}

#[derive(Debug)]
struct LazyRetryStreamModel(Arc<std::sync::atomic::AtomicUsize>);
#[async_trait]
impl ChatModel<()> for LazyRetryStreamModel {
    fn tracks_physical_attempts(&self) -> bool {
        true
    }

    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        unreachable!()
    }
    async fn stream(&self, _: &(), _: ModelRequest) -> crate::Result<ModelStream> {
        let calls = Arc::clone(&self.0);
        let mut completed = false;
        Ok(ModelStream::new(Box::pin(futures::stream::poll_fn(
            move |_| {
                if completed {
                    return std::task::Poll::Ready(None);
                }
                completed = true;
                for _ in 0..2 {
                    if let Err(error) = before_physical_attempt() {
                        return std::task::Poll::Ready(Some(ModelStreamItem::Failed(
                            error.to_string(),
                        )));
                    }
                    calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
                std::task::Poll::Ready(Some(ModelStreamItem::Completed(ModelResponse::assistant(
                    "unsafe",
                ))))
            },
        ))))
    }
}
#[tokio::test]
async fn lazy_stream_physical_retry_reserves_before_dispatch_and_charges_terminal_failure() {
    use futures::StreamExt;
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let ledger = Budget::new(SpendLimits {
        tokens: None,
        cost_micros: Some(100),
    });
    let model = BudgetedModel::new(
        Arc::new(LazyRetryStreamModel(Arc::clone(&calls))),
        ledger.clone(),
        call_policy(),
    );
    let mut stream = model.stream(&(), request()).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::Failed(_))
    ));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(ledger.snapshot().reserved.cost_micros, 0);
    assert_eq!(ledger.snapshot().spent.cost_micros, 100);
    assert!(ledger.refusal().is_some());
}
#[derive(Debug)]
struct FailedStreamModel;
#[async_trait]
impl ChatModel<()> for FailedStreamModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        unreachable!()
    }
    async fn stream(&self, _: &(), _: ModelRequest) -> crate::Result<ModelStream> {
        Ok(ModelStream::new(Box::pin(futures::stream::iter(vec![
            ModelStreamItem::ProviderFailed(crate::model::ProviderError::default()),
        ]))))
    }
}
#[tokio::test]
async fn terminal_stream_failure_is_charged_while_the_consumer_retains_the_stream() {
    use futures::StreamExt;
    let ledger = Budget::new(SpendLimits::default());
    let model = BudgetedModel::new(Arc::new(FailedStreamModel), ledger.clone(), call_policy());
    let mut stream = model.stream(&(), request()).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::ProviderFailed(_))
    ));
    assert_eq!(ledger.snapshot().reserved.cost_micros, 0);
    assert_eq!(ledger.snapshot().spent.cost_micros, 100);
    drop(stream);
    assert_eq!(ledger.snapshot().spent.cost_micros, 100);
}
