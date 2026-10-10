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
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        before_physical_attempt()?;
        before_physical_attempt()?;
        Ok(ModelResponse::assistant("ok"))
    }
}
#[tokio::test]
async fn provider_internal_retry_cannot_reuse_the_first_attempt_reservation() {
    let budget = Budget::new(SpendLimits {
        tokens: None,
        cost_micros: Some(100),
    });
    let model = BudgetedModel::new(Arc::new(InternalRetryModel), budget.clone(), call_policy());
    assert!(matches!(
        model.invoke(&(), request()).await,
        Err(crate::Error::BudgetExceeded(_))
    ));
    assert_eq!(budget.snapshot().spent.cost_micros, 100);
    assert_eq!(budget.snapshot().reserved.cost_micros, 0);
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
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
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
        let outcome = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            model.invoke(&(), request),
        )
        .await;
        assert!(matches!(outcome, Ok(Err(crate::Error::Validation(_)))));
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
        assert_eq!(budget.snapshot(), BudgetSnapshot::default());
    }
}
