use super::*;
use futures::{FutureExt, poll};
use std::task::Poll;

fn spend(amount: u64) -> Spend {
    Spend {
        tokens: amount,
        cost_micros: amount,
    }
}
fn ledger(limit: u64) -> Budget {
    Budget::new(SpendLimits {
        tokens: Some(limit),
        cost_micros: Some(limit),
    })
}

#[tokio::test]
async fn live_reservations_wait_then_admit_after_known_spend_settles() {
    let budget = ledger(100);
    let live = budget.reserve(spend(80)).unwrap();
    let mut waiting = Box::pin(budget.reserve_wait(spend(60)));
    assert!(
        poll!(waiting.as_mut()).is_pending(),
        "live reservations must queue, not reject"
    );
    live.settle(spend(10));
    let admitted = waiting.await.expect("settled spend leaves room");
    assert_eq!(budget.snapshot().reserved, spend(60));
    admitted.settle(spend(20));
    assert_eq!(budget.snapshot().spent, spend(30));
}

#[tokio::test]
async fn affordable_reservations_do_not_serialize() {
    let budget = ledger(100);
    let first = budget
        .reserve_wait(spend(60))
        .now_or_never()
        .unwrap()
        .unwrap();
    let second = budget
        .reserve_wait(spend(30))
        .now_or_never()
        .unwrap()
        .unwrap();
    assert_eq!(budget.snapshot().reserved, spend(90));
    first.settle(spend(10));
    second.settle(spend(10));
}

#[tokio::test]
async fn unknown_settlement_wakes_waiters_to_a_terminal_refusal() {
    let budget = ledger(100);
    let live = budget.reserve(spend(80)).unwrap();
    let mut waiting = Box::pin(budget.reserve_wait(spend(60)));
    assert!(poll!(waiting.as_mut()).is_pending());
    drop(live);
    let error = waiting.await.unwrap_err();
    assert_eq!(error.snapshot.spent, spend(80));
    assert_eq!(error.snapshot.reserved, Spend::default());
    assert_eq!(budget.snapshot().spent, spend(80));
}

#[tokio::test]
async fn settled_spend_and_zero_limits_refuse_without_waiting() {
    let budget = ledger(100);
    budget.reserve(spend(80)).unwrap().settle(spend(80));
    assert!(matches!(
        poll!(Box::pin(budget.reserve_wait(spend(60))).as_mut()),
        Poll::Ready(Err(_))
    ));
    let zero = ledger(0);
    assert!(matches!(
        poll!(Box::pin(zero.reserve_wait(Spend::default())).as_mut()),
        Poll::Ready(Err(_))
    ));
}

#[tokio::test]
async fn permanent_child_limit_is_not_hidden_by_a_live_parent_refusal() {
    let parent = ledger(100);
    let held = parent.reserve(spend(100)).unwrap();
    let child = parent.child(SpendLimits {
        tokens: Some(40),
        cost_micros: Some(40),
    });
    let error = child
        .reserve_wait(spend(60))
        .now_or_never()
        .expect("permanently unaffordable")
        .unwrap_err();
    assert_eq!(error.limits.tokens, Some(40));
    assert_eq!(child.refusal().unwrap().limits.tokens, Some(40));
    assert_eq!(parent.snapshot().reserved, spend(100));
    held.settle(Spend::default());
}

#[tokio::test]
async fn settlement_notification_is_retained_before_the_receiver_awaits() {
    let parent = ledger(100);
    let child = parent.child(SpendLimits::default());
    let live = child.reserve(spend(80)).unwrap();
    // The same shared event used by reserve_wait is armed before the failed
    // check. Settlement happens before changed() even creates its future.
    let mut changes = parent.changes.subscribe();
    assert!(parent.reserve(spend(60)).is_err());
    live.settle(spend(10));
    assert!(changes.changed().now_or_never().unwrap().is_ok());
    parent
        .reserve_wait(spend(60))
        .await
        .unwrap()
        .settle(spend(10));
}

#[tokio::test]
async fn woken_sibling_contenders_recheck_shared_parent_capacity_atomically() {
    let parent = ledger(100);
    let held = parent.reserve(spend(100)).unwrap();
    let a = parent.child(SpendLimits::default());
    let b = parent.child(SpendLimits::default());
    let mut first = Box::pin(a.reserve_wait(spend(60)));
    let mut second = Box::pin(b.reserve_wait(spend(60)));
    assert!(poll!(first.as_mut()).is_pending());
    assert!(poll!(second.as_mut()).is_pending());
    held.settle(Spend::default());
    let Poll::Ready(Ok(admitted)) = poll!(first.as_mut()) else {
        panic!("one fits")
    };
    assert!(
        poll!(second.as_mut()).is_pending(),
        "two cannot over-admit together"
    );
    assert_eq!(parent.snapshot().reserved, spend(60));
    admitted.settle(spend(10));
    let next = second.await.unwrap();
    assert_eq!(parent.snapshot().reserved, spend(60));
    next.settle(spend(10));
    assert_eq!(parent.snapshot().spent, spend(20));
}

#[tokio::test]
async fn dropping_a_waiting_future_reserves_and_charges_nothing() {
    let parent = ledger(100);
    let held = parent.reserve(spend(80)).unwrap();
    let mut waiting = Box::pin(parent.reserve_wait(spend(60)));
    assert!(poll!(waiting.as_mut()).is_pending());
    drop(waiting);
    assert_eq!(
        parent.snapshot(),
        BudgetSnapshot {
            spent: Spend::default(),
            reserved: spend(80)
        }
    );
    held.settle(spend(10));
    assert_eq!(parent.snapshot().spent, spend(10));
}

#[derive(Debug)]
struct GatedModel {
    calls: std::sync::atomic::AtomicUsize,
    release: tokio::sync::Semaphore,
}
impl Default for GatedModel {
    fn default() -> Self {
        Self {
            calls: Default::default(),
            release: tokio::sync::Semaphore::new(0),
        }
    }
}
#[async_trait]
impl ChatModel<()> for GatedModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.release.acquire().await.unwrap().forget();
        Ok(
            ModelResponse::assistant("ok").with_usage(crate::usage::Usage {
                input_tokens: 5,
                output_tokens: 5,
                charged_amount: Some(crate::usage::ChargedAmount::usd_micros(10)),
                ..Default::default()
            }),
        )
    }
}
fn policy() -> CallBudget {
    CallBudget {
        input_tokens: 1_000,
        output_tokens: 20,
        cost_micros: 100,
    }
}
fn request() -> ModelRequest {
    ModelRequest::new(vec![crate::message::Message::user("fixture")])
}
fn cost_ledger(limit: u64) -> Budget {
    Budget::new(SpendLimits {
        tokens: None,
        cost_micros: Some(limit),
    })
}

#[tokio::test]
async fn opt_in_models_run_affordable_calls_concurrently_and_wait_only_for_capacity() {
    let ledger = cost_ledger(250).wait_for_capacity();
    let inner = Arc::new(GatedModel::default());
    let model = BudgetedModel::new(
        inner.clone(),
        ledger.child(SpendLimits::default()),
        policy(),
    );
    let mut first = Box::pin(model.invoke(&(), request()));
    let mut second = Box::pin(model.invoke(&(), request()));
    let mut third = Box::pin(model.invoke(&(), request()));
    assert!(poll!(first.as_mut()).is_pending());
    assert!(poll!(second.as_mut()).is_pending());
    assert!(poll!(third.as_mut()).is_pending());
    assert_eq!(inner.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(ledger.snapshot().reserved.cost_micros, 200);
    inner.release.add_permits(1);
    assert!(matches!(poll!(first.as_mut()), Poll::Ready(Ok(_))));
    assert!(poll!(third.as_mut()).is_pending());
    assert_eq!(inner.calls.load(std::sync::atomic::Ordering::SeqCst), 3);
    assert_eq!(ledger.snapshot().spent.cost_micros, 10);
    assert_eq!(ledger.snapshot().reserved.cost_micros, 200);
    inner.release.add_permits(2);
    assert!(matches!(poll!(second.as_mut()), Poll::Ready(Ok(_))));
    assert!(matches!(poll!(third.as_mut()), Poll::Ready(Ok(_))));
    assert_eq!(ledger.snapshot().spent.cost_micros, 30);
}

#[tokio::test]
async fn cancelled_paid_model_call_retains_charge_and_prevents_waiting_provider_dispatch() {
    let ledger = cost_ledger(100).wait_for_capacity();
    let inner = Arc::new(GatedModel::default());
    let model = BudgetedModel::new(inner.clone(), ledger.clone(), policy());
    let mut paid = Box::pin(model.invoke(&(), request()));
    let mut waiting = Box::pin(model.invoke(&(), request()));
    assert!(poll!(paid.as_mut()).is_pending());
    assert!(poll!(waiting.as_mut()).is_pending());
    assert_eq!(inner.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    drop(paid);
    assert!(matches!(
        poll!(waiting.as_mut()),
        Poll::Ready(Err(crate::Error::BudgetExceeded(_)))
    ));
    assert_eq!(inner.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(ledger.snapshot().spent.cost_micros, 100);
    assert_eq!(ledger.snapshot().reserved.cost_micros, 0);
}

#[tokio::test]
async fn cancelling_queued_model_admission_does_not_dispatch_or_charge_it() {
    let ledger = cost_ledger(100).wait_for_capacity();
    let inner = Arc::new(GatedModel::default());
    let model = BudgetedModel::new(inner.clone(), ledger.clone(), policy());
    let mut paid = Box::pin(model.invoke(&(), request()));
    let mut queued = Box::pin(model.invoke(&(), request()));
    assert!(poll!(paid.as_mut()).is_pending());
    assert!(poll!(queued.as_mut()).is_pending());
    drop(queued);
    inner.release.add_permits(1);
    assert!(matches!(poll!(paid.as_mut()), Poll::Ready(Ok(_))));
    assert_eq!(inner.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(ledger.snapshot().spent.cost_micros, 10);
    assert_eq!(ledger.snapshot().reserved.cost_micros, 0);
}

#[tokio::test]
async fn temporary_wait_and_its_cancellation_publish_no_refusal() {
    let budget = ledger(100);
    let live = budget.reserve(spend(80)).unwrap();
    let mut waiting = Box::pin(budget.reserve_wait(spend(60)));
    assert!(poll!(waiting.as_mut()).is_pending());
    assert!(
        budget.refusal().is_none(),
        "temporary capacity pressure is not a terminal refusal"
    );
    drop(waiting);
    assert!(
        budget.refusal().is_none(),
        "canceling an unadmitted waiter must not poison later calls"
    );
    live.settle(spend(10));
    let admitted = budget.reserve_wait(spend(60)).await.unwrap();
    assert!(budget.refusal().is_none());
    admitted.settle(spend(10));
}

#[tokio::test]
async fn temporary_wait_preserves_an_existing_fail_fast_refusal() {
    let budget = ledger(100);
    let live = budget.reserve(spend(80)).unwrap();
    let original = budget.reserve(spend(70)).unwrap_err();
    let mut waiting = Box::pin(budget.reserve_wait(spend(60)));
    assert!(poll!(waiting.as_mut()).is_pending());
    assert_eq!(budget.refusal().unwrap().requested, original.requested);
    live.settle(spend(10));
    let admitted = waiting.await.unwrap();
    assert_eq!(budget.refusal().unwrap().requested, original.requested);
    admitted.settle(spend(10));
}

#[derive(Debug)]
struct RefuseDispatch;
impl super::BeforeDispatch for RefuseDispatch {
    fn check(&self) -> crate::Result<()> {
        Err(crate::Error::Validation("soft deadline elapsed".into()))
    }
}

#[tokio::test]
async fn before_dispatch_rechecks_after_wait_and_refunds_unspent_reservation() {
    let ledger = cost_ledger(150).wait_for_capacity();
    let inner = Arc::new(GatedModel::default());
    let admitted = BudgetedModel::new(inner.clone(), ledger.clone(), policy());
    let refused = BudgetedModel::new(inner.clone(), ledger.clone(), policy())
        .before_dispatch(Arc::new(RefuseDispatch));
    let mut first = Box::pin(admitted.invoke(&(), request()));
    let mut waiting = Box::pin(refused.invoke(&(), request()));

    assert!(poll!(first.as_mut()).is_pending());
    assert!(poll!(waiting.as_mut()).is_pending());
    assert_eq!(inner.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(ledger.snapshot().reserved.cost_micros, 100);

    inner.release.add_permits(1);
    assert!(matches!(poll!(first.as_mut()), Poll::Ready(Ok(_))));
    assert!(matches!(
        poll!(waiting.as_mut()),
        Poll::Ready(Err(crate::Error::Validation(message))) if message == "soft deadline elapsed"
    ));

    assert_eq!(inner.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(ledger.snapshot().spent.cost_micros, 10);
    assert_eq!(ledger.snapshot().reserved, Spend::default());
}
