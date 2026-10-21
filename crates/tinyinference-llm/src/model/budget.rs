//! Atomic spend admission around physical provider calls, including retries.
//!
//! Wrap each concrete provider below any retry, cache or fallback adapter.
//! Cloned budgets share one ledger; child budgets share the same lock and
//! debit every ancestor. Missing usage retains the reservation as charged:
//! a timed-out request may have been billed even when its response is lost.

use super::{ChatModel, ModelRequest, ModelResponse, ModelStream, ModelStreamItem};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// A token count and USD charge in integer microdollars.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spend {
    /// Input plus output tokens (reasoning is already included in output).
    pub tokens: u64,
    /// USD multiplied by one million, rounded up.
    pub cost_micros: u64,
}
impl Spend {
    fn add(self, rhs: Self) -> Self {
        Self {
            tokens: self.tokens.saturating_add(rhs.tokens),
            cost_micros: self.cost_micros.saturating_add(rhs.cost_micros),
        }
    }
    fn sub(self, rhs: Self) -> Self {
        Self {
            tokens: self.tokens.saturating_sub(rhs.tokens),
            cost_micros: self.cost_micros.saturating_sub(rhs.cost_micros),
        }
    }
}
/// Independent optional ceilings. Zero refuses all calls on that dimension.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpendLimits {
    /// Token ceiling across all calls.
    pub tokens: Option<u64>,
    /// Cost ceiling in microdollars across all calls.
    pub cost_micros: Option<u64>,
}
/// Snapshot separates known or conservatively charged spend from live reservations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetSnapshot {
    /// Completed calls, including retained reservations for unknown usage.
    pub spent: Spend,
    /// Calls still in flight.
    pub reserved: Spend,
}
/// Typed admission refusal; no provider call was started.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("budget exceeded: spent {snapshot:?}, requested {requested:?}, limits {limits:?}")]
pub struct BudgetExceeded {
    /// Ledger that refused the call (the turn or any ancestor run).
    pub snapshot: BudgetSnapshot,
    /// Amount the call attempted to reserve.
    pub requested: Spend,
    /// Ceiling that refused the reservation.
    pub limits: SpendLimits,
}
#[derive(Debug)]
struct Entry {
    limits: SpendLimits,
    snapshot: BudgetSnapshot,
    ancestors: Vec<usize>,
    refusal: Option<(usize, BudgetExceeded)>,
}
/// Shareable run budget. Child ledgers debit parent ledgers atomically.
#[derive(Debug, Clone)]
pub struct Budget {
    ledger: Arc<Mutex<Vec<Entry>>>,
    id: usize,
    changes: tokio::sync::watch::Sender<()>,
    wait_capacity: bool,
}
impl Budget {
    /// Start an independent run ledger.
    pub fn new(limits: SpendLimits) -> Self {
        Self {
            ledger: Arc::new(Mutex::new(vec![Entry {
                limits,
                snapshot: BudgetSnapshot::default(),
                ancestors: vec![],
                refusal: None,
            }])),
            id: 0,
            changes: tokio::sync::watch::channel(()).0,
            wait_capacity: false,
        }
    }
    /// Add a per-turn ceiling, still charged to this run and all ancestors.
    pub fn child(&self, limits: SpendLimits) -> Self {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut ancestors = ledger[self.id].ancestors.clone();
        ancestors.push(self.id);
        let id = ledger.len();
        ledger.push(Entry {
            limits,
            snapshot: BudgetSnapshot::default(),
            ancestors,
            refusal: None,
        });
        Self {
            ledger: self.ledger.clone(),
            id,
            changes: self.changes.clone(),
            wait_capacity: self.wait_capacity,
        }
    }
    /// Read this ledger's spend, including its children.
    pub fn snapshot(&self) -> BudgetSnapshot {
        self.ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[self.id]
            .snapshot
    }
    /// Most recent admission refusal made directly through this ledger.
    pub fn refusal(&self) -> Option<BudgetExceeded> {
        let ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ledger[self.id]
            .refusal
            .as_ref()
            .map(|(refusing_id, error)| {
                let mut error = error.clone();
                error.snapshot = ledger[*refusing_id].snapshot;
                error
            })
    }

    /// Opt this handle's budgeted models into waiting for live capacity.
    /// Clones and children created from this handle inherit the preference;
    /// existing other handles and direct [`Self::reserve`] calls stay fail-fast.
    /// Waiting never permits a call that settled spend cannot afford.
    pub fn wait_for_capacity(mut self) -> Self {
        self.wait_capacity = true;
        self
    }

    /// Atomically admit a physical call against this ledger and its ancestors.
    pub fn reserve(&self, requested: Spend) -> Result<Reservation, BudgetExceeded> {
        self.reserve_inner(requested, false)
            .map_err(|refusal| refusal.error)
    }

    fn reserve_inner(
        &self,
        requested: Spend,
        waiting: bool,
    ) -> Result<Reservation, AdmissionRefusal> {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut ids = ledger[self.id].ancestors.clone();
        ids.push(self.id);
        let capacity = ids.iter().copied().find(|&id| {
            let entry = &ledger[id];
            exceeds(
                entry.limits,
                entry.snapshot.spent.add(entry.snapshot.reserved),
                requested,
            )
        });
        if let Some(capacity_id) = capacity {
            // Check all settled ceilings under the same lock. A live parent
            // reservation must not hide a permanently unaffordable child.
            let permanent = waiting
                .then(|| {
                    ids.iter().copied().find(|&id| {
                        let entry = &ledger[id];
                        exceeds(entry.limits, entry.snapshot.spent, requested)
                    })
                })
                .flatten();
            let terminal = !waiting || permanent.is_some();
            let id = permanent.unwrap_or(capacity_id);
            let entry = &ledger[id];
            let error = BudgetExceeded {
                snapshot: entry.snapshot,
                requested,
                limits: entry.limits,
            };
            // Only terminal refusals feed facade error classification. A
            // temporary wait must neither publish nor clear another refusal.
            if terminal {
                ledger[self.id].refusal = Some((id, error.clone()));
            }
            return Err(AdmissionRefusal { error, terminal });
        }
        for &id in &ids {
            ledger[id].snapshot.reserved = ledger[id].snapshot.reserved.add(requested);
        }
        Ok(Reservation {
            budget: self.clone(),
            ids,
            requested,
            settled: false,
        })
    }
}
struct AdmissionRefusal {
    error: BudgetExceeded,
    terminal: bool,
}
/// An admitted call. Dropping without settlement charges the full reservation.
#[derive(Debug)]
pub struct Reservation {
    budget: Budget,
    ids: Vec<usize>,
    requested: Spend,
    settled: bool,
}
impl Reservation {
    /// Reconcile known usage. Unknown dimensions must use the reserved value.
    pub fn settle(mut self, actual: Spend) {
        self.finish(actual);
    }
    fn finish(&mut self, actual: Spend) {
        if self.settled {
            return;
        }
        let mut ledger = self
            .budget
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for &id in &self.ids {
            let snapshot = &mut ledger[id].snapshot;
            snapshot.reserved = snapshot.reserved.sub(self.requested);
            snapshot.spent = snapshot.spent.add(actual);
        }
        self.settled = true;
        drop(ledger);
        // Every settlement, including conservative Drop, changes admission.
        self.budget.changes.send_replace(());
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.finish(self.requested);
    }
}
/// Host-supplied conservative bound for any one call on the wrapped route.
///
/// The host must include all input modalities and provider pricing when choosing
/// these bounds. They are admission reservations, not provider billing caps.
#[derive(Debug, Clone, Copy)]
pub struct CallBudget {
    /// Conservative maximum input token count. Text requests exceeding this
    /// in serialized request bytes are refused; non-text modalities are refused.
    pub input_tokens: u64,
    /// Output cap imposed on every physical provider request.
    pub output_tokens: u32,
    /// Conservative cost bound including prompt, cache, reasoning and output.
    pub cost_micros: u64,
}

/// A synchronous, provider-neutral check before dispatching a reserved model
/// call and before each physical provider attempt.
///
/// The host can use this hook to recheck an absolute deadline or other
/// admission policy after [`BudgetedModel`] has waited for monetary capacity.
/// The check is repeated at each physical transport attempt, including a
/// retry or a lazy stream's first poll. It must be quick and must not perform
/// I/O. Implementations must tolerate repeated checks for the same logical
/// call; a committed host permit should make this operation idempotent.
pub trait BeforeDispatch: Send + Sync {
    /// Returns an error when the reserved call must not reach the provider.
    ///
    /// # Errors
    ///
    /// Returns the host's typed error when dispatch must be refused.
    fn check(&self) -> crate::Result<()>;

    /// Accounts one concrete physical provider send after the policy check.
    ///
    /// This is called once per transport send, including warm-up and retry
    /// requests. The default keeps existing host adapters source compatible;
    /// hosts that enforce a physical-send ceiling should override it.
    /// With independent nested policies, an earlier policy can conservatively
    /// admit an attempt before a later policy refuses it. These values bound
    /// admitted sends and are not exact wire-send metrics.
    ///
    /// # Errors
    ///
    /// Returns the host's typed error when this physical send exceeds policy.
    fn physical_attempt(&self) -> crate::Result<()> {
        Ok(())
    }
}

/// Supplies a final dispatch policy to an existing provider without adding a
/// monetary budget.
///
/// Built-in transports call `before_physical_attempt` immediately before
/// each send. The wrapper scopes the hook across invocation and stream polling,
/// and checks once before generic model work for compatibility with adapters
/// that do not expose a separate physical-send seam.
pub struct DispatchCheckedModel<State: Send + Sync> {
    inner: Arc<dyn ChatModel<State>>,
    before_dispatch: Arc<dyn BeforeDispatch>,
}

impl<State: Send + Sync> std::fmt::Debug for DispatchCheckedModel<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DispatchCheckedModel")
            .finish_non_exhaustive()
    }
}

impl<State: Send + Sync> DispatchCheckedModel<State> {
    /// Wraps a provider with a host-owned synchronous dispatch check.
    pub fn new(inner: Arc<dyn ChatModel<State>>, before_dispatch: Arc<dyn BeforeDispatch>) -> Self {
        Self {
            inner,
            before_dispatch,
        }
    }
}

struct DispatchHookScope {
    before_dispatch: Arc<dyn BeforeDispatch>,
    attempt_gate: std::sync::Mutex<()>,
    attempts: std::sync::atomic::AtomicUsize,
}

tokio::task_local! { static DISPATCH_HOOK: Vec<Arc<DispatchHookScope>>; }

/// Admits an unbudgeted physical attempt made by a concrete provider.
fn checked_unbudgeted_physical_attempt() -> crate::Result<Vec<Arc<DispatchHookScope>>> {
    DISPATCH_HOOK
        .try_with(|scope| {
            for scope in scope {
                scope.before_dispatch.check()?;
            }
            Ok(scope.clone())
        })
        .unwrap_or_else(|_| Ok(Vec::new()))
}

fn before_unbudgeted_physical_attempt() -> crate::Result<()> {
    let scopes = checked_unbudgeted_physical_attempt()?;
    let _attempts: Vec<_> = scopes
        .iter()
        .map(|scope| {
            scope
                .attempt_gate
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        })
        .collect();
    for scope in &scopes {
        scope.before_dispatch.physical_attempt()?;
    }
    for scope in &scopes {
        scope
            .attempts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    Ok(())
}

fn nested_dispatch_scopes(scope: Arc<DispatchHookScope>) -> Vec<Arc<DispatchHookScope>> {
    let mut scopes = DISPATCH_HOOK.try_with(Clone::clone).unwrap_or_default();
    scopes.push(scope);
    scopes
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for DispatchCheckedModel<State> {
    fn profile(&self) -> Option<&super::ModelProfile> {
        self.inner.profile()
    }

    fn supports_input(
        &self,
        modality: super::InputModality,
        mime: &str,
        source: super::InputSource,
    ) -> bool {
        self.inner.supports_input(modality, mime, source)
    }

    fn cache_identity(&self) -> Option<String> {
        self.inner.cache_identity()
    }

    fn tracks_physical_attempts(&self) -> bool {
        self.inner.tracks_physical_attempts()
    }

    fn has_pre_invocation_admission_gate(&self) -> bool {
        true
    }

    async fn fetch_deferred(
        &self,
        handle: &super::DeferredHandle,
    ) -> crate::Result<super::DeferredStatus> {
        if let Err(error) = self.before_dispatch.check() {
            mark_dispatch_refused_before_invocation();
            return Err(error);
        }
        if !self.inner.tracks_physical_attempts() && !self.inner.has_pre_invocation_admission_gate()
        {
            mark_untracked_inner_invoked();
        }
        let scope = Arc::new(DispatchHookScope {
            before_dispatch: self.before_dispatch.clone(),
            attempt_gate: std::sync::Mutex::new(()),
            attempts: std::sync::atomic::AtomicUsize::new(0),
        });
        DISPATCH_HOOK
            .scope(
                nested_dispatch_scopes(scope),
                self.inner.fetch_deferred(handle),
            )
            .await
    }

    async fn invoke(&self, state: &State, request: ModelRequest) -> crate::Result<ModelResponse> {
        if let Err(error) = self.before_dispatch.check() {
            mark_dispatch_refused_before_invocation();
            return Err(error);
        }
        if !self.inner.tracks_physical_attempts() && !self.inner.has_pre_invocation_admission_gate()
        {
            mark_untracked_inner_invoked();
        }
        let scope = Arc::new(DispatchHookScope {
            before_dispatch: self.before_dispatch.clone(),
            attempt_gate: std::sync::Mutex::new(()),
            attempts: std::sync::atomic::AtomicUsize::new(0),
        });
        DISPATCH_HOOK
            .scope(
                nested_dispatch_scopes(scope),
                self.inner.invoke(state, request),
            )
            .await
    }

    async fn stream(&self, state: &State, request: ModelRequest) -> crate::Result<ModelStream> {
        if let Err(error) = self.before_dispatch.check() {
            mark_dispatch_refused_before_invocation();
            return Err(error);
        }
        if !self.inner.tracks_physical_attempts() && !self.inner.has_pre_invocation_admission_gate()
        {
            mark_untracked_inner_invoked();
        }
        let scope = Arc::new(DispatchHookScope {
            before_dispatch: self.before_dispatch.clone(),
            attempt_gate: std::sync::Mutex::new(()),
            attempts: std::sync::atomic::AtomicUsize::new(0),
        });
        let mut dispatch_scopes = Some(nested_dispatch_scopes(Arc::clone(&scope)));
        let stream = DISPATCH_HOOK
            .scope(
                dispatch_scopes
                    .as_ref()
                    .expect("dispatch scopes are initialized")
                    .clone(),
                self.inner.stream(state, request),
            )
            .await?;
        let metadata = stream.metadata().clone();
        let mut stream = Box::pin(stream);
        let mut scope = Some(scope);
        let guarded = futures::stream::poll_fn(move |cx| {
            let Some(current_scope) = scope.as_ref().cloned() else {
                return std::task::Poll::Ready(None);
            };
            let Some(mut current_scopes) = dispatch_scopes.as_ref().cloned() else {
                return std::task::Poll::Ready(None);
            };
            if !current_scopes
                .iter()
                .any(|scope| Arc::ptr_eq(scope, &current_scope))
            {
                current_scopes.push(current_scope);
            }
            let item = DISPATCH_HOOK.sync_scope(current_scopes, || {
                futures::Stream::poll_next(stream.as_mut(), cx)
            });
            if matches!(
                &item,
                std::task::Poll::Ready(Some(
                    ModelStreamItem::Completed(_)
                        | ModelStreamItem::Failed(_)
                        | ModelStreamItem::ProviderFailed(_)
                        | ModelStreamItem::Deferred(_)
                )) | std::task::Poll::Ready(None)
            ) {
                scope.take();
                dispatch_scopes.take();
            }
            item
        });
        Ok(ModelStream::new(Box::pin(guarded)).with_metadata(metadata))
    }
}

// Concrete transports call this before sending: adapters may retry request
// shape internally, below ChatModel. The first attempt owns the outer permit;
// each extra attempt is conservatively charged before dispatch.
struct AttemptScope {
    budget: Budget,
    requested: Spend,
    before_dispatch: Option<Arc<dyn BeforeDispatch>>,
    has_pre_invocation_admission_gate: bool,
    attempt_gate: std::sync::Mutex<()>,
    attempts: std::sync::atomic::AtomicUsize,
    refused_before_attempt: std::sync::atomic::AtomicBool,
    untracked_inner_invoked: std::sync::atomic::AtomicBool,
}

impl AttemptScope {
    fn can_refund_before_first_attempt(&self) -> bool {
        (self
            .refused_before_attempt
            .load(std::sync::atomic::Ordering::SeqCst)
            || self.has_pre_invocation_admission_gate)
            && self.attempts.load(std::sync::atomic::Ordering::SeqCst) == 0
            && !self
                .untracked_inner_invoked
                .load(std::sync::atomic::Ordering::SeqCst)
    }
}
tokio::task_local! { static PHYSICAL_CALL: Vec<Arc<AttemptScope>>; }

fn mark_physical_refused_before_attempt() {
    let _ = PHYSICAL_CALL.try_with(|scopes| {
        for scope in scopes {
            scope
                .refused_before_attempt
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    });
}

fn mark_dispatch_refused_before_invocation() {
    let _ = PHYSICAL_CALL.try_with(|scopes| {
        for scope in scopes {
            scope
                .refused_before_attempt
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    });
}

fn mark_untracked_inner_invoked() {
    let _ = PHYSICAL_CALL.try_with(|scopes| {
        for scope in scopes {
            scope
                .untracked_inner_invoked
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    });
}

pub(super) fn mark_untracked_fallback_child_invoked() {
    mark_untracked_inner_invoked();
}

fn nested_attempt_scopes(scope: Arc<AttemptScope>) -> Vec<Arc<AttemptScope>> {
    let mut scopes = PHYSICAL_CALL.try_with(Clone::clone).unwrap_or_default();
    scopes.push(scope);
    scopes
}
/// Admit an extra physical attempt made internally by a concrete provider.
///
/// The first attempt uses the invocation reservation; later attempts must
/// atomically reserve again. Checks active dispatch and budget scopes
/// immediately before a physical provider send. Provider implementations
/// should call this for every send,
/// including retries, and report [`ChatModel::tracks_physical_attempts`] as
/// true only when every send is covered.
///
/// Retry admission is synchronous and does not wait for live reservations to
/// settle. When a wait-mode budget cannot reserve a retry only because of live
/// capacity, this returns [`crate::Error::BudgetExceeded`] without publishing
/// a terminal refusal through [`Budget::refusal`]. A retry that settled spend
/// cannot afford remains a terminal refusal.
///
/// Outside an active budgeted or dispatch-checked invocation this is a no-op.
pub fn before_physical_attempt() -> crate::Result<()> {
    PHYSICAL_CALL
        .try_with(|scopes| {
            let _attempts: Vec<_> = scopes
                .iter()
                .map(|scope| {
                    scope
                        .attempt_gate
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                })
                .collect();
            for scope in scopes {
                if let Some(hook) = &scope.before_dispatch
                    && let Err(error) = hook.check()
                {
                    mark_physical_refused_before_attempt();
                    return Err(error);
                }
            }
            // A DispatchCheckedModel may wrap this budgeted model. Its
            // independent admission scope must also be checked at every
            // physical send, including provider retries.
            let dispatch_scopes = match checked_unbudgeted_physical_attempt() {
                Ok(scope) => scope,
                Err(error) => {
                    mark_physical_refused_before_attempt();
                    return Err(error);
                }
            };
            // Reserve retry money before consuming a host call slot. A denied
            // monetary reservation therefore never charges an unsent attempt.
            let mut retry_reservations = Vec::new();
            for scope in scopes {
                if scope.attempts.load(std::sync::atomic::Ordering::SeqCst) > 0 {
                    match scope
                        .budget
                        .reserve_inner(scope.requested, scope.budget.wait_capacity)
                    {
                        Ok(reservation) => retry_reservations.push(reservation),
                        Err(refusal) => {
                            for reservation in retry_reservations {
                                reservation.settle(Spend::default());
                            }
                            mark_physical_refused_before_attempt();
                            return Err(crate::Error::BudgetExceeded(refusal.error));
                        }
                    }
                }
            }
            let _dispatch_attempts: Vec<_> = dispatch_scopes
                .iter()
                .map(|scope| {
                    scope
                        .attempt_gate
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                })
                .collect();
            for scope in scopes {
                if let Some(hook) = &scope.before_dispatch
                    && let Err(error) = hook.physical_attempt()
                {
                    mark_physical_refused_before_attempt();
                    for reservation in retry_reservations {
                        reservation.settle(Spend::default());
                    }
                    return Err(error);
                }
            }
            for dispatch_scope in &dispatch_scopes {
                if let Err(error) = dispatch_scope.before_dispatch.physical_attempt() {
                    mark_physical_refused_before_attempt();
                    for reservation in retry_reservations {
                        reservation.settle(Spend::default());
                    }
                    return Err(error);
                }
            }
            for dispatch_scope in &dispatch_scopes {
                dispatch_scope
                    .attempts
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            for scope in scopes {
                scope
                    .attempts
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            // An accepted attempt remains charged even if the HTTP send fails.
            drop(retry_reservations);
            Ok(())
        })
        .unwrap_or_else(|_| before_unbudgeted_physical_attempt())
}

/// Applies a shared budget to a concrete provider, before every invocation.
pub struct BudgetedModel<State: Send + Sync> {
    inner: Arc<dyn ChatModel<State>>,
    budget: Budget,
    call: CallBudget,
    before_dispatch: Option<Arc<dyn BeforeDispatch>>,
}
impl<State: Send + Sync> std::fmt::Debug for BudgetedModel<State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetedModel")
            .field("budget", &self.budget)
            .field("call", &self.call)
            .finish_non_exhaustive()
    }
}
impl<State: Send + Sync> BudgetedModel<State> {
    /// Wrap a provider below retries/fallbacks so each attempt reserves separately.
    pub fn new(inner: Arc<dyn ChatModel<State>>, budget: Budget, call: CallBudget) -> Self {
        Self {
            inner,
            budget,
            call,
            before_dispatch: None,
        }
    }

    /// Install a synchronous policy check after monetary admission, before
    /// provider invocation or stream construction, and before each physical
    /// transport attempt.
    pub fn before_dispatch(mut self, hook: Arc<dyn BeforeDispatch>) -> Self {
        self.before_dispatch = Some(hook);
        self
    }
    async fn admit(&self, request: &mut ModelRequest) -> crate::Result<Reservation> {
        if ["max_tokens", "max_completion_tokens", "max_output_tokens"]
            .iter()
            .any(|key| request.provider_options.get(*key).is_some())
        {
            return Err(crate::Error::Validation(
                "budgeted model output caps must use typed max_tokens".into(),
            ));
        }
        // Count the complete typed request: response schemas and provider
        // instructions also become model input. Framing remains host-bounded.
        let bytes = serde_json::to_vec(&request)?.len() as u64;
        let text_only = request.messages.iter().all(|message| {
            use crate::message::{ContentBlock, Message};
            let blocks = match message {
                Message::User(message) => &message.content,
                Message::Assistant(message) => &message.content,
                Message::Tool(message) => &message.content,
                Message::System(_) | Message::Custom(_) => return true,
            };
            blocks.iter().all(|block| {
                matches!(
                    block,
                    ContentBlock::Text(_) | ContentBlock::Json(_) | ContentBlock::Thinking { .. }
                )
            })
        });
        if bytes > self.call.input_tokens || !text_only {
            return Err(crate::Error::Validation(
                "budgeted model requires text input within its conservative input-token bound"
                    .into(),
            ));
        }
        request.max_tokens = Some(
            request
                .max_tokens
                .unwrap_or(self.call.output_tokens)
                .min(self.call.output_tokens),
        );
        let amount = Spend {
            tokens: self
                .call
                .input_tokens
                .saturating_add(request.max_tokens.unwrap_or_default() as u64),
            cost_micros: self.call.cost_micros,
        };
        let admission = if self.budget.wait_capacity {
            self.budget.reserve_wait(amount).await
        } else {
            self.budget.reserve(amount)
        };
        admission.map_err(crate::Error::BudgetExceeded)
    }
    fn settle(reservation: Reservation, response: &ModelResponse) {
        let reserved = reservation.requested;
        let spent = if response.served_from_cache {
            Spend::default()
        } else {
            Spend {
                tokens: response.usage.map_or(reserved.tokens, |usage| {
                    usage
                        .input_tokens
                        .saturating_add(usage.output_tokens)
                        .max(usage.total_tokens)
                }),
                cost_micros: response
                    .usage
                    .and_then(|usage| usage.charged_amount)
                    .and_then(|amount| u64::try_from(amount.micros).ok())
                    .unwrap_or(reserved.cost_micros),
            }
        };
        reservation.settle(spent);
    }
}
#[async_trait]
impl<State: Send + Sync> ChatModel<State> for BudgetedModel<State> {
    fn profile(&self) -> Option<&super::ModelProfile> {
        self.inner.profile()
    }
    fn supports_input(
        &self,
        _modality: super::InputModality,
        _mime: &str,
        _source: super::InputSource,
    ) -> bool {
        false
    }
    fn cache_identity(&self) -> Option<String> {
        self.inner.cache_identity()
    }

    fn tracks_physical_attempts(&self) -> bool {
        self.inner.tracks_physical_attempts()
    }

    fn has_pre_invocation_admission_gate(&self) -> bool {
        true
    }
    async fn invoke(
        &self,
        state: &State,
        mut request: ModelRequest,
    ) -> crate::Result<ModelResponse> {
        let reservation = self.admit(&mut request).await?;
        let scope = Arc::new(AttemptScope {
            budget: self.budget.clone(),
            requested: reservation.requested,
            before_dispatch: self.before_dispatch.clone(),
            has_pre_invocation_admission_gate: self.inner.has_pre_invocation_admission_gate(),
            attempt_gate: std::sync::Mutex::new(()),
            attempts: std::sync::atomic::AtomicUsize::new(0),
            refused_before_attempt: std::sync::atomic::AtomicBool::new(false),
            untracked_inner_invoked: std::sync::atomic::AtomicBool::new(false),
        });
        if let Some(hook) = &self.before_dispatch
            && let Err(error) = hook.check()
        {
            reservation.settle(Spend::default());
            return Err(error);
        }
        if !self.inner.tracks_physical_attempts() && !self.inner.has_pre_invocation_admission_gate()
        {
            mark_untracked_inner_invoked();
            scope
                .untracked_inner_invoked
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
        let response = match PHYSICAL_CALL
            .scope(
                nested_attempt_scopes(Arc::clone(&scope)),
                self.inner.invoke(state, request),
            )
            .await
        {
            Ok(response) => response,
            Err(error) if scope.can_refund_before_first_attempt() => {
                reservation.settle(Spend::default());
                return Err(error);
            }
            Err(error) => return Err(error),
        };
        Self::settle(reservation, &response);
        Ok(response)
    }
    async fn stream(&self, state: &State, mut request: ModelRequest) -> crate::Result<ModelStream> {
        let reservation = self.admit(&mut request).await?;
        let scope = Arc::new(AttemptScope {
            budget: self.budget.clone(),
            requested: reservation.requested,
            before_dispatch: self.before_dispatch.clone(),
            has_pre_invocation_admission_gate: self.inner.has_pre_invocation_admission_gate(),
            attempt_gate: std::sync::Mutex::new(()),
            attempts: std::sync::atomic::AtomicUsize::new(0),
            refused_before_attempt: std::sync::atomic::AtomicBool::new(false),
            untracked_inner_invoked: std::sync::atomic::AtomicBool::new(false),
        });
        if let Some(hook) = &self.before_dispatch
            && let Err(error) = hook.check()
        {
            reservation.settle(Spend::default());
            return Err(error);
        }
        if !self.inner.tracks_physical_attempts() && !self.inner.has_pre_invocation_admission_gate()
        {
            mark_untracked_inner_invoked();
            scope
                .untracked_inner_invoked
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
        let mut attempt_scopes = Some(nested_attempt_scopes(Arc::clone(&scope)));
        let stream = match PHYSICAL_CALL
            .scope(
                attempt_scopes
                    .as_ref()
                    .expect("attempt scopes are initialized")
                    .clone(),
                self.inner.stream(state, request),
            )
            .await
        {
            Ok(stream) => stream,
            Err(error) if scope.can_refund_before_first_attempt() => {
                reservation.settle(Spend::default());
                return Err(error);
            }
            Err(error) => return Err(error),
        };
        let metadata = stream.metadata().clone();
        let mut stream = Box::pin(stream);
        let mut reservation = Some(reservation);
        let mut scope = Some(scope);
        // Lazy providers can send/retry while polled; retain the same attempt
        // counter and admission scope over both construction and consumption.
        let stream = futures::stream::poll_fn(move |cx| {
            let Some(current_scope) = scope.as_ref().cloned() else {
                return std::task::Poll::Ready(None);
            };
            let Some(mut current_scopes) = attempt_scopes.as_ref().cloned() else {
                return std::task::Poll::Ready(None);
            };
            if !current_scopes
                .iter()
                .any(|scope| Arc::ptr_eq(scope, &current_scope))
            {
                current_scopes.push(current_scope.clone());
            }
            let item = PHYSICAL_CALL.sync_scope(current_scopes, || {
                let item = futures::Stream::poll_next(stream.as_mut(), cx);
                match &item {
                    std::task::Poll::Ready(Some(ModelStreamItem::Completed(response))) => {
                        if let Some(permit) = reservation.take() {
                            Self::settle(permit, response);
                        }
                    }
                    std::task::Poll::Ready(Some(
                        ModelStreamItem::Failed(_)
                        | ModelStreamItem::ProviderFailed(_)
                        | ModelStreamItem::Deferred(_),
                    ))
                    | std::task::Poll::Ready(None) => {
                        if current_scope.can_refund_before_first_attempt() {
                            if let Some(reservation) = reservation.take() {
                                reservation.settle(Spend::default());
                            }
                        } else {
                            drop(reservation.take());
                        }
                    }
                    _ => (),
                }
                item
            });
            if matches!(
                &item,
                std::task::Poll::Ready(Some(
                    ModelStreamItem::Completed(_)
                        | ModelStreamItem::Failed(_)
                        | ModelStreamItem::ProviderFailed(_)
                        | ModelStreamItem::Deferred(_)
                )) | std::task::Poll::Ready(None)
            ) {
                // Drop the host permit at terminal delivery, even if the
                // caller retains the fused stream object for later polling.
                scope.take();
                attempt_scopes.take();
            }
            item
        });
        Ok(ModelStream::new(Box::pin(stream)).with_metadata(metadata))
    }
}
#[cfg(test)]
#[path = "budget_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "budget_wait_tests.rs"]
mod wait_tests;

// Keep fail-fast and waiting admission on the same ceiling arithmetic,
// including explicit zero limits and saturating committed spend.
fn exceeds(limits: SpendLimits, committed: Spend, requested: Spend) -> bool {
    limits.tokens.is_some_and(|max| {
        committed.tokens >= max || requested.tokens > max.saturating_sub(committed.tokens)
    }) || limits.cost_micros.is_some_and(|max| {
        committed.cost_micros >= max
            || requested.cost_micros > max.saturating_sub(committed.cost_micros)
    })
}

#[path = "budget_wait.rs"]
mod wait;
