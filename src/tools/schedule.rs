//! Tool-round scheduling: pending calls, detached tasks, budgets, and the
//! auto-background policy.
//!
//! Extracted from `agent.rs` so the turn/UI code does not own the async
//! bookkeeping. Everything here is state plus pure helpers; the systems that
//! drive it live beside the producers (`poll_js_events`, `poll_script_events`).

use std::{
    collections::HashMap,
    env,
    time::{Duration, Instant},
};

use bevy::prelude::*;
use llm_history_model::history::{ToolCall, ToolCallResult};

use crate::messages::ToolRoundFinished;

/// Maximum nesting of tool calls below one root call.
pub(crate) const MAX_TOOL_DEPTH: u32 = 50;
/// Nested calls allowed per root call (the model's own batch entries are roots).
pub(crate) const MAX_CALLS_PER_ROOT: u32 = 2_000;

struct Budget {
    agent: Entity,
    remaining: u32,
}

/// One in-flight tool round. Results are stored by call index so asynchronous
/// completions cannot reorder the history entry.
pub(crate) struct PendingToolRound {
    pub(crate) agent: Entity,
    pub(crate) turn_id: u64,
    pub(crate) calls: Vec<ToolCall>,
    pub(crate) results: Vec<Option<ToolCallResult>>,
    pub(crate) remaining: usize,
}

/// A detached task whose result arrives as an agent message.
pub(crate) struct DetachedTask {
    pub(crate) task_id: u64,
    pub(crate) agent: Entity,
}

/// A non-detached async call inside a round.
struct PendingCall {
    round_id: u64,
    call_index: usize,
    started_at: Option<Instant>,
    /// `None` means "never detach"; `Some` is the grace period.
    auto_background: Option<Duration>,
}

/// Where a nested call's result goes.
pub(crate) enum NestedReply {
    /// A script worker blocked on its own channel.
    Channel(std::sync::mpsc::Sender<Result<String, String>>),
    /// An agent's interpreter thread blocked on its host token.
    Agent { agent: Entity, token: u64 },
}

/// A nested call made by `execute()` from a tool context.
pub(crate) struct NestedCall {
    pub(crate) agent: Entity,
    pub(crate) request: u64,
    deadline: Option<Instant>,
    reply: Option<NestedReply>,
}

/// Per-call background request from `execute(name, args, background_after)` or
/// `run_in_background`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AutoBackground {
    /// Use the tool's own policy (usually "never").
    Default,
    Never,
    Immediately,
    After(Duration),
}

/// Parse the `background_after` argument: missing/null -> default, negative ->
/// never, zero -> immediately, positive -> seconds.
pub(crate) fn parse_background_after(value: Option<f64>) -> Result<AutoBackground, String> {
    let Some(seconds) = value else {
        return Ok(AutoBackground::Default);
    };
    if !seconds.is_finite() {
        return Err("`background_after` must be a finite number of seconds".to_owned());
    }
    if seconds < 0.0 {
        return Ok(AutoBackground::Never);
    }
    if seconds == 0.0 {
        return Ok(AutoBackground::Immediately);
    }
    // Clamp before converting: `Duration::from_secs_f64` panics on values that
    // overflow `Duration`, and this value comes from model-authored code.
    let seconds = seconds.clamp(0.001, 24.0 * 60.0 * 60.0);
    Ok(AutoBackground::After(Duration::from_secs_f64(seconds)))
}

/// `run_in_background` on the JavaScript tools is the boolean spelling of the
/// same enum.
pub(crate) fn background_from_flag(value: Option<&serde_json::Value>) -> AutoBackground {
    match value {
        None | Some(serde_json::Value::Null) => AutoBackground::Default,
        Some(value) if crate::tools::is_truthy(Some(value)) => AutoBackground::Immediately,
        Some(_) => AutoBackground::Never,
    }
}

/// Resolve a per-call request against the tool default and the global policy.
pub(crate) fn resolve_auto_background(
    request: AutoBackground,
    tool_defaults_to_auto: bool,
    policy: Option<Duration>,
) -> Option<Duration> {
    match request {
        AutoBackground::Default => {
            if tool_defaults_to_auto {
                policy
            } else {
                None
            }
        }
        AutoBackground::Never => None,
        AutoBackground::Immediately => Some(Duration::ZERO),
        AutoBackground::After(duration) => Some(duration),
    }
}

/// Global grace period used by tools whose default is "auto".
#[derive(Resource)]
pub(crate) struct AutoBackgroundPolicy {
    pub(crate) auto_after: Option<Duration>,
}

impl Default for AutoBackgroundPolicy {
    fn default() -> Self {
        const DEFAULT_MS: u64 = 5_000;
        let millis = env::var("JS_AUTO_BACKGROUND_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MS);
        Self {
            auto_after: (millis > 0).then(|| Duration::from_millis(millis)),
        }
    }
}

/// Tracks tool rounds, async calls, detached tasks, nested calls, and budgets.
#[derive(Resource, Default)]
pub(crate) struct ToolRounds {
    active: HashMap<u64, PendingToolRound>,
    /// Request id -> round slot.
    pending: HashMap<u64, PendingCall>,
    /// Detached requests are deliberately not part of a round.
    detached: HashMap<u64, DetachedTask>,
    /// Calls made by scripts/tools through `execute`, keyed by request id.
    nested: HashMap<u64, NestedCall>,
    /// Remaining nested-call budget per root call, tagged with its agent.
    budgets: HashMap<u64, Budget>,
    pub(crate) next_round: u64,
    next_task_id: u64,
}

impl ToolRounds {
    /// Insert a round under a pre-allocated id.
    pub(crate) fn insert_round(&mut self, round_id: u64, round: PendingToolRound) {
        self.active.insert(round_id, round);
    }

    pub(crate) fn track(&mut self, request: u64, pending: PendingCallFields) {
        self.pending.insert(
            request,
            PendingCall {
                round_id: pending.round_id,
                call_index: pending.call_index,
                started_at: None,
                auto_background: pending.auto_background,
            },
        );
    }

    pub(crate) fn mark_started(&mut self, request: u64, at: Instant) {
        if let Some(pending) = self.pending.get_mut(&request) {
            pending.started_at = Some(at);
        }
    }

    /// Stop tracking a round (used by cancellation and failures).
    pub(crate) fn cancel_rounds(&mut self, agent: Entity) {
        let rounds = self
            .active
            .iter()
            .filter_map(|(id, round)| (round.agent == agent).then_some(*id))
            .collect::<Vec<_>>();
        for round_id in rounds {
            self.active.remove(&round_id);
        }
        self.pending
            .retain(|_, pending| self.active.contains_key(&pending.round_id));
        self.budgets.retain(|_, budget| budget.agent != agent);
    }

    /// Drop everything belonging to an agent, including detached work.
    pub(crate) fn remove_agent(&mut self, agent: Entity) {
        self.cancel_rounds(agent);
        self.detached.retain(|_, task| task.agent != agent);
        self.nested.retain(|_, call| call.agent != agent);
    }

    /// Register a detached request and return its user-visible id.
    pub(crate) fn detach(&mut self, request: u64, agent: Entity) -> u64 {
        self.next_task_id = self.next_task_id.wrapping_add(1);
        let task_id = self.next_task_id;
        self.detached
            .insert(request, DetachedTask { task_id, agent });
        task_id
    }

    pub(crate) fn take_detached(&mut self, request: u64) -> Option<DetachedTask> {
        self.detached.remove(&request)
    }

    pub(crate) fn add_nested(&mut self, call: NestedCall) {
        self.nested.insert(call.request, call);
    }

    pub(crate) fn take_nested(&mut self, request: u64) -> Option<NestedCall> {
        self.nested.remove(&request)
    }

    /// Nested calls whose grace period expired; returns their request ids.
    pub(crate) fn nested_due(&self, now: Instant) -> Vec<u64> {
        self.nested
            .iter()
            .filter(|(_, call)| {
                call.deadline.is_some_and(|deadline| {
                    now.saturating_duration_since(deadline) > Duration::ZERO
                })
            })
            .map(|(request, _)| *request)
            .collect()
    }

    /// Tool calls still in flight for `agent`: unresolved round slots, nested
    /// calls, and detached tasks that outlived their round.
    pub(crate) fn running_for(&self, agent: Entity) -> usize {
        let pending = self
            .active
            .values()
            .filter(|round| round.agent == agent)
            .map(|round| round.remaining)
            .sum::<usize>();
        let detached = self
            .detached
            .values()
            .filter(|task| task.agent == agent)
            .count();
        let nested = self
            .nested
            .values()
            .filter(|call| call.agent == agent)
            .count();
        pending + detached + nested
    }

    /// Open the nested-call budget for one root call.
    pub(crate) fn open_budget(&mut self, root: u64, agent: Entity) {
        self.budgets.insert(
            root,
            Budget {
                agent,
                remaining: MAX_CALLS_PER_ROOT,
            },
        );
    }

    /// Spend one nested call. Returns false when the root's budget is gone.
    pub(crate) fn spend_budget(&mut self, root: u64) -> bool {
        match self.budgets.get_mut(&root) {
            Some(budget) if budget.remaining > 0 => {
                budget.remaining -= 1;
                true
            }
            _ => false,
        }
    }

    /// Give one call back after a submission failed.
    pub(crate) fn refund_budget(&mut self, root: u64) {
        if let Some(budget) = self.budgets.get_mut(&root) {
            budget.remaining = budget.remaining.saturating_add(1);
        }
    }

    /// Close every budget belonging to one agent (called when its next tool
    /// batch starts, so detached work from the previous turn keeps its budget
    /// for that turn's lifetime).
    pub(crate) fn close_agent_budgets(&mut self, agent: Entity) {
        self.budgets.retain(|_, budget| budget.agent != agent);
    }

    #[cfg(test)]
    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len()
    }

    #[cfg(test)]
    pub(crate) fn has_pending(&self, request: u64) -> bool {
        self.pending.contains_key(&request)
    }

    #[cfg(test)]
    pub(crate) fn has_detached(&self, request: u64) -> bool {
        self.detached.contains_key(&request)
    }

    #[cfg(test)]
    pub(crate) fn has_active(&self, round_id: u64) -> bool {
        self.active.contains_key(&round_id)
    }
}

/// Fields needed when tracking an async call.
pub(crate) struct PendingCallFields {
    pub(crate) round_id: u64,
    pub(crate) call_index: usize,
    pub(crate) auto_background: Option<Duration>,
}

impl NestedCall {
    pub(crate) fn new(
        agent: Entity,
        request: u64,
        deadline: Option<Instant>,
        reply: NestedReply,
    ) -> Self {
        Self {
            agent,
            request,
            deadline,
            reply: Some(reply),
        }
    }

    /// Take the reply target, leaving the call registered as detached.
    pub(crate) fn take_reply(&mut self) -> Option<NestedReply> {
        self.reply.take()
    }
}

pub(crate) fn background_started_message(task_id: u64) -> String {
    format!(
        "Task {task_id} running in background, do not poll for it, continue your task or end turn to await result."
    )
}

/// Emit a finished round once every call has a result.
pub(crate) fn finish_round_if_ready(
    rounds: &mut ToolRounds,
    round_id: u64,
    finished: &mut MessageWriter<ToolRoundFinished>,
) {
    let Some(round) = rounds.active.get(&round_id) else {
        return;
    };
    if round.remaining != 0 {
        return;
    }

    let round = rounds
        .active
        .remove(&round_id)
        .expect("the round was checked above");
    finished.write(ToolRoundFinished {
        agent: round.agent,
        turn_id: round.turn_id,
        calls: round.calls,
        results: round
            .results
            .into_iter()
            .map(|result| result.expect("all tool results are available"))
            .collect(),
    });
}

/// Apply one finished async request to its round, if the round still exists.
pub(crate) fn complete_tool_request(
    rounds: &mut ToolRounds,
    request: u64,
    agent: Entity,
    result: Result<String, String>,
    finished: &mut MessageWriter<ToolRoundFinished>,
) {
    let Some(pending) = rounds.pending.remove(&request) else {
        return;
    };
    let round_id = pending.round_id;

    {
        let Some(round) = rounds.active.get_mut(&round_id) else {
            return;
        };
        if round.agent != agent {
            return;
        }

        let (text, is_error) = match result {
            Ok(text) => (text, false),
            Err(err) => (err, true),
        };
        round.results[pending.call_index] = Some(ToolCallResult {
            result: text,
            is_error,
            tool_use_id: round.calls[pending.call_index].tool_use_id.clone(),
        });
        round.remaining -= 1;
    }

    finish_round_if_ready(rounds, round_id, finished);
}

/// Detach round-tracked calls whose grace period has elapsed.
pub(crate) fn auto_background_long_calls(
    mut rounds: ResMut<ToolRounds>,
    mut finished: MessageWriter<ToolRoundFinished>,
) {
    let now = Instant::now();
    let due = rounds
        .pending
        .iter()
        .filter(|(_, pending)| {
            pending.auto_background.is_some_and(|grace| {
                pending
                    .started_at
                    .is_some_and(|started| now.saturating_duration_since(started) >= grace)
            })
        })
        .map(|(request, _)| *request)
        .collect::<Vec<_>>();

    for request in due {
        let Some(pending) = rounds.pending.remove(&request) else {
            continue;
        };
        let Some(agent) = rounds
            .active
            .get(&pending.round_id)
            .map(|round| round.agent)
        else {
            continue;
        };

        let task_id = rounds.detach(request, agent);
        let Some(round) = rounds.active.get_mut(&pending.round_id) else {
            continue;
        };
        round.results[pending.call_index] = Some(ToolCallResult {
            result: background_started_message(task_id),
            is_error: false,
            tool_use_id: round.calls[pending.call_index].tool_use_id.clone(),
        });
        round.remaining -= 1;

        finish_round_if_ready(&mut rounds, pending.round_id, &mut finished);
    }
}

/// Release nested callers whose grace period elapsed; their task keeps running
/// and reports through `TaskResult`.
pub(crate) fn auto_background_nested_calls(
    mut rounds: ResMut<ToolRounds>,
    js_handles: Query<&crate::js_runtime::JsRuntimeHandle>,
) {
    let now = Instant::now();
    for request in rounds.nested_due(now) {
        let Some(mut call) = rounds.take_nested(request) else {
            continue;
        };
        let Some(reply) = call.take_reply() else {
            continue;
        };
        let task_id = rounds.detach(request, call.agent);
        send_nested_reply(&js_handles, reply, Ok(background_started_message(task_id)));
    }
}

/// Deliver a nested result to whichever caller is waiting.
pub(crate) fn send_nested_reply(
    js_handles: &Query<&crate::js_runtime::JsRuntimeHandle>,
    reply: NestedReply,
    result: Result<String, String>,
) {
    match reply {
        NestedReply::Channel(sender) => {
            let _ = sender.send(result);
        }
        NestedReply::Agent { agent, token } => {
            if let Ok(handle) = js_handles.get(agent) {
                handle.respond(token, result);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_after_uses_the_documented_cases() {
        assert_eq!(
            parse_background_after(None).unwrap(),
            AutoBackground::Default
        );
        assert_eq!(
            parse_background_after(Some(-1.0)).unwrap(),
            AutoBackground::Never
        );
        assert_eq!(
            parse_background_after(Some(-0.0)).unwrap(),
            AutoBackground::Immediately
        );
        assert_eq!(
            parse_background_after(Some(0.0)).unwrap(),
            AutoBackground::Immediately
        );
        assert_eq!(
            parse_background_after(Some(2.5)).unwrap(),
            AutoBackground::After(Duration::from_millis(2500))
        );
        assert!(parse_background_after(Some(f64::NAN)).is_err());
        assert!(parse_background_after(Some(f64::INFINITY)).is_err());
    }

    #[test]
    fn huge_durations_are_clamped_not_panicking() {
        for value in [1e300, f64::MAX] {
            let AutoBackground::After(duration) = parse_background_after(Some(value)).unwrap()
            else {
                panic!("expected a clamped duration");
            };
            assert_eq!(duration, Duration::from_secs(86_400));
        }
    }

    #[test]
    fn budgets_are_closed_per_agent() {
        let mut world = bevy::prelude::World::new();
        let first = world.spawn_empty().id();
        let second = world.spawn_empty().id();
        let mut rounds = ToolRounds::default();
        rounds.open_budget(1, first);
        rounds.open_budget(2, second);

        rounds.close_agent_budgets(first);

        assert!(!rounds.spend_budget(1));
        assert!(rounds.spend_budget(2));
    }

    #[test]
    fn per_call_request_overrides_the_tool_default() {
        let policy = Some(Duration::from_secs(5));
        assert_eq!(
            resolve_auto_background(AutoBackground::Default, true, policy),
            policy
        );
        assert_eq!(
            resolve_auto_background(AutoBackground::Default, false, policy),
            None
        );
        assert_eq!(
            resolve_auto_background(AutoBackground::Never, true, policy),
            None
        );
        assert_eq!(
            resolve_auto_background(AutoBackground::Immediately, false, policy),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn budgets_are_scoped_to_one_root_call() {
        let mut rounds = ToolRounds::default();
        rounds.open_budget(1, bevy::prelude::Entity::PLACEHOLDER);
        rounds.open_budget(2, bevy::prelude::Entity::PLACEHOLDER);
        for _ in 0..MAX_CALLS_PER_ROOT {
            assert!(rounds.spend_budget(1));
        }
        assert!(!rounds.spend_budget(1));
        assert!(rounds.spend_budget(2));
    }
}
