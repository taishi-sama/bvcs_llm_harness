//! Agent state, turn loop, tool rounds, and per-agent systems.
//!
//! Chat-log rendering, cards, collapsibles, and scrolling live in
//! [`crate::transcript`].

use std::{
    collections::{HashMap, HashSet, VecDeque},
    time::Duration,
};

use bevy::{
    input_focus::{FocusCause, InputFocus},
    log::error,
    prelude::*,
    text::EditableText,
    ui_widgets::Activate,
};
use bevy_tokio_tasks::TokioTasksRuntime;
use llm_api::{
    anthropic,
    model_config::{ModelConfig, ModelKind},
    openai_completions, openai_responses,
    stream::{RawStreamBytes, StreamEvent, finish_draft_for},
    usage::{Usage, UsageProvider},
};
use llm_history_model::history::{DraftBlock, Entry, MessageContent, ToolCallResult};
use serde_json::Value;
use tokio::sync::watch;

use crate::{
    config::{AppConfig, Provider},
    js_runtime::{HostCommand, HostRequest, JsChannels, JsEvent, JsRuntimeHandle},
    messages::*,
    tools::{
        AdvertisedTools, CallContext, Execution, ScriptHostRequest, ScriptRuntime, ScriptTask,
        ToolSourceCache, ToolWorld, advertised_tools, background_started_message, execute_tool,
        find, interpreter_task,
        schedule::{
            AutoBackground, AutoBackgroundPolicy, MAX_TOOL_DEPTH, NestedCall, NestedReply,
            PendingCallFields, PendingToolRound, ToolRounds, background_from_flag,
            complete_tool_request, finish_round_if_ready, parse_background_after,
            resolve_auto_background, send_nested_reply,
        },
        validate_script_arguments,
    },
    transcript::{
        AgentTab, AgentView, CollapsibleText, RetryAgentButton, TextCollapseConfig,
        begin_live_assistant, create_agent_tab, ensure_live_assistant, response_display_text,
        response_thinking_text, set_collapsible_text, set_display, set_text, stats_text,
    },
    transport::{request_body, stream_transport},
};

#[cfg(test)]
use crate::transcript::{
    AgentPanel, AssistantView, TranscriptCard, TranscriptLog, close_active_agent_menu,
};

/// Prevent accidental runaway agent spawning from exhausting UI/VRAM.
pub(crate) const MAX_AGENTS: usize = 64;

// ---------------------------------------------------------------------------
// Agent components
// ---------------------------------------------------------------------------

/// One independent conversation. All agent state lives on this component.
#[derive(Component)]
pub(crate) struct Agent {
    name: String,
    provider: Provider,
    model_name: String,
    history: Vec<Entry>,
    pending: bool,
    current_turn: Option<u64>,
    streaming_text: String,
    streaming_thinking: String,
    /// Messages waiting for the next legal history boundary (after a reply or
    /// a completed tool-call round), in arrival order.
    pending_messages: VecDeque<QueuedMessage>,
    /// Last unanswered user message, kept so Retry can resend it after a
    /// failure or cancellation.
    retry_message: Option<String>,
    /// History immediately before the most recent applied change.
    ///
    /// Diagnostics dump both snapshots when a request fails or the prompt
    /// cache regresses, so the exact step that caused the problem is visible.
    previous_history: Vec<Entry>,
    /// Short label of the most recent applied history change.
    last_history_change: &'static str,
    /// Usage reported by the previous completed request.
    last_usage: Option<Usage>,
    /// Whether a turn was rolled back since the last usage report. Rollback is
    /// the most common legitimate reason for a prompt-cache regression.
    rolled_back_since_last_usage: bool,
    /// Serialized body of the most recent request, kept for error dumps.
    last_request_body: Option<String>,
    /// Serialized body of the request before `last_request_body`.
    ///
    /// Comparing the two bodies distinguishes a provider-side cache eviction
    /// from the harness changing a committed prefix between requests.
    previous_request_body: Option<String>,
}

impl Agent {
    pub(crate) fn new(name: String, provider: Provider, model_name: String) -> Self {
        let history = vec![Entry::SystemPrompt {
            content: format!(
                "You are the LLM agent named {name}. Avoid using markdown and LaTeX \
                 notation."
            ),
        }];
        Self {
            name: name.clone(),
            provider,
            model_name,
            previous_history: history.clone(),
            history,
            pending: false,
            current_turn: None,
            streaming_text: String::new(),
            streaming_thinking: String::new(),
            pending_messages: VecDeque::new(),
            retry_message: None,
            last_history_change: "initial",
            last_usage: None,
            rolled_back_since_last_usage: false,
            last_request_body: None,
            previous_request_body: None,
        }
    }

    /// Snapshot the current history before a mutation is applied.
    fn note_history_change(&mut self, change: &'static str) {
        self.previous_history = self.history.clone();
        self.last_history_change = change;
    }

    /// Append one entry while preserving the previous history step.
    fn push_history(&mut self, change: &'static str, entry: Entry) {
        self.note_history_change(change);
        self.history.push(entry);
    }

    /// Append several entries while preserving the previous history step.
    fn extend_history(&mut self, change: &'static str, entries: Vec<Entry>) {
        if entries.is_empty() {
            return;
        }
        self.note_history_change(change);
        self.history.extend(entries);
    }

    /// Remove the pending user turn while preserving the pre-rollback history.
    fn rollback_pending_turn(&mut self) -> Option<String> {
        self.note_history_change("turn_rollback");
        self.rolled_back_since_last_usage = true;
        rollback_pending_turn(&mut self.history)
    }

    fn model_config(&self, tools: Vec<llm_api::generic_tool::Tool>) -> ModelConfig {
        ModelConfig {
            kind: self.provider.kind(&self.model_name),
            tools,
            max_tokens: Some(8192),
        }
    }

    /// Agent title as shown in tabs and used for cross-agent lookups.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Full conversation history, including the system prompt.
    pub(crate) fn history(&self) -> &[Entry] {
        &self.history
    }

    /// Text received so far for the in-flight response.
    pub(crate) fn streaming_text(&self) -> &str {
        &self.streaming_text
    }

    /// Reasoning received so far for the in-flight response.
    pub(crate) fn streaming_thinking(&self) -> &str {
        &self.streaming_thinking
    }

    /// Whether a request is currently streaming or waiting for tools.
    pub(crate) fn has_active_turn(&self) -> bool {
        self.current_turn.is_some()
    }

    /// Append a history entry from display-layer tests.
    #[cfg(test)]
    pub(crate) fn push_entry_for_test(&mut self, entry: Entry) {
        self.push_history("test", entry);
    }

    /// Pretend a response is in flight from display-layer tests.
    #[cfg(test)]
    pub(crate) fn mark_active_turn_for_test(&mut self) {
        self.current_turn = Some(1);
    }
}

/// Append-only JSONL file under the process working directory. Kept separate
/// from the normal log stream so a full history can be inspected without
/// digging through unrelated output.
const HISTORY_DUMP_FILE: &str = "history_dumps.jsonl";

/// Whether the previous request body is an exact prefix (all non-message
/// fields equal, messages a prefix) of the current one.
///
/// `None` when either body is unavailable or not JSON.
fn request_prefix_append_only(previous: Option<&str>, current: Option<&str>) -> Option<bool> {
    let previous: Value = serde_json::from_str(previous?).ok()?;
    let current: Value = serde_json::from_str(current?).ok()?;
    let previous = previous.as_object()?;
    let current = current.as_object()?;

    if previous
        .iter()
        .any(|(key, value)| key != "messages" && current.get(key) != Some(value))
    {
        return Some(false);
    }

    let previous_messages = previous.get("messages")?.as_array()?;
    let current_messages = current.get("messages")?.as_array()?;
    Some(
        previous_messages.len() <= current_messages.len()
            && previous_messages
                .iter()
                .zip(current_messages)
                .all(|(before, after)| before == after),
    )
}

/// Emit the current and previous history as one parseable JSON record.
///
/// The dump deliberately contains both snapshots: `previous_history` is the
/// state before the last applied change and `current_history` is what is about
/// to be (or was just) sent. `request_body` is the exact body of the most
/// recent request, so a missing `reasoning_content` can be confirmed directly.
fn log_history_dump(agent: &Agent, reason: &str, details: Value) {
    let request_body = agent
        .last_request_body
        .as_deref()
        .and_then(|body| serde_json::from_str::<Value>(body).ok());
    let previous_request_body = agent
        .previous_request_body
        .as_deref()
        .and_then(|body| serde_json::from_str::<Value>(body).ok());
    let payload = serde_json::json!({
        "event": "history_dump",
        "reason": reason,
        "timestamp_unix_ms": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis() as u64)
            .unwrap_or(0),
        "agent": agent.name,
        "provider": agent.provider.label(),
        "model": agent.model_name,
        "last_history_change": agent.last_history_change,
        "previous_history": &agent.previous_history,
        "current_history": &agent.history,
        "request_prefix_append_only": request_prefix_append_only(
            agent.previous_request_body.as_deref(),
            agent.last_request_body.as_deref(),
        ),
        "previous_request_body": previous_request_body,
        "request_body": request_body,
        "details": details,
    });

    error!("{payload}");
    if let Err(err) = append_history_dump(&payload.to_string()) {
        error!("could not append to {HISTORY_DUMP_FILE}: {err}");
    }
}

fn append_history_dump(line: &str) -> std::io::Result<()> {
    use std::io::Write as _;

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(HISTORY_DUMP_FILE)?;
    writeln!(file, "{line}")
}

/// DeepSeek-style prompt caches are reported in fixed 128-token blocks, so a
/// one-block decrease is accounting granularity rather than a real regression.
const CACHE_BLOCK_TOKENS: u64 = 128;

/// Whether `previous` is an append-only prefix of `current`.
///
/// A cache regression that is not append-only means a committed entry was
/// rewritten or removed, which is the harness-side failure we want to catch.
fn history_is_append_only(previous: &[Entry], current: &[Entry]) -> bool {
    previous.len() <= current.len()
        && previous.iter().zip(current).all(|(before, after)| {
            serde_json::to_value(before).ok() == serde_json::to_value(after).ok()
        })
}

/// Classify a cached-token regression. `None` means a known benign cause (one
/// cache block of rounding noise) that should not be dumped.
fn classify_cache_drop(before: u64, after: u64, rolled_back: bool) -> Option<&'static str> {
    if rolled_back {
        // Rollback is expected but worth recording: it explains the miss.
        return Some("turn_rollback");
    }
    if after == 0 {
        return Some("full_cache_miss");
    }
    if before.saturating_sub(after) <= CACHE_BLOCK_TOKENS {
        return None;
    }
    Some("unexplained")
}

/// Track usage between requests and dump both history snapshots when the
/// cached prompt prefix shrinks unexpectedly.
fn record_usage(agent: &mut Agent, usage: &Usage) {
    let previous = agent.last_usage.clone();
    let previous_cached = previous
        .as_ref()
        .and_then(|usage| usage.cached_input_tokens);
    let current_cached = usage.cached_input_tokens;

    if let (Some(previous_cached), Some(current_cached)) = (previous_cached, current_cached)
        && current_cached < previous_cached
        && let Some(likely_reason) = classify_cache_drop(
            previous_cached,
            current_cached,
            agent.rolled_back_since_last_usage,
        )
    {
        log_history_dump(
            agent,
            "cache_miss",
            serde_json::json!({
                "previous_usage": previous,
                "current_usage": usage,
                "cached_input_tokens_before": previous_cached,
                "cached_input_tokens_after": current_cached,
                "cached_input_tokens_delta": previous_cached - current_cached,
                "history_change": agent.last_history_change,
                "history_append_only": history_is_append_only(
                    &agent.previous_history,
                    &agent.history,
                ),
                "likely_reason": likely_reason,
            }),
        );
    }

    agent.last_usage = Some(usage.clone());
    agent.rolled_back_since_last_usage = false;
}

/// A message waiting for the next legal insertion point in an agent's history.
pub(crate) enum QueuedMessage {
    /// Typed by the user into this agent's input box.
    User(String),
    /// Sent by another agent through `send_message_to_agent`.
    Agent { from: String, text: String },
    /// Initial task assigned by `spawn_agent`; delivered exactly as written.
    Task(String),
    /// Result of a detached `run_in_background` interpreter task.
    BackgroundResult {
        task_id: u64,
        result: String,
        is_error: bool,
    },
}

/// Handles for the shared shell that hosts all agents.
#[derive(Resource)]
pub(crate) struct AgentLayout {
    pub(crate) tabs: Entity,
    pub(crate) panels: Entity,
    pub(crate) next_number: usize,
}

/// Marks the single agent whose transcript menu currently exists.
///
/// The menu panel and all transcript cards live only while this component is
/// present. Insert/remove observers in [`crate::transcript`] own that lifetime.
#[derive(Component)]
pub(crate) struct ActiveAgent;

/// Move the active-agent marker to `next` (or clear it when `next` is `None`).
pub(crate) fn activate_agent(
    commands: &mut Commands,
    active: &Query<Entity, With<ActiveAgent>>,
    next: Option<Entity>,
) {
    for entity in active {
        if Some(entity) != next {
            // Agent removal despawns the previous owner, so this command can be
            // applied after the entity is gone. Removal is best-effort here.
            commands.entity(entity).try_remove::<ActiveAgent>();
        }
    }
    if let Some(next) = next
        && !active.contains(next)
    {
        commands.entity(next).insert(ActiveAgent);
    }
}

/// Name -> entity lookup used by the cross-agent tool.
#[derive(Resource, Default)]
pub(crate) struct AgentIndex {
    pub(crate) by_name: HashMap<String, Entity>,
}

pub(crate) struct ActiveTurn {
    turn_id: u64,
    kind: ModelKind,
    decoder: Decoder,
    draft: Vec<DraftBlock>,
    stop_reason: Option<String>,
    /// Cooperative cancellation. Aborting the Tokio task directly is unsafe
    /// because it may be awaiting `run_on_main_thread`, whose callback panics
    /// when the oneshot receiver has been dropped.
    cancel: watch::Sender<bool>,
}

#[derive(Resource, Default)]
pub(crate) struct AgentRuntime {
    next_turn_id: u64,
    active: HashMap<Entity, ActiveTurn>,
}

/// Uniform wrapper over the three provider stream decoders.
pub(crate) enum Decoder {
    Anthropic(anthropic::stream::StreamDecoder),
    Completions(openai_completions::stream::StreamDecoder),
    Responses(openai_responses::stream::StreamDecoder),
}

impl Decoder {
    fn new(provider: Provider, usage_provider: UsageProvider) -> Self {
        match provider {
            Provider::Anthropic => Self::Anthropic(anthropic::stream::StreamDecoder::new()),
            Provider::Completions => Self::Completions(
                openai_completions::stream::StreamDecoder::with_provider(usage_provider),
            ),
            Provider::Responses => Self::Responses(
                openai_responses::stream::StreamDecoder::with_provider(usage_provider),
            ),
        }
    }

    fn push_bytes(
        &mut self,
        bytes: RawStreamBytes,
        blocks: &mut Vec<DraftBlock>,
    ) -> Result<Vec<StreamEvent>, llm_api::stream::StreamError> {
        match self {
            Decoder::Anthropic(decoder) => decoder.push_bytes(bytes, blocks),
            Decoder::Completions(decoder) => decoder.push_bytes(bytes, blocks),
            Decoder::Responses(decoder) => decoder.push_bytes(bytes, blocks),
        }
    }

    fn finish(
        &mut self,
        blocks: &mut Vec<DraftBlock>,
    ) -> Result<Vec<StreamEvent>, llm_api::stream::StreamError> {
        match self {
            Decoder::Anthropic(decoder) => decoder.finish(blocks),
            Decoder::Completions(decoder) => decoder.finish(blocks),
            Decoder::Responses(decoder) => decoder.finish(blocks),
        }
    }

    fn usage(&self) -> Option<&Usage> {
        match self {
            Decoder::Anthropic(decoder) => decoder.usage(),
            Decoder::Completions(decoder) => decoder.usage(),
            Decoder::Responses(decoder) => decoder.usage(),
        }
    }
}

pub(crate) fn create_requested_agents(
    mut requests: MessageReader<NewAgentRequested>,
    mut commands: Commands,
    mut layout: ResMut<AgentLayout>,
    config: Res<AppConfig>,
    active: Query<Entity, With<ActiveAgent>>,
    mut index: ResMut<AgentIndex>,
) {
    let mut created = 0;
    let mut newest = None;
    for _ in requests.read() {
        if index.by_name.len() + created >= MAX_AGENTS {
            continue;
        }
        let number = layout.next_number;
        layout.next_number += 1;
        let name = format!("LLM Agent {number}");
        let agent = create_agent(&mut commands, &layout, &config, name.clone(), number as i32);
        index.by_name.insert(name, agent);
        newest = Some(agent);
        created += 1;
    }
    if let Some(agent) = newest {
        activate_agent(&mut commands, &active, Some(agent));
    }
}

pub(crate) fn create_agent(
    commands: &mut Commands,
    layout: &AgentLayout,
    config: &AppConfig,
    name: String,
    tab_index: i32,
) -> Entity {
    let agent_entity = commands
        .spawn((
            Agent::new(name.clone(), config.provider, config.model_name.clone()),
            Name::new(name.clone()),
        ))
        .id();

    let (tab, tab_label) = create_agent_tab(commands, layout.tabs, agent_entity, &name, tab_index);

    commands
        .entity(agent_entity)
        .insert(AgentView::new(Entity::PLACEHOLDER, tab_label));

    // Select this tab when clicked.
    commands.entity(tab).observe(
        move |_: On<Activate>, active: Query<Entity, With<ActiveAgent>>, mut commands: Commands| {
            activate_agent(&mut commands, &active, Some(agent_entity));
        },
    );

    agent_entity
}

// ---------------------------------------------------------------------------
// Systems: tabs and input
// ---------------------------------------------------------------------------

/// Keep [`AgentIndex`] in sync without a per-frame rebuild.
pub(crate) fn index_agent_on_insert(
    insert: On<Insert<Agent>>,
    agents: Query<&Agent>,
    mut index: ResMut<AgentIndex>,
) {
    if let Ok(agent) = agents.get(insert.entity) {
        index.by_name.insert(agent.name.clone(), insert.entity);
    }
}

/// `Discard` also fires before replacement and despawn, so stale name mappings
/// disappear with the component they belong to.
pub(crate) fn index_agent_on_discard(
    discard: On<Discard<Agent>>,
    agents: Query<&Agent>,
    mut index: ResMut<AgentIndex>,
) {
    if let Ok(agent) = agents.get(discard.entity)
        && index.by_name.get(&agent.name) == Some(&discard.entity)
    {
        index.by_name.remove(&agent.name);
    }
}

/// Pick another visible agent when the active entity is removed.
///
/// `Remove` fires for explicit removal and despawn, but not for component
/// replacement, so replacing [`Agent`] data does not switch tabs.
pub(crate) fn deactivate_removed_agent(
    remove: On<Remove<Agent>>,
    views: Query<(Entity, &AgentView)>,
    active: Query<Entity, With<ActiveAgent>>,
    mut commands: Commands,
    mut focus: ResMut<InputFocus>,
) {
    let active_was_removed = active.contains(remove.entity);
    if !active_was_removed && !active.is_empty() {
        return;
    }

    let next = views
        .iter()
        .map(|(entity, _)| entity)
        .find(|entity| *entity != remove.entity);
    activate_agent(&mut commands, &active, next);
    if next.is_none() {
        focus.clear();
    }
}

/// Attach a [`JsRuntimeHandle`] to every agent.
///
/// The `Insert` observer in `js_runtime` starts the actual OS thread. Keeping
/// this as an observer means `create_agent` does not need to know anything
/// about JavaScript plumbing.
pub(crate) fn attach_js_runtime(
    add: On<Add<Agent>>,
    channels: Res<JsChannels>,
    mut commands: Commands,
) {
    commands
        .entity(add.entity)
        .insert(channels.handle_for(add.entity));
}

pub(crate) fn sync_retry_buttons(
    agents: Query<&Agent>,
    mut buttons: Query<(&RetryAgentButton, &mut BackgroundColor, &mut BorderColor)>,
) {
    for (button, mut background, mut border) in &mut buttons {
        let enabled = agents
            .get(button.agent)
            .is_ok_and(|agent| agent.retry_message.is_some());
        let (background_color, border_color) = if enabled {
            (Color::srgb(0.12, 0.22, 0.15), Color::srgb(0.38, 0.68, 0.43))
        } else {
            (Color::srgb(0.11, 0.12, 0.14), Color::srgb(0.22, 0.25, 0.30))
        };

        if background.0 != background_color {
            background.0 = background_color;
        }
        let border_color = BorderColor::all(border_color);
        if *border != border_color {
            *border = border_color;
        }
    }
}

pub(crate) fn retry_agent_turn(
    mut requests: MessageReader<RetryAgentTurn>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut texts: Query<&mut Text>,
    mut focus: ResMut<InputFocus>,
    mut start_turn: MessageWriter<StartAgentTurn>,
) {
    for request in requests.read() {
        let Ok((mut agent, mut view)) = agents.get_mut(request.agent) else {
            continue;
        };
        focus.set(view.input, FocusCause::Navigated);

        if agent.pending {
            continue;
        }
        let Some(text) = agent.retry_message.take() else {
            continue;
        };

        // The failed/canceled user card is already visible; only re-add the
        // history entry and let start_agent_turn create the next assistant card.
        agent.push_history(
            "retry_user_message",
            Entry::UserMessage {
                content: MessageContent {
                    text: Some(text),
                    image: None,
                },
            },
        );
        agent.pending = true;
        agent.streaming_text.clear();
        agent.streaming_thinking.clear();
        view.follow_tail = true;
        view.scroll_to_bottom = true;
        set_text(&mut texts, view.status, "Retrying...");

        start_turn.write(StartAgentTurn {
            agent: request.agent,
        });
    }
}

pub(crate) fn cancel_agent_turn(
    mut requests: MessageReader<CancelAgentRequest>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut runtime: ResMut<AgentRuntime>,
    mut rounds: ResMut<ToolRounds>,
    mut texts: Query<&mut Text>,
    mut focus: ResMut<InputFocus>,
    mut commands: Commands,
) {
    for request in requests.read() {
        if let Some(active) = runtime.active.remove(&request.agent) {
            let _ = active.cancel.send(true);
        }
        rounds.cancel_rounds(request.agent);

        let Ok((mut agent, mut view)) = agents.get_mut(request.agent) else {
            continue;
        };

        // Clicking the Stop button moves input focus to the button; return it
        // to the text field whether or not there was a turn to cancel.
        focus.set(view.input, FocusCause::Navigated);

        if !agent.pending && agent.current_turn.is_none() {
            continue;
        }

        // Stop only discards work that has not been committed to history yet:
        // the in-flight assistant response and queued messages. Earlier turns
        // (including the user message and completed tool rounds) are kept.
        agent.retry_message = None;
        agent.pending_messages.clear();
        agent.pending = false;
        agent.current_turn = None;
        agent.streaming_text.clear();
        agent.streaming_thinking.clear();

        if let Some(assistant) = view.assistant.take() {
            commands.entity(assistant.card).despawn();
        }
        view.scroll_to_bottom = true;
        set_text(&mut texts, view.status, "Interrupted");
    }
}

pub(crate) fn submit_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    active: Query<Entity, With<ActiveAgent>>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut inputs: Query<&mut EditableText>,
    mut texts: Query<&mut Text>,
) {
    if !keyboard.just_pressed(KeyCode::Enter) {
        return;
    }
    let Some(agent_entity) = active.iter().next() else {
        return;
    };
    let Ok((mut agent, view)) = agents.get_mut(agent_entity) else {
        return;
    };
    let Ok(mut input) = inputs.get_mut(view.input) else {
        return;
    };
    if input.is_composing() {
        return;
    }

    let prompt = input.value().to_string();
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return;
    }

    // A new submission means the user chose not to retry the previous turn.
    agent.retry_message = None;

    // Always queue the input. If the agent is idle, the delivery system picks
    // it up in this same frame; if it is busy, the message waits until the next
    // legal history boundary (after a reply or a completed tool-call round).
    input.clear();
    agent
        .pending_messages
        .push_back(QueuedMessage::User(prompt.to_owned()));

    let status = if agent.pending {
        format!("Queued message ({} waiting)", agent.pending_messages.len())
    } else {
        "Starting request...".to_owned()
    };
    set_text(&mut texts, view.status, status);
}

/// One queued message rendered for both history and the transcript.
pub(crate) struct PendingDelivery {
    text: String,
    status: &'static str,
}

pub(crate) fn take_pending_message(agent: &mut Agent) -> Option<PendingDelivery> {
    match agent.pending_messages.pop_front()? {
        QueuedMessage::User(text) => Some(PendingDelivery {
            text,
            status: "Starting request...",
        }),
        QueuedMessage::Agent { from, text } => Some(PendingDelivery {
            text: format!(
                "<system_message>[Message from {from}, answer with a tool \"send_message_to_agent\" if you need to reply]\n{text}</system_message>"
            ),
            status: "Message received; starting request...",
        }),
        QueuedMessage::Task(text) => Some(PendingDelivery {
            // The task text is already formatted with its creator prefix.
            text,
            status: "Initial task received; starting request...",
        }),
        QueuedMessage::BackgroundResult {
            task_id,
            result,
            is_error,
        } => Some(PendingDelivery {
            text: format!(
                "<system_message>[Background task {task_id} {}]\n{result}</system_message>",
                if is_error { "failed" } else { "completed" }
            ),
            status: if is_error {
                "Background task failed; starting request..."
            } else {
                "Background task finished; starting request..."
            },
        }),
    }
}

/// Moves waiting messages into `agent.history` at the current legal boundary.
///
/// When `all` is set every queued message is flushed (used after tool results,
/// where the agent is about to continue anyway); otherwise only the oldest
/// message is delivered so an idle agent answers one message per turn.
/// Returns how many messages were inserted.
pub(crate) fn flush_pending_messages(
    agent: &mut Agent,
    view: &mut AgentView,
    texts: &mut Query<&mut Text>,
    all: bool,
) -> usize {
    let mut combined = Vec::new();
    let mut status = None;

    while let Some(delivery) = take_pending_message(agent) {
        combined.push(delivery.text);
        status = Some(delivery.status);
        if !all {
            break;
        }
    }

    if combined.is_empty() {
        return 0;
    }

    // Consecutive user turns are rejected by history validation, so one batch
    // of pending messages becomes a single user entry in arrival order.
    agent.push_history(
        "pending_messages",
        Entry::UserMessage {
            content: MessageContent {
                text: Some(combined.join("\n\n")),
                image: None,
            },
        },
    );
    view.follow_tail = true;
    view.scroll_to_bottom = true;
    if let Some(status) = status {
        set_text(texts, view.status, status);
    }
    combined.len()
}

pub(crate) fn deliver_pending_messages(
    mut agents: Query<(Entity, &mut Agent, &mut AgentView)>,
    mut texts: Query<&mut Text>,
    mut start_turn: MessageWriter<StartAgentTurn>,
) {
    for (entity, mut agent, mut view) in &mut agents {
        if agent.pending {
            continue;
        }

        if flush_pending_messages(&mut agent, &mut view, &mut texts, false) == 0 {
            continue;
        }

        agent.pending = true;
        agent.streaming_text.clear();
        agent.streaming_thinking.clear();

        start_turn.write(StartAgentTurn { agent: entity });
    }
}

// ---------------------------------------------------------------------------
// Systems: request startup
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn start_agent_turn(
    mut requests: MessageReader<StartAgentTurn>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut runtime: ResMut<AgentRuntime>,
    mut texts: Query<&mut Text>,
    collapse: Res<TextCollapseConfig>,
    tokio_runtime: Res<TokioTasksRuntime>,
    app_config: Res<AppConfig>,
    tools: AdvertisedTools,
    mut commands: Commands,
) {
    for request in requests.read() {
        if runtime.active.contains_key(&request.agent) {
            continue;
        }
        let Ok((mut agent, mut view)) = agents.get_mut(request.agent) else {
            continue;
        };

        agent.retry_message = None;
        agent.previous_request_body = agent.last_request_body.take();
        let config = agent.model_config(advertised_tools(&tools));
        let body = match request_body(agent.provider, &agent.history, &config) {
            Ok(body) => body,
            Err(err) => {
                error!("could not build request for {}: {err:#}", agent.name);
                agent.retry_message = agent.rollback_pending_turn();
                agent.pending = false;
                log_history_dump(
                    &agent,
                    "request_build_error",
                    serde_json::json!({ "error": format!("{err:#}") }),
                );
                set_text(
                    &mut texts,
                    view.status,
                    format!("Could not build request: {err:#}"),
                );
                continue;
            }
        };
        agent.last_request_body = Some(body.clone());

        let provider = agent.provider;
        let kind = config.kind.clone();
        let turn_id = runtime.next_turn_id;
        runtime.next_turn_id += 1;
        agent.current_turn = Some(turn_id);
        agent.pending = true;
        agent.streaming_text.clear();
        agent.streaming_thinking.clear();

        if view.transcript_visible {
            begin_live_assistant(&mut commands, &mut view, &agent, &collapse);
        } else {
            view.assistant = None;
        }
        view.scroll_to_bottom = true;
        set_text(&mut texts, view.status, "Connecting...");

        let agent_entity = request.agent;
        let endpoint = app_config.endpoint.clone();
        let token = app_config.bearer_token().map(str::to_owned);
        let session_id = app_config.session_id.clone();
        let (cancel_tx, cancel_rx) = watch::channel(false);
        tokio_runtime.spawn_background_task(move |ctx| async move {
            stream_transport(
                ctx,
                cancel_rx,
                agent_entity,
                turn_id,
                endpoint,
                token,
                session_id,
                body,
            )
            .await;
        });

        runtime.active.insert(
            request.agent,
            ActiveTurn {
                turn_id,
                kind: kind.clone(),
                decoder: Decoder::new(provider, kind.usage_provider()),
                draft: Vec::new(),
                stop_reason: None,
                cancel: cancel_tx,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Systems: transport decoding and stream events
// ---------------------------------------------------------------------------

pub(crate) fn decode_stream_chunks(
    mut runtime: ResMut<AgentRuntime>,
    mut chunks: MessageReader<StreamChunk>,
    mut ended: MessageReader<StreamEnded>,
    mut transport_failed: MessageReader<StreamFailed>,
    mut stream_events: MessageWriter<AgentStreamEvent>,
    mut finished: MessageWriter<AgentTurnFinished>,
    mut failed: MessageWriter<AgentFailed>,
) {
    let mut decode_failure = None;

    for chunk in chunks.read() {
        let Some(active) = runtime.active.get_mut(&chunk.agent) else {
            continue;
        };
        if active.turn_id != chunk.turn_id {
            continue;
        }

        match active
            .decoder
            .push_bytes(RawStreamBytes::new(chunk.bytes.clone()), &mut active.draft)
        {
            Ok(events) => write_stream_events(&mut stream_events, chunk.agent, active, events),
            Err(err) => {
                decode_failure = Some((chunk.agent, active.turn_id, err.to_string()));
                break;
            }
        }
    }

    if let Some((agent, turn_id, message)) = decode_failure {
        runtime.active.remove(&agent);
        failed.write(AgentFailed {
            agent,
            turn_id,
            message,
        });
        return;
    }

    for end in ended.read() {
        if !runtime
            .active
            .get(&end.agent)
            .is_some_and(|active| active.turn_id == end.turn_id)
        {
            continue;
        }

        let mut active = runtime
            .active
            .remove(&end.agent)
            .expect("active turn checked above");
        match active.decoder.finish(&mut active.draft) {
            Ok(events) => write_stream_events(&mut stream_events, end.agent, &mut active, events),
            Err(err) => {
                failed.write(AgentFailed {
                    agent: end.agent,
                    turn_id: active.turn_id,
                    message: err.to_string(),
                });
                continue;
            }
        }

        let usage = active.decoder.usage().cloned();
        match finish_draft_for(
            Entry::Draft {
                blocks: active.draft,
            },
            &active.kind,
        ) {
            Ok(entries) => {
                finished.write(AgentTurnFinished {
                    agent: end.agent,
                    turn_id: active.turn_id,
                    entries,
                    usage,
                    stop_reason: active.stop_reason,
                });
            }
            Err(err) => {
                failed.write(AgentFailed {
                    agent: end.agent,
                    turn_id: active.turn_id,
                    message: err.to_string(),
                });
            }
        }
    }

    for fail in transport_failed.read() {
        if !runtime
            .active
            .get(&fail.agent)
            .is_some_and(|active| active.turn_id == fail.turn_id)
        {
            continue;
        }
        runtime.active.remove(&fail.agent);
        failed.write(AgentFailed {
            agent: fail.agent,
            turn_id: fail.turn_id,
            message: fail.message.clone(),
        });
    }
}

pub(crate) fn write_stream_events(
    stream_events: &mut MessageWriter<AgentStreamEvent>,
    agent: Entity,
    active: &mut ActiveTurn,
    events: Vec<StreamEvent>,
) {
    for event in events {
        if let StreamEvent::Completed { stop_reason } = &event {
            active.stop_reason = stop_reason.clone();
        }
        stream_events.write(AgentStreamEvent {
            agent,
            turn_id: active.turn_id,
            event,
        });
    }
}

pub(crate) fn handle_stream_events(
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut stream_events: MessageReader<AgentStreamEvent>,
    mut texts: Query<&mut Text>,
    mut collapsibles: Query<&mut CollapsibleText>,
    mut nodes: Query<&mut Node>,
    collapse: Res<TextCollapseConfig>,
    mut commands: Commands,
) {
    for message in stream_events.read() {
        let Ok((mut agent, mut view)) = agents.get_mut(message.agent) else {
            continue;
        };
        if agent.current_turn != Some(message.turn_id) {
            continue;
        }

        match &message.event {
            StreamEvent::TextDelta(text) => {
                agent.streaming_text.push_str(text);
                ensure_live_assistant(&mut commands, &mut view, &agent, &collapse);
                if let Some(assistant) = view.assistant {
                    set_text(
                        &mut texts,
                        assistant.answer_body,
                        agent.streaming_text.clone(),
                    );
                }
                view.scroll_to_bottom = true;
            }
            StreamEvent::ThinkingDelta(text) => {
                agent.streaming_thinking.push_str(text);
                ensure_live_assistant(&mut commands, &mut view, &agent, &collapse);
                if let Some(assistant) = view.assistant {
                    set_display(&mut nodes, assistant.thinking_container, Display::Flex);
                    set_collapsible_text(
                        &mut texts,
                        &mut collapsibles,
                        assistant.thinking_body,
                        agent.streaming_thinking.clone(),
                    );
                }
                view.scroll_to_bottom = true;
                set_text(
                    &mut texts,
                    view.status,
                    format!(
                        "Thinking... ({} chars)",
                        agent.streaming_thinking.chars().count()
                    ),
                );
            }
            StreamEvent::ToolCallDelta {
                name: Some(name), ..
            } => {
                set_text(
                    &mut texts,
                    view.status,
                    format!("Preparing tool call `{name}`..."),
                );
            }
            StreamEvent::Completed { .. } => {
                set_text(&mut texts, view.status, "Finishing...");
            }
            StreamEvent::Usage(usage) => {
                set_text(&mut texts, view.stats, stats_text(Some(usage)));
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Systems: committed turns and tool calls
// ---------------------------------------------------------------------------

pub(crate) fn handle_turn_finished(
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut finished: MessageReader<AgentTurnFinished>,
    mut texts: Query<&mut Text>,
    mut collapsibles: Query<&mut CollapsibleText>,
    mut nodes: Query<&mut Node>,
    mut run_tools: MessageWriter<RunToolCalls>,
    mut commands: Commands,
) {
    for message in finished.read() {
        let Ok((mut agent, mut view)) = agents.get_mut(message.agent) else {
            continue;
        };
        if agent.current_turn != Some(message.turn_id) {
            continue;
        }

        if let Some(usage) = &message.usage {
            record_usage(&mut agent, usage);
            set_text(&mut texts, view.stats, stats_text(Some(usage)));
        }

        let answer = if !agent.streaming_text.is_empty() {
            agent.streaming_text.clone()
        } else if let Some(text) = response_display_text(&message.entries) {
            text
        } else if !agent.streaming_thinking.is_empty() {
            "".to_owned()
        } else {
            "(empty response)".to_owned()
        };
        let thinking = response_thinking_text(&message.entries).or_else(|| {
            (!agent.streaming_thinking.is_empty()).then(|| agent.streaming_thinking.clone())
        });

        if let Some(assistant) = view.assistant {
            set_text(&mut texts, assistant.answer_body, answer);
            if let Some(thinking) = thinking {
                set_display(&mut nodes, assistant.thinking_container, Display::Flex);
                set_collapsible_text(
                    &mut texts,
                    &mut collapsibles,
                    assistant.thinking_body,
                    thinking,
                );
            } else {
                set_display(&mut nodes, assistant.thinking_container, Display::None);
            }
        }

        let tool_call = match message.entries.last() {
            Some(Entry::ResponseToolCall { tool_call, .. }) => Some(tool_call.clone()),
            _ => None,
        };

        if message.entries.is_empty() {
            agent.retry_message = agent.rollback_pending_turn();
            agent.pending = false;
            agent.current_turn = None;
            if let Some(assistant) = view.assistant.take() {
                commands.entity(assistant.card).despawn();
            }
            set_text(&mut texts, view.status, "Empty response");
        } else {
            agent.extend_history("assistant_response", message.entries.clone());
            agent.streaming_text.clear();
            agent.streaming_thinking.clear();

            if let Some(tool_call) = tool_call {
                agent.pending = true;
                set_text(
                    &mut texts,
                    view.status,
                    format!(
                        "Running tools({})...",
                        tool_call
                            .iter()
                            .map(|x| x.tool_name.clone())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                );
                run_tools.write(RunToolCalls {
                    agent: message.agent,
                    turn_id: message.turn_id,
                    tool_call,
                });
            } else {
                agent.pending = false;
                agent.current_turn = None;
                agent.retry_message = None;
                let mut status = String::from("Ready");
                if let Some(reason) = message
                    .stop_reason
                    .as_deref()
                    .filter(|reason| *reason != "end_turn" && *reason != "stop")
                {
                    status.push_str(" | ");
                    status.push_str(reason);
                }
                set_text(&mut texts, view.status, status);
            }
        }

        view.scroll_to_bottom = true;
    }
}

/// Dispatch a finished assistant turn's tool calls.
///
/// World tools run immediately. JavaScript tools become [`JsTask`]s on the
/// owning context thread, so this system never blocks on a Boa heap. Results
/// are collected by [`poll_js_events`] and committed by [`apply_tool_results`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_tool_calls(
    mut requests: MessageReader<RunToolCalls>,
    mut world: ToolWorld,
    js_handles: Query<&JsRuntimeHandle>,
    script: Res<ScriptRuntime>,
    sources: Res<ToolSourceCache>,
    policy: Res<AutoBackgroundPolicy>,
    mut rounds: ResMut<ToolRounds>,
    mut finished: MessageWriter<ToolRoundFinished>,
) {
    for request in requests.read() {
        let Ok(agent) = world.agents.get(request.agent) else {
            continue;
        };
        if agent.current_turn != Some(request.turn_id) || request.tool_call.is_empty() {
            continue;
        }
        let source_name = agent.name().to_owned();

        let round_id = rounds.next_round;
        rounds.next_round += 1;

        let mut results: Vec<Option<ToolCallResult>> =
            (0..request.tool_call.len()).map(|_| None).collect();
        let mut remaining = request.tool_call.len();

        for (call_index, call) in request.tool_call.iter().enumerate() {
            let raw_arguments = call.toolcall_params.trim();
            let arguments = if raw_arguments.is_empty() {
                Value::Null
            } else {
                match serde_json::from_str::<Value>(raw_arguments) {
                    Ok(value) => value,
                    Err(err) => {
                        results[call_index] = Some(ToolCallResult {
                            result: format!(
                                "invalid JSON arguments for `{}`: {err}",
                                call.tool_name
                            ),
                            is_error: true,
                            tool_use_id: call.tool_use_id.clone(),
                        });
                        remaining -= 1;
                        continue;
                    }
                }
            };
            let root = (round_id << 32) | call_index as u64;
            rounds.open_budget(root, request.agent);
            let context = CallContext {
                source_entity: request.agent,
                source_agent: &source_name,
                depth: 0,
                root,
            };

            let found = find(&world, &call.tool_name);
            if let Some(reason) = found.and_then(|found| found.disabled) {
                results[call_index] = Some(ToolCallResult {
                    result: format!("Tool `{}` is disabled: {reason}", call.tool_name),
                    is_error: true,
                    tool_use_id: call.tool_use_id.clone(),
                });
                remaining -= 1;
                continue;
            }
            match found {
                Some(found) if matches!(found.execution, Execution::Interpreter) => {
                    let auto = resolve_auto_background(
                        background_from_flag(arguments.get("run_in_background")),
                        found.auto_background,
                        policy.auto_after,
                    );
                    if !rounds.spend_budget(root) {
                        results[call_index] = Some(ToolCallResult {
                            result: budget_exhausted_message(),
                            is_error: true,
                            tool_use_id: call.tool_use_id.clone(),
                        });
                        remaining -= 1;
                        continue;
                    }
                    let submitted =
                        interpreter_task(&call.tool_name, &arguments, &world.fs, 1, root).and_then(
                            |task| {
                                js_handles
                                    .get(request.agent)
                                    .map_err(|_| {
                                        format!("agent {} has no JavaScript runtime", request.agent)
                                    })?
                                    .submit(task)
                            },
                        );
                    match submitted {
                        Ok(request_id) if auto == Some(Duration::ZERO) => {
                            let task_id = rounds.detach(request_id, request.agent, root);
                            results[call_index] = Some(ToolCallResult {
                                result: background_started_message(task_id),
                                is_error: false,
                                tool_use_id: call.tool_use_id.clone(),
                            });
                            remaining -= 1;
                        }
                        Ok(request_id) => rounds.track(
                            request_id,
                            PendingCallFields {
                                round_id,
                                call_index,
                                auto_background: auto,
                            },
                        ),
                        Err(err) => {
                            rounds.refund_budget(root);
                            results[call_index] = Some(ToolCallResult {
                                result: err,
                                is_error: true,
                                tool_use_id: call.tool_use_id.clone(),
                            });
                            remaining -= 1;
                        }
                    }
                }
                Some(found) if matches!(found.execution, Execution::Gateway) => {
                    match dispatch_gateway(
                        &mut world,
                        &js_handles,
                        &script,
                        &sources,
                        &policy,
                        &mut rounds,
                        request.agent,
                        &source_name,
                        &arguments,
                        root,
                    ) {
                        GatewayDispatch::Done { message, is_error } => {
                            results[call_index] = Some(ToolCallResult {
                                result: message,
                                is_error,
                                tool_use_id: call.tool_use_id.clone(),
                            });
                            remaining -= 1;
                        }
                        GatewayDispatch::Pending {
                            request: target,
                            auto_background,
                        } => rounds.track(
                            target,
                            PendingCallFields {
                                round_id,
                                call_index,
                                auto_background,
                            },
                        ),
                    }
                }
                Some(found) if matches!(found.execution, Execution::Script) => {
                    if let Err(err) = validate_script_arguments(&world, &call.tool_name, &arguments)
                    {
                        results[call_index] = Some(ToolCallResult {
                            result: err,
                            is_error: true,
                            tool_use_id: call.tool_use_id.clone(),
                        });
                        remaining -= 1;
                        continue;
                    }
                    let auto = resolve_auto_background(
                        AutoBackground::Default,
                        found.auto_background,
                        policy.auto_after,
                    );
                    let spec = world.scripts.get(found.entity).cloned();
                    match spec {
                        Ok(spec) => {
                            if !rounds.spend_budget(root) {
                                results[call_index] = Some(ToolCallResult {
                                    result: budget_exhausted_message(),
                                    is_error: true,
                                    tool_use_id: call.tool_use_id.clone(),
                                });
                                remaining -= 1;
                                continue;
                            }
                            let submitted = script.submit(
                                request.agent,
                                ScriptTask {
                                    sources: sources.0.clone(),
                                    entry: spec.entry,
                                    export: spec.export,
                                    arguments: arguments.clone(),
                                    depth: 1,
                                    root,
                                },
                            );
                            match submitted {
                                Ok(request_id) if auto == Some(Duration::ZERO) => {
                                    let task_id = rounds.detach(request_id, request.agent, root);
                                    results[call_index] = Some(ToolCallResult {
                                        result: background_started_message(task_id),
                                        is_error: false,
                                        tool_use_id: call.tool_use_id.clone(),
                                    });
                                    remaining -= 1;
                                }
                                Ok(request_id) => rounds.track(
                                    request_id,
                                    PendingCallFields {
                                        round_id,
                                        call_index,
                                        auto_background: auto,
                                    },
                                ),
                                Err(err) => {
                                    rounds.refund_budget(root);
                                    results[call_index] = Some(ToolCallResult {
                                        result: err,
                                        is_error: true,
                                        tool_use_id: call.tool_use_id.clone(),
                                    });
                                    remaining -= 1;
                                }
                            }
                        }
                        Err(_) => {
                            results[call_index] = Some(ToolCallResult {
                                result: format!(
                                    "script tool `{}` is missing its execution data",
                                    call.tool_name
                                ),
                                is_error: true,
                                tool_use_id: call.tool_use_id.clone(),
                            });
                            remaining -= 1;
                        }
                    }
                }
                _ => {
                    let (result, is_error) =
                        execute_tool(&mut world, &call.tool_name, &arguments, &context);
                    results[call_index] = Some(ToolCallResult {
                        result,
                        is_error,
                        tool_use_id: call.tool_use_id.clone(),
                    });
                    remaining -= 1;
                }
            }
        }

        rounds.insert_round(
            round_id,
            PendingToolRound {
                agent: request.agent,
                turn_id: request.turn_id,
                calls: request.tool_call.clone(),
                results,
                remaining,
            },
        );
        finish_round_if_ready(&mut rounds, round_id, &mut finished);
    }
}

/// Drain completions from all context threads and finish any round that now
/// has every result.
pub(crate) fn poll_js_events(
    channels: Res<JsChannels>,
    js_handles: Query<&JsRuntimeHandle>,
    mut rounds: ResMut<ToolRounds>,
    mut finished: MessageWriter<ToolRoundFinished>,
    mut task_results: MessageWriter<TaskResult>,
) {
    for event in channels.drain_events() {
        let (agent, request, result) = match event {
            JsEvent::Started { request, at } => {
                rounds.mark_started(request, at);
                continue;
            }
            JsEvent::Done {
                agent,
                request,
                result,
            } => (agent, request, result),
        };

        route_async_result(
            &mut rounds,
            request,
            agent,
            result,
            &js_handles,
            &mut task_results,
            &mut finished,
        );
    }
}

/// Route one completed async request: detached task, blocked nested caller, or
/// a slot in an active round.
fn route_async_result(
    rounds: &mut ToolRounds,
    request: u64,
    agent: Entity,
    result: Result<String, String>,
    js_handles: &Query<&JsRuntimeHandle>,
    task_results: &mut MessageWriter<TaskResult>,
    finished: &mut MessageWriter<ToolRoundFinished>,
) {
    if let Some(task) = rounds.take_detached(request) {
        rounds.release_budget(task.root);
        let (text, is_error) = match result {
            Ok(text) => (text, false),
            Err(err) => (err, true),
        };
        task_results.write(TaskResult {
            agent: task.agent,
            task_id: task.task_id,
            result: text,
            is_error,
        });
        return;
    }

    if let Some(mut nested) = rounds.take_nested(request) {
        if let Some(reply) = nested.take_reply() {
            send_nested_reply(js_handles, reply, result);
        }
        return;
    }

    complete_tool_request(rounds, request, agent, result, finished);
}

/// Drain completions from script workers.
pub(crate) fn poll_script_events(
    runtime: Res<ScriptRuntime>,
    js_handles: Query<&JsRuntimeHandle>,
    mut rounds: ResMut<ToolRounds>,
    mut finished: MessageWriter<ToolRoundFinished>,
    mut task_results: MessageWriter<TaskResult>,
) {
    for event in runtime.drain_events() {
        match event {
            crate::tools::script::ScriptEvent::Started { request, at } => {
                rounds.mark_started(request, at);
            }
            crate::tools::script::ScriptEvent::Done {
                agent,
                request,
                result,
            } => route_async_result(
                &mut rounds,
                request,
                agent,
                result,
                &js_handles,
                &mut task_results,
                &mut finished,
            ),
        }
    }
}

/// Result of calling the model-facing `execute` gateway.
enum GatewayDispatch {
    Done {
        message: String,
        is_error: bool,
    },
    /// An async target was submitted; its result fills the gateway's round slot.
    Pending {
        request: u64,
        auto_background: Option<Duration>,
    },
}

fn gateway_error(message: impl Into<String>) -> GatewayDispatch {
    GatewayDispatch::Done {
        message: message.into(),
        is_error: true,
    }
}

/// Parse `execute(tool_name, args, background_after)`.
fn parse_gateway_arguments(arguments: &Value) -> Result<(String, Value, AutoBackground), String> {
    let Some(target) = arguments.get("tool_name").and_then(Value::as_str) else {
        return Err("`execute` requires a `tool_name` string".to_owned());
    };
    let target_arguments = arguments.get("args").cloned().unwrap_or(Value::Null);
    let background = match arguments.get("background_after") {
        None | Some(Value::Null) => AutoBackground::Default,
        Some(value) => {
            let Some(seconds) = value.as_f64() else {
                return Err("`background_after` must be a number of seconds".to_owned());
            };
            parse_background_after(Some(seconds))?
        }
    };
    Ok((target.to_owned(), target_arguments, background))
}

/// Call another tool by name on behalf of the model.
///
/// This is the model-facing twin of the JavaScript `execute()` global: same
/// argument shape, same background semantics, same depth/budget rules.
#[allow(clippy::too_many_arguments)]
fn dispatch_gateway(
    world: &mut ToolWorld<'_, '_>,
    js_handles: &Query<&JsRuntimeHandle>,
    script: &ScriptRuntime,
    sources: &ToolSourceCache,
    policy: &AutoBackgroundPolicy,
    rounds: &mut ToolRounds,
    agent: Entity,
    source_name: &str,
    arguments: &Value,
    root: u64,
) -> GatewayDispatch {
    let (target_name, target_arguments, background) = match parse_gateway_arguments(arguments) {
        Ok(parsed) => parsed,
        Err(err) => return gateway_error(err),
    };

    let context = CallContext {
        source_entity: agent,
        source_agent: source_name,
        depth: 1,
        root,
    };

    let Some(found) = find(world, &target_name) else {
        return gateway_error(format!("Unknown tool: {target_name}"));
    };
    if let Some(reason) = found.disabled {
        return gateway_error(format!("Tool `{target_name}` is disabled: {reason}"));
    }
    if matches!(found.execution, Execution::Gateway) {
        return gateway_error("`execute` cannot call `execute`");
    }
    if !rounds.spend_budget(root) {
        return gateway_error(budget_exhausted_message());
    }

    let auto = resolve_auto_background(background, found.auto_background, policy.auto_after);
    match found.execution {
        Execution::Inline(_) => {
            if !matches!(background, AutoBackground::Default | AutoBackground::Never) {
                rounds.refund_budget(root);
                return gateway_error(format!("tool `{target_name}` cannot run in the background"));
            }
            let (message, is_error) =
                execute_tool(world, &target_name, &target_arguments, &context);
            GatewayDispatch::Done { message, is_error }
        }
        Execution::Interpreter => {
            let submitted = interpreter_task(&target_name, &target_arguments, &world.fs, 2, root)
                .and_then(|task| {
                    js_handles
                        .get(agent)
                        .map_err(|_| format!("agent {agent} has no JavaScript runtime"))?
                        .submit(task)
                });
            match submitted {
                Err(err) => {
                    rounds.refund_budget(root);
                    gateway_error(err)
                }
                Ok(request) if auto == Some(Duration::ZERO) => {
                    let task_id = rounds.detach(request, agent, root);
                    GatewayDispatch::Done {
                        message: background_started_message(task_id),
                        is_error: false,
                    }
                }
                Ok(request) => GatewayDispatch::Pending {
                    request,
                    auto_background: auto,
                },
            }
        }
        Execution::Script => {
            if let Err(err) = validate_script_arguments(world, &target_name, &target_arguments) {
                rounds.refund_budget(root);
                return gateway_error(err);
            }
            let Ok(spec) = world.scripts.get(found.entity).cloned() else {
                rounds.refund_budget(root);
                return gateway_error(format!(
                    "script tool `{target_name}` is missing its execution data"
                ));
            };
            let submitted = script.submit(
                agent,
                ScriptTask {
                    sources: sources.0.clone(),
                    entry: spec.entry,
                    export: spec.export,
                    arguments: target_arguments,
                    depth: 2,
                    root,
                },
            );
            match submitted {
                Err(err) => {
                    rounds.refund_budget(root);
                    gateway_error(err)
                }
                Ok(request) if auto == Some(Duration::ZERO) => {
                    let task_id = rounds.detach(request, agent, root);
                    GatewayDispatch::Done {
                        message: background_started_message(task_id),
                        is_error: false,
                    }
                }
                Ok(request) => GatewayDispatch::Pending {
                    request,
                    auto_background: auto,
                },
            }
        }
        Execution::Gateway => unreachable!("checked above"),
        Execution::Missing => {
            rounds.refund_budget(root);
            gateway_error(format!("tool `{target_name}` has no registered handler"))
        }
    }
}

/// Which context is asking the main loop to run a tool.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HostOrigin {
    /// The agent's persistent interpreter thread.
    AgentInterpreter,
    /// A script worker with a throwaway context.
    Script,
}

/// Outcome of a host tool call.
enum HostOutcome {
    /// The reply was already sent.
    Immediate,
    /// Registered as a nested/detached call; the reply is sent on completion.
    Pending,
}

/// Send an immediate host reply to whichever caller is waiting.
fn reply_now(
    js_handles: &Query<&JsRuntimeHandle>,
    reply: NestedReply,
    result: Result<String, String>,
) -> HostOutcome {
    send_nested_reply(js_handles, reply, result);
    HostOutcome::Immediate
}

/// Shared handling for `execute(...)` from JavaScript and script workers.
#[allow(clippy::too_many_arguments)]
fn handle_host_tool(
    world: &mut ToolWorld<'_, '_>,
    rounds: &mut ToolRounds,
    js_handles: &Query<&JsRuntimeHandle>,
    script: &ScriptRuntime,
    sources: &ToolSourceCache,
    policy: &AutoBackgroundPolicy,
    agent: Entity,
    name: &str,
    arguments: &Value,
    depth: u32,
    root: u64,
    background: Option<f64>,
    origin: HostOrigin,
    reply: NestedReply,
) -> HostOutcome {
    if depth > MAX_TOOL_DEPTH {
        return reply_now(
            js_handles,
            reply,
            Err(format!(
                "tool call nesting limit ({MAX_TOOL_DEPTH}) reached"
            )),
        );
    }
    let request = match parse_background_after(background) {
        Ok(request) => request,
        Err(err) => return reply_now(js_handles, reply, Err(err)),
    };
    if !matches!(request, AutoBackground::Default | AutoBackground::Never) {
        return reply_now(
            js_handles,
            reply,
            Err("only a model-level tool call can detach into the background".to_owned()),
        );
    }
    let source_name = world
        .agents
        .get(agent)
        .map(|agent| agent.name().to_owned())
        .unwrap_or_else(|_| "agent".to_owned());
    let context = CallContext {
        source_entity: agent,
        source_agent: &source_name,
        depth,
        root,
    };

    let Some(found) = find(world, name) else {
        return reply_now(js_handles, reply, Err(format!("Unknown tool: {name}")));
    };
    if let Some(disabled) = found.disabled {
        return reply_now(
            js_handles,
            reply,
            Err(format!("Tool `{name}` is disabled: {disabled}")),
        );
    }
    if found.model_only {
        return reply_now(
            js_handles,
            reply,
            Err(format!(
                "tool `{name}` can only be called by the model, not from JavaScript or scripts"
            )),
        );
    }

    // Nested calls always run synchronously from the caller's point of view.
    let _ = policy;
    match found.execution {
        Execution::Inline(_) => {
            if !matches!(request, AutoBackground::Default | AutoBackground::Never) {
                return reply_now(
                    js_handles,
                    reply,
                    Err(format!("tool `{name}` cannot run in the background")),
                );
            }
            if !rounds.spend_budget(root) {
                return reply_now(
                    js_handles,
                    reply,
                    Err(format!(
                        "tool call budget for this root call is exhausted ({})",
                        crate::tools::schedule::MAX_CALLS_PER_ROOT
                    )),
                );
            }
            let (text, is_error) = execute_tool(world, name, arguments, &context);
            let result = if is_error { Err(text) } else { Ok(text) };
            reply_now(js_handles, reply, result)
        }
        Execution::Interpreter => {
            // The agent has exactly one interpreter thread. Submitting to it
            // while that thread is blocked in `execute()` would never run.
            if origin == HostOrigin::AgentInterpreter {
                return reply_now(
                    js_handles,
                    reply,
                    Err(format!(
                        "JavaScript cannot call another JavaScript tool (`{name}`)"
                    )),
                );
            }
            if !rounds.spend_budget(root) {
                return reply_now(js_handles, reply, Err(budget_exhausted_message()));
            }
            let submitted =
                interpreter_task(name, arguments, &world.fs, depth + 1, root).and_then(|task| {
                    js_handles
                        .get(agent)
                        .map_err(|_| format!("agent {agent} has no JavaScript runtime"))?
                        .submit(task)
                });
            match submitted {
                Err(err) => {
                    rounds.refund_budget(root);
                    reply_now(js_handles, reply, Err(err))
                }
                Ok(request_id) => {
                    rounds.add_nested(NestedCall::new(agent, request_id, reply));
                    HostOutcome::Pending
                }
            }
        }
        Execution::Gateway => reply_now(
            js_handles,
            reply,
            Err(format!("tool `{name}` can only be called by the model")),
        ),
        Execution::Missing => reply_now(
            js_handles,
            reply,
            Err(format!("tool `{name}` has no registered handler")),
        ),
        Execution::Script => {
            if let Err(err) = validate_script_arguments(world, name, arguments) {
                return reply_now(js_handles, reply, Err(err));
            }
            let Ok(spec) = world.scripts.get(found.entity).cloned() else {
                return reply_now(
                    js_handles,
                    reply,
                    Err(format!(
                        "script tool `{name}` is missing its execution data"
                    )),
                );
            };
            if !rounds.spend_budget(root) {
                return reply_now(js_handles, reply, Err(budget_exhausted_message()));
            }
            let submitted = script.submit(
                agent,
                ScriptTask {
                    sources: sources.0.clone(),
                    entry: spec.entry,
                    export: spec.export,
                    arguments: arguments.clone(),
                    depth: depth + 1,
                    root,
                },
            );
            match submitted {
                Err(err) => {
                    rounds.refund_budget(root);
                    reply_now(js_handles, reply, Err(err))
                }
                Ok(request_id) => {
                    rounds.add_nested(NestedCall::new(agent, request_id, reply));
                    HostOutcome::Pending
                }
            }
        }
    }
}

/// Handle `execute(...)` and module reads coming from script workers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn poll_script_host_requests(
    runtime: Res<ScriptRuntime>,
    mut world: ToolWorld,
    js_handles: Query<&JsRuntimeHandle>,
    sources: Res<ToolSourceCache>,
    policy: Res<AutoBackgroundPolicy>,
    mut rounds: ResMut<ToolRounds>,
) {
    for request in runtime.drain_host_requests() {
        let ScriptHostRequest {
            agent,
            command,
            reply,
        } = request;

        // Work from a removed agent must not keep mutating the world.
        if world.agents.get(agent).is_err() {
            let _ = reply.send(Err(format!("agent entity {agent} no longer exists")));
            continue;
        }

        match command {
            HostCommand::Tool {
                name,
                arguments,
                depth,
                root,
                background,
            } => {
                let _ = handle_host_tool(
                    &mut world,
                    &mut rounds,
                    &js_handles,
                    &runtime,
                    &sources,
                    &policy,
                    agent,
                    &name,
                    &arguments,
                    depth,
                    root,
                    background,
                    HostOrigin::Script,
                    NestedReply::Channel(reply),
                );
            }
            HostCommand::ReadFile { path } => {
                let _ = reply.send(world.fs.read(&path));
            }
        }
    }
}

/// Execute host commands coming from context threads: JavaScript `execute`
/// tool calls and virtual-filesystem reads for module imports.
#[allow(clippy::too_many_arguments)]
pub(crate) fn poll_js_host_requests(
    channels: Res<JsChannels>,
    mut world: ToolWorld,
    js_handles: Query<&JsRuntimeHandle>,
    script: Res<ScriptRuntime>,
    sources: Res<ToolSourceCache>,
    policy: Res<AutoBackgroundPolicy>,
    mut rounds: ResMut<ToolRounds>,
) {
    for request in channels.drain_host_requests() {
        let HostRequest {
            agent,
            token,
            command,
        } = request;

        if world.agents.get(agent).is_err() {
            if let Ok(handle) = js_handles.get(agent) {
                handle.respond(token, Err(format!("agent entity {agent} no longer exists")));
            }
            continue;
        }

        match command {
            HostCommand::Tool {
                name,
                arguments,
                depth,
                root,
                background,
            } => {
                // The reply is delivered now or when the nested call completes;
                // the agent's interpreter stays blocked on `token`.
                let _ = handle_host_tool(
                    &mut world,
                    &mut rounds,
                    &js_handles,
                    &script,
                    &sources,
                    &policy,
                    agent,
                    &name,
                    &arguments,
                    depth,
                    root,
                    background,
                    HostOrigin::AgentInterpreter,
                    NestedReply::Agent { agent, token },
                );
            }
            HostCommand::ReadFile { path } => {
                if let Ok(handle) = js_handles.get(agent) {
                    handle.respond(token, world.fs.read(&path));
                }
            }
        }
    }
}

/// Commit a fully-resolved tool round to history and continue the agent loop.
pub(crate) fn apply_tool_results(
    mut finished: MessageReader<ToolRoundFinished>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut texts: Query<&mut Text>,
    mut start_turn: MessageWriter<StartAgentTurn>,
) {
    for message in finished.read() {
        let Ok((mut agent, mut view)) = agents.get_mut(message.agent) else {
            continue;
        };
        if agent.current_turn != Some(message.turn_id) {
            continue;
        }

        // `end_agent` queues a removal; do not commit a new turn for an agent
        // that is about to be despawned.
        if message
            .calls
            .iter()
            .any(|call| call.tool_name == "end_agent")
        {
            continue;
        }

        agent.push_history(
            "tool_results",
            Entry::ToolCallResult {
                tool_call: message.results.clone(),
            },
        );

        view.scroll_to_bottom = true;
        set_text(&mut texts, view.status, "Tools finished; continuing...");

        // Force waiting user and cross-agent messages into the history now:
        // a completed tool round is a legal boundary, and leaving them queued
        // would delay them until the whole agent loop finishes.
        let inserted = flush_pending_messages(&mut agent, &mut view, &mut texts, true);
        if inserted > 0 {
            set_text(
                &mut texts,
                view.status,
                format!("Inserted {inserted} pending message(s); continuing..."),
            );
        }

        agent.current_turn = None;
        start_turn.write(StartAgentTurn {
            agent: message.agent,
        });
    }
}

pub(crate) fn rename_agents(
    mut requests: MessageReader<RenameAgentRequest>,
    mut agents: Query<(&mut Agent, &AgentView)>,
    mut index: ResMut<AgentIndex>,
    mut texts: Query<&mut Text>,
    mut commands: Commands,
) {
    for request in requests.read() {
        let Ok((mut agent, view)) = agents.get_mut(request.agent) else {
            continue;
        };
        if index
            .by_name
            .get(&request.name)
            .is_some_and(|existing| *existing != request.agent)
        {
            continue;
        }

        let old_name = std::mem::replace(&mut agent.name, request.name.clone());
        if index.by_name.get(&old_name) == Some(&request.agent) {
            index.by_name.remove(&old_name);
        }
        index.by_name.insert(request.name.clone(), request.agent);

        commands
            .entity(request.agent)
            .insert(Name::new(request.name.clone()));
        set_text(&mut texts, view.tab_label, request.name.clone());
        set_text(
            &mut texts,
            view.status,
            format!("Renamed to {}", request.name),
        );
    }
}

pub(crate) fn handle_send_agent_message(
    mut messages: MessageReader<SendAgentMessage>,
    mut agents: Query<&mut Agent>,
) {
    for message in messages.read() {
        if let Ok(mut agent) = agents.get_mut(message.target) {
            agent.pending_messages.push_back(if message.verbatim {
                QueuedMessage::Task(message.message.clone())
            } else {
                QueuedMessage::Agent {
                    from: message.from.clone(),
                    text: message.message.clone(),
                }
            });
        }
    }
}

pub(crate) fn handle_task_result(
    mut results: MessageReader<TaskResult>,
    mut agents: Query<&mut Agent>,
) {
    for result in results.read() {
        if let Ok(mut agent) = agents.get_mut(result.agent) {
            agent
                .pending_messages
                .push_back(QueuedMessage::BackgroundResult {
                    task_id: result.task_id,
                    result: result.result.clone(),
                    is_error: result.is_error,
                });
        }
    }
}

pub(crate) fn handle_agent_failed(
    mut failures: MessageReader<AgentFailed>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut rounds: ResMut<ToolRounds>,
    mut texts: Query<&mut Text>,
) {
    for failure in failures.read() {
        rounds.cancel_rounds(failure.agent);
        let Ok((mut agent, mut view)) = agents.get_mut(failure.agent) else {
            continue;
        };
        if agent.current_turn != Some(failure.turn_id) {
            continue;
        }

        error!(
            "agent request for {} failed: {}",
            agent.name, failure.message
        );
        if let Some(assistant) = view.assistant {
            set_text(
                &mut texts,
                assistant.answer_body,
                format!("Request failed: {}", failure.message),
            );
        }
        let reason = if failure.message.starts_with("HTTP ") {
            "endpoint_error"
        } else {
            "agent_error"
        };
        agent.retry_message = agent.rollback_pending_turn();
        log_history_dump(
            &agent,
            reason,
            serde_json::json!({ "error": &failure.message }),
        );
        agent.pending = false;
        agent.current_turn = None;
        agent.streaming_text.clear();
        agent.streaming_thinking.clear();
        view.scroll_to_bottom = true;
        set_text(&mut texts, view.status, "Request failed");
    }
}

pub(crate) fn remove_agents(
    mut requests: MessageReader<RemoveAgentRequest>,
    mut commands: Commands,
    tabs: Query<(Entity, &AgentTab)>,
    mut runtime: ResMut<AgentRuntime>,
    mut rounds: ResMut<ToolRounds>,
) {
    let mut seen = HashSet::new();
    for request in requests.read() {
        let agent = request.agent;
        if !seen.insert(agent) {
            continue;
        }

        // Stop any stream task and drop asynchronous tool work first.
        if let Some(active_turn) = runtime.active.remove(&agent) {
            let _ = active_turn.cancel.send(true);
        }
        rounds.remove_agent(agent);

        // The panel is despawned by the ActiveAgent remove observer (if this
        // agent owned it); tabs are separate roots and are removed here.
        for (tab, tab_agent) in &tabs {
            if tab_agent.agent == agent {
                commands.entity(tab).despawn();
            }
        }

        // Despawning the agent drops its JsRuntimeHandle, which triggers the
        // Discard observer and shuts down that agent's interpreter thread.
        commands.entity(agent).despawn();
    }
}

/// Remove the current user turn and anything generated after it. Returns the
/// removed user text so it can be offered through Retry.
pub(crate) fn rollback_pending_turn(history: &mut Vec<Entry>) -> Option<String> {
    let mut retry = None;
    while let Some(last) = history.last() {
        match last {
            Entry::ToolCallResult { .. }
            | Entry::ResponseToolCall { .. }
            | Entry::Response { .. } => {
                history.pop();
            }
            Entry::UserMessage { content } => {
                retry = content.text.clone();
                history.pop();
                break;
            }
            Entry::SystemMessage { .. } => {
                history.pop();
            }
            _ => break,
        }
    }
    retry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_readout_keeps_fixed_width() {
        let usage = Usage {
            input_tokens: Some(534),
            output_tokens: Some(3),
            cached_input_tokens: Some(512),
            ..Default::default()
        };
        assert_eq!(
            stats_text(Some(&usage)),
            "in    534   out      3   cached    512"
        );
        assert_eq!(stats_text(None), "in      -   out      -   cached      -");
    }

    #[test]
    fn initial_task_is_delivered_verbatim() {
        let mut agent = Agent::new(
            "Worker".to_owned(),
            Provider::Responses,
            "test-model".to_owned(),
        );
        agent.pending_messages.push_back(QueuedMessage::Task(
            "You were created by Boss, your task:\nwork".to_owned(),
        ));

        let delivery = take_pending_message(&mut agent).expect("task should be queued");
        assert_eq!(delivery.text, "You were created by Boss, your task:\nwork");
    }

    #[test]
    fn rollback_returns_user_text_for_retry() {
        let mut history = vec![
            Entry::SystemPrompt {
                content: "system".into(),
            },
            Entry::UserMessage {
                content: MessageContent {
                    text: Some("hello".into()),
                    image: None,
                },
            },
            Entry::response_text("partial answer"),
        ];

        let retry = rollback_pending_turn(&mut history);
        assert_eq!(retry.as_deref(), Some("hello"));
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn cache_drops_are_classified_before_dumping() {
        // One 128-token block of rounding noise is not a real regression.
        assert_eq!(classify_cache_drop(3584, 3456, false), None);
        assert_eq!(classify_cache_drop(5120, 4864, false), Some("unexplained"));
        // A complete miss is always worth recording.
        assert_eq!(classify_cache_drop(2688, 0, false), Some("full_cache_miss"));
        // Rollback explains a drop, but is still recorded.
        assert_eq!(classify_cache_drop(500, 0, true), Some("turn_rollback"));
    }

    #[test]
    fn request_prefix_detects_body_rewrites() {
        let first = r#"{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        let appended = r#"{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}"#;
        assert_eq!(
            request_prefix_append_only(Some(first), Some(appended)),
            Some(true)
        );

        // A changed top-level field invalidates the whole prefix.
        let model_changed = r#"{"model":"m2","stream":true,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}"#;
        assert_eq!(
            request_prefix_append_only(Some(first), Some(model_changed)),
            Some(false)
        );

        // A rewritten committed message is not append-only either.
        let rewritten = r#"{"model":"m","stream":true,"messages":[{"role":"user","content":"bye"},{"role":"assistant","content":"yo"}]}"#;
        assert_eq!(
            request_prefix_append_only(Some(first), Some(rewritten)),
            Some(false)
        );

        assert_eq!(request_prefix_append_only(None, Some(appended)), None);
    }

    #[test]
    fn append_only_detection_finds_rewritten_history() {
        let first = vec![Entry::SystemPrompt {
            content: "sys".into(),
        }];
        let mut appended = first.clone();
        appended.push(Entry::response_text("answer"));

        assert!(history_is_append_only(&first, &appended));
        assert!(!history_is_append_only(&appended, &first));

        let mut rewritten = appended.clone();
        rewritten[0] = Entry::SystemPrompt {
            content: "different".into(),
        };
        assert!(!history_is_append_only(&appended, &rewritten));
    }

    #[test]
    fn history_changes_preserve_the_previous_step() {
        let mut agent = Agent::new(
            "Worker".to_owned(),
            Provider::Completions,
            "test-model".to_owned(),
        );
        agent.push_history(
            "user_message",
            Entry::UserMessage {
                content: MessageContent {
                    text: Some("hello".into()),
                    image: None,
                },
            },
        );
        let before_assistant = agent.history.clone();

        agent.push_history("assistant_response", Entry::response_text("partial"));
        assert_eq!(
            serde_json::to_value(&agent.previous_history).unwrap(),
            serde_json::to_value(&before_assistant).unwrap()
        );

        let retry = agent.rollback_pending_turn();
        assert_eq!(retry.as_deref(), Some("hello"));
        assert!(agent.rolled_back_since_last_usage);
        assert_eq!(agent.last_history_change, "turn_rollback");
        // `previous_history` is the pre-rollback state, so an error dump can
        // show exactly what was removed.
        assert!(matches!(
            agent.previous_history.last(),
            Some(Entry::Response { .. })
        ));
        assert!(matches!(
            agent.history.last(),
            Some(Entry::SystemPrompt { .. })
        ));
    }
}

#[cfg(test)]
mod removal_tests {
    use super::*;

    fn spawn_test_agent(app: &mut App, name: &str) -> (Entity, Entity, Entity) {
        let panel = app.world_mut().spawn_empty().id();
        let tab = app.world_mut().spawn_empty().id();
        let agent = app
            .world_mut()
            .spawn((
                Agent::new(name.to_owned(), Provider::Responses, "model".to_owned()),
                AgentView {
                    panel,
                    ..AgentView::new(Entity::PLACEHOLDER, Entity::PLACEHOLDER)
                },
            ))
            .id();
        app.world_mut().entity_mut(tab).insert(AgentTab { agent });
        app.world_mut()
            .entity_mut(panel)
            .insert(AgentPanel { agent });
        (agent, panel, tab)
    }

    #[test]
    fn removing_an_agent_clears_ui_tabs_index_and_active_selection() {
        let mut app = App::new();
        app.set_error_handler(bevy::ecs::error::panic);
        app.init_resource::<AgentRuntime>();
        app.init_resource::<ToolRounds>();
        app.init_resource::<AgentIndex>();
        app.init_resource::<InputFocus>();
        app.add_message::<RemoveAgentRequest>();
        app.add_observer(index_agent_on_insert);
        app.add_observer(index_agent_on_discard);
        app.add_observer(deactivate_removed_agent);
        app.add_observer(close_active_agent_menu);
        app.add_systems(Update, remove_agents);

        let (first, first_panel, first_tab) = spawn_test_agent(&mut app, "First");
        let (second, _, _) = spawn_test_agent(&mut app, "Second");
        app.world_mut().entity_mut(first).insert(ActiveAgent);

        app.world_mut()
            .write_message(RemoveAgentRequest { agent: first });
        app.update();

        assert!(app.world().get_entity(first).is_err());
        assert!(app.world().get_entity(first_panel).is_err());
        assert!(app.world().get_entity(first_tab).is_err());

        let index = app.world().resource::<AgentIndex>();
        assert!(!index.by_name.contains_key("First"));
        assert!(index.by_name.contains_key("Second"));

        assert!(app.world().get::<ActiveAgent>(second).is_some());
        assert!(
            !app.world()
                .resource::<AgentRuntime>()
                .active
                .contains_key(&first)
        );
    }
}

#[cfg(test)]
mod background_tests {
    use super::*;
    use crate::js_runtime::{JsTask, start_js_runtime};
    use crate::tools::schedule::{PendingCallFields, PendingToolRound};
    use llm_history_model::history::ToolCall;
    use std::time::{Duration, Instant};

    #[test]
    fn removing_agent_drops_background_tasks() {
        let mut rounds = ToolRounds::default();
        let agent = Entity::PLACEHOLDER;
        rounds.detach(7, agent, 7);
        rounds.remove_agent(agent);
        assert!(rounds.take_detached(7).is_none());
    }

    #[test]
    fn long_running_js_is_auto_backgrounded() {
        let mut rounds = ToolRounds::default();
        let agent = Entity::PLACEHOLDER;
        let round_id = 10;
        let request = 20;

        rounds.insert_round(
            round_id,
            PendingToolRound {
                agent,
                turn_id: 1,
                calls: vec![ToolCall {
                    tool_name: "eval_javascript".to_owned(),
                    toolcall_params: "{}".to_owned(),
                    tool_use_id: "tool-1".to_owned(),
                }],
                results: vec![None],
                remaining: 1,
            },
        );
        rounds.open_budget(round_id << 32, agent);
        rounds.track(
            request,
            PendingCallFields {
                round_id,
                call_index: 0,
                auto_background: Some(Duration::from_secs(5)),
            },
        );
        rounds.mark_started(request, Instant::now() - Duration::from_secs(10));

        let mut app = App::new();
        app.insert_resource(rounds);
        app.add_message::<ToolRoundFinished>();
        app.add_systems(Update, crate::tools::schedule::auto_background_long_calls);
        app.update();

        let rounds = app.world().resource::<ToolRounds>();
        assert_eq!(rounds.pending_len(), 0);
        assert!(rounds.has_detached(request));
        assert!(!rounds.has_active(round_id));
    }

    #[test]
    fn explicit_false_prevents_auto_backgrounding() {
        let mut rounds = ToolRounds::default();
        let agent = Entity::PLACEHOLDER;
        let round_id = 10;
        let request = 20;

        rounds.insert_round(
            round_id,
            PendingToolRound {
                agent,
                turn_id: 1,
                calls: vec![ToolCall {
                    tool_name: "eval_javascript".to_owned(),
                    toolcall_params: "{}".to_owned(),
                    tool_use_id: "tool-1".to_owned(),
                }],
                results: vec![None],
                remaining: 1,
            },
        );
        rounds.open_budget(round_id << 32, agent);
        rounds.track(
            request,
            PendingCallFields {
                round_id,
                call_index: 0,
                auto_background: None,
            },
        );
        rounds.mark_started(request, Instant::now() - Duration::from_secs(10));

        let mut app = App::new();
        app.insert_resource(rounds);
        app.add_message::<ToolRoundFinished>();
        app.add_systems(Update, crate::tools::schedule::auto_background_long_calls);
        app.update();

        let rounds = app.world().resource::<ToolRounds>();
        assert!(rounds.has_pending(request));
        assert!(!rounds.has_detached(request));
    }

    #[test]
    fn gateway_arguments_follow_execute_semantics() {
        let (name, args, background) =
            parse_gateway_arguments(&serde_json::json!({ "tool_name": "fs_ls" })).unwrap();
        assert_eq!(name, "fs_ls");
        assert_eq!(args, Value::Null);
        assert_eq!(background, AutoBackground::Default);

        let (_, args, _) = parse_gateway_arguments(&serde_json::json!({
            "tool_name": "demo",
            "args": { "message": "hi" }
        }))
        .unwrap();
        assert_eq!(args["message"], "hi");

        for (value, expected) in [
            (-1.0, AutoBackground::Never),
            (0.0, AutoBackground::Immediately),
        ] {
            let (_, _, background) = parse_gateway_arguments(&serde_json::json!({
                "tool_name": "x",
                "background_after": value
            }))
            .unwrap();
            assert_eq!(background, expected);
        }

        assert!(parse_gateway_arguments(&serde_json::json!({})).is_err());
        assert!(
            parse_gateway_arguments(&serde_json::json!({
                "tool_name": "x",
                "background_after": "soon"
            }))
            .is_err()
        );
    }

    #[test]
    fn background_error_is_marked_failed() {
        let mut agent = Agent::new("Worker".to_owned(), Provider::Responses, "model".to_owned());
        agent
            .pending_messages
            .push_back(QueuedMessage::BackgroundResult {
                task_id: 3,
                result: "ReferenceError: boom".to_owned(),
                is_error: true,
            });

        let delivery = take_pending_message(&mut agent).expect("queued background error");
        assert!(delivery.text.contains("[Background task 3 failed]"));
        assert!(delivery.text.contains("ReferenceError: boom"));
    }

    #[test]
    fn background_result_is_queued_as_a_message() {
        let mut app = App::new();
        app.init_resource::<JsChannels>();
        app.init_resource::<ToolRounds>();
        app.add_message::<TaskResult>();
        app.add_message::<ToolRoundFinished>();
        app.add_observer(attach_js_runtime);
        app.add_observer(start_js_runtime);
        app.add_systems(Update, (poll_js_events, handle_task_result).chain());

        let agent = app
            .world_mut()
            .spawn(Agent::new(
                "Worker".to_owned(),
                Provider::Responses,
                "model".to_owned(),
            ))
            .id();

        let request = app
            .world()
            .get::<JsRuntimeHandle>(agent)
            .expect("handle attached by observer")
            .submit(JsTask::Eval {
                source: "6 * 7".to_owned(),
                name: None,
                depth: 0,
                root: 0,
            })
            .unwrap();
        app.world_mut()
            .resource_mut::<ToolRounds>()
            .detach(request, agent, request);

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            app.update();
            let received = app
                .world()
                .get::<Agent>(agent)
                .is_some_and(|agent| !agent.pending_messages.is_empty());
            if received {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "background result did not arrive"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        let mut agent = app.world_mut().get_mut::<Agent>(agent).unwrap();
        let delivery = take_pending_message(&mut agent).expect("queued background result");
        assert!(delivery.text.contains("[Background task 1 completed]"));
        assert!(delivery.text.contains("42"));
    }
}

#[cfg(test)]
mod cancel_tests {
    use super::*;

    #[test]
    fn stop_keeps_committed_history_and_drops_only_the_stream() {
        let mut app = App::new();
        app.init_resource::<AgentRuntime>();
        app.init_resource::<ToolRounds>();
        app.init_resource::<InputFocus>();
        app.add_message::<CancelAgentRequest>();
        app.add_systems(Update, cancel_agent_turn);

        let status = app.world_mut().spawn(Text::new("Streaming")).id();
        let live_card = app
            .world_mut()
            .spawn((Node::default(), TranscriptCard))
            .id();
        let agent = app
            .world_mut()
            .spawn((
                Agent::new("Worker".to_owned(), Provider::Responses, "model".to_owned()),
                AgentView {
                    status,
                    assistant: Some(AssistantView {
                        card: live_card,
                        thinking_container: Entity::PLACEHOLDER,
                        thinking_body: Entity::PLACEHOLDER,
                        answer_body: Entity::PLACEHOLDER,
                    }),
                    transcript_visible: true,
                    rendered_entries: 3,
                    ..AgentView::new(Entity::PLACEHOLDER, Entity::PLACEHOLDER)
                },
            ))
            .id();

        {
            let mut agent_ref = app.world_mut().get_mut::<Agent>(agent).unwrap();
            agent_ref.history.push(Entry::UserMessage {
                content: MessageContent {
                    text: Some("hello".to_owned()),
                    image: None,
                },
            });
            agent_ref
                .history
                .push(Entry::response_text("committed answer"));
            agent_ref.pending = true;
            agent_ref.current_turn = Some(7);
            agent_ref.streaming_text = "partial stream".to_owned();
            agent_ref.streaming_thinking = "partial thinking".to_owned();
            agent_ref
                .pending_messages
                .push_back(QueuedMessage::User("queued".to_owned()));
            agent_ref.retry_message = Some("hello".to_owned());
        }

        let history_len_before = app.world().get::<Agent>(agent).unwrap().history.len();
        app.world_mut().write_message(CancelAgentRequest { agent });
        app.update();

        let agent_ref = app.world().get::<Agent>(agent).unwrap();
        assert_eq!(agent_ref.history.len(), history_len_before);
        assert!(matches!(
            agent_ref.history.last(),
            Some(Entry::Response { .. })
        ));
        assert!(!agent_ref.pending);
        assert!(agent_ref.current_turn.is_none());
        assert!(agent_ref.streaming_text.is_empty());
        assert!(agent_ref.streaming_thinking.is_empty());
        assert!(agent_ref.pending_messages.is_empty());
        assert!(agent_ref.retry_message.is_none());

        assert!(app.world().get_entity(live_card).is_err());
        let view = app.world().get::<AgentView>(agent).unwrap();
        assert!(view.assistant.is_none());
        assert_eq!(app.world().get::<Text>(status).unwrap().0, "Interrupted");
    }
}

#[cfg(test)]
mod rename_tests {
    use super::*;

    #[test]
    fn rename_updates_agent_index_tab_and_status() {
        let mut app = App::new();
        app.init_resource::<AgentIndex>();
        app.add_message::<RenameAgentRequest>();
        app.add_systems(Update, rename_agents);

        let tab_label = app.world_mut().spawn(Text::new("Old")).id();
        let status = app.world_mut().spawn(Text::new("Ready")).id();
        let agent = app
            .world_mut()
            .spawn((
                Agent::new("Old".to_owned(), Provider::Responses, "model".to_owned()),
                AgentView {
                    status,
                    tab_label,
                    ..AgentView::new(Entity::PLACEHOLDER, Entity::PLACEHOLDER)
                },
            ))
            .id();
        app.world_mut()
            .resource_mut::<AgentIndex>()
            .by_name
            .insert("Old".to_owned(), agent);

        app.world_mut().write_message(RenameAgentRequest {
            agent,
            name: "New".to_owned(),
        });
        app.update();

        assert_eq!(app.world().get::<Agent>(agent).unwrap().name, "New");
        let index = app.world().resource::<AgentIndex>();
        assert_eq!(index.by_name.get("New"), Some(&agent));
        assert!(!index.by_name.contains_key("Old"));

        assert_eq!(app.world().get::<Text>(tab_label).unwrap().0, "New");
        assert_eq!(app.world().get::<Text>(status).unwrap().0, "Renamed to New");
    }
}

#[cfg(test)]
mod empty_response_tests {
    use super::*;

    #[test]
    fn empty_turn_removes_live_card_and_clears_turn() {
        let mut app = App::new();

        let log = app.world_mut().spawn((Node::default(), TranscriptLog)).id();
        let live_card = app
            .world_mut()
            .spawn((Node::default(), TranscriptCard))
            .id();
        app.world_mut().entity_mut(log).add_children(&[live_card]);
        let status = app.world_mut().spawn(Text::new("Ready")).id();

        let agent = app
            .world_mut()
            .spawn(Agent::new(
                "Worker".to_owned(),
                Provider::Responses,
                "model".to_owned(),
            ))
            .id();
        {
            let mut agent_ref = app.world_mut().get_mut::<Agent>(agent).unwrap();
            agent_ref.current_turn = Some(1);
            agent_ref.pending = true;
        }
        app.world_mut().entity_mut(agent).insert(AgentView {
            log,
            status,
            assistant: Some(AssistantView {
                card: live_card,
                thinking_container: Entity::PLACEHOLDER,
                thinking_body: Entity::PLACEHOLDER,
                answer_body: Entity::PLACEHOLDER,
            }),
            transcript_visible: true,
            rendered_entries: 1,
            ..AgentView::new(Entity::PLACEHOLDER, Entity::PLACEHOLDER)
        });

        app.add_message::<AgentTurnFinished>();
        app.add_message::<RunToolCalls>();
        app.add_systems(Update, handle_turn_finished);
        app.world_mut().write_message(AgentTurnFinished {
            agent,
            turn_id: 1,
            entries: Vec::new(),
            usage: None,
            stop_reason: None,
        });
        app.update();

        assert!(app.world().get_entity(live_card).is_err());
        let view = app.world().get::<AgentView>(agent).unwrap();
        assert!(view.assistant.is_none());
        let agent_ref = app.world().get::<Agent>(agent).unwrap();
        assert!(agent_ref.current_turn.is_none());
        assert!(!agent_ref.pending);
    }
}

fn budget_exhausted_message() -> String {
    format!(
        "tool call budget for this root call is exhausted ({})",
        crate::tools::schedule::MAX_CALLS_PER_ROOT
    )
}
