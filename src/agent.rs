//! Agent state, UI, and per-agent systems.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    env,
    time::{Duration, Instant},
};

use bevy::{
    input::mouse::MouseScrollUnit,
    input_focus::{FocusCause, InputFocus, tab_navigation::TabIndex},
    log::error,
    picking::events::{PointerClick, PointerScroll},
    prelude::*,
    text::{EditableText, TextCursorStyle},
    ui_widgets::{Activate, Button, TextInput},
};
use bevy_tokio_tasks::TokioTasksRuntime;
use llm_api::{
    anthropic,
    model_config::{ModelConfig, ModelKind},
    openai_completions, openai_responses,
    stream::{RawStreamBytes, StreamEvent, finish_draft_for},
    usage::{Usage, UsageProvider},
    wire_common::thinking_joined,
};
use llm_history_model::history::{
    DraftBlock, Entry, MessageContent, Thinking, ToolCall, ToolCallResult,
};
use serde_json::Value;
use tokio::sync::watch;

use crate::{
    config::{AppConfig, Provider},
    js_runtime::{HostCommand, HostRequest, JsChannels, JsEvent, JsRuntimeHandle},
    messages::*,
    tools::{ToolTarget, background_setting, classify_tool_call, execute_tool, interpreter_task},
    transport::{request_body, stream_transport},
    virtual_fs::VirtualFs,
};

/// Keep the rendered transcript bounded even when the conversation keeps growing.
const MAX_TRANSCRIPT_CARDS: usize = 120;
/// Prevent accidental runaway agent spawning from exhausting UI/VRAM.
pub(crate) const MAX_AGENTS: usize = 64;

// ---------------------------------------------------------------------------
// Tool rounds
// ---------------------------------------------------------------------------

/// One in-flight tool round. Results are stored by call index so asynchronous
/// JavaScript completions cannot reorder the history entry.
pub(crate) struct PendingToolRound {
    agent: Entity,
    turn_id: u64,
    calls: Vec<ToolCall>,
    results: Vec<Option<ToolCallResult>>,
    remaining: usize,
}

/// A detached interpreter task whose result arrives as an agent message.
struct BackgroundTask {
    task_id: u64,
    agent: Entity,
}

/// A normal (non-detached) interpreter request inside a tool round.
struct PendingJsRequest {
    round_id: u64,
    call_index: usize,
    started_at: Option<Instant>,
    /// False when the model explicitly passed `run_in_background: false`.
    auto_background: bool,
}

/// Detaches long-running interpreter tasks after a grace period.
#[derive(Resource)]
pub(crate) struct BackgroundPolicy {
    auto_after: Option<Duration>,
}

impl Default for BackgroundPolicy {
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

/// Truncation limits for long reasoning and tool-call bodies. A body collapses
/// when it exceeds either limit.
#[derive(Resource, Clone, Debug)]
pub(crate) struct TextCollapseConfig {
    pub(crate) max_chars: usize,
    pub(crate) max_lines: usize,
}

impl Default for TextCollapseConfig {
    fn default() -> Self {
        const DEFAULT_CHARS: usize = 1_000;
        const DEFAULT_LINES: usize = 10;
        Self {
            max_chars: env_limit("COLLAPSE_MAX_CHARS").unwrap_or(DEFAULT_CHARS),
            max_lines: env_limit("COLLAPSE_MAX_LINES").unwrap_or(DEFAULT_LINES),
        }
    }
}

fn env_limit(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

/// Tracks tool rounds that are waiting for one or more JS contexts.
#[derive(Resource, Default)]
pub(crate) struct ToolRounds {
    active: HashMap<u64, PendingToolRound>,
    /// Request id returned by [`JsRuntimeHandle::submit`] -> pending round slot.
    js_requests: HashMap<u64, PendingJsRequest>,
    /// Detached requests are deliberately not part of a round.
    background: HashMap<u64, BackgroundTask>,
    next_round: u64,
    next_task_id: u64,
}

impl ToolRounds {
    /// Stop tracking a round (used by cancellation and failures).
    fn cancel_rounds(&mut self, agent: Entity) {
        let rounds = self
            .active
            .iter()
            .filter_map(|(id, round)| (round.agent == agent).then_some(*id))
            .collect::<Vec<_>>();
        for round_id in rounds {
            self.active.remove(&round_id);
        }
        self.js_requests
            .retain(|_, pending| self.active.contains_key(&pending.round_id));
    }

    /// Drop everything belonging to an agent, including detached tasks.
    fn remove_agent(&mut self, agent: Entity) {
        self.cancel_rounds(agent);
        self.background.retain(|_, task| task.agent != agent);
    }

    /// Register a detached interpreter request and return its user-visible id.
    fn start_background(&mut self, request: u64, agent: Entity) -> u64 {
        self.next_task_id = self.next_task_id.wrapping_add(1);
        let task_id = self.next_task_id;
        self.background
            .insert(request, BackgroundTask { task_id, agent });
        task_id
    }

    fn take_background(&mut self, request: u64) -> Option<BackgroundTask> {
        self.background.remove(&request)
    }
}

fn background_started_message(task_id: u64) -> String {
    format!(
        "Task {task_id} running in background, do not poll for it, continue your task or end turn to await result."
    )
}

fn finish_round_if_ready(
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
}

impl Agent {
    fn new(name: String, provider: Provider, model_name: String) -> Self {
        Self {
            name: name.clone(),
            provider,
            model_name,
            history: vec![Entry::SystemPrompt {
                content: format!(
                    "You are the LLM agent named {name}. Avoid using markdown and LaTeX \
                     notation."
                ),
            }],
            pending: false,
            current_turn: None,
            streaming_text: String::new(),
            streaming_thinking: String::new(),
            pending_messages: VecDeque::new(),
            retry_message: None,
        }
    }

    fn model_config(&self) -> ModelConfig {
        ModelConfig {
            kind: self.provider.kind(&self.model_name),
            tools: crate::tools::tools(),
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

/// Entity handles for one agent's UI.
#[derive(Component)]
pub(crate) struct AgentView {
    panel: Entity,
    log: Entity,
    input: Entity,
    status: Entity,
    /// Fixed-width usage readout (input/output/cached tokens).
    stats: Entity,
    tab_label: Entity,
    assistant: Option<AssistantView>,
    /// Whether the transcript cards for this agent are currently spawned.
    transcript_visible: bool,
    /// Number of history entries the current transcript reflects.
    rendered_entries: usize,
    scroll_to_bottom: bool,
    /// Whether streaming should keep the transcript pinned to the newest text.
    /// Cleared when the user scrolls away from the bottom.
    follow_tail: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct AssistantView {
    card: Entity,
    thinking_container: Entity,
    thinking_body: Entity,
    answer_body: Entity,
}

/// The visible agent tab for [`Agent`].
#[derive(Component)]
pub(crate) struct AgentTab {
    agent: Entity,
}

/// Retry button belonging to one agent.
#[derive(Component)]
pub(crate) struct RetryAgentButton {
    agent: Entity,
}

/// Destructive button that removes the agent it belongs to.
#[derive(Component)]
pub(crate) struct KillAgentButton {
    agent: Entity,
}

#[derive(Component)]
pub(crate) struct MessageBody;

#[derive(Component)]
pub(crate) struct StatusText;

/// The scroll container that owns one agent's transcript cards.
#[derive(Component)]
pub(crate) struct TranscriptLog;

/// A removable transcript card root.
#[derive(Component)]
pub(crate) struct TranscriptCard;

/// How a collapsed body chooses the text it keeps visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CollapseMode {
    /// Show the leading lines (used for tool output).
    Head,
    /// Show the leading and trailing lines (used for live reasoning, so the
    /// newest reasoning stays visible while it streams in).
    HeadAndTail,
}

/// A text body that collapses behind a "Show more" header when it exceeds the
/// configured character or line limit.
#[derive(Component)]
pub(crate) struct CollapsibleText {
    full: String,
    collapsed: bool,
    max_chars: usize,
    max_lines: usize,
    mode: CollapseMode,
    /// Text entity in the header that shows the expand/collapse affordance.
    hint: Entity,
}

impl CollapsibleText {
    fn new(full: String, config: &TextCollapseConfig, mode: CollapseMode, hint: Entity) -> Self {
        Self {
            full,
            collapsed: true,
            max_chars: config.max_chars,
            max_lines: config.max_lines,
            mode,
            hint,
        }
    }

    fn exceeds_limit(&self) -> bool {
        self.full.chars().count() > self.max_chars || self.full.lines().count() > self.max_lines
    }

    /// Text that should currently be rendered. Long text starts truncated and
    /// is only shown in full after the user expands it.
    fn visible_text(&self) -> String {
        if self.collapsed && self.exceeds_limit() {
            match self.mode {
                CollapseMode::Head => collapsed_text(&self.full, self.max_chars, self.max_lines),
                CollapseMode::HeadAndTail => {
                    collapsed_head_tail_text(&self.full, self.max_chars, self.max_lines)
                }
            }
        } else {
            self.full.clone()
        }
    }

    fn hint_text(&self) -> String {
        if !self.exceeds_limit() {
            return String::new();
        }
        if self.collapsed {
            format!(
                "Show more ({} chars, {} lines)",
                self.full.chars().count(),
                self.full.lines().count()
            )
        } else {
            "Show less".to_owned()
        }
    }
}

/// Clickable header that toggles the [`CollapsibleText`] stored on `body`.
#[derive(Component)]
pub(crate) struct CollapsibleToggle {
    body: Entity,
}

/// Keep the first `max_lines` lines and then at most `max_chars` characters.
fn collapsed_text(full: &str, max_chars: usize, max_lines: usize) -> String {
    let line_limited = full.lines().take(max_lines).collect::<Vec<_>>().join("\n");
    let truncated = line_limited.chars().take(max_chars).collect::<String>();
    let hidden = full
        .chars()
        .count()
        .saturating_sub(truncated.chars().count());
    if hidden == 0 {
        truncated
    } else {
        format!("{truncated}\n... ({hidden} more characters)")
    }
}

/// Show the first half and the last half of the line budget, with an ellipsis
/// between them. Each side also gets half of the character budget, so the
/// collapsed block stays compact even when lines are very long.
fn collapsed_head_tail_text(full: &str, max_chars: usize, max_lines: usize) -> String {
    const SEPARATOR: &str = "\n...\n";
    let lines = full.lines().collect::<Vec<_>>();

    if lines.len() <= max_lines {
        // Only the character limit was exceeded. Split the single block into
        // leading and trailing character windows instead.
        return head_tail_chars(full, max_chars, SEPARATOR);
    }

    let head_lines = (max_lines / 2).max(1);
    let tail_lines = max_lines.saturating_sub(head_lines).max(1);
    let head_end = head_lines.min(lines.len());
    let tail_start = lines.len().saturating_sub(tail_lines).max(head_end);
    let head = lines[..head_end].join("\n");
    let tail = lines[tail_start..].join("\n");

    let budget = max_chars.saturating_sub(SEPARATOR.chars().count());
    let head_budget = budget / 2;
    let tail_budget = budget - head_budget;
    let head = head.chars().take(head_budget).collect::<String>();
    let tail = tail
        .chars()
        .rev()
        .take(tail_budget)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();

    format!("{head}{SEPARATOR}{tail}")
}

fn head_tail_chars(full: &str, max_chars: usize, separator: &str) -> String {
    let budget = max_chars.saturating_sub(separator.chars().count());
    let head_budget = budget / 2;
    let tail_budget = budget - head_budget;
    let head = full.chars().take(head_budget).collect::<String>();
    let tail = full
        .chars()
        .rev()
        .take(tail_budget)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!("{head}{separator}{tail}")
}

/// Toggles a collapsible body when its header is clicked. Pointer events bubble
/// from the header's text children, so `click.entity` is the header entity.
fn toggle_collapsible(
    click: On<PointerClick>,
    toggles: Query<&CollapsibleToggle>,
    mut bodies: Query<&mut CollapsibleText>,
    mut texts: Query<&mut Text>,
) {
    let Ok(toggle) = toggles.get(click.entity) else {
        return;
    };
    let Ok(mut body) = bodies.get_mut(toggle.body) else {
        return;
    };
    if !body.exceeds_limit() {
        return;
    }

    body.collapsed = !body.collapsed;
    let visible = body.visible_text();
    let hint = body.hint_text();
    let hint_entity = body.hint;
    if let Ok(mut text) = texts.get_mut(toggle.body) {
        text.0 = visible;
    }
    if let Ok(mut text) = texts.get_mut(hint_entity) {
        text.0 = hint;
    }
}

/// Handles for the shared shell that hosts all agents.
#[derive(Resource)]
pub(crate) struct AgentLayout {
    pub(crate) tabs: Entity,
    pub(crate) panels: Entity,
    pub(crate) next_number: usize,
}

#[derive(Resource)]
pub(crate) struct ActiveAgent(pub(crate) Option<Entity>);

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
    mut active: ResMut<ActiveAgent>,
    mut index: ResMut<AgentIndex>,
) {
    let mut created = 0;
    for _ in requests.read() {
        if index.by_name.len() + created >= MAX_AGENTS {
            continue;
        }
        let number = layout.next_number;
        layout.next_number += 1;
        let name = format!("LLM Agent {number}");
        let agent = create_agent(&mut commands, &layout, &config, name.clone(), number as i32);
        index.by_name.insert(name, agent);
        active.0 = Some(agent);
        created += 1;
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

    let panel = create_agent_panel(commands, layout.panels, agent_entity);
    let (tab, tab_label) = create_agent_tab(commands, layout.tabs, agent_entity, &name, tab_index);

    commands.entity(agent_entity).insert(AgentView {
        panel: panel.panel,
        log: panel.log,
        input: panel.input,
        status: panel.status,
        stats: panel.stats,
        tab_label,
        assistant: None,
        transcript_visible: false,
        rendered_entries: 0,
        scroll_to_bottom: false,
        follow_tail: true,
    });

    // Select this tab when clicked.
    commands
        .entity(tab)
        .observe(move |_: On<Activate>, mut active: ResMut<ActiveAgent>| {
            active.0 = Some(agent_entity);
        });

    agent_entity
}

pub(crate) struct PanelEntities {
    panel: Entity,
    log: Entity,
    input: Entity,
    status: Entity,
    stats: Entity,
}

pub(crate) fn create_agent_panel(
    commands: &mut Commands,
    panels_root: Entity,
    agent: Entity,
) -> PanelEntities {
    let panel = commands
        .spawn((
            Node {
                width: percent(100.),
                height: percent(100.),
                min_height: px(0.),
                display: Display::None,
                flex_direction: FlexDirection::Column,
                row_gap: px(12.),
                ..default()
            },
            Visibility::Hidden,
        ))
        .id();

    let log = commands
        .spawn((
            Node {
                width: percent(100.),
                flex_basis: px(0.),
                flex_grow: 1.,
                min_height: px(0.),
                flex_direction: FlexDirection::Column,
                row_gap: px(10.),
                padding: px(4.).all(),
                overflow: Overflow::scroll_y(),
                ..default()
            },
            TranscriptLog,
        ))
        .id();

    // Manual scrolling. The observer is attached to the scroll container so
    // wheel events that hit child message cards bubble up to it.
    commands.entity(log).observe(
        move |on_scroll: On<PointerScroll>,
              mut logs: Query<(&mut ScrollPosition, &ComputedNode)>,
              mut views: Query<&mut AgentView>| {
            let Ok((mut scroll_position, computed)) = logs.get_mut(on_scroll.entity) else {
                return;
            };

            let dy = match on_scroll.unit {
                MouseScrollUnit::Line => on_scroll.y * 20.,
                MouseScrollUnit::Pixel => on_scroll.y,
            };
            let max_offset = ((computed.content_size().y - computed.size().y).max(0.))
                * computed.inverse_scale_factor();
            let next = (scroll_position.y - dy).clamp(0., max_offset);
            scroll_position.y = next;

            if let Ok(mut view) = views.get_mut(agent) {
                view.follow_tail = next >= max_offset - 1.;
                view.scroll_to_bottom = false;
            }
        },
    );

    let input = commands
        .spawn((
            Node {
                flex_grow: 1.,
                flex_basis: px(0.),
                min_width: px(0.),
                min_height: px(46.),
                padding: px(12.).all(),
                border: px(2.).all(),
                border_radius: BorderRadius::all(px(10.)),
                align_items: AlignItems::Center,
                ..default()
            },
            TextInput,
            EditableText {
                visible_lines: Some(1.),
                allow_newlines: false,
                ..default()
            },
            TextLayout::no_wrap(),
            TextFont {
                font_size: FontSize::Px(18.),
                ..default()
            },
            TextCursorStyle::default(),
            BackgroundColor(Color::srgb(0.10, 0.115, 0.15)),
            BorderColor::all(Color::srgb(0.24, 0.28, 0.36)),
            TabIndex(1),
        ))
        .id();

    let retry = commands
        .spawn((
            Button,
            RetryAgentButton { agent },
            Node {
                min_width: px(82.),
                height: px(46.),
                padding: UiRect::horizontal(px(14.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(2.).all(),
                border_radius: BorderRadius::all(px(10.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.11, 0.12, 0.14)),
            BorderColor::all(Color::srgb(0.22, 0.25, 0.30)),
            TabIndex(2),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Retry"),
                TextFont {
                    font_size: FontSize::Px(16.),
                    ..default()
                },
                TextColor(Color::srgb(0.82, 0.88, 0.96)),
            ));
        })
        .id();

    commands
        .entity(retry)
        .observe(move |_: On<Activate>, mut commands: Commands| {
            commands.write_message(RetryAgentTurn { agent });
        });

    let stop = commands
        .spawn((
            Button,
            Node {
                min_width: px(82.),
                height: px(46.),
                padding: UiRect::horizontal(px(14.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(2.).all(),
                border_radius: BorderRadius::all(px(10.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.22, 0.10, 0.11)),
            BorderColor::all(Color::srgb(0.55, 0.25, 0.27)),
            TabIndex(3),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Stop"),
                TextFont {
                    font_size: FontSize::Px(16.),
                    ..default()
                },
                TextColor(Color::srgb(0.98, 0.82, 0.82)),
            ));
        })
        .id();

    commands
        .entity(stop)
        .observe(move |_: On<Activate>, mut commands: Commands| {
            commands.write_message(CancelAgentRequest { agent });
        });

    let kill = commands
        .spawn((
            Button,
            KillAgentButton { agent },
            Node {
                min_width: px(82.),
                height: px(46.),
                padding: UiRect::horizontal(px(14.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(2.).all(),
                border_radius: BorderRadius::all(px(10.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.30, 0.07, 0.08)),
            BorderColor::all(Color::srgb(0.72, 0.20, 0.22)),
            TabIndex(5),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Kill"),
                TextFont {
                    font_size: FontSize::Px(16.),
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.78, 0.78)),
            ));
        })
        .id();

    commands.entity(kill).observe(
        |activate: On<Activate>, buttons: Query<&KillAgentButton>, mut commands: Commands| {
            if let Ok(button) = buttons.get(activate.entity) {
                commands.write_message(RemoveAgentRequest {
                    agent: button.agent,
                });
            }
        },
    );

    let scroll_to_bottom = commands
        .spawn((
            Button,
            Node {
                min_width: px(140.),
                height: px(28.),
                flex_shrink: 0.,
                padding: UiRect::horizontal(px(10.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(1.).all(),
                border_radius: BorderRadius::all(px(8.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.13, 0.16, 0.21)),
            BorderColor::all(Color::srgb(0.30, 0.36, 0.46)),
            TabIndex(4),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Scroll to bottom"),
                TextFont {
                    font_size: FontSize::Px(12.),
                    ..default()
                },
                TextColor(Color::srgb(0.82, 0.88, 0.96)),
            ));
        })
        .id();

    commands.entity(scroll_to_bottom).observe(
        move |_: On<Activate>, mut views: Query<&mut AgentView>| {
            if let Ok(mut view) = views.get_mut(agent) {
                // Jump now and keep following new streaming output.
                view.follow_tail = true;
                view.scroll_to_bottom = true;
            }
        },
    );

    let input_row = commands
        .spawn(Node {
            width: percent(100.),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(8.),
            ..default()
        })
        .add_children(&[input, retry, stop, kill])
        .id();

    let status = commands
        .spawn((
            Text::new("Ready"),
            TextFont {
                font_size: FontSize::Px(13.),
                ..default()
            },
            TextColor(Color::srgb(0.55, 0.60, 0.70)),
            StatusText,
        ))
        .id();

    let mut stats = Entity::PLACEHOLDER;
    let stats_box = commands
        .spawn((
            Node {
                width: px(320.),
                height: px(28.),
                flex_shrink: 0.,
                padding: UiRect::horizontal(px(10.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(1.).all(),
                border_radius: BorderRadius::all(px(8.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.10, 0.12, 0.16)),
            BorderColor::all(Color::srgb(0.22, 0.26, 0.33)),
        ))
        .with_children(|parent| {
            stats = parent
                .spawn((
                    Text::new(stats_text(None)),
                    TextFont {
                        font_size: FontSize::Px(12.),
                        ..default()
                    },
                    TextColor(Color::srgb(0.62, 0.68, 0.78)),
                    TextLayout::no_wrap(),
                ))
                .id();
        })
        .id();

    let footer = commands
        .spawn(Node {
            width: percent(100.),
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: px(8.),
            ..default()
        })
        .add_children(&[status, stats_box, scroll_to_bottom])
        .id();

    commands
        .entity(panel)
        .add_children(&[log, input_row, footer]);
    commands.entity(panels_root).add_children(&[panel]);

    PanelEntities {
        panel,
        log,
        input,
        status,
        stats,
    }
}

pub(crate) fn create_agent_tab(
    commands: &mut Commands,
    tabs: Entity,
    agent: Entity,
    name: &str,
    tab_index: i32,
) -> (Entity, Entity) {
    let mut label = Entity::PLACEHOLDER;
    let tab = commands
        .spawn((
            Button,
            AgentTab { agent },
            Node {
                min_width: px(84.),
                height: px(34.),
                padding: UiRect::horizontal(px(14.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(1.).all(),
                border_radius: BorderRadius::all(px(8.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.12, 0.14, 0.18)),
            BorderColor::all(Color::srgb(0.24, 0.28, 0.36)),
            TabIndex(tab_index),
        ))
        .with_children(|parent| {
            label = parent
                .spawn((
                    Text::new(name.to_owned()),
                    TextFont {
                        font_size: FontSize::Px(14.),
                        ..default()
                    },
                    TextColor(Color::srgb(0.82, 0.87, 0.95)),
                ))
                .id();
        })
        .id();

    commands.entity(tabs).add_children(&[tab]);
    (tab, label)
}

/// Header and body entities for a collapsible text block.
struct CollapsibleParts {
    header: Entity,
    body: Entity,
}

/// Spawn a text block with a clickable header. The body starts collapsed when
/// its contents exceed the configured limit.
#[allow(clippy::too_many_arguments)]
fn spawn_collapsible_text(
    commands: &mut Commands,
    label: &str,
    label_font_size: f32,
    label_color: Color,
    text: String,
    font_size: f32,
    text_color: Color,
    mode: CollapseMode,
    collapse: &TextCollapseConfig,
) -> CollapsibleParts {
    let mut hint = Entity::PLACEHOLDER;
    let header = commands
        .spawn((
            Node {
                width: percent(100.),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                column_gap: px(8.),
                ..default()
            },
            CollapsibleToggle {
                body: Entity::PLACEHOLDER,
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(label.to_owned()),
                TextFont {
                    font_size: FontSize::Px(label_font_size),
                    ..default()
                },
                TextColor(label_color),
            ));
            hint = parent
                .spawn((
                    Text::new(""),
                    TextFont {
                        font_size: FontSize::Px(12.),
                        ..default()
                    },
                    TextColor(Color::srgb(0.55, 0.75, 1.0)),
                    TextLayout::no_wrap(),
                ))
                .id();
        })
        .id();

    let state = CollapsibleText::new(text, collapse, mode, hint);
    let hint_text = state.hint_text();
    let visible = state.visible_text();
    let body = commands
        .spawn((
            Text::new(visible),
            TextFont {
                font_size: FontSize::Px(font_size),
                ..default()
            },
            TextColor(text_color),
            TextLayout {
                linebreak: LineBreak::WordOrCharacter,
                ..default()
            },
            Node {
                width: percent(100.),
                ..default()
            },
            state,
        ))
        .id();

    commands.entity(hint).insert(Text::new(hint_text));
    commands
        .entity(header)
        .insert(CollapsibleToggle { body })
        .observe(toggle_collapsible);

    CollapsibleParts { header, body }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_message_card(
    commands: &mut Commands,
    log: Entity,
    role: &str,
    role_color: Color,
    text: String,
    background: Color,
    collapsible: bool,
    collapse: &TextCollapseConfig,
) -> Entity {
    let card = commands
        .spawn((
            Node {
                width: percent(100.),
                flex_direction: FlexDirection::Column,
                row_gap: px(4.),
                padding: px(12.).all(),
                border_radius: BorderRadius::all(px(10.)),
                ..default()
            },
            BackgroundColor(background),
            TranscriptCard,
        ))
        .id();

    let body = if collapsible {
        let parts = spawn_collapsible_text(
            commands,
            role,
            13.,
            role_color,
            text,
            18.,
            Color::srgb(0.90, 0.92, 0.96),
            CollapseMode::Head,
            collapse,
        );
        commands
            .entity(card)
            .add_children(&[parts.header, parts.body]);
        parts.body
    } else {
        let mut body = Entity::PLACEHOLDER;
        commands.entity(card).with_children(|parent| {
            parent.spawn((
                Text::new(role),
                TextFont {
                    font_size: FontSize::Px(13.),
                    ..default()
                },
                TextColor(role_color),
            ));
            body = parent
                .spawn((
                    Text::new(text),
                    TextFont {
                        font_size: FontSize::Px(18.),
                        ..default()
                    },
                    TextColor(Color::srgb(0.90, 0.92, 0.96)),
                    TextLayout {
                        linebreak: LineBreak::WordOrCharacter,
                        ..default()
                    },
                    Node {
                        width: percent(100.),
                        ..default()
                    },
                    MessageBody,
                ))
                .id();
        });
        body
    };

    commands.entity(log).add_children(&[card]);
    body
}

pub(crate) fn spawn_assistant_card(
    commands: &mut Commands,
    log: Entity,
    text: &str,
    thinking: Option<&str>,
    collapse: &TextCollapseConfig,
) -> AssistantView {
    let card = commands
        .spawn((
            Node {
                width: percent(100.),
                flex_direction: FlexDirection::Column,
                row_gap: px(8.),
                padding: px(12.).all(),
                border_radius: BorderRadius::all(px(10.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.10, 0.115, 0.15)),
            TranscriptCard,
        ))
        .id();

    let role = commands
        .spawn((
            Text::new("Assistant"),
            TextFont {
                font_size: FontSize::Px(13.),
                ..default()
            },
            TextColor(Color::srgb(0.55, 0.85, 0.70)),
        ))
        .id();

    let thinking_container = commands
        .spawn(Node {
            width: percent(100.),
            display: if thinking.is_some() {
                Display::Flex
            } else {
                Display::None
            },
            flex_direction: FlexDirection::Column,
            row_gap: px(3.),
            padding: px(9.).all(),
            border_radius: BorderRadius::all(px(7.)),
            ..default()
        })
        .insert(BackgroundColor(Color::srgb(0.085, 0.095, 0.125)))
        .id();

    let thinking_parts = spawn_collapsible_text(
        commands,
        "Thinking",
        12.,
        Color::srgb(0.62, 0.66, 0.76),
        thinking.map(str::to_owned).unwrap_or_default(),
        15.,
        Color::srgb(0.70, 0.74, 0.82),
        CollapseMode::HeadAndTail,
        collapse,
    );

    let answer = if !text.is_empty() {
        text.to_owned()
    } else if thinking.is_some() {
        "".to_owned()
    } else {
        "(empty response)".to_owned()
    };

    let answer_body = commands
        .spawn((
            Text::new(answer),
            TextFont {
                font_size: FontSize::Px(18.),
                ..default()
            },
            TextColor(Color::srgb(0.90, 0.92, 0.96)),
            TextLayout {
                linebreak: LineBreak::WordOrCharacter,
                ..default()
            },
            Node {
                width: percent(100.),
                ..default()
            },
            MessageBody,
        ))
        .id();

    commands
        .entity(thinking_container)
        .add_children(&[thinking_parts.header, thinking_parts.body]);
    commands
        .entity(card)
        .add_children(&[role, thinking_container, answer_body]);
    commands.entity(log).add_children(&[card]);

    AssistantView {
        card,
        thinking_container,
        thinking_body: thinking_parts.body,
        answer_body,
    }
}

fn tool_result_summary(results: &[ToolCallResult], tool_names: &HashMap<String, String>) -> String {
    results
        .iter()
        .map(|result| {
            let name = tool_names
                .get(&result.tool_use_id)
                .map(String::as_str)
                .unwrap_or(&result.tool_use_id);
            format!(
                "{name} [{}]\n{}",
                if result.is_error { "error" } else { "ok" },
                result.result
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn tool_name_map(history: &[Entry]) -> HashMap<String, String> {
    let mut names = HashMap::new();
    for entry in history {
        if let Entry::ResponseToolCall { tool_call, .. } = entry {
            for call in tool_call {
                names.insert(call.tool_use_id.clone(), call.tool_name.clone());
            }
        }
    }
    names
}

fn render_entry(
    commands: &mut Commands,
    log: Entity,
    entry: &Entry,
    tool_names: &HashMap<String, String>,
    collapse: &TextCollapseConfig,
) {
    match entry {
        Entry::SystemPrompt { content } | Entry::SystemMessage { content, .. } => {
            spawn_message_card(
                commands,
                log,
                "System",
                Color::srgb(0.55, 0.60, 0.70),
                content.clone(),
                Color::srgb(0.09, 0.10, 0.13),
                false,
                collapse,
            );
        }
        Entry::UserMessage { content } => {
            if let Some(text) = &content.text {
                spawn_message_card(
                    commands,
                    log,
                    "You",
                    Color::srgb(0.55, 0.75, 1.0),
                    text.clone(),
                    Color::srgb(0.12, 0.17, 0.26),
                    false,
                    collapse,
                );
            }
        }
        Entry::Response { .. } | Entry::ResponseToolCall { .. } => {
            let text = entry.response_joined_text().unwrap_or_default();
            let thinking = response_thinking_text(std::slice::from_ref(entry));
            if !text.is_empty() || thinking.is_some() {
                spawn_assistant_card(commands, log, &text, thinking.as_deref(), collapse);
            }

            if let Entry::ResponseToolCall { tool_call, .. } = entry
                && !tool_call.is_empty()
            {
                let names = tool_call
                    .iter()
                    .map(|call| format!("`{}`", call.tool_name))
                    .collect::<Vec<_>>()
                    .join(", ");
                spawn_message_card(
                    commands,
                    log,
                    "Tool calls",
                    Color::srgb(0.95, 0.75, 0.40),
                    format!("Calling {names}"),
                    Color::srgb(0.13, 0.115, 0.09),
                    true,
                    collapse,
                );
            }
        }
        Entry::ToolCallResult { tool_call } => {
            spawn_message_card(
                commands,
                log,
                "Tool calls",
                Color::srgb(0.95, 0.75, 0.40),
                tool_result_summary(tool_call, tool_names),
                Color::srgb(0.13, 0.115, 0.09),
                true,
                collapse,
            );
        }
        Entry::Draft { .. } => {}
    }
}

/// Rebuild the visible transcript from the last [`MAX_TRANSCRIPT_CARDS`]
/// history entries plus the live streaming card, if any.
pub(crate) fn render_transcript(
    commands: &mut Commands,
    view: &mut AgentView,
    agent: &Agent,
    collapse: &TextCollapseConfig,
) {
    commands.entity(view.log).despawn_children();

    let start = agent.history.len().saturating_sub(MAX_TRANSCRIPT_CARDS);
    let recent = &agent.history[start..];
    let tool_names = tool_name_map(recent);
    for entry in recent {
        render_entry(commands, view.log, entry, &tool_names, collapse);
    }

    view.assistant = None;
    if !agent.streaming_text.is_empty() || !agent.streaming_thinking.is_empty() {
        let thinking =
            (!agent.streaming_thinking.is_empty()).then_some(agent.streaming_thinking.as_str());
        view.assistant = Some(spawn_assistant_card(
            commands,
            view.log,
            &agent.streaming_text,
            thinking,
            collapse,
        ));
    }

    view.transcript_visible = true;
    view.rendered_entries = agent.history.len();
    view.scroll_to_bottom = true;
}

/// Render only the newest history entry when exactly one was appended. This is
/// the "draft popped" path: the live assistant card is replaced by the
/// finalized block without rebuilding the whole transcript.
pub(crate) fn refresh_transcripts(
    mut commands: Commands,
    mut views: Query<(&mut AgentView, &Agent)>,
    collapse: Res<TextCollapseConfig>,
) {
    for (mut view, agent) in &mut views {
        if !view.transcript_visible {
            continue;
        }

        let history_len = agent.history.len();
        if history_len < view.rendered_entries {
            // Rollback or reset: the transcript is no longer trustworthy.
            render_transcript(&mut commands, &mut view, agent, &collapse);
            continue;
        }
        if history_len > view.rendered_entries + 1 {
            render_transcript(&mut commands, &mut view, agent, &collapse);
            continue;
        }
        if history_len == view.rendered_entries + 1 {
            if let Some(assistant) = view.assistant.take() {
                commands.entity(assistant.card).despawn();
            }
            if let Some(entry) = agent.history.last() {
                let start = history_len.saturating_sub(MAX_TRANSCRIPT_CARDS);
                let tool_names = tool_name_map(&agent.history[start..history_len]);
                render_entry(&mut commands, view.log, entry, &tool_names, &collapse);
            }
            view.rendered_entries = history_len;
            view.scroll_to_bottom = true;
        }
    }
}

/// Keep the number of live transcript cards bounded.
pub(crate) fn trim_transcripts(
    mut commands: Commands,
    logs: Query<&Children, With<TranscriptLog>>,
    cards: Query<(), With<TranscriptCard>>,
) {
    for children in &logs {
        let excess = children.len().saturating_sub(MAX_TRANSCRIPT_CARDS);
        if excess == 0 {
            continue;
        }

        let mut removed = 0;
        for child in children.iter() {
            if cards.contains(child) {
                commands.entity(child).despawn();
                removed += 1;
                if removed == excess {
                    break;
                }
            }
        }
    }
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
    mut active: ResMut<ActiveAgent>,
    mut focus: ResMut<InputFocus>,
) {
    if active.0 == Some(remove.entity) {
        active.0 = views
            .iter()
            .map(|(entity, _)| entity)
            .find(|entity| *entity != remove.entity);
        if active.0.is_none() {
            focus.clear();
        }
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn sync_active_agent(
    mut active: ResMut<ActiveAgent>,
    mut focus: ResMut<InputFocus>,
    mut views: Query<(Entity, &mut AgentView, &Agent)>,
    mut nodes: Query<&mut Node>,
    mut visibilities: Query<&mut Visibility>,
    mut tabs: Query<(&AgentTab, &mut BackgroundColor, &mut BorderColor)>,
    collapse: Res<TextCollapseConfig>,
    mut commands: Commands,
) {
    let active_valid = active.0.is_some_and(|entity| views.get(entity).is_ok());
    if !active_valid {
        active.0 = views.iter().next().map(|(entity, _, _)| entity);
    }
    let Some(active_entity) = active.0 else {
        return;
    };

    for (entity, mut view, agent) in &mut views {
        let is_active = entity == active_entity;
        if let Ok(mut node) = nodes.get_mut(view.panel) {
            node.display = if is_active {
                Display::Flex
            } else {
                Display::None
            };
        }
        if let Ok(mut visibility) = visibilities.get_mut(view.panel) {
            *visibility = if is_active {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
        }

        if is_active {
            if !view.transcript_visible {
                render_transcript(&mut commands, &mut view, agent, &collapse);
            }
        } else if view.transcript_visible {
            commands.entity(view.log).despawn_children();
            view.transcript_visible = false;
            view.rendered_entries = 0;
            view.assistant = None;
        }
    }

    for (tab, mut background, mut border) in &mut tabs {
        if tab.agent == active_entity {
            background.0 = Color::srgb(0.18, 0.24, 0.34);
            *border = BorderColor::all(Color::srgb(0.45, 0.62, 0.88));
        } else {
            background.0 = Color::srgb(0.12, 0.14, 0.18);
            *border = BorderColor::all(Color::srgb(0.24, 0.28, 0.36));
        }
    }

    if active.is_changed()
        && let Ok((_, view, _)) = views.get(active_entity)
    {
        focus.set(view.input, FocusCause::Navigated);
    }
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
        agent.history.push(Entry::UserMessage {
            content: MessageContent {
                text: Some(text),
                image: None,
            },
        });
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
    active: Res<ActiveAgent>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut inputs: Query<&mut EditableText>,
    mut texts: Query<&mut Text>,
) {
    if !keyboard.just_pressed(KeyCode::Enter) {
        return;
    }
    let Some(agent_entity) = active.0 else {
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
    agent.history.push(Entry::UserMessage {
        content: MessageContent {
            text: Some(combined.join("\n\n")),
            image: None,
        },
    });
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
        let config = agent.model_config();
        let body = match request_body(agent.provider, &agent.history, &config) {
            Ok(body) => body,
            Err(err) => {
                error!("could not build request for {}: {err:#}", agent.name);
                agent.retry_message = rollback_pending_turn(&mut agent.history);
                agent.pending = false;
                set_text(
                    &mut texts,
                    view.status,
                    format!("Could not build request: {err:#}"),
                );
                continue;
            }
        };

        let provider = agent.provider;
        let kind = config.kind.clone();
        let turn_id = runtime.next_turn_id;
        runtime.next_turn_id += 1;
        agent.current_turn = Some(turn_id);
        agent.pending = true;
        agent.streaming_text.clear();
        agent.streaming_thinking.clear();

        if view.transcript_visible {
            let assistant = spawn_assistant_card(&mut commands, view.log, "...", None, &collapse);
            view.assistant = Some(assistant);
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

fn ensure_live_assistant(
    commands: &mut Commands,
    view: &mut AgentView,
    agent: &Agent,
    collapse: &TextCollapseConfig,
) {
    if view.assistant.is_none() && view.transcript_visible {
        let thinking =
            (!agent.streaming_thinking.is_empty()).then_some(agent.streaming_thinking.as_str());
        view.assistant = Some(spawn_assistant_card(
            commands,
            view.log,
            &agent.streaming_text,
            thinking,
            collapse,
        ));
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

        if message.usage.is_some() {
            set_text(&mut texts, view.stats, stats_text(message.usage.as_ref()));
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
            agent.retry_message = rollback_pending_turn(&mut agent.history);
            agent.pending = false;
            agent.current_turn = None;
            if let Some(assistant) = view.assistant.take() {
                commands.entity(assistant.card).despawn();
            }
            set_text(&mut texts, view.status, "Empty response");
        } else {
            agent.history.extend(message.entries.iter().cloned());
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
    agents: Query<&Agent>,
    mut index: ResMut<AgentIndex>,
    mut layout: ResMut<AgentLayout>,
    mut fs: ResMut<VirtualFs>,
    config: Res<AppConfig>,
    mut send_agent_message: MessageWriter<SendAgentMessage>,
    mut remove_agent: MessageWriter<RemoveAgentRequest>,
    mut rename_agent: MessageWriter<RenameAgentRequest>,
    js_handles: Query<&JsRuntimeHandle>,
    mut rounds: ResMut<ToolRounds>,
    mut finished: MessageWriter<ToolRoundFinished>,
    mut commands: Commands,
) {
    for request in requests.read() {
        let Ok(agent) = agents.get(request.agent) else {
            continue;
        };
        if agent.current_turn != Some(request.turn_id) || request.tool_call.is_empty() {
            continue;
        }

        let round_id = rounds.next_round;
        rounds.next_round += 1;

        let mut results: Vec<Option<ToolCallResult>> =
            (0..request.tool_call.len()).map(|_| None).collect();
        let mut remaining = request.tool_call.len();

        for (call_index, call) in request.tool_call.iter().enumerate() {
            let arguments =
                serde_json::from_str::<Value>(&call.toolcall_params).unwrap_or(Value::Null);

            match classify_tool_call(&call.tool_name) {
                ToolTarget::Agent => {
                    let (result, is_error) = execute_tool(
                        &call.tool_name,
                        &arguments,
                        request.agent,
                        &agent.name,
                        &mut index,
                        &agents,
                        &mut layout,
                        &mut fs,
                        &config,
                        &mut commands,
                        &mut send_agent_message,
                        &mut remove_agent,
                        &mut rename_agent,
                    );
                    results[call_index] = Some(ToolCallResult {
                        result,
                        is_error,
                        tool_use_id: call.tool_use_id.clone(),
                    });
                    remaining -= 1;
                }
                ToolTarget::Interpreter => {
                    let supports_background = matches!(
                        call.tool_name.as_str(),
                        "run_javascript" | "eval_javascript"
                    );
                    let background = if supports_background {
                        background_setting(arguments.get("run_in_background"))
                    } else {
                        None
                    };
                    let immediate_background = background == Some(true);
                    let auto_background = supports_background && background != Some(false);

                    let submitted =
                        interpreter_task(&call.tool_name, &arguments, &fs).and_then(|task| {
                            js_handles
                                .get(request.agent)
                                .map_err(|_| {
                                    format!("agent {} has no JavaScript runtime", request.agent)
                                })?
                                .submit(task)
                        });
                    match submitted {
                        Ok(request_id) if immediate_background => {
                            let task_id = rounds.start_background(request_id, request.agent);
                            results[call_index] = Some(ToolCallResult {
                                result: background_started_message(task_id),
                                is_error: false,
                                tool_use_id: call.tool_use_id.clone(),
                            });
                            remaining -= 1;
                        }
                        Ok(request_id) => {
                            rounds.js_requests.insert(
                                request_id,
                                PendingJsRequest {
                                    round_id,
                                    call_index,
                                    started_at: None,
                                    auto_background,
                                },
                            );
                        }
                        Err(err) => {
                            results[call_index] = Some(ToolCallResult {
                                result: err,
                                is_error: true,
                                tool_use_id: call.tool_use_id.clone(),
                            });
                            remaining -= 1;
                        }
                    }
                }
            }
        }

        rounds.active.insert(
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
    mut rounds: ResMut<ToolRounds>,
    mut finished: MessageWriter<ToolRoundFinished>,
    mut task_results: MessageWriter<TaskResult>,
) {
    for event in channels.drain_events() {
        let (agent, request, result) = match event {
            JsEvent::Started { request, at } => {
                if let Some(pending) = rounds.js_requests.get_mut(&request) {
                    pending.started_at = Some(at);
                }
                continue;
            }
            JsEvent::Done {
                agent,
                request,
                result,
            } => (agent, request, result),
        };

        if let Some(task) = rounds.take_background(request) {
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
            continue;
        }

        let Some(pending) = rounds.js_requests.remove(&request) else {
            // The round was cancelled (or the agent failed) while JavaScript
            // was still running. Nothing to apply.
            continue;
        };
        let round_id = pending.round_id;

        {
            let Some(round) = rounds.active.get_mut(&round_id) else {
                continue;
            };
            if round.agent != agent {
                continue;
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

        finish_round_if_ready(&mut rounds, round_id, &mut finished);
    }
}

/// Detach normal interpreter requests that have been running longer than the
/// configured grace period.
pub(crate) fn auto_background_long_js_tasks(
    policy: Res<BackgroundPolicy>,
    mut rounds: ResMut<ToolRounds>,
    mut finished: MessageWriter<ToolRoundFinished>,
) {
    let Some(after) = policy.auto_after else {
        return;
    };

    let now = Instant::now();
    let due = rounds
        .js_requests
        .iter()
        .filter(|(_, pending)| {
            pending.auto_background
                && pending
                    .started_at
                    .is_some_and(|started| now.saturating_duration_since(started) >= after)
        })
        .map(|(request, _)| *request)
        .collect::<Vec<_>>();

    for request in due {
        let Some(pending) = rounds.js_requests.remove(&request) else {
            continue;
        };
        let Some(agent) = rounds
            .active
            .get(&pending.round_id)
            .map(|round| round.agent)
        else {
            continue;
        };

        let task_id = rounds.start_background(request, agent);
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

/// Execute host commands coming from context threads: JavaScript `execute`
/// tool calls and virtual-filesystem reads for module imports.
#[allow(clippy::too_many_arguments)]
pub(crate) fn poll_js_host_requests(
    channels: Res<JsChannels>,
    mut fs: ResMut<VirtualFs>,
    mut index: ResMut<AgentIndex>,
    mut layout: ResMut<AgentLayout>,
    config: Res<AppConfig>,
    agents: Query<&Agent>,
    js_handles: Query<&JsRuntimeHandle>,
    mut send_agent_message: MessageWriter<SendAgentMessage>,
    mut remove_agent: MessageWriter<RemoveAgentRequest>,
    mut rename_agent: MessageWriter<RenameAgentRequest>,
    mut commands: Commands,
) {
    for request in channels.drain_host_requests() {
        let HostRequest {
            agent: agent_entity,
            token,
            command,
        } = request;

        let (result, is_error) = match command {
            HostCommand::Tool { name, arguments } => match agents.get(agent_entity) {
                Ok(agent) => execute_tool(
                    &name,
                    &arguments,
                    agent_entity,
                    &agent.name,
                    &mut index,
                    &agents,
                    &mut layout,
                    &mut fs,
                    &config,
                    &mut commands,
                    &mut send_agent_message,
                    &mut remove_agent,
                    &mut rename_agent,
                ),
                Err(_) => (
                    format!("agent entity {agent_entity} no longer exists"),
                    true,
                ),
            },
            HostCommand::ReadFile { path } => match fs.read(&path) {
                Ok(content) => (content, false),
                Err(err) => (err, true),
            },
        };

        if let Ok(handle) = js_handles.get(agent_entity) {
            handle.respond(token, if is_error { Err(result) } else { Ok(result) });
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

        agent.history.push(Entry::ToolCallResult {
            tool_call: message.results.clone(),
        });

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
        agent.retry_message = rollback_pending_turn(&mut agent.history);
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
    agents: Query<(Entity, &AgentView)>,
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

        // The panel and tab are separate roots, not children of the agent.
        if let Ok((_, view)) = agents.get(agent) {
            commands.entity(view.panel).despawn();
        }
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

pub(crate) fn response_display_text(entries: &[Entry]) -> Option<String> {
    let text = entries
        .iter()
        .filter_map(Entry::response_joined_text)
        .collect::<Vec<_>>()
        .join("\n\n");
    if !text.is_empty() {
        return Some(text);
    }

    for entry in entries {
        if let Entry::ResponseToolCall { tool_call, .. } = entry
            && !tool_call.is_empty()
        {
            return Some(format!(
                "Calling {}...",
                tool_call
                    .iter()
                    .map(|call| format!("`{}`", call.tool_name))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }

    None
}

pub(crate) fn response_thinking_text(entries: &[Entry]) -> Option<String> {
    let thinking = entries
        .iter()
        .flat_map(|entry| entry.response_thinking_blocks())
        .cloned()
        .collect::<Vec<Thinking>>();
    thinking_joined(&thinking)
}

pub(crate) fn set_text(texts: &mut Query<&mut Text>, entity: Entity, value: impl Into<String>) {
    if let Ok(mut text) = texts.get_mut(entity) {
        text.0 = value.into();
    }
}

/// Update a collapsible body, re-applying its current collapsed/expanded state
/// and refreshing the header hint. Falls back to a plain text update for bodies
/// that are not collapsible.
fn set_collapsible_text(
    texts: &mut Query<&mut Text>,
    collapsibles: &mut Query<&mut CollapsibleText>,
    entity: Entity,
    value: impl Into<String>,
) {
    let value = value.into();
    let Ok(mut collapsible) = collapsibles.get_mut(entity) else {
        if let Ok(mut text) = texts.get_mut(entity) {
            text.0 = value;
        }
        return;
    };

    collapsible.full = value;
    let visible = collapsible.visible_text();
    let hint = collapsible.hint_text();
    let hint_entity = collapsible.hint;
    if let Ok(mut text) = texts.get_mut(entity) {
        text.0 = visible;
    }
    if let Ok(mut text) = texts.get_mut(hint_entity) {
        text.0 = hint;
    }
}

pub(crate) fn set_display(nodes: &mut Query<&mut Node>, entity: Entity, display: Display) {
    if let Ok(mut node) = nodes.get_mut(entity) {
        node.display = display;
    }
}

/// Renders the fixed-width usage readout. Fields are padded so the text does
/// not shift as token counts grow (Bevy's default font is monospaced).
pub(crate) fn stats_text(usage: Option<&Usage>) -> String {
    fn field(value: Option<u64>) -> String {
        value.map_or_else(|| "-".to_owned(), |value| value.to_string())
    }

    let input = field(usage.and_then(|usage| usage.input_tokens));
    let output = field(usage.and_then(|usage| usage.output_tokens));
    let cached = field(usage.and_then(|usage| usage.cached_input_tokens));
    format!("in {input:>6}   out {output:>6}   cached {cached:>6}")
}

pub(crate) fn auto_scroll(
    mut views: Query<&mut AgentView>,
    mut logs: Query<(&mut ScrollPosition, &ComputedNode)>,
) {
    for mut view in &mut views {
        if !view.follow_tail || !view.scroll_to_bottom {
            continue;
        }
        let Ok((mut scroll_position, computed)) = logs.get_mut(view.log) else {
            continue;
        };

        let max_offset =
            (computed.content_size() - computed.size()) * computed.inverse_scale_factor();
        scroll_position.y = max_offset.y.max(0.);
        view.scroll_to_bottom = false;
    }
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
                    log: Entity::PLACEHOLDER,
                    input: Entity::PLACEHOLDER,
                    status: Entity::PLACEHOLDER,
                    stats: Entity::PLACEHOLDER,
                    tab_label: Entity::PLACEHOLDER,
                    assistant: None,
                    transcript_visible: false,
                    rendered_entries: 0,
                    scroll_to_bottom: false,
                    follow_tail: true,
                },
            ))
            .id();
        app.world_mut().entity_mut(tab).insert(AgentTab { agent });
        (agent, panel, tab)
    }

    #[test]
    fn removing_an_agent_clears_ui_tabs_index_and_active_selection() {
        let mut app = App::new();
        app.init_resource::<AgentRuntime>();
        app.init_resource::<ToolRounds>();
        app.init_resource::<AgentIndex>();
        app.init_resource::<InputFocus>();
        app.insert_resource(ActiveAgent(None));
        app.add_message::<RemoveAgentRequest>();
        app.add_observer(index_agent_on_insert);
        app.add_observer(index_agent_on_discard);
        app.add_observer(deactivate_removed_agent);
        app.add_systems(Update, remove_agents);

        let (first, first_panel, first_tab) = spawn_test_agent(&mut app, "First");
        let (second, _, _) = spawn_test_agent(&mut app, "Second");
        app.world_mut().resource_mut::<ActiveAgent>().0 = Some(first);

        app.world_mut()
            .write_message(RemoveAgentRequest { agent: first });
        app.update();

        assert!(app.world().get_entity(first).is_err());
        assert!(app.world().get_entity(first_panel).is_err());
        assert!(app.world().get_entity(first_tab).is_err());

        let index = app.world().resource::<AgentIndex>();
        assert!(!index.by_name.contains_key("First"));
        assert!(index.by_name.contains_key("Second"));

        assert_eq!(app.world().resource::<ActiveAgent>().0, Some(second));
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
    use std::time::{Duration, Instant};

    #[test]
    fn removing_agent_drops_background_tasks() {
        let mut rounds = ToolRounds::default();
        let agent = Entity::PLACEHOLDER;
        rounds.start_background(7, agent);
        rounds.remove_agent(agent);
        assert!(rounds.take_background(7).is_none());
    }

    #[test]
    fn long_running_js_is_auto_backgrounded() {
        let mut rounds = ToolRounds::default();
        let agent = Entity::PLACEHOLDER;
        let round_id = 10;
        let request = 20;

        rounds.active.insert(
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
        rounds.js_requests.insert(
            request,
            PendingJsRequest {
                round_id,
                call_index: 0,
                started_at: Some(Instant::now() - Duration::from_secs(10)),
                auto_background: true,
            },
        );

        let mut app = App::new();
        app.insert_resource(rounds);
        app.insert_resource(BackgroundPolicy {
            auto_after: Some(Duration::from_secs(5)),
        });
        app.add_message::<ToolRoundFinished>();
        app.add_systems(Update, auto_background_long_js_tasks);
        app.update();

        let rounds = app.world().resource::<ToolRounds>();
        assert!(rounds.js_requests.is_empty());
        assert!(rounds.background.contains_key(&request));
        assert!(!rounds.active.contains_key(&round_id));
    }

    #[test]
    fn explicit_false_prevents_auto_backgrounding() {
        let mut rounds = ToolRounds::default();
        let agent = Entity::PLACEHOLDER;
        let round_id = 10;
        let request = 20;

        rounds.active.insert(
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
        rounds.js_requests.insert(
            request,
            PendingJsRequest {
                round_id,
                call_index: 0,
                started_at: Some(Instant::now() - Duration::from_secs(10)),
                auto_background: false,
            },
        );

        let mut app = App::new();
        app.insert_resource(rounds);
        app.insert_resource(BackgroundPolicy {
            auto_after: Some(Duration::from_secs(5)),
        });
        app.add_message::<ToolRoundFinished>();
        app.add_systems(Update, auto_background_long_js_tasks);
        app.update();

        let rounds = app.world().resource::<ToolRounds>();
        assert!(rounds.js_requests.contains_key(&request));
        assert!(!rounds.background.contains_key(&request));
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
            })
            .unwrap();
        app.world_mut()
            .resource_mut::<ToolRounds>()
            .start_background(request, agent);

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
                    panel: Entity::PLACEHOLDER,
                    log: Entity::PLACEHOLDER,
                    input: Entity::PLACEHOLDER,
                    status,
                    stats: Entity::PLACEHOLDER,
                    tab_label: Entity::PLACEHOLDER,
                    assistant: Some(AssistantView {
                        card: live_card,
                        thinking_container: Entity::PLACEHOLDER,
                        thinking_body: Entity::PLACEHOLDER,
                        answer_body: Entity::PLACEHOLDER,
                    }),
                    transcript_visible: true,
                    rendered_entries: 3,
                    scroll_to_bottom: false,
                    follow_tail: true,
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
                    panel: Entity::PLACEHOLDER,
                    log: Entity::PLACEHOLDER,
                    input: Entity::PLACEHOLDER,
                    status,
                    stats: Entity::PLACEHOLDER,
                    tab_label,
                    assistant: None,
                    transcript_visible: false,
                    rendered_entries: 0,
                    scroll_to_bottom: false,
                    follow_tail: true,
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
mod transcript_tests {
    use super::*;

    fn agent_view(log: Entity, assistant: Option<AssistantView>) -> AgentView {
        AgentView {
            panel: Entity::PLACEHOLDER,
            log,
            input: Entity::PLACEHOLDER,
            status: Entity::PLACEHOLDER,
            stats: Entity::PLACEHOLDER,
            tab_label: Entity::PLACEHOLDER,
            assistant,
            transcript_visible: true,
            rendered_entries: 1,
            scroll_to_bottom: false,
            follow_tail: true,
        }
    }

    #[test]
    fn draft_pop_replaces_only_the_live_card() {
        let mut app = App::new();

        let log = app.world_mut().spawn((Node::default(), TranscriptLog)).id();
        let old_card = app
            .world_mut()
            .spawn((Node::default(), TranscriptCard))
            .id();
        let live_card = app
            .world_mut()
            .spawn((Node::default(), TranscriptCard))
            .id();
        app.world_mut()
            .entity_mut(log)
            .add_children(&[old_card, live_card]);

        let agent = app
            .world_mut()
            .spawn(Agent::new(
                "Worker".to_owned(),
                Provider::Responses,
                "model".to_owned(),
            ))
            .id();
        app.world_mut()
            .get_mut::<Agent>(agent)
            .unwrap()
            .history
            .push(Entry::response_text("final answer"));

        app.world_mut().entity_mut(agent).insert(agent_view(
            log,
            Some(AssistantView {
                card: live_card,
                thinking_container: Entity::PLACEHOLDER,
                thinking_body: Entity::PLACEHOLDER,
                answer_body: Entity::PLACEHOLDER,
            }),
        ));
        app.add_systems(Update, refresh_transcripts);
        app.init_resource::<TextCollapseConfig>();
        app.update();

        assert!(
            app.world().get_entity(live_card).is_err(),
            "the old draft card should be gone"
        );
        assert!(
            app.world().get_entity(old_card).is_ok(),
            "earlier transcript cards must not be rebuilt"
        );
        let children = app.world().get::<Children>(log).unwrap();
        assert_eq!(children.len(), 2);
        assert_eq!(
            app.world()
                .get::<AgentView>(agent)
                .unwrap()
                .rendered_entries,
            2
        );
    }
}

#[cfg(test)]
mod display_tests {
    use super::*;

    fn config() -> TextCollapseConfig {
        TextCollapseConfig {
            max_chars: 1_000,
            max_lines: 10,
        }
    }

    fn numbered_lines(count: usize) -> String {
        (0..count)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn short_text_is_not_collapsed() {
        let state = CollapsibleText::new(
            "short".to_owned(),
            &config(),
            CollapseMode::HeadAndTail,
            Entity::PLACEHOLDER,
        );
        assert!(!state.exceeds_limit());
        assert_eq!(state.visible_text(), "short");
        assert_eq!(state.hint_text(), "");
    }

    #[test]
    fn tool_output_collapses_to_the_leading_lines() {
        let state = CollapsibleText::new(
            numbered_lines(20),
            &config(),
            CollapseMode::Head,
            Entity::PLACEHOLDER,
        );
        let visible = state.visible_text();
        assert!(visible.starts_with("line 0"), "{visible}");
        assert!(!visible.contains("line 10"), "{visible}");
        assert!(visible.contains("more characters"), "{visible}");
        assert!(state.hint_text().contains("Show more"));
    }

    #[test]
    fn reasoning_shows_the_first_and_last_lines() {
        let mut state = CollapsibleText::new(
            numbered_lines(20),
            &config(),
            CollapseMode::HeadAndTail,
            Entity::PLACEHOLDER,
        );
        let visible = state.visible_text();
        for index in 0..5 {
            assert!(visible.contains(&format!("line {index}")), "{visible}");
        }
        for index in 15..20 {
            assert!(visible.contains(&format!("line {index}")), "{visible}");
        }
        assert!(!visible.contains("line 7"), "{visible}");
        assert!(visible.contains("..."), "{visible}");
        assert!(state.hint_text().contains("Show more"));

        // New reasoning updates the trailing window while the body stays collapsed.
        state.full.push_str("\nline 20");
        let visible = state.visible_text();
        assert!(visible.contains("line 16"), "{visible}");
        assert!(visible.contains("line 20"), "{visible}");
        assert!(state.collapsed);
    }

    #[test]
    fn long_tool_result_card_starts_collapsed() {
        let mut world = World::new();
        let log = world.spawn_empty().id();
        let body = {
            let mut commands = world.commands();
            spawn_message_card(
                &mut commands,
                log,
                "Tool calls",
                Color::srgb(0.95, 0.75, 0.40),
                numbered_lines(20),
                Color::srgb(0.13, 0.115, 0.09),
                true,
                &config(),
            )
        };
        world.flush();

        let state = world
            .get::<CollapsibleText>(body)
            .expect("tool card body is collapsible");
        assert!(state.collapsed);
        let rendered = &world.get::<Text>(body).unwrap().0;
        assert!(rendered.starts_with("line 0"), "{rendered}");
        assert!(!rendered.contains("line 10"), "{rendered}");
    }

    #[test]
    fn clicking_a_collapsible_header_expands_the_body() {
        use bevy::{
            camera::NormalizedRenderTarget,
            picking::{
                backend::HitData,
                events::Pointer,
                pointer::{Location, PointerButton, PointerId},
            },
        };

        let mut world = World::new();
        let parts = {
            let mut commands = world.commands();
            spawn_collapsible_text(
                &mut commands,
                "Thinking",
                12.,
                Color::WHITE,
                numbered_lines(20),
                15.,
                Color::WHITE,
                CollapseMode::HeadAndTail,
                &config(),
            )
        };
        world.flush();

        let collapsed = world.get::<Text>(parts.body).unwrap().0.clone();
        assert!(!collapsed.contains("line 10"), "{collapsed}");

        let camera = world.spawn_empty().id();
        world.trigger(PointerClick {
            entity: parts.header,
            pointer: Pointer::new(
                PointerId::Mouse,
                Location {
                    position: Vec2::ZERO,
                    target: NormalizedRenderTarget::None {
                        width: 1,
                        height: 1,
                    },
                },
            ),
            button: PointerButton::Primary,
            hit: HitData::new(camera, 0., None, None),
            duration: Duration::ZERO,
            count: 1,
        });

        let expanded = world.get::<Text>(parts.body).unwrap().0.clone();
        assert_eq!(expanded, numbered_lines(20));
        assert!(!world.get::<CollapsibleText>(parts.body).unwrap().collapsed);
    }

    #[test]
    fn thinking_only_assistant_card_is_not_ellipsis() {
        let mut world = World::new();
        let log = world.spawn_empty().id();
        let assistant = {
            let mut commands = world.commands();
            spawn_assistant_card(
                &mut commands,
                log,
                "",
                Some("reasoning"),
                &TextCollapseConfig::default(),
            )
        };
        world.flush();

        assert_eq!(world.get::<Text>(assistant.answer_body).unwrap().0, "");
        assert_eq!(
            world.get::<Text>(assistant.thinking_body).unwrap().0,
            "reasoning"
        );
    }

    #[test]
    fn tool_results_show_the_called_tool_name() {
        let results = vec![ToolCallResult {
            result: "file contents".to_owned(),
            is_error: false,
            tool_use_id: "call-1".to_owned(),
        }];
        let names = HashMap::from([("call-1".to_owned(), "fs_read".to_owned())]);
        let summary = tool_result_summary(&results, &names);
        assert!(summary.contains("fs_read"), "{summary}");
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
            panel: Entity::PLACEHOLDER,
            log,
            input: Entity::PLACEHOLDER,
            status,
            stats: Entity::PLACEHOLDER,
            tab_label: Entity::PLACEHOLDER,
            assistant: Some(AssistantView {
                card: live_card,
                thinking_container: Entity::PLACEHOLDER,
                thinking_body: Entity::PLACEHOLDER,
                answer_body: Entity::PLACEHOLDER,
            }),
            transcript_visible: true,
            rendered_entries: 1,
            scroll_to_bottom: false,
            follow_tail: true,
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
