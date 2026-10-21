//! Demo LLM agent client for a local llama.cpp server.
//!
//! Select the wire API with `PROVIDER` (the endpoint URLs mirror
//! `crates/llm_connector_example`). OpenAI Responses is the default;
//! Anthropic support is best effort:
//!
//! ```text
//! cargo run                            # http://127.0.0.1:8080/v1/responses
//! PROVIDER=completions  cargo run      # http://127.0.0.1:8080/v1/chat/completions
//! PROVIDER=anthropic    cargo run      # http://127.0.0.1:8080/v1/messages (best effort)
//! ```
//!
//! `MODEL` overrides the model name sent to the server. `ENDPOINT` overrides
//! the provider URL (for example an HTTPS OpenCode Go endpoint), `API_TOKEN`
//! is sent as `Authorization: Bearer <token>`, and `SESSION_ID` is sent as
//! `x-opencode-session` for request routing:
//!
//! ```text
//! ENDPOINT=https://opencode.ai/inference/go/openai/v1/chat/completions \
//! PROVIDER=completions MODEL=deepseek-v4.1-flash API_TOKEN=... cargo run
//! ```
//!
//! Each conversation is a [`Agent`] component on its own entity. Tabs select
//! the visible agent while all agents can stream independently.
//!
//! Stages communicate through Bevy messages:
//!
//! transport task -> `StreamChunk`/`StreamEnded` -> decoder system
//! -> `AgentStreamEvent` (UI) and `AgentTurnFinished` (history/tool calls)
//! -> `RunToolCalls` -> `StartAgentTurn` -> next transport task.

use std::{
    collections::{HashMap, VecDeque},
    env,
    sync::mpsc::{self, Receiver},
};

use bevy::{
    input::mouse::MouseScrollUnit,
    input_focus::{
        FocusCause, InputFocus,
        tab_navigation::{TabGroup, TabIndex, TabNavigationPlugin},
    },
    picking::events::PointerScroll,
    prelude::*,
    text::{EditableText, TextCursorStyle},
    ui::UiSystems,
    ui_widgets::{Activate, Button, TextInput},
    window::WindowResolution,
};
use bevy_brp_extras::BrpExtrasPlugin;
use bevy_tokio_tasks::{TaskContext, TokioTasksPlugin, TokioTasksRuntime};
use boa_engine::{Context as JsContext, JsValue, NativeFunction, Source};
use futures_util::StreamExt as _;
use llm_api::{
    anthropic,
    generic_tool::{ParameterDescription, ParameterType, Tool},
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

/// Which server-side streaming API to exercise.
#[derive(Clone, Copy, Debug)]
enum Provider {
    Anthropic,
    Completions,
    Responses,
}

impl Provider {
    fn from_env() -> Self {
        match env::var("PROVIDER").as_deref() {
            Ok("anthropic") => Self::Anthropic,
            Ok("completions") => Self::Completions,
            _ => Self::Responses,
        }
    }

    /// Endpoints copied from `crates/llm_connector_example`.
    fn endpoint(self) -> &'static str {
        match self {
            Provider::Anthropic => "http://127.0.0.1:8080/v1/messages",
            Provider::Completions => "http://127.0.0.1:8080/v1/chat/completions",
            Provider::Responses => "http://127.0.0.1:8080/v1/responses",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::Completions => "completions",
            Provider::Responses => "responses",
        }
    }

    fn kind(self, model_name: &str) -> ModelKind {
        match self {
            Provider::Anthropic => ModelKind::AnthropicCompatible {
                model_name: model_name.into(),
                streaming: true,
                thinking: None,
                // llama.cpp returns thinking blocks with `signature: ""`.
                allow_unsigned_thinking: true,
                system_message_policy: Default::default(),
            },
            Provider::Completions | Provider::Responses => ModelKind::OpenAICompatible {
                model_name: model_name.into(),
                streaming: true,
                prompt_cache_key: String::new(),
                provider: UsageProvider::Other,
                include_usage: true,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Persistent JavaScript runtimes
// ---------------------------------------------------------------------------

/// One Boa context per LLM Agent, plus that context's private output channel.
/// The map is stored as a non-send resource, so the JavaScript heap survives
/// for the whole agent session.
struct JsSession {
    context: JsContext,
    output: Receiver<String>,
}

#[derive(Default)]
struct JsRuntimes {
    sessions: HashMap<Entity, JsSession>,
}

impl JsRuntimes {
    fn eval(&mut self, agent: Entity, code: &str) -> Result<String, String> {
        if !self.sessions.contains_key(&agent) {
            self.sessions.insert(agent, new_js_context()?);
        }
        let session = self
            .sessions
            .get_mut(&agent)
            .expect("JavaScript context was just inserted");

        // Drop anything left over from an earlier evaluation before running.
        session.output.try_iter().for_each(drop);

        let value = session
            .context
            .eval(Source::from_bytes(code))
            .map_err(|err| format!("JavaScript error: {err}"))?;
        let _async_result = session
            .context
            .run_jobs()
            .map_err(|err| format!("JavaScript async error: {err}"))?;
        let rendered = value
            .to_string(&mut session.context)
            .map(|text| text.to_std_string_escaped())
            .unwrap_or_else(|_| value.display().to_string());
        let printed = session.output.try_iter().collect::<Vec<_>>().join("\n");

        Ok(match (printed.is_empty(), rendered == "undefined") {
            (true, _) => rendered,
            (false, true) => printed,
            (false, false) => format!("{printed}\n{rendered}"),
        })
    }

    fn reset(&mut self, agent: Entity) -> String {
        if self.sessions.remove(&agent).is_some() {
            "Dropped the whole JavaScript context for this agent. The next evaluation starts fresh."
                .to_owned()
        } else {
            "No JavaScript context was active for this agent; the next evaluation will start fresh."
                .to_owned()
        }
    }

    fn drop_symbol(&mut self, agent: Entity, symbol: &str) -> Result<String, String> {
        if !is_js_identifier(symbol) {
            return Err(format!(
                "`{symbol}` is not a single JavaScript identifier. Omit `symbol` to reset the whole context."
            ));
        }
        if !self.sessions.contains_key(&agent) {
            return Ok(format!(
                "No JavaScript context is active, so `{symbol}` does not exist."
            ));
        }

        let before = self.eval(agent, &format!("typeof {symbol}"))?;
        if before == "undefined" {
            return Ok(format!("No symbol `{symbol}` is defined."));
        }

        // `delete` only removes configurable global-object properties. Lexical
        // bindings (`let`/`const`) and non-configurable `var` bindings survive,
        // which the follow-up `typeof` check detects.
        self.eval(agent, &format!("delete globalThis.{symbol}"))?;
        let after = self.eval(agent, &format!("typeof {symbol}"))?;

        if after == "undefined" {
            Ok(format!("Dropped `{symbol}`."))
        } else {
            Ok(format!(
                "Could not drop `{symbol}`: it is a lexical or non-configurable binding. \
                 Reset the whole context instead."
            ))
        }
    }
}

fn is_js_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_' || first == '$') {
        return false;
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '$')
}

fn new_js_context() -> Result<JsSession, String> {
    let mut context = JsContext::default();

    let (tx, rx) = mpsc::channel();
    // SAFETY: the closure only captures an `mpsc::Sender<String>`, which is not
    // a GC-managed Boa value, so the collector has nothing to trace or move.
    let print = unsafe {
        NativeFunction::from_closure(move |_this, args, context| {
            let line = args
                .iter()
                .map(|value| {
                    value
                        .to_string(context)
                        .map(|text| text.to_std_string_escaped())
                        .unwrap_or_else(|_| value.display().to_string())
                })
                .collect::<Vec<_>>()
                .join(" ");
            let _ = tx.send(line);
            Ok(JsValue::undefined())
        })
    };

    // Protect the host from runaway loops in model-generated code.
    context
        .runtime_limits_mut()
        .set_loop_iteration_limit(10_000_000);
    context
        .register_global_builtin_callable("print".into(), 1, print)
        .map_err(|err| format!("failed to initialize JavaScript runtime: {err}"))?;
    context
        .eval(Source::from_bytes(
            "globalThis.console = { log: print, info: print, warn: print, error: print };",
        ))
        .map_err(|err| format!("failed to initialize JavaScript console: {err}"))?;

    Ok(JsSession {
        context,
        output: rx,
    })
}

// ---------------------------------------------------------------------------
// Bevy messages
// ---------------------------------------------------------------------------

/// Ask the agent coordinator to start a request for the current history.
#[derive(Message)]
struct StartAgentTurn {
    agent: Entity,
}

/// Create a new agent tab and panel.
#[derive(Message)]
struct NewAgentRequested;

/// Abort the current generation for an agent and drop its pending user turn.
#[derive(Message)]
struct CancelAgentRequest {
    agent: Entity,
}

/// Resend the last user turn after a failure or cancellation.
#[derive(Message)]
struct RetryAgentTurn {
    agent: Entity,
}

/// Raw HTTP body bytes produced by the transport task.
#[derive(Message)]
struct StreamChunk {
    agent: Entity,
    turn_id: u64,
    bytes: Vec<u8>,
}

/// The HTTP body ended normally.
#[derive(Message)]
struct StreamEnded {
    agent: Entity,
    turn_id: u64,
}

/// The HTTP request failed.
#[derive(Message)]
struct StreamFailed {
    agent: Entity,
    turn_id: u64,
    message: String,
}

/// One decoded provider event. UI systems consume these directly.
#[derive(Message)]
struct AgentStreamEvent {
    agent: Entity,
    turn_id: u64,
    event: StreamEvent,
}

/// The stream decoder finished and produced replayable history entries.
#[derive(Message)]
struct AgentTurnFinished {
    agent: Entity,
    turn_id: u64,
    entries: Vec<Entry>,
    usage: Option<Usage>,
    stop_reason: Option<String>,
}

/// An agent turn failed after decoding or during the final commit.
#[derive(Message)]
struct AgentFailed {
    agent: Entity,
    turn_id: u64,
    message: String,
}

/// Execute the tool calls carried by a finished assistant turn.
#[derive(Message)]
struct RunToolCalls {
    agent: Entity,
    turn_id: u64,
    tool_call: Vec<ToolCall>,
}

/// Queue a message for another agent (used by the cross-agent tool).
#[derive(Message)]
struct SendAgentMessage {
    target: Entity,
    from: String,
    message: String,
}

// ---------------------------------------------------------------------------
// Agent components
// ---------------------------------------------------------------------------

/// One independent conversation. All agent state lives on this component.
#[derive(Component)]
struct Agent {
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
                     notation. You can list and message other agents with the list_agents and \
                     send_message_to_agent tools."
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
            tools: tools(),
            max_tokens: Some(8192),
        }
    }
}

/// A message waiting for the next legal insertion point in an agent's history.
enum QueuedMessage {
    /// Typed by the user into this agent's input box.
    User(String),
    /// Sent by another agent through `send_message_to_agent`.
    Agent { from: String, text: String },
}

/// Entity handles for one agent's UI.
#[derive(Component)]
struct AgentView {
    panel: Entity,
    log: Entity,
    input: Entity,
    status: Entity,
    /// Fixed-width usage readout (input/output/cached tokens).
    stats: Entity,
    assistant: Option<AssistantView>,
    scroll_to_bottom: bool,
    /// Whether streaming should keep the transcript pinned to the newest text.
    /// Cleared when the user scrolls away from the bottom.
    follow_tail: bool,
}

#[derive(Clone, Copy)]
struct AssistantView {
    thinking_container: Entity,
    thinking_body: Entity,
    answer_body: Entity,
}

/// The visible agent tab for [`Agent`].
#[derive(Component)]
struct AgentTab {
    agent: Entity,
}

/// Retry button belonging to one agent.
#[derive(Component)]
struct RetryAgentButton {
    agent: Entity,
}

#[derive(Component)]
struct MessageBody;

#[derive(Component)]
struct StatusText;

/// Global configuration shared by newly created agents.
#[derive(Resource)]
struct AppConfig {
    provider: Provider,
    model_name: String,
    endpoint: String,
    token: String,
    /// Stable `x-opencode-session` value used for request routing.
    session_id: String,
}

impl AppConfig {
    /// Returns the token for an `Authorization: Bearer <token>` header.
    fn bearer_token(&self) -> Option<&str> {
        let token = self.token.trim();
        (!token.is_empty()).then_some(token)
    }
}

/// Stable identifier for OpenCode Go request routing. `SESSION_ID` overrides
/// it; otherwise one id is derived once per process so prompt caching stays
/// effective for the whole session.
fn session_id_from_env() -> String {
    env::var("SESSION_ID").unwrap_or_else(|_| llm_api::model_config::generate_session_id())
}

/// Handles for the shared shell that hosts all agents.
#[derive(Resource)]
struct AgentLayout {
    tabs: Entity,
    panels: Entity,
    next_number: usize,
}

#[derive(Resource)]
struct ActiveAgent(Option<Entity>);

/// Name -> entity lookup used by the cross-agent tool.
#[derive(Resource, Default)]
struct AgentIndex {
    by_name: HashMap<String, Entity>,
}

/// Shared in-memory flat filesystem. Every agent reads and writes the same
/// files, and nothing touches the host filesystem.
#[derive(Resource, Default)]
struct VirtualFs {
    files: HashMap<String, String>,
}

fn split_virtual_file(content: &str) -> (Vec<String>, bool) {
    (
        content.lines().map(str::to_owned).collect(),
        content.ends_with('\n'),
    )
}

impl VirtualFs {
    /// The virtual filesystem has no directories: a path is a plain file name.
    fn normalize_path(path: &str) -> Result<String, String> {
        let path = path.trim();
        if path.is_empty() {
            return Err("path must not be empty".to_owned());
        }
        if path.contains('/') || path.contains('\\') {
            return Err(format!(
                "invalid path `{path}`: the virtual filesystem is flat (no directories)"
            ));
        }
        if path == "." || path == ".." {
            return Err(format!("invalid file name `{path}`"));
        }
        Ok(path.to_owned())
    }

    fn list(&self) -> String {
        if self.files.is_empty() {
            return "Virtual filesystem is empty.".to_owned();
        }

        let mut names: Vec<&String> = self.files.keys().collect();
        names.sort_unstable();
        let mut listing = format!("Files ({}):", names.len());
        for name in names {
            let content = &self.files[name];
            listing.push_str(&format!(
                "\n{name} - {} bytes, {} line(s)",
                content.len(),
                content.lines().count()
            ));
        }
        listing
    }

    fn read(&self, path: &str) -> Result<String, String> {
        let path = Self::normalize_path(path)?;
        self.files
            .get(&path)
            .cloned()
            .ok_or_else(|| format!("No such file: `{path}`. Use fs_write to create it."))
    }

    fn write(&mut self, path: &str, content: String) -> Result<String, String> {
        let path = Self::normalize_path(path)?;
        let lines = content.lines().count();
        self.files.insert(path.clone(), content);
        Ok(format!("Wrote `{path}` ({lines} lines)."))
    }

    fn delete(&mut self, path: &str) -> Result<String, String> {
        let path = Self::normalize_path(path)?;
        let content = self
            .files
            .remove(&path)
            .ok_or_else(|| format!("No such file: `{path}`"))?;
        Ok(format!("Deleted `{path}` ({} bytes).", content.len()))
    }

    /// Replaces an inclusive, 1-based line range. An empty `content` deletes
    /// the selected lines.
    fn edit_lines(
        &mut self,
        path: &str,
        start_line: u64,
        end_line: u64,
        content: String,
    ) -> Result<String, String> {
        let path = Self::normalize_path(path)?;
        let original = self
            .files
            .get(&path)
            .cloned()
            .ok_or_else(|| format!("No such file: `{path}`"))?;
        let (mut lines, trailing_newline) = split_virtual_file(&original);

        let start = usize::try_from(start_line)
            .map_err(|_| format!("invalid start_line `{start_line}`"))?;
        let end =
            usize::try_from(end_line).map_err(|_| format!("invalid end_line `{end_line}`"))?;
        if start == 0 {
            return Err("start_line is 1-based and must be at least 1".to_owned());
        }
        if end < start {
            return Err(format!(
                "end_line ({end_line}) must be greater than or equal to start_line ({start_line})"
            ));
        }
        if end > lines.len() {
            return Err(format!(
                "line range {start_line}-{end_line} is out of bounds; `{path}` has {} line(s)",
                lines.len()
            ));
        }

        let replacement: Vec<String> = if content.is_empty() {
            Vec::new()
        } else {
            content.lines().map(str::to_owned).collect()
        };
        lines.splice(start - 1..end, replacement);

        let mut updated = lines.join("\n");
        if trailing_newline && !updated.is_empty() {
            updated.push('\n');
        }
        let line_count = lines.len();
        self.files.insert(path.clone(), updated);
        Ok(format!(
            "Replaced lines {start_line}-{end_line} in `{path}`; file now has {line_count} line(s)."
        ))
    }

    /// Replaces a literal substring only when it occurs exactly once.
    fn edit_pattern(
        &mut self,
        path: &str,
        pattern: &str,
        replacement: &str,
    ) -> Result<String, String> {
        let path = Self::normalize_path(path)?;
        if pattern.is_empty() {
            return Err("pattern must not be empty".to_owned());
        }
        let original = self
            .files
            .get(&path)
            .cloned()
            .ok_or_else(|| format!("No such file: `{path}`"))?;

        match original.matches(pattern).count() {
            0 => Err(format!("Pattern not found in `{path}`.")),
            1 => {
                self.files
                    .insert(path.clone(), original.replacen(pattern, replacement, 1));
                Ok(format!(
                    "Replaced the single match of the pattern in `{path}`."
                ))
            }
            count => Err(format!(
                "Pattern matches {count} times in `{path}`; refusing to edit because exactly one match is required."
            )),
        }
    }
}

/// Incremental decoder state keyed by agent entity.
struct ActiveTurn {
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
struct AgentRuntime {
    next_turn_id: u64,
    active: HashMap<Entity, ActiveTurn>,
}

/// Uniform wrapper over the three provider stream decoders.
enum Decoder {
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

fn main() {
    let provider = Provider::from_env();
    let model_name = env::var("MODEL").unwrap_or_else(|_| "qwen3.6-35b-a3b".to_owned());
    let endpoint = env::var("ENDPOINT").unwrap_or_else(|_| provider.endpoint().to_owned());
    let token = env::var("API_TOKEN").unwrap_or_default();
    let session_id = session_id_from_env();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "LLM Agents".into(),
                resolution: WindowResolution::new(1120, 800),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((TokioTasksPlugin::default(), TabNavigationPlugin))
        .add_plugins(BrpExtrasPlugin) // will listen on BRP default port 15702
        .insert_resource(ClearColor(Color::srgb(0.055, 0.065, 0.085)))
        .insert_resource(AppConfig {
            provider,
            model_name,
            endpoint,
            token,
            session_id,
        })
        .init_resource::<AgentRuntime>()
        .init_resource::<AgentIndex>()
        .init_resource::<VirtualFs>()
        .init_non_send::<JsRuntimes>()
        .add_message::<StartAgentTurn>()
        .add_message::<NewAgentRequested>()
        .add_message::<CancelAgentRequest>()
        .add_message::<RetryAgentTurn>()
        .add_message::<StreamChunk>()
        .add_message::<StreamEnded>()
        .add_message::<StreamFailed>()
        .add_message::<AgentStreamEvent>()
        .add_message::<AgentTurnFinished>()
        .add_message::<AgentFailed>()
        .add_message::<RunToolCalls>()
        .add_message::<SendAgentMessage>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                create_requested_agents,
                sync_agent_index,
                sync_active_agent,
                sync_retry_buttons,
                cancel_agent_turn,
                retry_agent_turn,
                submit_input,
                deliver_pending_messages,
                decode_stream_chunks,
                handle_stream_events,
                handle_turn_finished,
                execute_tool_calls,
                handle_send_agent_message,
                start_agent_turn,
                handle_agent_failed,
            )
                .chain(),
        )
        .add_systems(PostUpdate, auto_scroll.after(UiSystems::Layout))
        .run();
}

fn setup(mut commands: Commands, config: Res<AppConfig>) {
    commands.spawn((Camera2d, IsDefaultUiCamera));

    let root = commands
        .spawn((
            Node {
                width: percent(100.),
                height: percent(100.),
                flex_direction: FlexDirection::Column,
                padding: px(18.).all(),
                row_gap: px(12.),
                ..default()
            },
            BackgroundColor(Color::srgb(0.055, 0.065, 0.085)),
        ))
        .id();

    let header = commands
        .spawn(Node {
            width: percent(100.),
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                Text::new("LLM Agents"),
                TextFont {
                    font_size: FontSize::Px(24.),
                    ..default()
                },
                TextColor(Color::srgb(0.92, 0.94, 0.98)),
            ));
            parent.spawn((
                Text::new(format!(
                    "{} | {} | {}",
                    config.provider.label(),
                    config.model_name,
                    config.endpoint
                )),
                TextFont {
                    font_size: FontSize::Px(13.),
                    ..default()
                },
                TextColor(Color::srgb(0.55, 0.60, 0.70)),
            ));
        })
        .id();

    let tabs = commands
        .spawn((
            Node {
                flex_grow: 1.,
                flex_direction: FlexDirection::Row,
                flex_wrap: FlexWrap::Wrap,
                align_items: AlignItems::Center,
                column_gap: px(8.),
                row_gap: px(6.),
                ..default()
            },
            TabGroup::new(0),
        ))
        .id();

    let plus = commands
        .spawn((
            Button,
            Node {
                min_width: px(38.),
                height: px(34.),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(1.).all(),
                border_radius: BorderRadius::all(px(8.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.14, 0.17, 0.22)),
            BorderColor::all(Color::srgb(0.30, 0.36, 0.46)),
            TabIndex(0),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("+"),
                TextFont {
                    font_size: FontSize::Px(20.),
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.90, 0.98)),
            ));
        })
        .id();
    commands
        .entity(plus)
        .observe(|_: On<Activate>, mut commands: Commands| {
            commands.write_message(NewAgentRequested);
        });

    let tab_bar = commands
        .spawn(Node {
            width: percent(100.),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(8.),
            ..default()
        })
        .add_children(&[tabs, plus])
        .id();

    let panels = commands
        .spawn((
            Node {
                width: percent(100.),
                flex_basis: px(0.),
                flex_grow: 1.,
                min_height: px(0.),
                ..default()
            },
            TabGroup::new(1),
        ))
        .id();

    commands
        .entity(root)
        .add_children(&[header, tab_bar, panels]);

    let layout = AgentLayout {
        tabs,
        panels,
        next_number: 2,
    };
    let agent = create_agent(&mut commands, &layout, &config, "LLM Agent 1".to_owned(), 1);

    commands.insert_resource(layout);
    commands.insert_resource(ActiveAgent(Some(agent)));
}

fn create_requested_agents(
    mut requests: MessageReader<NewAgentRequested>,
    mut commands: Commands,
    mut layout: ResMut<AgentLayout>,
    config: Res<AppConfig>,
    mut active: ResMut<ActiveAgent>,
) {
    for _ in requests.read() {
        let number = layout.next_number;
        layout.next_number += 1;
        let name = format!("LLM Agent {number}");
        let agent = create_agent(&mut commands, &layout, &config, name, number as i32);
        active.0 = Some(agent);
    }
}

fn create_agent(
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

    let panel = create_agent_panel(commands, layout.panels, agent_entity, &name, config);
    let tab = create_agent_tab(commands, layout.tabs, agent_entity, &name, tab_index);

    commands.entity(agent_entity).insert(AgentView {
        panel: panel.panel,
        log: panel.log,
        input: panel.input,
        status: panel.status,
        stats: panel.stats,
        assistant: None,
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

struct PanelEntities {
    panel: Entity,
    log: Entity,
    input: Entity,
    status: Entity,
    stats: Entity,
}

fn create_agent_panel(
    commands: &mut Commands,
    panels_root: Entity,
    agent: Entity,
    name: &str,
    config: &AppConfig,
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
        .spawn(Node {
            width: percent(100.),
            flex_basis: px(0.),
            flex_grow: 1.,
            min_height: px(0.),
            flex_direction: FlexDirection::Column,
            row_gap: px(10.),
            padding: px(4.).all(),
            overflow: Overflow::scroll_y(),
            ..default()
        })
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
        .add_children(&[input, retry, stop])
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

    spawn_message_card(
        commands,
        log,
        "System",
        Color::srgb(0.55, 0.60, 0.70),
        format!(
            "{name} connected to {} at {}. Type a message and press Enter.",
            config.provider.label(),
            config.endpoint
        ),
        Color::srgb(0.09, 0.10, 0.13),
    );

    PanelEntities {
        panel,
        log,
        input,
        status,
        stats,
    }
}

fn create_agent_tab(
    commands: &mut Commands,
    tabs: Entity,
    agent: Entity,
    name: &str,
    tab_index: i32,
) -> Entity {
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
            parent.spawn((
                Text::new(name.to_owned()),
                TextFont {
                    font_size: FontSize::Px(14.),
                    ..default()
                },
                TextColor(Color::srgb(0.82, 0.87, 0.95)),
            ));
        })
        .id();

    commands.entity(tabs).add_children(&[tab]);
    tab
}

fn spawn_message_card(
    commands: &mut Commands,
    log: Entity,
    role: &str,
    role_color: Color,
    text: String,
    background: Color,
) -> Entity {
    let mut body = None;
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
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(role),
                TextFont {
                    font_size: FontSize::Px(13.),
                    ..default()
                },
                TextColor(role_color),
            ));
            body = Some(
                parent
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
                    .id(),
            );
        })
        .id();
    commands.entity(log).add_children(&[card]);
    body.expect("message body was spawned")
}

fn spawn_assistant_card(commands: &mut Commands, log: Entity) -> AssistantView {
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
            display: Display::None,
            flex_direction: FlexDirection::Column,
            row_gap: px(3.),
            padding: px(9.).all(),
            border_radius: BorderRadius::all(px(7.)),
            ..default()
        })
        .insert(BackgroundColor(Color::srgb(0.085, 0.095, 0.125)))
        .id();

    let thinking_label = commands
        .spawn((
            Text::new("Thinking"),
            TextFont {
                font_size: FontSize::Px(12.),
                ..default()
            },
            TextColor(Color::srgb(0.62, 0.66, 0.76)),
        ))
        .id();

    let thinking_body = commands
        .spawn((
            Text::new(""),
            TextFont {
                font_size: FontSize::Px(15.),
                ..default()
            },
            TextColor(Color::srgb(0.70, 0.74, 0.82)),
            TextLayout {
                linebreak: LineBreak::WordOrCharacter,
                ..default()
            },
            Node {
                width: percent(100.),
                ..default()
            },
        ))
        .id();

    let answer_body = commands
        .spawn((
            Text::new("..."),
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
        .add_children(&[thinking_label, thinking_body]);
    commands
        .entity(card)
        .add_children(&[role, thinking_container, answer_body]);
    commands.entity(log).add_children(&[card]);

    AssistantView {
        thinking_container,
        thinking_body,
        answer_body,
    }
}

// ---------------------------------------------------------------------------
// Systems: tabs and input
// ---------------------------------------------------------------------------

fn sync_agent_index(agents: Query<(Entity, &Agent)>, mut index: ResMut<AgentIndex>) {
    index.by_name.clear();
    for (entity, agent) in &agents {
        index.by_name.insert(agent.name.clone(), entity);
    }
}

fn sync_active_agent(
    mut active: ResMut<ActiveAgent>,
    mut focus: ResMut<InputFocus>,
    views: Query<(Entity, &AgentView)>,
    mut nodes: Query<&mut Node>,
    mut visibilities: Query<&mut Visibility>,
    mut tabs: Query<(&AgentTab, &mut BackgroundColor, &mut BorderColor)>,
) {
    let active_valid = active.0.is_some_and(|entity| views.get(entity).is_ok());
    if !active_valid {
        active.0 = views.iter().next().map(|(entity, _)| entity);
    }
    let Some(active_entity) = active.0 else {
        return;
    };

    for (entity, view) in &views {
        let active = entity == active_entity;
        if let Ok(mut node) = nodes.get_mut(view.panel) {
            node.display = if active { Display::Flex } else { Display::None };
        }
        if let Ok(mut visibility) = visibilities.get_mut(view.panel) {
            *visibility = if active {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
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
        && let Ok((_, view)) = views.get(active_entity)
    {
        focus.set(view.input, FocusCause::Navigated);
    }
}

fn sync_retry_buttons(
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

fn retry_agent_turn(
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

fn cancel_agent_turn(
    mut requests: MessageReader<CancelAgentRequest>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut runtime: ResMut<AgentRuntime>,
    mut texts: Query<&mut Text>,
    mut focus: ResMut<InputFocus>,
) {
    for request in requests.read() {
        if let Some(active) = runtime.active.remove(&request.agent) {
            let _ = active.cancel.send(true);
        }

        let Ok((mut agent, mut view)) = agents.get_mut(request.agent) else {
            continue;
        };

        // Clicking the Stop button moves input focus to the button; return it
        // to the text field whether or not there was a turn to cancel.
        focus.set(view.input, FocusCause::Navigated);

        if !agent.pending && agent.current_turn.is_none() {
            continue;
        }

        // Drop the unanswered user turn and any queued cross-agent messages so
        // stopping does not immediately start another request. Keep the user
        // text so Retry can resend it.
        agent.retry_message = rollback_pending_turn(&mut agent.history);
        agent.pending_messages.clear();
        agent.pending = false;
        agent.current_turn = None;
        agent.streaming_text.clear();
        agent.streaming_thinking.clear();

        if let Some(assistant) = view.assistant {
            set_text(&mut texts, assistant.answer_body, "Generation interrupted.");
        }
        view.scroll_to_bottom = true;
        set_text(&mut texts, view.status, "Interrupted");
    }
}

fn submit_input(
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
struct PendingDelivery {
    text: String,
    role: String,
    role_color: Color,
    background: Color,
    status: &'static str,
}

fn take_pending_message(agent: &mut Agent) -> Option<PendingDelivery> {
    match agent.pending_messages.pop_front()? {
        QueuedMessage::User(text) => Some(PendingDelivery {
            text,
            role: "You".to_owned(),
            role_color: Color::srgb(0.55, 0.75, 1.0),
            background: Color::srgb(0.12, 0.17, 0.26),
            status: "Starting request...",
        }),
        QueuedMessage::Agent { from, text } => Some(PendingDelivery {
            text: format!(
                "<system_message>[Message from {from}, answer with a tool \"send_message_to_agent\" if you need to reply]\n{text}</system_message>"
            ),
            role: format!("From {from}"),
            role_color: Color::srgb(0.78, 0.64, 1.0),
            background: Color::srgb(0.17, 0.13, 0.24),
            status: "Message received; starting request...",
        }),
    }
}

/// Moves waiting messages into `agent.history` at the current legal boundary.
///
/// When `all` is set every queued message is flushed (used after tool results,
/// where the agent is about to continue anyway); otherwise only the oldest
/// message is delivered so an idle agent answers one message per turn.
/// Returns how many messages were inserted.
fn flush_pending_messages(
    agent: &mut Agent,
    view: &mut AgentView,
    commands: &mut Commands,
    texts: &mut Query<&mut Text>,
    all: bool,
) -> usize {
    let mut combined = Vec::new();
    let mut status = None;

    while let Some(delivery) = take_pending_message(agent) {
        spawn_message_card(
            commands,
            view.log,
            &delivery.role,
            delivery.role_color,
            delivery.text.clone(),
            delivery.background,
        );
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

fn deliver_pending_messages(
    mut agents: Query<(Entity, &mut Agent, &mut AgentView)>,
    mut texts: Query<&mut Text>,
    mut start_turn: MessageWriter<StartAgentTurn>,
    mut commands: Commands,
) {
    for (entity, mut agent, mut view) in &mut agents {
        if agent.pending {
            continue;
        }

        if flush_pending_messages(&mut agent, &mut view, &mut commands, &mut texts, false) == 0 {
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

fn start_agent_turn(
    mut requests: MessageReader<StartAgentTurn>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut runtime: ResMut<AgentRuntime>,
    mut texts: Query<&mut Text>,
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

        let assistant = spawn_assistant_card(&mut commands, view.log);
        view.assistant = Some(assistant);
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

fn decode_stream_chunks(
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

fn write_stream_events(
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

fn handle_stream_events(
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut stream_events: MessageReader<AgentStreamEvent>,
    mut texts: Query<&mut Text>,
    mut nodes: Query<&mut Node>,
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
                if let Some(assistant) = view.assistant {
                    set_display(&mut nodes, assistant.thinking_container, Display::Flex);
                    set_text(
                        &mut texts,
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

fn handle_turn_finished(
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut finished: MessageReader<AgentTurnFinished>,
    mut texts: Query<&mut Text>,
    mut nodes: Query<&mut Node>,
    mut run_tools: MessageWriter<RunToolCalls>,
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
            "(thinking only - no visible text)".to_owned()
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
                set_text(&mut texts, assistant.thinking_body, thinking);
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
            set_text(&mut texts, view.status, "Empty response");
        } else {
            agent.history.extend(message.entries.iter().cloned());
            agent.streaming_text.clear();
            agent.streaming_thinking.clear();

            if let Some(tool_call) = tool_call {
                agent.pending = true;
                set_text(&mut texts, view.status, "Running tools...");
                run_tools.write(RunToolCalls {
                    agent: message.agent,
                    turn_id: message.turn_id,
                    tool_call,
                });
            } else {
                agent.pending = false;
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

fn execute_tool_calls(
    mut requests: MessageReader<RunToolCalls>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut index: ResMut<AgentIndex>,
    mut layout: ResMut<AgentLayout>,
    mut fs: ResMut<VirtualFs>,
    config: Res<AppConfig>,
    mut texts: Query<&mut Text>,
    mut send_agent_message: MessageWriter<SendAgentMessage>,
    mut start_turn: MessageWriter<StartAgentTurn>,
    mut js_runtimes: NonSendMut<JsRuntimes>,
    mut commands: Commands,
) {
    for request in requests.read() {
        let Ok((mut agent, mut view)) = agents.get_mut(request.agent) else {
            continue;
        };
        if agent.current_turn != Some(request.turn_id) || request.tool_call.is_empty() {
            continue;
        }

        let mut results = Vec::with_capacity(request.tool_call.len());
        for call in &request.tool_call {
            if call.tool_name == "spawn_agent" {
                let number = layout.next_number;
                layout.next_number += 1;
                let name = format!("LLM Agent {number}");
                let agent =
                    create_agent(&mut commands, &layout, &config, name.clone(), number as i32);
                index.by_name.insert(name.clone(), agent);
                results.push(ToolCallResult {
                    result: format!("Spawned `{name}` in a new tab."),
                    is_error: false,
                    tool_use_id: call.tool_use_id.clone(),
                });
            } else {
                results.push(execute_tool_call(
                    call,
                    request.agent,
                    &agent.name,
                    &index,
                    &mut fs,
                    &mut send_agent_message,
                    &mut js_runtimes,
                ));
            }
        }

        let summary = request
            .tool_call
            .iter()
            .zip(&results)
            .map(|(call, result)| {
                format!(
                    "{} [{}]\n{}",
                    call.tool_name,
                    if result.is_error { "error" } else { "ok" },
                    result.result
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        agent
            .history
            .push(Entry::ToolCallResult { tool_call: results });

        spawn_message_card(
            &mut commands,
            view.log,
            "Tools",
            Color::srgb(0.95, 0.75, 0.40),
            summary,
            Color::srgb(0.13, 0.115, 0.09),
        );
        view.scroll_to_bottom = true;
        set_text(&mut texts, view.status, "Tools finished; continuing...");

        // Force waiting user and cross-agent messages into the history now:
        // a completed tool round is a legal boundary, and leaving them queued
        // would delay them until the whole agent loop finishes.
        let inserted =
            flush_pending_messages(&mut agent, &mut view, &mut commands, &mut texts, true);
        if inserted > 0 {
            set_text(
                &mut texts,
                view.status,
                format!("Inserted {inserted} pending message(s); continuing..."),
            );
        }

        start_turn.write(StartAgentTurn {
            agent: request.agent,
        });
    }
}

fn handle_send_agent_message(
    mut messages: MessageReader<SendAgentMessage>,
    mut agents: Query<&mut Agent>,
) {
    for message in messages.read() {
        if let Ok(mut agent) = agents.get_mut(message.target) {
            agent.pending_messages.push_back(QueuedMessage::Agent {
                from: message.from.clone(),
                text: message.message.clone(),
            });
        }
    }
}

fn handle_agent_failed(
    mut failures: MessageReader<AgentFailed>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut texts: Query<&mut Text>,
) {
    for failure in failures.read() {
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
        agent.streaming_text.clear();
        agent.streaming_thinking.clear();
        view.scroll_to_bottom = true;
        set_text(&mut texts, view.status, "Request failed");
    }
}

// ---------------------------------------------------------------------------
// Transport task
// ---------------------------------------------------------------------------

async fn stream_transport(
    mut ctx: TaskContext,
    mut cancel: watch::Receiver<bool>,
    agent: Entity,
    turn_id: u64,
    endpoint: String,
    token: Option<String>,
    session_id: String,
    body: String,
) {
    if let Err(err) = stream_request(
        &mut ctx,
        &mut cancel,
        agent,
        turn_id,
        endpoint,
        token,
        session_id,
        body,
    )
    .await
    {
        let message = format!("{err:#}");
        ctx.run_on_main_thread(move |ctx| {
            ctx.world.write_message(StreamFailed {
                agent,
                turn_id,
                message,
            });
        })
        .await;
    }
}

#[allow(clippy::too_many_arguments)]
async fn stream_request(
    ctx: &mut TaskContext,
    cancel: &mut watch::Receiver<bool>,
    agent: Entity,
    turn_id: u64,
    endpoint: String,
    token: Option<String>,
    session_id: String,
    body: String,
) -> Result<(), anyhow::Error> {
    let client = reqwest::Client::new();
    let mut request = client
        .post(&endpoint)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::USER_AGENT, "bvcs_llm_harness/0.1")
        .header("x-opencode-session", session_id)
        .body(body);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }

    let res = request.send().await?;
    if !res.status().is_success() {
        anyhow::bail!("HTTP {}: {}", res.status(), res.text().await?);
    }

    let mut stream = res.bytes_stream();
    loop {
        if *cancel.borrow() {
            return Ok(());
        }

        let next = tokio::select! {
            next = stream.next() => next,
            changed = cancel.changed() => {
                let _ = changed;
                return Ok(());
            }
        };

        let Some(chunk) = next else {
            break;
        };
        let bytes = chunk?.to_vec();
        ctx.run_on_main_thread(move |ctx| {
            ctx.world.write_message(StreamChunk {
                agent,
                turn_id,
                bytes,
            });
        })
        .await;

        if *cancel.borrow() {
            return Ok(());
        }
    }

    if *cancel.borrow() {
        return Ok(());
    }

    ctx.run_on_main_thread(move |ctx| {
        ctx.world.write_message(StreamEnded { agent, turn_id });
    })
    .await;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "get_weather".into(),
            description: Some("Returns the current weather for a city.".into()),
            parameters: HashMap::from([(
                "city".to_owned(),
                ParameterDescription {
                    r#type: ParameterType::String,
                    description: "City name".into(),
                },
            )]),
        },
        Tool {
            name: "fs_read".into(),
            description: Some(
                "Read a file from the shared virtual flat filesystem. Every LLM Agent sees \
                 the same files. Paths are plain file names without directories."
                    .into(),
            ),
            parameters: HashMap::from([(
                "path".to_owned(),
                ParameterDescription {
                    r#type: ParameterType::String,
                    description: "File name".into(),
                },
            )]),
        },
        Tool {
            name: "fs_ls".into(),
            description: Some(
                "List files in the shared virtual flat filesystem with their sizes and line \
                 counts. Every LLM Agent sees the same files."
                    .into(),
            ),
            parameters: HashMap::new(),
        },
        Tool {
            name: "fs_write".into(),
            description: Some(
                "Create or overwrite a file in the shared virtual flat filesystem. Every LLM \
                 Agent sees the same files. Paths are plain file names without directories."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "path".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "File name".into(),
                    },
                ),
                (
                    "content".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Complete new file content".into(),
                    },
                ),
            ]),
        },
        Tool {
            name: "fs_edit_lines".into(),
            description: Some(
                "Replace an inclusive, 1-based line range in a shared virtual file. An empty \
                 content string deletes the selected lines."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "path".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "File name".into(),
                    },
                ),
                (
                    "start_line".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "First line to replace (1-based)".into(),
                    },
                ),
                (
                    "end_line".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Last line to replace (inclusive)".into(),
                    },
                ),
                (
                    "content".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Replacement text; may contain newlines".into(),
                    },
                ),
            ]),
        },
        Tool {
            name: "fs_edit_pattern".into(),
            description: Some(
                "Replace a literal substring in a shared virtual file. Fails unless the \
                 pattern occurs exactly once, so ambiguous edits cannot happen."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "path".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "File name".into(),
                    },
                ),
                (
                    "pattern".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Literal text to find".into(),
                    },
                ),
                (
                    "replacement".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Replacement text (may be empty)".into(),
                    },
                ),
            ]),
        },
        Tool {
            name: "fs_delete".into(),
            description: Some("Delete a file from the shared virtual flat filesystem.".into()),
            parameters: HashMap::from([(
                "path".to_owned(),
                ParameterDescription {
                    r#type: ParameterType::String,
                    description: "File name".into(),
                },
            )]),
        },
        // Disabled: `read_file` lets the model read process command lines
        // (for example /proc/self/cmdline) and exfiltrate API tokens.
        // Tool {
        //     name: "read_file".into(),
        //     description: Some("Reads a UTF-8 text file from the user's computer.".into()),
        //     parameters: HashMap::from([(
        //         "file_path".to_owned(),
        //         ParameterDescription {
        //             r#type: ParameterType::String,
        //             description: "Absolute or relative path to the file".into(),
        //         },
        //     )]),
        // },
        Tool {
            name: "run_javascript".into(),
            description: Some(
                "Read a JavaScript file from the shared virtual filesystem and execute it in \
                 this agent's persistent Boa context. Returns printed output (print/console.log) \
                 followed by the last expression value. Variables and globals persist across runs."
                    .into(),
            ),
            parameters: HashMap::from([(
                "file".to_owned(),
                ParameterDescription {
                    r#type: ParameterType::String,
                    description: "Virtual file name containing JavaScript source".into(),
                },
            )]),
        },
        Tool {
            name: "eval_javascript".into(),
            description: Some(
                "Execute JavaScript in a persistent Boa runtime owned by this LLM Agent. \
                 Variables and globals persist across calls for the whole agent session. \
                 Returns printed output (print/console.log) followed by the last expression value."
                    .into(),
            ),
            parameters: HashMap::from([(
                "code".to_owned(),
                ParameterDescription {
                    r#type: ParameterType::String,
                    description: "JavaScript source code to execute".into(),
                },
            )]),
        },
        Tool {
            name: "eval_javascript_reset_context".into(),
            description: Some(
                "Drop JavaScript state for this LLM Agent. Pass `symbol` to remove one \
                 configurable global variable, or omit/empty `symbol` to drop the whole \
                 persistent context. Lexical let/const bindings cannot be removed individually."
                    .into(),
            ),
            parameters: HashMap::from([(
                "symbol".to_owned(),
                ParameterDescription {
                    r#type: ParameterType::String,
                    description: "Optional single identifier to drop; omit or use an empty \
                                  string to reset the whole context"
                        .into(),
                },
            )]),
        },
        Tool {
            name: "send_message_to_agent".into(),
            description: Some(
                "Send a message to another agent. The target agent receives it as a user \
                 message and starts its own response."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "agent".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Exact target agent title, for example `LLM Agent 2`".into(),
                    },
                ),
                (
                    "message".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Message to send to the target agent".into(),
                    },
                ),
            ]),
        },
        Tool {
            name: "list_agents".into(),
            description: Some("List every LLM Agent currently available to message.".into()),
            parameters: HashMap::new(),
        },
        Tool {
            name: "spawn_agent".into(),
            description: Some("Create a new LLM Agent in a new tab.".into()),
            parameters: HashMap::new(),
        },
    ]
}

fn execute_tool_call(
    call: &ToolCall,
    source_entity: Entity,
    source_agent: &str,
    index: &AgentIndex,
    fs: &mut VirtualFs,
    send_agent_message: &mut MessageWriter<SendAgentMessage>,
    js_runtimes: &mut JsRuntimes,
) -> ToolCallResult {
    let arguments = serde_json::from_str::<Value>(&call.toolcall_params).unwrap_or(Value::Null);

    let (result, is_error) = match call.tool_name.as_str() {
        "get_weather" => match arguments.get("city").and_then(Value::as_str) {
            Some(city) => (format!("{city}: 30 degrees Celsius"), false),
            None => ("missing required string parameter `city`".to_owned(), true),
        },
        "fs_read" => match arguments.get("path").and_then(Value::as_str) {
            Some(path) => match fs.read(path) {
                Ok(content) => (content, false),
                Err(err) => (err, true),
            },
            None => ("missing required string parameter `path`".to_owned(), true),
        },
        "fs_ls" => (fs.list(), false),
        "fs_write" => match (
            arguments.get("path").and_then(Value::as_str),
            arguments.get("content").and_then(Value::as_str),
        ) {
            (Some(path), Some(content)) => match fs.write(path, content.to_owned()) {
                Ok(message) => (message, false),
                Err(err) => (err, true),
            },
            _ => (
                "missing required string parameters `path` and `content`".to_owned(),
                true,
            ),
        },
        "fs_edit_lines" => match (
            arguments.get("path").and_then(Value::as_str),
            arguments.get("start_line").and_then(Value::as_u64),
            arguments.get("end_line").and_then(Value::as_u64),
            arguments.get("content").and_then(Value::as_str),
        ) {
            (Some(path), Some(start), Some(end), Some(content)) => {
                match fs.edit_lines(path, start, end, content.to_owned()) {
                    Ok(message) => (message, false),
                    Err(err) => (err, true),
                }
            }
            _ => (
                "missing required parameters `path`, `start_line`, `end_line`, `content`"
                    .to_owned(),
                true,
            ),
        },
        "fs_edit_pattern" => match (
            arguments.get("path").and_then(Value::as_str),
            arguments.get("pattern").and_then(Value::as_str),
            arguments.get("replacement").and_then(Value::as_str),
        ) {
            (Some(path), Some(pattern), Some(replacement)) => {
                match fs.edit_pattern(path, pattern, replacement) {
                    Ok(message) => (message, false),
                    Err(err) => (err, true),
                }
            }
            _ => (
                "missing required string parameters `path`, `pattern`, `replacement`".to_owned(),
                true,
            ),
        },
        "fs_delete" => match arguments.get("path").and_then(Value::as_str) {
            Some(path) => match fs.delete(path) {
                Ok(message) => (message, false),
                Err(err) => (err, true),
            },
            None => ("missing required string parameter `path`".to_owned(), true),
        },
        // Disabled together with the tool definition above: arbitrary file
        // reads leak secrets such as tokens from /proc/self/cmdline.
        // "read_file" => match arguments.get("file_path").and_then(Value::as_str) {
        //     Some(path) => match fs::read_to_string(path) {
        //         Ok(contents) => (contents, false),
        //         Err(err) => (err.to_string(), true),
        //     },
        //     None => (
        //         "missing required string parameter `file_path`".to_owned(),
        //         true,
        //     ),
        // },
        "run_javascript" => match arguments.get("file").and_then(Value::as_str) {
            Some(file) => match fs.read(file) {
                Ok(source) => match js_runtimes.eval(source_entity, &source) {
                    Ok(output) => (output, false),
                    Err(err) => (err, true),
                },
                Err(err) => (err, true),
            },
            None => ("missing required string parameter `file`".to_owned(), true),
        },
        "eval_javascript" => match arguments.get("code").and_then(Value::as_str) {
            Some(code) => match js_runtimes.eval(source_entity, code) {
                Ok(output) => (output, false),
                Err(err) => (err, true),
            },
            None => ("missing required string parameter `code`".to_owned(), true),
        },
        "eval_javascript_reset_context" => {
            let symbol = arguments
                .get("symbol")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if symbol.is_empty() {
                (js_runtimes.reset(source_entity), false)
            } else {
                match js_runtimes.drop_symbol(source_entity, symbol) {
                    Ok(message) => (message, false),
                    Err(message) => (message, true),
                }
            }
        }
        "list_agents" => {
            let mut names = index.by_name.keys().cloned().collect::<Vec<_>>();
            names.sort();
            let available = names
                .into_iter()
                .map(|name| {
                    if name == source_agent {
                        format!("{name} (you)")
                    } else {
                        name
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            (format!("Available agents: {available}"), false)
        }
        "send_message_to_agent" => {
            let target_name = arguments.get("agent").and_then(Value::as_str);
            let message = arguments.get("message").and_then(Value::as_str);
            match (target_name, message) {
                (Some(target_name), Some(_)) if target_name == source_agent => {
                    ("Cannot send a message to the same agent.".to_owned(), true)
                }
                (Some(target_name), Some(message)) => {
                    if let Some(target) = index.by_name.get(target_name) {
                        send_agent_message.write(SendAgentMessage {
                            target: *target,
                            from: source_agent.to_owned(),
                            message: message.to_owned(),
                        });
                        (
                            format!(
                                "Queued message for `{target_name}`. Continue your task or end turn to await reply."
                            ),
                            false,
                        )
                    } else {
                        let mut names = index.by_name.keys().cloned().collect::<Vec<_>>();
                        names.sort();
                        (
                            format!(
                                "No agent named `{target_name}`. Available agents: {}",
                                names.join(", ")
                            ),
                            true,
                        )
                    }
                }
                _ => (
                    "missing required string parameters `agent` and `message`".to_owned(),
                    true,
                ),
            }
        }
        other => (format!("Unknown tool: {other}"), true),
    };

    ToolCallResult {
        result,
        is_error,
        tool_use_id: call.tool_use_id.clone(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn request_body(
    provider: Provider,
    history: &[Entry],
    config: &ModelConfig,
) -> Result<String, anyhow::Error> {
    Ok(match provider {
        Provider::Anthropic => serde_json::to_string(
            &anthropic::to_anthropic_messages(history, config)
                .map_err(|errors| anyhow::anyhow!("history validation failed: {errors:?}"))?,
        )?,
        Provider::Completions => serde_json::to_string(
            &openai_completions::to_completions_input(history, config)
                .map_err(|errors| anyhow::anyhow!("history validation failed: {errors:?}"))?,
        )?,
        Provider::Responses => serde_json::to_string(
            &openai_responses::to_responses_input(history, config)
                .map_err(|errors| anyhow::anyhow!("history validation failed: {errors:?}"))?,
        )?,
    })
}

/// Remove the current user turn and anything generated after it. Returns the
/// removed user text so it can be offered through Retry.
fn rollback_pending_turn(history: &mut Vec<Entry>) -> Option<String> {
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

fn response_display_text(entries: &[Entry]) -> Option<String> {
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

fn response_thinking_text(entries: &[Entry]) -> Option<String> {
    let thinking = entries
        .iter()
        .flat_map(|entry| entry.response_thinking_blocks())
        .cloned()
        .collect::<Vec<Thinking>>();
    thinking_joined(&thinking)
}

fn set_text(texts: &mut Query<&mut Text>, entity: Entity, value: impl Into<String>) {
    if let Ok(mut text) = texts.get_mut(entity) {
        text.0 = value.into();
    }
}

fn set_display(nodes: &mut Query<&mut Node>, entity: Entity, display: Display) {
    if let Ok(mut node) = nodes.get_mut(entity) {
        node.display = display;
    }
}

/// Renders the fixed-width usage readout. Fields are padded so the text does
/// not shift as token counts grow (Bevy's default font is monospaced).
fn stats_text(usage: Option<&Usage>) -> String {
    fn field(value: Option<u64>) -> String {
        value.map_or_else(|| "-".to_owned(), |value| value.to_string())
    }

    let input = field(usage.and_then(|usage| usage.input_tokens));
    let output = field(usage.and_then(|usage| usage.output_tokens));
    let cached = field(usage.and_then(|usage| usage.cached_input_tokens));
    format!("in {input:>6}   out {output:>6}   cached {cached:>6}")
}

fn auto_scroll(
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
    fn javascript_state_persists_per_agent() {
        let mut runtimes = JsRuntimes::default();
        let agent = Entity::PLACEHOLDER;

        assert_eq!(
            runtimes.eval(agent, "var counter = 41; counter").unwrap(),
            "41"
        );
        assert_eq!(runtimes.eval(agent, "counter + 1").unwrap(), "42");
        assert_eq!(
            runtimes
                .eval(agent, "console.log('hello'); counter * 2")
                .unwrap(),
            "hello\n82"
        );
    }

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
    fn virtual_fs_supports_shared_file_lifecycle() {
        let mut fs = VirtualFs::default();
        assert!(fs.read("notes.txt").is_err());

        fs.write("notes.txt", "one\ntwo\nthree\n".into()).unwrap();
        assert_eq!(fs.read("notes.txt").unwrap(), "one\ntwo\nthree\n");
        let listing = fs.list();
        assert!(listing.contains("Files (1):"), "{listing}");
        assert!(
            listing.contains("notes.txt - 14 bytes, 3 line(s)"),
            "{listing}"
        );

        fs.edit_lines("notes.txt", 2, 2, "TWO".into()).unwrap();
        assert_eq!(fs.read("notes.txt").unwrap(), "one\nTWO\nthree\n");

        fs.edit_pattern("notes.txt", "TWO", "two").unwrap();
        assert_eq!(fs.read("notes.txt").unwrap(), "one\ntwo\nthree\n");

        fs.delete("notes.txt").unwrap();
        assert!(fs.read("notes.txt").is_err());
        assert_eq!(fs.list(), "Virtual filesystem is empty.");
    }

    #[test]
    fn virtual_fs_edits_are_unambiguous_and_bounded() {
        let mut fs = VirtualFs::default();
        fs.write("a.txt", "same same\nsecond".into()).unwrap();

        let err = fs
            .edit_pattern("a.txt", "same", "other")
            .expect_err("ambiguous pattern must fail");
        assert!(err.contains("2 times"), "{err}");

        assert!(fs.edit_pattern("a.txt", "missing", "x").is_err());
        assert!(fs.edit_lines("a.txt", 0, 1, "x".into()).is_err());
        assert!(fs.edit_lines("a.txt", 3, 3, "x".into()).is_err());
        assert!(fs.edit_lines("a.txt", 2, 1, "x".into()).is_err());
        assert!(fs.write("dir/file.txt", "x".into()).is_err());
        assert!(fs.read("..").is_err());
    }

    #[test]
    fn virtual_fs_line_edits_can_delete_lines() {
        let mut fs = VirtualFs::default();
        fs.write("d.txt", "a\nb\nc".into()).unwrap();

        fs.edit_lines("d.txt", 2, 2, String::new()).unwrap();
        assert_eq!(fs.read("d.txt").unwrap(), "a\nc");

        fs.edit_lines("d.txt", 1, 2, String::new()).unwrap();
        assert_eq!(fs.read("d.txt").unwrap(), "");
    }

    #[test]
    fn virtual_javascript_file_runs_in_agent_context() {
        let mut fs = VirtualFs::default();
        fs.write(
            "script.js",
            "var from_file = 6 * 7; print('ran'); from_file".into(),
        )
        .unwrap();
        let source = fs.read("script.js").unwrap();

        let mut runtimes = JsRuntimes::default();
        let agent = Entity::PLACEHOLDER;
        assert_eq!(runtimes.eval(agent, &source).unwrap(), "ran\n42");
        // The file runs in the same persistent context as eval_javascript.
        assert_eq!(runtimes.eval(agent, "from_file").unwrap(), "42");
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
    fn javascript_context_can_be_reset() {
        let mut runtimes = JsRuntimes::default();
        let agent = Entity::PLACEHOLDER;

        runtimes.eval(agent, "var value = 7").unwrap();
        runtimes.reset(agent);
        assert_eq!(runtimes.eval(agent, "typeof value").unwrap(), "undefined");
    }

    #[test]
    fn javascript_global_symbol_can_be_dropped() {
        let mut runtimes = JsRuntimes::default();
        let agent = Entity::PLACEHOLDER;

        runtimes.eval(agent, "globalThis.value = 7").unwrap();
        let message = runtimes.drop_symbol(agent, "value").unwrap();
        assert!(message.contains("Dropped"));
        assert_eq!(runtimes.eval(agent, "typeof value").unwrap(), "undefined");
    }

    #[test]
    fn javascript_lexical_binding_cannot_be_dropped_individually() {
        let mut runtimes = JsRuntimes::default();
        let agent = Entity::PLACEHOLDER;

        runtimes.eval(agent, "let value = 7").unwrap();
        let message = runtimes.drop_symbol(agent, "value").unwrap();
        assert!(message.contains("Could not drop"));
        assert_eq!(runtimes.eval(agent, "value").unwrap(), "7");
    }

    #[test]
    fn javascript_output_is_isolated_per_context() {
        let mut world = World::new();
        let first = world.spawn_empty().id();
        let second = world.spawn_empty().id();
        let mut runtimes = JsRuntimes::default();

        assert_eq!(
            runtimes.eval(first, "console.log('first')").unwrap(),
            "first"
        );
        assert_eq!(
            runtimes.eval(second, "console.log('second')").unwrap(),
            "second"
        );
        assert_eq!(
            runtimes.eval(first, "console.log('again'); 1 + 1").unwrap(),
            "again\n2"
        );
    }

    #[test]
    fn javascript_state_is_isolated_between_agents() {
        let mut world = World::new();
        let first = world.spawn_empty().id();
        let second = world.spawn_empty().id();
        let mut runtimes = JsRuntimes::default();

        runtimes.eval(first, "var value = 'first'").unwrap();
        assert_eq!(runtimes.eval(first, "value").unwrap(), "first");
        assert_eq!(runtimes.eval(second, "typeof value").unwrap(), "undefined");
    }
}
