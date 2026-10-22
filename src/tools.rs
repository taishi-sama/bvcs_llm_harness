//! Central tool registry and executor.
//!
//! Both LLM-driven tool calls and JavaScript `execute(name, args)` calls end up
//! in [`execute_tool`]. Tools that start JavaScript are deliberately not handled
//! here; callers dispatch those asynchronously to the owning JS context.

use std::collections::HashMap;

use bevy::prelude::{Commands, Entity, MessageWriter, Query};
use llm_api::generic_tool::{ParameterDescription, ParameterType, Tool};
use llm_history_model::history::Entry;
use serde_json::Value;

use crate::{
    agent::{Agent, AgentIndex, AgentLayout, MAX_AGENTS, create_agent},
    config::AppConfig,
    js_runtime::JsTask,
    messages::{RemoveAgentRequest, RenameAgentRequest, SendAgentMessage},
    virtual_fs::VirtualFs,
};

/// Where one tool call must run.
///
/// LLM tool calls are classified by [`classify_tool_call`]. JavaScript
/// `execute(name, args)` always goes to [`ToolTarget::Agent`], where
/// [`execute_tool`] rejects [`JS_TOOLS`], so an interpreter can never
/// recursively start another interpreter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolTarget {
    /// Execute immediately on the Bevy main loop (fast, world-state tools).
    Agent,
    /// Send to the agent's JavaScript thread and finish asynchronously.
    Interpreter,
}

pub(crate) fn classify_tool_call(name: &str) -> ToolTarget {
    if is_js_tool(name) {
        ToolTarget::Interpreter
    } else {
        ToolTarget::Agent
    }
}

/// Build the interpreter task for one JavaScript tool call.
pub(crate) fn interpreter_task(
    name: &str,
    arguments: &Value,
    fs: &VirtualFs,
) -> Result<JsTask, String> {
    match name {
        "run_javascript" => {
            let file = arguments
                .get("file")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing required string parameter `file`".to_owned())?;
            let source = fs.read(file)?;
            Ok(JsTask::Eval { source })
        }
        "eval_javascript" => {
            let code = arguments
                .get("code")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing required string parameter `code`".to_owned())?;
            Ok(JsTask::Eval {
                source: code.to_owned(),
            })
        }
        "eval_javascript_reset_context" => {
            let symbol = arguments
                .get("symbol")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_owned();
            Ok(JsTask::Reset {
                symbol: (!symbol.is_empty()).then_some(symbol),
            })
        }
        other => Err(format!("not a JavaScript tool: {other}")),
    }
}

/// Tools that run JavaScript. They are handled by the async JS runtime instead
/// of [`execute_tool`], which keeps JavaScript from recursively starting itself.
pub(crate) const JS_TOOLS: &[&str] = &[
    "run_javascript",
    "eval_javascript",
    "eval_javascript_reset_context",
];

pub(crate) fn is_js_tool(name: &str) -> bool {
    JS_TOOLS.contains(&name)
}

/// First message for an agent created with an initial prompt.
pub(crate) fn initial_task_message(creator: &str, prompt: &str) -> String {
    format!("You were created by {creator}, your task:\n{prompt}")
}

fn optional_u64(arguments: &Value, name: &str) -> Result<Option<u64>, String> {
    match arguments.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("`{name}` must be a non-negative integer")),
    }
}

/// Like [`optional_u64`], but negative values are meaningful (relative indices).
fn optional_i64(arguments: &Value, name: &str) -> Result<Option<i64>, String> {
    match arguments.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_i64()
            .map(Some)
            .ok_or_else(|| format!("`{name}` must be an integer")),
    }
}

/// Messages returned by `get_agent_history` when `amount` is omitted.
const DEFAULT_HISTORY_AMOUNT: usize = 20;
/// Upper bound on `amount`, so one call cannot dump an unbounded history.
const MAX_HISTORY_AMOUNT: usize = 1000;

/// Resolve the inclusive-exclusive slice of history to return.
///
/// - omitted `position` starts at the last `amount` entries
/// - a negative `position` counts back from the end (`-1` is the newest entry)
/// - otherwise `position` is a plain zero-based index
fn history_window(len: usize, position: Option<i64>, amount: Option<u64>) -> (usize, usize) {
    let amount = amount
        .unwrap_or(DEFAULT_HISTORY_AMOUNT as u64)
        .min(MAX_HISTORY_AMOUNT as u64) as usize;

    let start = match position {
        None => len.saturating_sub(amount),
        Some(position) if position < 0 => len.saturating_sub(position.unsigned_abs() as usize),
        Some(position) => (position as usize).min(len),
    };
    let end = start.saturating_add(amount).min(len);
    (start, end)
}

/// Snake-case name of the history entry kind, exposed to the model.
fn entry_type(entry: &Entry) -> &'static str {
    match entry {
        Entry::SystemPrompt { .. } => "system_prompt",
        Entry::SystemMessage { .. } => "system_message",
        Entry::UserMessage { .. } => "user_message",
        Entry::Response { .. } => "response",
        Entry::ResponseToolCall { .. } => "response_tool_call",
        Entry::ToolCallResult { .. } => "tool_call_result",
        Entry::Draft { .. } => "draft",
    }
}

/// Who produced the entry, as far as it can be recovered from its content.
fn entry_sender(entry: &Entry, agent_name: &str) -> String {
    match entry {
        Entry::SystemPrompt { .. } | Entry::SystemMessage { .. } => "system".to_owned(),
        Entry::UserMessage { content } => content
            .text
            .as_deref()
            .and_then(user_message_sender)
            .unwrap_or_else(|| "user".to_owned()),
        Entry::Response { .. } | Entry::ResponseToolCall { .. } | Entry::Draft { .. } => {
            agent_name.to_owned()
        }
        Entry::ToolCallResult { .. } => "tool".to_owned(),
    }
}

/// Recover the sender from the envelopes used for queued user messages.
fn user_message_sender(text: &str) -> Option<String> {
    const AGENT_PREFIX: &str = "<system_message>[Message from ";
    const TASK_PREFIX: &str = "You were created by ";
    const BACKGROUND_PREFIX: &str = "<system_message>[Background task ";

    let name = if let Some(rest) = text.strip_prefix(AGENT_PREFIX) {
        rest.split(',').next()
    } else if let Some(rest) = text.strip_prefix(TASK_PREFIX) {
        rest.split(", your task:").next()
    } else if text.starts_with(BACKGROUND_PREFIX) {
        return Some("background_task".to_owned());
    } else {
        return None;
    };

    name.map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
}

/// Look up an agent by title, with a helpful error listing what exists.
fn history_agent<'a>(
    name: &str,
    index: &AgentIndex,
    agents: &'a Query<&Agent>,
) -> Result<&'a Agent, String> {
    let Some(entity) = index.by_name.get(name) else {
        let mut names = index.by_name.keys().cloned().collect::<Vec<_>>();
        names.sort();
        return Err(format!(
            "No agent named `{name}`. Available agents: {}",
            names.join(", ")
        ));
    };
    agents
        .get(*entity)
        .map_err(|_| format!("Agent `{name}` no longer exists."))
}

/// Distinguish an omitted background flag (`None`) from an explicit falsey one
/// (`Some(false)`), so callers can opt out of automatic backgrounding.
pub(crate) fn background_setting(value: Option<&Value>) -> Option<bool> {
    match value {
        None | Some(Value::Null) => None,
        Some(value) => Some(is_truthy(Some(value))),
    }
}

/// Treat common model-generated spellings as boolean true.
pub(crate) fn is_truthy(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value.as_f64().is_some_and(|value| value != 0.0),
        Some(Value::String(value)) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "true" | "1" | "yes" | "on"
        ),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
        Some(Value::Null) | None => false,
    }
}

/// Every tool advertised to the model. This is the single source of truth for
/// tool names, descriptions, and parameter schemas.
pub(crate) fn tools() -> Vec<Tool> {
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
                 the same files. Paths are plain file names without directories. Use `seek` \
                 and `len` to read a byte range instead of the whole file."
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
                    "seek".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Optional byte offset to start reading from (default 0)"
                            .into(),
                    },
                ),
                (
                    "len".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Optional number of bytes to read (default: to end of file)"
                            .into(),
                    },
                ),
            ]),
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
                 followed by the last expression value. Variables and globals persist across runs. \
                 JavaScript code can call other tools with execute(name, arguments). Set \
                 run_in_background to true to continue immediately and receive the result as a \
                 later message. If omitted, evaluations running longer than the auto-background \
                 grace period are detached automatically; set it to false to always wait."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "file".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Virtual file name containing JavaScript source".into(),
                    },
                ),
                (
                    "run_in_background".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Boolean,
                        description: "Run asynchronously and deliver the result as a message"
                            .into(),
                    },
                ),
            ]),
        },
        Tool {
            name: "eval_javascript".into(),
            description: Some(
                "Execute JavaScript in a persistent Boa runtime owned by this LLM Agent. \
                 Variables and globals persist across calls for the whole agent session. \
                 Returns printed output (print/console.log) followed by the last expression value. \
                 JavaScript code can call other tools with execute(name, arguments). Set \
                 run_in_background to true to continue immediately and receive the result as a \
                 later message. If omitted, evaluations running longer than the auto-background \
                 grace period are detached automatically; set it to false to always wait."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "code".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "JavaScript source code to execute".into(),
                    },
                ),
                (
                    "run_in_background".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Boolean,
                        description: "Run asynchronously and deliver the result as a message"
                            .into(),
                    },
                ),
            ]),
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
            name: "get_agent_history".into(),
            description: Some(
                "List the messages in any agent's conversation history, including this agent's \
                 own. Accepts an exact agent name. Each item has `type`, `sender`, and \
                 `message_index`; read the full message with get_agent_message. By default the \
                 last 20 messages are returned."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "agent".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Exact agent title, or this agent's own name".into(),
                    },
                ),
                (
                    "position".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Optional message index to start from. Negative values \
                                      count back from the end (-1 is the newest message); omit \
                                      to start at the most recent messages."
                            .into(),
                    },
                ),
                (
                    "amount".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Optional number of messages to return (default 20)".into(),
                    },
                ),
            ]),
        },
        Tool {
            name: "get_agent_message".into(),
            description: Some(
                "Read one message from an agent's conversation history, serialized as JSON. Use \
                 the `message_index` values from get_agent_history."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "agent".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Exact agent title, or this agent's own name".into(),
                    },
                ),
                (
                    "index".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Message index from get_agent_history".into(),
                    },
                ),
            ]),
        },
        Tool {
            name: "spawn_agent".into(),
            description: Some(
                "Create a new LLM Agent in a new tab. Optional `name` sets its title. If \
                 `prompt` is provided, the new agent immediately starts on that task; otherwise \
                 it waits for a message."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "name".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Optional unique agent title, for example `Researcher`".into(),
                    },
                ),
                (
                    "prompt".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Optional initial task for the new agent".into(),
                    },
                ),
            ]),
        },
        Tool {
            name: "rename_self".into(),
            description: Some(
                "Change this agent's local name/title. The new name must not already be used \
                 by another agent; tabs, index lookups, and later messages use it immediately."
                    .into(),
            ),
            parameters: HashMap::from([(
                "name".to_owned(),
                ParameterDescription {
                    r#type: ParameterType::String,
                    description: "New unique name for this agent".into(),
                },
            )]),
        },
        Tool {
            name: "end_agent".into(),
            description: Some(
                "Permanently end this agent. Its tab, panel, conversation, and JavaScript \
                 context are removed. This cannot be undone."
                    .into(),
            ),
            parameters: HashMap::new(),
        },
    ]
}

/// Execute one non-JavaScript tool.
///
/// Returns `(message, is_error)`. JavaScript tools are rejected here so that a
/// JS `execute` call can never recursively start another JS evaluation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_tool(
    name: &str,
    arguments: &Value,
    source_entity: Entity,
    source_agent: &str,
    index: &mut AgentIndex,
    agents: &Query<&Agent>,
    layout: &mut AgentLayout,
    fs: &mut VirtualFs,
    config: &AppConfig,
    commands: &mut Commands,
    send_agent_message: &mut MessageWriter<SendAgentMessage>,
    remove_agent: &mut MessageWriter<RemoveAgentRequest>,
    rename_agent: &mut MessageWriter<RenameAgentRequest>,
) -> (String, bool) {
    match name {
        "get_weather" => match arguments.get("city").and_then(Value::as_str) {
            Some(city) => (format!("{city}: 30 degrees Celsius"), false),
            None => ("missing required string parameter `city`".to_owned(), true),
        },
        "fs_read" => match arguments.get("path").and_then(Value::as_str) {
            Some(path) => match (
                optional_u64(arguments, "seek"),
                optional_u64(arguments, "len"),
            ) {
                (Ok(seek), Ok(len)) => match fs.read_range(path, seek, len) {
                    Ok(content) => (content, false),
                    Err(err) => (err, true),
                },
                (Err(err), _) | (_, Err(err)) => (err, true),
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
        "get_agent_history" => {
            let Some(target_name) = arguments.get("agent").and_then(Value::as_str) else {
                return ("missing required string parameter `agent`".to_owned(), true);
            };
            let target = match history_agent(target_name, index, agents) {
                Ok(target) => target,
                Err(err) => return (err, true),
            };
            let position = match optional_i64(arguments, "position") {
                Ok(position) => position,
                Err(err) => return (err, true),
            };
            let amount = match optional_u64(arguments, "amount") {
                Ok(amount) => amount,
                Err(err) => return (err, true),
            };

            let history = target.history();
            let (start, end) = history_window(history.len(), position, amount);
            let items = history[start..end]
                .iter()
                .enumerate()
                .map(|(offset, entry)| {
                    serde_json::json!({
                        "type": entry_type(entry),
                        "sender": entry_sender(entry, target.name()),
                        "message_index": start + offset,
                    })
                })
                .collect::<Vec<_>>();

            match serde_json::to_string_pretty(&items) {
                Ok(json) => (json, false),
                Err(err) => (format!("could not serialize history: {err}"), true),
            }
        }
        "get_agent_message" => {
            let Some(target_name) = arguments.get("agent").and_then(Value::as_str) else {
                return ("missing required string parameter `agent`".to_owned(), true);
            };
            let target = match history_agent(target_name, index, agents) {
                Ok(target) => target,
                Err(err) => return (err, true),
            };
            let Some(message_index) = arguments.get("index").and_then(Value::as_u64) else {
                return (
                    "missing required non-negative integer parameter `index`".to_owned(),
                    true,
                );
            };

            let history = target.history();
            let Some(entry) = history.get(message_index as usize) else {
                return (
                    format!(
                        "index {message_index} is out of range; `{target_name}` has {} messages (0..{})",
                        history.len(),
                        history.len().saturating_sub(1)
                    ),
                    true,
                );
            };

            match serde_json::to_string_pretty(entry) {
                Ok(json) => (json, false),
                Err(err) => (format!("could not serialize message: {err}"), true),
            }
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
                            verbatim: false,
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
        "rename_self" => {
            let Some(new_name) = arguments
                .get("name")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|name| !name.is_empty())
            else {
                return (
                    "missing required non-empty string parameter `name`".to_owned(),
                    true,
                );
            };

            if new_name == source_agent {
                return ("Name unchanged.".to_owned(), false);
            }
            if index.by_name.contains_key(new_name) {
                return (format!("An agent named `{new_name}` already exists."), true);
            }

            // Reserve the name immediately so two renames in one tool round
            // cannot both claim it.
            index.by_name.remove(source_agent);
            index.by_name.insert(new_name.to_owned(), source_entity);
            rename_agent.write(RenameAgentRequest {
                agent: source_entity,
                name: new_name.to_owned(),
            });
            (format!("Renamed this agent to `{new_name}`."), false)
        }
        "end_agent" => {
            remove_agent.write(RemoveAgentRequest {
                agent: source_entity,
            });
            ("Ending this agent...".to_owned(), false)
        }
        "spawn_agent" => {
            if index.by_name.len() >= MAX_AGENTS {
                return (
                    format!("Agent limit reached ({MAX_AGENTS}); remove an agent first."),
                    true,
                );
            }

            let requested_name = arguments
                .get("name")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|name| !name.is_empty());
            let prompt = arguments
                .get("prompt")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|prompt| !prompt.is_empty());

            if let Some(name) = requested_name
                && index.by_name.contains_key(name)
            {
                return (format!("An agent named `{name}` already exists."), true);
            }

            let mut number = layout.next_number;
            let name = if let Some(name) = requested_name {
                name.to_owned()
            } else {
                let mut candidate = format!("LLM Agent {number}");
                while index.by_name.contains_key(&candidate) {
                    number += 1;
                    candidate = format!("LLM Agent {number}");
                }
                candidate
            };
            layout.next_number = number + 1;

            let agent = create_agent(commands, layout, config, name.clone(), number as i32);
            index.by_name.insert(name.clone(), agent);

            if let Some(prompt) = prompt {
                // Delivered verbatim (no cross-agent envelope), so the new
                // agent's first user message starts exactly with this text.
                send_agent_message.write(SendAgentMessage {
                    target: agent,
                    from: source_agent.to_owned(),
                    message: initial_task_message(source_agent, prompt),
                    verbatim: true,
                });
                (
                    format!("Spawned `{name}` in a new tab with its initial task."),
                    false,
                )
            } else {
                (format!("Spawned `{name}` in a new tab."), false)
            }
        }
        other if is_js_tool(other) => (
            format!(
                "tool `{other}` cannot be called from JavaScript; recursive JavaScript execution is disabled"
            ),
            true,
        ),
        other => (format!("Unknown tool: {other}"), true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn javascript_tools_route_to_the_interpreter() {
        for name in JS_TOOLS {
            assert_eq!(classify_tool_call(name), ToolTarget::Interpreter);
        }
        assert_eq!(classify_tool_call("fs_read"), ToolTarget::Agent);
        assert_eq!(classify_tool_call("spawn_agent"), ToolTarget::Agent);
        assert_eq!(classify_tool_call("end_agent"), ToolTarget::Agent);
    }

    #[test]
    fn initial_task_message_has_the_required_prefix() {
        assert_eq!(
            initial_task_message("LLM Agent 1", "Summarize the notes."),
            "You were created by LLM Agent 1, your task:\nSummarize the notes."
        );
    }

    #[test]
    fn fs_read_exposes_seek_and_len() {
        let tools = tools();
        let fs_read = tools
            .iter()
            .find(|tool| tool.name == "fs_read")
            .expect("fs_read should exist");
        assert!(fs_read.parameters.contains_key("seek"));
        assert!(fs_read.parameters.contains_key("len"));

        assert_eq!(
            optional_u64(&serde_json::json!({ "seek": 3 }), "seek").unwrap(),
            Some(3)
        );
        assert!(optional_u64(&serde_json::json!({ "seek": -1 }), "seek").is_err());
        assert_eq!(optional_u64(&serde_json::json!({}), "len").unwrap(), None);
    }

    #[test]
    fn javascript_tools_expose_a_background_flag() {
        let tools = tools();
        for name in ["run_javascript", "eval_javascript"] {
            let tool = tools
                .iter()
                .find(|tool| tool.name == name)
                .expect("JavaScript tool should exist");
            assert!(
                tool.parameters.contains_key("run_in_background"),
                "{name} should expose run_in_background"
            );
        }
    }

    #[test]
    fn background_flag_accepts_true_ish_values() {
        assert!(is_truthy(Some(&Value::Bool(true))));
        assert!(is_truthy(Some(&serde_json::json!(1))));
        assert!(is_truthy(Some(&serde_json::json!("true"))));
        assert!(is_truthy(Some(&serde_json::json!("YES"))));
        assert!(!is_truthy(Some(&Value::Bool(false))));
        assert!(!is_truthy(Some(&serde_json::json!(0))));
        assert!(!is_truthy(Some(&serde_json::json!("false"))));
        assert!(!is_truthy(None));
    }

    #[test]
    fn background_setting_distinguishes_absent_from_false() {
        assert_eq!(background_setting(None), None);
        assert_eq!(background_setting(Some(&Value::Bool(true))), Some(true));
        assert_eq!(background_setting(Some(&Value::Bool(false))), Some(false));
        assert_eq!(
            background_setting(Some(&serde_json::json!("false"))),
            Some(false)
        );
    }

    #[test]
    fn interpreter_tasks_only_accept_javascript_tools() {
        let fs = VirtualFs::default();

        assert!(interpreter_task("fs_read", &Value::Null, &fs).is_err());
        assert!(matches!(
            interpreter_task(
                "eval_javascript",
                &serde_json::json!({ "code": "1 + 1" }),
                &fs
            ),
            Ok(JsTask::Eval { .. })
        ));
        assert!(matches!(
            interpreter_task(
                "eval_javascript_reset_context",
                &serde_json::json!({ "symbol": "" }),
                &fs
            ),
            Ok(JsTask::Reset { symbol: None })
        ));
    }
}

#[cfg(test)]
mod rename_tool_tests {
    use super::*;

    #[test]
    fn rename_self_tool_exists() {
        let tools = tools();
        let tool = tools
            .iter()
            .find(|tool| tool.name == "rename_self")
            .expect("rename_self should exist");
        assert!(tool.parameters.contains_key("name"));
    }
}

#[cfg(test)]
mod history_tool_tests {
    use super::*;
    use llm_history_model::history::MessageContent;

    fn user_text(text: &str) -> Entry {
        Entry::UserMessage {
            content: MessageContent {
                text: Some(text.to_owned()),
                image: None,
            },
        }
    }

    #[test]
    fn history_window_defaults_to_the_last_twenty_messages() {
        assert_eq!(history_window(100, None, None), (80, 100));
        assert_eq!(history_window(5, None, None), (0, 5));
        assert_eq!(history_window(0, None, None), (0, 0));
    }

    #[test]
    fn history_window_supports_negative_and_positive_positions() {
        assert_eq!(history_window(100, Some(-3), None), (97, 100));
        assert_eq!(history_window(100, Some(-3), Some(1)), (97, 98));
        assert_eq!(history_window(100, Some(10), Some(5)), (10, 15));
        assert_eq!(history_window(100, Some(98), Some(5)), (98, 100));
        assert_eq!(history_window(100, Some(200), Some(5)), (100, 100));
        assert_eq!(history_window(100, Some(-200), Some(5)), (0, 5));
        assert_eq!(history_window(100, None, Some(0)), (100, 100));
    }

    #[test]
    fn history_window_caps_the_amount() {
        let (start, end) = history_window(10_000, Some(0), Some(u64::MAX));
        assert_eq!((start, end), (0, MAX_HISTORY_AMOUNT));
    }

    #[test]
    fn history_tools_are_advertised() {
        let tools = tools();
        let history = tools
            .iter()
            .find(|tool| tool.name == "get_agent_history")
            .expect("get_agent_history should exist");
        assert!(history.parameters.contains_key("agent"));
        assert!(history.parameters.contains_key("position"));
        assert!(history.parameters.contains_key("amount"));

        let message = tools
            .iter()
            .find(|tool| tool.name == "get_agent_message")
            .expect("get_agent_message should exist");
        assert!(message.parameters.contains_key("agent"));
        assert!(message.parameters.contains_key("index"));
    }

    #[test]
    fn entry_sender_recovers_cross_agent_envelopes() {
        assert_eq!(
            entry_sender(
                &user_text("<system_message>[Message from LLM Agent 2, answer with a tool]\nhi"),
                "LLM Agent 1"
            ),
            "LLM Agent 2"
        );
        assert_eq!(
            entry_sender(
                &user_text("You were created by LLM Agent 1, your task:\nwork"),
                "Researcher"
            ),
            "LLM Agent 1"
        );
        assert_eq!(
            entry_sender(
                &user_text("<system_message>[Background task 3 completed]\nok"),
                "LLM Agent 1"
            ),
            "background_task"
        );
        assert_eq!(
            entry_sender(&user_text("plain prompt"), "LLM Agent 1"),
            "user"
        );
        assert_eq!(
            entry_sender(&Entry::response_text("answer"), "LLM Agent 1"),
            "LLM Agent 1"
        );
        assert_eq!(
            entry_sender(
                &Entry::SystemPrompt {
                    content: "x".into()
                },
                "LLM Agent 1"
            ),
            "system"
        );
    }

    #[test]
    fn entry_type_uses_snake_case_names() {
        assert_eq!(entry_type(&user_text("hi")), "user_message");
        assert_eq!(
            entry_type(&Entry::SystemPrompt {
                content: "x".into()
            }),
            "system_prompt"
        );
        assert_eq!(entry_type(&Entry::response_text("answer")), "response");
        assert_eq!(entry_type(&Entry::draft()), "draft");
        assert_eq!(
            entry_type(&Entry::ToolCallResult { tool_call: vec![] }),
            "tool_call_result"
        );
    }
}
