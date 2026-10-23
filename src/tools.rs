//! Tool catalog, registry, and executor.
//!
//! `definitions` is the single source of truth for names, descriptions, and
//! parameter schemas. `catalog` decides how each one is exposed, and `registry`
//! spawns one entity per tool so the live catalog is an ordinary ECS query.
//! Both LLM-driven tool calls and JavaScript `execute(name, args)` calls end up
//! in [`execute_tool`]. Tools that start JavaScript are deliberately not handled
//! here; callers dispatch those asynchronously to the owning JS context.

pub(crate) mod manifest;
pub(crate) mod mirror;
pub(crate) mod rescan;
pub(crate) mod schedule;
pub(crate) mod script;
pub(crate) mod tree;

mod agent_tools;
mod catalog;
mod fs;
mod registry;
mod search;
mod world;

use std::collections::HashMap;

use bevy::prelude::Query;
use llm_api::generic_tool::{ParameterDescription, ParameterType, Tool};
use llm_history_model::history::Entry;
use serde_json::Value;

use crate::{
    agent::{Agent, AgentIndex},
    js_runtime::JsTask,
    virtual_fs::VirtualFs,
};

pub(crate) use catalog::{AdvertisedTools, advertised_tools, register_script_tools};
pub(crate) use mirror::{
    MirrorDebounce, MirrorStatus, ToolsDir, flush_tool_mirror, seed_tool_tree,
};
pub(crate) use registry::{CallContext, Execution, find, register_static_tools};
pub(crate) use rescan::{ToolRegistryStatus, reconcile_script_tools};
pub(crate) use schedule::background_started_message;
pub(crate) use script::{ScriptHostRequest, ScriptRuntime, ScriptTask, ToolSourceCache};
pub(crate) use world::ToolWorld;

/// Build the interpreter task for one JavaScript tool call.
pub(crate) fn interpreter_task(
    name: &str,
    arguments: &Value,
    fs: &VirtualFs,
    depth: u32,
    root: u64,
) -> Result<JsTask, String> {
    match name {
        "run_javascript" => {
            let file = arguments
                .get("file")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing required string parameter `file`".to_owned())?;
            let source = fs.read(file)?;
            Ok(JsTask::Eval {
                source,
                depth,
                root,
            })
        }
        "eval_javascript" => {
            let code = arguments
                .get("code")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing required string parameter `code`".to_owned())?;
            Ok(JsTask::Eval {
                source: code.to_owned(),
                depth,
                root,
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

pub(crate) fn optional_u64(arguments: &Value, name: &str) -> Result<Option<u64>, String> {
    match arguments.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("`{name}` must be a non-negative integer")),
    }
}

/// Required counterpart to [`optional_u64`].
pub(crate) fn required_u64(arguments: &Value, name: &str) -> Result<u64, String> {
    optional_u64(arguments, name)?
        .ok_or_else(|| format!("missing required integer parameter `{name}`"))
}

/// Like [`optional_u64`], but negative values are meaningful (relative indices).
pub(crate) fn optional_i64(arguments: &Value, name: &str) -> Result<Option<i64>, String> {
    match arguments.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_i64()
            .map(Some)
            .ok_or_else(|| format!("`{name}` must be an integer")),
    }
}

/// Boolean option with a default when the argument is omitted or null.
pub(crate) fn bool_option(arguments: &Value, name: &str, default: bool) -> bool {
    match arguments.get(name) {
        None | Some(Value::Null) => default,
        Some(value) => is_truthy(Some(value)),
    }
}

/// Default and upper bounds for `fs_grep` output.
pub(crate) const DEFAULT_GREP_MATCHES: u64 = 200;
pub(crate) const MAX_GREP_MATCHES: u64 = 5_000;
pub(crate) const MAX_GREP_CONTEXT: u64 = 100;

/// Messages returned by `get_agent_history` when `amount` is omitted.
pub(crate) const DEFAULT_HISTORY_AMOUNT: usize = 20;
/// Upper bound on `amount`, so one call cannot dump an unbounded history.
pub(crate) const MAX_HISTORY_AMOUNT: usize = 1000;

/// Resolve the inclusive-exclusive slice of history to return.
///
/// - omitted `position` starts at the last `amount` entries
/// - a negative `position` counts back from the end (`-1` is the newest entry)
/// - otherwise `position` is a plain zero-based index
pub(crate) fn history_window(
    len: usize,
    position: Option<i64>,
    amount: Option<u64>,
) -> (usize, usize) {
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
pub(crate) fn entry_type(entry: &Entry) -> &'static str {
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
pub(crate) fn entry_sender(entry: &Entry, agent_name: &str) -> String {
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
pub(crate) fn user_message_sender(text: &str) -> Option<String> {
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

/// Every tool definition. This is the single source of truth for names,
/// descriptions, and parameter schemas; exposure is decided by `catalog`.
pub(crate) fn definitions() -> Vec<Tool> {
    vec![
        crate::tools::rescan::rescan_tool_definition(),
        Tool {
            name: "execute".into(),
            description: Some(
                "Call any registered tool by name. Use search_tools to find names. \
                 `args` is the argument object for the target tool. Optional \
                 `background_after` is seconds until the call detaches: 0 detaches \
                 immediately, a negative value never detaches, and omitting it uses \
                 the target tool's default. A detached call returns a task id and \
                 reports its result later as a message."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "tool_name".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Name of the tool to call".into(),
                        required: true,
                    },
                ),
                (
                    "args".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Object,
                        description: "Arguments object passed to the target tool (default empty)"
                            .into(),
                        required: false,
                    },
                ),
                (
                    "background_after".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Optional seconds until the call detaches; 0 = now,                                       negative = never"
                            .into(),
                        required: false,
                    },
                ),
            ]),
        },
        Tool {
            name: "search_tools".into(),
            description: Some(
                "Search registered tools by name and/or description and return a compact \
                 signature per match: `name(required: type, optional?: type) — description`. \
                 Use it to discover tools that are not in your schema, then call a discovery \
                 result directly by name. `filter` is an optional case-insensitive regular \
                 expression; `max` defaults to 20 (max 100); `search_descriptions` defaults \
                 to true."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "filter".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Optional case-insensitive regular expression matched \
                                      against tool names, and against descriptions when \
                                      `search_descriptions` is true"
                            .into(),
                        required: false,
                    },
                ),
                (
                    "max".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Maximum number of matches to return (default 20, max 100)"
                            .into(),
                        required: false,
                    },
                ),
                (
                    "search_descriptions".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Boolean,
                        description: "Also match the filter against descriptions (default true)"
                            .into(),
                        required: false,
                    },
                ),
            ]),
        },
        Tool {
            name: "fs_read".into(),
            description: Some(
                "Read a file from the shared virtual flat filesystem. Every LLM Agent sees \
                 the same files. Paths are plain file names without directories. Use `offset` \
                 and the required `limit` to read a byte range instead of the whole file."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "path".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "File name".into(),
                        required: true,
                    },
                ),
                (
                    "offset".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Optional byte offset to start reading from (default 0)"
                            .into(),
                        required: false,
                    },
                ),
                (
                    "limit".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Number of bytes to read (required)".into(),
                        required: true,
                    },
                ),
            ]),
        },
        Tool {
            name: "fs_grep".into(),
            description: Some(
                "Search the shared virtual files for a literal substring, like grep/rg. Prints \
                 `file:line:text` for matches and `file-line-text` for context lines. Omit `path` \
                 to search every file, use `context` to include lines around each match, set \
                 `ignore_case` for a case-insensitive search, and `max_matches` to bound the \
                 number of matching lines (default 200, max 5000)."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "pattern".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Literal substring to find".into(),
                        required: true,
                    },
                ),
                (
                    "path".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Optional file name; omit to search all files".into(),
                        required: false,
                    },
                ),
                (
                    "context".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Optional number of lines before and after each match (default 0, max 100)"
                            .into(),
                        required: false,
                    },
                ),
                (
                    "ignore_case".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Boolean,
                        description: "Match without regard to case (default false)".into(),
                        required: false,
                    },
                ),
                (
                    "max_matches".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Optional maximum number of matching lines to report (default 200)"
                            .into(),
                        required: false,
                    },
                ),
            ]),
        },
        Tool {
            name: "fs_sed".into(),
            description: Some(
                "Print an inclusive 1-based line range from a shared virtual file, like \
                 `sed -n 'START,ENDp'`. `start_line` defaults to 1; omit `end_line` to print \
                 through the end of the file. Lines are prefixed with their 1-based line number \
                 unless `line_numbers` is false."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "path".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "File name".into(),
                        required: true,
                    },
                ),
                (
                    "start_line".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "First line to print (1-based, default 1)".into(),
                        required: false,
                    },
                ),
                (
                    "end_line".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Last line to print (inclusive, default: end of file)".into(),
                        required: false,
                    },
                ),
                (
                    "line_numbers".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Boolean,
                        description: "Prefix each line with its 1-based line number (default true)"
                            .into(),
                        required: false,
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
                        required: true,
                    },
                ),
                (
                    "content".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Complete new file content".into(),
                        required: true,
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
                        required: true,
                    },
                ),
                (
                    "start_line".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "First line to replace (1-based)".into(),
                        required: true,
                    },
                ),
                (
                    "end_line".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Last line to replace (inclusive)".into(),
                        required: true,
                    },
                ),
                (
                    "content".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Replacement text; may contain newlines".into(),
                        required: true,
                    },
                ),
            ]),
        },
        Tool {
            name: "fs_edit_pattern".into(),
            description: Some(
                "Replace text in a shared virtual file. By default the pattern is literal and \
                 must occur exactly once; set `allow_multiple` to replace every match and \
                 `use_regex` to use a regular expression (replacement supports `$1`/`${name}` \
                 captures)."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "path".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "File name".into(),
                        required: true,
                    },
                ),
                (
                    "pattern".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Literal text or regular expression to find".into(),
                        required: true,
                    },
                ),
                (
                    "replacement".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Replacement text (may be empty)".into(),
                        required: true,
                    },
                ),
                (
                    "allow_multiple".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Boolean,
                        description: "Replace every match instead of failing on multiple matches"
                            .into(),
                        required: false,
                    },
                ),
                (
                    "use_regex".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Boolean,
                        description: "Treat `pattern` as a regular expression".into(),
                        required: false,
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
                    required: true,
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
                        required: true,
                    },
                ),
                (
                    "run_in_background".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Boolean,
                        description: "Run asynchronously and deliver the result as a message"
                            .into(),
                        required: false,
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
                        required: false,
                    },
                ),
                (
                    "run_in_background".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Boolean,
                        description: "Run asynchronously and deliver the result as a message"
                            .into(),
                        required: false,
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
                    required: false,
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
                        required: false,
                    },
                ),
                (
                    "message".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Message to send to the target agent".into(),
                        required: false,
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
                        required: false,
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
                        required: false,
                    },
                ),
                (
                    "amount".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Optional number of messages to return (default 20)".into(),
                        required: false,
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
                        required: false,
                    },
                ),
                (
                    "index".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Message index from get_agent_history".into(),
                        required: false,
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
                        required: false,
                    },
                ),
                (
                    "prompt".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Optional initial task for the new agent".into(),
                        required: false,
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
                    required: false,
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

/// Dispatch one tool call by name.
///
/// Returns `(message, is_error)`. Honours `DisabledTool`, reports unknown
/// names, and routes to the execution kind registered for the tool.
pub(crate) fn execute_tool(
    world: &mut ToolWorld<'_, '_>,
    name: &str,
    arguments: &Value,
    call: &CallContext<'_>,
) -> (String, bool) {
    if call.depth > crate::tools::schedule::MAX_TOOL_DEPTH {
        return (
            format!(
                "tool call nesting limit ({}) reached",
                crate::tools::schedule::MAX_TOOL_DEPTH
            ),
            true,
        );
    }
    let Some(tool) = registry::find(world, name) else {
        return (format!("Unknown tool: {name}"), true);
    };
    if let Some(reason) = tool.disabled {
        return (format!("Tool `{name}` is disabled: {reason}"), true);
    }
    match tool.execution {
        Execution::Inline(handler) => handler(world, arguments, call),
        Execution::Interpreter => (
            format!("tool `{name}` must run in the JavaScript interpreter"),
            true,
        ),
        Execution::Script => (
            format!("tool `{name}` must run as a script tool call"),
            true,
        ),
        Execution::Gateway => (
            format!("tool `{name}` can only be called by the model"),
            true,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentIndex, AgentLayout};
    use crate::config::{AppConfig, Provider};
    use crate::messages::{RemoveAgentRequest, RenameAgentRequest, SendAgentMessage};
    use crate::tools::catalog::static_tools;
    use crate::virtual_fs::VirtualFs;
    use bevy::prelude::{App, Entity, PreStartup, ResMut, Resource, Update};
    use serde_json::Value;

    #[test]
    fn javascript_tools_are_registered_as_interpreter_tools() {
        let tools = static_tools();
        for name in JS_TOOLS {
            let entry = tools
                .iter()
                .find(|entry| entry.definition.name == *name)
                .expect("JavaScript tool should be registered");
            assert!(
                matches!(entry.execution, Execution::Interpreter),
                "{name} should be an interpreter tool"
            );
        }
        for name in ["fs_read", "spawn_agent", "end_agent"] {
            let entry = tools
                .iter()
                .find(|entry| entry.definition.name == name)
                .expect("tool should be registered");
            assert!(
                !matches!(entry.execution, Execution::Interpreter),
                "{name} should not be an interpreter tool"
            );
        }
    }

    #[test]
    fn every_definition_is_registered_with_unique_names() {
        let definitions = definitions();
        let tools = static_tools();
        assert_eq!(tools.len(), definitions.len());

        let mut names = definitions
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>();
        names.sort_unstable();
        let unique = names.len();
        names.dedup();
        assert_eq!(names.len(), unique, "duplicate tool names: {names:?}");

        for definition in &definitions {
            assert!(
                tools
                    .iter()
                    .any(|entry| entry.definition.name == definition.name),
                "`{}` is defined but not registered",
                definition.name
            );
        }
    }

    #[test]
    fn advertisement_policy_matches_the_trim() {
        let tools = static_tools();
        let advertised = |name: &str| {
            tools
                .iter()
                .find(|entry| entry.definition.name == name)
                .unwrap_or_else(|| panic!("`{name}` is not registered"))
                .advertised
        };

        for name in [
            "execute",
            "rescan_tools",
            "search_tools",
            "fs_read",
            "fs_write",
            "fs_edit_pattern",
            "fs_edit_lines",
            "fs_ls",
            "fs_grep",
            "fs_sed",
            "eval_javascript",
            "run_javascript",
            "send_message_to_agent",
            "spawn_agent",
            "rename_self",
        ] {
            assert!(advertised(name), "`{name}` should be advertised");
        }
        for name in [
            "fs_delete",
            "eval_javascript_reset_context",
            "list_agents",
            "get_agent_history",
            "get_agent_message",
            "end_agent",
        ] {
            assert!(!advertised(name), "`{name}` should be hidden");
        }
    }

    #[test]
    fn documented_optional_parameters_are_not_required() {
        let tools = definitions();
        let parameter = |tool: &str, parameter: &str| {
            tools
                .iter()
                .find(|entry| entry.name == tool)
                .unwrap_or_else(|| panic!("missing tool `{tool}`"))
                .parameters
                .get(parameter)
                .unwrap_or_else(|| panic!("missing parameter `{tool}.{parameter}`"))
        };

        for (tool, name) in [
            ("fs_read", "offset"),
            ("fs_grep", "path"),
            ("fs_grep", "context"),
            ("fs_grep", "ignore_case"),
            ("fs_grep", "max_matches"),
            ("fs_sed", "start_line"),
            ("fs_sed", "end_line"),
            ("fs_sed", "line_numbers"),
            ("fs_edit_pattern", "allow_multiple"),
            ("fs_edit_pattern", "use_regex"),
            ("run_javascript", "run_in_background"),
            ("eval_javascript", "run_in_background"),
            ("eval_javascript_reset_context", "symbol"),
            ("get_agent_history", "position"),
            ("get_agent_history", "amount"),
            ("spawn_agent", "name"),
            ("spawn_agent", "prompt"),
        ] {
            assert!(
                !parameter(tool, name).required,
                "`{tool}.{name}` is documented as optional"
            );
        }

        for (tool, name) in [
            ("fs_read", "path"),
            ("fs_read", "limit"),
            ("fs_write", "path"),
            ("fs_write", "content"),
            ("fs_edit_pattern", "pattern"),
        ] {
            assert!(
                parameter(tool, name).required,
                "`{tool}.{name}` is required"
            );
        }
    }

    #[test]
    fn search_tools_has_only_optional_parameters() {
        let tools = definitions();
        let search = tools
            .iter()
            .find(|tool| tool.name == "search_tools")
            .expect("search_tools should exist");
        for (name, parameter) in &search.parameters {
            assert!(!parameter.required, "`{name}` should be optional");
        }
    }

    #[test]
    fn every_definition_has_a_handler() {
        for definition in definitions() {
            assert!(
                crate::tools::catalog::is_implemented(&definition.name),
                "`{}` is defined but no module handles it",
                definition.name
            );
        }
    }

    fn smoke_context() -> CallContext<'static> {
        CallContext {
            source_entity: Entity::PLACEHOLDER,
            source_agent: "Smoke",
            depth: 0,
            root: 0,
        }
    }

    #[derive(Resource, Default)]
    struct SmokeOutput(String);

    fn smoke_dispatch(mut world: ToolWorld, mut out: ResMut<SmokeOutput>) {
        let (inline_message, inline_error) = execute_tool(
            &mut world,
            "search_tools",
            &serde_json::json!({ "filter": "^fs_read$" }),
            &smoke_context(),
        );
        let (legacy_message, legacy_error) =
            execute_tool(&mut world, "fs_ls", &Value::Null, &smoke_context());
        let (unknown_message, unknown_error) =
            execute_tool(&mut world, "no_such_tool", &Value::Null, &smoke_context());
        let (agents_message, agents_error) =
            execute_tool(&mut world, "list_agents", &Value::Null, &smoke_context());
        out.0 = format!(
            "{inline_error}|{inline_message}\n{legacy_error}|{legacy_message}\n{unknown_error}|{unknown_message}\n{agents_error}|{agents_message}"
        );
    }

    #[test]
    fn registered_tools_dispatch_through_the_registry() {
        let mut app = App::new();
        app.insert_resource(AgentIndex::default())
            .insert_resource(AgentLayout {
                tabs: Entity::PLACEHOLDER,
                panels: Entity::PLACEHOLDER,
                next_number: 1,
            })
            .insert_resource(VirtualFs::default())
            .insert_resource(AppConfig {
                provider: Provider::Responses,
                model_name: "test".into(),
                endpoint: "http://localhost".into(),
                token: String::new(),
                session_id: "ses_test".into(),
            })
            .init_resource::<SmokeOutput>()
            .init_resource::<crate::tools::ToolRegistryStatus>()
            .init_resource::<crate::tools::ToolsDir>()
            .add_message::<SendAgentMessage>()
            .add_message::<RemoveAgentRequest>()
            .add_message::<RenameAgentRequest>()
            .add_systems(PreStartup, register_static_tools)
            .add_systems(Update, smoke_dispatch);
        app.update();

        let output = &app.world().resource::<SmokeOutput>().0;
        let lines = output.lines().collect::<Vec<_>>();
        assert!(
            lines[0].starts_with("false|fs_read("),
            "search_tools returned {lines:?}"
        );
        assert_eq!(lines[1], "false|Virtual filesystem is empty.");
        assert_eq!(lines[2], "true|Unknown tool: no_such_tool");
        assert_eq!(lines[3], "false|Available agents: ");
    }

    #[test]
    fn initial_task_message_has_the_required_prefix() {
        assert_eq!(
            initial_task_message("LLM Agent 1", "Summarize the notes."),
            "You were created by LLM Agent 1, your task:\nSummarize the notes."
        );
    }

    #[test]
    fn fs_read_exposes_offset_and_limit() {
        let tools = definitions();
        let fs_read = tools
            .iter()
            .find(|tool| tool.name == "fs_read")
            .expect("fs_read should exist");
        assert!(fs_read.parameters.contains_key("offset"));
        assert!(fs_read.parameters.contains_key("limit"));
        assert!(!fs_read.parameters.contains_key("seek"));
        assert!(!fs_read.parameters.contains_key("len"));

        assert_eq!(
            optional_u64(&serde_json::json!({ "offset": 3 }), "offset").unwrap(),
            Some(3)
        );
        assert!(optional_u64(&serde_json::json!({ "offset": -1 }), "offset").is_err());

        assert_eq!(
            required_u64(&serde_json::json!({ "limit": 3 }), "limit").unwrap(),
            3
        );
        assert!(required_u64(&serde_json::json!({}), "limit").is_err());
        assert!(required_u64(&serde_json::json!({ "limit": null }), "limit").is_err());
        assert!(required_u64(&serde_json::json!({ "limit": -1 }), "limit").is_err());
    }

    #[test]
    fn javascript_tools_expose_a_background_flag() {
        let tools = definitions();
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
    fn grep_and_sed_tools_are_advertised() {
        let tools = definitions();
        let grep = tools
            .iter()
            .find(|tool| tool.name == "fs_grep")
            .expect("fs_grep should exist");
        for parameter in ["pattern", "path", "context", "ignore_case", "max_matches"] {
            assert!(
                grep.parameters.contains_key(parameter),
                "fs_grep is missing `{parameter}`"
            );
        }

        let sed = tools
            .iter()
            .find(|tool| tool.name == "fs_sed")
            .expect("fs_sed should exist");
        for parameter in ["path", "start_line", "end_line", "line_numbers"] {
            assert!(
                sed.parameters.contains_key(parameter),
                "fs_sed is missing `{parameter}`"
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
    fn interpreter_tasks_only_accept_javascript_tools() {
        let fs = VirtualFs::default();

        assert!(interpreter_task("fs_read", &Value::Null, &fs, 0, 0).is_err());
        assert!(matches!(
            interpreter_task(
                "eval_javascript",
                &serde_json::json!({ "code": "1 + 1" }),
                &fs,
                0,
                0
            ),
            Ok(JsTask::Eval { .. })
        ));
        assert!(matches!(
            interpreter_task(
                "eval_javascript_reset_context",
                &serde_json::json!({ "symbol": "" }),
                &fs,
                0,
                0
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
        let tools = definitions();
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
        let tools = definitions();
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
