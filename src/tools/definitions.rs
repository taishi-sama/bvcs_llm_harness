//! Static tool definitions.
//!
//! `definitions()` is the single source of truth for model-facing names,
//! descriptions, and parameter schemas. Exposure policy lives in `catalog`.

use std::collections::HashMap;

use llm_api::generic_tool::{ParameterDescription, ParameterType, Tool};

/// Every tool definition. This is the single source of truth for names,
/// descriptions, and parameter schemas; exposure is decided by `catalog`.
pub(crate) fn definitions() -> Vec<Tool> {
    vec![
        Tool {
            name: "rescan_tools".into(),
            description: Some(
                "Reconcile `.tools` script tools with the live registry. Reports how many tools \
                 the scan found plus added/removed/changed from this pass. Hot reload applies \
                 changes after tool calls and before the next request, so an empty diff is \
                 followed by the last automatic change. `apply` defaults to true: it asks that \
                 reconciler to run even if a dry run already recorded the revision. Set it to \
                 false for a dry run that records the revision and prevents the automatic apply. \
                 `source` is `vfs` (default, use in-memory edits) or `disk` (reload `./.tools` \
                 first, discarding unmirrored in-memory edits)."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "apply".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Boolean,
                        description: "Apply the diff to the registry (default true)".into(),
                        required: false,
                    },
                ),
                (
                    "source".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "`vfs` (default) or `disk`".into(),
                        required: false,
                    },
                ),
            ]),
        },
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
                        description: "Maximum number of matches to return (default 20, min 1, max 100)"
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
                "Read a file from the shared virtual filesystem. Every LLM Agent sees \
                 the same files. Root paths are plain file names; `.tools/<dir>/<file>` is \
                 the only hierarchical mount. Use `offset` and the required `limit` to read \
                 a byte range. A result shorter than `limit` means the range reached end of \
                 file; there is no truncation marker."
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
                        description: "Optional maximum number of matching lines to report (default 200, min 1, max 5000)"
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
                "List files in the shared virtual filesystem with their sizes and line \
                 counts: root files first, then every `.tools` key. Every LLM Agent sees the \
                 same files."
                    .into(),
            ),
            parameters: HashMap::new(),
        },
        Tool {
            name: "fs_write".into(),
            description: Some(
                "Create or overwrite a file in the shared virtual filesystem. Every LLM \
                 Agent sees the same files. Root paths are plain file names; \
                 `.tools/<dir>/<file>` is the only hierarchical mount."
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
            description: Some(
                "Delete a file from the shared virtual filesystem: a root file name or \
                 `.tools/<dir>/<file>`."
                    .into(),
            ),
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
            name: "eval_javascript_reset_context".into(),
            description: Some(
                "Drop JavaScript state for this LLM Agent. Pass `symbol` to remove one \
                 configurable global variable, or omit/empty `symbol` to drop the whole \
                 persistent context. Lexical let/const bindings and non-configurable var \
                 bindings cannot be removed individually."
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
                        required: true,
                    },
                ),
                (
                    "message".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Message to send to the target agent".into(),
                        required: true,
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
                 own. Omit `agent` to read this agent's history. Each item has `type`, \
                 `sender`, and `message_index`; read the full message with get_agent_message. \
                 By default the last 20 messages are returned."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "agent".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Optional exact agent title; defaults to this agent"
                            .into(),
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
                "Read one message from an agent's conversation history, serialized as JSON. \
                 The result has `type`, `sender`, `message_index`, and the raw `message`. \
                 Omit `agent` to read this agent's history. Use the `message_index` values from \
                 get_agent_history."
                    .into(),
            ),
            parameters: HashMap::from([
                (
                    "agent".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: "Optional exact agent title; defaults to this agent"
                            .into(),
                        required: false,
                    },
                ),
                (
                    "index".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: "Message index from get_agent_history".into(),
                        required: true,
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
                    required: true,
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
