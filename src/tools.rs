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
mod definitions;
mod fs;
mod registration;
mod registry;
mod search;
mod world;

use bevy::{
    ecs::schedule::IntoScheduleConfigs,
    prelude::{App, Plugin, PreStartup, Query, Update},
};
use llm_api::generic_tool::{ParameterType, Tool};
use llm_history_model::history::Entry;
use serde_json::Value;

use crate::{
    AppSet,
    agent::{Agent, AgentIndex},
    js_runtime::JsTask,
    virtual_fs::VirtualFs,
};

pub(crate) use agent_tools::history_view;
pub(crate) use catalog::{AdvertisedTools, advertised_tools, register_script_tools};
pub(crate) use definitions::definitions;
pub(crate) use mirror::{
    MirrorDebounce, MirrorStatus, ToolsDir, flush_tool_mirror, seed_tool_tree,
};
pub(crate) use registry::{CallContext, Execution, find, register_static_tools};
pub(crate) use rescan::{ToolRegistryStatus, reconcile_script_tools};
pub(crate) use schedule::background_started_message;
pub(crate) use script::{ScriptHostRequest, ScriptRuntime, ScriptTask, ToolSourceCache};
pub(crate) use world::ToolWorld;

/// Tool catalog, execution resources, and the cross-thread tool pipeline.
pub(crate) struct ToolsPlugin;

impl Plugin for ToolsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ToolsDir>()
            .init_resource::<MirrorStatus>()
            .init_resource::<MirrorDebounce>()
            .init_resource::<ScriptRuntime>()
            .init_resource::<ToolSourceCache>()
            .init_resource::<ToolRegistryStatus>()
            .init_resource::<schedule::ToolRounds>()
            .init_resource::<schedule::AutoBackgroundPolicy>()
            .add_systems(
                PreStartup,
                (seed_tool_tree, register_static_tools, register_script_tools).chain(),
            )
            .add_systems(
                Update,
                script::refresh_tool_sources.in_set(AppSet::ToolSources),
            )
            .add_systems(
                Update,
                schedule::auto_background_long_calls.in_set(AppSet::AutoBackground),
            )
            .add_systems(Update, flush_tool_mirror.in_set(AppSet::MirrorFlush))
            .add_systems(
                Update,
                reconcile_script_tools.in_set(AppSet::ReconcileTools),
            );
    }
}

/// The tools handled by the agent's persistent JavaScript thread.
///
/// One enum keeps the name list, the task constructor, and [`is_js_tool`] from
/// drifting apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JsTool {
    RunJavascript,
    EvalJavascript,
    ResetContext,
}

impl JsTool {
    pub(crate) const ALL: [Self; 3] = [
        Self::RunJavascript,
        Self::EvalJavascript,
        Self::ResetContext,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::RunJavascript => "run_javascript",
            Self::EvalJavascript => "eval_javascript",
            Self::ResetContext => "eval_javascript_reset_context",
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|tool| tool.name() == name)
    }
}

/// Build the interpreter task for one JavaScript tool call.
pub(crate) fn interpreter_task(
    name: &str,
    arguments: &Value,
    fs: &VirtualFs,
    depth: u32,
    root: u64,
) -> Result<JsTask, String> {
    let tool = JsTool::from_name(name).ok_or_else(|| format!("not a JavaScript tool: {name}"))?;
    match tool {
        JsTool::RunJavascript => {
            let file = arguments
                .get("file")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing required string parameter `file`".to_owned())?;
            let source = fs.read(file)?;
            Ok(JsTask::Eval {
                source,
                name: Some(file.to_owned()),
                depth,
                root,
            })
        }
        JsTool::EvalJavascript => {
            let code = arguments
                .get("code")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing required string parameter `code`".to_owned())?;
            Ok(JsTask::Eval {
                source: code.to_owned(),
                name: None,
                depth,
                root,
            })
        }
        JsTool::ResetContext => {
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
    }
}

/// Whether `name` is handled by the async JS runtime rather than
/// [`execute_tool`], which keeps JavaScript from recursively starting itself.
pub(crate) fn is_js_tool(name: &str) -> bool {
    JsTool::from_name(name).is_some()
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

/// Required string argument.
pub(crate) fn required_str<'a>(arguments: &'a Value, name: &str) -> Result<&'a str, String> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing required string parameter `{name}`"))
}

/// Optional string argument. Non-string values count as absent, matching the
/// permissive policy used by the inline tools.
pub(crate) fn optional_str<'a>(arguments: &'a Value, name: &str) -> Option<&'a str> {
    arguments.get(name).and_then(Value::as_str)
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

/// Sorted copy of the agent names used in messages and diagnostics.
pub(crate) fn sorted_agent_names(index: &AgentIndex) -> Vec<String> {
    let mut names = index.by_name.keys().cloned().collect::<Vec<_>>();
    names.sort();
    names
}

/// Look up an agent by title, with a helpful error listing what exists.
fn history_agent<'a>(
    name: &str,
    index: &AgentIndex,
    agents: &'a Query<&Agent>,
) -> Result<&'a Agent, String> {
    let Some(entity) = index.by_name.get(name) else {
        return Err(format!(
            "No agent named `{name}`. Available agents: {}",
            sorted_agent_names(index).join(", ")
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

/// Validate one script tool's arguments against its advertised manifest.
///
/// Built-in tools keep their hand-written checks. Script tools previously used
/// `parameters` for documentation only, so a missing required argument produced
/// a plausible but wrong answer instead of an error. Unknown keys are passed
/// through (the same permissive policy as the built-ins).
pub(crate) fn validate_script_arguments(
    world: &ToolWorld<'_, '_>,
    name: &str,
    arguments: &Value,
) -> Result<(), String> {
    let Some(found) = find(world, name) else {
        return Ok(());
    };
    let Ok((_, entry, _, _, _)) = world.tools.get(found.entity) else {
        return Ok(());
    };
    validate_manifest_arguments(&entry.definition, arguments)
}

fn validate_manifest_arguments(definition: &Tool, arguments: &Value) -> Result<(), String> {
    let empty = serde_json::Map::new();
    let values = match arguments {
        Value::Object(values) => values,
        Value::Null => &empty,
        other => {
            return Err(format!(
                "`{}` arguments must be a JSON object, not {}",
                definition.name,
                json_type_name(other)
            ));
        }
    };

    for (name, parameter) in &definition.parameters {
        match values.get(name) {
            None | Some(Value::Null) if parameter.required => {
                return Err(format!(
                    "missing required {} parameter `{name}`",
                    parameter_type_name(&parameter.r#type)
                ));
            }
            None | Some(Value::Null) => {}
            Some(value) if !argument_matches(&parameter.r#type, value) => {
                return Err(format!(
                    "`{name}` must be a {}",
                    parameter_type_name(&parameter.r#type)
                ));
            }
            Some(_) => {}
        }
    }

    Ok(())
}

const fn parameter_type_name(parameter_type: &ParameterType) -> &'static str {
    match parameter_type {
        ParameterType::String => "string",
        ParameterType::Number => "number",
        ParameterType::Boolean => "boolean",
        ParameterType::Object => "object",
    }
}

fn argument_matches(parameter_type: &ParameterType, value: &Value) -> bool {
    match parameter_type {
        ParameterType::String => value.is_string(),
        ParameterType::Number => value.is_number(),
        ParameterType::Boolean => value.is_boolean(),
        ParameterType::Object => value.is_object(),
    }
}

const fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
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
        Execution::Missing => ("tool has no registered handler".to_owned(), true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, AgentIndex, AgentLayout};
    use crate::config::{AppConfig, Provider};
    use crate::messages::{RemoveAgentRequest, RenameAgentRequest, SendAgentMessage};
    use crate::tools::catalog::static_tools;
    use crate::virtual_fs::VirtualFs;
    use bevy::prelude::{App, Entity, PreStartup, ResMut, Resource, Update};
    use llm_api::generic_tool::ParameterDescription;
    use serde_json::Value;
    use std::collections::HashMap;

    #[test]
    fn javascript_tools_are_registered_as_interpreter_tools() {
        let tools = static_tools();
        for tool in JsTool::ALL {
            let name = tool.name();
            let entry = tools
                .iter()
                .find(|entry| entry.definition.name == name)
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
    fn definitions_have_unique_names() {
        let mut names = definitions()
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        names.sort_unstable();
        let unique = names.len();
        names.dedup();
        assert_eq!(names.len(), unique, "duplicate tool names: {names:?}");
    }

    #[test]
    fn registration_flags_match_the_policy_table() {
        // (name, advertised, model_only, auto_background)
        const EXPECTED: &[(&str, bool, bool, bool)] = &[
            ("execute", true, true, false),
            ("search_tools", true, false, false),
            ("rescan_tools", true, false, false),
            ("fs_read", true, false, false),
            ("fs_grep", true, false, false),
            ("fs_sed", true, false, false),
            ("fs_ls", true, false, false),
            ("fs_write", true, false, false),
            ("fs_edit_lines", true, false, false),
            ("fs_edit_pattern", true, false, false),
            ("fs_delete", false, false, false),
            ("run_javascript", true, true, true),
            ("eval_javascript", true, true, true),
            ("eval_javascript_reset_context", false, false, false),
            ("send_message_to_agent", true, false, false),
            ("list_agents", false, false, false),
            ("get_agent_history", false, false, false),
            ("get_agent_message", false, false, false),
            ("spawn_agent", true, true, false),
            ("rename_self", true, true, false),
            ("end_agent", false, true, false),
        ];

        let tools = static_tools();
        assert_eq!(
            tools.len(),
            EXPECTED.len(),
            "catalog/policy table size mismatch"
        );
        for (name, advertised, model_only, auto_background) in EXPECTED {
            let entry = tools
                .iter()
                .find(|entry| entry.definition.name == *name)
                .unwrap_or_else(|| panic!("`{name}` is not registered"));
            assert_eq!(entry.advertised, *advertised, "`{name}` advertised");
            assert_eq!(entry.model_only, *model_only, "`{name}` model_only");
            assert_eq!(
                entry.auto_background, *auto_background,
                "`{name}` auto_background"
            );
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
            ("get_agent_history", "agent"),
            ("get_agent_message", "agent"),
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
            ("get_agent_message", "index"),
            ("eval_javascript", "code"),
            ("send_message_to_agent", "agent"),
            ("send_message_to_agent", "message"),
            ("rename_self", "name"),
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
    fn script_manifest_arguments_are_validated() {
        let tool = Tool {
            name: "demo".into(),
            description: None,
            parameters: HashMap::from([
                (
                    "text".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::String,
                        description: String::new(),
                        required: true,
                    },
                ),
                (
                    "count".to_owned(),
                    ParameterDescription {
                        r#type: ParameterType::Number,
                        description: String::new(),
                        required: false,
                    },
                ),
            ]),
        };

        assert!(validate_manifest_arguments(&tool, &serde_json::json!({ "text": "hi" })).is_ok());
        assert!(
            validate_manifest_arguments(&tool, &serde_json::json!({ "text": "hi", "count": 2 }))
                .is_ok()
        );
        assert!(
            validate_manifest_arguments(&tool, &Value::Null)
                .unwrap_err()
                .contains("missing required string parameter `text`")
        );
        assert!(
            validate_manifest_arguments(&tool, &serde_json::json!({ "text": 1 }))
                .unwrap_err()
                .contains("`text` must be a string")
        );
        assert!(
            validate_manifest_arguments(
                &tool,
                &serde_json::json!({ "text": "hi", "count": "two" })
            )
            .unwrap_err()
            .contains("`count` must be a number")
        );
        assert!(
            validate_manifest_arguments(&tool, &serde_json::json!("hi"))
                .unwrap_err()
                .contains("must be a JSON object")
        );

        let no_parameters = Tool {
            name: "noparams".into(),
            description: None,
            parameters: HashMap::new(),
        };
        assert!(validate_manifest_arguments(&no_parameters, &serde_json::json!({})).is_ok());
        assert!(validate_manifest_arguments(&no_parameters, &Value::Null).is_ok());
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
        let (history_message, history_error) = execute_tool(
            &mut world,
            "get_agent_history",
            &serde_json::json!({}),
            &smoke_context(),
        );
        out.0 = format!(
            "{inline_error}|{inline_message}\n{legacy_error}|{legacy_message}\n{unknown_error}|{unknown_message}\n{agents_error}|{agents_message}\n{history_error}|{history_message}"
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

        let agent = app
            .world_mut()
            .spawn(Agent::new(
                "Smoke".to_owned(),
                Provider::Responses,
                "test".to_owned(),
            ))
            .id();
        app.world_mut()
            .resource_mut::<AgentIndex>()
            .by_name
            .insert("Smoke".to_owned(), agent);

        app.update();

        let output = &app.world().resource::<SmokeOutput>().0;
        let lines = output.lines().collect::<Vec<_>>();
        assert!(
            lines[0].starts_with("false|fs_read("),
            "search_tools returned {lines:?}"
        );
        assert_eq!(lines[1], "false|Virtual filesystem is empty.");
        assert_eq!(lines[2], "true|Unknown tool: no_such_tool");
        assert_eq!(lines[3], "false|Available agents: Smoke (you)");
        assert!(
            output.contains("false|[") && output.contains("\"type\": \"system_prompt\""),
            "get_agent_history without `agent` returned {lines:?}"
        );
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
    fn grep_and_sed_tools_expose_parameters() {
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
    fn history_tools_expose_parameters() {
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
