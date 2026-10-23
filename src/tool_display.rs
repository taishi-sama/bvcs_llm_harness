//! Display-only presentation of tool calls for the transcript.
//!
//! Nothing here participates in provider history: the model still sees the raw
//! call (for example `execute`) and its raw result. These helpers only decide
//! how the transcript labels a call and which result renderer to use.
//!
//! To add a special renderer (for example an edit diff), add a
//! [`ToolPresentation`] variant, return it from [`present_call`], and handle the
//! variant in `crate::transcript::render_entry`. Never add `Entry` variants for
//! display-only data; history is shared with the providers.

use llm_history_model::history::ToolCall;
use serde_json::Value;

/// The effective tool call as the transcript should present it.
#[derive(Clone, Debug)]
pub(crate) struct PresentedCall {
    /// Name shown in "Calling ..." and result headings.
    pub(crate) name: String,
    /// Arguments of the effective tool (the target's `args` for `execute`).
    ///
    /// Plain cards do not display arguments; special renderers (such as an
    /// edit diff) can use this instead of re-unwrapping the gateway call.
    #[allow(dead_code, reason = "consumed by future special renderers")]
    pub(crate) arguments: Value,
    /// Which result renderer the transcript should use.
    pub(crate) kind: ToolPresentation,
}

/// How the transcript renders a call's result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ToolPresentation {
    /// Show the raw result text under the tool name.
    #[default]
    Plain,
}

/// Resolve the display form of a model tool call.
///
/// `execute` is a gateway: the transcript shows the tool it targets instead of
/// the wrapper, so a gateway call looks the same as a direct inline call.
/// Malformed or missing gateway arguments fall back to displaying `execute`, so
/// the error result remains understandable.
pub(crate) fn present_call(call: &ToolCall) -> PresentedCall {
    let arguments = parse_arguments(&call.toolcall_params);
    if call.tool_name == "execute"
        && let Some(target) = arguments
            .as_ref()
            .and_then(|arguments| arguments.get("tool_name"))
            .and_then(Value::as_str)
            .filter(|target| !target.trim().is_empty())
    {
        return PresentedCall {
            name: target.to_owned(),
            arguments: arguments
                .as_ref()
                .and_then(|arguments| arguments.get("args"))
                .cloned()
                .unwrap_or(Value::Null),
            kind: ToolPresentation::Plain,
        };
    }

    PresentedCall {
        name: call.tool_name.clone(),
        arguments: arguments.unwrap_or(Value::Null),
        kind: ToolPresentation::Plain,
    }
}

fn parse_arguments(raw: &str) -> Option<Value> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    serde_json::from_str(raw).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, params: &str) -> ToolCall {
        ToolCall {
            tool_name: name.to_owned(),
            toolcall_params: params.to_owned(),
            tool_use_id: "call-1".to_owned(),
        }
    }

    #[test]
    fn gateway_execute_presents_its_target() {
        let presented = present_call(&call(
            "execute",
            r#"{"tool_name":"fs_read","args":{"path":"src/main.rs"}}"#,
        ));

        assert_eq!(presented.name, "fs_read");
        assert_eq!(presented.arguments["path"], "src/main.rs");
        assert_eq!(presented.kind, ToolPresentation::Plain);
    }

    #[test]
    fn gateway_execute_defaults_missing_target_arguments() {
        let presented = present_call(&call("execute", r#"{"tool_name":"fs_ls"}"#));

        assert_eq!(presented.name, "fs_ls");
        assert_eq!(presented.arguments, Value::Null);
    }

    #[test]
    fn unusable_gateway_arguments_stay_execute() {
        for params in ["", "not json", r#"{"args":{}}"#, r#"{"tool_name":""}"#] {
            let presented = present_call(&call("execute", params));
            assert_eq!(presented.name, "execute", "params: {params}");
            assert_eq!(presented.kind, ToolPresentation::Plain);
        }
    }

    #[test]
    fn regular_tools_keep_their_name_and_arguments() {
        let presented = present_call(&call("fs_grep", r#"{"pattern":"tool"}"#));

        assert_eq!(presented.name, "fs_grep");
        assert_eq!(presented.arguments["pattern"], "tool");
    }
}
