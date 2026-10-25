//! Shared constructors for unit tests.
//!
//! Provider modules build the same `Entry` shapes over and over. Keeping them
//! here makes a history-shape change a one-file edit instead of eight.

use llm_history_model::history::{
    Entry, MessageContent, SystemMessageKind, ToolCall, ToolCallResult,
};

use crate::{
    generic_tool::{ParameterDescription, ParameterType, Tool},
    model_config::ModelConfig,
    validation::validate_entries,
};

pub(crate) fn text_user(text: &str) -> Entry {
    Entry::UserMessage {
        content: MessageContent {
            text: Some(text.into()),
            image: None,
        },
    }
}

pub(crate) fn assistant_text(text: &str) -> Entry {
    Entry::response_text(text)
}

pub(crate) fn system_message(content: &str) -> Entry {
    Entry::SystemMessage {
        kind: SystemMessageKind::Environment,
        content: content.into(),
    }
}

pub(crate) fn tool_call(id: &str) -> Entry {
    Entry::ResponseToolCall {
        blocks: Vec::new(),
        tool_call: vec![ToolCall {
            tool_name: "get_weather".into(),
            toolcall_params: r#"{"city":"Oslo"}"#.into(),
            tool_use_id: id.into(),
        }],
    }
}

pub(crate) fn tool_result(id: &str) -> Entry {
    Entry::ToolCallResult {
        tool_call: vec![ToolCallResult {
            result: "sunny".into(),
            is_error: false,
            tool_use_id: id.into(),
        }],
    }
}

/// Prepend a user turn, validate assistant-side entries, and return the full
/// history for tests that need to keep using it.
pub(crate) fn validated_assistant_history(entries: Vec<Entry>, config: &ModelConfig) -> Vec<Entry> {
    let mut history = vec![text_user("hi")];
    history.extend(entries);
    assert_eq!(validate_entries(&history, config), Ok(()));
    history
}

/// Build a tool definition from a compact parameter list.
pub(crate) fn tool(
    name: &str,
    description: Option<&str>,
    parameters: &[(&str, ParameterType, &str, bool)],
) -> Tool {
    let parameters = parameters
        .iter()
        .map(|(name, parameter_type, description, required)| {
            (
                (*name).to_owned(),
                ParameterDescription {
                    r#type: parameter_type.clone(),
                    description: (*description).to_owned(),
                    required: *required,
                },
            )
        })
        .collect();
    Tool {
        name: name.to_owned(),
        description: description.map(str::to_owned),
        parameters,
    }
}
