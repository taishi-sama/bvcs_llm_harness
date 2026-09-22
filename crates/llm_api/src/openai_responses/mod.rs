//! Serialization of [`crate::history::Entry`] histories into the
//! OpenAI-compatible **Responses API** (`POST /responses`) wire format.
//!
//! Mapping overview:
//! - `SystemPrompt`      -> top-level `instructions` parameter
//! - `UserMessage`       -> `{type:"message", role:"user"}` (string content, or
//!   typed parts when an image is attached)
//! - `ResponseNormal`    -> optional `reasoning` item + `{type:"message", role:"assistant"}`
//! - `ResponseToolCall`  -> optional `reasoning` item + one `function_call` item per call
//! - `ToolCallResult`    -> one `function_call_output` item per result
//!
//! Policies:
//! - `Thinking::Normal` and `Thinking::Secret` are both emitted as plaintext
//!   `reasoning_text`; OpenAI round-trips its own reasoning via opaque
//!   `encrypted_content`, which this model does not store, so reasoning context
//!   is best-effort for o-series models.
//! - `is_error` has no native flag in Responses; errors are prefixed `"ERROR: "`.
//! - Reasoning items get synthesized ids (`rs_NNNNNN`) since the model does not
//!   track provider item ids.

pub mod response;
pub mod tools;

use crate::model_config::ModelConfig;
use crate::validation::validate_entries;
use crate::wire_common::{mime_str, thinking_text};
use llm_history_model::history::{Entry, MessageContent, Thinking};
use serde::Serialize;

pub use tools::{OpenAiFunction, OpenAiTool, tools_to_openai};

/// The `input` portion of a `POST /responses` request body.
#[derive(Debug, Serialize)]
pub struct ResponsesInput<'a> {
    /// Always false, we don't need statefulness
    store: bool,
    pub model: String,
    pub stream: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    prompt_cache_key: String,
    /// System prompt, hoisted out of the item list (top-level request parameter).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<&'a str>,
    /// Function tools forwarded from `ModelConfig.tools`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OpenAiTool>,
    pub input: Vec<InputItem>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    Message(MessageItem),
    FunctionCall(FunctionCallItem),
    FunctionCallOutput(FunctionCallOutputItem),
    Reasoning(ReasoningItem),
}

#[derive(Debug, Serialize)]
pub struct MessageItem {
    pub role: Role,
    pub content: Content,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Plain string for text-only messages, or typed content parts when an image
/// is attached (`content: string | ResponseInputMessageContentList`).
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "input_text")]
    Text { text: String },
    #[serde(rename = "input_image")]
    Image {
        /// Base64-encoded image as a data URL, e.g. `data:image/png;base64,...`
        image_url: String,
    },
}

#[derive(Debug, Serialize)]
pub struct FunctionCallItem {
    pub call_id: String,
    pub name: String,
    /// A JSON string of arguments — matches `ToolCall::toolcall_params` as-is.
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct FunctionCallOutputItem {
    pub call_id: String,
    pub output: String,
}

#[derive(Debug, Serialize)]
pub struct ReasoningItem {
    /// Synthesized id; the Responses API requires one per reasoning item.
    pub id: String,
    pub summary: Vec<SummaryText>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<ReasoningText>,
}

#[derive(Debug, Serialize)]
pub struct SummaryText {
    /// Always `"summary_text"`
    pub r#type: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ReasoningText {
    /// Always `"reasoning_text"`
    pub r#type: &'static str,
    pub text: String,
}

fn push_reasoning(items: &mut Vec<InputItem>, thinking: &[Thinking], counter: &mut usize) {
    for t in thinking {
        let id = format!("rs_{counter:06}");
        *counter += 1;
        items.push(InputItem::Reasoning(ReasoningItem {
            id,
            summary: vec![],
            content: vec![ReasoningText {
                r#type: "reasoning_text",
                text: thinking_text(t).to_owned(),
            }],
        }));
    }
}

fn user_content(content: &MessageContent) -> Content {
    match (&content.text, &content.image) {
        (Some(text), None) => Content::Text(text.clone()),
        (text, image) => {
            let mut parts = Vec::new();
            if let Some(t) = text
                && !t.is_empty()
            {
                parts.push(ContentPart::Text { text: t.clone() });
            }
            if let Some(img) = image {
                parts.push(ContentPart::Image {
                    image_url: format!("data:image/{};base64,{}", mime_str(&img.mime), img.base64),
                });
            }
            Content::Parts(parts)
        }
    }
}

/// Validates `entries` against `config` and converts them into the Responses
/// API `input` shape.
///
/// Returns all validation errors if the history is malformed; serialization is
/// total otherwise (no additional failure modes).
pub fn to_responses_input<'a>(
    entries: &'a [Entry],
    config: &ModelConfig,
) -> Result<ResponsesInput<'a>, Vec<crate::validation::ValidationError>> {
    validate_entries(entries, config)?;

    let mut instructions: Option<&str> = None;
    let mut items: Vec<InputItem> = Vec::new();
    let mut reasoning_count = 0usize;

    for entry in entries {
        match entry {
            Entry::SystemPrompt { content } => instructions = Some(content),

            Entry::UserMessage { content } => items.push(InputItem::Message(MessageItem {
                role: Role::User,
                content: user_content(content),
            })),

            Entry::ResponseNormal { thinking, content } => {
                push_reasoning(&mut items, thinking, &mut reasoning_count);
                items.push(InputItem::Message(MessageItem {
                    role: Role::Assistant,
                    content: Content::Text(content.clone()),
                }));
            }

            Entry::ResponseToolCall {
                thinking,
                content,
                tool_call,
            } => {
                push_reasoning(&mut items, thinking, &mut reasoning_count);
                if let Some(text) = content
                    && !text.is_empty()
                {
                    items.push(InputItem::Message(MessageItem {
                        role: Role::Assistant,
                        content: Content::Text(text.clone()),
                    }));
                }
                for tc in tool_call {
                    items.push(InputItem::FunctionCall(FunctionCallItem {
                        call_id: tc.tool_use_id.clone(),
                        name: tc.tool_name.clone(),
                        arguments: tc.toolcall_params.clone(),
                    }));
                }
            }

            Entry::ToolCallResult { tool_call } => {
                for r in tool_call {
                    items.push(InputItem::FunctionCallOutput(FunctionCallOutputItem {
                        call_id: r.tool_use_id.clone(),
                        output: if r.is_error {
                            format!("ERROR: {}", r.result)
                        } else {
                            r.result.clone()
                        },
                    }));
                }
            }
        }
    }
    let crate::model_config::ModelKind::OpenAICompatible {
        model_name,
        streaming,
        prompt_cache_key,
    } = &config.kind
    else {
        return Err(vec![crate::validation::ValidationError::WrongProviderKind {
            expected: "openai_compatible",
            actual: config.kind.name(),
        }]);
    };
    Ok(ResponsesInput {
        prompt_cache_key: prompt_cache_key.clone(),
        store: false,
        model: model_name.to_string(),
        instructions,
        tools: tools_to_openai(&config.tools),
        input: items,
        stream: *streaming,
    })
}

#[cfg(test)]
mod tests {
    use llm_history_model::history::ImageMIME;
    use super::*;
    use crate::model_config::ModelKind;
    use llm_history_model::history::{Image, MessageContent, ToolCall, ToolCallResult};
    use serde_json::{Value, json};

    fn config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::OpenAICompatible {
                prompt_cache_key: "prompt_cache_key".into(),
                model_name: "gpt-test".into(),
                streaming: false,
            },
            tools: Default::default(),
            max_tokens: None,
        }
    }

    fn text_user(t: &str) -> Entry {
        Entry::UserMessage {
            content: MessageContent {
                text: Some(t.into()),
                image: None,
            },
        }
    }

    fn serialize(entries: &[Entry]) -> Value {
        serde_json::to_value(to_responses_input(entries, &config()).expect("valid history"))
            .unwrap()
    }

    #[test]
    fn plain_conversation_with_instructions() {
        let v = serialize(&[
            Entry::SystemPrompt {
                content: "Be terse.".into(),
            },
            text_user("hi"),
            Entry::ResponseNormal {
                thinking: Vec::new(),
                content: "Hello!".into(),
            },
        ]);

        assert_eq!(
            v,
            json!({
                "store": false,
                "model": "gpt-test",
                "stream": false,
                "prompt_cache_key": "prompt_cache_key",
                "instructions": "Be terse.",
                "input": [
                    {"type": "message", "role": "user", "content": "hi"},
                    {"type": "message", "role": "assistant", "content": "Hello!"}
                ]
            })
        );
    }

    #[test]
    fn no_system_prompt_omits_instructions() {
        let v = serialize(&[text_user("hi")]);
        assert!(v.get("instructions").is_none());
        assert!(v.get("tools").is_none()); // empty config.tools omitted
    }

    #[test]
    fn config_tools_are_forwarded() {
        let mut parameters = std::collections::HashMap::new();
        parameters.insert(
            "city".to_owned(),
            crate::generic_tool::ParameterDescription {
                r#type: crate::generic_tool::ParameterType::String,
                description: "City name".to_owned(),
            },
        );
        let cfg = ModelConfig {
            tools: vec![crate::generic_tool::Tool {
                name: "get_weather".to_owned(),
                parameters,
                description: None,
            }],
            ..config()
        };

        let v = serde_json::to_value(
            to_responses_input(&[text_user("hi")], &cfg).expect("valid history"),
        )
        .unwrap();

        assert_eq!(
            v["tools"],
            json!([{
                "type": "function",
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                    "additionalProperties": false
                }
            }])
        );
    }

    #[test]
    fn tool_round_trip_with_reasoning() {
        let v = serialize(&[
            text_user("weather in Oslo?"),
            Entry::ResponseToolCall {
                thinking: vec![Thinking::Normal {
                    thinking: "need to check".into(),
                }],
                content: None,
                tool_call: vec![ToolCall {
                    tool_name: "get_weather".into(),
                    toolcall_params: r#"{"city":"Oslo"}"#.into(),
                    tool_use_id: "call_1".into(),
                }],
            },
            Entry::ToolCallResult {
                tool_call: vec![ToolCallResult {
                    result: "sunny".into(),
                    is_error: false,
                    tool_use_id: "call_1".into(),
                }],
            },
            Entry::ResponseNormal {
                thinking: vec![Thinking::Signed {
                    thinking: "it said sunny".into(),
                    signature: "sig==".into(),
                }],
                content: "Sunny!".into(),
            },
        ]);

        assert_eq!(
            v["input"],
            json!([
                {"type": "message", "role": "user", "content": "weather in Oslo?"},
                {
                    "type": "reasoning",
                    "id": "rs_000000",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "need to check"}]
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Oslo\"}"
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
                {
                    "type": "reasoning",
                    "id": "rs_000001",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "it said sunny"}]
                },
                {"type": "message", "role": "assistant", "content": "Sunny!"}
            ])
        );
    }

    #[test]
    fn tool_call_with_content_and_multiple_thinking() {
        let v = serialize(&[
            text_user("do both"),
            Entry::ResponseToolCall {
                thinking: vec![
                    Thinking::Normal {
                        thinking: "first".into(),
                    },
                    Thinking::Signed {
                        thinking: "second".into(),
                        signature: "sig".into(),
                    },
                ],
                content: Some("working on it".into()),
                tool_call: vec![ToolCall {
                    tool_name: "a".into(),
                    toolcall_params: "{}".into(),
                    tool_use_id: "c1".into(),
                }],
            },
            Entry::ToolCallResult {
                tool_call: vec![ToolCallResult {
                    result: "r".into(),
                    is_error: false,
                    tool_use_id: "c1".into(),
                }],
            },
        ]);

        assert_eq!(
            v["input"],
            json!([
                {"type": "message", "role": "user", "content": "do both"},
                {
                    "type": "reasoning",
                    "id": "rs_000000",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "first"}]
                },
                {
                    "type": "reasoning",
                    "id": "rs_000001",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "second"}]
                },
                {"type": "message", "role": "assistant", "content": "working on it"},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "a",
                    "arguments": "{}"
                },
                {"type": "function_call_output", "call_id": "c1", "output": "r"}
            ])
        );
    }

    #[test]
    fn parallel_calls_and_error_prefix() {
        let v = serialize(&[
            text_user("two things"),
            Entry::ResponseToolCall {
                thinking: Vec::new(),
                content: None,
                tool_call: vec![
                    ToolCall {
                        tool_name: "a".into(),
                        toolcall_params: "{}".into(),
                        tool_use_id: "c1".into(),
                    },
                    ToolCall {
                        tool_name: "b".into(),
                        toolcall_params: "{}".into(),
                        tool_use_id: "c2".into(),
                    },
                ],
            },
            Entry::ToolCallResult {
                tool_call: vec![
                    ToolCallResult {
                        result: "ok".into(),
                        is_error: false,
                        tool_use_id: "c1".into(),
                    },
                    ToolCallResult {
                        result: "boom".into(),
                        is_error: true,
                        tool_use_id: "c2".into(),
                    },
                ],
            },
            assistant_done(),
        ]);

        assert_eq!(
            v["input"][1],
            json!({"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"})
        );
        assert_eq!(
            v["input"][3],
            json!({"type": "function_call_output", "call_id": "c1", "output": "ok"})
        );
        assert_eq!(
            v["input"][4],
            json!({"type": "function_call_output", "call_id": "c2", "output": "ERROR: boom"})
        );
    }

    fn assistant_done() -> Entry {
        Entry::ResponseNormal {
            thinking: Vec::new(),
            content: "done".into(),
        }
    }

    #[test]
    fn image_message_becomes_typed_parts() {
        let v = serialize(&[Entry::UserMessage {
            content: MessageContent {
                text: Some("what is this?".into()),
                image: Some(Image {
                    mime: ImageMIME::PNG,
                    base64: "aGk=".into(),
                }),
            },
        }]);

        assert_eq!(
            v["input"],
            json!([{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,aGk="}
                ]
            }])
        );
    }

    #[test]
    fn invalid_history_propagates_errors() {
        let err = to_responses_input(&[text_user("a"), text_user("b")], &config()).unwrap_err();
        assert!(err.iter().any(|e| matches!(
            e,
            crate::validation::ValidationError::IllegalTransition { .. }
        )));
    }

    #[test]
    fn non_openai_kind_is_rejected() {
        let cfg = ModelConfig {
            kind: ModelKind::AnthropicCompatible {
                model_name: "claude".into(),
            },
            tools: Default::default(),
            max_tokens: None,
        };
        assert!(matches!(
            to_responses_input(&[text_user("hi")], &cfg),
            Err(e) if e.contains(&crate::validation::ValidationError::WrongProviderKind {
                expected: "openai_compatible",
                actual: "anthropic_compatible",
            })
        ));
    }
}
