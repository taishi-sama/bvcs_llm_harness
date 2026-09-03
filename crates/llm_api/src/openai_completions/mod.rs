//! Serialization of [`crate::history::Entry`] histories into the legacy
//! OpenAI-compatible **Chat Completions API** (`POST /chat/completions`) wire
//! format.
//!
//! Mapping overview:
//! - `SystemPrompt`      -> `{role:"system", content}` message
//! - `UserMessage`       -> `{role:"user"}` (string content, or content parts
//!   when an image is attached)
//! - `ResponseNormal`    -> `{role:"assistant", content}` message
//! - `ResponseToolCall`  -> ONE `{role:"assistant", tool_calls:[...]}` message
//!   (parallel calls grouped via `tool_calls`)
//! - `ToolCallResult`    -> one `{role:"tool", tool_call_id, content}` per result
//!
//! Notes:
//! - `Thinking` is preserved as `reasoning_content` on assistant messages
//!   (both plain and tool-call ones); only the plaintext is kept.
//! - `is_error` has no native flag; errors are prefixed `"ERROR: "`.

use crate::model_config::ModelConfig;
use crate::validation::validate_entries;
use llm_history_model::history::{Entry, ImageMIME, MessageContent, Thinking};
use serde::Serialize;

pub mod response;
pub mod tools;

pub use tools::{CompletionTool, FunctionDefinition, tools_to_completions};

/// The body of a `POST /chat/completions` request.
#[derive(Debug, Serialize)]
pub struct CompletionsInput {
    /// Always false, we don't need statefulness
    store: bool,
    pub model: String,
    pub stream: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    prompt_cache_key: String,
    /// Function tools forwarded from `ModelConfig.tools`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<CompletionTool>,
    pub messages: Vec<Message>,
}

#[derive(Debug, Serialize)]
pub struct Message {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    /// Reasoning text on assistant messages (`reasoning_content`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Present only on assistant tool-call messages.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Present only on tool-result messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// `content: string | [content parts]` — string for plain text, parts when an
/// image is attached.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    Image { image_url: ImageUrl },
}

#[derive(Debug, Serialize)]
pub struct ImageUrl {
    pub url: String,
}

/// A single `{id, type, function}` entry inside an assistant `tool_calls` array.
#[derive(Debug, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ToolCallFunction,
}

#[derive(Debug, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// A JSON string of arguments — matches `ToolCall::toolcall_params` as-is.
    pub arguments: String,
}

fn mime_str(mime: &ImageMIME) -> &'static str {
    match mime {
        ImageMIME::PNG => "png",
        ImageMIME::JPEG => "jpeg",
    }
}

fn thinking_text(t: &Thinking) -> &str {
    match t {
        Thinking::Normal { thinking } => thinking,
        Thinking::Secret { thinking, .. } => thinking,
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
                    image_url: ImageUrl {
                        url: format!("data:image/{};base64,{}", mime_str(&img.mime), img.base64),
                    },
                });
            }
            Content::Parts(parts)
        }
    }
}

/// Validates `entries` against `config` and converts them into the Chat
/// Completions request body.
///
/// Returns all validation errors if the history is malformed. Serialization is
/// total otherwise (no additional failure modes).
pub fn to_completions_input(
    entries: &[Entry],
    config: &ModelConfig,
) -> Result<CompletionsInput, Vec<crate::validation::ValidationError>> {
    validate_entries(entries, config)?;

    let crate::model_config::ModelKind::OpenAICompatible {
        model_name,
        streaming,
        prompt_cache_key,
    } = &config.kind
    else {
        return Err(vec![]);
    };

    let mut messages: Vec<Message> = Vec::new();

    for entry in entries {
        match entry {
            Entry::SystemPrompt { content } => messages.push(Message {
                role: Role::System,
                content: Some(Content::Text(content.clone())),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            }),

            Entry::UserMessage { content } => messages.push(Message {
                role: Role::User,
                content: Some(user_content(content)),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            }),

            Entry::ResponseNormal { thinking, content } => messages.push(Message {
                role: Role::Assistant,
                content: Some(Content::Text(content.clone())),
                reasoning_content: thinking.as_ref().map(thinking_text).map(str::to_owned),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }),

            Entry::ResponseToolCall {
                thinking,
                tool_call,
            } => {
                let tool_calls = tool_call
                    .iter()
                    .map(|tc| ToolCall {
                        id: tc.tool_use_id.clone(),
                        kind: "function",
                        function: ToolCallFunction {
                            name: tc.tool_name.clone(),
                            arguments: tc.toolcall_params.clone(),
                        },
                    })
                    .collect();
                messages.push(Message {
                    role: Role::Assistant,
                    content: None,
                    reasoning_content: thinking.as_ref().map(thinking_text).map(str::to_owned),
                    tool_calls,
                    tool_call_id: None,
                });
            }

            Entry::ToolCallResult { tool_call } => {
                for r in tool_call {
                    messages.push(Message {
                        role: Role::Tool,
                        content: Some(Content::Text(if r.is_error {
                            format!("ERROR: {}", r.result)
                        } else {
                            r.result.clone()
                        })),
                        reasoning_content: None,
                        tool_calls: Vec::new(),
                        tool_call_id: Some(r.tool_use_id.clone()),
                    });
                }
            }
        }
    }
    Ok(CompletionsInput {
        model: model_name.to_string(),
        stream: *streaming,
        tools: tools_to_completions(&config.tools),
        messages,
        store: false,
        prompt_cache_key: prompt_cache_key.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::ModelKind;
    use llm_history_model::history::{Image, ToolCall as HistoryToolCall, ToolCallResult};

    fn config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::OpenAICompatible {
                prompt_cache_key: "prompt_cache_key".into(),
                model_name: "gpt-test".into(),
                streaming: false,
            },
            tools: Default::default(),
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

    fn serialize(entries: &[Entry]) -> serde_json::Value {
        serde_json::to_value(to_completions_input(entries, &config()).expect("valid history"))
            .unwrap()
    }

    #[test]
    fn plain_conversation() {
        let v = serialize(&[
            Entry::SystemPrompt {
                content: "Be terse.".into(),
            },
            text_user("hi"),
            Entry::ResponseNormal {
                thinking: None,
                content: "Hello!".into(),
            },
        ]);

        assert_eq!(
            v,
            serde_json::json!({
                "store": false,
                "model": "gpt-test",
                "stream": false,
                "prompt_cache_key": "prompt_cache_key",
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "Hello!"}
                ]
            })
        );
    }

    #[test]
    fn tool_call_and_result() {
        let v = serialize(&[
            text_user("weather?"),
            Entry::ResponseToolCall {
                thinking: None,
                tool_call: vec![HistoryToolCall {
                    tool_name: "get_weather".into(),
                    toolcall_params: r#"{"city":"Oslo"}"#.into(),
                    tool_use_id: "call_1".into(),
                }],
            },
            Entry::ToolCallResult {
                tool_call: vec![ToolCallResult {
                    result: "sunny".into(),
                    is_error: true,
                    tool_use_id: "call_1".into(),
                }],
            },
        ]);

        assert_eq!(
            v["messages"],
            serde_json::json!([
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ERROR: sunny"}
            ])
        );
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
                description: Some("Get the weather".to_owned()),
                parameters,
            }],
            ..config()
        };

        let v = serde_json::to_value(
            to_completions_input(&[text_user("hi")], &cfg).expect("valid history"),
        )
        .unwrap();

        assert_eq!(
            v["tools"],
            serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"],
                        "additionalProperties": false
                    }
                }
            }])
        );
    }

    #[test]
    fn image_user_message_becomes_parts() {
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
            v["messages"][0]["content"],
            serde_json::json!([
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}
            ])
        );
    }

    #[test]
    fn thinking_is_preserved_as_reasoning_content() {
        let v = serialize(&[
            text_user("q"),
            Entry::ResponseNormal {
                thinking: Some(llm_history_model::history::Thinking::Secret {
                    thinking: "secret".into(),
                    signature: "sig".into(),
                }),
                content: "answer".into(),
            },
        ]);
        assert_eq!(v["messages"][1]["content"], "answer");
        assert_eq!(v["messages"][1]["reasoning_content"], "secret");

        // tool-call assistant messages also carry reasoning_content
        let v = serialize(&[
            text_user("q"),
            Entry::ResponseToolCall {
                thinking: Some(llm_history_model::history::Thinking::Normal {
                    thinking: "need to check".into(),
                }),
                tool_call: vec![HistoryToolCall {
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
        assert_eq!(v["messages"][1]["reasoning_content"], "need to check");
    }
}
