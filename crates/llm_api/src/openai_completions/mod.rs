//! Serialization of [`crate::history::Entry`] histories into the legacy
//! OpenAI-compatible **Chat Completions API** (`POST /chat/completions`) wire
//! format.
//!
//! Mapping overview:
//! - `SystemPrompt`      -> `{role:"system", content}` message
//! - `SystemMessage`     -> `{role:"system"|"developer", content}` at its position
//! - `UserMessage`       -> `{role:"user"}` (string content, or content parts
//!   when an image is attached)
//! - `Response`          -> `{role:"assistant", reasoning_content?, content}`
//!   message (ordered blocks are flattened because the API has no thinking
//!   blocks; text-only and thinking-only responses use `content:""` and
//!   `reasoning_content` respectively)
//! - `ResponseToolCall`  -> ONE `{role:"assistant", tool_calls:[...]}` message
//!   (parallel calls grouped via `tool_calls`)
//! - `ToolCallResult`    -> one `{role:"tool", tool_call_id, content}` per result
//!
//! Notes:
//! - Plaintext `Thinking` is preserved as `reasoning_content` on assistant
//!   messages (both plain and tool-call ones). `Reasoning` content is a
//!   compatible-server extension, not part of the OpenAI schema.
//! - OpenAI `Thinking::Encrypted` has no Chat Completions field. It is skipped
//!   when plaintext reasoning is also available and rejected by
//!   [`to_completions_input`] when it is the only content of a turn.
//! - `is_error` has no native flag; errors are prefixed `"ERROR: "`.
//! - When `streaming` is enabled, `stream_options.include_usage` is sent
//!   (unless `include_usage` is disabled in `ModelKind`) so the server emits a
//!   final usage-only chunk.

use crate::model_config::ModelConfig;
use crate::validation::validate_entries;
use crate::wire_common::{mime_str, thinking_joined};
use llm_history_model::history::{
    Entry, MessageContent, ResponseBlock, SystemMessageKind, Thinking,
};
use serde::Serialize;

pub mod response;
pub mod stream;
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
    /// Asks the server for a final usage-only chunk when streaming.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Function tools forwarded from `ModelConfig.tools`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<CompletionTool>,
    pub messages: Vec<Message>,
}

#[derive(Debug, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
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
    Developer,
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

/// Flattens ordered response blocks into Chat Completions'
/// `reasoning_content` + `content` pair. Block-level interleaving cannot be
/// represented by this API.
fn flatten_blocks(blocks: &[ResponseBlock]) -> (Option<String>, String) {
    let thinking: Vec<Thinking> = blocks
        .iter()
        .filter_map(ResponseBlock::thinking)
        .cloned()
        .collect();
    let text = blocks
        .iter()
        .filter_map(ResponseBlock::text)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    (thinking_joined(&thinking), text)
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
        include_usage,
        ..
    } = &config.kind
    else {
        return Err(vec![
            crate::validation::ValidationError::WrongProviderKind {
                expected: "openai_compatible",
                actual: config.kind.name(),
            },
        ]);
    };

    let mut messages: Vec<Message> = Vec::new();

    for (i, entry) in entries.iter().enumerate() {
        match entry {
            Entry::SystemPrompt { content } => messages.push(Message {
                role: Role::System,
                content: Some(Content::Text(content.clone())),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            }),

            Entry::Draft { .. } => {
                return Err(vec![crate::validation::ValidationError::DraftNotFinal {
                    index: i,
                }]);
            }

            Entry::SystemMessage { kind, content } => messages.push(Message {
                role: if matches!(kind, SystemMessageKind::Developer) {
                    Role::Developer
                } else {
                    Role::System
                },
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

            Entry::Response { blocks } => {
                let (reasoning, text) = flatten_blocks(blocks);

                if text.is_empty() {
                    // An opaque-only turn (for example OpenAI
                    // `encrypted_content`) has no Chat Completions field.
                    let Some(reasoning) = reasoning else {
                        return Err(vec![
                            crate::validation::ValidationError::IncompatibleThinking {
                                index: i,
                                provider: "openai_chat_completions",
                                kind: "opaque",
                            },
                        ]);
                    };
                    messages.push(Message {
                        role: Role::Assistant,
                        // Chat Completions requires `content` unless tool calls
                        // are present; an empty string keeps it schema-valid.
                        content: Some(Content::Text(String::new())),
                        reasoning_content: Some(reasoning),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                } else {
                    messages.push(Message {
                        role: Role::Assistant,
                        content: Some(Content::Text(text)),
                        reasoning_content: reasoning,
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                }
            }

            Entry::ResponseToolCall { blocks, tool_call } => {
                let (reasoning, text) = flatten_blocks(blocks);
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
                    content: (!text.is_empty()).then_some(Content::Text(text)),
                    reasoning_content: reasoning,
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
        stream_options: (*streaming && *include_usage).then_some(StreamOptions {
            include_usage: true,
        }),
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
    use llm_history_model::history::ImageMIME;
    use llm_history_model::history::{
        Image, ResponseBlock, ToolCall as HistoryToolCall, ToolCallResult,
    };

    fn config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::OpenAICompatible {
                prompt_cache_key: "prompt_cache_key".into(),
                model_name: "gpt-test".into(),
                streaming: false,
                provider: crate::usage::UsageProvider::OpenAI,
                include_usage: true,
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
            Entry::response_text("Hello!"),
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
                blocks: Vec::new(),
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
            Entry::response(vec![
                ResponseBlock::Thinking(llm_history_model::history::Thinking::Signed {
                    thinking: "secret".into(),
                    signature: "sig".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]),
        ]);
        assert_eq!(v["messages"][1]["content"], "answer");
        assert_eq!(v["messages"][1]["reasoning_content"], "secret");

        // tool-call assistant messages also carry reasoning_content and content
        let v = serialize(&[
            text_user("q"),
            Entry::ResponseToolCall {
                blocks: vec![
                    ResponseBlock::Thinking(llm_history_model::history::Thinking::Normal {
                        thinking: "need to check".into(),
                    }),
                    ResponseBlock::Text("looking into it".into()),
                ],
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
        assert_eq!(v["messages"][1]["content"], "looking into it");
    }

    #[test]
    fn system_and_environment_messages_are_forwarded() {
        let v = serialize(&[
            Entry::SystemPrompt {
                content: "sys".into(),
            },
            text_user("hi"),
            Entry::response_text("hello"),
            Entry::SystemMessage {
                kind: SystemMessageKind::Environment,
                content: "workspace changed".into(),
            },
            Entry::SystemMessage {
                kind: SystemMessageKind::Developer,
                content: "use tabs".into(),
            },
        ]);

        assert_eq!(
            v["messages"][3],
            serde_json::json!({"role": "system", "content": "workspace changed"})
        );
        assert_eq!(
            v["messages"][4],
            serde_json::json!({"role": "developer", "content": "use tabs"})
        );
    }

    #[test]
    fn thinking_only_entry_uses_reasoning_content() {
        let v = serialize(&[
            text_user("q"),
            Entry::response_thinking(vec![llm_history_model::history::Thinking::Normal {
                thinking: "still thinking".into(),
            }]),
        ]);

        assert_eq!(v["messages"][1]["role"], "assistant");
        assert_eq!(v["messages"][1]["reasoning_content"], "still thinking");
        assert_eq!(v["messages"][1]["content"], "");
    }

    #[test]
    fn encrypted_only_thinking_is_rejected_for_chat_completions() {
        let err = to_completions_input(
            &[
                text_user("q"),
                Entry::response_thinking(vec![llm_history_model::history::Thinking::Encrypted {
                    thinking: "opaque".into(),
                }]),
            ],
            &config(),
        )
        .unwrap_err();

        assert!(
            err.contains(&crate::validation::ValidationError::IncompatibleThinking {
                index: 1,
                provider: "openai_chat_completions",
                kind: "opaque",
            })
        );
    }

    fn streaming_config(include_usage: bool) -> ModelConfig {
        ModelConfig {
            kind: ModelKind::OpenAICompatible {
                model_name: "gpt-test".into(),
                streaming: true,
                prompt_cache_key: "key".into(),
                provider: crate::usage::UsageProvider::OpenAI,
                include_usage,
            },
            tools: Default::default(),
            max_tokens: None,
        }
    }

    #[test]
    fn user_message_after_tool_result_is_serialized_in_order() {
        let v = serialize(&[
            text_user("q"),
            Entry::ResponseToolCall {
                blocks: Vec::new(),
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
            text_user("more"),
            Entry::response_text("ok"),
        ]);

        assert_eq!(v["messages"][2]["role"], "tool");
        assert_eq!(v["messages"][2]["tool_call_id"], "c1");
        assert_eq!(v["messages"][3]["role"], "user");
        assert_eq!(v["messages"][3]["content"], "more");
    }

    #[test]
    fn streaming_requests_include_usage() {
        let v = serde_json::to_value(
            to_completions_input(&[text_user("hi")], &streaming_config(true))
                .expect("valid history"),
        )
        .unwrap();

        assert_eq!(v["stream"], true);
        assert_eq!(
            v["stream_options"],
            serde_json::json!({"include_usage": true})
        );
    }

    #[test]
    fn non_streaming_requests_omit_stream_options() {
        let v = serialize(&[text_user("hi")]);
        assert_eq!(v["stream"], false);
        assert!(v.get("stream_options").is_none());
    }

    #[test]
    fn include_usage_can_be_disabled() {
        let v = serde_json::to_value(
            to_completions_input(&[text_user("hi")], &streaming_config(false))
                .expect("valid history"),
        )
        .unwrap();
        assert!(v.get("stream_options").is_none());
    }

    #[test]
    fn non_openai_kind_is_rejected() {
        let cfg = ModelConfig {
            kind: ModelKind::AnthropicCompatible {
                model_name: "claude".into(),
                streaming: false,
                thinking: None,
                allow_unsigned_thinking: false,
                system_message_policy: Default::default(),
            },
            tools: Default::default(),
            max_tokens: None,
        };
        assert!(matches!(
            to_completions_input(&[text_user("hi")], &cfg),
            Err(e) if e.contains(&crate::validation::ValidationError::WrongProviderKind {
                expected: "openai_compatible",
                actual: "anthropic_compatible",
            })
        ));
    }
}
