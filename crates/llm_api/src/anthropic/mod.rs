//! Serialization of [`crate::history::Entry`] histories into the
//! Anthropic-compatible **Messages API** (`POST /v1/messages`) wire format.
//!
//! Mapping overview:
//! - `SystemPrompt`      -> top-level `system` parameter (there is no `system`
//!   role in the Messages API)
//! - `UserMessage`       -> `{role:"user"}` (string shorthand for text-only,
//!   content blocks when an image is attached)
//! - `ResponseNormal`    -> `{role:"assistant"}` with `[thinking*, text]`
//! - `ResponseToolCall`  -> `{role:"assistant"}` with `[thinking*, text?,
//!   tool_use*]`
//! - `ToolCallResult`    -> `{role:"user"}` with `[tool_result*]`; consecutive
//!   batches merge into a single user message
//!
//! Notes:
//! - Thinking blocks carry `Thinking::Signed` verbatim (`thinking` +
//!   `signature`) and always come first in assistant content, as the API
//!   requires. `Thinking::Redacted` goes as `redacted_thinking` with the
//!   opaque payload in its `thinking` field. Unsigned `Thinking::Normal` is
//!   rejected by validation for the Anthropic kind.
//! - `max_tokens` is required by the API but lives outside `ModelConfig`, so
//!   it is a parameter of [`to_anthropic_messages`].

use crate::model_config::ModelConfig;
use crate::validation::{ValidationError, validate_entries};
use crate::wire_common::anthropic_media_type_str;
use llm_history_model::history::{Entry, Thinking};
use serde::Serialize;
use serde_json::Value;

pub mod response;
pub mod tools;

pub use tools::{AnthropicTool, tools_to_anthropic};

/// The body of a `POST /v1/messages` request.
#[derive(Debug, Serialize)]
pub struct MessagesInput<'a> {
    pub model: String,
    pub max_tokens: u32,
    /// System prompt, hoisted out of the message list (top-level parameter).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<&'a str>,
    /// Tool definitions forwarded from `ModelConfig.tools`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AnthropicTool>,
    pub messages: Vec<Message>,
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: Content,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// `content: string | [content blocks]` — string shorthand for text-only user
/// messages, blocks otherwise (assistant messages are always blocks).
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Blocks(Vec<Block>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
}

#[derive(Debug, Serialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub media_type: &'static str,
    pub data: String,
}

/// Pushes thinking blocks first in `blocks`, preserving order. `Signed` blocks
/// go verbatim (`thinking` + `signature`); `Redacted` blocks go as
/// `redacted_thinking` with the opaque payload carried in `Thinking::Redacted::thinking`.
/// Unsigned `Normal` thinking cannot be resent to the API; validation rejects
/// it for the Anthropic kind, so reaching this branch with `Normal` only
/// happens if validation was bypassed (it is then skipped).
fn push_thinking(blocks: &mut Vec<Block>, thinking: &[Thinking]) {
    for t in thinking {
        match t {
            Thinking::Signed {
                thinking,
                signature,
            } => blocks.push(Block::Thinking {
                thinking: thinking.clone(),
                signature: signature.clone(),
            }),
            Thinking::Redacted { thinking } => blocks.push(Block::RedactedThinking {
                data: thinking.clone(),
            }),
            Thinking::Normal { .. } => {}
        }
    }
}

/// Fallback `max_tokens` when `ModelConfig.max_tokens` is `None`. The API
/// requires the parameter; this default fits every current Claude model.
pub const DEFAULT_MAX_TOKENS: u32 = 262144;

/// Validates `entries` against `config` and converts them into the Messages
/// API request body.
///
/// `max_tokens` comes from `config.max_tokens` (`DEFAULT_MAX_TOKENS` when
/// unset), clamped to `u32`.
///
/// Returns all validation errors if the history is malformed or the config
/// kind is not Anthropic-compatible. Serialization is total otherwise.
pub fn to_anthropic_messages<'a>(
    entries: &'a [Entry],
    config: &ModelConfig,
) -> Result<MessagesInput<'a>, Vec<ValidationError>> {
    validate_entries(entries, config)?;

    let crate::model_config::ModelKind::AnthropicCompatible { model_name } = &config.kind else {
        return Err(vec![ValidationError::WrongProviderKind {
            expected: "anthropic_compatible",
            actual: config.kind.name(),
        }]);
    };

    let max_tokens = config
        .max_tokens
        .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
        .unwrap_or(DEFAULT_MAX_TOKENS);

    let mut system: Option<&str> = None;
    let mut messages: Vec<Message> = Vec::new();

    for (i, entry) in entries.iter().enumerate() {
        match entry {
            Entry::SystemPrompt { content } => system = Some(content),

            Entry::UserMessage { content } => {
                let message = match (&content.text, &content.image) {
                    (Some(text), None) => Message {
                        role: Role::User,
                        content: Content::Text(text.clone()),
                    },
                    (text, image) => {
                        let mut blocks = Vec::new();
                        if let Some(t) = text
                            && !t.is_empty()
                        {
                            blocks.push(Block::Text { text: t.clone() });
                        }
                        if let Some(img) = image {
                            blocks.push(Block::Image {
                                source: ImageSource {
                                    kind: "base64",
                                    media_type: anthropic_media_type_str(&img.mime),
                                    data: img.base64.clone(),
                                },
                            });
                        }
                        Message {
                            role: Role::User,
                            content: Content::Blocks(blocks),
                        }
                    }
                };
                messages.push(message);
            }

            Entry::ResponseNormal { thinking, content } => {
                let mut blocks = Vec::new();
                push_thinking(&mut blocks, thinking);
                blocks.push(Block::Text {
                    text: content.clone(),
                });
                messages.push(Message {
                    role: Role::Assistant,
                    content: Content::Blocks(blocks),
                });
            }

            Entry::ResponseToolCall {
                thinking,
                content,
                tool_call,
            } => {
                let mut blocks = Vec::new();
                push_thinking(&mut blocks, thinking);
                if let Some(text) = content
                    && !text.is_empty()
                {
                    blocks.push(Block::Text { text: text.clone() });
                }
                for (ci, tc) in tool_call.iter().enumerate() {
                    // `tool_use.input` must be a JSON object (`validate_entries`
                    // checks this too; mapped here so the error carries position).
                    let input = match serde_json::from_str::<Value>(&tc.toolcall_params) {
                        Ok(Value::Object(map)) => Value::Object(map),
                        _ => {
                            return Err(vec![ValidationError::InvalidToolParams {
                                entry: i,
                                call: ci,
                            }]);
                        }
                    };
                    blocks.push(Block::ToolUse {
                        id: tc.tool_use_id.clone(),
                        name: tc.tool_name.clone(),
                        input,
                    });
                }
                messages.push(Message {
                    role: Role::Assistant,
                    content: Content::Blocks(blocks),
                });
            }

            Entry::ToolCallResult { tool_call } => {
                let blocks: Vec<Block> = tool_call
                    .iter()
                    .map(|r| Block::ToolResult {
                        tool_use_id: r.tool_use_id.clone(),
                        content: r.result.clone(),
                        is_error: r.is_error,
                    })
                    .collect();
                // Consecutive result batches merge into a single user message.
                if let Some(Message {
                    role: Role::User,
                    content: Content::Blocks(existing),
                }) = messages.last_mut()
                {
                    existing.extend(blocks);
                } else {
                    messages.push(Message {
                        role: Role::User,
                        content: Content::Blocks(blocks),
                    });
                }
            }
        }
    }

    Ok(MessagesInput {
        model: model_name.to_string(),
        max_tokens,
        system,
        tools: tools_to_anthropic(&config.tools),
        messages,
        stream: config.kind.get_streamable(),
    })
}

#[cfg(test)]
mod tests {
    use llm_history_model::history::ImageMIME;
    use super::*;
    use crate::model_config::ModelKind;
    use llm_history_model::history::{
        Image, MessageContent, ToolCall as HistoryToolCall, ToolCallResult,
    };

    fn config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::AnthropicCompatible {
                model_name: "claude-test".into(),
            },
            tools: Default::default(),
            max_tokens: Some(1024),
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
        serde_json::to_value(to_anthropic_messages(entries, &config()).expect("valid history"))
            .unwrap()
    }

    #[test]
    fn system_prompt_hoisted_and_plain_conversation() {
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
            serde_json::json!({
                "model": "claude-test",
                "max_tokens": 1024,
                "stream": false,
                "system": "Be terse.",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]}
                ]
            })
        );
    }

    #[test]
    fn max_tokens_comes_from_config_with_fallback() {
        // Explicit value is forwarded as-is.
        assert_eq!(serialize(&[text_user("hi")])["max_tokens"], 1024);

        // Unset falls back to DEFAULT_MAX_TOKENS.
        let cfg = ModelConfig {
            max_tokens: None,
            ..config()
        };
        let v = serde_json::to_value(
            to_anthropic_messages(&[text_user("hi")], &cfg).expect("valid history"),
        )
        .unwrap();
        assert_eq!(v["max_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn signed_thinking_comes_first() {
        let v = serialize(&[
            text_user("q"),
            Entry::ResponseNormal {
                thinking: vec![Thinking::Signed {
                    thinking: "hmm".into(),
                    signature: "sig==".into(),
                }],
                content: "answer".into(),
            },
        ]);

        assert_eq!(
            v["messages"][1]["content"],
            serde_json::json!([
                {"type": "thinking", "thinking": "hmm", "signature": "sig=="},
                {"type": "text", "text": "answer"}
            ])
        );
    }

    #[test]
    fn redacted_thinking_and_tool_call_content() {
        let v = serialize(&[
            text_user("q"),
            Entry::ResponseToolCall {
                thinking: vec![
                    Thinking::Signed {
                        thinking: "hmm".into(),
                        signature: "sig==".into(),
                    },
                    Thinking::Redacted {
                        thinking: "opaque".into(),
                    },
                ],
                content: Some("calling now".into()),
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

        assert_eq!(
            v["messages"][1]["content"],
            serde_json::json!([
                {"type": "thinking", "thinking": "hmm", "signature": "sig=="},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "calling now"},
                {"type": "tool_use", "id": "c1", "name": "a", "input": {}}
            ])
        );
    }

    #[test]
    fn tool_round_trip_with_results() {
        let v = serialize(&[
            text_user("price?"),
            Entry::ResponseToolCall {
                thinking: Vec::new(),
                content: None,
                tool_call: vec![
                    HistoryToolCall {
                        tool_name: "get_stock_price".into(),
                        toolcall_params: r#"{"ticker":"AAPL"}"#.into(),
                        tool_use_id: "toolu_1".into(),
                    },
                    HistoryToolCall {
                        tool_name: "get_time".into(),
                        toolcall_params: "{}".into(),
                        tool_use_id: "toolu_2".into(),
                    },
                ],
            },
            Entry::ToolCallResult {
                tool_call: vec![
                    ToolCallResult {
                        result: "259.75 USD".into(),
                        is_error: false,
                        tool_use_id: "toolu_1".into(),
                    },
                    ToolCallResult {
                        result: "boom".into(),
                        is_error: true,
                        tool_use_id: "toolu_2".into(),
                    },
                ],
            },
        ]);

        assert_eq!(
            v["messages"],
            serde_json::json!([
                {"role": "user", "content": "price?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "get_stock_price", "input": {"ticker": "AAPL"}},
                    {"type": "tool_use", "id": "toolu_2", "name": "get_time", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "259.75 USD", "is_error": false},
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "boom", "is_error": true}
                ]}
            ])
        );
    }

    #[test]
    fn consecutive_result_batches_merge() {
        let v = serialize(&[
            text_user("q"),
            Entry::ResponseToolCall {
                thinking: Vec::new(),
                content: None,
                tool_call: vec![
                    HistoryToolCall {
                        tool_name: "a".into(),
                        toolcall_params: "{}".into(),
                        tool_use_id: "c1".into(),
                    },
                    HistoryToolCall {
                        tool_name: "b".into(),
                        toolcall_params: "{}".into(),
                        tool_use_id: "c2".into(),
                    },
                ],
            },
            Entry::ToolCallResult {
                tool_call: vec![ToolCallResult {
                    result: "r1".into(),
                    is_error: false,
                    tool_use_id: "c1".into(),
                }],
            },
            Entry::ToolCallResult {
                tool_call: vec![ToolCallResult {
                    result: "r2".into(),
                    is_error: false,
                    tool_use_id: "c2".into(),
                }],
            },
        ]);

        let messages = v["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn image_user_message_becomes_blocks() {
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
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}
            ])
        );
    }

    #[test]
    fn config_tools_are_forwarded() {
        let mut parameters = std::collections::HashMap::new();
        parameters.insert(
            "ticker".to_owned(),
            crate::generic_tool::ParameterDescription {
                r#type: crate::generic_tool::ParameterType::String,
                description: "Ticker symbol".to_owned(),
            },
        );
        let cfg = ModelConfig {
            tools: vec![crate::generic_tool::Tool {
                name: "get_stock_price".to_owned(),
                description: Some("Get the current stock price.".to_owned()),
                parameters,
            }],
            ..config()
        };

        let v = serde_json::to_value(
            to_anthropic_messages(&[text_user("hi")], &cfg).expect("valid history"),
        )
        .unwrap();

        assert_eq!(
            v["tools"],
            serde_json::json!([{
                "name": "get_stock_price",
                "description": "Get the current stock price.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Ticker symbol"}
                    },
                    "required": ["ticker"],
                    "additionalProperties": false
                }
            }])
        );
    }

    #[test]
    fn non_anthropic_kind_is_rejected() {
        let cfg = ModelConfig {
            kind: ModelKind::OpenAICompatible {
                model_name: "gpt".into(),
                streaming: false,
                prompt_cache_key: String::new(),
            },
            tools: Default::default(),
            max_tokens: None,
        };
        assert!(matches!(
            to_anthropic_messages(&[text_user("hi")], &cfg),
            Err(e) if e.contains(&ValidationError::WrongProviderKind {
                expected: "anthropic_compatible",
                actual: "openai_compatible",
            })
        ));
    }

    #[test]
    fn unsigned_thinking_rejected_for_anthropic() {
        let err = to_anthropic_messages(
            &[
                text_user("q"),
                Entry::ResponseNormal {
                    thinking: vec![Thinking::Normal {
                        thinking: "hmm".into(),
                    }],
                    content: "answer".into(),
                },
            ],
            &config(),
        )
        .unwrap_err();
        assert!(err.contains(&ValidationError::UnsignedThinkingRejected { index: 1 }));
    }
}
