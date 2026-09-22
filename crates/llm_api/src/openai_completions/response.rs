//! Parsing of legacy OpenAI-compatible **Chat Completions API** response
//! bodies (`POST /chat/completions`) back into
//! [`llm_history_model::history::Entry`] histories.
//!
//! Mapping overview (inverse of [`super`]'s serialization):
//! - assistant message with `tool_calls[]` -> `Entry::ResponseToolCall`
//! - assistant message with `content`      -> `Entry::ResponseNormal`
//! - `n` > 1 produces multiple choices; only `index` 0 is honored
//!
//! Tool results arrive as later `role:"tool"` messages produced client-side, so
//! this parser yields assistant-side entries only. `reasoning_content` is
//! attached as [`Thinking::Normal`] to the produced assistant entry.

use llm_history_model::history::{Entry, Thinking, ToolCall};
use serde::Deserialize;

/// Top-level `POST /chat/completions` response body.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    #[serde(default)]
    pub choices: Vec<Choice>,
    pub created: Option<i64>,
    pub model: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub index: Option<u32>,
    pub finish_reason: Option<String>,
    pub message: AssistantMessage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssistantMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub refusal: Option<String>,
    /// Reasoning text produced by the model (`reasoning_content`).
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<MessageToolCall>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MessageToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: MessageToolCallFunction,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MessageToolCallFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

impl CompletionResponse {
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn from_value(value: serde_json::Value) -> serde_json::Result<Self> {
        serde_json::from_value(value)
    }

    /// Converts the response's first choice back into history entries.
    ///
    /// - `tool_calls[]` merge into a single [`Entry::ResponseToolCall`]
    /// - otherwise the message text becomes an [`Entry::ResponseNormal`]
    ///   (`content` and `refusal` are concatenated; empty text is skipped)
    ///
    /// The result contains assistant-side entries only — append them to an
    /// existing user-first history before validation or re-serialization.
    pub fn to_entries(&self) -> Vec<Entry> {
        let Some(choice) = self.choices.first() else {
            return Vec::new();
        };
        let msg = &choice.message;

        let thinking: Vec<Thinking> = msg
            .reasoning_content
            .as_deref()
            .filter(|t| !t.is_empty())
            .map(|t| Thinking::Normal {
                thinking: t.to_owned(),
            })
            .into_iter()
            .collect();

        // Assistant text is valid alongside `tool_calls` in this API.
        let content: Option<String> = msg
            .content
            .as_deref()
            .filter(|t| !t.is_empty())
            .map(str::to_owned);

        if !msg.tool_calls.is_empty() {
            let tool_call = msg
                .tool_calls
                .iter()
                .map(|tc| ToolCall {
                    tool_name: tc.function.name.clone(),
                    toolcall_params: if tc.function.arguments.trim().is_empty() {
                        "{}".to_owned()
                    } else {
                        tc.function.arguments.clone()
                    },
                    tool_use_id: tc.id.clone(),
                })
                .collect();
            return vec![Entry::ResponseToolCall {
                thinking,
                content,
                tool_call,
            }];
        }

        let mut text_parts: Vec<&str> = Vec::new();
        if let Some(content) = msg.content.as_deref()
            && !content.is_empty()
        {
            text_parts.push(content);
        }
        if let Some(refusal) = msg.refusal.as_deref()
            && !refusal.is_empty()
        {
            text_parts.push(refusal);
        }
        if text_parts.is_empty() {
            return Vec::new();
        }

        vec![Entry::ResponseNormal {
            thinking,
            content: text_parts.join("\n"),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::{ModelConfig, ModelKind};
    use crate::validation::validate_entries;
    use llm_history_model::history::Entry as E;
    use llm_history_model::history::ToolCallResult;

    fn config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::OpenAICompatible {
                model_name: "gpt-test".into(),
                streaming: false,
                prompt_cache_key: "key".into(),
            },
            tools: Default::default(),
            max_tokens: None,
        }
    }

    fn text_user(t: &str) -> E {
        E::UserMessage {
            content: llm_history_model::history::MessageContent {
                text: Some(t.into()),
                image: None,
            },
        }
    }

    fn validated_history(entries: Vec<E>) {
        let mut history = vec![text_user("hi")];
        history.extend(entries);
        assert_eq!(validate_entries(&history, &config()), Ok(()));
    }

    #[test]
    fn parses_plain_text_message() {
        let resp = CompletionResponse::from_value(serde_json::json!({
            "id": "x", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello!",
                            "reasoning_content": "thinking here"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }))
        .expect("parses");

        let entries = resp.to_entries();
        assert_eq!(entries.len(), 1);
        let E::ResponseNormal { content, thinking } = &entries[0] else {
            panic!("expected ResponseNormal");
        };
        assert_eq!(content, "Hello!");
        assert!(matches!(
            thinking.as_slice(),
            [Thinking::Normal { thinking }] if thinking == "thinking here"
        ));
        validated_history(entries);
    }

    #[test]
    fn parses_tool_calls_into_one_entry() {
        let resp = CompletionResponse::from_value(serde_json::json!({
            "id": "x", "model": "m",
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": null,
                            "reasoning_content": "planning",
                            "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{\"x\":1}"}},
                    {"id": "c2", "type": "function", "function": {"name": "b", "arguments": ""}}
                ]}}]
        }))
        .expect("parses");

        let entries = resp.to_entries();
        assert_eq!(entries.len(), 1);
        let E::ResponseToolCall {
            thinking,
            content,
            tool_call,
        } = &entries[0]
        else {
            panic!("expected ResponseToolCall");
        };
        assert!(matches!(
            thinking.as_slice(),
            [Thinking::Normal { thinking }] if thinking == "planning"
        ));
        assert_eq!(*content, None);
        assert_eq!(tool_call.len(), 2);
        assert_eq!(tool_call[0].tool_name, "a");
        assert_eq!(tool_call[1].toolcall_params, "{}"); // empty arguments defaulted
        // NB: not validated — tool results arrive as later client-side entries
    }

    #[test]
    fn empty_response_yields_no_entries() {
        let resp = CompletionResponse::from_value(serde_json::json!({
            "id": "x", "model": "m", "choices": []
        }))
        .expect("parses");
        assert!(resp.to_entries().is_empty());
    }

    #[test]
    fn content_and_refusal_concatenated() {
        let resp = CompletionResponse::from_value(serde_json::json!({
            "id": "x", "model": "m",
            "choices": [{"index": 0, "finish_reason": "content_filter",
                "message": {"role": "assistant", "content": "partial", "refusal": "blocked"}}]
        }))
        .expect("parses");
        let entries = resp.to_entries();
        assert_eq!(entries.len(), 1);
        let E::ResponseNormal { content, .. } = &entries[0] else {
            panic!();
        };
        assert_eq!(content, "partial\nblocked");
    }

    /// Serialize a history into the request, feed the assistant message back
    /// through the response parser, and confirm re-serializing reproduces the
    /// original assistant message verbatim.
    fn assert_round_trip(history: &[E]) {
        use crate::openai_completions::to_completions_input;

        let input = to_completions_input(history, &config()).expect("valid history");
        let assistant = serde_json::to_value(&input.messages[1]).unwrap();
        let response = CompletionResponse::from_value(serde_json::json!({
            "id": "x", "model": "gpt-test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": assistant}]
        }))
        .expect("parses");

        let mut full = vec![history[0].clone()];
        full.extend(response.to_entries());
        // Client-side entries (tool results) replay unchanged.
        full.extend_from_slice(&history[2..]);
        let reinput = to_completions_input(&full, &config()).expect("valid history");
        assert_eq!(
            serde_json::to_value(&reinput.messages[1]).unwrap(),
            assistant
        );
    }

    #[test]
    fn round_trip_text_with_reasoning() {
        assert_round_trip(&[
            text_user("q"),
            E::ResponseNormal {
                thinking: vec![Thinking::Normal {
                    thinking: "hmm".into(),
                }],
                content: "answer".into(),
            },
        ]);
    }

    #[test]
    fn round_trip_tool_calls_with_preamble() {
        assert_round_trip(&[
            text_user("q"),
            E::ResponseToolCall {
                thinking: vec![Thinking::Normal {
                    thinking: "plan".into(),
                }],
                content: Some("calling now".into()),
                tool_call: vec![
                    ToolCall {
                        tool_name: "get_weather".into(),
                        toolcall_params: r#"{"city":"Oslo"}"#.into(),
                        tool_use_id: "c1".into(),
                    },
                    ToolCall {
                        tool_name: "b".into(),
                        toolcall_params: "{}".into(),
                        tool_use_id: "c2".into(),
                    },
                ],
            },
            E::ToolCallResult {
                tool_call: vec![
                    ToolCallResult {
                        result: "r1".into(),
                        is_error: false,
                        tool_use_id: "c1".into(),
                    },
                    ToolCallResult {
                        result: "r2".into(),
                        is_error: false,
                        tool_use_id: "c2".into(),
                    },
                ],
            },
        ]);
    }
}
