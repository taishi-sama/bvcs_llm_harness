//! Parsing of Anthropic-compatible **Messages API** response bodies
//! (`POST /v1/messages`) back into [`llm_history_model::history::Entry`]
//! histories.
//!
//! Mapping overview (inverse of [`super`]'s serialization):
//! - `thinking` blocks          -> `Thinking::Signed` (verbatim `thinking` +
//!   `signature`, all blocks in order) attached to the assistant entry
//! - `redacted_thinking` blocks -> `Thinking::Redacted` (opaque payload in
//!   `thinking` field), now round-trippable via the serializer
//! - `text` blocks              -> `Entry::ResponseNormal` (parts joined)
//! - `tool_use` blocks          -> merged into one `Entry::ResponseToolCall`
//!   (adjacent text becomes its `content`)
//! - other blocks               -> skipped
//!
//! Notes:
//! - Assistant text accompanying `tool_use` blocks becomes the entry's
//!   `content`.
//! - Thinking-only responses yield no entries (empty assistant content fails
//!   validation).

use llm_history_model::history::{Entry, Thinking, ToolCall};
use serde::Deserialize;
use serde_json::Value;

/// Top-level `POST /v1/messages` response body.
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    pub model: String,
    #[serde(default)]
    pub role: Option<String>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    #[serde(default)]
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    #[serde(other)]
    Other,
}

impl MessagesResponse {
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn from_value(value: Value) -> serde_json::Result<Self> {
        serde_json::from_value(value)
    }

    /// Converts the response content back into history entries.
    ///
    /// - `tool_use` blocks merge into a single [`Entry::ResponseToolCall`]
    /// - otherwise the text parts become an [`Entry::ResponseNormal`]
    /// - the first `thinking` block attaches as [`Thinking::Secret`]
    /// - thinking-only responses yield no entries (empty assistant content
    ///   fails validation)
    ///
    /// The result contains assistant-side entries only — append them to an
    /// existing user-first history before validation or re-serialization.
    pub fn to_entries(&self) -> Vec<Entry> {
        let mut thinking: Vec<Thinking> = Vec::new();
        let mut texts: Vec<&str> = Vec::new();
        let mut calls: Vec<ToolCall> = Vec::new();

        for block in &self.content {
            match block {
                ContentBlock::Other => {}
                ContentBlock::RedactedThinking { data } => {
                    thinking.push(Thinking::Redacted {
                        thinking: data.clone(),
                    });
                }
                ContentBlock::Thinking {
                    thinking: text,
                    signature,
                } => {
                    // Preserve every block verbatim in order; merging would
                    // break the signatures.
                    thinking.push(Thinking::Signed {
                        thinking: text.clone(),
                        signature: signature.clone(),
                    });
                }
                ContentBlock::Text { text } => {
                    if !text.is_empty() {
                        texts.push(text.as_str());
                    }
                }
                ContentBlock::ToolUse { id, name, input } => {
                    calls.push(ToolCall {
                        tool_name: name.clone(),
                        toolcall_params: match input {
                            Value::Null => "{}".to_owned(),
                            Value::String(s) if s.trim().is_empty() => "{}".to_owned(),
                            Value::Object(_) | Value::Array(_) => input.to_string(),
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        },
                        tool_use_id: id.clone(),
                    });
                }
            }
        }

        let content = {
            let joined = texts.join("\n");
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        };

        if !calls.is_empty() {
            return vec![Entry::ResponseToolCall {
                thinking,
                content,
                tool_call: calls,
            }];
        }
        let Some(content) = content else {
            return Vec::new();
        };
        vec![Entry::ResponseNormal { thinking, content }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::{ModelConfig, ModelKind};
    use crate::validation::validate_entries;
    use llm_history_model::history::Entry as E;

    fn config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::AnthropicCompatible {
                model_name: "claude-test".into(),
            },
            tools: Default::default(),
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

    /// Response entries are assistant-side; they validate only when appended
    /// onto an existing user-first history.
    fn validated_history(entries: Vec<E>) {
        let mut history = vec![text_user("hi")];
        history.extend(entries);
        assert_eq!(validate_entries(&history, &config()), Ok(()));
    }

    #[test]
    fn parses_text_and_signed_thinking() {
        let resp = MessagesResponse::from_value(serde_json::json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig=="},
                {"type": "text", "text": "Hello!"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }))
        .expect("parses");

        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(resp.usage.as_ref().unwrap().input_tokens, Some(10));

        let entries = resp.to_entries();
        assert_eq!(entries.len(), 1);
        let E::ResponseNormal { content, thinking } = &entries[0] else {
            panic!("expected ResponseNormal");
        };
        assert_eq!(content, "Hello!");
        assert!(matches!(
            thinking.as_slice(),
            [Thinking::Signed { thinking, signature }]
                if thinking == "hmm" && signature == "sig=="
        ));
        validated_history(entries);
    }

    #[test]
    fn parses_tool_use_blocks_into_one_entry() {
        let resp = MessagesResponse::from_value(serde_json::json!({
            "id": "msg_2", "type": "message", "role": "assistant", "model": "m",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "plan", "signature": "s1"},
                {"type": "tool_use", "id": "toolu_1", "name": "a", "input": {"x": 1}},
                {"type": "tool_use", "id": "toolu_2", "name": "b"}
            ]
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
        assert!(matches!(thinking.as_slice(), [Thinking::Signed { .. }]));
        assert_eq!(*content, None);
        assert_eq!(tool_call.len(), 2);
        assert_eq!(tool_call[0].tool_name, "a");
        assert_eq!(tool_call[0].toolcall_params, r#"{"x":1}"#);
        assert_eq!(tool_call[1].toolcall_params, "{}"); // missing input defaulted
        // NB: not validated — tool results arrive as later client-side entries
    }

    #[test]
    fn redacted_thinking_round_trips_and_unknown_blocks_skipped() {
        let resp = MessagesResponse::from_value(serde_json::json!({
            "id": "msg_3", "type": "message", "role": "assistant", "model": "m",
            "content": [
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "server_tool_use", "id": "srvtoolu_1"},
                {"type": "text", "text": "done"}
            ]
        }))
        .expect("parses");

        let entries = resp.to_entries();
        assert_eq!(entries.len(), 1);
        let E::ResponseNormal { content, thinking } = &entries[0] else {
            panic!("expected ResponseNormal");
        };
        assert_eq!(content, "done");
        assert!(matches!(
            thinking.as_slice(),
            [Thinking::Redacted { thinking }] if thinking == "opaque"
        ));
        validated_history(entries);
    }

    #[test]
    fn thinking_only_response_yields_no_entries() {
        let resp = MessagesResponse::from_value(serde_json::json!({
            "id": "msg_4", "type": "message", "role": "assistant", "model": "m",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"}
            ]
        }))
        .expect("parses");
        assert!(resp.to_entries().is_empty());
    }

    #[test]
    fn preserves_all_thinking_blocks_in_order() {
        let resp = MessagesResponse::from_value(serde_json::json!({
            "id": "msg_5", "type": "message", "role": "assistant", "model": "m",
            "content": [
                {"type": "thinking", "thinking": "one", "signature": "s1"},
                {"type": "thinking", "thinking": "two", "signature": "s2"},
                {"type": "text", "text": "answer"}
            ]
        }))
        .expect("parses");

        let entries = resp.to_entries();
        let E::ResponseNormal { thinking, .. } = &entries[0] else {
            panic!("expected ResponseNormal");
        };
        assert!(matches!(
            thinking.as_slice(),
            [
                Thinking::Signed { thinking: t1, signature: s1 },
                Thinking::Signed { thinking: t2, signature: s2 }
            ] if t1 == "one" && s1 == "s1" && t2 == "two" && s2 == "s2"
        ));
        validated_history(entries);
    }
}
