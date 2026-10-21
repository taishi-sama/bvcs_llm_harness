//! Parsing of Anthropic-compatible **Messages API** response bodies
//! (`POST /v1/messages`) back into [`llm_history_model::history::Entry`]
//! histories.
//!
//! Mapping overview (inverse of [`super`]'s serialization):
//! - `thinking` blocks          -> `Thinking::Signed` (verbatim `thinking` +
//!   `signature`, all blocks in order) attached to the assistant entry
//! - `redacted_thinking` blocks -> `Thinking::Redacted` (opaque payload in
//!   `thinking` field), now round-trippable via the serializer
//! - `thinking`/`text` blocks   -> ordered [`ResponseBlock`]s in
//!   `Entry::Response`, preserving interleaving
//! - `tool_use` blocks          -> merged into one `Entry::ResponseToolCall`
//!   (adjacent thinking/text is flattened into its legacy fields)
//! - other blocks               -> skipped
//!
//! Notes:
//! - Assistant thinking/text accompanying `tool_use` blocks becomes the
//!   entry's `thinking`/`content`.
//! - A thinking block without a signature is preserved as
//!   [`Thinking::Normal`]; such a block cannot be replayed to Anthropic and is
//!   reported by validation.

use llm_history_model::history::{Entry, ResponseBlock, Thinking, ToolCall};
use serde::Deserialize;
use serde_json::Value;

/// Top-level `POST /v1/messages` response body.
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesResponse {
    /// Optional for compatibility with lightweight compatible servers.
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    #[serde(default)]
    pub model: Option<String>,
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
    Thinking {
        thinking: String,
        /// Some compatible endpoints stream thinking without a signature.
        #[serde(default)]
        signature: String,
    },
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
    /// - thinking and text blocks are preserved in their original order as
    ///   [`Entry::Response`] blocks
    /// - `tool_use` blocks merge into a single [`Entry::ResponseToolCall`]
    ///   (its thinking/text fields are flattened because tool calls still use
    ///   the legacy split representation)
    ///
    /// The result contains assistant-side entries only — append them to an
    /// existing user-first history before validation or re-serialization.
    pub fn to_entries(&self) -> Vec<Entry> {
        let mut blocks: Vec<ResponseBlock> = Vec::new();
        let mut calls: Vec<ToolCall> = Vec::new();

        for block in &self.content {
            match block {
                ContentBlock::Other => {}
                ContentBlock::RedactedThinking { data } if !data.is_empty() => {
                    blocks.push(ResponseBlock::Thinking(Thinking::Redacted {
                        thinking: data.clone(),
                    }));
                }
                ContentBlock::Thinking {
                    thinking: text,
                    signature,
                } => {
                    // Preserve every signed block verbatim in order; merging
                    // would break the signatures. Compatible endpoints may omit
                    // the signature, in which case the block is plaintext
                    // thinking that Anthropic itself refuses to replay (the
                    // validation layer reports that).
                    let thinking = if signature.is_empty() {
                        if text.is_empty() {
                            None
                        } else {
                            Some(Thinking::Normal {
                                thinking: text.clone(),
                            })
                        }
                    } else {
                        Some(Thinking::Signed {
                            thinking: text.clone(),
                            signature: signature.clone(),
                        })
                    };
                    if let Some(thinking) = thinking {
                        blocks.push(ResponseBlock::Thinking(thinking));
                    }
                }
                ContentBlock::Text { text } if !text.is_empty() => {
                    blocks.push(ResponseBlock::Text(text.clone()));
                }
                ContentBlock::Text { .. } | ContentBlock::RedactedThinking { .. } => {}
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

        blocks.retain(|block| !block.is_empty());

        if !calls.is_empty() {
            let thinking = blocks
                .iter()
                .filter_map(ResponseBlock::thinking)
                .cloned()
                .collect();
            let content = {
                let joined = blocks
                    .iter()
                    .filter_map(ResponseBlock::text)
                    .filter(|text| !text.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");
                if joined.is_empty() {
                    None
                } else {
                    Some(joined)
                }
            };
            return vec![Entry::ResponseToolCall {
                thinking,
                content,
                tool_call: calls,
            }];
        }
        if blocks.is_empty() {
            return Vec::new();
        }
        vec![Entry::Response { blocks }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::{ModelConfig, ModelKind};
    use crate::validation::validate_entries;
    use llm_history_model::history::Entry as E;
    use llm_history_model::history::{ResponseBlock, ToolCallResult};

    fn config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::AnthropicCompatible {
                model_name: "claude-test".into(),
                streaming: false,
                thinking: None,
                allow_unsigned_thinking: false,
            },
            tools: Default::default(),
            max_tokens: Some(2048),
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
        assert_eq!(entries[0].response_joined_text().as_deref(), Some("Hello!"));
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
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
        assert_eq!(entries[0].response_joined_text().as_deref(), Some("done"));
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Redacted { thinking }] if thinking == "opaque"
        ));
        validated_history(entries);
    }

    #[test]
    fn missing_metadata_is_tolerated() {
        let resp = MessagesResponse::from_value(serde_json::json!({
            "content": [{"type": "text", "text": "Hi"}]
        }))
        .expect("parses without id/model");

        assert!(resp.id.is_none());
        assert!(resp.model.is_none());
        assert_eq!(resp.to_entries().len(), 1);
    }

    #[test]
    fn thinking_only_response_becomes_thinking_only_entry() {
        let resp = MessagesResponse::from_value(serde_json::json!({
            "id": "msg_4", "type": "message", "role": "assistant", "model": "m",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"}
            ]
        }))
        .expect("parses");
        let entries = resp.to_entries();
        assert_eq!(entries.len(), 1);
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Signed { thinking, signature }]
                if thinking == "hmm" && signature == "sig"
        ));
        assert!(entries[0].response_text_blocks().is_empty());
    }

    #[test]
    fn unsigned_thinking_block_is_parsed_as_normal() {
        let resp = MessagesResponse::from_value(serde_json::json!({
            "id": "msg_u", "type": "message", "role": "assistant", "model": "m",
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "answer"}
            ]
        }))
        .expect("parses");
        let entries = resp.to_entries();
        assert_eq!(entries[0].response_joined_text().as_deref(), Some("answer"));
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Normal { thinking }] if thinking == "hmm"
        ));
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
        assert_eq!(
            entries[0].as_response_blocks().expect("expected Response"),
            [
                ResponseBlock::Thinking(Thinking::Signed {
                    thinking: "one".into(),
                    signature: "s1".into(),
                }),
                ResponseBlock::Thinking(Thinking::Signed {
                    thinking: "two".into(),
                    signature: "s2".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]
        );
        validated_history(entries);
    }

    /// Serialize a history into the request, feed the assistant content blocks
    /// back through the response parser, and confirm re-serializing reproduces
    /// the original assistant message verbatim.
    fn assert_round_trip(history: &[E]) {
        use crate::anthropic::to_anthropic_messages;

        let input = to_anthropic_messages(history, &config()).expect("valid history");
        let assistant = serde_json::to_value(&input.messages[1]).unwrap();
        let response = MessagesResponse::from_value(serde_json::json!({
            "id": "msg_rt", "type": "message", "role": "assistant",
            "model": "claude-test", "stop_reason": "end_turn",
            "content": assistant["content"],
        }))
        .expect("parses");

        let mut full = vec![history[0].clone()];
        full.extend(response.to_entries());
        // Client-side entries (tool results) replay unchanged.
        full.extend_from_slice(&history[2..]);
        let reinput = to_anthropic_messages(&full, &config()).expect("valid history");
        assert_eq!(
            serde_json::to_value(&reinput.messages[1]).unwrap(),
            assistant
        );
    }

    #[test]
    fn round_trip_text_with_signed_thinking() {
        assert_round_trip(&[
            text_user("q"),
            E::response(vec![
                ResponseBlock::Thinking(Thinking::Signed {
                    thinking: "hmm".into(),
                    signature: "sig==".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]),
        ]);
    }

    #[test]
    fn round_trip_interleaved_thinking_and_text() {
        assert_round_trip(&[
            text_user("q"),
            E::response(vec![
                ResponseBlock::Thinking(Thinking::Signed {
                    thinking: "first plan".into(),
                    signature: "s1".into(),
                }),
                ResponseBlock::Text("first part".into()),
                ResponseBlock::Thinking(Thinking::Signed {
                    thinking: "second plan".into(),
                    signature: "s2".into(),
                }),
                ResponseBlock::Text("second part".into()),
            ]),
        ]);
    }

    #[test]
    fn round_trip_redacted_thinking() {
        assert_round_trip(&[
            text_user("q"),
            E::response(vec![
                ResponseBlock::Thinking(Thinking::Redacted {
                    thinking: "opaque".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]),
        ]);
    }

    #[test]
    fn round_trip_tool_call_with_preamble_text() {
        assert_round_trip(&[
            text_user("q"),
            E::ResponseToolCall {
                thinking: vec![Thinking::Signed {
                    thinking: "plan".into(),
                    signature: "s1".into(),
                }],
                content: Some("calling now".into()),
                tool_call: vec![
                    ToolCall {
                        tool_name: "get_stock_price".into(),
                        toolcall_params: r#"{"ticker":"AAPL"}"#.into(),
                        tool_use_id: "toolu_1".into(),
                    },
                    ToolCall {
                        tool_name: "b".into(),
                        toolcall_params: "{}".into(),
                        tool_use_id: "toolu_2".into(),
                    },
                ],
            },
            E::ToolCallResult {
                tool_call: vec![
                    ToolCallResult {
                        result: "r1".into(),
                        is_error: false,
                        tool_use_id: "toolu_1".into(),
                    },
                    ToolCallResult {
                        result: "r2".into(),
                        is_error: false,
                        tool_use_id: "toolu_2".into(),
                    },
                ],
            },
        ]);
    }
}
