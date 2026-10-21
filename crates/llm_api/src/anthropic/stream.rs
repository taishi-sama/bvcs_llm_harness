//! Streaming support for the **Anthropic Messages API** (`stream: true`).
//!
//! Anthropic sends typed SSE events:
//!
//! ```text
//! event: content_block_start
//! data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}
//!
//! event: content_block_delta
//! data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me"}}
//!
//! event: content_block_delta
//! data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"..."}}
//! ```
//!
//! The decoder mutates a [`Entry::Draft`] in place. Thinking text arrives before
//! its signature, which is why [`DraftBlock::Thinking::signature`] is optional.
//! Tool-call arguments arrive as `input_json_delta` fragments and are only valid
//! JSON once the block closes.

use crate::sse::SseDecoder;
use crate::stream::{StreamError, StreamEvent};
use llm_history_model::history::DraftBlock;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

/// Incremental Anthropic Messages stream decoder.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    sse: SseDecoder,
    /// Content-block `index` -> position in `blocks`.
    positions: HashMap<u32, usize>,
    /// Position in `blocks` of the currently open content block, used as a
    /// fallback when a compatible provider omits/rewrites indices.
    current: Option<usize>,
    /// Whether the current tool block was seeded with complete `input` in
    /// `content_block_start`; the first `input_json_delta` replaces it instead
    /// of appending to it.
    input_seeded: bool,
    stop_reason: Option<String>,
    completed: bool,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_completed(&self) -> bool {
        self.completed
    }

    /// Feed raw HTTP body bytes, mutating `blocks` as events arrive.
    ///
    /// Returns progress events for UI/logging. The caller owns the draft; this
    /// decoder never adds or removes history entries.
    pub fn push_bytes(
        &mut self,
        bytes: &[u8],
        blocks: &mut Vec<DraftBlock>,
    ) -> Result<Vec<StreamEvent>, StreamError> {
        let mut out = Vec::new();
        for sse in self.sse.push(bytes) {
            if sse.data == "[DONE]" {
                self.completed = true;
                out.push(StreamEvent::Completed {
                    stop_reason: self.stop_reason.clone(),
                });
                continue;
            }
            if sse.data.is_empty() {
                continue;
            }
            let event: Event =
                serde_json::from_str(&sse.data).map_err(|e| StreamError::InvalidJson {
                    event: sse.event.clone(),
                    details: e.to_string(),
                })?;
            self.handle(event, blocks, &mut out)?;
        }
        Ok(out)
    }

    /// Flush any pending SSE frame at end of stream.
    pub fn finish(
        &mut self,
        blocks: &mut Vec<DraftBlock>,
    ) -> Result<Vec<StreamEvent>, StreamError> {
        let mut out = Vec::new();
        for sse in self.sse.finish() {
            if sse.data.is_empty() {
                continue;
            }
            let event: Event =
                serde_json::from_str(&sse.data).map_err(|e| StreamError::InvalidJson {
                    event: sse.event.clone(),
                    details: e.to_string(),
                })?;
            self.handle(event, blocks, &mut out)?;
        }
        Ok(out)
    }

    fn handle(
        &mut self,
        event: Event,
        blocks: &mut Vec<DraftBlock>,
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), StreamError> {
        match event {
            Event::MessageStart | Event::Ping | Event::Other => {}

            Event::ContentBlockStart {
                index,
                content_block,
            } => {
                self.input_seeded = false;
                let pos = match content_block {
                    StartBlock::Text { text } => {
                        blocks.push(DraftBlock::Text { text });
                        Some(blocks.len() - 1)
                    }
                    StartBlock::Thinking {
                        thinking,
                        signature,
                    } => {
                        blocks.push(DraftBlock::Thinking {
                            text: thinking,
                            signature: (!signature.is_empty()).then_some(signature),
                            encrypted_content: None,
                        });
                        Some(blocks.len() - 1)
                    }
                    StartBlock::RedactedThinking { data } => {
                        blocks.push(DraftBlock::RedactedThinking { data });
                        Some(blocks.len() - 1)
                    }
                    StartBlock::ToolUse { id, name, input } => {
                        // `input` is normally `{}`; some compatible endpoints
                        // send the complete arguments upfront. If they then
                        // also stream `input_json_delta` fragments, the first
                        // fragment replaces the seeded value (see below).
                        let arguments = match &input {
                            Value::Object(map) if map.is_empty() => String::new(),
                            Value::Null => String::new(),
                            other => other.to_string(),
                        };
                        self.input_seeded = !arguments.is_empty();
                        blocks.push(DraftBlock::ToolCall {
                            index,
                            tool_use_id: Some(id),
                            tool_name: Some(name),
                            arguments,
                        });
                        Some(blocks.len() - 1)
                    }
                    StartBlock::Other => None,
                };
                if let Some(pos) = pos {
                    self.positions.insert(index, pos);
                }
                self.current = pos;
            }

            Event::ContentBlockDelta { index, delta } => {
                // Prefer the provider's block index; fall back to the last open
                // block for compatible servers that omit it.
                let Some(pos) = self.positions.get(&index).copied().or(self.current) else {
                    return Ok(());
                };
                let Some(block) = blocks.get_mut(pos) else {
                    return Ok(());
                };
                match (delta, block) {
                    (Delta::Text { text }, DraftBlock::Text { text: existing }) => {
                        existing.push_str(&text);
                        out.push(StreamEvent::TextDelta(text));
                    }
                    (Delta::Thinking { thinking }, DraftBlock::Thinking { text, .. }) => {
                        text.push_str(&thinking);
                        out.push(StreamEvent::ThinkingDelta(thinking));
                    }
                    (
                        Delta::Signature { signature },
                        DraftBlock::Thinking {
                            signature: slot, ..
                        },
                    ) => {
                        *slot = (!signature.is_empty()).then_some(signature.clone());
                        out.push(StreamEvent::SignatureDelta(signature));
                    }
                    (
                        Delta::InputJson { partial_json },
                        DraftBlock::ToolCall {
                            index: call_index,
                            arguments,
                            ..
                        },
                    ) => {
                        if self.input_seeded {
                            // The start block already carried the complete
                            // JSON; do not concatenate a second copy.
                            arguments.clear();
                            self.input_seeded = false;
                        }
                        arguments.push_str(&partial_json);
                        out.push(StreamEvent::ToolCallDelta {
                            index: *call_index,
                            id: None,
                            name: None,
                            arguments: partial_json,
                        });
                    }
                    // Citations and unknown deltas do not affect history.
                    _ => {}
                }
            }

            Event::ContentBlockStop { index } => {
                self.positions.remove(&index);
                self.current = None;
                self.input_seeded = false;
            }

            Event::MessageDelta { delta } => {
                if delta.stop_reason.is_some() {
                    self.stop_reason = delta.stop_reason;
                }
            }

            Event::MessageStop => {
                self.completed = true;
                out.push(StreamEvent::Completed {
                    stop_reason: self.stop_reason.clone(),
                });
            }

            Event::Error { error } => {
                return Err(StreamError::Provider {
                    message: error
                        .message
                        .unwrap_or_else(|| "unknown Anthropic stream error".into()),
                });
            }
        }
        Ok(())
    }
}

/// Top-level payload of an Anthropic streaming event.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event {
    MessageStart,
    ContentBlockStart {
        index: u32,
        content_block: StartBlock,
    },
    ContentBlockDelta {
        index: u32,
        delta: Delta,
    },
    ContentBlockStop {
        #[allow(dead_code)]
        index: u32,
    },
    MessageDelta {
        delta: MessageDeltaBody,
    },
    MessageStop,
    Ping,
    Error {
        error: ApiError,
    },
    /// Forward compatibility: ignore event types we do not model.
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StartBlock {
    Text {
        #[serde(default)]
        text: String,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Delta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaBody {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    #[serde(default)]
    message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::{ModelConfig, ModelKind};
    use crate::stream::{finish_draft, finish_draft_for};
    use llm_history_model::history::{Entry, MessageContent, Thinking};

    fn config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::AnthropicCompatible {
                model_name: "claude-test".into(),
                streaming: true,
                thinking: None,
                allow_unsigned_thinking: false,
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

    fn feed(decoder: &mut StreamDecoder, raw: &str) -> Vec<DraftBlock> {
        let mut blocks = Vec::new();
        // Split into tiny chunks to exercise SSE buffering.
        for chunk in raw.as_bytes().chunks(7) {
            decoder.push_bytes(chunk, &mut blocks).expect("decodes");
        }
        decoder.finish(&mut blocks).expect("flushes");
        blocks
    }

    #[test]
    fn text_stream_commits_response_normal() {
        let raw = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\"}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"!\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        assert!(decoder.is_completed());
        assert_eq!(
            blocks,
            vec![DraftBlock::Text {
                text: "Hello!".into()
            }]
        );

        let entries = finish_draft(Entry::Draft { blocks }).unwrap();
        assert_eq!(entries[0].response_joined_text().as_deref(), Some("Hello!"));
    }

    #[test]
    fn thinking_signature_arrives_after_text() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let me \"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"check\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig==\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        let entries = finish_draft(Entry::Draft { blocks }).unwrap();
        assert_eq!(entries[0].response_joined_text().as_deref(), Some("answer"));
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Signed { thinking, signature }]
                if thinking == "Let me check" && signature == "sig=="
        ));
    }

    #[test]
    fn unsigned_thinking_is_dropped_so_committed_history_is_serializable() {
        // llama.cpp and some other compatible endpoints stream thinking but
        // never send a signature. The preview shows it; the committed history
        // must drop it or the next Anthropic request fails validation.
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi!\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        let entries = finish_draft_for(Entry::Draft { blocks }, &config().kind).unwrap();

        let mut history = vec![text_user("q")];
        history.extend(entries);
        let request = crate::anthropic::to_anthropic_messages(&history, &config())
            .expect("committed stream is serializable");

        let assistant = &request.messages[1];
        let crate::anthropic::Content::Blocks(blocks) = &assistant.content else {
            panic!("expected assistant blocks");
        };
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], crate::anthropic::Block::Text { text } if text == "Hi!"));
    }

    #[test]
    fn tool_use_arguments_accumulate() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Oslo\\\"}\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        assert_eq!(
            blocks,
            vec![DraftBlock::ToolCall {
                index: 0,
                tool_use_id: Some("toolu_1".into()),
                tool_name: Some("get_weather".into()),
                arguments: r#"{"city":"Oslo"}"#.into(),
            }]
        );

        let entries = finish_draft(Entry::Draft { blocks }).unwrap();
        let Entry::ResponseToolCall { tool_call, .. } = &entries[0] else {
            panic!("expected ResponseToolCall");
        };
        assert_eq!(tool_call[0].toolcall_params, r#"{"city":"Oslo"}"#);
    }

    #[test]
    fn provider_error_becomes_stream_error() {
        let raw = "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n";
        let mut decoder = StreamDecoder::new();
        let mut blocks = Vec::new();
        let err = decoder.push_bytes(raw.as_bytes(), &mut blocks).unwrap_err();
        assert_eq!(
            err,
            StreamError::Provider {
                message: "Overloaded".into()
            }
        );
    }

    #[test]
    fn deltas_are_routed_by_content_block_index() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"zero\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"one\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        assert_eq!(
            blocks,
            vec![
                DraftBlock::Text {
                    text: "zero".into()
                },
                DraftBlock::Text { text: "one".into() },
            ]
        );
    }

    #[test]
    fn input_seeded_at_block_start_is_not_doubled_by_deltas() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"a\",\"input\":{\"x\":1}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"x\\\":1}\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        assert_eq!(
            blocks,
            vec![DraftBlock::ToolCall {
                index: 0,
                tool_use_id: Some("toolu_1".into()),
                tool_name: Some("a".into()),
                arguments: r#"{"x":1}"#.into(),
            }]
        );
    }

    #[test]
    fn unknown_events_are_ignored() {
        let raw = concat!(
            "data: {\"type\":\"future_event\",\"foo\":1}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        assert_eq!(blocks, vec![DraftBlock::Text { text: "ok".into() }]);
    }

    #[test]
    fn redacted_thinking_is_preserved() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"opaque\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        let entries = finish_draft(Entry::Draft { blocks }).unwrap();
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Redacted { thinking }] if thinking == "opaque"
        ));
    }
}
