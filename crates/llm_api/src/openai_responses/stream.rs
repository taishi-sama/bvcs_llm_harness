//! Streaming support for the **OpenAI Responses API** (`stream: true`).
//!
//! Unlike Chat Completions, the Responses API emits semantic events keyed by
//! `output_index` / `item_id`:
//!
//! ```text
//! event: response.output_item.added
//! data: {"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}
//!
//! event: response.output_text.delta
//! data: {"type":"response.output_text.delta","output_index":1,"delta":"Hello"}
//!
//! event: response.completed
//! data: {"type":"response.completed","response":{...,"output":[...]}}
//! ```
//!
//! `response.completed` contains the full response object. When present, the
//! decoder rebuilds the draft from it, so the streamed history matches the
//! non-streaming parser exactly.

use crate::openai_responses::response::{OutputItem, ResponsesResponse};
use crate::sse::SseDecoder;
use crate::stream::{StreamError, StreamEvent};
use llm_history_model::history::DraftBlock;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

/// Incremental Responses API stream decoder.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    sse: SseDecoder,
    /// `output_index` -> position in `blocks`.
    positions: HashMap<u32, usize>,
    last_position: Option<usize>,
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

    pub fn push_bytes(
        &mut self,
        bytes: &[u8],
        blocks: &mut Vec<DraftBlock>,
    ) -> Result<Vec<StreamEvent>, StreamError> {
        let mut out = Vec::new();
        for sse in self.sse.push(bytes) {
            if sse.data.is_empty() {
                continue;
            }
            let event: RawEvent =
                serde_json::from_str(&sse.data).map_err(|e| StreamError::InvalidJson {
                    event: sse.event.clone(),
                    details: e.to_string(),
                })?;
            self.handle(event, blocks, &mut out)?;
        }
        Ok(out)
    }

    pub fn finish(
        &mut self,
        blocks: &mut Vec<DraftBlock>,
    ) -> Result<Vec<StreamEvent>, StreamError> {
        let mut out = Vec::new();
        for sse in self.sse.finish() {
            if sse.data.is_empty() {
                continue;
            }
            let event: RawEvent =
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
        event: RawEvent,
        blocks: &mut Vec<DraftBlock>,
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), StreamError> {
        match event.kind.as_str() {
            "response.output_item.added" => {
                if let Some(item) = event.item {
                    let index = event.output_index.unwrap_or(blocks.len() as u32);
                    let pos = push_item(blocks, index, &item);
                    self.positions.insert(index, pos);
                    self.last_position = Some(pos);
                }
            }

            "response.content_part.added" => {
                // The item block already exists; initial part text is usually
                // empty. If an endpoint sends text here, recover it.
                if let Some(part) = event.part
                    && let Some(text) = part.text()
                    && !text.is_empty()
                {
                    let pos = self.position_for_text(event.output_index, blocks);
                    if let Some(DraftBlock::Text { text: existing }) = blocks.get_mut(pos) {
                        existing.push_str(&text);
                    }
                }
            }

            "response.output_text.delta" => {
                if let Some(delta) = event.delta
                    && !delta.is_empty()
                {
                    let pos = self.position_for_text(event.output_index, blocks);
                    if let Some(DraftBlock::Text { text }) = blocks.get_mut(pos) {
                        text.push_str(&delta);
                    }
                    self.last_position = Some(pos);
                    out.push(StreamEvent::TextDelta(delta));
                }
            }

            "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                if let Some(delta) = event.delta
                    && !delta.is_empty()
                {
                    let pos = self.position_for_thinking(event.output_index, blocks);
                    if let Some(DraftBlock::Thinking { text, .. }) = blocks.get_mut(pos) {
                        text.push_str(&delta);
                    }
                    self.last_position = Some(pos);
                    out.push(StreamEvent::ThinkingDelta(delta));
                }
            }

            "response.function_call_arguments.delta" => {
                if let Some(delta) = event.delta
                    && !delta.is_empty()
                {
                    let index = event.output_index.unwrap_or(blocks.len() as u32);
                    let pos = self.position_for_tool_call(index, blocks);
                    if let Some(DraftBlock::ToolCall {
                        arguments,
                        index: block_index,
                        ..
                    }) = blocks.get_mut(pos)
                    {
                        arguments.push_str(&delta);
                        out.push(StreamEvent::ToolCallDelta {
                            index: *block_index,
                            id: None,
                            name: None,
                            arguments: delta,
                        });
                    }
                    self.last_position = Some(pos);
                }
            }

            "response.output_item.done" => {
                if let Some(item) = event.item {
                    let index = event.output_index.unwrap_or(blocks.len() as u32);
                    let pos = match self.positions.get(&index).copied() {
                        Some(pos) if pos < blocks.len() => pos,
                        _ => push_item(blocks, index, &item),
                    };
                    sync_item(&mut blocks[pos], &item);
                    self.positions.insert(index, pos);
                    self.last_position = Some(pos);
                }
            }

            "response.completed" | "response.incomplete" => {
                self.completed = true;
                self.stop_reason = Some(if event.kind == "response.completed" {
                    "completed".into()
                } else {
                    "incomplete".into()
                });

                // The completed payload is authoritative and guarantees parity
                // with the non-streaming parser. Some compatible endpoints send
                // an empty `output`; keep the accumulated deltas in that case.
                if let Some(response) = event.response
                    && let Ok(parsed) = serde_json::from_value::<ResponsesResponse>(response)
                    && !parsed.output.is_empty()
                {
                    rebuild_from_response(&parsed, blocks);
                }

                out.push(StreamEvent::Completed {
                    stop_reason: self.stop_reason.clone(),
                });
            }

            "response.failed" => {
                let message = event
                    .response
                    .and_then(|r| r.get("error").cloned())
                    .and_then(|e| e.get("message").and_then(Value::as_str).map(str::to_owned))
                    .unwrap_or_else(|| "response failed".into());
                return Err(StreamError::Provider { message });
            }

            "error" => {
                return Err(StreamError::Provider {
                    message: event.message.unwrap_or_else(|| "stream error".into()),
                });
            }

            // created / in_progress / content_part.done / *.done without
            // history-relevant data.
            _ => {}
        }
        Ok(())
    }

    fn position_for_text(
        &mut self,
        output_index: Option<u32>,
        blocks: &mut Vec<DraftBlock>,
    ) -> usize {
        let index = output_index.unwrap_or(blocks.len() as u32);
        if let Some(pos) = output_index.and_then(|i| self.positions.get(&i).copied())
            && matches!(blocks.get(pos), Some(DraftBlock::Text { .. }))
        {
            return pos;
        }
        if let Some(pos) = self
            .last_position
            .filter(|pos| matches!(blocks.get(*pos), Some(DraftBlock::Text { .. })))
        {
            return pos;
        }
        blocks.push(DraftBlock::Text {
            text: String::new(),
        });
        let pos = blocks.len() - 1;
        self.positions.insert(index, pos);
        pos
    }

    fn position_for_thinking(
        &mut self,
        output_index: Option<u32>,
        blocks: &mut Vec<DraftBlock>,
    ) -> usize {
        let index = output_index.unwrap_or(blocks.len() as u32);
        if let Some(pos) = output_index.and_then(|i| self.positions.get(&i).copied())
            && matches!(blocks.get(pos), Some(DraftBlock::Thinking { .. }))
        {
            return pos;
        }
        if let Some(pos) = self
            .last_position
            .filter(|pos| matches!(blocks.get(*pos), Some(DraftBlock::Thinking { .. })))
        {
            return pos;
        }
        blocks.push(DraftBlock::Thinking {
            text: String::new(),
            signature: None,
        });
        let pos = blocks.len() - 1;
        self.positions.insert(index, pos);
        pos
    }

    fn position_for_tool_call(&mut self, index: u32, blocks: &mut Vec<DraftBlock>) -> usize {
        if let Some(pos) = self.positions.get(&index).copied()
            && matches!(blocks.get(pos), Some(DraftBlock::ToolCall { .. }))
        {
            return pos;
        }
        if let Some(pos) = blocks.iter().position(|block| {
            matches!(block, DraftBlock::ToolCall { index: block_index, .. } if *block_index == index)
        }) {
            self.positions.insert(index, pos);
            return pos;
        }
        blocks.push(DraftBlock::ToolCall {
            index,
            tool_use_id: None,
            tool_name: None,
            arguments: String::new(),
        });
        let pos = blocks.len() - 1;
        self.positions.insert(index, pos);
        pos
    }
}

fn push_item(blocks: &mut Vec<DraftBlock>, index: u32, item: &Item) -> usize {
    let block = match item {
        Item::Message { content } => DraftBlock::Text {
            text: join_parts(content),
        },
        Item::Reasoning { summary, content } => DraftBlock::Thinking {
            text: reasoning_text(summary, content),
            signature: None,
        },
        Item::FunctionCall {
            call_id,
            name,
            arguments,
        } => DraftBlock::ToolCall {
            index,
            tool_use_id: Some(call_id.clone()),
            tool_name: Some(name.clone()),
            arguments: arguments.clone(),
        },
        Item::Other => DraftBlock::Text {
            text: String::new(),
        },
    };
    blocks.push(block);
    blocks.len() - 1
}

fn sync_item(block: &mut DraftBlock, item: &Item) {
    match (block, item) {
        (DraftBlock::Text { text }, Item::Message { content }) => {
            let final_text = join_parts(content);
            if !final_text.is_empty() {
                *text = final_text;
            }
        }
        (DraftBlock::Thinking { text, .. }, Item::Reasoning { summary, content }) => {
            let final_text = reasoning_text(summary, content);
            if !final_text.is_empty() {
                *text = final_text;
            }
        }
        (
            DraftBlock::ToolCall {
                tool_use_id,
                tool_name,
                arguments,
                ..
            },
            Item::FunctionCall {
                call_id,
                name,
                arguments: final_arguments,
            },
        ) => {
            *tool_use_id = Some(call_id.clone());
            *tool_name = Some(name.clone());
            if !final_arguments.is_empty() {
                *arguments = final_arguments.clone();
            }
        }
        _ => {}
    }
}

fn rebuild_from_response(response: &ResponsesResponse, blocks: &mut Vec<DraftBlock>) {
    blocks.clear();
    let mut index = 0u32;
    for item in &response.output {
        match item {
            OutputItem::Reasoning(r) => {
                blocks.push(DraftBlock::Thinking {
                    text: r.text(),
                    signature: None,
                });
                index += 1;
            }
            OutputItem::Message(m) => {
                let text = m.text();
                if !text.is_empty() {
                    blocks.push(DraftBlock::Text { text });
                    index += 1;
                }
            }
            OutputItem::FunctionCall(fc) => {
                blocks.push(DraftBlock::ToolCall {
                    index,
                    tool_use_id: Some(fc.call_id.clone()),
                    tool_name: Some(fc.name.clone()),
                    arguments: fc.arguments.clone(),
                });
                index += 1;
            }
            // Tool outputs are client-side history entries, not assistant
            // blocks; unknown items are skipped like `response.rs` does.
            OutputItem::FunctionCallOutput(_) | OutputItem::Other => {}
        }
    }
}

fn join_parts(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(ContentPart::text)
        .collect::<Vec<_>>()
        .join("\n")
}

fn reasoning_text(summary: &[TextPart], content: &Option<Vec<TextPart>>) -> String {
    if let Some(parts) = content {
        let text = parts
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if !text.is_empty() {
            return text;
        }
    }
    summary
        .iter()
        .map(|p| p.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Deserialize)]
struct RawEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    output_index: Option<u32>,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    item: Option<Item>,
    #[serde(default)]
    part: Option<ContentPart>,
    #[serde(default)]
    response: Option<Value>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Item {
    Message {
        #[serde(default)]
        content: Vec<ContentPart>,
    },
    Reasoning {
        #[serde(default)]
        summary: Vec<TextPart>,
        #[serde(default)]
        content: Option<Vec<TextPart>>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        #[serde(default)]
        arguments: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart {
    OutputText {
        #[serde(default)]
        text: String,
    },
    Refusal {
        #[serde(default)]
        refusal: String,
    },
    #[serde(other)]
    Other,
}

impl ContentPart {
    fn text(&self) -> Option<String> {
        match self {
            ContentPart::OutputText { text } => Some(text.clone()),
            ContentPart::Refusal { refusal } => Some(refusal.clone()),
            ContentPart::Other => None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct TextPart {
    #[serde(default)]
    text: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::finish_draft;
    use llm_history_model::history::{Entry, Thinking};

    fn feed(decoder: &mut StreamDecoder, raw: &str) -> Vec<DraftBlock> {
        let mut blocks = Vec::new();
        for chunk in raw.as_bytes().chunks(13) {
            decoder.push_bytes(chunk, &mut blocks).expect("decodes");
        }
        decoder.finish(&mut blocks).expect("flushes");
        blocks
    }

    #[test]
    fn text_and_reasoning_deltas() {
        let raw = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[]}}\n\n",
            "data: {\"type\":\"response.reasoning_text.delta\",\"output_index\":0,\"delta\":\"hmm\"}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"content\":[]}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"Hel\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"lo\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"model\":\"m\",\"status\":\"completed\",\"output\":[]}}\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        assert!(decoder.is_completed());
        // An empty `output` in `response.completed` must not erase deltas.
        assert_eq!(
            blocks,
            vec![
                DraftBlock::Thinking {
                    text: "hmm".into(),
                    signature: None,
                },
                DraftBlock::Text {
                    text: "Hello".into(),
                },
            ]
        );

        let entries = finish_draft(Entry::Draft { blocks }).unwrap();
        let Entry::ResponseNormal { thinking, content } = &entries[0] else {
            panic!("expected ResponseNormal");
        };
        assert_eq!(content, "Hello");
        assert!(matches!(
            thinking.as_slice(),
            [Thinking::Normal { thinking }] if thinking == "hmm"
        ));
    }

    #[test]
    fn completes_from_authoritative_response_output() {
        let raw = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"partial\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"model\":\"m\",\"status\":\"completed\",\"output\":[",
            "{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[],\"content\":[{\"type\":\"reasoning_text\",\"text\":\"plan\"}]},",
            "{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello!\"}]}",
            "]}}\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        let entries = finish_draft(Entry::Draft { blocks }).unwrap();
        let Entry::ResponseNormal { thinking, content } = &entries[0] else {
            panic!("expected ResponseNormal");
        };
        assert_eq!(content, "Hello!");
        assert!(matches!(
            thinking.as_slice(),
            [Thinking::Normal { thinking }] if thinking == "plan"
        ));
    }

    #[test]
    fn function_call_arguments_stream() {
        let raw = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"a\",\"arguments\":\"\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"x\\\":\"}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"1}\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"model\":\"m\",\"status\":\"completed\",\"output\":[",
            "{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"a\",\"arguments\":\"{\\\"x\\\":1}\"}",
            "]}}\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        let entries = finish_draft(Entry::Draft { blocks }).unwrap();
        let Entry::ResponseToolCall { tool_call, .. } = &entries[0] else {
            panic!("expected ResponseToolCall");
        };
        assert_eq!(tool_call[0].toolcall_params, r#"{"x":1}"#);
    }

    #[test]
    fn failed_response_becomes_error() {
        let raw = "data: {\"type\":\"response.failed\",\"response\":{\"id\":\"r\",\"error\":{\"code\":\"server_error\",\"message\":\"boom\"}}}\n\n";
        let mut decoder = StreamDecoder::new();
        let mut blocks = Vec::new();
        let err = decoder.push_bytes(raw.as_bytes(), &mut blocks).unwrap_err();
        assert_eq!(
            err,
            StreamError::Provider {
                message: "boom".into()
            }
        );
    }
}
