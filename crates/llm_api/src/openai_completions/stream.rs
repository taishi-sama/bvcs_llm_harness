//! Streaming support for the **OpenAI-compatible Chat Completions API**
//! (`stream: true`).
//!
//! Chunks carry a `delta` object:
//!
//! ```text
//! data: {"choices":[{"index":0,"delta":{"reasoning_content":"Let me"}}]}
//! data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}
//! data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"a","arguments":"{\"x\":"}}]}}]}
//! data: [DONE]
//! ```
//!
//! Reasoning streams first (`reasoning_content`, or the OpenRouter-style
//! `reasoning`), then visible text, then tool calls. Tool-call id/name usually
//! appear only in the first fragment for each `index`; later fragments may carry
//! just `index` + `arguments`, so blocks are merged by index.

use crate::sse::SseDecoder;
use crate::stream::{StreamError, StreamEvent, append_text, tool_call_position};
use llm_history_model::history::DraftBlock;
use serde::Deserialize;

/// Incremental Chat Completions stream decoder.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    sse: SseDecoder,
    thinking_pos: Option<usize>,
    text_pos: Option<usize>,
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
            self.handle_data(&sse.data, blocks, &mut out)?;
        }
        Ok(out)
    }

    pub fn finish(
        &mut self,
        blocks: &mut Vec<DraftBlock>,
    ) -> Result<Vec<StreamEvent>, StreamError> {
        let mut out = Vec::new();
        for sse in self.sse.finish() {
            self.handle_data(&sse.data, blocks, &mut out)?;
        }
        // Some compatible servers close the body without sending `[DONE]`.
        // The end of the HTTP stream is still the end of the completion.
        if !self.completed {
            self.completed = true;
            if self.stop_reason.is_some() || !blocks.is_empty() {
                out.push(StreamEvent::Completed {
                    stop_reason: self.stop_reason.clone(),
                });
            }
        }
        Ok(out)
    }

    fn handle_data(
        &mut self,
        data: &str,
        blocks: &mut Vec<DraftBlock>,
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), StreamError> {
        if data.is_empty() {
            return Ok(());
        }
        if data.trim() == "[DONE]" {
            self.completed = true;
            out.push(StreamEvent::Completed {
                stop_reason: self.stop_reason.clone(),
            });
            return Ok(());
        }

        let chunk: Chunk = serde_json::from_str(data).map_err(|e| StreamError::InvalidJson {
            event: None,
            details: e.to_string(),
        })?;

        // `n > 1` is not supported by the response model; honor choice 0 only,
        // matching `response.rs`.
        let Some(choice) = chunk.choices.iter().find(|c| c.index.unwrap_or(0) == 0) else {
            return Ok(());
        };

        // Reasoning (`reasoning_content` from vLLM/DeepSeek, `reasoning` from
        // OpenRouter-style endpoints).
        if let Some(reasoning) = choice
            .delta
            .reasoning_content
            .as_deref()
            .or(choice.delta.reasoning.as_deref())
            && !reasoning.is_empty()
        {
            append_text(blocks, &mut self.thinking_pos, reasoning, true);
            out.push(StreamEvent::ThinkingDelta(reasoning.to_owned()));
        }

        if let Some(content) = choice.delta.content.as_deref()
            && !content.is_empty()
        {
            append_text(blocks, &mut self.text_pos, content, false);
            out.push(StreamEvent::TextDelta(content.to_owned()));
        }

        for call in &choice.delta.tool_calls {
            let pos = match tool_call_position(blocks, call.index) {
                Some(pos) => pos,
                None => {
                    blocks.push(DraftBlock::ToolCall {
                        index: call.index,
                        tool_use_id: call.id.clone(),
                        tool_name: call.function.name.clone(),
                        arguments: String::new(),
                    });
                    blocks.len() - 1
                }
            };

            let mut arguments = String::new();
            if let DraftBlock::ToolCall {
                tool_use_id,
                tool_name,
                arguments: existing,
                ..
            } = &mut blocks[pos]
            {
                if call.id.is_some() {
                    *tool_use_id = call.id.clone();
                }
                if call.function.name.is_some() {
                    *tool_name = call.function.name.clone();
                }
                if let Some(fragment) = &call.function.arguments {
                    existing.push_str(fragment);
                    arguments = fragment.clone();
                }
            }

            out.push(StreamEvent::ToolCallDelta {
                index: call.index,
                id: call.id.clone(),
                name: call.function.name.clone(),
                arguments,
            });
        }

        if choice.finish_reason.is_some() {
            self.stop_reason = choice.finish_reason.clone();
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct Chunk {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Vec<CallDelta>,
}

#[derive(Debug, Deserialize)]
struct CallDelta {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: FunctionDelta,
}

#[derive(Debug, Deserialize, Default)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::finish_draft;
    use llm_history_model::history::{Entry, Thinking};

    fn feed(decoder: &mut StreamDecoder, raw: &str) -> Vec<DraftBlock> {
        let mut blocks = Vec::new();
        for chunk in raw.as_bytes().chunks(11) {
            decoder.push_bytes(chunk, &mut blocks).expect("decodes");
        }
        decoder.finish(&mut blocks).expect("flushes");
        blocks
    }

    #[test]
    fn text_and_reasoning_stream() {
        let raw = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"hmm\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        assert!(decoder.is_completed());

        let entries = finish_draft(Entry::Draft { blocks }).unwrap();
        assert_eq!(entries[0].response_joined_text().as_deref(), Some("Hello"));
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Normal { thinking }] if thinking == "hmm"
        ));
    }

    #[test]
    fn tool_call_fragments_merge_by_index() {
        let raw = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"a\",\"arguments\":\"{\\\"x\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"c2\",\"function\":{\"name\":\"b\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        assert_eq!(
            blocks,
            vec![
                DraftBlock::ToolCall {
                    index: 0,
                    tool_use_id: Some("c1".into()),
                    tool_name: Some("a".into()),
                    arguments: r#"{"x":1}"#.into(),
                },
                DraftBlock::ToolCall {
                    index: 1,
                    tool_use_id: Some("c2".into()),
                    tool_name: Some("b".into()),
                    arguments: String::new(),
                },
            ]
        );

        let entries = finish_draft(Entry::Draft { blocks }).unwrap();
        let Entry::ResponseToolCall { tool_call, .. } = &entries[0] else {
            panic!("expected ResponseToolCall");
        };
        assert_eq!(tool_call[0].toolcall_params, r#"{"x":1}"#);
        assert_eq!(tool_call[1].toolcall_params, "{}");
    }

    #[test]
    fn ignores_non_zero_choices() {
        let raw = concat!(
            "data: {\"choices\":[{\"index\":1,\"delta\":{\"content\":\"nope\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"yes\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        assert_eq!(blocks, vec![DraftBlock::Text { text: "yes".into() }]);
    }

    #[test]
    fn finishes_when_server_closes_without_done_marker() {
        let raw = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        );

        let mut decoder = StreamDecoder::new();
        let mut blocks = Vec::new();
        let mut events = decoder
            .push_bytes(raw.as_bytes(), &mut blocks)
            .expect("decodes");
        events.extend(decoder.finish(&mut blocks).expect("flushes"));

        assert!(decoder.is_completed());
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Completed {
                stop_reason: Some(reason)
            }) if reason == "stop"
        ));
    }

    #[test]
    fn usage_only_chunk_is_ignored() {
        let raw = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n",
        );
        let mut decoder = StreamDecoder::new();
        let blocks = feed(&mut decoder, raw);
        assert_eq!(blocks, vec![DraftBlock::Text { text: "ok".into() }]);
    }
}
