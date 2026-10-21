//! Parsing of OpenAI-compatible **Responses API** response bodies
//! (`POST /responses`) back into [`llm_history_model::history::Entry`] histories.
//!
//! Mapping overview (inverse of [`super`]'s serialization):
//! - `reasoning` item          -> `Thinking::Normal` attached to the next assistant entry
//! - `message` item            -> `ResponseBlock::Text` in `Entry::Response`
//! - `function_call` items     -> consecutive items merge into one `Entry::ResponseToolCall`
//! - `function_call_output`s   -> consecutive items merge into one `Entry::ToolCallResult`
//! - unknown item types        -> skipped (forward compatibility)
//!
//! Note: `encrypted_content` is intentionally not preserved — the history model
//! stores plaintext thinking only. `function_call_output.is_error` has no wire
//! representation, so results always deserialize as `is_error: false`.

use crate::usage::{ParsedResponse, Usage, UsageProvider};
use llm_history_model::history::{Entry, ResponseBlock, Thinking, ToolCall, ToolCallResult};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

/// Top-level `POST /responses` response body.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesResponse {
    /// Optional for compatibility with lightweight OpenAI-compatible servers.
    #[serde(default)]
    pub id: Option<String>,
    /// Always `"response"`; optional for lenient parsing.
    #[serde(default)]
    pub object: Option<String>,
    pub created_at: Option<i64>,
    pub completed_at: Option<i64>,
    #[serde(default)]
    pub model: Option<String>,
    /// `"queued" | "in_progress" | "completed" | "incomplete" | "failed"`
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
    #[serde(default)]
    pub error: Option<ResponseError>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseError {
    pub code: Option<String>,
    pub message: Option<String>,
}

/// Responses API wire usage.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
    #[serde(default)]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(default)]
    pub output_tokens_details: Option<OutputTokensDetails>,
    /// Unknown numeric usage fields, preserved by the normalizer.
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    Message(OutputMessage),
    Reasoning(Reasoning),
    FunctionCall(FunctionCall),
    FunctionCallOutput(FunctionCallOutput),
    /// Any other (or future) item type; skipped during conversion.
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OutputMessage {
    pub id: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    pub status: Option<String>,
    #[serde(default)]
    pub content: Vec<OutputContentPart>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum OutputContentPart {
    #[serde(rename = "output_text")]
    Text { text: String },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Reasoning {
    pub id: Option<String>,
    #[serde(default)]
    pub summary: Vec<TextPart>,
    #[serde(default)]
    pub content: Option<Vec<TextPart>>,
    pub encrypted_content: Option<String>,
    pub status: Option<String>,
}

/// `{ type: "summary_text" | "reasoning_text", text: String }` — the `type`
/// discriminator is ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct TextPart {
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionCall {
    pub id: Option<String>,
    pub call_id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: String,
    pub status: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionCallOutput {
    pub id: Option<String>,
    /// Optional per spec; items without one are skipped during conversion.
    pub call_id: Option<String>,
    /// `string | [content parts]`; normalized to plain text.
    #[serde(default)]
    pub output: Option<Value>,
    pub status: Option<String>,
}

fn join_texts<'a>(texts: impl Iterator<Item = &'a str>) -> String {
    texts.collect::<Vec<_>>().join("\n")
}

impl Reasoning {
    /// Plaintext reasoning: prefers `content[]`, falls back to `summary[]`.
    pub(crate) fn text(&self) -> String {
        if let Some(parts) = &self.content {
            let s = join_texts(parts.iter().map(|p| p.text.as_str()));
            if !s.is_empty() {
                return s;
            }
        }
        join_texts(self.summary.iter().map(|p| p.text.as_str()))
    }

    /// History representation of this item.
    ///
    /// `encrypted_content` is preferred when present: it is the only form
    /// OpenAI accepts for stateless replay. Plaintext is retained for
    /// compatible endpoints that do not return encrypted payloads.
    pub(crate) fn thinking(&self) -> Option<Thinking> {
        if let Some(encrypted) = self
            .encrypted_content
            .as_deref()
            .filter(|content| !content.is_empty())
        {
            return Some(Thinking::Encrypted {
                thinking: encrypted.to_owned(),
            });
        }
        let text = self.text();
        if text.is_empty() {
            None
        } else {
            Some(Thinking::Normal { thinking: text })
        }
    }
}

impl OutputMessage {
    /// Visible text produced by the model (`output_text` + `refusal` parts).
    pub(crate) fn text(&self) -> String {
        join_texts(self.content.iter().filter_map(|p| match p {
            OutputContentPart::Text { text } => Some(text.as_str()),
            OutputContentPart::Refusal { refusal } => Some(refusal.as_str()),
            OutputContentPart::Other => None,
        }))
    }
}

fn normalize_output(output: &Option<Value>) -> String {
    match output {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => join_texts(
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str)),
        ),
        Some(other) => other.to_string(),
    }
}

impl ResponsesResponse {
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn from_value(value: Value) -> serde_json::Result<Self> {
        serde_json::from_value(value)
    }

    /// Normalized usage for this response, labeled as OpenAI.
    pub fn usage(&self) -> Option<Usage> {
        self.usage_for(UsageProvider::OpenAI)
    }

    /// Normalized usage with an explicit provider label.
    pub fn usage_for(&self, provider: UsageProvider) -> Option<Usage> {
        Usage::from_responses_for(self.usage.as_ref()?, provider)
    }

    /// Assistant-side entries plus the normalized usage of this response.
    pub fn parse(&self) -> ParsedResponse {
        self.parse_for(UsageProvider::OpenAI)
    }

    /// Assistant-side entries plus usage labeled with `provider`.
    pub fn parse_for(&self, provider: UsageProvider) -> ParsedResponse {
        ParsedResponse {
            entries: self.to_entries(),
            usage: self.usage_for(provider),
        }
    }

    /// Converts the response output back into history entries.
    ///
    /// - reasoning and message items become ordered [`ResponseBlock`]s inside
    ///   one [`Entry::Response`]; `encrypted_content` becomes
    ///   [`Thinking::Encrypted`], otherwise plaintext becomes
    ///   [`Thinking::Normal`]
    /// - consecutive `function_call` items merge into one
    ///   [`Entry::ResponseToolCall`] (parallel calls live in one entry); the
    ///   ordered reasoning/message blocks precede the calls
    /// - consecutive `function_call_output` items merge into one
    ///   [`Entry::ToolCallResult`]
    /// - empty messages and items without `call_id` are skipped
    ///
    /// The result contains assistant-side entries only (`Response`,
    /// `ResponseToolCall`, `ToolCallResult`) — append them to an existing
    /// user-first history before validation or re-serialization.
    pub fn to_entries(&self) -> Vec<Entry> {
        fn flush_assistant(
            blocks: &mut Vec<ResponseBlock>,
            calls: &mut Vec<ToolCall>,
            entries: &mut Vec<Entry>,
        ) {
            if !calls.is_empty() {
                entries.push(Entry::ResponseToolCall {
                    blocks: std::mem::take(blocks),
                    tool_call: std::mem::take(calls),
                });
            } else if !blocks.is_empty() {
                entries.push(Entry::Response {
                    blocks: std::mem::take(blocks),
                });
            }
        }

        fn flush_results(results: &mut Vec<ToolCallResult>, entries: &mut Vec<Entry>) {
            if !results.is_empty() {
                entries.push(Entry::ToolCallResult {
                    tool_call: std::mem::take(results),
                });
            }
        }

        let mut entries = Vec::with_capacity(self.output.len());
        let mut blocks: Vec<ResponseBlock> = Vec::new();
        let mut calls: Vec<ToolCall> = Vec::new();
        let mut results: Vec<ToolCallResult> = Vec::new();

        for item in &self.output {
            match item {
                OutputItem::Other => {}

                OutputItem::Reasoning(r) => {
                    let Some(thinking) = r.thinking() else {
                        continue;
                    };
                    flush_results(&mut results, &mut entries);
                    blocks.push(ResponseBlock::Thinking(thinking));
                }

                OutputItem::Message(m) => {
                    let text = m.text();
                    if text.is_empty() {
                        continue;
                    }
                    flush_results(&mut results, &mut entries);
                    blocks.push(ResponseBlock::Text(text));
                }

                OutputItem::FunctionCall(fc) => {
                    flush_results(&mut results, &mut entries);
                    let arguments = if fc.arguments.trim().is_empty() {
                        "{}".to_owned()
                    } else {
                        fc.arguments.clone()
                    };
                    calls.push(ToolCall {
                        tool_name: fc.name.clone(),
                        toolcall_params: arguments,
                        tool_use_id: fc.call_id.clone(),
                    });
                }

                OutputItem::FunctionCallOutput(out) => {
                    flush_assistant(&mut blocks, &mut calls, &mut entries);
                    let Some(tool_use_id) = &out.call_id else {
                        continue;
                    };
                    results.push(ToolCallResult {
                        result: normalize_output(&out.output),
                        is_error: false,
                        tool_use_id: tool_use_id.clone(),
                    });
                }
            }
        }

        flush_assistant(&mut blocks, &mut calls, &mut entries);
        flush_results(&mut results, &mut entries);
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::{ModelConfig, ModelKind};
    use crate::validation::validate_entries;

    fn config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::OpenAICompatible {
                model_name: "qwen3.6-35b-a3b".into(),
                streaming: false,
                prompt_cache_key: "prompt_cache_key".into(),
                provider: crate::usage::UsageProvider::OpenAI,
                include_usage: true,
            },
            tools: Default::default(),
            max_tokens: None,
        }
    }

    const REAL_PAYLOAD: &str = r#"{
        "completed_at":1787591421,"created_at":1787591421,
        "id":"resp_oMsgUGht6NWYDLP6eS8xyFoyFn9KymTz",
        "model":"qwen3.6-35b-a3b","object":"response",
        "output":[
            {"id":"rs_Rb8FHpHd5xvFtRWwUsXkruRrKTs2dsjB","summary":[],"type":"reasoning",
             "content":[{"text":"Here's a thinking process:\n\n1.  **Analyze User Input:**\n   - User says: \"Hello!\"\n","type":"reasoning_text"}],
             "encrypted_content":"","status":"completed"},
            {"content":[{"type":"output_text","annotations":[],"logprobs":[],"text":"Hello! How can I assist you today?"}],
             "id":"msg_BhsIdbcvSpMAujtvSrxoLGZ9URf39V20","role":"assistant","status":"completed","type":"message"}
        ],
        "status":"completed",
        "usage":{"input_tokens":21,"output_tokens":247,"total_tokens":268,"input_tokens_details":{"cached_tokens":0}}
    }"#;

    fn text_user(t: &str) -> Entry {
        Entry::UserMessage {
            content: llm_history_model::history::MessageContent {
                text: Some(t.into()),
                image: None,
            },
        }
    }

    /// Response entries are assistant-side; they validate only when appended
    /// onto an existing user-first history.
    fn validated_history(resp_entries: Vec<Entry>) -> Vec<Entry> {
        let mut history = vec![text_user("hi")];
        history.extend(resp_entries);
        assert_eq!(validate_entries(&history, &config()), Ok(()));
        history
    }

    #[test]
    fn parses_real_endpoint_payload() {
        let resp = ResponsesResponse::from_json(REAL_PAYLOAD).expect("parses");
        assert_eq!(resp.model.as_deref(), Some("qwen3.6-35b-a3b"));
        assert_eq!(resp.status.as_deref(), Some("completed"));
        let usage = resp.usage.as_ref().expect("usage");
        assert_eq!(usage.input_tokens, Some(21));
        assert_eq!(
            usage.input_tokens_details.as_ref().unwrap().cached_tokens,
            Some(0)
        );

        let normalized = resp.usage().expect("normalized usage");
        assert_eq!(normalized.input_tokens, Some(21));
        assert_eq!(normalized.output_tokens, Some(247));
        assert_eq!(normalized.cached_input_tokens, Some(0));
        assert_eq!(normalized.total(), Some(268));

        let entries = resp.to_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].response_joined_text().as_deref(),
            Some("Hello! How can I assist you today?")
        );
        let [Thinking::Normal { thinking }] = entries[0].response_thinking_blocks().as_slice()
        else {
            panic!("expected one attached thinking block");
        };
        assert!(thinking.contains("Analyze User Input"));

        validated_history(entries);
    }

    #[test]
    fn merges_parallel_calls_and_outputs() {
        let resp = ResponsesResponse::from_value(serde_json::json!({
            "id": "resp_x", "model": "m", "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_0", "summary": [], "content": [{"type": "reasoning_text", "text": "plan"}]},
                {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{\"x\":1}"},
                {"type": "function_call", "call_id": "c2", "name": "b", "arguments": ""},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                {"type": "function_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "line1"}, {"type": "input_text", "text": "line2"}]},
                {"type": "web_search_call", "id": "ws_0", "status": "completed"}
            ]
        }))
        .expect("parses");

        let entries = resp.to_entries();
        assert_eq!(entries.len(), 2);

        let Entry::ResponseToolCall { blocks, tool_call } = &entries[0] else {
            panic!("expected merged ResponseToolCall");
        };
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Normal { .. }]
        ));
        assert!(entries[0].response_text_blocks().is_empty());
        assert_eq!(blocks.len(), 1);
        assert_eq!(tool_call.len(), 2);
        assert_eq!(tool_call[1].toolcall_params, "{}"); // empty arguments defaulted

        let Entry::ToolCallResult { tool_call } = &entries[1] else {
            panic!("expected merged ToolCallResult");
        };
        assert_eq!(tool_call.len(), 2);
        assert_eq!(tool_call[1].result, "line1\nline2"); // array output normalized
        assert!(!tool_call[0].is_error);

        validated_history(entries);
    }

    #[test]
    fn missing_metadata_is_tolerated() {
        let resp = ResponsesResponse::from_value(serde_json::json!({
            "output": [{
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi"}]
            }]
        }))
        .expect("parses without id/model");

        assert!(resp.id.is_none());
        assert!(resp.model.is_none());
        assert_eq!(resp.to_entries().len(), 1);
    }

    #[test]
    fn preserves_interleaved_reasoning_and_messages() {
        let resp = ResponsesResponse::from_value(serde_json::json!({
            "id": "resp_i", "model": "m",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": [],
                 "content": [{"type": "reasoning_text", "text": "first"}]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "one"}]},
                {"type": "reasoning", "id": "rs_2", "summary": [],
                 "content": [{"type": "reasoning_text", "text": "second"}]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "two"}]}
            ]
        }))
        .expect("parses");

        let entries = resp.to_entries();
        assert_eq!(
            entries[0].as_response_blocks().expect("expected Response"),
            [
                ResponseBlock::Thinking(Thinking::Normal {
                    thinking: "first".into()
                }),
                ResponseBlock::Text("one".into()),
                ResponseBlock::Thinking(Thinking::Normal {
                    thinking: "second".into()
                }),
                ResponseBlock::Text("two".into()),
            ]
        );
    }

    #[test]
    fn prefers_encrypted_reasoning_and_keeps_synthesized_id_input_intact() {
        let resp = ResponsesResponse::from_value(serde_json::json!({
            "id": "resp_e", "model": "m",
            "output": [
                {
                    "type": "reasoning", "id": "rs_1", "summary": [],
                    "content": [{"type": "reasoning_text", "text": "plain"}],
                    "encrypted_content": "opaque-openai-blob"
                },
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "answer"}]}
            ]
        }))
        .expect("parses");

        let entries = resp.to_entries();
        assert_eq!(entries[0].response_joined_text().as_deref(), Some("answer"));
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Encrypted { thinking }] if thinking == "opaque-openai-blob"
        ));
    }

    #[test]
    fn reasoning_without_message_becomes_thinking_only() {
        let resp = ResponsesResponse::from_value(serde_json::json!({
            "id": "resp_t", "model": "m",
            "output": [
                {
                    "type": "reasoning", "id": "rs_1", "summary": [],
                    "encrypted_content": "opaque"
                }
            ]
        }))
        .expect("parses");

        let entries = resp.to_entries();
        assert_eq!(entries.len(), 1);
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Encrypted { thinking }] if thinking == "opaque"
        ));
        assert!(entries[0].response_text_blocks().is_empty());
    }

    #[test]
    fn skips_items_without_call_id_and_empty_messages() {
        let resp = ResponsesResponse::from_value(serde_json::json!({
            "id": "resp_y", "model": "m",
            "output": [
                {"type": "function_call_output", "output": "orphan"},
                {"type": "message", "id": "msg_0", "role": "assistant", "content": []},
                {"type": "message", "id": "msg_1", "role": "assistant",
                 "content": [{"type": "refusal", "refusal": "cannot comply"}]}
            ]
        }))
        .expect("parses");

        let entries = resp.to_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].response_joined_text().as_deref(),
            Some("cannot comply")
        );
        assert!(entries[0].response_thinking_blocks().is_empty());
    }

    /// Serialize a history into the request, feed the assistant-side output
    /// items back through the response parser, and confirm re-serializing
    /// reproduces the original `input` items verbatim (including synthesized
    /// reasoning ids).
    fn assert_round_trip(history: &[Entry]) {
        use crate::openai_responses::to_responses_input;

        let input = to_responses_input(history, &config()).expect("valid history");
        let items = serde_json::to_value(&input.input).unwrap();
        // The wire `output` never contains the user message; assistant-side
        // items start after the first (user) item. Output-side messages also
        // always carry content as typed parts (never the string shorthand).
        let assistant_items: Vec<Value> = items.as_array().unwrap()[1..]
            .iter()
            .cloned()
            .map(|mut item| {
                if item["type"] == "message"
                    && let Some(text) = item["content"].as_str().map(str::to_owned)
                {
                    item["content"] = serde_json::json!([{"type": "output_text", "text": text}]);
                }
                item
            })
            .collect();
        let response = ResponsesResponse::from_value(serde_json::json!({
            "id": "resp_rt", "model": "qwen3.6-35b-a3b", "status": "completed",
            "output": assistant_items,
        }))
        .expect("parses");

        let mut full = vec![history[0].clone()];
        full.extend(response.to_entries());
        let reinput = to_responses_input(&full, &config()).expect("valid history");
        assert_eq!(serde_json::to_value(&reinput.input).unwrap(), items);
    }

    #[test]
    fn round_trip_interleaved_text_and_thinking() {
        assert_round_trip(&[
            text_user("q"),
            Entry::response(vec![
                ResponseBlock::Text("first part".into()),
                ResponseBlock::Thinking(Thinking::Normal {
                    thinking: "hmm".into(),
                }),
                ResponseBlock::Text("second part".into()),
            ]),
        ]);
    }

    #[test]
    fn round_trip_message_with_reasoning() {
        assert_round_trip(&[
            text_user("q"),
            Entry::response(vec![
                ResponseBlock::Thinking(Thinking::Normal {
                    thinking: "hmm".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]),
        ]);
    }

    #[test]
    fn round_trip_tool_call_with_preamble_text() {
        assert_round_trip(&[
            text_user("q"),
            Entry::ResponseToolCall {
                blocks: vec![
                    ResponseBlock::Thinking(Thinking::Normal {
                        thinking: "plan".into(),
                    }),
                    ResponseBlock::Text("working on it".into()),
                ],
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
            Entry::ToolCallResult {
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

    #[test]
    fn message_before_calls_becomes_tool_call_content() {
        let resp = ResponsesResponse::from_value(serde_json::json!({
            "id": "resp_z", "model": "m",
            "output": [
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "working on it"}]},
                {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "r"}
            ]
        }))
        .expect("parses");

        let entries = resp.to_entries();
        assert_eq!(entries.len(), 2);
        let Entry::ResponseToolCall { blocks, tool_call } = &entries[0] else {
            panic!("expected ResponseToolCall with content");
        };
        assert!(entries[0].response_thinking_blocks().is_empty());
        assert_eq!(
            entries[0].response_joined_text().as_deref(),
            Some("working on it")
        );
        assert_eq!(blocks.len(), 1);
        assert_eq!(tool_call.len(), 1);

        validated_history(entries);
    }
}
