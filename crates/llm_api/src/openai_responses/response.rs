//! Parsing of OpenAI-compatible **Responses API** response bodies
//! (`POST /responses`) back into [`llm_history_model::history::Entry`] histories.
//!
//! Mapping overview (inverse of [`super`]'s serialization):
//! - `reasoning` item          -> `Thinking::Normal` attached to the next assistant entry
//! - `message` item            -> `Entry::ResponseNormal` (`output_text`/`refusal` parts joined)
//! - `function_call` items     -> consecutive items merge into one `Entry::ResponseToolCall`
//! - `function_call_output`s   -> consecutive items merge into one `Entry::ToolCallResult`
//! - unknown item types        -> skipped (forward compatibility)
//!
//! Note: `encrypted_content` is intentionally not preserved — the history model
//! stores plaintext thinking only. `function_call_output.is_error` has no wire
//! representation, so results always deserialize as `is_error: false`.

use llm_history_model::history::{Entry, Thinking, ToolCall, ToolCallResult};
use serde::Deserialize;
use serde_json::Value;

/// Top-level `POST /responses` response body.
#[derive(Debug, Clone, Deserialize)]
pub struct Response {
    pub id: String,
    /// Always `"response"`; optional for lenient parsing.
    #[serde(default)]
    pub object: Option<String>,
    pub created_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub model: String,
    /// `"queued" | "in_progress" | "completed" | "incomplete" | "failed"`
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub error: Option<ResponseError>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseError {
    pub code: Option<String>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub input_tokens_details: Option<InputTokensDetails>,
    pub output_tokens_details: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InputTokensDetails {
    pub cached_tokens: Option<u64>,
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
    fn text(&self) -> String {
        if let Some(parts) = &self.content {
            let s = join_texts(parts.iter().map(|p| p.text.as_str()));
            if !s.is_empty() {
                return s;
            }
        }
        join_texts(self.summary.iter().map(|p| p.text.as_str()))
    }
}

impl OutputMessage {
    /// Visible text produced by the model (`output_text` + `refusal` parts).
    fn text(&self) -> String {
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

impl Response {
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn from_value(value: Value) -> serde_json::Result<Self> {
        serde_json::from_value(value)
    }

    /// Converts the response output back into history entries.
    ///
    /// - consecutive `function_call` items merge into one
    ///   [`Entry::ResponseToolCall`] (parallel calls live in one entry)
    /// - consecutive `function_call_output` items merge into one
    ///   [`Entry::ToolCallResult`]
    /// - `reasoning` items attach as [`Thinking::Normal`] to the following
    ///   assistant entry; multiple consecutive reasonings are concatenated
    /// - empty messages and items without `call_id` are skipped
    ///
    /// The result contains assistant-side entries only (`ResponseNormal`,
    /// `ResponseToolCall`, `ToolCallResult`) — append them to an existing
    /// user-first history before validation or re-serialization.
    pub fn to_entries(&self) -> Vec<Entry> {
        fn flush_calls(
            thinking: &mut Option<Thinking>,
            calls: &mut Vec<ToolCall>,
            entries: &mut Vec<Entry>,
        ) {
            if !calls.is_empty() {
                entries.push(Entry::ResponseToolCall {
                    thinking: thinking.take(),
                    tool_call: std::mem::take(calls),
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
        let mut pending_thinking: Option<Thinking> = None;
        let mut calls: Vec<ToolCall> = Vec::new();
        let mut results: Vec<ToolCallResult> = Vec::new();

        for item in &self.output {
            match item {
                OutputItem::Other => {}

                OutputItem::Reasoning(r) => {
                    let text = r.text();
                    if text.is_empty() {
                        continue;
                    }
                    flush_results(&mut results, &mut entries);
                    match &mut pending_thinking {
                        Some(Thinking::Normal { thinking }) => {
                            thinking.push_str("\n\n");
                            thinking.push_str(&text);
                        }
                        _ => pending_thinking = Some(Thinking::Normal { thinking: text }),
                    }
                }

                OutputItem::Message(m) => {
                    let text = m.text();
                    if text.is_empty() {
                        continue;
                    }
                    flush_results(&mut results, &mut entries);
                    flush_calls(&mut pending_thinking, &mut calls, &mut entries);
                    entries.push(Entry::ResponseNormal {
                        thinking: pending_thinking.take(),
                        content: text,
                    });
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
                    flush_calls(&mut pending_thinking, &mut calls, &mut entries);
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

        flush_calls(&mut pending_thinking, &mut calls, &mut entries);
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
            },
            tools: Default::default(),
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
        let resp = Response::from_json(REAL_PAYLOAD).expect("parses");
        assert_eq!(resp.model, "qwen3.6-35b-a3b");
        assert_eq!(resp.status.as_deref(), Some("completed"));
        let usage = resp.usage.as_ref().expect("usage");
        assert_eq!(usage.input_tokens, Some(21));
        assert_eq!(
            usage.input_tokens_details.as_ref().unwrap().cached_tokens,
            Some(0)
        );

        let entries = resp.to_entries();
        assert_eq!(entries.len(), 1);
        let Entry::ResponseNormal { thinking, content } = &entries[0] else {
            panic!("expected ResponseNormal");
        };
        assert_eq!(content, "Hello! How can I assist you today?");
        let Some(Thinking::Normal { thinking }) = thinking else {
            panic!("expected attached thinking");
        };
        assert!(thinking.contains("Analyze User Input"));

        validated_history(entries);
    }

    #[test]
    fn merges_parallel_calls_and_outputs() {
        let resp = Response::from_value(serde_json::json!({
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

        let Entry::ResponseToolCall {
            thinking,
            tool_call,
        } = &entries[0]
        else {
            panic!("expected merged ResponseToolCall");
        };
        assert!(matches!(thinking, Some(Thinking::Normal { .. })));
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
    fn skips_items_without_call_id_and_empty_messages() {
        let resp = Response::from_value(serde_json::json!({
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
        let Entry::ResponseNormal { content, thinking } = &entries[0] else {
            panic!("expected ResponseNormal");
        };
        assert_eq!(content, "cannot comply");
        assert!(thinking.is_none());
    }
}
