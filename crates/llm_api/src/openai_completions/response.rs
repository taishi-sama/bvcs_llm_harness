//! Parsing of legacy OpenAI-compatible **Chat Completions API** response
//! bodies (`POST /chat/completions`) back into
//! [`llm_history_model::history::Entry`] histories.
//!
//! Mapping overview (inverse of [`super`]'s serialization):
//! - assistant message with `tool_calls[]` -> `Entry::ResponseToolCall`
//! - assistant message with `content`      -> `Entry::Response` (thinking first,
//!   then text; Chat Completions cannot represent finer interleaving)
//! - `n` > 1 produces multiple choices; only `index` 0 is honored
//!
//! Tool results arrive as later `role:"tool"` messages produced client-side, so
//! this parser yields assistant-side entries only. `reasoning_content` is
//! attached as [`Thinking::Normal`] to the produced assistant entry.

use crate::usage::{ParsedResponse, Usage, UsageProvider};
use llm_history_model::history::{Entry, ResponseBlock, Thinking, ToolCall};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

/// Top-level `POST /chat/completions` response body.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionResponse {
    /// Optional for compatibility with lightweight compatible servers.
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub choices: Vec<Choice>,
    pub created: Option<i64>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}

/// Chat Completions wire usage, shared by OpenAI, DeepSeek and Z.ai.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// DeepSeek cache-hit prompt tokens.
    #[serde(default)]
    pub prompt_cache_hit_tokens: Option<u64>,
    /// DeepSeek cache-miss prompt tokens.
    #[serde(default)]
    pub prompt_cache_miss_tokens: Option<u64>,
    /// Unknown numeric usage fields, preserved by the normalizer.
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
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
    pub content: Option<MessageContent>,
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

/// Chat Completions normally returns a string, but some compatible servers
/// return typed content parts. Accept both.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
    #[serde(other)]
    Other,
}

impl MessageContent {
    /// Flattens the content into plain text (`text` and `refusal` parts).
    pub fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::Refusal { refusal } => Some(refusal.as_str()),
                    ContentPart::Other => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
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

    /// Normalized usage for this response, labeled as OpenAI.
    pub fn usage(&self) -> Option<Usage> {
        self.usage_for(UsageProvider::OpenAI)
    }

    /// Normalized usage with an explicit provider label (for example DeepSeek
    /// or Z.ai, which share this wire format).
    pub fn usage_for(&self, provider: UsageProvider) -> Option<Usage> {
        Usage::from_chat_for(self.usage.as_ref()?, provider)
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

    /// Converts the response's first choice back into history entries.
    ///
    /// - `tool_calls[]` merge into a single [`Entry::ResponseToolCall`]
    /// - otherwise the message becomes an [`Entry::Response`]
    ///   (`content` and `refusal` are concatenated; empty text is skipped)
    ///
    /// The result contains assistant-side entries only — append them to an
    /// existing user-first history before validation or re-serialization.
    pub fn to_entries(&self) -> Vec<Entry> {
        // Honor choice index 0 even when the server orders choices differently.
        let Some(choice) = self
            .choices
            .iter()
            .find(|choice| choice.index.unwrap_or(0) == 0)
            .or_else(|| self.choices.first())
        else {
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
        let content = msg
            .content
            .as_ref()
            .map(MessageContent::text)
            .filter(|text| !text.is_empty());

        // Chat Completions exposes reasoning as a message-level field, so the
        // best ordering we can reconstruct is thinking followed by text.
        let mut blocks: Vec<ResponseBlock> =
            thinking.into_iter().map(ResponseBlock::Thinking).collect();

        if !msg.tool_calls.is_empty() {
            if let Some(text) = content {
                blocks.push(ResponseBlock::Text(text));
            }
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
            return vec![Entry::ResponseToolCall { blocks, tool_call }];
        }

        let mut text = content.unwrap_or_default();
        if let Some(refusal) = msg.refusal.as_deref()
            && !refusal.is_empty()
        {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(refusal);
        }
        if !text.is_empty() {
            blocks.push(ResponseBlock::Text(text));
        }
        if blocks.is_empty() {
            Vec::new()
        } else {
            vec![Entry::Response { blocks }]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::{ModelConfig, ModelKind};
    use crate::test_support::{text_user, validated_assistant_history};
    use llm_history_model::history::Entry as E;
    use llm_history_model::history::{ResponseBlock, ToolCallResult};

    fn config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::OpenAICompatible {
                model_name: "gpt-test".into(),
                streaming: false,
                prompt_cache_key: "key".into(),
                provider: crate::usage::UsageProvider::OpenAI,
                include_usage: true,
            },
            tools: Default::default(),
            max_tokens: None,
        }
    }

    fn validated_history(entries: Vec<E>) {
        validated_assistant_history(entries, &config());
    }

    #[test]
    fn usage_is_normalized_and_parsed() {
        let resp = CompletionResponse::from_value(serde_json::json!({
            "id": "x", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
                "prompt_tokens_details": {"cached_tokens": 8},
                "completion_tokens_details": {"reasoning_tokens": 2}
            }
        }))
        .expect("parses");

        let parsed = resp.parse();
        assert_eq!(parsed.entries.len(), 1);
        let usage = parsed.usage.expect("usage");
        assert_eq!(usage.input_tokens, Some(20));
        assert_eq!(usage.output_tokens, Some(5));
        assert_eq!(usage.cached_input_tokens, Some(8));
        assert_eq!(usage.reasoning_tokens, Some(2));
        assert_eq!(usage.total(), Some(25));
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
        assert_eq!(entries[0].response_joined_text().as_deref(), Some("Hello!"));
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
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
        let E::ResponseToolCall { blocks, tool_call } = &entries[0] else {
            panic!("expected ResponseToolCall");
        };
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Normal { thinking }] if thinking == "planning"
        ));
        assert!(entries[0].response_text_blocks().is_empty());
        assert_eq!(blocks.len(), 1);
        assert_eq!(tool_call.len(), 2);
        assert_eq!(tool_call[0].tool_name, "a");
        assert_eq!(tool_call[1].toolcall_params, "{}"); // empty arguments defaulted
        // NB: not validated — tool results arrive as later client-side entries
    }

    #[test]
    fn parses_reasoning_only_message() {
        let resp = CompletionResponse::from_value(serde_json::json!({
            "id": "x", "model": "m",
            "choices": [{"index": 0, "finish_reason": "length",
                "message": {"role": "assistant", "content": "",
                            "reasoning_content": "still thinking"}}]
        }))
        .expect("parses");

        let entries = resp.to_entries();
        assert_eq!(entries.len(), 1);
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Normal { thinking }] if thinking == "still thinking"
        ));
        assert!(entries[0].response_text_blocks().is_empty());
    }

    #[test]
    fn parses_content_parts() {
        let resp = CompletionResponse::from_value(serde_json::json!({
            "id": "x", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "world"}
                    ]}}]
        }))
        .expect("parses");

        let entries = resp.to_entries();
        assert_eq!(
            entries[0].response_joined_text().as_deref(),
            Some("Hello\nworld")
        );
    }

    #[test]
    fn picks_choice_zero_regardless_of_array_order() {
        let resp = CompletionResponse::from_value(serde_json::json!({
            "choices": [
                {"index": 1, "message": {"role": "assistant", "content": "wrong"}},
                {"index": 0, "message": {"role": "assistant", "content": "right"}}
            ]
        }))
        .expect("parses");

        let entries = resp.to_entries();
        assert_eq!(entries[0].response_joined_text().as_deref(), Some("right"));
    }

    #[test]
    fn missing_metadata_is_tolerated() {
        let resp = CompletionResponse::from_value(serde_json::json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}]
        }))
        .expect("parses without id/model/created");

        assert!(resp.id.is_none());
        assert!(resp.model.is_none());
        assert_eq!(resp.to_entries().len(), 1);
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
        assert_eq!(
            entries[0].response_joined_text().as_deref(),
            Some("partial\nblocked")
        );
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
            E::response(vec![
                ResponseBlock::Thinking(Thinking::Normal {
                    thinking: "hmm".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]),
        ]);
    }

    #[test]
    fn round_trip_tool_calls_with_preamble() {
        assert_round_trip(&[
            text_user("q"),
            E::ResponseToolCall {
                blocks: vec![
                    ResponseBlock::Thinking(Thinking::Normal {
                        thinking: "plan".into(),
                    }),
                    ResponseBlock::Text("calling now".into()),
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
