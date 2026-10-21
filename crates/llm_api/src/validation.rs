use crate::model_config::{ModelConfig, ModelKind, SystemMessagePolicy};
use llm_history_model::history::{Entry, ResponseBlock, Thinking};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// SystemPrompt must appear exactly once, at index 0
    MisplacedSystemPrompt { index: usize },
    /// Conversation must open with a user turn (the API enforces this)
    MustStartWithUser { index: usize },
    /// Illegal transition between entry kinds (breaks role alternation)
    IllegalTransition {
        index: usize,
        from: &'static str,
        to: &'static str,
    },
    /// UserMessage with neither text nor image
    EmptyUserMessage { index: usize },
    /// Assistant text content is empty
    EmptyAssistantContent { index: usize },
    /// toolcall_params is not a valid JSON *object*
    InvalidToolParams { entry: usize, call: usize },
    /// tool_use_id referenced by a result was not called in the immediately preceding batch
    OrphanToolResult {
        entry: usize,
        result: usize,
        tool_use_id: String,
    },
    /// tool_use_id used more than once across history
    DuplicateToolCallId { tool_use_id: String },
    /// History ends with tool calls that have no results
    UnansweredToolCalls { tool_use_ids: Vec<String> },
    /// Anthropic requires a signature to resend thinking blocks;
    /// `Thinking::Normal` cannot be round-tripped and would be silently dropped.
    UnsignedThinkingRejected { index: usize },
    /// A thinking block belongs to a different provider's opaque format and
    /// cannot be replayed to the configured one (for example Anthropic
    /// `redacted_thinking` sent to OpenAI, or OpenAI `encrypted_content` sent
    /// to Anthropic).
    IncompatibleThinking {
        index: usize,
        provider: &'static str,
        kind: &'static str,
    },
    /// A tool call entry has no calls and no visible content.
    EmptyToolCall { index: usize },
    /// A tool result entry contains no results.
    EmptyToolResult { index: usize },
    /// A tool call/result identifier is empty.
    EmptyToolIdentifier {
        entry: usize,
        call: usize,
        field: &'static str,
    },
    /// OpenAI exposes reasoning per assistant message, not as multiple
    /// independent thinking blocks in one message.
    MultipleThinkingBlocks { index: usize, count: usize },
    /// A system/environment message has no content.
    EmptySystemMessage { index: usize },
    /// A system/environment message cannot be inserted while tool calls are
    /// still waiting for results.
    SystemMessageDuringToolCall { index: usize },
    /// A user message cannot be inserted while tool calls are still waiting
    /// for results.
    UserMessageDuringToolCall { index: usize },
    /// The configured provider cannot represent an in-conversation system
    /// message (Anthropic with `SystemMessagePolicy::Reject`).
    SystemMessageUnsupported { index: usize },
    /// Anthropic extended thinking requires `1024 <= budget_tokens < max_tokens`.
    InvalidThinkingBudget { budget_tokens: u32, max_tokens: u32 },
    /// Config kind does not match the serializer used (e.g. calling an
    /// Anthropic serializer with an `OpenAICompatible` config).
    WrongProviderKind {
        expected: &'static str,
        actual: &'static str,
    },
    /// A streaming draft is still present. Drafts are live state and cannot be
    /// replayed to a provider; finish the stream and commit the draft first.
    DraftNotFinal { index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    System,
    SystemMessage,
    UserMsg,
    Assistant,
    Results,
    /// Live streaming state; rejected before serialization.
    Draft,
}

const fn kind_of(e: &Entry) -> Kind {
    match e {
        Entry::SystemPrompt { .. } => Kind::System,
        Entry::SystemMessage { .. } => Kind::SystemMessage,
        Entry::UserMessage { .. } => Kind::UserMsg,
        Entry::Response { .. } | Entry::ResponseToolCall { .. } => Kind::Assistant,
        Entry::ToolCallResult { .. } => Kind::Results,
        Entry::Draft { .. } => Kind::Draft,
    }
}

const fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::System => "system",
        Kind::SystemMessage => "system_message",
        Kind::UserMsg => "user",
        Kind::Assistant => "assistant",
        Kind::Results => "results",
        Kind::Draft => "draft",
    }
}

/// Legal turn structure once grouped the way wire APIs require:
/// Start -> User, User -> Assistant, Assistant -> User | Results,
/// Results -> Assistant | Results | User
fn transition_ok(from: Option<Kind>, to: Kind) -> bool {
    match from {
        None => to == Kind::UserMsg,
        Some(Kind::UserMsg) => matches!(to, Kind::Assistant),
        Some(Kind::Assistant) => matches!(to, Kind::UserMsg | Kind::Results),
        // A user message may follow a completed tool-result batch. Anthropic
        // folds it into the same user turn; Chat Completions and Responses
        // send it immediately after the tool outputs.
        Some(Kind::Results) => matches!(to, Kind::Assistant | Kind::Results | Kind::UserMsg),
        // Drafts never reach this check (they are rejected and skipped), but
        // keep the transition permissive so an earlier draft cannot cascade
        // into unrelated transition errors.
        Some(Kind::Draft) => true,
        Some(Kind::System) => true,
        Some(Kind::SystemMessage) => true,
    }
}

/// Previous non-system, non-draft entry kind before `upto`
fn last_kind(entries: &[Entry], upto: usize) -> Option<Kind> {
    entries[..upto]
        .iter()
        .rev()
        .map(kind_of)
        .find(|k| !matches!(k, Kind::System | Kind::SystemMessage | Kind::Draft))
}

fn is_json_object(s: &str) -> bool {
    matches!(serde_json::from_str::<Value>(s), Ok(Value::Object(_)))
}

fn assistant_thinking(entry: &Entry) -> Vec<&Thinking> {
    entry
        .as_response_blocks()
        .map(|blocks| blocks.iter().filter_map(ResponseBlock::thinking).collect())
        .unwrap_or_default()
}

/// Provider-specific checks layered on top of the shared structural rules.
fn check_provider_rules(
    entries: &[Entry],
    config: &ModelConfig,
    errors: &mut Vec<ValidationError>,
) {
    match &config.kind {
        ModelKind::AnthropicCompatible {
            allow_unsigned_thinking,
            system_message_policy,
            ..
        } => {
            if matches!(system_message_policy, SystemMessagePolicy::Reject) {
                for (i, entry) in entries.iter().enumerate() {
                    if matches!(entry, Entry::SystemMessage { .. }) {
                        errors.push(ValidationError::SystemMessageUnsupported { index: i });
                    }
                }
            }
            for (i, entry) in entries.iter().enumerate() {
                let thinking = assistant_thinking(entry);
                // Real Anthropic requires a signature to replay thinking.
                // Compatible servers (llama.cpp) send an empty signature.
                if !*allow_unsigned_thinking
                    && thinking
                        .iter()
                        .any(|t| matches!(t, Thinking::Normal { .. }))
                {
                    errors.push(ValidationError::UnsignedThinkingRejected { index: i });
                }
                // OpenAI encrypted reasoning is opaque in a different format.
                if thinking
                    .iter()
                    .any(|t| matches!(t, Thinking::Encrypted { .. }))
                {
                    errors.push(ValidationError::IncompatibleThinking {
                        index: i,
                        provider: "anthropic_compatible",
                        kind: "encrypted",
                    });
                }
            }
        }
        ModelKind::OpenAICompatible { .. } => {
            for (i, entry) in entries.iter().enumerate() {
                let thinking = assistant_thinking(entry);
                // OpenAI reasoning is attached per assistant message. Multiple
                // thinking blocks in one `Response` (or one `ResponseToolCall`)
                // would be emitted as multiple reasoning items for the same
                // message, which is not a valid OpenAI history.
                if thinking.len() > 1 {
                    errors.push(ValidationError::MultipleThinkingBlocks {
                        index: i,
                        count: thinking.len(),
                    });
                }
                // Anthropic redacted thinking is an opaque payload that cannot
                // be converted to OpenAI reasoning text.
                if thinking
                    .iter()
                    .any(|t| matches!(t, Thinking::Redacted { .. }))
                {
                    errors.push(ValidationError::IncompatibleThinking {
                        index: i,
                        provider: "openai_compatible",
                        kind: "redacted",
                    });
                }
            }
        }
    }
}

/// Validates a raw history against `config` for serializability into the
/// configured provider's wire format.
///
/// Enforced rules:
/// - at most one [`Entry::SystemPrompt`], at index 0
/// - strict alternation `User -> Assistant -> Results* -> (User | Assistant)`,
///   starting with user; a user turn may follow a completed tool-result batch
///   (Anthropic folds it into the same turn, OpenAI sends it after the tool
///   messages)
/// - [`Entry::SystemMessage`] is a non-turn entry: it may appear between turns
///   but not while tool calls are waiting for results
/// - non-empty user messages (text or image) and non-empty assistant responses
///   ([`Entry::Response`] must have at least one non-empty block)
/// - every `toolcall_params` parses as a JSON object
/// - non-empty tool-call/tool-result batches and non-empty identifiers
/// - every result references a call from the immediately preceding batch
/// - no duplicate tool_use_ids, no unanswered calls at the end of history
/// - no [`Entry::Draft`] (finish the stream and commit it first)
///
/// Provider-specific (`config.kind`):
/// - `AnthropicCompatible`: rejects unsigned [`Thinking::Normal`] blocks
///   unless `allow_unsigned_thinking` is set, always rejects OpenAI
///   [`Thinking::Encrypted`] blocks, and rejects [`Entry::SystemMessage`]
///   when `system_message_policy` is `Reject`
/// - `OpenAICompatible`: rejects Anthropic [`Thinking::Redacted`] blocks and
///   more than one thinking block in a single assistant message
pub fn validate_entries(
    entries: &[Entry],
    config: &ModelConfig,
) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();
    let mut call_ids: HashSet<&str> = HashSet::new();
    let mut result_ids: HashSet<&str> = HashSet::new();
    let mut pending: HashSet<&str> = HashSet::new();

    for (i, entry) in entries.iter().enumerate() {
        if let Entry::SystemPrompt { .. } = entry {
            if i != 0 {
                errors.push(ValidationError::MisplacedSystemPrompt { index: i });
            }
            continue;
        }

        // A draft is live streaming state. It is structurally allowed while a
        // response is being received, but it can never be sent back to a
        // provider, so any draft makes the history non-serializable.
        if entry.is_draft() {
            errors.push(ValidationError::DraftNotFinal { index: i });
            continue;
        }

        // System/environment messages are non-turn entries: they may appear
        // between turns but must not break tool-call/result pairing.
        if let Entry::SystemMessage { content, .. } = entry {
            if content.is_empty() {
                errors.push(ValidationError::EmptySystemMessage { index: i });
            }
            if !pending.is_empty() {
                errors.push(ValidationError::SystemMessageDuringToolCall { index: i });
            }
            continue;
        }

        let prev = last_kind(entries, i);
        let k = kind_of(entry);
        if !transition_ok(prev, k) {
            errors.push(ValidationError::IllegalTransition {
                index: i,
                from: prev.map(kind_name).unwrap_or("start"),
                to: kind_name(k),
            });
        }

        match entry {
            Entry::SystemPrompt { .. } => unreachable!("handled above"),
            Entry::SystemMessage { .. } => unreachable!("handled above"),
            Entry::Draft { .. } => unreachable!("handled above"),
            Entry::UserMessage { content } => {
                // Placing a user turn between a tool call and its results would
                // break the call/result pairing on the wire.
                if !pending.is_empty() {
                    errors.push(ValidationError::UserMessageDuringToolCall { index: i });
                }
                let has_text = content.text.as_ref().is_some_and(|t| !t.is_empty());
                if !has_text && content.image.is_none() {
                    errors.push(ValidationError::EmptyUserMessage { index: i });
                }
            }
            Entry::Response { blocks } => {
                if blocks.is_empty() || blocks.iter().all(|block| block.is_empty()) {
                    errors.push(ValidationError::EmptyAssistantContent { index: i });
                }
            }
            Entry::ResponseToolCall { blocks, tool_call } => {
                if tool_call.is_empty()
                    && (blocks.is_empty() || blocks.iter().all(|block| block.is_empty()))
                {
                    errors.push(ValidationError::EmptyToolCall { index: i });
                }
                // A previous tool-call batch must have been answered before a
                // new one starts. Without this, an empty ToolCallResult entry
                // between two batches would silently clear the pending set.
                if !pending.is_empty() {
                    errors.push(ValidationError::UnansweredToolCalls {
                        tool_use_ids: pending.iter().map(|s| (*s).to_string()).collect(),
                    });
                    pending.clear();
                }
                for (ci, tc) in tool_call.iter().enumerate() {
                    if tc.tool_use_id.trim().is_empty() {
                        errors.push(ValidationError::EmptyToolIdentifier {
                            entry: i,
                            call: ci,
                            field: "tool_use_id",
                        });
                    }
                    if tc.tool_name.trim().is_empty() {
                        errors.push(ValidationError::EmptyToolIdentifier {
                            entry: i,
                            call: ci,
                            field: "tool_name",
                        });
                    }
                    if !call_ids.insert(tc.tool_use_id.as_str()) {
                        errors.push(ValidationError::DuplicateToolCallId {
                            tool_use_id: tc.tool_use_id.clone(),
                        });
                    }
                    pending.insert(tc.tool_use_id.as_str());
                    if !is_json_object(&tc.toolcall_params) {
                        errors.push(ValidationError::InvalidToolParams { entry: i, call: ci });
                    }
                }
            }
            Entry::ToolCallResult { tool_call } => {
                if tool_call.is_empty() {
                    errors.push(ValidationError::EmptyToolResult { index: i });
                }
                for (ri, r) in tool_call.iter().enumerate() {
                    if r.tool_use_id.trim().is_empty() {
                        errors.push(ValidationError::EmptyToolIdentifier {
                            entry: i,
                            call: ri,
                            field: "tool_use_id",
                        });
                    }
                    if !result_ids.insert(r.tool_use_id.as_str()) {
                        errors.push(ValidationError::DuplicateToolCallId {
                            tool_use_id: r.tool_use_id.clone(),
                        });
                    }
                    if !pending.remove(r.tool_use_id.as_str()) {
                        errors.push(ValidationError::OrphanToolResult {
                            entry: i,
                            result: ri,
                            tool_use_id: r.tool_use_id.clone(),
                        });
                    }
                }
            }
        }
    }

    if !pending.is_empty() {
        errors.push(ValidationError::UnansweredToolCalls {
            tool_use_ids: pending.iter().map(|s| (*s).to_string()).collect(),
        });
    }

    check_provider_rules(entries, config, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_history_model::history::{
        DraftBlock, Image, ImageMIME, MessageContent, SystemMessageKind, ToolCall, ToolCallResult,
    };

    fn openai_config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::OpenAICompatible {
                model_name: "gpt-test".into(),
                streaming: false,
                prompt_cache_key: "test_key".into(),
                provider: crate::usage::UsageProvider::OpenAI,
                include_usage: true,
            },
            tools: Default::default(),
            max_tokens: None,
        }
    }

    fn anthropic_config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::AnthropicCompatible {
                model_name: "claude-test".into(),
                streaming: false,
                thinking: None,
                allow_unsigned_thinking: false,
                system_message_policy: Default::default(),
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

    fn assistant_text(t: &str) -> Entry {
        Entry::response_text(t)
    }

    fn system_message(content: &str) -> Entry {
        Entry::SystemMessage {
            kind: SystemMessageKind::Environment,
            content: content.into(),
        }
    }

    fn tool_call(id: &str) -> Entry {
        Entry::ResponseToolCall {
            blocks: Vec::new(),
            tool_call: vec![ToolCall {
                tool_name: "get_weather".into(),
                toolcall_params: r#"{"city":"Oslo"}"#.into(),
                tool_use_id: id.into(),
            }],
        }
    }

    fn tool_result(id: &str) -> Entry {
        Entry::ToolCallResult {
            tool_call: vec![ToolCallResult {
                result: "sunny".into(),
                is_error: false,
                tool_use_id: id.into(),
            }],
        }
    }

    #[test]
    fn accepts_valid_tool_round_trip() {
        let h = vec![
            text_user("weather?"),
            tool_call("call_1"),
            tool_result("call_1"),
            assistant_text("It is sunny."),
        ];
        assert_eq!(validate_entries(&h, &openai_config()), Ok(()));
        assert_eq!(validate_entries(&h, &anthropic_config()), Ok(()));
    }

    #[test]
    fn accepts_user_message_after_tool_results() {
        // The harness inserts queued user/agent messages at tool boundaries.
        let h = vec![
            text_user("weather?"),
            tool_call("call_1"),
            tool_result("call_1"),
            text_user("also check Oslo tomorrow"),
            assistant_text("Done."),
        ];
        assert_eq!(validate_entries(&h, &openai_config()), Ok(()));
        assert_eq!(validate_entries(&h, &anthropic_config()), Ok(()));
    }

    #[test]
    fn rejects_user_message_before_tool_results() {
        let h = vec![
            text_user("q"),
            tool_call("c1"),
            text_user("interrupt"),
            tool_result("c1"),
        ];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::UserMessageDuringToolCall { index: 2 })
        ));
    }

    #[test]
    fn accepts_parallel_calls_split_batches() {
        let h = vec![
            text_user("q"),
            Entry::ResponseToolCall {
                blocks: vec![ResponseBlock::Thinking(Thinking::Signed {
                    thinking: "hmm".into(),
                    signature: "sig".into(),
                })],
                tool_call: vec![
                    ToolCall {
                        tool_name: "a".into(),
                        toolcall_params: "{}".into(),
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
            Entry::response(vec![
                ResponseBlock::Thinking(Thinking::Signed {
                    thinking: "t".into(),
                    signature: "sig".into(),
                }),
                ResponseBlock::Text("done".into()),
            ]),
        ];
        assert_eq!(validate_entries(&h, &openai_config()), Ok(()));
        assert_eq!(validate_entries(&h, &anthropic_config()), Ok(()));
    }

    #[test]
    fn anthropic_rejects_unsigned_thinking_but_openai_accepts() {
        let h = vec![
            text_user("q"),
            Entry::response(vec![
                ResponseBlock::Thinking(Thinking::Normal {
                    thinking: "hmm".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]),
        ];
        assert_eq!(validate_entries(&h, &openai_config()), Ok(()));
        assert_eq!(
            validate_entries(&h, &anthropic_config()),
            Err(vec![ValidationError::UnsignedThinkingRejected { index: 1 }])
        );
    }

    #[test]
    fn anthropic_accepts_unsigned_thinking_when_allowed() {
        let cfg = ModelConfig {
            kind: ModelKind::AnthropicCompatible {
                model_name: "llama.cpp".into(),
                streaming: false,
                thinking: None,
                allow_unsigned_thinking: true,
                system_message_policy: Default::default(),
            },
            tools: Default::default(),
            max_tokens: Some(1024),
        };
        let h = vec![
            text_user("q"),
            Entry::response(vec![
                ResponseBlock::Thinking(Thinking::Normal {
                    thinking: "hmm".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]),
        ];
        assert_eq!(validate_entries(&h, &cfg), Ok(()));
    }

    #[test]
    fn anthropic_accepts_signed_thinking() {
        let h = vec![
            text_user("q"),
            Entry::response(vec![
                ResponseBlock::Thinking(Thinking::Signed {
                    thinking: "hmm".into(),
                    signature: "sig==".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]),
        ];
        assert_eq!(validate_entries(&h, &anthropic_config()), Ok(()));
    }

    #[test]
    fn rejects_misplaced_system_prompt() {
        let h = vec![
            text_user("hi"),
            Entry::SystemPrompt {
                content: "sys".into(),
            },
        ];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::MisplacedSystemPrompt { index: 1 })
        ));
    }

    #[test]
    fn rejects_history_starting_with_assistant() {
        let h = vec![assistant_text("hello")];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::IllegalTransition {
                index: 0,
                from: "start",
                to: "assistant"
            })
        ));
    }

    #[test]
    fn rejects_consecutive_user_messages() {
        let h = vec![text_user("a"), text_user("b")];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.iter().any(|err| matches!(
                err,
                ValidationError::IllegalTransition { index: 1, .. }
            ))
        ));
    }

    #[test]
    fn rejects_empty_user_message() {
        let h = vec![Entry::UserMessage {
            content: MessageContent {
                text: None,
                image: None,
            },
        }];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::EmptyUserMessage { index: 0 })
        ));
    }

    #[test]
    fn accepts_image_only_user_message() {
        let h = vec![Entry::UserMessage {
            content: MessageContent {
                text: None,
                image: Some(Image {
                    mime: ImageMIME::PNG,
                    base64: "aGk=".into(),
                }),
            },
        }];
        assert_eq!(validate_entries(&h, &openai_config()), Ok(()));
    }

    #[test]
    fn rejects_orphan_tool_result() {
        let h = vec![text_user("q"), tool_result("ghost")];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.iter().any(|err| matches!(
                err,
                ValidationError::OrphanToolResult { tool_use_id, .. } if tool_use_id == "ghost"
            ))
        ));
    }

    #[test]
    fn rejects_duplicate_call_id() {
        let h = vec![
            text_user("q"),
            tool_call("dup"),
            tool_result("dup"),
            tool_call("dup"),
            tool_result("dup"),
        ];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::DuplicateToolCallId {
                tool_use_id: "dup".into()
            })
        ));
    }

    #[test]
    fn rejects_non_object_params() {
        let h = vec![
            text_user("q"),
            Entry::ResponseToolCall {
                blocks: Vec::new(),
                tool_call: vec![ToolCall {
                    tool_name: "t".into(),
                    toolcall_params: "[1,2]".into(),
                    tool_use_id: "c".into(),
                }],
            },
            tool_result("c"),
        ];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::InvalidToolParams { entry: 1, call: 0 })
        ));
    }

    #[test]
    fn rejects_unanswered_trailing_calls() {
        let h = vec![text_user("q"), tool_call("dangling")];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::UnansweredToolCalls {
                tool_use_ids: vec!["dangling".into()]
            })
        ));
    }

    #[test]
    fn rejects_streaming_draft() {
        let h = vec![text_user("q"), Entry::draft()];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::DraftNotFinal { index: 1 })
        ));
    }

    #[test]
    fn rejects_draft_in_the_middle() {
        let mut draft = Entry::draft();
        draft
            .as_draft_mut()
            .unwrap()
            .push(llm_history_model::history::DraftBlock::Text {
                text: "partial".into(),
            });
        let h = vec![text_user("q"), draft, assistant_text("done")];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::DraftNotFinal { index: 1 })
        ));
    }

    #[test]
    fn rescued_draft_is_valid_history() {
        let draft = Entry::Draft {
            blocks: vec![
                DraftBlock::Thinking {
                    text: "hmm".into(),
                    signature: None,
                    encrypted_content: None,
                },
                DraftBlock::Text {
                    text: "partial answer".into(),
                },
                DraftBlock::ToolCall {
                    index: 0,
                    tool_use_id: Some("c1".into()),
                    tool_name: Some("get_weather".into()),
                    arguments: r#"{"city":"Os"#.into(),
                },
            ],
        };
        let rescued = draft
            .try_to_rescue_draft("[thinking only]", false)
            .expect("text is rescuable");
        let h = vec![text_user("q"), rescued];
        assert_eq!(validate_entries(&h, &openai_config()), Ok(()));
    }

    #[test]
    fn accepts_system_message_between_turns() {
        let h = vec![
            text_user("q"),
            assistant_text("answer"),
            system_message("workspace changed"),
            text_user("next"),
        ];
        assert_eq!(validate_entries(&h, &openai_config()), Ok(()));
        assert_eq!(validate_entries(&h, &anthropic_config()), Ok(()));
    }

    #[test]
    fn rejects_system_message_while_tool_call_is_pending() {
        let h = vec![
            text_user("q"),
            tool_call("c1"),
            system_message("workspace changed"),
        ];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::SystemMessageDuringToolCall { index: 2 })
        ));
    }

    #[test]
    fn rejects_empty_system_message() {
        let h = vec![text_user("q"), system_message("")];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::EmptySystemMessage { index: 1 })
        ));
    }

    #[test]
    fn anthropic_rejects_system_message_when_policy_is_reject() {
        let cfg = ModelConfig {
            kind: ModelKind::AnthropicCompatible {
                model_name: "claude-test".into(),
                streaming: false,
                thinking: None,
                allow_unsigned_thinking: false,
                system_message_policy: SystemMessagePolicy::Reject,
            },
            tools: Default::default(),
            max_tokens: Some(1024),
        };
        let h = vec![text_user("q"), system_message("workspace changed")];
        assert!(matches!(
            validate_entries(&h, &cfg),
            Err(e) if e.contains(&ValidationError::SystemMessageUnsupported { index: 1 })
        ));
    }

    #[test]
    fn collects_multiple_errors() {
        let h = vec![tool_call("x"), tool_result("y")];
        assert!(matches!(validate_entries(&h, &openai_config()), Err(e) if e.len() >= 3));
    }

    #[test]
    fn rejects_redacted_thinking_for_openai() {
        let h = vec![
            text_user("q"),
            Entry::response(vec![
                ResponseBlock::Thinking(Thinking::Redacted {
                    thinking: "opaque".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]),
        ];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::IncompatibleThinking {
                index: 1,
                provider: "openai_compatible",
                kind: "redacted",
            })
        ));
    }

    #[test]
    fn rejects_encrypted_thinking_for_anthropic() {
        let h = vec![
            text_user("q"),
            Entry::response(vec![
                ResponseBlock::Thinking(Thinking::Encrypted {
                    thinking: "opaque".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]),
        ];
        assert!(matches!(
            validate_entries(&h, &anthropic_config()),
            Err(e) if e.contains(&ValidationError::IncompatibleThinking {
                index: 1,
                provider: "anthropic_compatible",
                kind: "encrypted",
            })
        ));
        // OpenAI accepts its own encrypted payload.
        assert_eq!(validate_entries(&h, &openai_config()), Ok(()));
    }

    #[test]
    fn thinking_only_requires_non_empty_thinking() {
        let valid = vec![
            text_user("q"),
            Entry::response_thinking(vec![Thinking::Encrypted {
                thinking: "opaque".into(),
            }]),
        ];
        assert_eq!(validate_entries(&valid, &openai_config()), Ok(()));

        let empty = vec![text_user("q"), Entry::response(Vec::new())];
        assert!(matches!(
            validate_entries(&empty, &openai_config()),
            Err(e) if e.contains(&ValidationError::EmptyAssistantContent { index: 1 })
        ));
    }

    #[test]
    fn openai_rejects_multiple_thinking_blocks_in_one_response() {
        let h = vec![
            text_user("q"),
            Entry::response(vec![
                ResponseBlock::Thinking(Thinking::Encrypted {
                    thinking: "one".into(),
                }),
                ResponseBlock::Text("answer".into()),
                ResponseBlock::Thinking(Thinking::Encrypted {
                    thinking: "two".into(),
                }),
            ]),
        ];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::MultipleThinkingBlocks {
                index: 1,
                count: 2,
            })
        ));

        // Anthropic allows multiple signed thinking blocks (each keeps its own
        // signature), so the same shape is valid there.
        let h = vec![
            text_user("q"),
            Entry::response(vec![
                ResponseBlock::Thinking(Thinking::Signed {
                    thinking: "one".into(),
                    signature: "s1".into(),
                }),
                ResponseBlock::Text("answer".into()),
                ResponseBlock::Thinking(Thinking::Signed {
                    thinking: "two".into(),
                    signature: "s2".into(),
                }),
            ]),
        ];
        assert_eq!(validate_entries(&h, &anthropic_config()), Ok(()));
    }

    #[test]
    fn openai_rejects_multiple_thinking_blocks_in_tool_call_message() {
        let h = vec![
            text_user("q"),
            Entry::ResponseToolCall {
                blocks: vec![
                    ResponseBlock::Thinking(Thinking::Encrypted {
                        thinking: "one".into(),
                    }),
                    ResponseBlock::Thinking(Thinking::Encrypted {
                        thinking: "two".into(),
                    }),
                    ResponseBlock::Text("answer".into()),
                ],
                tool_call: Vec::new(),
            },
        ];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::MultipleThinkingBlocks {
                index: 1,
                count: 2,
            })
        ));
    }

    #[test]
    fn rejects_empty_tool_call_batch() {
        let h = vec![
            text_user("q"),
            Entry::ResponseToolCall {
                blocks: Vec::new(),
                tool_call: Vec::new(),
            },
        ];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::EmptyToolCall { index: 1 })
        ));
    }

    #[test]
    fn rejects_empty_tool_result_batch() {
        let h = vec![
            text_user("q"),
            tool_call("c1"),
            Entry::ToolCallResult {
                tool_call: Vec::new(),
            },
        ];
        assert!(matches!(
            validate_entries(&h, &openai_config()),
            Err(e) if e.contains(&ValidationError::EmptyToolResult { index: 2 })
        ));
    }

    #[test]
    fn rejects_empty_tool_identifiers() {
        let h = vec![
            text_user("q"),
            Entry::ResponseToolCall {
                blocks: Vec::new(),
                tool_call: vec![ToolCall {
                    tool_name: String::new(),
                    toolcall_params: "{}".into(),
                    tool_use_id: String::new(),
                }],
            },
        ];
        let err = validate_entries(&h, &openai_config()).unwrap_err();
        assert!(err.contains(&ValidationError::EmptyToolIdentifier {
            entry: 1,
            call: 0,
            field: "tool_use_id",
        }));
        assert!(err.contains(&ValidationError::EmptyToolIdentifier {
            entry: 1,
            call: 0,
            field: "tool_name",
        }));
    }

    #[test]
    fn empty_result_batch_cannot_hide_unanswered_calls() {
        let h = vec![
            text_user("q"),
            tool_call("c1"),
            Entry::ToolCallResult {
                tool_call: Vec::new(),
            },
            tool_call("c2"),
        ];
        let err = validate_entries(&h, &openai_config()).unwrap_err();
        assert!(err.contains(&ValidationError::UnansweredToolCalls {
            tool_use_ids: vec!["c1".into()],
        }));
        assert!(err.contains(&ValidationError::EmptyToolResult { index: 2 }));
    }
}
