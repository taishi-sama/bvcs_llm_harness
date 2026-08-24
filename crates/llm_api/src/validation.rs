use crate::model_config::{ModelConfig, ModelKind};
use llm_history_model::history::{Entry, Thinking};
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    System,
    UserMsg,
    Assistant,
    Results,
}

const fn kind_of(e: &Entry) -> Kind {
    match e {
        Entry::SystemPrompt { .. } => Kind::System,
        Entry::UserMessage { .. } => Kind::UserMsg,
        Entry::ResponseNormal { .. } | Entry::ResponseToolCall { .. } => Kind::Assistant,
        Entry::ToolCallResult { .. } => Kind::Results,
    }
}

const fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::System => "system",
        Kind::UserMsg => "user",
        Kind::Assistant => "assistant",
        Kind::Results => "results",
    }
}

/// Legal turn structure once grouped the way wire APIs require:
/// Start -> User, User -> Assistant, Assistant -> User | Results, Results -> Assistant | Results
fn transition_ok(from: Option<Kind>, to: Kind) -> bool {
    match from {
        None => to == Kind::UserMsg,
        Some(Kind::UserMsg) => matches!(to, Kind::Assistant),
        Some(Kind::Assistant) => matches!(to, Kind::UserMsg | Kind::Results),
        Some(Kind::Results) => matches!(to, Kind::Assistant | Kind::Results),
        Some(Kind::System) => true,
    }
}

/// Previous non-system entry kind before `upto`
fn last_kind(entries: &[Entry], upto: usize) -> Option<Kind> {
    entries[..upto]
        .iter()
        .rev()
        .map(kind_of)
        .find(|k| *k != Kind::System)
}

fn is_json_object(s: &str) -> bool {
    matches!(serde_json::from_str::<Value>(s), Ok(Value::Object(_)))
}

/// Provider-specific checks layered on top of the shared structural rules.
fn check_provider_rules(
    entries: &[Entry],
    config: &ModelConfig,
    errors: &mut Vec<ValidationError>,
) {
    match &config.kind {
        ModelKind::AnthropicCompatible { .. } => {
            // Unsigned thinking cannot be resent to Anthropic (signature required).
            for (i, entry) in entries.iter().enumerate() {
                let thinking = match entry {
                    Entry::ResponseNormal { thinking, .. }
                    | Entry::ResponseToolCall { thinking, .. } => thinking,
                    _ => continue,
                };
                if matches!(thinking, Some(Thinking::Normal { .. })) {
                    errors.push(ValidationError::UnsignedThinkingRejected { index: i });
                }
            }
        }
        ModelKind::OpenAICompatible { .. } => {
            // No additional constraints yet; both signed and unsigned thinking
            // serialize fine as reasoning items.
        }
    }
}

/// Validates a raw history against `config` for serializability into the
/// configured provider's wire format.
///
/// Enforced rules:
/// - at most one [`Entry::SystemPrompt`], at index 0
/// - strict alternation `User -> Assistant -> Results* -> Assistant ...`, starting with user
/// - non-empty user messages (text or image) and non-empty assistant text
/// - every `toolcall_params` parses as a JSON object
/// - every result references a call from the immediately preceding batch
/// - no duplicate tool_use_ids, no unanswered calls at the end of history
///
/// Provider-specific (`config.kind`):
/// - `AnthropicCompatible`: rejects unsigned [`Thinking::Normal`] blocks
/// - `OpenAICompatible`: no extra rules yet
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
            Entry::UserMessage { content } => {
                let has_text = content.text.as_ref().is_some_and(|t| !t.is_empty());
                if !has_text && content.image.is_none() {
                    errors.push(ValidationError::EmptyUserMessage { index: i });
                }
            }
            Entry::ResponseNormal { content, .. } => {
                if content.is_empty() {
                    errors.push(ValidationError::EmptyAssistantContent { index: i });
                }
            }
            Entry::ResponseToolCall { tool_call, .. } => {
                pending.clear();
                for (ci, tc) in tool_call.iter().enumerate() {
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
                for (ri, r) in tool_call.iter().enumerate() {
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
    use llm_history_model::history::{Image, ImageMIME, MessageContent, ToolCall, ToolCallResult};

    fn openai_config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::OpenAICompatible {
                model_name: "gpt-test".into(), streaming: false,
            },
        }
    }

    fn anthropic_config() -> ModelConfig {
        ModelConfig {
            kind: ModelKind::AnthropicCompatible {
                model_name: "claude-test".into(),
            },
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
        Entry::ResponseNormal {
            thinking: None,
            content: t.into(),
        }
    }

    fn tool_call(id: &str) -> Entry {
        Entry::ResponseToolCall {
            thinking: None,
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
    fn accepts_parallel_calls_split_batches() {
        let h = vec![
            text_user("q"),
            Entry::ResponseToolCall {
                thinking: Some(Thinking::Secret {
                    thinking: "hmm".into(),
                    signature: "sig".into(),
                }),
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
            Entry::ResponseNormal {
                thinking: Some(Thinking::Secret {
                    thinking: "t".into(),
                    signature: "sig".into(),
                }),
                content: "done".into(),
            },
        ];
        assert_eq!(validate_entries(&h, &openai_config()), Ok(()));
        assert_eq!(validate_entries(&h, &anthropic_config()), Ok(()));
    }

    #[test]
    fn anthropic_rejects_unsigned_thinking_but_openai_accepts() {
        let h = vec![
            text_user("q"),
            Entry::ResponseNormal {
                thinking: Some(Thinking::Normal {
                    thinking: "hmm".into(),
                }),
                content: "answer".into(),
            },
        ];
        assert_eq!(validate_entries(&h, &openai_config()), Ok(()));
        assert_eq!(
            validate_entries(&h, &anthropic_config()),
            Err(vec![ValidationError::UnsignedThinkingRejected { index: 1 }])
        );
    }

    #[test]
    fn anthropic_accepts_signed_thinking() {
        let h = vec![
            text_user("q"),
            Entry::ResponseNormal {
                thinking: Some(Thinking::Secret {
                    thinking: "hmm".into(),
                    signature: "sig==".into(),
                }),
                content: "answer".into(),
            },
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
                thinking: None,
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
    fn collects_multiple_errors() {
        let h = vec![tool_call("x"), tool_result("y")];
        assert!(matches!(validate_entries(&h, &openai_config()), Err(e) if e.len() >= 3));
    }
}
