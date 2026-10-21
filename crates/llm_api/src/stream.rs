//! Provider-neutral streaming support for [`Entry::Draft`].
//!
//! A draft lives in the history while a response is being received. Provider
//! stream decoders mutate its [`DraftBlock`]s in place; once the stream is
//! complete, [`finish_draft`] converts the draft into finalized history entries.
//!
//! Drafts are live state only: [`crate::validation::validate_entries`] rejects
//! them, so a draft can never leak into a provider request.

use crate::model_config::ModelKind;
use llm_history_model::history::{DraftBlock, Entry, ResponseBlock, Thinking, ToolCall};
use std::fmt;

/// A provider-neutral notification about stream progress.
///
/// Decoders mutate the draft directly; these events are for logging, UI
/// updates, or cancellation, not for reconstructing state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    ThinkingDelta(String),
    SignatureDelta(String),
    TextDelta(String),
    ToolCallDelta {
        index: u32,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
    /// The provider finished the response.
    Completed {
        stop_reason: Option<String>,
    },
}

/// Errors produced while decoding a stream or committing its draft.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    /// A provider SSE payload was not valid JSON for its event type.
    InvalidJson {
        event: Option<String>,
        details: String,
    },
    /// The provider reported an error event.
    Provider { message: String },
    /// A finalized tool call is missing its id.
    MissingToolCallId { index: u32 },
    /// A finalized tool call is missing its function name.
    MissingToolCallName { index: u32 },
    /// `finish_draft` was called on an entry that is not a draft.
    NotADraft,
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::InvalidJson { event, details } => match event {
                Some(event) => write!(f, "invalid `{event}` SSE payload: {details}"),
                None => write!(f, "invalid SSE payload: {details}"),
            },
            StreamError::Provider { message } => write!(f, "provider stream error: {message}"),
            StreamError::MissingToolCallId { index } => {
                write!(f, "tool call block {index} finished without an id")
            }
            StreamError::MissingToolCallName { index } => {
                write!(f, "tool call block {index} finished without a name")
            }
            StreamError::NotADraft => write!(f, "entry is not a streaming draft"),
        }
    }
}

impl std::error::Error for StreamError {}

/// Converts a finished draft into replayable history entries.
///
/// Mirrors the mapping used by the non-streaming response parsers:
/// - blocks keep their order
/// - thinking with OpenAI `encrypted_content` becomes
///   [`Thinking::Encrypted`]; otherwise thinking with a signature becomes
///   [`Thinking::Signed`], unsigned thinking becomes [`Thinking::Normal`]
///   (Anthropic validation rejects the latter, as it does for non-streamed
///   histories)
/// - tool calls merge into one [`Entry::ResponseToolCall`] with adjacent text as
///   its `content`
/// - otherwise the ordered thinking/text blocks become [`Entry::Response`]
///   (preserving interleaving)
/// - empty drafts yield no entries
pub fn finish_draft(entry: Entry) -> Result<Vec<Entry>, StreamError> {
    finish_draft_unfiltered(entry)
}

/// Provider-aware variant of [`finish_draft`] for committing a stream.
///
/// Anthropic needs a signature to replay a thinking block. Some compatible
/// endpoints stream thinking without ever sending `signature_delta`; that
/// thinking is dropped here so the committed history can be sent again. Other
/// providers keep unsigned thinking as normal reasoning.
pub fn finish_draft_for(entry: Entry, kind: &ModelKind) -> Result<Vec<Entry>, StreamError> {
    let mut entries = finish_draft_unfiltered(entry)?;
    if let ModelKind::AnthropicCompatible {
        allow_unsigned_thinking,
        ..
    } = kind
    {
        let replayable = |t: &Thinking| {
            (matches!(t, Thinking::Signed { .. } | Thinking::Redacted { .. })
                || (*allow_unsigned_thinking && matches!(t, Thinking::Normal { .. })))
                && !crate::wire_common::thinking_is_empty(t)
        };
        for entry in &mut entries {
            match entry {
                Entry::Response { blocks } => {
                    blocks.retain(|block| match block {
                        ResponseBlock::Thinking(t) => replayable(t),
                        ResponseBlock::Text(_) => true,
                    });
                }
                Entry::ResponseToolCall { thinking, .. } => {
                    thinking.retain(replayable);
                }
                _ => {}
            }
        }
        // A response left with no replayable content is empty.
        entries.retain(|entry| match entry {
            Entry::Response { blocks } => {
                !blocks.is_empty() && blocks.iter().any(|block| !block.is_empty())
            }
            _ => true,
        });
    }
    Ok(entries)
}

fn finish_draft_unfiltered(entry: Entry) -> Result<Vec<Entry>, StreamError> {
    let Entry::Draft { blocks } = entry else {
        return Err(StreamError::NotADraft);
    };

    let mut response: Vec<ResponseBlock> = Vec::new();
    let mut calls: Vec<ToolCall> = Vec::new();

    for block in blocks {
        match block {
            DraftBlock::Thinking {
                text,
                signature,
                encrypted_content,
            } => {
                let thinking = if let Some(data) = encrypted_content.filter(|data| !data.is_empty())
                {
                    // The opaque payload is the authoritative replay form.
                    Some(Thinking::Encrypted { thinking: data })
                } else if let Some(signature) = signature {
                    Some(Thinking::Signed {
                        thinking: text,
                        signature,
                    })
                } else if !text.is_empty() {
                    Some(Thinking::Normal { thinking: text })
                } else {
                    None
                };
                if let Some(thinking) = thinking {
                    response.push(ResponseBlock::Thinking(thinking));
                }
            }
            DraftBlock::RedactedThinking { data } => {
                if !data.is_empty() {
                    response.push(ResponseBlock::Thinking(Thinking::Redacted {
                        thinking: data,
                    }));
                }
            }
            DraftBlock::Text { text } => {
                if !text.is_empty() {
                    response.push(ResponseBlock::Text(text));
                }
            }
            DraftBlock::ToolCall {
                index,
                tool_use_id,
                tool_name,
                arguments,
            } => {
                let tool_use_id = tool_use_id.ok_or(StreamError::MissingToolCallId { index })?;
                let tool_name = tool_name.ok_or(StreamError::MissingToolCallName { index })?;
                let toolcall_params = if arguments.trim().is_empty() {
                    "{}".to_owned()
                } else {
                    arguments
                };
                calls.push(ToolCall {
                    tool_name,
                    toolcall_params,
                    tool_use_id,
                });
            }
        }
    }

    if !calls.is_empty() {
        let thinking = response
            .iter()
            .filter_map(ResponseBlock::thinking)
            .cloned()
            .collect();
        let content = {
            let joined = response
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
        return Ok(vec![Entry::ResponseToolCall {
            thinking,
            content,
            tool_call: calls,
        }]);
    }

    if response.is_empty() {
        Ok(Vec::new())
    } else {
        Ok(vec![Entry::Response { blocks: response }])
    }
}

/// Finds the position of an existing tool-call block by provider index.
pub(crate) fn tool_call_position(blocks: &[DraftBlock], index: u32) -> Option<usize> {
    blocks.iter().position(|block| {
        matches!(
            block,
            DraftBlock::ToolCall {
                index: block_index,
                ..
            } if *block_index == index
        )
    })
}

/// Appends text to the last/open block of a given shape, or starts a new one.
pub(crate) fn append_text(
    blocks: &mut Vec<DraftBlock>,
    position: &mut Option<usize>,
    text: &str,
    thinking: bool,
) {
    let pos = match *position {
        Some(pos) if pos < blocks.len() => pos,
        _ => {
            blocks.push(if thinking {
                DraftBlock::Thinking {
                    text: String::new(),
                    signature: None,
                    encrypted_content: None,
                }
            } else {
                DraftBlock::Text {
                    text: String::new(),
                }
            });
            let pos = blocks.len() - 1;
            *position = Some(pos);
            pos
        }
    };

    match &mut blocks[pos] {
        DraftBlock::Thinking { text: existing, .. } if thinking => existing.push_str(text),
        DraftBlock::Text { text: existing } if !thinking => existing.push_str(text),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_block(text: &str) -> DraftBlock {
        DraftBlock::Text {
            text: text.to_owned(),
        }
    }

    #[test]
    fn text_only_draft_becomes_response_normal() {
        let entries = finish_draft(Entry::Draft {
            blocks: vec![
                DraftBlock::Thinking {
                    text: "hmm".into(),
                    signature: Some("sig".into()),
                    encrypted_content: None,
                },
                text_block("Hello"),
                text_block("world"),
            ],
        })
        .unwrap();

        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].response_joined_text().as_deref(),
            Some("Hello\nworld")
        );
        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Signed { thinking, signature }]
                if thinking == "hmm" && signature == "sig"
        ));
    }

    #[test]
    fn unsigned_thinking_becomes_normal() {
        let entries = finish_draft(Entry::Draft {
            blocks: vec![
                DraftBlock::Thinking {
                    text: "hmm".into(),
                    signature: None,
                    encrypted_content: None,
                },
                text_block("answer"),
            ],
        })
        .unwrap();

        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Normal { thinking }] if thinking == "hmm"
        ));
    }

    #[test]
    fn anthropic_commit_drops_unsigned_thinking() {
        let kind = ModelKind::AnthropicCompatible {
            model_name: "claude".into(),
            streaming: true,
            thinking: None,
            allow_unsigned_thinking: false,
        };
        let entries = finish_draft_for(
            Entry::Draft {
                blocks: vec![
                    DraftBlock::Thinking {
                        text: "hmm".into(),
                        signature: None,
                        encrypted_content: None,
                    },
                    text_block("answer"),
                ],
            },
            &kind,
        )
        .unwrap();

        assert_eq!(entries[0].response_joined_text().as_deref(), Some("answer"));
        assert!(entries[0].response_thinking_blocks().is_empty());
    }

    #[test]
    fn anthropic_commit_keeps_unsigned_thinking_when_allowed() {
        let kind = ModelKind::AnthropicCompatible {
            model_name: "llama.cpp".into(),
            streaming: true,
            thinking: None,
            allow_unsigned_thinking: true,
        };
        let entries = finish_draft_for(
            Entry::Draft {
                blocks: vec![
                    DraftBlock::Thinking {
                        text: "hmm".into(),
                        signature: None,
                        encrypted_content: None,
                    },
                    text_block("answer"),
                ],
            },
            &kind,
        )
        .unwrap();

        assert_eq!(
            entries[0].as_response_blocks().expect("expected Response"),
            [
                ResponseBlock::Thinking(Thinking::Normal {
                    thinking: "hmm".into()
                }),
                ResponseBlock::Text("answer".into()),
            ]
        );
    }

    #[test]
    fn openai_commit_keeps_unsigned_thinking() {
        let kind = ModelKind::OpenAICompatible {
            model_name: "gpt".into(),
            streaming: true,
            prompt_cache_key: String::new(),
        };
        let entries = finish_draft_for(
            Entry::Draft {
                blocks: vec![
                    DraftBlock::Thinking {
                        text: "hmm".into(),
                        signature: None,
                        encrypted_content: None,
                    },
                    text_block("answer"),
                ],
            },
            &kind,
        )
        .unwrap();

        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Normal { thinking }] if thinking == "hmm"
        ));
    }

    #[test]
    fn tool_calls_merge_and_keep_preamble_text() {
        let entries = finish_draft(Entry::Draft {
            blocks: vec![
                DraftBlock::Thinking {
                    text: "plan".into(),
                    signature: Some("s1".into()),
                    encrypted_content: None,
                },
                text_block("calling now"),
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
            ],
        })
        .unwrap();

        assert_eq!(entries.len(), 1);
        let Entry::ResponseToolCall {
            thinking,
            content,
            tool_call,
        } = &entries[0]
        else {
            panic!("expected ResponseToolCall");
        };
        assert_eq!(content.as_deref(), Some("calling now"));
        assert_eq!(tool_call.len(), 2);
        assert_eq!(tool_call[1].toolcall_params, "{}");
        assert_eq!(thinking.len(), 1);
    }

    #[test]
    fn empty_drafts_yield_nothing_and_thinking_only_drafts_are_kept() {
        assert!(finish_draft(Entry::draft()).unwrap().is_empty());

        let entries = finish_draft(Entry::Draft {
            blocks: vec![DraftBlock::Thinking {
                text: "hmm".into(),
                signature: Some("s".into()),
                encrypted_content: None,
            }],
        })
        .unwrap();
        assert_eq!(
            entries[0].as_response_blocks().expect("expected Response"),
            [ResponseBlock::Thinking(Thinking::Signed {
                thinking: "hmm".into(),
                signature: "s".into(),
            })]
        );
    }

    #[test]
    fn encrypted_thinking_prefers_opaque_payload() {
        let entries = finish_draft(Entry::Draft {
            blocks: vec![DraftBlock::Thinking {
                text: "plaintext preview".into(),
                signature: None,
                encrypted_content: Some("opaque".into()),
            }],
        })
        .unwrap();

        assert!(matches!(
            entries[0].response_thinking_blocks().as_slice(),
            [Thinking::Encrypted { thinking }] if thinking == "opaque"
        ));
    }

    #[test]
    fn tool_call_without_id_is_an_error() {
        let err = finish_draft(Entry::Draft {
            blocks: vec![DraftBlock::ToolCall {
                index: 3,
                tool_use_id: None,
                tool_name: Some("a".into()),
                arguments: "{}".into(),
            }],
        })
        .unwrap_err();
        assert_eq!(err, StreamError::MissingToolCallId { index: 3 });
    }

    #[test]
    fn appending_respects_block_kind() {
        let mut blocks = Vec::new();
        let mut pos = None;
        append_text(&mut blocks, &mut pos, "a", true);
        append_text(&mut blocks, &mut pos, "b", true);
        assert_eq!(
            blocks,
            vec![DraftBlock::Thinking {
                text: "ab".into(),
                signature: None,
                encrypted_content: None,
            }]
        );
    }
}
