use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct History {
    pub entries: Vec<Entry>,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    pub tool_name: String,
    pub toolcall_params: String,
    pub tool_use_id: String,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCallResult {
    pub result: String,
    pub is_error: bool,
    pub tool_use_id: String,
}
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum Thinking {
    Normal {
        thinking: String,
    },
    /// OpenAI-style encrypted thinking
    Encrypted {
        thinking: String,
    },
    /// Anthropic-style signed thinking
    Signed {
        thinking: String,
        signature: String,
    },
    /// Anthropic-style redacted thinking
    Redacted {
        thinking: String,
    },
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ImageMIME {
    PNG,
    JPEG,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Image {
    pub mime: ImageMIME,
    pub base64: String,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MessageContent {
    pub text: Option<String>,
    pub image: Option<Image>, //Support passing as Base64-encoded image for now
}

/// Pending assistant turn that is filled by SSE stream.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum DraftBlock {
    Thinking {
        text: String,
        /// Anthropic sends the signature as a separate delta after the
        /// thinking text. `None` means it has not arrived yet.
        signature: Option<String>,
        /// OpenAI sends the complete reasoning item as an opaque
        /// `encrypted_content` blob when stateless replay is requested.
        /// `None` for providers that only expose plaintext thinking.
        encrypted_content: Option<String>,
    },
    RedactedThinking {
        /// Opaque provider payload (Anthropic `redacted_thinking.data`).
        data: String,
    },
    Text {
        text: String,
    },
    ToolCall {
        /// Provider block index (Anthropic `index`, Responses `output_index`,
        /// completions `tool_calls[].index`). Used to merge argument deltas.
        index: u32,
        /// `None` until the provider sends it.
        tool_use_id: Option<String>,
        tool_name: Option<String>,
        /// Partial JSON. Not necessarily valid until the block closes.
        arguments: String,
    },
}

/// One ordered block of an assistant response.
///
/// Keeping thinking and visible text in the same vector preserves the wire
/// order (for example `[thinking, text, thinking, text]`) instead of forcing
/// all thinking to the front.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum ResponseBlock {
    Thinking(Thinking),
    Text(String),
}

impl ResponseBlock {
    pub fn thinking(&self) -> Option<&Thinking> {
        match self {
            Self::Thinking(thinking) => Some(thinking),
            Self::Text(_) => None,
        }
    }

    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Thinking(_) => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Thinking(thinking) => match thinking {
                Thinking::Normal { thinking }
                | Thinking::Encrypted { thinking }
                | Thinking::Redacted { thinking } => thinking.is_empty(),
                Thinking::Signed {
                    thinking,
                    signature,
                } => thinking.is_empty() && signature.is_empty(),
            },
            Self::Text(text) => text.is_empty(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Entry {
    /// Should be only one in beginning of history
    SystemPrompt {
        content: String,
    },
    UserMessage {
        content: MessageContent,
    },
    /// A completed assistant response, stored as ordered thinking/text blocks.
    ///
    /// A response may be thinking-only, text-only, or interleaved. The vector
    /// must contain at least one non-empty block.
    Response {
        blocks: Vec<ResponseBlock>,
    },
    ResponseToolCall {
        thinking: Vec<Thinking>,
        content: Option<String>,
        tool_call: Vec<ToolCall>,
    },
    ToolCallResult {
        tool_call: Vec<ToolCallResult>,
    },
    /// Unfinished history block that is still in progress, should be trimmed to be sendable again.
    /// Should be either finished by `llm_api::stream::finish_draft` or `crate::history::Entry::try_to_resque_draft`
    Draft {
        blocks: Vec<DraftBlock>,
    },
}

impl Entry {
    /// A fresh, empty streaming draft.
    pub fn draft() -> Self {
        Entry::Draft { blocks: Vec::new() }
    }

    /// An ordered assistant response.
    pub fn response(blocks: Vec<ResponseBlock>) -> Self {
        Entry::Response { blocks }
    }

    /// A text-only assistant response.
    pub fn response_text(text: impl Into<String>) -> Self {
        Entry::Response {
            blocks: vec![ResponseBlock::Text(text.into())],
        }
    }

    /// A thinking-only assistant response.
    pub fn response_thinking(thinking: Vec<Thinking>) -> Self {
        Entry::Response {
            blocks: thinking.into_iter().map(ResponseBlock::Thinking).collect(),
        }
    }

    /// The ordered response blocks, if this is an [`Entry::Response`].
    pub fn as_response_blocks(&self) -> Option<&[ResponseBlock]> {
        match self {
            Entry::Response { blocks } => Some(blocks),
            _ => None,
        }
    }

    /// Thinking blocks of an [`Entry::Response`], in order.
    pub fn response_thinking_blocks(&self) -> Vec<&Thinking> {
        self.as_response_blocks()
            .map(|blocks| blocks.iter().filter_map(ResponseBlock::thinking).collect())
            .unwrap_or_default()
    }

    /// Non-empty text blocks of an [`Entry::Response`], in order.
    pub fn response_text_blocks(&self) -> Vec<&str> {
        self.as_response_blocks()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(ResponseBlock::text)
                    .filter(|text| !text.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Visible text of an [`Entry::Response`], joining text blocks with `\n`.
    pub fn response_joined_text(&self) -> Option<String> {
        let joined = self.response_text_blocks().join("\n");
        (!joined.is_empty()).then_some(joined)
    }

    pub fn is_draft(&self) -> bool {
        matches!(self, Entry::Draft { .. })
    }

    /// Mutable access to the draft blocks, if this entry is a draft.
    pub fn as_draft_mut(&mut self) -> Option<&mut Vec<DraftBlock>> {
        match self {
            Entry::Draft { blocks } => Some(blocks),
            _ => None,
        }
    }

    /// Best-effort conversion of an unfinished draft into one finalized
    /// assistant entry, so an interrupted stream does not lose its output.
    ///
    /// Unlike the strict commit path, this drops unusable pieces instead of
    /// failing:
    ///
    /// - thinking and text are kept in their original block order as
    ///   [`ResponseBlock`]s
    /// - OpenAI encrypted thinking is kept as [`Thinking::Encrypted`]
    /// - otherwise thinking is kept (signed when a signature arrived,
    ///   [`Thinking::Normal`] when unsigned); empty blocks are skipped
    /// - when `drop_unsigned_thinking` is `true`, thinking blocks that never
    ///   received a signature are omitted instead of becoming
    ///   [`Thinking::Normal`] (useful for providers like Anthropic that reject
    ///   unsigned thinking on replay); OpenAI [`Thinking::Encrypted`] payloads
    ///   are omitted as well because they are not replayable to Anthropic
    /// - tool calls are kept only when they are finished: non-empty id and
    ///   name, and arguments that are empty or a complete JSON object. Partial
    ///   arguments drop the whole call rather than preserving unusable JSON.
    /// - if any finished tool call remains, the result is
    ///   [`Entry::ResponseToolCall`] with the ordered thinking/text flattened
    ///   into its fields; otherwise the ordered blocks become
    ///   [`Entry::Response`]
    ///
    /// The `thinking_only_message` parameter is retained for API
    /// compatibility and is no longer used.
    ///
    /// Returns `None` when `self` is not a draft or when nothing usable
    /// remains.
    ///
    /// The result is structurally valid, but provider rules still apply (for
    /// example, Anthropic cannot replay unsigned thinking).
    pub fn try_to_rescue_draft(
        &self,
        _thinking_only_message: &str,
        drop_unsigned_thinking: bool,
    ) -> Option<Entry> {
        let Entry::Draft { blocks } = self else {
            return None;
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
                    let thinking = if let Some(data) =
                        encrypted_content.as_deref().filter(|s| !s.is_empty())
                    {
                        // OpenAI encrypted reasoning is opaque; it supersedes
                        // the plaintext copy for replay. When the caller asked
                        // to drop non-Anthropic thinking, it is omitted too.
                        (!drop_unsigned_thinking).then(|| Thinking::Encrypted {
                            thinking: data.to_owned(),
                        })
                    } else if let Some(signature) = signature.as_deref().filter(|s| !s.is_empty()) {
                        // Keep signature-only blocks (`display: "omitted"`).
                        Some(Thinking::Signed {
                            thinking: text.clone(),
                            signature: signature.to_owned(),
                        })
                    } else if !text.is_empty() && !drop_unsigned_thinking {
                        Some(Thinking::Normal {
                            thinking: text.clone(),
                        })
                    } else {
                        None
                    };
                    if let Some(thinking) = thinking {
                        response.push(ResponseBlock::Thinking(thinking));
                    }
                }
                DraftBlock::RedactedThinking { data } if !data.is_empty() => {
                    response.push(ResponseBlock::Thinking(Thinking::Redacted {
                        thinking: data.clone(),
                    }));
                }
                DraftBlock::Text { text } if !text.is_empty() => {
                    response.push(ResponseBlock::Text(text.clone()));
                }
                DraftBlock::ToolCall {
                    tool_use_id: Some(tool_use_id),
                    tool_name: Some(tool_name),
                    arguments,
                    ..
                } if !tool_use_id.trim().is_empty()
                    && !tool_name.trim().is_empty()
                    && is_finished_arguments(arguments) =>
                {
                    calls.push(ToolCall {
                        tool_name: tool_name.clone(),
                        toolcall_params: if arguments.trim().is_empty() {
                            "{}".to_owned()
                        } else {
                            arguments.clone()
                        },
                        tool_use_id: tool_use_id.clone(),
                    });
                }
                // Unfinished tool calls (missing id/name or partial arguments)
                // cannot be answered or serialized; the rest of the draft is
                // still salvageable.
                _ => {}
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
            return Some(Entry::ResponseToolCall {
                thinking,
                content,
                tool_call: calls,
            });
        }

        if response.is_empty() {
            None
        } else {
            Some(Entry::Response { blocks: response })
        }
    }
}

/// Tool arguments are finished when the provider sent none (empty) or they
/// already parse as a complete JSON object.
fn is_finished_arguments(arguments: &str) -> bool {
    let arguments = arguments.trim();
    arguments.is_empty()
        || matches!(
            serde_json::from_str::<serde_json::Value>(arguments),
            Ok(serde_json::Value::Object(_))
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(text: &str) -> DraftBlock {
        DraftBlock::Text { text: text.into() }
    }

    fn thinking(text: &str, signature: Option<&str>) -> DraftBlock {
        DraftBlock::Thinking {
            text: text.into(),
            signature: signature.map(str::to_owned),
            encrypted_content: None,
        }
    }

    fn encrypted_thinking(data: &str) -> DraftBlock {
        DraftBlock::Thinking {
            text: String::new(),
            signature: None,
            encrypted_content: Some(data.into()),
        }
    }

    fn tool_call(id: Option<&str>, name: Option<&str>, arguments: &str) -> DraftBlock {
        DraftBlock::ToolCall {
            index: 0,
            tool_use_id: id.map(str::to_owned),
            tool_name: name.map(str::to_owned),
            arguments: arguments.into(),
        }
    }

    #[test]
    fn rescues_text_with_unsigned_thinking() {
        let rescued = Entry::Draft {
            blocks: vec![thinking("hmm", None), text("Hello"), text("world")],
        }
        .try_to_rescue_draft("[thinking only]", false)
        .expect("text is rescuable");

        assert_eq!(
            rescued.as_response_blocks().expect("expected Response"),
            [
                ResponseBlock::Thinking(Thinking::Normal {
                    thinking: "hmm".into()
                }),
                ResponseBlock::Text("Hello".into()),
                ResponseBlock::Text("world".into()),
            ]
        );
    }

    #[test]
    fn preserves_signed_and_redacted_thinking() {
        let rescued = Entry::Draft {
            blocks: vec![
                thinking("signed", Some("sig==")),
                DraftBlock::RedactedThinking {
                    data: "opaque".into(),
                },
                text("answer"),
            ],
        }
        .try_to_rescue_draft("[thinking only]", false)
        .expect("text is rescuable");

        assert_eq!(
            rescued.as_response_blocks().expect("expected Response"),
            [
                ResponseBlock::Thinking(Thinking::Signed {
                    thinking: "signed".into(),
                    signature: "sig==".into(),
                }),
                ResponseBlock::Thinking(Thinking::Redacted {
                    thinking: "opaque".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]
        );
    }

    #[test]
    fn rescues_complete_tool_calls_with_preamble_text() {
        let rescued = Entry::Draft {
            blocks: vec![
                thinking("plan", Some("s1")),
                text("calling now"),
                tool_call(Some("c1"), Some("get_weather"), r#"{"city":"Oslo"}"#),
                tool_call(Some("c2"), Some("get_time"), ""),
            ],
        }
        .try_to_rescue_draft("[thinking only]", false)
        .expect("tool calls are rescuable");

        let Entry::ResponseToolCall {
            thinking,
            content,
            tool_call,
        } = rescued
        else {
            panic!("expected ResponseToolCall");
        };
        assert_eq!(content.as_deref(), Some("calling now"));
        assert_eq!(tool_call.len(), 2);
        assert_eq!(tool_call[1].toolcall_params, "{}");
        assert_eq!(thinking.len(), 1);
    }

    #[test]
    fn drops_incomplete_and_partial_tool_calls() {
        // Missing id, missing name and partial arguments are all dropped; the
        // text is still rescued.
        let rescued = Entry::Draft {
            blocks: vec![
                text("partial answer"),
                tool_call(None, Some("a"), "{}"),
                tool_call(Some("c2"), None, "{}"),
                tool_call(Some("c3"), Some("c"), r#"{"city":"Os"#),
                tool_call(Some(""), Some("d"), "{}"),
            ],
        }
        .try_to_rescue_draft("[thinking only]", false)
        .expect("text is rescuable");

        assert_eq!(
            rescued.as_response_blocks().expect("expected Response"),
            [ResponseBlock::Text("partial answer".into())]
        );

        // Nothing else usable -> no entry.
        let rescued = Entry::Draft {
            blocks: vec![tool_call(Some("c3"), Some("c"), r#"{"city":"Os"#)],
        }
        .try_to_rescue_draft("[thinking only]", false);
        assert!(rescued.is_none());
    }

    #[test]
    fn returns_none_for_non_draft_and_empty_drafts() {
        let final_entry = Entry::response_text("done");
        assert!(
            final_entry
                .try_to_rescue_draft("[thinking only]", false)
                .is_none()
        );
        assert!(
            Entry::draft()
                .try_to_rescue_draft("[thinking only]", false)
                .is_none()
        );
    }

    #[test]
    fn can_drop_unsigned_thinking() {
        // Unsigned thinking is dropped when asked; text survives.
        let draft = Entry::Draft {
            blocks: vec![thinking("hmm", None), text("answer")],
        };
        let rescued = draft
            .try_to_rescue_draft("[thinking only]", true)
            .expect("text is rescuable");
        assert_eq!(
            rescued.as_response_blocks().expect("expected Response"),
            [ResponseBlock::Text("answer".into())]
        );

        // Signed thinking is unaffected by the toggle.
        let draft = Entry::Draft {
            blocks: vec![thinking("hmm", Some("sig")), text("answer")],
        };
        let rescued = draft
            .try_to_rescue_draft("", true)
            .expect("text is rescuable");
        assert_eq!(
            rescued.as_response_blocks().expect("expected Response"),
            [
                ResponseBlock::Thinking(Thinking::Signed {
                    thinking: "hmm".into(),
                    signature: "sig".into(),
                }),
                ResponseBlock::Text("answer".into()),
            ]
        );

        // If unsigned thinking was the only content, dropping it leaves
        // nothing and the fallback cannot manufacture a message.
        let draft = Entry::Draft {
            blocks: vec![thinking("hmm", None)],
        };
        assert!(draft.try_to_rescue_draft("[thinking only]", true).is_none());
    }

    #[test]
    fn thinking_only_draft_becomes_response_thinking_only() {
        let draft = Entry::Draft {
            blocks: vec![thinking("only thoughts", Some("sig"))],
        };

        let rescued = draft
            .try_to_rescue_draft("[thinking only]", false)
            .expect("thinking is rescuable");
        assert_eq!(
            rescued.as_response_blocks().expect("expected Response"),
            [ResponseBlock::Thinking(Thinking::Signed {
                thinking: "only thoughts".into(),
                signature: "sig".into(),
            })]
        );

        // The legacy fallback text is ignored.
        assert!(draft.try_to_rescue_draft("", false).is_some());
    }

    #[test]
    fn rescues_encrypted_thinking() {
        let rescued = Entry::Draft {
            blocks: vec![encrypted_thinking("opaque-openai-blob")],
        }
        .try_to_rescue_draft("", false)
        .expect("encrypted thinking is rescuable");

        assert_eq!(
            rescued.as_response_blocks().expect("expected Response"),
            [ResponseBlock::Thinking(Thinking::Encrypted {
                thinking: "opaque-openai-blob".into(),
            })]
        );
    }
}
