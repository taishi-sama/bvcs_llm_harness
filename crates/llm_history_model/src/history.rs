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
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Thinking {
    Normal { thinking: String },
    Signed { thinking: String, signature: String },
    Redacted { thinking: String },
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Entry {
    /// Should be only one in beginning of history
    SystemPrompt {
        content: String,
    },
    UserMessage {
        content: MessageContent,
    },
    ResponseNormal {
        thinking: Vec<Thinking>,
        content: String,
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
    /// - non-empty text blocks are joined with `\n`
    /// - thinking is kept (signed when a signature arrived, otherwise
    ///   [`Thinking::Normal`]); empty blocks are skipped
    /// - when `drop_unsigned_thinking` is `true`, thinking blocks that never
    ///   received a signature are omitted instead of becoming
    ///   [`Thinking::Normal`] (useful for providers like Anthropic that reject
    ///   unsigned thinking on replay)
    /// - tool calls are kept only when they are finished: non-empty id and
    ///   name, and arguments that are empty or a complete JSON object. Partial
    ///   arguments drop the whole call rather than preserving unusable JSON.
    /// - if any finished tool call remains, the result is
    ///   [`Entry::ResponseToolCall`] with the text as `content`; otherwise
    ///   non-empty text becomes [`Entry::ResponseNormal`]
    /// - if the draft contains only thinking, `thinking_only_message` becomes
    ///   the message content, so the turn can still be appended to history
    ///
    /// Returns `None` when `self` is not a draft, when nothing usable remains,
    /// or when the draft is thinking-only and `thinking_only_message` is empty.
    ///
    /// The result is structurally valid, but provider rules still apply (for
    /// example, Anthropic cannot replay unsigned thinking).
    pub fn try_to_rescue_draft(
        &self,
        thinking_only_message: &str,
        drop_unsigned_thinking: bool,
    ) -> Option<Entry> {
        let Entry::Draft { blocks } = self else {
            return None;
        };

        let mut thinking: Vec<Thinking> = Vec::new();
        let mut texts: Vec<String> = Vec::new();
        let mut calls: Vec<ToolCall> = Vec::new();

        for block in blocks {
            match block {
                DraftBlock::Thinking { text, signature } if !text.is_empty() => {
                    match signature.as_deref().filter(|s| !s.is_empty()) {
                        Some(signature) => thinking.push(Thinking::Signed {
                            thinking: text.clone(),
                            signature: signature.to_owned(),
                        }),
                        None if !drop_unsigned_thinking => thinking.push(Thinking::Normal {
                            thinking: text.clone(),
                        }),
                        None => {}
                    }
                }
                DraftBlock::RedactedThinking { data } if !data.is_empty() => {
                    thinking.push(Thinking::Redacted {
                        thinking: data.clone(),
                    });
                }
                DraftBlock::Text { text } if !text.is_empty() => texts.push(text.clone()),
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

        let content = {
            let joined = texts.join("\n");
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        };

        if !calls.is_empty() {
            return Some(Entry::ResponseToolCall {
                thinking,
                content,
                tool_call: calls,
            });
        }

        if let Some(content) = content {
            return Some(Entry::ResponseNormal { thinking, content });
        }

        if thinking.is_empty() || thinking_only_message.is_empty() {
            return None;
        }
        Some(Entry::ResponseNormal {
            thinking,
            content: thinking_only_message.to_owned(),
        })
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

        let Entry::ResponseNormal { thinking, content } = rescued else {
            panic!("expected ResponseNormal");
        };
        assert_eq!(content, "Hello\nworld");
        assert!(matches!(
            thinking.as_slice(),
            [Thinking::Normal { thinking }] if thinking == "hmm"
        ));
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

        let Entry::ResponseNormal { thinking, .. } = rescued else {
            panic!("expected ResponseNormal");
        };
        assert!(matches!(
            thinking.as_slice(),
            [
                Thinking::Signed { thinking, signature },
                Thinking::Redacted { thinking: redacted }
            ] if thinking == "signed" && signature == "sig==" && redacted == "opaque"
        ));
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

        assert!(matches!(
            rescued,
            Entry::ResponseNormal { content, .. } if content == "partial answer"
        ));

        // Nothing else usable -> no entry.
        let rescued = Entry::Draft {
            blocks: vec![tool_call(Some("c3"), Some("c"), r#"{"city":"Os"#)],
        }
        .try_to_rescue_draft("[thinking only]", false);
        assert!(rescued.is_none());
    }

    #[test]
    fn returns_none_for_non_draft_and_empty_drafts() {
        let final_entry = Entry::ResponseNormal {
            thinking: Vec::new(),
            content: "done".into(),
        };
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
        let Entry::ResponseNormal {
            thinking: rescued_thinking,
            content,
        } = rescued
        else {
            panic!("expected ResponseNormal");
        };
        assert_eq!(content, "answer");
        assert!(rescued_thinking.is_empty());

        // Signed thinking is unaffected by the toggle.
        let draft = Entry::Draft {
            blocks: vec![thinking("hmm", Some("sig")), text("answer")],
        };
        let rescued = draft
            .try_to_rescue_draft("", true)
            .expect("text is rescuable");
        let Entry::ResponseNormal {
            thinking: rescued_thinking,
            ..
        } = rescued
        else {
            panic!("expected ResponseNormal");
        };
        assert!(matches!(
            rescued_thinking.as_slice(),
            [Thinking::Signed { thinking, signature }]
                if thinking == "hmm" && signature == "sig"
        ));

        // If unsigned thinking was the only content, dropping it leaves
        // nothing and the fallback cannot manufacture a message.
        let draft = Entry::Draft {
            blocks: vec![thinking("hmm", None)],
        };
        assert!(draft.try_to_rescue_draft("[thinking only]", true).is_none());
    }

    #[test]
    fn thinking_only_draft_uses_fallback_message() {
        let draft = Entry::Draft {
            blocks: vec![thinking("only thoughts", Some("sig"))],
        };

        let rescued = draft
            .try_to_rescue_draft("[thinking only]", false)
            .expect("fallback text creates a message");
        let Entry::ResponseNormal { thinking, content } = rescued else {
            panic!("expected ResponseNormal");
        };
        assert_eq!(content, "[thinking only]");
        assert!(matches!(
            thinking.as_slice(),
            [Thinking::Signed { thinking, signature }]
                if thinking == "only thoughts" && signature == "sig"
        ));

        // An empty fallback cannot produce valid assistant content.
        assert!(draft.try_to_rescue_draft("", false).is_none());
    }
}
