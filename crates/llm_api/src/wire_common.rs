//! Helpers shared by the provider wire-format adapters.
//!
//! This module centralizes the small conversions that every serializer needs:
//! JSON Schema construction from [`Tool`] definitions, image MIME strings,
//! and plaintext extraction from [`Thinking`] blocks.

use llm_history_model::history::{ImageMIME, Thinking};
use serde_json::{Map, Value};

use crate::generic_tool::{ParameterType, Tool};

const fn parameter_type_str(t: &ParameterType) -> &'static str {
    match t {
        ParameterType::String => "string",
        ParameterType::Number => "number",
    }
}

/// Builds the `properties` + sorted `required` parts of a JSON Schema object
/// from a tool's parameter map. Returns the ready-to-embed schema value.
///
/// Every declared parameter is listed as required and unknown extra arguments
/// are rejected (`additionalProperties: false`) — matching what OpenAI's
/// `strict` mode expects.
pub fn parameters_schema(tool: &Tool) -> Value {
    // `serde_json` preserves insertion order, so iterate in a stable order.
    // `Tool.parameters` is a `HashMap`; leaving its iteration order in the
    // output would reshuffle the tool schema between requests and invalidate
    // provider prompt caches (the schema is part of the cached prefix).
    let mut names: Vec<&str> = tool.parameters.keys().map(String::as_str).collect();
    names.sort_unstable();

    let mut properties = Map::new();
    for name in &names {
        let param = &tool.parameters[*name];
        properties.insert(
            (*name).to_owned(),
            serde_json::json!({
                "type": parameter_type_str(&param.r#type),
                "description": param.description,
            }),
        );
    }

    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": names,
        "additionalProperties": false,
    })
}

/// Canonical MIME string for image blocks (`image/png`, `image/jpeg`).
pub const fn mime_str(mime: &ImageMIME) -> &'static str {
    match mime {
        ImageMIME::PNG => "png",
        ImageMIME::JPEG => "jpeg",
    }
}

/// Anthropic wire MIME spelling (`image/png`, `image/jpeg`).
pub const fn anthropic_media_type_str(mime: &ImageMIME) -> &'static str {
    match mime {
        ImageMIME::PNG => "image/png",
        ImageMIME::JPEG => "image/jpeg",
    }
}

/// Plaintext of a thinking block, when it has one.
///
/// `Thinking::Encrypted` and `Thinking::Redacted` are opaque provider payloads;
/// they must be replayed through their native wire form (`encrypted_content`
/// or `redacted_thinking`) rather than converted to visible text.
pub fn thinking_text(t: &Thinking) -> Option<&str> {
    match t {
        Thinking::Normal { thinking } | Thinking::Signed { thinking, .. } => Some(thinking),
        Thinking::Encrypted { .. } | Thinking::Redacted { .. } => None,
    }
}

/// Whether a thinking block carries no payload at all.
pub fn thinking_is_empty(t: &Thinking) -> bool {
    match t {
        Thinking::Normal { thinking }
        | Thinking::Encrypted { thinking }
        | Thinking::Redacted { thinking } => thinking.is_empty(),
        // `display: "omitted"` can return a signature without visible text;
        // such a block is still replayable and must not be treated as empty.
        Thinking::Signed {
            thinking,
            signature,
        } => thinking.is_empty() && signature.is_empty(),
    }
}

/// Joins the plaintext thinking blocks into a single string, dropping opaque
/// payloads and empty results to `None`.
pub fn thinking_joined(thinking: &[Thinking]) -> Option<String> {
    let joined = thinking
        .iter()
        .filter_map(thinking_text)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic_tool::ParameterDescription;
    use std::collections::HashMap;

    fn sample_tool() -> Tool {
        let mut parameters = HashMap::new();
        parameters.insert(
            "city".to_owned(),
            ParameterDescription {
                r#type: ParameterType::String,
                description: "City name".to_owned(),
            },
        );
        parameters.insert(
            "days".to_owned(),
            ParameterDescription {
                r#type: ParameterType::Number,
                description: "Forecast length".to_owned(),
            },
        );
        Tool {
            name: "get_weather".to_owned(),
            parameters,
            description: None,
        }
    }

    #[test]
    fn schema_lists_sorted_required_and_closes_extras() {
        let schema = parameters_schema(&sample_tool());
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["required"], serde_json::json!(["city", "days"]));
        assert_eq!(schema["properties"]["city"]["type"], "string");
        assert_eq!(schema["properties"]["days"]["type"], "number");
    }

    #[test]
    fn schema_property_order_is_stable() {
        // `Tool.parameters` is a `HashMap`; the serialized schema must not
        // inherit its randomized iteration order or prompt caches break.
        let mut parameters = HashMap::new();
        for name in ["zulu", "alpha", "mike", "bravo", "yankee", "charlie"] {
            parameters.insert(
                name.to_owned(),
                ParameterDescription {
                    r#type: ParameterType::String,
                    description: name.to_owned(),
                },
            );
        }
        let tool = Tool {
            name: "many".to_owned(),
            parameters,
            description: None,
        };

        let schema = parameters_schema(&tool);
        let keys: Vec<&str> = schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            ["alpha", "bravo", "charlie", "mike", "yankee", "zulu"]
        );
    }

    #[test]
    fn thinking_joined_only_includes_plaintext_variants() {
        let thinking = vec![
            Thinking::Normal {
                thinking: "a".into(),
            },
            Thinking::Signed {
                thinking: "b".into(),
                signature: "sig".into(),
            },
            Thinking::Encrypted {
                thinking: "opaque-openai".into(),
            },
            Thinking::Redacted {
                thinking: "opaque-anthropic".into(),
            },
        ];
        assert_eq!(thinking_joined(&thinking).as_deref(), Some("a\n\nb"));
        assert_eq!(thinking_joined(&[]), None);
        assert_eq!(
            thinking_joined(&[Thinking::Encrypted {
                thinking: "opaque".into()
            }]),
            None
        );
    }

    #[test]
    fn thinking_is_empty_checks_the_payload() {
        assert!(thinking_is_empty(&Thinking::Normal {
            thinking: String::new()
        }));
        assert!(!thinking_is_empty(&Thinking::Encrypted {
            thinking: "opaque".into()
        }));
        assert!(!thinking_is_empty(&Thinking::Redacted {
            thinking: "opaque".into()
        }));
    }
}
