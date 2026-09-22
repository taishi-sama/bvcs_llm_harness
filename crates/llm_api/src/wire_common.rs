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
    let mut properties = Map::new();
    for (name, param) in &tool.parameters {
        properties.insert(
            name.clone(),
            serde_json::json!({
                "type": parameter_type_str(&param.r#type),
                "description": param.description,
            }),
        );
    }
    let mut required: Vec<&str> = tool.parameters.keys().map(String::as_str).collect();
    required.sort_unstable();

    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
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

/// Plaintext of a thinking block. Every provider adapter that cannot represent
/// signatures or redaction natively falls back to this.
pub fn thinking_text(t: &Thinking) -> &str {
    match t {
        Thinking::Normal { thinking } => thinking,
        Thinking::Signed { thinking, .. } => thinking,
        Thinking::Redacted { thinking } => thinking,
    }
}

/// Joins multiple thinking blocks into a single plaintext string, dropping
/// empty results to `None`.
pub fn thinking_joined(thinking: &[Thinking]) -> Option<String> {
    if thinking.is_empty() {
        return None;
    }
    let joined = thinking
        .iter()
        .map(thinking_text)
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
    fn thinking_joined_concatenates_all_variants() {
        let thinking = vec![
            Thinking::Normal { thinking: "a".into() },
            Thinking::Signed {
                thinking: "b".into(),
                signature: "sig".into(),
            },
            Thinking::Redacted { thinking: "c".into() },
        ];
        assert_eq!(thinking_joined(&thinking).as_deref(), Some("a\n\nb\n\nc"));
        assert_eq!(thinking_joined(&[]), None);
    }
}
