//! Conversion of provider-neutral [`Tool`] definitions into the
//! OpenAI-compatible **Responses API** `tools` array format
//! (`{"type":"function","name":...,"parameters":<json schema>}`).

use crate::generic_tool::Tool;
use crate::wire_common::parameters_schema;
use serde::Serialize;
use serde_json::Value;

/// A function tool in Responses API wire format (flat layout — unlike Chat
/// Completions, there is no nested `"function"` object).
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiTool {
    /// Always `"function"`
    pub r#type: &'static str,
    #[serde(flatten)]
    pub function: OpenAiFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiFunction {
    pub name: String,
    /// The generic `Tool` carries no top-level description; omitted when empty.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// JSON Schema object describing the arguments.
    pub parameters: Value,
}

impl OpenAiTool {
    pub fn from_generic(tool: &Tool) -> Self {
        Self {
            r#type: "function",
            function: OpenAiFunction {
                name: tool.name.clone(),
                description: tool.description.clone().unwrap_or_default(),
                parameters: parameters_schema(tool),
            },
        }
    }
}

/// Converts the full `ModelConfig.tools` list into Responses API wire format.
pub fn tools_to_openai(tools: &[Tool]) -> Vec<OpenAiTool> {
    tools.iter().map(OpenAiTool::from_generic).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic_tool::{ParameterDescription, ParameterType};
    use std::collections::HashMap;

    fn sample() -> Tool {
        let mut parameters = HashMap::new();
        parameters.insert(
            "city".to_owned(),
            ParameterDescription {
                r#type: ParameterType::String,
                description: "City name".to_owned(),
                required: true,
            },
        );
        parameters.insert(
            "days".to_owned(),
            ParameterDescription {
                r#type: ParameterType::Number,
                description: "Forecast length".to_owned(),
                required: true,
            },
        );
        Tool {
            name: "get_weather".to_owned(),
            parameters,
            description: None,
        }
    }

    #[test]
    fn converts_tool_to_wire_format() {
        let wire = tools_to_openai(&[sample()]);
        assert_eq!(wire.len(), 1);
        let value = serde_json::to_value(&wire[0]).unwrap();

        assert_eq!(value["type"], "function");
        assert_eq!(value["name"], "get_weather");
        assert!(value.get("description").is_none()); // omitted when empty

        let params = &value["parameters"];
        assert_eq!(params["type"], "object");
        assert_eq!(params["additionalProperties"], false);
        assert_eq!(
            params["required"],
            serde_json::json!(["city", "days"]) // sorted, all required
        );
        assert_eq!(params["properties"]["city"]["type"], "string");
        assert_eq!(params["properties"]["city"]["description"], "City name");
        assert_eq!(params["properties"]["days"]["type"], "number");
    }

    #[test]
    fn empty_tools_convert_to_empty_array() {
        assert!(tools_to_openai(&[]).is_empty());
    }
}
