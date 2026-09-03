//! Conversion of provider-neutral [`Tool`] definitions into the
//! Anthropic **Messages API** `tools` array format:
//! `{"name":...,"description":...,"input_schema":<json schema>}`.

use crate::generic_tool::{ParameterType, Tool};
use serde::Serialize;
use serde_json::{Map, Value};

/// A client tool definition in Messages API wire format.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    /// The generic `Tool` carries an optional description; omitted when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema object describing the tool `input` shape.
    pub input_schema: Value,
}

const fn parameter_type_str(t: &ParameterType) -> &'static str {
    match t {
        ParameterType::String => "string",
        ParameterType::Number => "number",
    }
}

/// Builds the `input_schema` JSON Schema for one tool.
///
/// Every declared parameter is listed as required and unknown extra arguments
/// are rejected (`additionalProperties: false`).
fn input_schema(tool: &Tool) -> Value {
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

impl AnthropicTool {
    pub fn from_generic(tool: &Tool) -> Self {
        Self {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: input_schema(tool),
        }
    }
}

/// Converts the full `ModelConfig.tools` list into Messages API wire format.
pub fn tools_to_anthropic(tools: &[Tool]) -> Vec<AnthropicTool> {
    tools.iter().map(AnthropicTool::from_generic).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic_tool::ParameterDescription;
    use std::collections::HashMap;

    fn sample() -> Tool {
        let mut parameters = HashMap::new();
        parameters.insert(
            "ticker".to_owned(),
            ParameterDescription {
                r#type: ParameterType::String,
                description: "Ticker symbol".to_owned(),
            },
        );
        Tool {
            name: "get_stock_price".to_owned(),
            parameters,
            description: Some("Get the current stock price.".to_owned()),
        }
    }

    #[test]
    fn converts_tool_to_wire_format() {
        let wire = tools_to_anthropic(&[sample()]);
        assert_eq!(wire.len(), 1);
        let value = serde_json::to_value(&wire[0]).unwrap();

        assert_eq!(value["name"], "get_stock_price");
        assert_eq!(value["description"], "Get the current stock price.");
        assert!(value.get("type").is_none()); // no type discriminator on Tool
        let schema = &value["input_schema"];
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["required"], serde_json::json!(["ticker"]));
        assert_eq!(schema["properties"]["ticker"]["type"], "string");
        assert_eq!(
            schema["properties"]["ticker"]["description"],
            "Ticker symbol"
        );
    }

    #[test]
    fn omits_description_when_absent() {
        let wire = tools_to_anthropic(&[Tool {
            name: "t".into(),
            parameters: HashMap::new(),
            description: None,
        }]);
        let value = serde_json::to_value(&wire[0]).unwrap();
        assert!(value.get("description").is_none());
    }

    #[test]
    fn empty_tools_convert_to_empty_array() {
        assert!(tools_to_anthropic(&[]).is_empty());
    }
}
