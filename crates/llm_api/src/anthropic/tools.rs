//! Conversion of provider-neutral [`Tool`] definitions into the
//! Anthropic **Messages API** `tools` array format:
//! `{"name":...,"description":...,"input_schema":<json schema>}`.

use crate::generic_tool::Tool;
use crate::wire_common::parameters_schema;
use serde::Serialize;
use serde_json::Value;

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

impl AnthropicTool {
    pub fn from_generic(tool: &Tool) -> Self {
        Self {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: parameters_schema(tool),
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
    use crate::generic_tool::ParameterType;
    use crate::test_support::tool;

    fn sample() -> Tool {
        tool(
            "get_stock_price",
            Some("Get the current stock price."),
            &[("ticker", ParameterType::String, "Ticker symbol", true)],
        )
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
        let wire = tools_to_anthropic(&[tool("t", None, &[])]);
        let value = serde_json::to_value(&wire[0]).unwrap();
        assert!(value.get("description").is_none());
    }

    #[test]
    fn empty_tools_convert_to_empty_array() {
        assert!(tools_to_anthropic(&[]).is_empty());
    }
}
