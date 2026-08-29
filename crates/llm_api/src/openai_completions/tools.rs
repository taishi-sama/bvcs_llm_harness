//! Conversion of provider-neutral [`Tool`] definitions into the legacy
//! **Chat Completions API** `tools` array format. Unlike the Responses API,
//! the Chat Completions layout nests the function definition under a
//! `"function"` key:
//! `{"type":"function","function":{"name":...,"parameters":<json schema>}}`.

use crate::generic_tool::{ParameterType, Tool};
use serde::Serialize;
use serde_json::{Map, Value};

/// A function tool in Chat Completions wire format.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionTool {
    /// Always `"function"`
    pub r#type: &'static str,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    /// The generic `Tool` carries an optional description; omitted when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema object describing the arguments.
    #[serde(skip_serializing_if = "Value::is_null")]
    pub parameters: Value,
}

const fn parameter_type_str(t: &ParameterType) -> &'static str {
    match t {
        ParameterType::String => "string",
        ParameterType::Number => "number",
    }
}

/// Builds the `parameters` JSON Schema for one tool.
///
/// Every declared parameter is listed as required and unknown extra arguments
/// are rejected (`additionalProperties: false`) — matching what OpenAI's
/// `strict` mode expects. A tool with no parameters yields `Value::Null` so it
/// is omitted (the spec treats that as an empty parameter list).
fn parameters_schema(tool: &Tool) -> Value {
    if tool.parameters.is_empty() {
        return Value::Null;
    }
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

impl CompletionTool {
    pub fn from_generic(tool: &Tool) -> Self {
        Self {
            r#type: "function",
            function: FunctionDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: parameters_schema(tool),
            },
        }
    }
}

/// Converts the full `ModelConfig.tools` list into Chat Completions wire format.
pub fn tools_to_completions(tools: &[Tool]) -> Vec<CompletionTool> {
    tools.iter().map(CompletionTool::from_generic).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic_tool::ParameterDescription;
    use std::collections::HashMap;

    fn sample() -> Tool {
        let mut parameters = HashMap::new();
        parameters.insert(
            "city".to_owned(),
            ParameterDescription {
                r#type: ParameterType::String,
                description: "City name".to_owned(),
            },
        );
        Tool {
            name: "get_weather".to_owned(),
            parameters,
            description: Some("Get the weather".to_owned()),
        }
    }

    #[test]
    fn converts_tool_to_nested_wire_format() {
        let wire = tools_to_completions(&[sample()]);
        let value = serde_json::to_value(&wire[0]).unwrap();

        assert_eq!(value["type"], "function");
        assert_eq!(value["function"]["name"], "get_weather");
        assert_eq!(value["function"]["description"], "Get the weather");
        let params = &value["function"]["parameters"];
        assert_eq!(params["type"], "object");
        assert_eq!(params["additionalProperties"], false);
        assert_eq!(params["required"], serde_json::json!(["city"]));
        assert_eq!(params["properties"]["city"]["type"], "string");
    }

    #[test]
    fn omits_description_when_absent() {
        let wire = tools_to_completions(&[Tool {
            name: "t".into(),
            parameters: HashMap::new(),
            description: None,
        }]);
        let value = serde_json::to_value(&wire[0]).unwrap();
        assert!(value["function"].get("description").is_none());
        assert!(value["function"].get("parameters").is_none());
    }

    #[test]
    fn empty_tools_convert_to_empty_array() {
        assert!(tools_to_completions(&[]).is_empty());
    }
}
