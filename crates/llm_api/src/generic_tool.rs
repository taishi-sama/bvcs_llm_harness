use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ParameterType {
    String,
    Number,
    Boolean,
}

const fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParameterDescription {
    pub r#type: ParameterType,
    pub description: String,
    /// Whether the parameter must be supplied. Defaults to `true` so older
    /// serialized tools (which omit the field) keep their previous meaning.
    #[serde(default = "default_true")]
    pub required: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: HashMap<String, ParameterDescription>,
}
