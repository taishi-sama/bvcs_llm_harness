use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ParameterType {
    String,
    Number,
    Boolean,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParameterDescription {
    pub r#type: ParameterType,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: HashMap<String, ParameterDescription>,
}
