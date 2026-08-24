use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Number,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ParameterDescription {
    pub r#type: ParameterType,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub parameters: HashMap<String, ParameterDescription>,
}
