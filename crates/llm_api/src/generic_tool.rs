use std::collections::HashMap;

pub enum ParameterType {
    String,
    Number,
}

pub struct ParameterDescription {
    pub r#type: ParameterType,
    pub description: String,
}

pub struct Tool {
    pub name: String,
    pub parameters: HashMap<String, ParameterDescription>
}