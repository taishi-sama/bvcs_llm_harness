use serde::{Deserialize, Serialize};


#[derive(Debug, Serialize, Deserialize)]
pub enum ModelKind {
    OpenAICompatible {
        model_name: String
    },
    AnthropicCompatible {
        model_name: String
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub kind: ModelKind,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GlobalConfig {
    pub model_config: ModelConfig,
    pub token: String,
    pub endpoint: String,
}