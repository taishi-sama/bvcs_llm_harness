use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum ModelKind {
    OpenAICompatible { model_name: String, streaming: bool },
    AnthropicCompatible { model_name: String },
}
impl ModelKind {
    pub fn get_model_name(&self) -> &str {
        match self {
            ModelKind::OpenAICompatible { model_name, .. }
            | ModelKind::AnthropicCompatible { model_name, .. } => model_name,
        }
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
