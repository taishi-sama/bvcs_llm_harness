use serde::{Deserialize, Serialize};

use crate::generic_tool::Tool;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ModelKind {
    OpenAICompatible {
        model_name: String,
        streaming: bool,
        prompt_cache_key: String,
    },
    AnthropicCompatible {
        model_name: String,
    },
}
impl ModelKind {
    pub fn get_model_name(&self) -> &str {
        match self {
            ModelKind::OpenAICompatible { model_name, .. }
            | ModelKind::AnthropicCompatible { model_name } => model_name,
        }
    }
    pub fn get_streamable(&self) -> bool {
        match self {
            ModelKind::OpenAICompatible { streaming, .. } => *streaming,
            ModelKind::AnthropicCompatible { .. } => false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    pub kind: ModelKind,
    pub tools: Vec<Tool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GlobalConfig {
    pub model_config: ModelConfig,
    pub token: String,
    pub endpoint: String,
}
