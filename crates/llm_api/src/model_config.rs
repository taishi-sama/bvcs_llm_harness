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
    /// Stable provider name, used in error reporting.
    pub const fn name(&self) -> &'static str {
        match self {
            ModelKind::OpenAICompatible { .. } => "openai_compatible",
            ModelKind::AnthropicCompatible { .. } => "anthropic_compatible",
        }
    }
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
    pub max_tokens: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GlobalConfig {
    pub model_config: ModelConfig,
    pub token: String,
    pub endpoint: String,
}
