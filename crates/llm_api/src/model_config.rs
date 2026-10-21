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
        /// Whether to request a server-sent event stream (`stream: true`).
        #[serde(default)]
        streaming: bool,
        /// Extended thinking configuration. `None` leaves thinking disabled.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thinking: Option<AnthropicThinking>,
        /// Accept thinking blocks that carry no signature. Compatible
        /// endpoints such as llama.cpp return `signature: ""`; the real
        /// Anthropic API requires a valid signature, so this defaults to
        /// `false` and should only be enabled for compatible servers.
        #[serde(default)]
        allow_unsigned_thinking: bool,
    },
}

/// Anthropic extended-thinking request configuration.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicThinking {
    /// Budget for internal reasoning. Must be at least 1024 and strictly less
    /// than the request's `max_tokens`.
    pub budget_tokens: u32,
    /// Whether the API should return summarized thinking or only a signature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<AnthropicThinkingDisplay>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicThinkingDisplay {
    Summarized,
    Omitted,
}

impl AnthropicThinkingDisplay {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Summarized => "summarized",
            Self::Omitted => "omitted",
        }
    }
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
            | ModelKind::AnthropicCompatible { model_name, .. } => model_name,
        }
    }
    pub fn get_streamable(&self) -> bool {
        match self {
            ModelKind::OpenAICompatible { streaming, .. }
            | ModelKind::AnthropicCompatible { streaming, .. } => *streaming,
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
