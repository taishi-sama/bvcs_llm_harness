//! Provider selection and application configuration.

use std::env;

use bevy::prelude::{App, Plugin, Resource};
use llm_api::{model_config::ModelKind, usage::UsageProvider};

/// Which server-side streaming API to exercise.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Provider {
    Anthropic,
    Completions,
    Responses,
}

impl Provider {
    pub(crate) fn from_env() -> Self {
        match env::var("PROVIDER").as_deref() {
            Ok("anthropic") => Self::Anthropic,
            Ok("completions") => Self::Completions,
            _ => Self::Responses,
        }
    }

    /// Endpoints copied from `crates/llm_connector_example`.
    pub(crate) fn endpoint(self) -> &'static str {
        match self {
            Provider::Anthropic => "http://127.0.0.1:8080/v1/messages",
            Provider::Completions => "http://127.0.0.1:8080/v1/chat/completions",
            Provider::Responses => "http://127.0.0.1:8080/v1/responses",
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::Completions => "completions",
            Provider::Responses => "responses",
        }
    }

    pub(crate) fn kind(self, model_name: &str) -> ModelKind {
        match self {
            Provider::Anthropic => ModelKind::AnthropicCompatible {
                model_name: model_name.into(),
                streaming: true,
                thinking: None,
                // llama.cpp returns thinking blocks with `signature: ""`.
                allow_unsigned_thinking: true,
                system_message_policy: Default::default(),
            },
            Provider::Completions | Provider::Responses => ModelKind::OpenAICompatible {
                model_name: model_name.into(),
                streaming: true,
                prompt_cache_key: String::new(),
                provider: UsageProvider::Other,
                include_usage: true,
            },
        }
    }
}

/// Global configuration shared by newly created agents.
#[derive(Resource)]
pub(crate) struct AppConfig {
    pub(crate) provider: Provider,
    pub(crate) model_name: String,
    pub(crate) endpoint: String,
    pub(crate) token: String,
    /// Stable `x-opencode-session` value used for request routing.
    pub(crate) session_id: String,
}

impl AppConfig {
    /// Build the global configuration from the process environment.
    pub(crate) fn from_env() -> Self {
        let provider = Provider::from_env();
        let model_name = env::var("MODEL").unwrap_or_else(|_| "qwen3.6-35b-a3b".to_owned());
        let endpoint = env::var("ENDPOINT").unwrap_or_else(|_| provider.endpoint().to_owned());
        let token = env::var("API_TOKEN").unwrap_or_default();
        Self {
            provider,
            model_name,
            endpoint,
            token,
            session_id: session_id_from_env(),
        }
    }

    /// Returns the token for an `Authorization: Bearer <token>` header.
    pub(crate) fn bearer_token(&self) -> Option<&str> {
        let token = self.token.trim();
        (!token.is_empty()).then_some(token)
    }
}

/// Installs the environment-derived [`AppConfig`].
pub(crate) struct ConfigPlugin;

impl Plugin for ConfigPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AppConfig::from_env());
    }
}

/// Stable identifier for OpenCode Go request routing. `SESSION_ID` overrides
/// it; otherwise one id is derived once per process so prompt caching stays
/// effective for the whole session.
pub(crate) fn session_id_from_env() -> String {
    env::var("SESSION_ID").unwrap_or_else(|_| llm_api::model_config::generate_session_id())
}
