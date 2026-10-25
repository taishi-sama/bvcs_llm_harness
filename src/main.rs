//! Demo LLM agent client for a local llama.cpp server.
//!
//! Select the wire API with `PROVIDER` (the endpoint URLs mirror
//! `crates/llm_connector_example`). OpenAI Responses is the default;
//! Anthropic support is best effort:
//!
//! ```text
//! cargo run                            # http://127.0.0.1:8080/v1/responses
//! PROVIDER=completions  cargo run      # http://127.0.0.1:8080/v1/chat/completions
//! PROVIDER=anthropic    cargo run      # http://127.0.0.1:8080/v1/messages (best effort)
//! ```
//!
//! `MODEL` overrides the model name sent to the server. `ENDPOINT` overrides
//! the provider URL (for example an HTTPS OpenCode Go endpoint), `API_TOKEN`
//! is sent as `Authorization: Bearer <token>`, and `SESSION_ID` is sent as
//! `x-opencode-session` for request routing:
//!
//! ```text
//! ENDPOINT=https://opencode.ai/inference/go/openai/v1/chat/completions \
//! PROVIDER=completions MODEL=deepseek-v4.1-flash API_TOKEN=... cargo run
//! ```
//!
//! Each conversation is a [`Agent`] component on its own entity. Tabs select
//! the visible agent while all agents can stream independently.
//!
//! Stages communicate through Bevy messages:
//!
//! transport task -> `StreamChunk`/`StreamEnded` -> decoder system
//! -> `AgentStreamEvent` (UI) and `AgentTurnFinished` (history/tool calls)
//! -> `RunToolCalls` -> `StartAgentTurn` -> next transport task.
//!
//! JavaScript is not part of that synchronous chain. Each agent's Boa context
//! lives on its own OS thread (`js_runtime`), and tool routing is centralized in
//! `tools`: world tools run on the main loop, while `eval_javascript`/
//! `run_javascript` are submitted to the interpreter and completed later.
//!
//! Module layout:
//! - `agent`: agent state, turn loop, tool rounds, and per-agent systems
//! - `transcript`: chat-log cards, collapsible text, bounded window, scrolling
//! - `tool_display`: display-only tool-call naming and result presentation
//! - `glyph_cache`: bounds the shared font-atlas cache
//! - `messages`: the Bevy message protocol shared by all layers
//! - `transport`: HTTP streaming and provider request bodies
//! - `tools`: tool definitions, routing, and execution
//! - `js_runtime`: off-thread Boa contexts
//! - `virtual_fs`: the shared flat filesystem
//! - `shell`: the top-level tab/panel UI
//! - `config`: provider selection and application configuration

mod agent;
mod brp;
mod config;
mod glyph_cache;
mod js_runtime;
mod messages;
mod sets;
mod shell;
mod tool_display;
mod tools;
mod transcript;
mod transport;
#[allow(dead_code, reason = "wired into VirtualFs during the .tools phase")]
mod vfs_path;
mod virtual_fs;

use bevy::{
    input_focus::tab_navigation::TabNavigationPlugin, prelude::*, window::WindowResolution,
};
use bevy_brp_extras::BrpExtrasPlugin;
use bevy_tokio_tasks::TokioTasksPlugin;

use agent::AgentPlugin;
use brp::HarnessBrpPlugin;
use config::ConfigPlugin;
use glyph_cache::GlyphCachePlugin;
use js_runtime::JsRuntimePlugin;
use messages::MessagesPlugin;
pub(crate) use sets::AppSet;
use shell::ShellPlugin;
use tools::ToolsPlugin;
use transcript::TranscriptPlugin;
use virtual_fs::VirtualFsPlugin;

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "LLM Agents".into(),
            resolution: WindowResolution::new(1120, 800),
            ..default()
        }),
        ..default()
    }))
    .add_plugins((TokioTasksPlugin::default(), TabNavigationPlugin))
    .add_plugins((
        ConfigPlugin,
        MessagesPlugin,
        VirtualFsPlugin,
        GlyphCachePlugin,
        JsRuntimePlugin,
        AgentPlugin,
        ToolsPlugin,
        TranscriptPlugin,
        ShellPlugin,
    ))
    .add_plugins(HarnessBrpPlugin)
    .add_plugins(BrpExtrasPlugin) // will listen on BRP default port 15702
    .insert_resource(ClearColor(Color::srgb(0.055, 0.065, 0.085)));

    sets::configure_update_sets(&mut app);
    app.run();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_plugins_build_and_register_resources() {
        let mut app = App::new();
        app.add_plugins((
            MessagesPlugin,
            VirtualFsPlugin,
            GlyphCachePlugin,
            JsRuntimePlugin,
            AgentPlugin,
            ToolsPlugin,
            TranscriptPlugin,
            ShellPlugin,
        ))
        .add_plugins(HarnessBrpPlugin);
        sets::configure_update_sets(&mut app);
        app.finish();

        assert!(
            app.world()
                .contains_resource::<crate::agent::AgentRuntime>()
        );
        assert!(app.world().contains_resource::<crate::agent::AgentIndex>());
        assert!(
            app.world()
                .contains_resource::<crate::virtual_fs::VirtualFs>()
        );
        assert!(
            app.world()
                .contains_resource::<crate::js_runtime::JsChannels>()
        );
        assert!(app.world().contains_resource::<crate::tools::ToolsDir>());
        assert!(
            app.world()
                .contains_resource::<crate::tools::schedule::ToolRounds>()
        );
        assert!(
            app.world()
                .contains_resource::<crate::transcript::TextCollapseConfig>()
        );
    }
}
