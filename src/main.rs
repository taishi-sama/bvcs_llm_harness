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
//! - `glyph_cache`: bounds the shared font-atlas cache
//! - `messages`: the Bevy message protocol shared by all layers
//! - `transport`: HTTP streaming and provider request bodies
//! - `tools`: tool definitions, routing, and execution
//! - `js_runtime`: off-thread Boa contexts
//! - `virtual_fs`: the shared flat filesystem
//! - `shell`: the top-level tab/panel UI
//! - `config`: provider selection and application configuration

mod agent;
mod config;
mod glyph_cache;
mod js_runtime;
mod messages;
mod shell;
mod tools;
mod transcript;
mod transport;
#[allow(dead_code, reason = "wired into VirtualFs during the .tools phase")]
mod vfs_path;
mod virtual_fs;

use std::env;

use bevy::{
    input_focus::tab_navigation::TabNavigationPlugin, prelude::*, ui::UiSystems,
    window::WindowResolution,
};
use bevy_brp_extras::BrpExtrasPlugin;
use bevy_tokio_tasks::TokioTasksPlugin;
use js_runtime::{JsChannels, start_js_runtime, stop_js_runtime};

use agent::*;
use config::{AppConfig, Provider, session_id_from_env};
use glyph_cache::{GlyphCacheLimit, limit_glyph_cache, log_glyph_cache_limit};
use messages::*;
use shell::setup;
use transcript::{
    TextCollapseConfig, auto_scroll, close_active_agent_menu, open_active_agent_menu,
    refresh_transcripts, update_tool_activity,
};
use virtual_fs::{VfsStateDir, VirtualFs};

fn main() {
    let provider = Provider::from_env();
    let model_name = env::var("MODEL").unwrap_or_else(|_| "qwen3.6-35b-a3b".to_owned());
    let endpoint = env::var("ENDPOINT").unwrap_or_else(|_| provider.endpoint().to_owned());
    let token = env::var("API_TOKEN").unwrap_or_default();
    let session_id = session_id_from_env();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "LLM Agents".into(),
                resolution: WindowResolution::new(1120, 800),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((TokioTasksPlugin::default(), TabNavigationPlugin))
        .add_plugins(BrpExtrasPlugin) // will listen on BRP default port 15702
        .insert_resource(ClearColor(Color::srgb(0.055, 0.065, 0.085)))
        .insert_resource(AppConfig {
            provider,
            model_name,
            endpoint,
            token,
            session_id,
        })
        .init_resource::<AgentRuntime>()
        .init_resource::<AgentIndex>()
        .init_resource::<VirtualFs>()
        .init_resource::<VfsStateDir>()
        .init_resource::<JsChannels>()
        .init_resource::<tools::ToolsDir>()
        .init_resource::<tools::MirrorStatus>()
        .init_resource::<tools::MirrorDebounce>()
        .init_resource::<ToolRounds>()
        .init_resource::<BackgroundPolicy>()
        .init_resource::<TextCollapseConfig>()
        .init_resource::<GlyphCacheLimit>()
        .add_observer(attach_js_runtime)
        .add_observer(index_agent_on_insert)
        .add_observer(index_agent_on_discard)
        .add_observer(deactivate_removed_agent)
        .add_observer(open_active_agent_menu)
        .add_observer(close_active_agent_menu)
        .add_observer(start_js_runtime)
        .add_observer(stop_js_runtime)
        .add_message::<StartAgentTurn>()
        .add_message::<NewAgentRequested>()
        .add_message::<CancelAgentRequest>()
        .add_message::<RetryAgentTurn>()
        .add_message::<StreamChunk>()
        .add_message::<StreamEnded>()
        .add_message::<StreamFailed>()
        .add_message::<AgentStreamEvent>()
        .add_message::<AgentTurnFinished>()
        .add_message::<AgentFailed>()
        .add_message::<RunToolCalls>()
        .add_message::<ToolRoundFinished>()
        .add_message::<SendAgentMessage>()
        .add_message::<RemoveAgentRequest>()
        .add_message::<RenameAgentRequest>()
        .add_message::<TaskResult>()
        .add_systems(
            PreStartup,
            (
                tools::seed_tool_tree,
                tools::register_static_tools,
                tools::register_script_tools,
            )
                .chain(),
        )
        .add_systems(Startup, (setup, log_glyph_cache_limit))
        .add_systems(Update, limit_glyph_cache)
        .add_systems(
            Update,
            (
                (
                    create_requested_agents,
                    sync_retry_buttons,
                    cancel_agent_turn,
                    retry_agent_turn,
                    submit_input,
                    deliver_pending_messages,
                    decode_stream_chunks,
                    handle_stream_events,
                    handle_turn_finished,
                )
                    .chain(),
                (
                    poll_js_events,
                    handle_task_result,
                    poll_js_host_requests,
                    execute_tool_calls,
                    auto_background_long_js_tasks,
                    apply_tool_results,
                    update_tool_activity,
                    tools::flush_tool_mirror,
                    handle_send_agent_message,
                    start_agent_turn,
                    handle_agent_failed,
                    refresh_transcripts,
                    rename_agents,
                    remove_agents,
                )
                    .chain(),
            )
                .chain(),
        )
        .add_systems(PostUpdate, auto_scroll.after(UiSystems::Layout))
        .run();
}
