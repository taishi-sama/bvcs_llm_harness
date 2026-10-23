//! The shared ECS world handed to inline tool handlers.
//!
//! Handlers take one `&mut ToolWorld` instead of the dozen separate parameters
//! the old `execute_tool` needed, so adding a resource does not touch every
//! call site.

use bevy::ecs::system::SystemParam;
use bevy::prelude::{Commands, MessageWriter, Query, Res, ResMut};

use crate::{
    agent::{Agent, AgentIndex, AgentLayout},
    config::AppConfig,
    messages::{RemoveAgentRequest, RenameAgentRequest, SendAgentMessage},
    tools::{
        registry::{Advertised, DisabledTool, ModelOnly, ToolEntry},
        script::ScriptTool,
    },
    virtual_fs::VirtualFs,
};

pub(crate) type ToolRow = (
    bevy::prelude::Entity,
    &'static ToolEntry,
    bevy::ecs::query::Has<Advertised>,
    Option<&'static DisabledTool>,
    bevy::ecs::query::Has<ModelOnly>,
);

/// Every resource and query an inline tool may touch.
#[derive(SystemParam)]
pub(crate) struct ToolWorld<'w, 's> {
    pub index: ResMut<'w, AgentIndex>,
    pub layout: ResMut<'w, AgentLayout>,
    pub fs: ResMut<'w, VirtualFs>,
    pub config: Res<'w, AppConfig>,
    /// Read-only: handlers inspect agents but mutate them through messages.
    pub agents: Query<'w, 's, &'static Agent>,
    /// Every registered tool plus the marker state needed to advertise,
    /// dispatch, and search it. `ToolEntry` is deliberately immutable here so
    /// a handler can never change the catalog mid-round; that goes through a
    /// message handled outside dispatch.
    pub tools: Query<'w, 's, ToolRow>,
    /// Script-tool execution data, keyed by tool entity.
    pub scripts: Query<'w, 's, &'static ScriptTool>,
    pub commands: Commands<'w, 's>,
    pub send_agent_message: MessageWriter<'w, SendAgentMessage>,
    pub remove_agent: MessageWriter<'w, RemoveAgentRequest>,
    pub rename_agent: MessageWriter<'w, RenameAgentRequest>,
}
