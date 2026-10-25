//! Bevy messages shared by the transport, agent, and UI layers.

use bevy::prelude::{App, Entity, Message, Plugin};
use llm_api::{stream::StreamEvent, usage::Usage};
use llm_history_model::history::{Entry, ToolCall, ToolCallResult};

/// Registers the complete message protocol exactly once.
pub(crate) struct MessagesPlugin;

impl Plugin for MessagesPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<StartAgentTurn>()
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
            .add_message::<TaskResult>();
    }
}

/// Ask the agent coordinator to start a request for the current history.
#[derive(Message)]
pub(crate) struct StartAgentTurn {
    pub(crate) agent: Entity,
}

/// Create a new agent tab and panel.
#[derive(Message)]
pub(crate) struct NewAgentRequested;

/// Abort the current generation for an agent and drop its pending user turn.
#[derive(Message)]
pub(crate) struct CancelAgentRequest {
    pub(crate) agent: Entity,
}

/// Resend the last user turn after a failure or cancellation.
#[derive(Message)]
pub(crate) struct RetryAgentTurn {
    pub(crate) agent: Entity,
}

/// Raw HTTP body bytes produced by the transport task.
#[derive(Message)]
pub(crate) struct StreamChunk {
    pub(crate) agent: Entity,
    pub(crate) turn_id: u64,
    pub(crate) bytes: Vec<u8>,
}

/// The HTTP body ended normally.
#[derive(Message)]
pub(crate) struct StreamEnded {
    pub(crate) agent: Entity,
    pub(crate) turn_id: u64,
}

/// The HTTP request failed.
#[derive(Message)]
pub(crate) struct StreamFailed {
    pub(crate) agent: Entity,
    pub(crate) turn_id: u64,
    pub(crate) message: String,
}

/// One decoded provider event. UI systems consume these directly.
#[derive(Message)]
pub(crate) struct AgentStreamEvent {
    pub(crate) agent: Entity,
    pub(crate) turn_id: u64,
    pub(crate) event: StreamEvent,
}

/// The stream decoder finished and produced replayable history entries.
#[derive(Message)]
pub(crate) struct AgentTurnFinished {
    pub(crate) agent: Entity,
    pub(crate) turn_id: u64,
    pub(crate) entries: Vec<Entry>,
    pub(crate) usage: Option<Usage>,
    pub(crate) stop_reason: Option<String>,
}

/// An agent turn failed after decoding or during the final commit.
#[derive(Message)]
pub(crate) struct AgentFailed {
    pub(crate) agent: Entity,
    pub(crate) turn_id: u64,
    pub(crate) message: String,
}

/// Execute the tool calls carried by a finished assistant turn.
#[derive(Message)]
pub(crate) struct RunToolCalls {
    pub(crate) agent: Entity,
    pub(crate) turn_id: u64,
    pub(crate) tool_call: Vec<ToolCall>,
}

/// All results for one tool-call round are available.
#[derive(Message)]
pub(crate) struct ToolRoundFinished {
    pub(crate) agent: Entity,
    pub(crate) turn_id: u64,
    pub(crate) calls: Vec<ToolCall>,
    pub(crate) results: Vec<ToolCallResult>,
}

/// Queue a message for another agent (used by the cross-agent tool).
#[derive(Message)]
pub(crate) struct SendAgentMessage {
    pub(crate) target: Entity,
    pub(crate) from: String,
    pub(crate) message: String,
    /// When true, deliver `message` exactly as written instead of wrapping it
    /// in the cross-agent `<system_message>` envelope.
    pub(crate) verbatim: bool,
}

/// Permanently remove an agent, its UI, and every runtime associated with it.
#[derive(Message)]
pub(crate) struct RemoveAgentRequest {
    pub(crate) agent: Entity,
}

/// A detached background task finished; deliver its result as a message.
#[derive(Message)]
pub(crate) struct TaskResult {
    pub(crate) agent: Entity,
    pub(crate) task_id: u64,
    pub(crate) result: String,
    pub(crate) is_error: bool,
}

/// Rename an agent after a tool call has reserved the new name.
#[derive(Message)]
pub(crate) struct RenameAgentRequest {
    pub(crate) agent: Entity,
    pub(crate) name: String,
}
