//! Ordered phases of the main `Update` schedule.
//!
//! Plugins attach their systems to a phase; [`configure_update_sets`] chains the
//! phases once. Keep the chain below ordered: it is the single source of truth
//! for cross-plugin scheduling.

use bevy::{
    ecs::schedule::IntoScheduleConfigs,
    prelude::{App, SystemSet, Update},
};

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AppSet {
    /// User input and agent creation.
    Input,
    /// Decode the transport stream and finish a turn.
    Turn,
    /// Refresh the immutable `.tools` source snapshot.
    ToolSources,
    /// Drain async JavaScript and script completions.
    PollEvents,
    /// Handle `execute()` / module-read requests from worker threads.
    HostRequests,
    /// Dispatch the tool calls of a finished turn.
    ExecuteTools,
    /// Detach long-running calls that exceeded their grace period.
    AutoBackground,
    /// Commit finished tool results.
    ApplyResults,
    /// Update the tool-activity readout.
    ToolActivity,
    /// Best-effort disk mirror of `.tools`.
    MirrorFlush,
    /// Queue cross-agent messages.
    SendAgentMessage,
    /// Reconcile `.tools` manifests with the registry.
    ReconcileTools,
    /// Resume or fail a turn after tool results.
    Resume,
    /// Rebuild/append transcript cards.
    Transcripts,
    /// Rename and remove agents.
    AgentMaintenance,
}

/// Chain every phase once, in execution order.
pub(crate) fn configure_update_sets(app: &mut App) {
    app.configure_sets(
        Update,
        (
            AppSet::Input,
            AppSet::Turn,
            AppSet::ToolSources,
            AppSet::PollEvents,
            AppSet::HostRequests,
            AppSet::ExecuteTools,
            AppSet::AutoBackground,
            AppSet::ApplyResults,
            AppSet::ToolActivity,
            AppSet::MirrorFlush,
            AppSet::SendAgentMessage,
            AppSet::ReconcileTools,
            AppSet::Resume,
            AppSet::Transcripts,
            AppSet::AgentMaintenance,
        )
            .chain(),
    );
}
