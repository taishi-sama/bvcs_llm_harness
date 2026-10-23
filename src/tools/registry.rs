//! Tool registry: every tool is an ECS entity.
//!
//! A tool entity carries its model-facing definition, how it executes, and its
//! policy markers. Registration order never leaks outward: advertisement sorts
//! by `order` then name so the model-facing list is stable between frames
//! (the tool array is part of the provider prompt cache prefix).

use bevy::prelude::{Commands, Component, Entity};
use llm_api::generic_tool::Tool;
use serde_json::Value;

use crate::tools::{
    catalog::{StaticTool, static_tools},
    world::ToolWorld,
};

/// Per-call information that is not world state.
///
/// This is where `origin`, `depth`, and the per-root call budget will live;
/// keeping them here avoids changing every handler signature again.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CallContext<'a> {
    /// Agent on whose behalf the call is running.
    pub source_entity: Entity,
    /// Its name at dispatch time (owned by `execute_tool` so the `Agent`
    /// borrow can end before the handler gets `&mut ToolWorld`).
    pub source_agent: &'a str,
}

/// Handler signature for tools executed synchronously on the main loop.
pub(crate) type ToolFn =
    for<'w, 's, 'a> fn(&mut ToolWorld<'w, 's>, &Value, &CallContext<'a>) -> (String, bool);

/// How one tool call is carried out.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Execution {
    /// Synchronous, on the main loop.
    Inline(ToolFn),
    /// Submitted to the agent's JavaScript thread and completed later.
    Interpreter,
    /// Submitted to a fresh `.tools` context and completed later.
    Script,
}

/// Whether a tool may only be reached by naming it explicitly.
///
/// Advertisement is a separate marker ([`Advertised`]), because a tool can be
/// in the model's schema without being search-only, and vice versa.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Availability {
    /// Reachable from any origin, including search results.
    #[default]
    AnyWay,
    /// Hidden from the schema and from `search_tools`; still callable when
    /// named directly. This is a discovery policy, not access control.
    OnlyManual,
}

/// One registered tool.
#[derive(Component)]
pub(crate) struct ToolEntry {
    /// Name, description, and parameter schema advertised to providers.
    pub definition: Tool,
    pub execution: Execution,
    pub availability: Availability,
    /// Stable advertisement order; lower comes first.
    pub order: u32,
    /// Coarse grouping ("fs", "js", "agent") for filtering and diagnostics.
    #[allow(dead_code, reason = "consumed by tag filters and diagnostics")]
    pub tags: &'static [&'static str],
}

/// Include this tool in the model-facing tool schema.
#[derive(Component)]
pub(crate) struct Advertised;

/// Keep the tool registered but reject every call, explaining why.
#[derive(Component)]
pub(crate) struct DisabledTool {
    pub reason: &'static str,
}

/// Tools that scripts (`execute`) must not call, such as agent lifecycle.
#[derive(Component)]
pub(crate) struct ModelOnly;

/// Dispatch-relevant copy of one tool's state, taken before the handler runs
/// so `&mut ToolWorld` is free of the registry borrow.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Found {
    /// The tool entity, for execution kinds that carry their own data
    /// (`ScriptTool`).
    pub entity: Entity,
    pub execution: Execution,
    /// Used by the origin checks that replace the old name-based routing.
    #[allow(dead_code, reason = "consumed by the gateway/origin checks")]
    pub availability: Availability,
    #[allow(dead_code, reason = "consumed by the gateway/origin checks")]
    pub advertised: bool,
    pub disabled: Option<&'static str>,
    #[allow(dead_code, reason = "consumed by the script-origin checks")]
    pub model_only: bool,
}

/// Locate a tool by the name the model or JavaScript used.
pub(crate) fn find(world: &ToolWorld<'_, '_>, name: &str) -> Option<Found> {
    world
        .tools
        .iter()
        .find_map(|(entity, entry, advertised, disabled, model_only)| {
            (entry.definition.name == name).then_some(Found {
                entity,
                execution: entry.execution,
                availability: entry.availability,
                advertised,
                disabled: disabled.map(|disabled| disabled.reason),
                model_only,
            })
        })
}

/// Spawn one entity per statically registered tool.
///
/// Runs in `PreStartup`, so every `Startup` consumer already sees the whole
/// catalog.
pub(crate) fn register_static_tools(mut commands: Commands) {
    for StaticTool {
        definition,
        execution,
        availability,
        advertised,
        order,
        tags,
    } in static_tools()
    {
        let mut entity = commands.spawn(ToolEntry {
            definition,
            execution,
            availability,
            order,
            tags,
        });
        if advertised {
            entity.insert(Advertised);
        }
    }
}
