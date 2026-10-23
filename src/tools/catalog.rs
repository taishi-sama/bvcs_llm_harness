//! Static tool catalog and advertisement policy.
//!
//! Definitions still live in [`crate::tools::tools`]; this module only decides
//! how each one is exposed and carried out. Splitting the policy out keeps the
//! model-facing trim reviewable in one small table instead of being spread
//! across the definitions.

use bevy::log::{info, warn};
use bevy::prelude::{Commands, Query, Res, With, Without};
use llm_api::generic_tool::Tool;
use serde_json::Value;

use crate::{
    tools::{
        definitions,
        registry::{Advertised, Availability, DisabledTool, Execution, ToolEntry},
    },
    virtual_fs::VirtualFs,
};

/// One definition plus its registration policy.
pub(crate) struct StaticTool {
    pub definition: Tool,
    pub execution: Execution,
    pub availability: Availability,
    pub advertised: bool,
    pub order: u32,
    pub tags: &'static [&'static str],
    pub auto_background: bool,
    pub model_only: bool,
}

/// Advertisement query used to build a request's tool list.
pub(crate) type AdvertisedTools<'w, 's> =
    Query<'w, 's, &'static ToolEntry, (With<Advertised>, Without<DisabledTool>)>;

struct Policy {
    advertised: bool,
    availability: Availability,
    order: u32,
    tags: &'static [&'static str],
}

const fn advertises(order: u32, tags: &'static [&'static str]) -> Policy {
    Policy {
        advertised: true,
        availability: Availability::AnyWay,
        order,
        tags,
    }
}

const fn manual(order: u32, tags: &'static [&'static str]) -> Policy {
    Policy {
        advertised: false,
        availability: Availability::OnlyManual,
        order,
        tags,
    }
}

/// Per-tool exposure policy. Keep this table in one place so a trim is a
/// single reviewable diff.
fn policy(name: &str) -> Policy {
    match name {
        "execute" => advertises(2, &["dispatch"]),
        "search_tools" => advertises(1, &["search"]),

        "fs_read" => advertises(10, &["fs"]),
        "fs_write" => advertises(11, &["fs"]),
        "fs_edit_pattern" => advertises(12, &["fs"]),
        "fs_edit_lines" => advertises(13, &["fs"]),
        "fs_ls" => advertises(14, &["fs"]),
        "fs_grep" => advertises(15, &["fs"]),
        "fs_sed" => advertises(16, &["fs"]),
        "fs_delete" => manual(17, &["fs"]),

        "eval_javascript" => advertises(20, &["js"]),
        "run_javascript" => advertises(21, &["js"]),
        "eval_javascript_reset_context" => manual(22, &["js"]),

        "send_message_to_agent" => advertises(30, &["agent"]),
        "spawn_agent" => advertises(31, &["agent"]),
        "rename_self" => advertises(32, &["agent"]),
        "list_agents" => manual(33, &["agent"]),
        "get_agent_history" => manual(34, &["agent"]),
        "get_agent_message" => manual(35, &["agent"]),
        "end_agent" => manual(36, &["agent"]),

        // Unknown names stay reachable but hidden: a definition without a
        // policy is a bug to fix, not a reason to break dispatch.
        _ => manual(900, &[]),
    }
}

fn execution(name: &str) -> Execution {
    if name == "execute" {
        return Execution::Gateway;
    }
    if let Some(handler) = crate::tools::fs::handler(name) {
        return Execution::Inline(handler);
    }
    if let Some(handler) = crate::tools::agent_tools::handler(name) {
        return Execution::Inline(handler);
    }
    if name == "search_tools" {
        return Execution::Inline(crate::tools::search::search_tools);
    }
    if crate::tools::is_js_tool(name) {
        return Execution::Interpreter;
    }
    // A definition with no handler is a bug; `every_definition_has_a_handler`
    // fails in tests rather than letting the model hit this at runtime.
    Execution::Inline(no_handler)
}

fn no_handler(
    _world: &mut crate::tools::world::ToolWorld<'_, '_>,
    _arguments: &Value,
    _call: &crate::tools::registry::CallContext<'_>,
) -> (String, bool) {
    ("tool has no registered handler".to_owned(), true)
}

/// Tools that opt into the global auto-background grace period by default.
pub(crate) fn auto_background(name: &str) -> bool {
    matches!(name, "run_javascript" | "eval_javascript")
}

/// Tools that scripts and JavaScript must not call (agent lifecycle and the
/// interpreter itself).
pub(crate) fn model_only(name: &str) -> bool {
    matches!(
        name,
        "spawn_agent"
            | "rename_self"
            | "end_agent"
            | "eval_javascript"
            | "run_javascript"
            | "execute"
    )
}

/// Whether a definition has a real handler.
///
/// Tests use this to catch a tool added to `definitions` but never wired into
/// an execution module, which would otherwise surface only as a runtime error.
#[cfg(test)]
pub(crate) fn is_implemented(name: &str) -> bool {
    crate::tools::fs::handler(name).is_some()
        || crate::tools::agent_tools::handler(name).is_some()
        || name == "search_tools"
        || name == "execute"
        || crate::tools::is_js_tool(name)
}

/// Register script tools found under `.tools`.
///
/// Execution lands in the next phase; registering them now makes the scan and
/// policy observable while keeping them out of the model's schema (`DisabledTool`
/// excludes them from advertisement and explains why a direct call fails).
pub(crate) fn register_script_tools(vfs: Res<VirtualFs>, mut commands: Commands) {
    let mut warnings = Vec::new();
    let specs = crate::tools::manifest::scan_tools(vfs.tools(), &mut warnings);
    let libraries = crate::tools::manifest::scan_libraries(vfs.tools(), &mut warnings);

    for warning in warnings {
        warn!("tool manifest: {warning}");
    }

    for spec in specs {
        info!(
            "registered script tool `{}` from `{}`",
            spec.name, spec.entry
        );
        commands.spawn((
            ToolEntry {
                definition: spec.definition,
                execution: Execution::Script,
                availability: Availability::AnyWay,
                order: 1000u32.saturating_add(spec.order),
                tags: &["script"],
                auto_background: false,
            },
            crate::tools::script::ScriptTool {
                entry: spec.entry,
                export: spec.export,
            },
            Advertised,
        ));
    }

    if !libraries.is_empty() {
        info!(
            "found {} importable tool library(ies) under `.tools`",
            libraries.len()
        );
    }
}

/// Every static tool, ready to be spawned.
pub(crate) fn static_tools() -> Vec<StaticTool> {
    definitions()
        .into_iter()
        .map(|definition| {
            let Policy {
                advertised,
                availability,
                order,
                tags,
            } = policy(&definition.name);
            StaticTool {
                execution: execution(&definition.name),
                auto_background: auto_background(&definition.name),
                model_only: model_only(&definition.name),
                definition,
                availability,
                advertised,
                order,
                tags,
            }
        })
        .collect()
}

/// Model-facing tool list for one request.
///
/// Sorted by `(order, name)` so it does not reshuffle when an entity changes
/// archetype (adding `DisabledTool`, for example).
pub(crate) fn advertised_tools(tools: &AdvertisedTools<'_, '_>) -> Vec<Tool> {
    let mut entries: Vec<&ToolEntry> = tools.iter().collect();
    entries.sort_by(|a, b| {
        a.order
            .cmp(&b.order)
            .then_with(|| a.definition.name.cmp(&b.definition.name))
    });
    entries
        .into_iter()
        .map(|entry| entry.definition.clone())
        .collect()
}
