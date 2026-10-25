//! Static tool catalog and advertisement policy.
//!
//! Definitions still live in [`crate::tools::tools`]; this module only decides
//! how each one is exposed and carried out. Splitting the policy out keeps the
//! model-facing trim reviewable in one small table instead of being spread
//! across the definitions.

use bevy::log::{info, warn};
use bevy::prelude::{Commands, Query, Res, With, Without};
use llm_api::generic_tool::Tool;

use crate::{
    tools::{
        definitions,
        registry::{Advertised, Availability, DisabledTool, Execution, ToolEntry, ToolFn},
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
    auto_background: bool,
    model_only: bool,
}

const fn advertises(order: u32, tags: &'static [&'static str]) -> Policy {
    Policy {
        advertised: true,
        availability: Availability::AnyWay,
        order,
        tags,
        auto_background: false,
        model_only: false,
    }
}

const fn manual(order: u32, tags: &'static [&'static str]) -> Policy {
    Policy {
        advertised: false,
        availability: Availability::OnlyManual,
        order,
        tags,
        auto_background: false,
        model_only: false,
    }
}

impl Policy {
    /// Tools that scripts and JavaScript must not call (agent lifecycle and
    /// the interpreter itself).
    const fn model_only(mut self) -> Self {
        self.model_only = true;
        self
    }

    /// Tools that opt into the global auto-background grace period by default.
    const fn auto_background(mut self) -> Self {
        self.auto_background = true;
        self
    }
}

/// Per-tool exposure policy. Keep this table in one place so a trim is a
/// single reviewable diff.
fn policy(name: &str) -> Policy {
    match name {
        "execute" => advertises(2, &["dispatch"]).model_only(),
        "rescan_tools" => advertises(3, &["dispatch", "tools"]),
        "search_tools" => advertises(1, &["search"]),

        "fs_read" => advertises(10, &["fs"]),
        "fs_write" => advertises(11, &["fs"]),
        "fs_edit_pattern" => advertises(12, &["fs"]),
        "fs_edit_lines" => advertises(13, &["fs"]),
        "fs_ls" => advertises(14, &["fs"]),
        "fs_grep" => advertises(15, &["fs"]),
        "fs_sed" => advertises(16, &["fs"]),
        "fs_delete" => manual(17, &["fs"]),

        "eval_javascript" => advertises(20, &["js"]).model_only().auto_background(),
        "run_javascript" => advertises(21, &["js"]).model_only().auto_background(),
        "eval_javascript_reset_context" => manual(22, &["js"]),

        "send_message_to_agent" => advertises(30, &["agent"]),
        "spawn_agent" => advertises(31, &["agent"]).model_only(),
        "rename_self" => advertises(32, &["agent"]).model_only(),
        "list_agents" => manual(33, &["agent"]),
        "get_agent_history" => manual(34, &["agent"]),
        "get_agent_message" => manual(35, &["agent"]),
        "end_agent" => manual(36, &["agent"]).model_only(),

        // Unknown names stay reachable but hidden: a definition without a
        // policy is a bug to fix, not a reason to break dispatch.
        _ => manual(900, &[]),
    }
}

/// Inline handler lookup across every tool module.
///
/// Modules register through the shared `tool_handlers!` macro, so adding a
/// module means adding exactly one line here; dispatch order never changes
/// which name wins because each module owns a disjoint name set.
fn inline_handler(name: &str) -> Option<ToolFn> {
    crate::tools::fs::handler(name)
        .or_else(|| crate::tools::agent_tools::handler(name))
        .or_else(|| crate::tools::search::handler(name))
        .or_else(|| crate::tools::rescan::handler(name))
}

fn execution(name: &str) -> Execution {
    if name == "execute" {
        return Execution::Gateway;
    }
    if crate::tools::is_js_tool(name) {
        return Execution::Interpreter;
    }
    match inline_handler(name) {
        Some(handler) => Execution::Inline(handler),
        // A definition with no handler is a bug; `every_definition_has_a_handler`
        // fails in tests rather than letting the model hit this at runtime.
        None => Execution::Missing,
    }
}

/// Whether a definition has a real handler.
///
/// Tests use this to catch a tool added to `definitions` but never wired into
/// an execution module, which would otherwise surface only as a runtime error.
#[cfg(test)]
pub(crate) fn is_implemented(name: &str) -> bool {
    !matches!(execution(name), Execution::Missing)
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
        crate::tools::rescan::spawn_script_tool(&mut commands, spec);
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
                auto_background,
                model_only,
            } = policy(&definition.name);
            StaticTool {
                execution: execution(&definition.name),
                auto_background,
                model_only,
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
