//! Live reconciliation of `.tools` script tools.
//!
//! The tree revision is cheap to compare, so a system reconciles entities
//! whenever `.tools` changes. `rescan_tools` exposes the same diff to the model
//! and can reload the tree from disk first.
//!
//! Registered through the shared `tool_handlers!` macro, like every inline
//! tool module.

use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

use bevy::log::{info, warn};
use bevy::prelude::*;
use llm_api::generic_tool::Tool;
use serde_json::Value;

use crate::{
    tools::{
        CallContext, bool_option,
        manifest::{self, ScriptToolSpec},
        registration::tool_handlers,
        registry::{Availability, Execution, ToolEntry},
        script::ScriptTool,
        tree::ToolsTree,
        world::ToolWorld,
    },
    virtual_fs::VirtualFs,
};

/// Result of one reconciliation pass.
pub(crate) struct RescanReport {
    pub registered: Vec<String>,
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<String>,
    pub broken: Vec<String>,
    pub libraries: usize,
}

impl RescanReport {
    /// Added/removed/changed lines, shared by the rendered report and the
    /// one-line automatic-change summary.
    fn diff_parts(&self) -> Vec<String> {
        let mut parts = Vec::new();
        if !self.added.is_empty() {
            parts.push(format!("added: {}", self.added.join(", ")));
        }
        if !self.removed.is_empty() {
            parts.push(format!("removed: {}", self.removed.join(", ")));
        }
        if !self.changed.is_empty() {
            parts.push(format!("changed: {}", self.changed.join(", ")));
        }
        parts
    }

    pub(crate) fn render(&self) -> String {
        let mut output = format!(
            "{} script tool(s) found in `.tools` ({} added, {} removed, {} changed); {} broken manifest(s); {} library(ies)",
            self.registered.len(),
            self.added.len(),
            self.removed.len(),
            self.changed.len(),
            self.broken.len(),
            self.libraries
        );
        for part in self.diff_parts() {
            output.push('\n');
            output.push_str(&part);
        }
        for broken in &self.broken {
            output.push_str(&format!("\nbroken: {broken}"));
        }
        output
    }

    /// One-line summary of what this pass changed, for the "last automatic
    /// change" note in a later `rescan_tools` result.
    fn diff_summary(&self) -> String {
        self.diff_parts().join("; ")
    }
}

/// Registry health, exposed to the UI and BRP.
#[derive(Resource, Default, Clone, Reflect)]
#[reflect(Resource)]
pub(crate) struct ToolRegistryStatus {
    pub revision: u64,
    pub registered: usize,
    pub broken: usize,
    pub report: String,
    /// Summary of the last automatic reconciliation that actually changed the
    /// registry. `rescan_tools` surfaces it when no newer diff exists.
    pub last_auto: String,
    /// Tree revision described by `last_auto`.
    pub last_auto_revision: u64,
    /// `rescan_tools(apply: true)` asks the automatic reconciler to run even if
    /// the revision was already recorded by an earlier dry run.
    pub pending_apply: bool,
}

fn fingerprint(definition: &Tool, entry: &str, export: &str, order: u32) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    serde_json::to_string(definition)
        .unwrap_or_default()
        .hash(&mut hasher);
    entry.hash(&mut hasher);
    export.hash(&mut hasher);
    order.hash(&mut hasher);
    hasher.finish()
}

fn script_tool_entry(spec: &ScriptToolSpec) -> (ToolEntry, ScriptTool) {
    (
        ToolEntry {
            definition: spec.definition.clone(),
            execution: Execution::Script,
            availability: Availability::AnyWay,
            order: 1000u32.saturating_add(spec.order),
            tags: &["script"],
            auto_background: false,
        },
        ScriptTool {
            entry: spec.entry.clone(),
            export: spec.export.clone(),
            fingerprint: fingerprint(&spec.definition, &spec.entry, &spec.export, spec.order),
        },
    )
}

pub(crate) fn spawn_script_tool(commands: &mut Commands, spec: ScriptToolSpec) -> u64 {
    let (entry, script) = script_tool_entry(&spec);
    let fingerprint = script.fingerprint;
    let entity = crate::tools::registry::spawn_tool(commands, entry, true, false);
    commands.entity(entity).insert(script);
    fingerprint
}

fn update_script_tool(commands: &mut Commands, entity: Entity, spec: ScriptToolSpec) {
    let (entry, script) = script_tool_entry(&spec);
    commands.entity(entity).insert((entry, script));
}

/// Diff the tree against live script entities and optionally apply the result.
pub(crate) fn reconcile(
    tree: &ToolsTree,
    existing: &[(Entity, String, u64)],
    commands: &mut Commands,
    apply: bool,
) -> RescanReport {
    let mut warnings = Vec::new();
    let specs = manifest::scan_tools(tree, &mut warnings);
    let libraries = manifest::scan_libraries(tree, &mut warnings).len();

    let mut live: HashMap<String, (Entity, u64)> = existing
        .iter()
        .map(|(entity, name, fingerprint)| (name.clone(), (*entity, *fingerprint)))
        .collect();

    let mut report = RescanReport {
        registered: Vec::new(),
        added: Vec::new(),
        removed: Vec::new(),
        changed: Vec::new(),
        broken: warnings,
        libraries,
    };

    for spec in specs {
        let name = spec.definition.name.clone();
        let printed = fingerprint(&spec.definition, &spec.entry, &spec.export, spec.order);
        match live.remove(&name) {
            None => {
                report.added.push(name.clone());
                if apply {
                    spawn_script_tool(commands, spec);
                }
            }
            Some((entity, old)) if old != printed => {
                report.changed.push(name.clone());
                if apply {
                    update_script_tool(commands, entity, spec);
                }
            }
            Some(_) => {}
        }
        report.registered.push(name);
    }

    let mut removed = live.into_iter().collect::<Vec<_>>();
    removed.sort_by(|(left, _), (right, _)| left.cmp(right));
    for (name, (entity, _)) in removed {
        report.removed.push(name);
        if apply {
            commands.entity(entity).despawn();
        }
    }

    report
}

/// Reconcile script tools whenever the `.tools` revision changed.
pub(crate) fn reconcile_script_tools(
    vfs: Res<VirtualFs>,
    existing: Query<(Entity, &ToolEntry, &ScriptTool)>,
    mut commands: Commands,
    mut status: ResMut<ToolRegistryStatus>,
) {
    let revision = vfs.tools().revision();
    if revision == status.revision && !status.pending_apply {
        return;
    }

    let rows: Vec<(Entity, String, u64)> = existing
        .iter()
        .map(|(entity, entry, script)| (entity, entry.definition.name.clone(), script.fingerprint))
        .collect();

    let report = reconcile(vfs.tools(), &rows, &mut commands, true);
    for broken in &report.broken {
        warn!("tool manifest: {broken}");
    }
    let changed =
        !report.added.is_empty() || !report.removed.is_empty() || !report.changed.is_empty();
    let rendered = report.render();
    if changed {
        info!("tool rescan: {rendered}");
        status.last_auto = report.diff_summary();
        status.last_auto_revision = revision;
    }

    status.pending_apply = false;
    status.revision = revision;
    status.registered = report.registered.len();
    status.broken = report.broken.len();
    status.report = rendered;
}

/// Record what a rescan call wants the automatic reconciler to do.
///
/// Last call wins: an `apply: false` after an `apply: true` cancels the pending
/// apply, matching the dry-run wording in the result.
fn record_rescan_intent(
    status: &mut ToolRegistryStatus,
    apply: bool,
    changed: bool,
    revision: u64,
) {
    if apply {
        if changed {
            status.pending_apply = true;
        }
    } else {
        status.revision = revision;
        status.pending_apply = false;
    }
}

/// Model-facing `rescan_tools`.
pub(crate) fn rescan_tools(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _call: &CallContext<'_>,
) -> (String, bool) {
    let apply = bool_option(arguments, "apply", true);
    let source = arguments
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or("vfs");

    match source {
        "vfs" => {}
        "disk" => {
            let root = world.tools_dir.root.clone();
            if let Err(err) = world.fs.tools_mut().seed_from_dir(&root) {
                return (
                    format!("could not reload `.tools` from `{}`: {err}", root.display()),
                    true,
                );
            }
        }
        other => {
            return (
                format!("`source` must be `vfs` or `disk`, not `{other}`"),
                true,
            );
        }
    }

    let mut rows = Vec::new();
    for (entity, entry, _, _, _) in world.tools.iter() {
        if let Ok(script) = world.scripts.get(entity) {
            rows.push((entity, entry.definition.name.clone(), script.fingerprint));
        }
    }

    let last_auto = world.registry_status.last_auto.clone();
    let last_auto_revision = world.registry_status.last_auto_revision;
    let revision = world.fs.tools().revision();

    // Never spawn/despawn through deferred commands here: a second rescan in
    // the same assistant batch would not see the first pass's entities and
    // would create duplicates. The automatic reconciler applies the diff later
    // in this frame, before the next request is built.
    let report = reconcile(world.fs.tools(), &rows, &mut world.commands, false);
    let changed =
        !report.added.is_empty() || !report.removed.is_empty() || !report.changed.is_empty();

    record_rescan_intent(&mut world.registry_status, apply, changed, revision);

    let mut output = report.render();
    if !changed && last_auto_revision == revision && !last_auto.is_empty() {
        output.push_str(&format!("\nlast automatic change: {last_auto}"));
    }
    if !apply {
        output.push_str(
            "\n(dry run: no entities changed; the automatic reconciler will skip this revision)",
        );
    } else if changed {
        output.push_str("\n(the automatic reconciler will apply this before the next request)");
    }
    (output, false)
}

tool_handlers! {
    "rescan_tools" => rescan_tools,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::registry::Advertised;

    fn write_tool(vfs: &mut VirtualFs, name: &str, description: &str) {
        vfs.write(
            &format!(".tools/{name}/tool.json"),
            format!(r#"{{"name":"{name}","description":"{description}"}}"#),
        )
        .unwrap();
        vfs.write(
            &format!(".tools/{name}/main.js"),
            "export function main() { return 'ok'; }".to_owned(),
        )
        .unwrap();
    }

    fn script_tools(app: &mut App) -> Vec<String> {
        let world = app.world_mut();
        let mut query = world.query::<(&ToolEntry, &ScriptTool)>();
        let mut names: Vec<String> = query
            .iter(world)
            .map(|(entry, _)| entry.definition.name.clone())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn reconciliation_tracks_add_update_and_remove() {
        let mut app = App::new();
        app.init_resource::<ToolRegistryStatus>();
        app.insert_resource(VirtualFs::default());
        app.add_systems(Update, reconcile_script_tools);

        write_tool(
            &mut app.world_mut().resource_mut::<VirtualFs>(),
            "demo",
            "first",
        );
        app.update();
        assert_eq!(script_tools(&mut app), ["demo"]);
        let status = app.world().resource::<ToolRegistryStatus>();
        assert_eq!(status.registered, 1);
        assert_eq!(status.broken, 0);
        assert_eq!(status.last_auto, "added: demo");
        assert!(
            status.report.contains("found in `.tools`"),
            "{}",
            status.report
        );

        write_tool(
            &mut app.world_mut().resource_mut::<VirtualFs>(),
            "demo",
            "second",
        );
        app.update();
        assert_eq!(script_tools(&mut app), ["demo"]);
        assert_eq!(
            app.world().resource::<ToolRegistryStatus>().last_auto,
            "changed: demo"
        );

        write_tool(
            &mut app.world_mut().resource_mut::<VirtualFs>(),
            "extra",
            "extra",
        );
        app.update();
        assert_eq!(script_tools(&mut app), ["demo", "extra"]);
        assert_eq!(
            app.world().resource::<ToolRegistryStatus>().last_auto,
            "added: extra"
        );

        app.world_mut()
            .resource_mut::<VirtualFs>()
            .delete(".tools/demo/main.js")
            .unwrap();
        app.update();
        assert_eq!(script_tools(&mut app), ["extra"]);
        assert_eq!(
            app.world().resource::<ToolRegistryStatus>().last_auto,
            "removed: demo"
        );
    }

    #[test]
    fn source_only_edit_does_not_replay_the_last_change() {
        let mut app = App::new();
        app.init_resource::<ToolRegistryStatus>();
        app.insert_resource(VirtualFs::default());
        app.add_systems(Update, reconcile_script_tools);

        write_tool(
            &mut app.world_mut().resource_mut::<VirtualFs>(),
            "demo",
            "first",
        );
        app.update();
        let change_revision = app
            .world()
            .resource::<ToolRegistryStatus>()
            .last_auto_revision;

        app.world_mut()
            .resource_mut::<VirtualFs>()
            .write(
                ".tools/demo/main.js",
                "export function main() { return 'changed'; }".to_owned(),
            )
            .unwrap();
        app.update();

        let status = app.world().resource::<ToolRegistryStatus>();
        assert_eq!(status.last_auto, "added: demo");
        assert_eq!(
            status.last_auto_revision, change_revision,
            "a source-only edit must not make the old change look current"
        );
        assert_ne!(
            app.world().resource::<VirtualFs>().tools().revision(),
            change_revision
        );
    }

    #[test]
    fn removal_order_is_stable() {
        let mut app = App::new();
        app.init_resource::<ToolRegistryStatus>();
        app.insert_resource(VirtualFs::default());
        app.add_systems(Update, reconcile_script_tools);

        for name in ["beta", "alpha"] {
            write_tool(&mut app.world_mut().resource_mut::<VirtualFs>(), name, name);
        }
        app.update();

        let vfs = &mut app.world_mut().resource_mut::<VirtualFs>();
        vfs.delete(".tools/alpha/main.js").unwrap();
        vfs.delete(".tools/beta/main.js").unwrap();
        app.update();

        assert_eq!(
            app.world().resource::<ToolRegistryStatus>().last_auto,
            "removed: alpha, beta"
        );
    }

    #[test]
    fn dry_run_cancels_a_pending_apply() {
        let mut status = ToolRegistryStatus::default();
        record_rescan_intent(&mut status, true, true, 7);
        assert!(status.pending_apply);

        record_rescan_intent(&mut status, false, true, 7);
        assert!(!status.pending_apply, "the later dry run must win");
        assert_eq!(status.revision, 7);
    }

    #[test]
    fn apply_after_a_dry_run_forces_the_reconciler() {
        let mut status = ToolRegistryStatus::default();
        record_rescan_intent(&mut status, false, true, 7);
        assert!(!status.pending_apply);
        assert_eq!(status.revision, 7);

        record_rescan_intent(&mut status, true, true, 7);
        assert!(status.pending_apply);
    }

    #[derive(Resource, Default)]
    struct Seen(Vec<String>);

    fn record_advertised(tools: Query<&ToolEntry, With<Advertised>>, mut seen: ResMut<Seen>) {
        seen.0 = tools
            .iter()
            .map(|entry| entry.definition.name.clone())
            .collect();
    }

    #[test]
    fn reconcile_before_request_sees_new_tools() {
        let mut app = App::new();
        app.init_resource::<ToolRegistryStatus>();
        app.insert_resource(VirtualFs::default());
        app.init_resource::<Seen>();
        app.add_systems(Update, (reconcile_script_tools, record_advertised).chain());

        write_tool(
            &mut app.world_mut().resource_mut::<VirtualFs>(),
            "demo",
            "first",
        );
        app.update();

        assert!(
            app.world().resource::<Seen>().0.iter().any(|n| n == "demo"),
            "the request-facing query must see the tool spawned earlier in the frame"
        );
    }
}
