//! Live reconciliation of `.tools` script tools.
//!
//! The tree revision is cheap to compare, so a system reconciles entities
//! whenever `.tools` changes. `rescan_tools` exposes the same diff to the model
//! and can reload the tree from disk first.

use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

use bevy::log::{info, warn};
use bevy::prelude::*;
use llm_api::generic_tool::{ParameterDescription, ParameterType, Tool};
use serde_json::Value;

use crate::{
    tools::{
        manifest::{self, ScriptToolSpec},
        registry::{Advertised, Availability, Execution, ToolEntry},
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
    pub(crate) fn render(&self) -> String {
        let mut output = format!(
            "{} script tool(s) registered ({} added, {} removed, {} changed); {} broken manifest(s); {} library(ies)",
            self.registered.len(),
            self.added.len(),
            self.removed.len(),
            self.changed.len(),
            self.broken.len(),
            self.libraries
        );
        if !self.added.is_empty() {
            output.push_str(&format!("\nadded: {}", self.added.join(", ")));
        }
        if !self.removed.is_empty() {
            output.push_str(&format!("\nremoved: {}", self.removed.join(", ")));
        }
        if !self.changed.is_empty() {
            output.push_str(&format!("\nchanged: {}", self.changed.join(", ")));
        }
        for broken in &self.broken {
            output.push_str(&format!("\nbroken: {broken}"));
        }
        output
    }
}

/// Registry health, exposed to the UI and BRP.
#[derive(Resource, Default, Clone)]
pub(crate) struct ToolRegistryStatus {
    pub revision: u64,
    pub registered: usize,
    pub broken: usize,
    pub report: String,
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
    commands.spawn((entry, script, Advertised));
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

    for (name, (entity, _)) in live {
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
    if revision == status.revision {
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
    if !report.added.is_empty() || !report.removed.is_empty() || !report.changed.is_empty() {
        info!("tool rescan: {}", report.render());
    }

    status.revision = revision;
    status.registered = report.registered.len();
    status.broken = report.broken.len();
    status.report = report.render();
}

/// Model-facing `rescan_tools`.
pub(crate) fn rescan_tools(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _call: &crate::tools::registry::CallContext<'_>,
) -> (String, bool) {
    let apply = match arguments.get("apply") {
        None | Some(Value::Null) => true,
        value => crate::tools::is_truthy(value),
    };
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

    let report = reconcile(world.fs.tools(), &rows, &mut world.commands, apply);
    // Record the revision even for a dry run: the caller asked for no changes,
    // so the automatic reconciler must not apply them on the next frame.
    world.registry_status.revision = world.fs.tools().revision();
    world.registry_status.registered = report.registered.len();
    world.registry_status.broken = report.broken.len();
    world.registry_status.report = report.render();

    let mut output = report.render();
    if !apply {
        output.push_str("\n(dry run: no entities changed)");
    }
    (output, false)
}

/// Definition for the `rescan_tools` tool.
pub(crate) fn rescan_tool_definition() -> Tool {
    Tool {
        name: "rescan_tools".into(),
        description: Some(
            "Reconcile `.tools` script tools with the live registry. Reports added, \
             removed, and changed tools plus every invalid manifest. `apply` defaults \
             to true; set it to false for a dry run. `source` is `vfs` (default, use \
             in-memory edits) or `disk` (reload `./.tools` first, discarding unmirrored \
             in-memory edits)."
                .into(),
        ),
        parameters: HashMap::from([
            (
                "apply".to_owned(),
                ParameterDescription {
                    r#type: ParameterType::Boolean,
                    description: "Apply the diff to the registry (default true)".into(),
                    required: false,
                },
            ),
            (
                "source".to_owned(),
                ParameterDescription {
                    r#type: ParameterType::String,
                    description: "`vfs` (default) or `disk`".into(),
                    required: false,
                },
            ),
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        write_tool(
            &mut app.world_mut().resource_mut::<VirtualFs>(),
            "demo",
            "second",
        );
        app.update();
        assert_eq!(script_tools(&mut app), ["demo"]);

        write_tool(
            &mut app.world_mut().resource_mut::<VirtualFs>(),
            "extra",
            "extra",
        );
        app.update();
        assert_eq!(script_tools(&mut app), ["demo", "extra"]);

        app.world_mut()
            .resource_mut::<VirtualFs>()
            .delete(".tools/demo/main.js")
            .unwrap();
        app.update();
        assert_eq!(script_tools(&mut app), ["extra"]);
    }
}
