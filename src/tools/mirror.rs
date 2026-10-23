//! Boot seeding and best-effort on-disk mirroring for `.tools`.
//!
//! The in-memory tree is the source of truth. The mirror is only a
//! convenience: it lets the user edit tool sources with a normal editor and
//! keeps them across restarts. A mirror failure is surfaced, never fatal.

use std::{env, fs, path::PathBuf};

use bevy::prelude::*;

use crate::virtual_fs::VirtualFs;

/// Directory that mirrors the `.tools` tree (defaults to `./.tools`).
#[derive(Resource, Clone, Debug)]
pub(crate) struct ToolsDir {
    pub root: PathBuf,
}

impl Default for ToolsDir {
    fn default() -> Self {
        let root = env::var_os("TOOLS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".tools"));
        let root = if root.is_absolute() {
            root
        } else {
            env::current_dir()
                .map(|cwd| cwd.join(&root))
                .unwrap_or(root)
        };
        Self { root }
    }
}

/// Mirror health, exposed for the UI and BRP inspection.
#[derive(Resource, Default)]
pub(crate) struct MirrorStatus {
    pub last_error: Option<String>,
    pub pending: usize,
}

#[derive(Resource)]
pub(crate) struct MirrorDebounce(Timer);

impl Default for MirrorDebounce {
    fn default() -> Self {
        Self(Timer::from_seconds(0.25, TimerMode::Repeating))
    }
}

/// Import `./.tools` into the tree once, before tools are registered.
///
/// Seeding never marks files dirty, so boot does not queue write-backs for the
/// files it just read.
pub(crate) fn seed_tool_tree(dir: Res<ToolsDir>, mut vfs: ResMut<VirtualFs>) {
    if let Err(err) = fs::create_dir_all(&dir.root) {
        warn!("could not create `{}`: {err}", dir.root.display());
    }
    match vfs.tools_mut().seed_from_dir(&dir.root) {
        Ok(0) => info!("`{}` is empty", dir.root.display()),
        Ok(count) => info!(
            "seeded {count} tool file(s) from `{}` (mirroring suppressed)",
            dir.root.display()
        ),
        Err(err) => warn!(
            "could not seed `.tools` from `{}`: {err}",
            dir.root.display()
        ),
    }
}

/// Debounced mirror flush. Runs on the main loop, so no locking is needed.
pub(crate) fn flush_tool_mirror(
    time: Res<Time>,
    dir: Res<ToolsDir>,
    mut vfs: ResMut<VirtualFs>,
    mut status: ResMut<MirrorStatus>,
    mut debounce: ResMut<MirrorDebounce>,
) {
    if !vfs.tools().has_dirty() {
        status.pending = 0;
        status.last_error = None;
        return;
    }

    debounce.0.tick(time.delta());
    if !debounce.0.just_finished() {
        status.pending = 1;
        return;
    }

    let failures = vfs.tools_mut().flush_to_dir(&dir.root);
    status.last_error = failures.first().cloned();
    status.pending = if vfs.tools().has_dirty() { 1 } else { 0 };
    for failure in failures {
        warn!("tool mirror: {failure}");
    }
}
