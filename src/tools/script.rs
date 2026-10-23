//! Script tools: `.tools` modules run in fresh, throwaway Boa contexts.
//!
//! A script call is exactly "reset the context, evaluate the entry module, call
//! its exported `main(args)`", so it shares the runtime plumbing (print
//! channel, `execute`, result rendering) with `eval_javascript` while never
//! inheriting an agent's globals. Each call runs on its own thread with a
//! deadline; a timed-out worker is abandoned rather than reused.

use std::{
    collections::HashMap,
    panic::AssertUnwindSafe,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread,
    time::Duration,
};

use bevy::prelude::{Component, Entity, Res, ResMut, Resource};
use serde_json::Value;

use crate::{
    js_runtime::{HostCommand, HostDispatch, JsSession},
    tools::{manifest, tree::ToolsTree},
    virtual_fs::VirtualFs,
};

/// Deadline for a script call when the manifest does not set `timeout_ms`.
pub(crate) const DEFAULT_SCRIPT_TIMEOUT: Duration = Duration::from_secs(30);
/// Concurrent script calls; beyond this, calls fail fast instead of queueing.
const MAX_CONCURRENT_SCRIPTS: usize = 32;
/// Keeps script request ids disjoint from agent-interpreter ids.
const SCRIPT_REQUEST_TAG: u64 = 1 << 63;

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

/// Immutable snapshot of `.tools` handed to worker threads.
///
/// A snapshot per call gives one consistent module graph even if the tree is
/// edited mid-call; edits apply to the next call.
#[derive(Default)]
pub(crate) struct ToolSources {
    pub revision: u64,
    files: HashMap<String, Arc<str>>,
    /// `directory -> entry module` for the import resolver.
    entries: HashMap<String, String>,
}

impl ToolSources {
    pub(crate) fn build(tree: &ToolsTree) -> Self {
        Self {
            revision: tree.revision(),
            files: tree
                .iter()
                .map(|(path, content)| (path.to_owned(), Arc::<str>::from(content)))
                .collect(),
            entries: manifest::entry_index(tree),
        }
    }

    pub(crate) fn get(&self, key: &str) -> Option<&str> {
        self.files.get(key).map(|source| &**source)
    }

    /// Keys, for diagnostics and tests.
    #[allow(dead_code, reason = "used by resolver diagnostics and tests")]
    pub(crate) fn contains(&self, key: &str) -> bool {
        self.files.contains_key(key)
    }
}

/// Keeps one published [`ToolSources`] in sync with the tree.
#[derive(Resource, Default)]
pub(crate) struct ToolSourceCache(pub Arc<ToolSources>);

pub(crate) fn refresh_tool_sources(vfs: Res<VirtualFs>, mut cache: ResMut<ToolSourceCache>) {
    let revision = vfs.tools().revision();
    if cache.0.revision != revision {
        cache.0 = Arc::new(ToolSources::build(vfs.tools()));
    }
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

/// Resolve a module specifier to a `.tools` key.
///
/// Relative specifiers resolve against the referrer's directory; bare ones are
/// a directory (`tool.json`/`library.json` entry, then `index.js`/`main.js`) or
/// a file. Nothing may escape the mount.
pub(crate) fn resolve_module(
    specifier: &str,
    referrer: Option<&Path>,
    sources: &ToolSources,
) -> Result<String, String> {
    if specifier.is_empty() {
        return Err("empty module specifier".to_owned());
    }
    if specifier.starts_with('/') || specifier.contains('\\') {
        return Err(format!("module specifier `{specifier}` must be relative"));
    }

    let candidate = if specifier.starts_with("./") || specifier.starts_with("../") {
        let referrer = referrer
            .and_then(Path::to_str)
            .ok_or_else(|| format!("relative import `{specifier}` has no referrer path"))?;
        let directory = referrer
            .rsplit_once('/')
            .map(|(directory, _)| directory)
            .unwrap_or("");
        normalize(&format!("{directory}/{specifier}"))?
    } else {
        normalize(specifier)?
    };

    resolve_key(&candidate, sources).ok_or_else(|| {
        let mut tried = vec![candidate.clone()];
        if let Some(entry) = sources.entries.get(&candidate) {
            tried.push(entry.clone());
        }
        tried.push(format!("{candidate}/index.js"));
        tried.push(format!("{candidate}/main.js"));
        if !candidate.ends_with(".js") {
            tried.push(format!("{candidate}.js"));
        }
        format!(
            "cannot resolve module `{specifier}` under `.tools` (tried {})",
            tried.join(", ")
        )
    })
}

fn resolve_key(candidate: &str, sources: &ToolSources) -> Option<String> {
    if sources.contains(candidate) {
        return Some(candidate.to_owned());
    }
    if let Some(entry) = sources.entries.get(candidate) {
        return Some(entry.clone());
    }
    for name in ["index.js", "main.js"] {
        let key = format!("{candidate}/{name}");
        if sources.contains(&key) {
            return Some(key);
        }
    }
    let key = format!("{candidate}.js");
    sources.contains(&key).then_some(key)
}

fn normalize(path: &str) -> Result<String, String> {
    let mut segments: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => continue,
            ".." => {
                if segments.pop().is_none() {
                    return Err(format!("module path `{path}` escapes `.tools`"));
                }
            }
            other => segments.push(other),
        }
    }
    if segments.is_empty() {
        return Err("module path resolves to the `.tools` root".to_owned());
    }
    Ok(segments.join("/"))
}

// ---------------------------------------------------------------------------
// Script tool component
// ---------------------------------------------------------------------------

/// Execution data for one script tool entity.
#[derive(Component, Clone, Debug)]
pub(crate) struct ScriptTool {
    /// Entry module relative to `.tools` (`tool1/main.js`).
    pub entry: String,
    /// Exported function the host calls with the parsed arguments object.
    pub export: String,
    pub timeout_ms: Option<u64>,
}

// ---------------------------------------------------------------------------
// Thread runtime
// ---------------------------------------------------------------------------

pub(crate) struct ScriptTask {
    pub sources: Arc<ToolSources>,
    pub entry: String,
    pub export: String,
    pub arguments: Value,
    pub timeout: Duration,
    /// Nesting below the model's root call; script nested calls add one.
    pub depth: u32,
    pub root: u64,
}

pub(crate) struct ScriptHostRequest {
    pub agent: Entity,
    pub command: HostCommand,
    pub reply: Sender<Result<String, String>>,
}

pub(crate) enum ScriptEvent {
    Started {
        request: u64,
        at: std::time::Instant,
    },
    Done {
        agent: Entity,
        request: u64,
        result: Result<String, String>,
    },
}

/// Owns the script workers' channels and the concurrency budget.
#[derive(Resource)]
pub(crate) struct ScriptRuntime {
    host_tx: Sender<ScriptHostRequest>,
    host_requests: Mutex<Receiver<ScriptHostRequest>>,
    event_tx: Sender<ScriptEvent>,
    events: Mutex<Receiver<ScriptEvent>>,
    next_request: AtomicU64,
    live: Arc<AtomicUsize>,
}

impl Default for ScriptRuntime {
    fn default() -> Self {
        let (host_tx, host_requests) = mpsc::channel();
        let (event_tx, events) = mpsc::channel();
        Self {
            host_tx,
            host_requests: Mutex::new(host_requests),
            event_tx,
            events: Mutex::new(events),
            next_request: AtomicU64::new(0),
            live: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl ScriptRuntime {
    pub(crate) fn drain_host_requests(&self) -> Vec<ScriptHostRequest> {
        self.host_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .try_iter()
            .collect()
    }

    pub(crate) fn drain_events(&self) -> Vec<ScriptEvent> {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .try_iter()
            .collect()
    }

    /// Start one script call. Returns the request id to track in `ToolRounds`.
    pub(crate) fn submit(&self, agent: Entity, task: ScriptTask) -> Result<u64, String> {
        let live = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        if live > MAX_CONCURRENT_SCRIPTS {
            self.live.fetch_sub(1, Ordering::SeqCst);
            return Err(format!(
                "too many script tools are running ({MAX_CONCURRENT_SCRIPTS} max); try again shortly"
            ));
        }

        let request = SCRIPT_REQUEST_TAG | self.next_request.fetch_add(1, Ordering::Relaxed);
        let done = Arc::new(AtomicBool::new(false));
        let deadline_done = done.clone();
        // Set when the deadline fires; blocked host calls notice and unwind.
        let expired = Arc::new(AtomicBool::new(false));
        let expired_worker = expired.clone();
        let expired_watchdog = expired.clone();
        // Signals the watchdog that the worker finished; dropping it (panic or
        // return) also counts as finished.
        let (finished_tx, finished_rx) = mpsc::channel::<()>();
        let event_tx = self.event_tx.clone();
        let host_tx = self.host_tx.clone();
        let live_counter = self.live.clone();
        let timeout = task.timeout;

        let _ = self.event_tx.send(ScriptEvent::Started {
            request,
            at: std::time::Instant::now(),
        });

        let spawned = thread::Builder::new()
            .name(format!("tool-script-{request:x}"))
            .spawn(move || {
                let dispatch = ScriptHost {
                    agent,
                    host: host_tx,
                    expired: expired_worker,
                };
                // A Boa panic must not strand the round or consume a slot.
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    JsSession::new_tools(
                        std::rc::Rc::new(dispatch),
                        task.sources,
                        task.depth,
                        task.root,
                    )
                    .and_then(|mut session| {
                        session.eval_module_call(&task.entry, &task.export, &task.arguments)
                    })
                }))
                .unwrap_or_else(|_| Err("script tool panicked".to_owned()));
                if done
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    let _ = event_tx.send(ScriptEvent::Done {
                        agent,
                        request,
                        result,
                    });
                }
                let _ = finished_tx.send(());
                live_counter.fetch_sub(1, Ordering::SeqCst);
            });

        if let Err(err) = spawned {
            self.live.fetch_sub(1, Ordering::SeqCst);
            return Err(format!("could not start a script worker: {err}"));
        }

        // Deadline watchdog: the worker keeps its slot until Boa's loop limit
        // or the host call returns, so a runaway script cannot be reused but
        // also cannot silently hold the round forever.
        let done = deadline_done;
        let event_tx = self.event_tx.clone();
        let deadline = thread::Builder::new()
            .name(format!("tool-script-deadline-{request:x}"))
            .spawn(move || {
                // Exits as soon as the call finishes, so at most the number of
                // in-flight scripts (not the number of calls) have watchdogs.
                match finished_rx.recv_timeout(timeout) {
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {
                        if done
                            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                            .is_ok()
                        {
                            let _ = event_tx.send(ScriptEvent::Done {
                                agent,
                                request,
                                result: Err("script tool stopped unexpectedly".to_owned()),
                            });
                        }
                        return;
                    }
                    Ok(()) => return,
                }
                expired_watchdog.store(true, Ordering::SeqCst);
                if done
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    let _ = event_tx.send(ScriptEvent::Done {
                        agent,
                        request,
                        result: Err(format!(
                            "script tool timed out after {} ms",
                            timeout.as_millis()
                        )),
                    });
                }
            });
        if let Err(err) = deadline {
            bevy::log::warn!("could not start a script deadline watchdog: {err}");
        }

        Ok(request)
    }
}

/// Blocking host bridge used by `execute()` and the module loader inside a
/// script worker.
struct ScriptHost {
    agent: Entity,
    host: Sender<ScriptHostRequest>,
    expired: Arc<AtomicBool>,
}

impl HostDispatch for ScriptHost {
    fn call(&self, command: HostCommand) -> Result<String, String> {
        if self.expired.load(Ordering::SeqCst) {
            return Err("script tool timed out".to_owned());
        }
        let (reply, wait) = mpsc::channel();
        self.host
            .send(ScriptHostRequest {
                agent: self.agent,
                command,
                reply,
            })
            .map_err(|_| "the harness stopped accepting tool calls".to_owned())?;
        // Poll the deadline so a timed-out worker cannot block here forever.
        loop {
            match wait.recv_timeout(Duration::from_millis(100)) {
                Ok(result) => return result,
                Err(RecvTimeoutError::Timeout) => {
                    if self.expired.load(Ordering::SeqCst) {
                        return Err("script tool timed out".to_owned());
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("the harness stopped before answering the tool call".to_owned());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::js_runtime::JsSession;

    fn sources(files: &[(&str, &str)]) -> ToolSources {
        let mut sources = ToolSources::default();
        for (path, content) in files {
            sources
                .files
                .insert((*path).to_owned(), Arc::<str>::from(*content));
        }
        sources
            .entries
            .insert("tool2".into(), "tool2/main.js".into());
        sources
    }

    #[test]
    fn relative_specifiers_resolve_against_the_referrer() {
        let sources = sources(&[
            ("tool1/main.js", ""),
            ("tool1/lib/x.js", ""),
            ("shared/y.js", ""),
        ]);
        assert_eq!(
            resolve_module("./lib/x.js", Some(Path::new("tool1/main.js")), &sources).unwrap(),
            "tool1/lib/x.js"
        );
        assert_eq!(
            resolve_module(
                "../../shared/y.js",
                Some(Path::new("tool1/lib/x.js")),
                &sources
            )
            .unwrap(),
            "shared/y.js"
        );
        assert!(
            resolve_module(
                "../../../escape.js",
                Some(Path::new("tool1/lib/x.js")),
                &sources
            )
            .is_err()
        );
        assert!(resolve_module("./x.js", None, &sources).is_err());
    }

    #[test]
    fn script_modules_import_relative_files() {
        let sources = Arc::new(sources(&[
            (
                "tool1/main.js",
                "import { value } from './lib/x.js'; export function main(args) { return `${args.prefix}:${value}`; }",
            ),
            ("tool1/lib/x.js", "export const value = 42;"),
        ]));
        let mut session = JsSession::new_tools_isolated(sources).unwrap();
        let result = session
            .eval_module_call(
                "tool1/main.js",
                "main",
                &serde_json::json!({ "prefix": "ok" }),
            )
            .unwrap();
        assert_eq!(result, "ok:42");
    }

    #[test]
    fn script_calls_get_fresh_contexts() {
        let sources = Arc::new(sources(&[(
            "demo/main.js",
            "globalThis.leaked = (globalThis.leaked ?? 0) + 1; export function main() { return String(globalThis.leaked); }",
        )]));
        let mut first = JsSession::new_tools_isolated(sources.clone()).unwrap();
        assert_eq!(
            first
                .eval_module_call("demo/main.js", "main", &Value::Null)
                .unwrap(),
            "1"
        );
        let mut second = JsSession::new_tools_isolated(sources).unwrap();
        assert_eq!(
            second
                .eval_module_call("demo/main.js", "main", &Value::Null)
                .unwrap(),
            "1"
        );
    }

    #[test]
    fn bare_specifiers_use_manifests_and_conventions() {
        let sources = sources(&[
            ("tool1/main.js", ""),
            ("tool2/main.js", ""),
            ("lib/index.js", ""),
            ("flat.js", ""),
        ]);
        assert_eq!(
            resolve_module("tool2", None, &sources).unwrap(),
            "tool2/main.js"
        );
        assert_eq!(
            resolve_module("lib", None, &sources).unwrap(),
            "lib/index.js"
        );
        assert_eq!(resolve_module("flat", None, &sources).unwrap(), "flat.js");
        assert!(resolve_module("missing", None, &sources).is_err());
        assert!(resolve_module("../tool1", None, &sources).is_err());
    }
}
