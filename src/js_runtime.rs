//! Off-thread JavaScript runtimes.
//!
//! Boa's [`Context`] is `!Send`, so each agent gets its own OS thread that owns
//! exactly one context. The Bevy main loop never touches a JavaScript heap.
//!
//! Each context thread has two channel pairs:
//!
//! 1. **Evaluation channel** — the main loop sends [`JsTask::Eval`]/
//!    [`JsTask::Reset`] requests; the thread answers with [`JsEvent::Done`].
//!    This is what keeps `eval_javascript` and friends from blocking a frame.
//! 2. **Host channel** — JavaScript `execute(name, args)` and the virtual
//!    module loader send [`HostCommand`]s to the main loop and block the
//!    context thread until a matching [`HostResponse`] arrives. The main loop
//!    only ever drains these channels, so it can never deadlock on a JS heap.

use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    fmt,
    panic::AssertUnwindSafe,
    path::PathBuf,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::Instant,
};

use bevy::{
    log::error,
    prelude::{Component, Discard, Entity, Insert, On, Query, Resource},
};
use boa_engine::{
    Context as JsContext, JsNativeError, JsResult, JsString, JsValue, Module, NativeFunction,
    Source,
    builtins::promise::PromiseState,
    js_string,
    module::{ModuleLoader, ModuleRequest, Referrer},
};
use serde_json::{Map as JsonMap, Value};

use crate::tools::script::{ToolSources, resolve_module};

/// Where a context's `import` specifiers resolve.
#[derive(Clone)]
pub(crate) enum ModuleRoot {
    /// The flat virtual filesystem (agent contexts).
    Vfs,
    /// The `.tools` tree (script workers).
    Tools(Arc<ToolSources>),
}

// ---------------------------------------------------------------------------
// Public channel types
// ---------------------------------------------------------------------------

/// One piece of work for a context thread. The main loop always keeps the
/// returned request id so it can match the eventual [`JsEvent::Done`].
#[derive(Debug)]
pub(crate) enum JsTask {
    Eval {
        source: String,
        /// Nesting below the model's root call.
        depth: u32,
        root: u64,
    },
    Reset {
        symbol: Option<String>,
    },
}

/// Work entering a context thread.
#[derive(Debug)]
enum JsCommand {
    Run { request: u64, task: JsTask },
    Shutdown,
}

/// Work leaving a context thread.
#[derive(Debug)]
pub(crate) enum JsEvent {
    Started {
        request: u64,
        at: Instant,
    },
    Done {
        agent: Entity,
        request: u64,
        result: Result<String, String>,
    },
}

/// A command that must run on the Bevy main loop because it needs world state
/// (virtual filesystem, agent index, UI commands, ...).
#[derive(Debug)]
pub(crate) enum HostCommand {
    /// `execute(name, arguments, background_after)` from JavaScript.
    Tool {
        name: String,
        arguments: Value,
        depth: u32,
        root: u64,
        /// Third argument: seconds until detach; validated on the main loop.
        background: Option<f64>,
    },
    /// Source load for `import`/`require`-style access to the virtual FS.
    ReadFile { path: String },
}

/// Message sent by a context thread to the main loop.
#[derive(Debug)]
pub(crate) struct HostRequest {
    pub agent: Entity,
    pub token: u64,
    pub command: HostCommand,
}

/// Answer sent by the main loop back to the context thread that sent a
/// [`HostRequest`] with the same `token`.
#[derive(Debug)]
pub(crate) struct HostResponse {
    pub token: u64,
    pub result: Result<String, String>,
}

/// The subset of host behaviour a JavaScript session needs. The real
/// implementation is [`HostBridge`]; tests use a refusing stub.
pub(crate) trait HostDispatch {
    fn call(&self, command: HostCommand) -> Result<String, String>;
}

// ---------------------------------------------------------------------------
// The bridge living on the context thread
// ---------------------------------------------------------------------------

/// Blocking request/response bridge used by native functions and the module
/// loader running inside a context thread.
///
/// Only one request is in flight at a time (a native call cannot return until
/// it has an answer), so a plain blocking `recv` is enough. The small pending
/// queue still keeps late/out-of-order answers from being lost if a future
/// version allows pipelining.
struct HostBridge {
    agent: Entity,
    next_token: Cell<u64>,
    requests: Sender<HostRequest>,
    responses: Receiver<HostResponse>,
    pending: RefCell<VecDeque<HostResponse>>,
}

impl HostBridge {
    fn new(
        agent: Entity,
        requests: Sender<HostRequest>,
        responses: Receiver<HostResponse>,
    ) -> Self {
        Self {
            agent,
            next_token: Cell::new(0),
            requests,
            responses,
            pending: RefCell::new(VecDeque::new()),
        }
    }
}

impl HostDispatch for HostBridge {
    fn call(&self, command: HostCommand) -> Result<String, String> {
        let token = self.next_token.get();
        self.next_token.set(token.wrapping_add(1));

        self.requests
            .send(HostRequest {
                agent: self.agent,
                token,
                command,
            })
            .map_err(|_| "the Bevy main loop is no longer accepting host requests".to_owned())?;

        loop {
            let position = {
                let pending = self.pending.borrow();
                pending.iter().position(|response| response.token == token)
            };
            if let Some(index) = position {
                let response = self
                    .pending
                    .borrow_mut()
                    .remove(index)
                    .expect("index came from position()");
                return response.result;
            }

            let response = self
                .responses
                .recv()
                .map_err(|_| "the Bevy main loop dropped the host response channel".to_owned())?;

            if response.token == token {
                return response.result;
            }

            // A response for a request that was cancelled or timed out. Keep it
            // around so a later matching call can still consume it.
            self.pending.borrow_mut().push_back(response);
        }
    }
}

/// Test-only host that refuses every command.
#[cfg(test)]
pub(crate) struct NoHost;

#[cfg(test)]
impl HostDispatch for NoHost {
    fn call(&self, command: HostCommand) -> Result<String, String> {
        Err(format!(
            "host access is disabled in this test session: {command:?}"
        ))
    }
}

// ---------------------------------------------------------------------------
// Virtual-filesystem module loader
// ---------------------------------------------------------------------------

struct VirtualModuleLoader {
    host: Rc<dyn HostDispatch>,
    root: ModuleRoot,
}

impl fmt::Debug for VirtualModuleLoader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("VirtualModuleLoader")
    }
}

impl ModuleLoader for VirtualModuleLoader {
    async fn load_imported_module(
        self: Rc<Self>,
        referrer: Referrer,
        request: ModuleRequest,
        context: &RefCell<&mut JsContext>,
    ) -> JsResult<Module> {
        let specifier = request.specifier().to_std_string_escaped();
        match &self.root {
            // The virtual filesystem is flat, so a specifier is just a file name.
            // Accept the common `./foo.js` spelling as a convenience.
            ModuleRoot::Vfs => {
                let path = specifier
                    .strip_prefix("./")
                    .unwrap_or(&specifier)
                    .to_owned();

                let source = self
                    .host
                    .call(HostCommand::ReadFile { path })
                    .map_err(|err| JsNativeError::typ().with_message(err))?;

                let mut context = context.borrow_mut();
                Module::parse(Source::from_bytes(&source), None, &mut context)
            }
            // `.tools` modules resolve inside the published snapshot and every
            // returned module keeps its path so nested imports resolve.
            ModuleRoot::Tools(sources) => {
                let resolved = resolve_module(&specifier, referrer.path(), sources)
                    .map_err(|err| JsNativeError::typ().with_message(err))?;
                let source = sources
                    .get(&resolved)
                    .ok_or_else(|| {
                        JsNativeError::typ().with_message(format!(
                            "module `{resolved}` vanished from the `.tools` snapshot"
                        ))
                    })?
                    .to_owned();
                let path = PathBuf::from(&resolved);
                let mut context = context.borrow_mut();
                Module::parse(
                    Source::from_bytes(&source).with_path(&path),
                    None,
                    &mut context,
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// One Boa session (testable without threads)
// ---------------------------------------------------------------------------

/// A Boa context plus its private `print` output channel.
pub(crate) struct JsSession {
    context: JsContext,
    output: Receiver<String>,
    dispatch: Rc<dyn HostDispatch>,
    root: ModuleRoot,
    /// Current (depth, root) read by `execute()`; updated per task.
    origin: Rc<Cell<(u32, u64)>>,
}

impl JsSession {
    pub(crate) fn new(dispatch: Rc<dyn HostDispatch>) -> Result<Self, String> {
        Self::new_with(dispatch, ModuleRoot::Vfs, 0, 0)
    }

    /// Fresh session whose imports resolve inside `.tools`.
    pub(crate) fn new_tools(
        dispatch: Rc<dyn HostDispatch>,
        sources: Arc<ToolSources>,
        depth: u32,
        root_call: u64,
    ) -> Result<Self, String> {
        Self::new_with(dispatch, ModuleRoot::Tools(sources), depth, root_call)
    }

    fn new_with(
        dispatch: Rc<dyn HostDispatch>,
        root: ModuleRoot,
        depth: u32,
        root_call: u64,
    ) -> Result<Self, String> {
        let (context, output, origin) =
            build_context_with(dispatch.clone(), root.clone(), depth, root_call)?;
        Ok(Self {
            context,
            output,
            dispatch,
            root,
            origin,
        })
    }

    /// Depth and root call used by `execute()` during the next evaluation.
    pub(crate) fn set_origin(&self, depth: u32, root: u64) {
        self.origin.set((depth, root));
    }

    /// Convenience constructor for unit tests that never call `execute`.
    #[cfg(test)]
    pub(crate) fn new_isolated() -> Result<Self, String> {
        Self::new(Rc::new(NoHost))
    }

    #[cfg(test)]
    pub(crate) fn new_tools_isolated(sources: Arc<ToolSources>) -> Result<Self, String> {
        Self::new_tools(Rc::new(NoHost), sources, 0, 0)
    }

    pub(crate) fn eval(&mut self, code: &str) -> Result<String, String> {
        // Drop anything left over from an earlier evaluation before running.
        self.output.try_iter().for_each(drop);

        let value = self
            .context
            .eval(Source::from_bytes(code))
            .map_err(|err| format!("JavaScript error: {err}"))?;
        self.context
            .run_jobs()
            .map_err(|err| format!("JavaScript async error: {err}"))?;

        Ok(self.render(value))
    }

    /// Evaluate a `.tools` entry module and call its exported function with the
    /// parsed arguments object.
    ///
    /// This is the script-tool equivalent of resetting the context and running
    /// the module body: fresh globals, same `print`/`execute`/result rendering.
    pub(crate) fn eval_module_call(
        &mut self,
        entry: &str,
        export: &str,
        arguments: &Value,
    ) -> Result<String, String> {
        let ModuleRoot::Tools(sources) = self.root.clone() else {
            return Err("script session without `.tools` sources".to_owned());
        };
        let Some(source) = sources.get(entry).map(str::to_owned) else {
            return Err(format!("missing tool source `{entry}`"));
        };

        self.output.try_iter().for_each(drop);

        let path = PathBuf::from(entry);
        let module = Module::parse(
            Source::from_bytes(&source).with_path(&path),
            None,
            &mut self.context,
        )
        .map_err(|err| format!("could not parse `{entry}`: {err}"))?;

        let promise = module.load_link_evaluate(&mut self.context);
        self.context
            .run_jobs()
            .map_err(|err| format!("JavaScript async error: {err}"))?;
        match promise.state() {
            PromiseState::Fulfilled(_) => {}
            PromiseState::Rejected(reason) => {
                return Err(format!(
                    "`{entry}` failed to evaluate: {}",
                    reason.display()
                ));
            }
            PromiseState::Pending => {
                return Err(format!("`{entry}` did not finish evaluating"));
            }
        }

        let namespace = module.namespace(&mut self.context);
        let function = namespace
            .get(JsString::from(export), &mut self.context)
            .map_err(|err| format!("could not read export `{export}`: {err}"))?;
        let Some(function) = function.as_callable() else {
            return Err(format!("`{entry}` does not export a callable `{export}`"));
        };

        let argument = JsValue::from_json(arguments, &mut self.context)
            .map_err(|err| format!("could not convert tool arguments: {err}"))?;
        let value = function
            .call(&JsValue::undefined(), &[argument], &mut self.context)
            .map_err(|err| format!("`{export}` threw: {err}"))?;
        self.context
            .run_jobs()
            .map_err(|err| format!("JavaScript async error: {err}"))?;

        Ok(self.render(value))
    }

    /// Printed output followed by the last expression, matching `eval`.
    fn render(&mut self, value: JsValue) -> String {
        let rendered = value
            .to_string(&mut self.context)
            .map(|text| text.to_std_string_escaped())
            .unwrap_or_else(|_| value.display().to_string());
        let printed = self.output.try_iter().collect::<Vec<_>>().join("\n");

        match (printed.is_empty(), rendered == "undefined") {
            (true, _) => rendered,
            (false, true) => printed,
            (false, false) => format!("{printed}\n{rendered}"),
        }
    }

    pub(crate) fn reset(&mut self) -> Result<String, String> {
        let (context, output, origin) =
            build_context_with(self.dispatch.clone(), self.root.clone(), 0, 0)?;
        self.context = context;
        self.output = output;
        self.origin = origin;
        Ok("Dropped the whole JavaScript context for this agent. The next evaluation starts fresh."
            .to_owned())
    }

    pub(crate) fn drop_symbol(&mut self, symbol: &str) -> Result<String, String> {
        if !is_js_identifier(symbol) {
            return Err(format!(
                "`{symbol}` is not a single JavaScript identifier. Omit `symbol` to reset the whole context."
            ));
        }

        let before = self.eval(&format!("typeof {symbol}"))?;
        if before == "undefined" {
            return Ok(format!("No symbol `{symbol}` is defined."));
        }

        // `delete` only removes configurable global-object properties. Lexical
        // bindings (`let`/`const`) and non-configurable `var` bindings survive,
        // which the follow-up `typeof` check detects.
        self.eval(&format!("delete globalThis.{symbol}"))?;
        let after = self.eval(&format!("typeof {symbol}"))?;

        if after == "undefined" {
            Ok(format!("Dropped `{symbol}`."))
        } else {
            Ok(format!(
                "Could not drop `{symbol}`: it is a lexical or non-configurable binding. \
                 Reset the whole context instead."
            ))
        }
    }
}

fn is_js_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_' || first == '$') {
        return false;
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '$')
}

/// A fresh context, its print channel, and the shared `execute` origin cell.
type BuiltContext = (JsContext, Receiver<String>, Rc<Cell<(u32, u64)>>);

fn build_context_with(
    dispatch: Rc<dyn HostDispatch>,
    root: ModuleRoot,
    depth: u32,
    root_call: u64,
) -> Result<BuiltContext, String> {
    let (tx, rx) = mpsc::channel();

    // SAFETY: the closure only captures an `mpsc::Sender<String>`, which is not
    // a GC-managed Boa value, so the collector has nothing to trace or move.
    let print = unsafe {
        NativeFunction::from_closure(move |_this, args, context| {
            let line = args
                .iter()
                .map(|value| {
                    value
                        .to_string(context)
                        .map(|text| text.to_std_string_escaped())
                        .unwrap_or_else(|_| value.display().to_string())
                })
                .collect::<Vec<_>>()
                .join(" ");
            let _ = tx.send(line);
            Ok(JsValue::undefined())
        })
    };

    let execute_dispatch = dispatch.clone();
    let origin = Rc::new(Cell::new((depth, root_call)));
    let execute_origin = origin.clone();
    // SAFETY: the closure captures an `Rc<dyn HostDispatch>`. It contains no
    // Boa GC-managed values; all JavaScript values are converted to
    // `serde_json::Value` before crossing the channel.
    let execute = unsafe {
        NativeFunction::from_closure(move |_this, args, context| {
            let name = args
                .first()
                .ok_or_else(|| {
                    JsNativeError::typ()
                        .with_message("execute(name, arguments) requires a tool name")
                })?
                .to_string(context)?
                .to_std_string_escaped();

            let arguments = match args.get(1) {
                Some(value) if !value.is_undefined() => {
                    value.to_json(context)?.unwrap_or(Value::Null)
                }
                _ => Value::Object(JsonMap::new()),
            };

            // Read the number directly: `to_json` maps NaN/infinity to null,
            // which would silently look like "use the default".
            let background = match args.get(2) {
                None => None,
                Some(value) if value.is_undefined() || value.is_null() => None,
                Some(value) => Some(value.as_number().ok_or_else(|| {
                    JsNativeError::typ().with_message(
                        "background_after must be a number of seconds (0 = now, negative = never)",
                    )
                })?),
            };
            let (depth, root) = execute_origin.get();

            match execute_dispatch.call(HostCommand::Tool {
                name,
                arguments,
                depth,
                root,
                background,
            }) {
                Ok(result) => Ok(JsValue::from(js_string!(result))),
                Err(err) => Err(JsNativeError::error().with_message(err).into()),
            }
        })
    };

    let mut context = JsContext::builder()
        .module_loader(Rc::new(VirtualModuleLoader {
            host: dispatch,
            root,
        }))
        .build()
        .map_err(|err| format!("failed to create JavaScript context: {err}"))?;

    context.strict(true);
    // Protect the host from runaway loops and allocation in model-generated code.
    context
        .runtime_limits_mut()
        .set_loop_iteration_limit(10_000_000);
    context
        .register_global_builtin_callable("print".into(), 1, print)
        .map_err(|err| format!("failed to initialize JavaScript runtime: {err}"))?;
    context
        .register_global_builtin_callable("execute".into(), 3, execute)
        .map_err(|err| format!("failed to initialize JavaScript execute(): {err}"))?;
    context
        .eval(Source::from_bytes(
            "globalThis.console = { log: print, info: print, warn: print, error: print };",
        ))
        .map_err(|err| format!("failed to initialize JavaScript console: {err}"))?;

    Ok((context, rx, origin))
}

// ---------------------------------------------------------------------------
// Bevy-side global channels and per-agent handle
// ---------------------------------------------------------------------------

/// Global channel endpoints shared by every context thread.
///
/// The contexts themselves live on their own threads; this resource only owns
/// the receivers that the main loop polls.
#[derive(Resource)]
pub(crate) struct JsChannels {
    host_tx: Sender<HostRequest>,
    host_requests: Mutex<Receiver<HostRequest>>,
    event_tx: Sender<JsEvent>,
    events: Mutex<Receiver<JsEvent>>,
    /// Request ids must be unique across every context because `ToolRounds`
    /// keys pending and background work by them.
    next_request: Arc<AtomicU64>,
}

impl Default for JsChannels {
    fn default() -> Self {
        let (host_tx, host_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        Self {
            host_tx,
            host_requests: Mutex::new(host_rx),
            event_tx,
            events: Mutex::new(event_rx),
            next_request: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl JsChannels {
    pub(crate) fn drain_events(&self) -> Vec<JsEvent> {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .try_iter()
            .collect()
    }

    pub(crate) fn drain_host_requests(&self) -> Vec<HostRequest> {
        self.host_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .try_iter()
            .collect()
    }

    /// Build (but do not start) the runtime handle for one agent.
    pub(crate) fn handle_for(&self, agent: Entity) -> JsRuntimeHandle {
        JsRuntimeHandle::new(agent, self)
    }
}

/// One agent's link to its JavaScript thread.
///
/// Inserting this component starts the thread (see [`start_js_runtime`]);
/// removing it or despawning the entity sends `Shutdown`. Dropping the
/// component also closes both channels, which releases a thread that is blocked
/// inside `execute()`.
#[derive(Component)]
pub(crate) struct JsRuntimeHandle {
    commands: Sender<JsCommand>,
    responses: Sender<HostResponse>,
    /// Consumed exactly once by the `Insert` observer.
    start: Mutex<Option<JsThreadStart>>,
    next_request: Arc<AtomicU64>,
}

struct JsThreadStart {
    agent: Entity,
    commands: Receiver<JsCommand>,
    responses: Receiver<HostResponse>,
    host_requests: Sender<HostRequest>,
    events: Sender<JsEvent>,
}

impl JsRuntimeHandle {
    fn new(agent: Entity, channels: &JsChannels) -> Self {
        let (commands, command_rx) = mpsc::channel();
        let (responses, response_rx) = mpsc::channel();
        Self {
            commands,
            responses,
            start: Mutex::new(Some(JsThreadStart {
                agent,
                commands: command_rx,
                responses: response_rx,
                host_requests: channels.host_tx.clone(),
                events: channels.event_tx.clone(),
            })),
            next_request: Arc::clone(&channels.next_request),
        }
    }

    /// Return the id the resulting [`JsEvent::Done`] will carry.
    pub(crate) fn submit(&self, task: JsTask) -> Result<u64, String> {
        let request = self.next_request.fetch_add(1, Ordering::Relaxed);

        self.commands
            .send(JsCommand::Run { request, task })
            .map_err(|_| "the JavaScript thread is no longer running".to_owned())?;

        Ok(request)
    }

    pub(crate) fn respond(&self, token: u64, result: Result<String, String>) {
        let _ = self.responses.send(HostResponse { token, result });
    }

    fn shutdown(&self) {
        let _ = self.commands.send(JsCommand::Shutdown);
    }

    fn take_start(&self) -> Option<JsThreadStart> {
        self.start
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}

/// Start an agent context when its [`JsRuntimeHandle`] is inserted.
pub(crate) fn start_js_runtime(
    insert: On<Insert<JsRuntimeHandle>>,
    handles: Query<&JsRuntimeHandle>,
) {
    let Ok(handle) = handles.get(insert.entity) else {
        return;
    };
    spawn_context(handle);
}

fn spawn_context(handle: &JsRuntimeHandle) {
    let Some(start) = handle.take_start() else {
        return;
    };

    let agent = start.agent;
    if let Err(err) = thread::Builder::new()
        .name(format!("js-context-{agent}"))
        .spawn(move || run_context_thread(start))
    {
        error!("failed to spawn JavaScript thread for {agent}: {err}");
    }
}

/// Stop the interpreter when its component is discarded.
///
/// `Discard` fires before removal, before a replacement, and before despawn,
/// so this one observer covers all three cases. The replacement's `Insert`
/// observer then starts a fresh thread.
pub(crate) fn stop_js_runtime(
    discard: On<Discard<JsRuntimeHandle>>,
    handles: Query<&JsRuntimeHandle>,
) {
    if let Ok(handle) = handles.get(discard.entity) {
        handle.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Context thread
// ---------------------------------------------------------------------------

fn run_context_thread(start: JsThreadStart) {
    let JsThreadStart {
        agent,
        commands,
        responses,
        host_requests,
        events,
    } = start;

    let bridge = Rc::new(HostBridge::new(agent, host_requests, responses));
    let dispatch: Rc<dyn HostDispatch> = bridge;
    let mut session = JsSession::new(dispatch.clone());

    while let Ok(command) = commands.recv() {
        match command {
            JsCommand::Run { request, task } => {
                let outcome = match session.as_mut() {
                    Ok(active) => {
                        // Tell the main loop when execution actually begins so
                        // queued requests are not auto-backgrounded early.
                        let _ = events.send(JsEvent::Started {
                            request,
                            at: Instant::now(),
                        });
                        Some(std::panic::catch_unwind(AssertUnwindSafe(|| {
                            run_task(active, task)
                        })))
                    }
                    Err(_) => None,
                };

                let result = match outcome {
                    Some(Ok(result)) => result,
                    Some(Err(_)) => {
                        // A panic can leave the Boa heap inconsistent. Drop it
                        // and create a clean context for the next evaluation.
                        session = JsSession::new(dispatch.clone());
                        Err("JavaScript evaluation panicked; the context was reset".to_owned())
                    }
                    None => match &session {
                        Err(err) => Err(err.clone()),
                        Ok(_) => unreachable!("outcome is None only when session is Err"),
                    },
                };

                if events
                    .send(JsEvent::Done {
                        agent,
                        request,
                        result,
                    })
                    .is_err()
                {
                    break;
                }
            }
            JsCommand::Shutdown => break,
        }
    }
}

fn run_task(session: &mut JsSession, task: JsTask) -> Result<String, String> {
    match task {
        JsTask::Eval {
            source,
            depth,
            root,
        } => {
            session.set_origin(depth, root);
            session.eval(&source)
        }
        JsTask::Reset { symbol } => match symbol.as_deref().map(str::trim) {
            Some(symbol) if !symbol.is_empty() => session.drop_symbol(symbol),
            _ => session.reset(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::World;
    use std::time::{Duration, Instant};

    fn wait_for_result(channels: &JsChannels, request: u64) -> Option<Result<String, String>> {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            for event in channels.drain_events() {
                if let JsEvent::Done {
                    request: id,
                    result,
                    ..
                } = event
                    && id == request
                {
                    return Some(result);
                }
            }
            thread::sleep(Duration::from_millis(5));
        }
        None
    }

    #[test]
    fn javascript_state_persists_in_a_session() {
        let mut session = JsSession::new_isolated().unwrap();

        assert_eq!(session.eval("var counter = 41; counter").unwrap(), "41");
        assert_eq!(session.eval("counter + 1").unwrap(), "42");
        assert_eq!(
            session.eval("console.log('hello'); counter * 2").unwrap(),
            "hello\n82"
        );
    }

    #[test]
    fn reset_starts_fresh() {
        let mut session = JsSession::new_isolated().unwrap();
        session.eval("var value = 7").unwrap();
        session.reset().unwrap();
        assert_eq!(session.eval("typeof value").unwrap(), "undefined");
    }

    #[test]
    fn configurable_global_can_be_dropped() {
        let mut session = JsSession::new_isolated().unwrap();
        session.eval("globalThis.value = 7").unwrap();
        let message = session.drop_symbol("value").unwrap();
        assert!(message.contains("Dropped"));
        assert_eq!(session.eval("typeof value").unwrap(), "undefined");
    }

    #[test]
    fn lexical_binding_cannot_be_dropped_individually() {
        let mut session = JsSession::new_isolated().unwrap();
        session.eval("let value = 7").unwrap();
        let message = session.drop_symbol("value").unwrap();
        assert!(message.contains("Could not drop"));
        assert_eq!(session.eval("value").unwrap(), "7");
    }

    #[test]
    fn context_thread_answers_eval_requests() {
        let channels = JsChannels::default();
        let agent = Entity::PLACEHOLDER;
        let handle = channels.handle_for(agent);
        spawn_context(&handle);

        let request = handle
            .submit(JsTask::Eval {
                source: "6 * 7".to_owned(),
                depth: 0,
                root: 0,
            })
            .unwrap();

        assert_eq!(wait_for_result(&channels, request).unwrap().unwrap(), "42");
    }

    #[test]
    fn inserting_the_component_starts_the_runtime() {
        let mut world = World::new();
        world.insert_resource(JsChannels::default());
        world.add_observer(start_js_runtime);

        let agent = world.spawn_empty().id();
        let handle = world.resource::<JsChannels>().handle_for(agent);
        world.entity_mut(agent).insert(handle);

        let request = world
            .get::<JsRuntimeHandle>(agent)
            .unwrap()
            .submit(JsTask::Eval {
                source: "40 + 2".to_owned(),
                depth: 0,
                root: 0,
            })
            .unwrap();

        let channels = world.resource::<JsChannels>();
        assert_eq!(wait_for_result(channels, request).unwrap().unwrap(), "42");
    }

    #[test]
    fn replacing_the_component_restarts_the_runtime() {
        let mut world = World::new();
        world.insert_resource(JsChannels::default());
        world.add_observer(start_js_runtime);
        world.add_observer(stop_js_runtime);

        let agent = world.spawn_empty().id();
        let handle = world.resource::<JsChannels>().handle_for(agent);
        world.entity_mut(agent).insert(handle);

        let first = world
            .get::<JsRuntimeHandle>(agent)
            .unwrap()
            .submit(JsTask::Eval {
                source: "var value = 7; value".to_owned(),
                depth: 0,
                root: 0,
            })
            .unwrap();
        let channels = world.resource::<JsChannels>();
        assert_eq!(wait_for_result(channels, first).unwrap().unwrap(), "7");

        // Replacing the component must shut down the old context and start a
        // fresh one from the new handle's `start` payload.
        let replacement = world.resource::<JsChannels>().handle_for(agent);
        world.entity_mut(agent).insert(replacement);

        let second = world
            .get::<JsRuntimeHandle>(agent)
            .unwrap()
            .submit(JsTask::Eval {
                source: "typeof value".to_owned(),
                depth: 0,
                root: 0,
            })
            .unwrap();
        let channels = world.resource::<JsChannels>();
        assert_eq!(
            wait_for_result(channels, second).unwrap().unwrap(),
            "undefined"
        );
    }

    #[test]
    fn execute_round_trips_through_the_host_channel() {
        let channels = JsChannels::default();
        let agent = Entity::PLACEHOLDER;
        let handle = channels.handle_for(agent);
        spawn_context(&handle);

        let request = handle
            .submit(JsTask::Eval {
                source: "execute('get_weather', { city: 'Paris' })".to_owned(),
                depth: 0,
                root: 0,
            })
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        let mut result = None;
        while Instant::now() < deadline && result.is_none() {
            for host_request in channels.drain_host_requests() {
                assert_eq!(host_request.agent, agent);
                match host_request.command {
                    HostCommand::Tool {
                        name, arguments, ..
                    } => {
                        assert_eq!(name, "get_weather");
                        assert_eq!(arguments["city"], "Paris");
                        handle.respond(host_request.token, Ok("Paris: sunny".to_owned()));
                    }
                    other => handle.respond(
                        host_request.token,
                        Err(format!("unexpected host command: {other:?}")),
                    ),
                }
            }

            for event in channels.drain_events() {
                if let JsEvent::Done {
                    request: id,
                    result: event_result,
                    ..
                } = event
                    && id == request
                {
                    result = Some(event_result);
                }
            }
            thread::sleep(Duration::from_millis(5));
        }

        assert_eq!(result.unwrap().unwrap(), "Paris: sunny");
    }

    struct FsHost;

    impl HostDispatch for FsHost {
        fn call(&self, command: HostCommand) -> Result<String, String> {
            match command {
                HostCommand::ReadFile { path } if path == "mod.js" => {
                    Ok("export const value = 42;".to_owned())
                }
                HostCommand::ReadFile { path } => Err(format!("no such file: {path}")),
                other => Err(format!("unexpected host command: {other:?}")),
            }
        }
    }

    #[test]
    fn module_loader_reads_from_the_virtual_filesystem() {
        let mut session = JsSession::new(Rc::new(FsHost)).unwrap();
        let output = session
            .eval("import('./mod.js').then(m => print(m.value));")
            .unwrap();
        assert!(output.contains("42"), "{output}");
    }
}

#[cfg(test)]
mod moved_tests {
    use super::*;
    use crate::virtual_fs::VirtualFs;

    #[test]
    fn virtual_javascript_file_runs_in_agent_context() {
        let mut fs = VirtualFs::default();
        fs.write(
            "script.js",
            "var from_file = 6 * 7; print('ran'); from_file".into(),
        )
        .unwrap();
        let source = fs.read("script.js").unwrap();

        let mut session = JsSession::new_isolated().unwrap();
        assert_eq!(session.eval(&source).unwrap(), "ran\n42");
        // The file runs in the same persistent context as eval_javascript.
        assert_eq!(session.eval("from_file").unwrap(), "42");
    }

    #[test]
    fn javascript_output_is_isolated_per_context() {
        let mut first = JsSession::new_isolated().unwrap();
        let mut second = JsSession::new_isolated().unwrap();

        assert_eq!(first.eval("console.log('first')").unwrap(), "first");
        assert_eq!(second.eval("console.log('second')").unwrap(), "second");
        assert_eq!(
            first.eval("console.log('again'); 1 + 1").unwrap(),
            "again\n2"
        );
    }

    #[test]
    fn javascript_state_is_isolated_between_agents() {
        let mut first = JsSession::new_isolated().unwrap();
        let mut second = JsSession::new_isolated().unwrap();

        first.eval("var value = 'first'").unwrap();
        assert_eq!(first.eval("value").unwrap(), "first");
        assert_eq!(second.eval("typeof value").unwrap(), "undefined");
    }
}

#[cfg(test)]
mod id_tests {
    use super::*;

    #[test]
    fn request_ids_are_unique_across_contexts() {
        let channels = JsChannels::default();
        let first = channels.handle_for(Entity::PLACEHOLDER);
        let second = channels.handle_for(Entity::PLACEHOLDER);

        let first_id = first
            .submit(JsTask::Eval {
                source: "1".to_owned(),
                depth: 0,
                root: 0,
            })
            .unwrap();
        let second_id = second
            .submit(JsTask::Eval {
                source: "2".to_owned(),
                depth: 0,
                root: 0,
            })
            .unwrap();

        assert_ne!(first_id, second_id);
    }
}
