# Tool calls: how they work and how to add one

This note describes the current tool-call architecture so a future agent can
extend it without re-deriving it from the code. File paths are relative to the
repository root.

## 1. The model of a tool

A tool is an **ECS entity**, not an entry in a central map.

| Piece | Type | Where |
| --- | --- | --- |
| Definition (name, description, params) | `ToolEntry.definition: llm_api::Tool` | `src/tools/registry.rs` |
| How it runs | `ToolEntry.execution: Execution` | `src/tools/registry.rs` |
| Script execution data | `ScriptTool { entry, export, fingerprint }` | `src/tools/script.rs` |
| Advertised to the model | `Advertised` marker | `src/tools/registry.rs` |
| Registered but never callable | `DisabledTool { reason }` | `src/tools/registry.rs` |
| Model-only (not callable from JS/scripts) | `ModelOnly` marker | `src/tools/registry.rs` |
| Discovery policy | `ToolEntry.availability: Availability` | `src/tools/registry.rs` |
| Default auto-background participation | `ToolEntry.auto_background: bool` | `src/tools/registry.rs` |

`Execution` has five cases:

```rust
enum Execution {
    Inline(ToolFn),   // fn(&mut ToolWorld, &Value, &CallContext) -> (String, bool)
    Interpreter,      // submitted to the agent's persistent Boa thread
    Script,           // submitted to a fresh .tools Boa context
    Gateway,          // the model-facing `execute` tool
    Missing,          // definition with no handler; rejected by tests
}
```

Definitions live in exactly one place: `definitions()` in `src/tools.rs`.
Exposure policy lives in `src/tools/catalog.rs`:

- `policy(name)` — advertised or manual, `order`, `tags`, `model_only`, and
  `auto_background`
- `execution(name)` — which `Execution` the name maps to
- `is_implemented(name)` — whether `execution` resolved a real handler

Inline handler registration is uniform. Every module that implements
synchronous tools ends with the shared `tool_handlers!` macro from
`src/tools/registration.rs`, which generates that module's
`handler(name) -> Option<ToolFn>` lookup. `catalog::execution` chains the lookups
(`fs`, `agent_tools`, `search`, `rescan`) through `inline_handler` and returns
`Execution::Missing` for a definition that was never wired up. Static and script
tools both spawn through `registry::spawn_tool`, so the `Advertised` and
`ModelOnly` markers cannot drift between them.

Registration runs in `PreStartup` (`src/main.rs`):

```
seed_tool_tree          // read ./.tools into VirtualFs::tools (no mirror writes)
register_static_tools   // one entity per definitions() entry
register_script_tools   // one entity per valid .tools manifest
```

`reconcile_script_tools` then runs every frame but does work only when
`VirtualFs::tools().revision` changed (adds/updates/despawns script tools).

## 2. What the model sees

`start_agent_turn` builds a per-request `ModelConfig`:

```rust
agent.model_config(advertised_tools(&tools))
```

`advertised_tools` queries `With<Advertised>, Without<DisabledTool>`, sorts by
`(order, name)` (stable across archetype changes, which matters for prompt
caching), and clones the definitions. The schema is a snapshot per request:
`reconcile_script_tools` runs after tool calls and before `start_agent_turn`, so
a manifest written or deleted by a tool call is reflected in the next request of
the same turn.

`search_tools` can expose non-advertised tools, but it does not change what is
advertised. The model can still name any registered tool; the harness
dispatches by name.

## 3. Life of one model tool call

```
assistant stream
  → AgentTurnFinished
  → handle_turn_finished
  → RunToolCalls { agent, turn_id, tool_call: Vec<ToolCall> }
  → execute_tool_calls                      (src/agent.rs)
      ├─ parse toolcall_params as JSON     (empty → Null; invalid → error result)
      ├─ root = (round_id << 32) | call_index
      ├─ rounds.open_budget(root, agent)   (2000 nested calls)
      ├─ find(&world, name) → Found { entity, execution, disabled, ... }
      │
      ├─ Execution::Inline     → execute_tool(&mut world, name, args, &CallContext)
      ├─ Execution::Gateway    → dispatch_gateway(...)
      ├─ Execution::Interpreter→ interpreter_task(...) → JsRuntimeHandle::submit
      ├─ Execution::Script     → ScriptRuntime::submit
      └─ Execution::Missing    → "tool has no registered handler" result
      │
      ├─ sync results fill results[call_index]; remaining -= 1
      ├─ async results are tracked:
      │     rounds.track(request_id, PendingCallFields {
      │         round_id, call_index, auto_background
      │     })
      │   or, when auto_background == Some(0):
      │     rounds.detach(request_id, agent, root)  + "started in background" result
      │
      └─ rounds.insert_round(round_id, PendingToolRound { ... })
         finish_round_if_ready(...)
```

`execute_tool` (`src/tools.rs`) is the **synchronous** entry point:

1. reject `depth > MAX_TOOL_DEPTH` (50)
2. `find` → unknown / disabled errors
3. `Execution::Inline(handler)` → call with `ToolWorld` and `CallContext`
4. `Interpreter` / `Script` / `Gateway` reached here are rejected — those are
   dispatched by `execute_tool_calls` (model path) or handled by the gateway.
5. `Execution::Missing` reports `tool has no registered handler`; tests reject
   that state before it can reach the model.

Every handler returns `(String, bool)` where the bool is `is_error`. Handlers
must use `world.commands` for world mutation and messages; they never panic.

## 4. Round completion

`finish_round_if_ready` (in `src/tools/schedule.rs`) runs when
`remaining == 0`:

1. removes the round and releases the budget references for its root calls
2. emits `ToolRoundFinished { agent, turn_id, calls, results }`
3. `apply_tool_results` appends one `Entry::ToolCallResult` per call to history
   (in call order) and continues the loop, or ends the agent if `end_agent`
   was called

Results are stored by `call_index`, so async completions never reorder history.

### Transcript presentation (UI only)

Cards are resolved through `src/tool_display.rs`. `present_call` maps a model
tool call to the effective call shown to the user: an `execute` gateway is
transparent, so `execute(fs_read, …)` renders as a direct `fs_read` call and
result. This is presentation only; history keeps the raw `execute` call and
result, so provider requests and prompt caching are unaffected.

To add a special renderer (for example an edit diff), add a
`ToolPresentation` variant, return it from `present_call`, and branch on it in
`transcript::render_entry`. Never add `Entry` variants for display-only data.

## 5. Async completions and the three request maps

`ToolRounds` (`src/tools/schedule.rs`) owns:

| Map | Meaning | Where it completes |
| --- | --- | --- |
| `pending` | async call that owns a round slot | `complete_tool_request` → `finish_round_if_ready` |
| `detached` | call detached from its round; result becomes a message | `route_async_result` → `TaskResult` |
| `nested` | call made by `execute()` inside JS/script; caller blocks | `route_async_result` → `send_nested_reply` |

Request ids:

- interpreter: one shared `AtomicU64` in `JsChannels`, so ids are unique across
  all agent contexts
- script: `1 << 63 | counter`, guaranteed disjoint from interpreter ids

Completion routing (`route_async_result` in `src/agent.rs`):

```
poll_js_events / poll_script_events
  → route_async_result(request, agent, result)
      ├─ take_detached → release budget ref → TaskResult (background message)
      ├─ take_nested   → send_nested_reply (agent host token or script channel)
      └─ otherwise     → complete_tool_request (round slot)
```

`poll_js_events` drains `JsEvent::{Started, Done}`;
`poll_script_events` drains `ScriptEvent::{Started, Done}`. `Started` records
`started_at`, which is what the auto-background grace is measured from. The
interpreter emits `Started` from the context thread when the task begins; the
script runtime emits it when the worker is submitted.

## 6. Nested calls (`execute` inside JS or a script)

The JavaScript global and script contexts use the same shape:

```
execute(tool_name, args, background_after)
  → HostCommand::Tool { name, arguments, depth, root, background }
  → agent interpreter: HostBridge blocks, main loop drains it in poll_js_host_requests
  → script worker:     ScriptHost blocks on a channel, drained in poll_script_host_requests
  → handle_host_tool(world, rounds, ..., origin, reply)
```

`handle_host_tool` enforces:

- `depth > 50` → error
- `background_after` other than default/never → error
  (**only root calls detach**)
- `ModelOnly` tools → error (scripts/JS cannot call agent lifecycle or the
  interpreter itself)
- `Execution::Gateway` → error (no `execute`→`execute`)
- `Execution::Interpreter` with `origin = AgentInterpreter` → error; that would
  submit to the same thread that is blocked waiting for the reply (deadlock).
  Script origins may call interpreter tools because they are a different thread.
- inline targets: spend one budget unit, call `execute_tool`, reply now
- async targets: spend one budget unit, submit, register a `NestedCall` with a
  `NestedReply` (`Agent { token }` or `Channel`), reply on completion

Nested calls never detach; the caller is blocked until the result arrives.

## 7. Backgrounding (root calls only)

Three inputs resolve to `Option<Duration>` via `resolve_auto_background`:

1. per-call request:
   - JS tools: `run_in_background: bool` (`true` → 0 s, `false` → never)
   - `execute` gateway: `background_after` (`0` → now, `< 0` → never,
     `> 0` → seconds, clamped to 1 ms–24 h)
2. per-tool default: `ToolEntry.auto_background` (true for `run_javascript` /
   `eval_javascript`; false for everything else)
3. global policy: `AutoBackgroundPolicy` (`JS_AUTO_BACKGROUND_MS`, default 5 s)

Outcomes:

- `None` → the round waits normally
- `Some(0)` → detach immediately at dispatch
- `Some(grace)` → `auto_background_long_calls` detaches once
  `now - started_at >= grace`

Detaching fills the round slot with `"Task N running in background..."`,
increments the root budget's reference count, and the eventual result becomes a
`TaskResult` message, which `handle_task_result` queues as a pending message for
the agent. Detach is **not cancellation**: the work keeps
running (bounded by Boa's loop iteration limit, `10_000_000`).

## 8. Budgets and depth

- Depth: model root call = 0. A context created for a call at depth `d` makes
  calls at depth `d + 1`. Limit: `MAX_TOOL_DEPTH = 50`.
- Budget: opening a root call grants `MAX_CALLS_PER_ROOT = 2000` nested calls.
  The root itself does not spend. Every nested call spends one unit before
  submission (refunded if submission fails).
- Lifespan: `Budget { remaining, refs }` where `refs` starts at 1 for the round
  slot and increments per detached task. `finish_round_if_ready` releases the
  slot reference; `route_async_result` releases the task reference when a
  detached result arrives. The budget disappears when `refs` reaches 0, so
  background work keeps calling tools across turns.
- Cancellation: `cancel_rounds(agent)` (Stop, failure) drops that agent's
  rounds, pending calls, and budgets. `remove_agent` also drops detached and
  nested calls. Late results are ignored.

## 9. How to add a tool

### Inline tool (Rust, main loop)

1. Add a `Tool` to `definitions()` in `src/tools.rs`:
   - mark optional parameters `required: false`
   - use `ParameterType::Object` for a generic object argument
2. Write the handler in a domain module (`src/tools/fs.rs`,
   `src/tools/agent_tools.rs`, or a new module):

   ```rust
   fn my_tool(
       world: &mut ToolWorld<'_, '_>,
       arguments: &Value,
       call: &CallContext<'_>,
   ) -> (String, bool) {
       // read arguments, do work, use world.fs / world.commands / messages
       ("done".to_owned(), false)
   }
   ```

3. Register it in that module's `tool_handlers!` block:

   ```rust
   tool_handlers! {
       "my_tool" => my_tool,
   }
   ```

   The shared macro in `src/tools/registration.rs` generates the module's
   `handler(name)` lookup; the handler signature is the same for every module.
4. `catalog::inline_handler` finds it automatically; there is no per-tool
   branch to add in `catalog::execution`.
5. Add policy in `catalog::policy(name)` (`advertises` or `manual`). Chain
   `.model_only()` when scripts and JavaScript must not call the tool (agent
   lifecycle and the interpreter itself), and `.auto_background()` to opt into
   the global background grace period.
6. Tests: `every_definition_has_a_handler`,
   `registration_flags_match_the_policy_table`, plus a handler-level test. If it
   mutates the world, drive it through an `App` test.

### Async tool (own thread or existing runtime)

1. Add a definition and policy as above.
2. Give it an `Execution` variant, or reuse `Interpreter`/`Script`.
3. Submit it from `execute_tool_calls` (model path) or `handle_host_tool`
   (nested path), spending budget first.
4. Return a request id, then `rounds.track(request_id, PendingCallFields { ... })`
   for a round slot, or `rounds.detach(request_id, agent, root)` to detach.
5. Emit completion through a channel drained by a system, and route it through
   `route_async_result` so rounds, nested callers, and background messages are
   handled uniformly.

### Script tool (`.tools`)

Create:

```
.tools/<name>/tool.json   # {"name","description","parameters","entry","export","order"}
.tools/<name>/main.js     # export function main(args) { ... }
```

The reconciler registers it automatically after tool calls and before the next
request. `rescan_tools` (`apply = true`, `source = "vfs"`, or `"disk"` to reload
`./.tools`) reports the diff and asks that reconciler to apply it. Script tools
run in a fresh Boa context per call; imports resolve relative to the referrer,
then bare names via manifest entries or `index.js`/`main.js`. Arguments are
checked against the manifest (required parameters and declared types) before the
call starts.

## 10. Invariants and gotchas

- `definitions()` is the single source of tool metadata; never duplicate a
  schema in a handler.
- Advertisement order must stay `(order, name)`; unsorted tool arrays invalidate
  provider prompt caches.
- Tool schemas are snapshotted per request; registry changes never affect the
  request in flight.
- Only root calls detach. Nested `execute(..., background_after)` is rejected.
- `execute_tool` is for inline tools only. Never call it for `Gateway`,
  `Script`, or `Interpreter` from the model dispatch path.
- Spend the budget **before** submitting async work; refund on submission
  failure.
- Results are written by `call_index`; never append to a round.
- A tool that needs another tool's result must not detach it.
- Script sources are snapshotted (`ToolSources`); an edit applies to the next
  frame/turn, not to a call in the same assistant batch.
- `.tools` writes go through `VfsPath` validation and are mirrored to disk
  best-effort; the in-memory tree is the source of truth.
- `end_agent` queues removal; the round still commits, then the agent is
  despawned.

## 11. Map of the code

| Concern | File |
| --- | --- |
| Definitions | `src/tools.rs` (`definitions`) |
| Exposure policy, registration, gateway definitions | `src/tools/catalog.rs` |
| Uniform handler-registration macro | `src/tools/registration.rs` |
| Tool entity spawn + markers | `src/tools/registry.rs` |
| Handler context | `src/tools/world.rs` |
| Round/budget/detach bookkeeping | `src/tools/schedule.rs` |
| Model dispatch, host bridges, completion routing | `src/agent.rs` |
| Script runtime, module resolver, sources | `src/tools/script.rs` |
| Tool manifests and entry index | `src/tools/manifest.rs` |
| `.tools` tree and mirror | `src/tools/tree.rs`, `src/tools/mirror.rs` |
| Live reconciliation / `rescan_tools` | `src/tools/rescan.rs` |
| `.tools` mount and flat VFS | `src/virtual_fs.rs`, `src/vfs_path.rs` |
| Boa contexts, `execute` global, module loader | `src/js_runtime.rs` |
| Transcript tool-call presentation | `src/tool_display.rs` |

## See also

- `docs/architecture.md` — threads, resources, schedules, messages
- `docs/scheduling.md` — rounds, budgets, detach, cancellation in detail
- `docs/js-runtime.md` — Boa contexts, the `execute` global, module loading
- `docs/tools-folder.md` — `.tools` manifests, imports, mirror, rescan
- `docs/virtual-fs.md` — the two mounts and `fs_*` semantics
- `docs/known-issues.md` — deliberate decisions and remaining gaps
