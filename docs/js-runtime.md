# JavaScript runtime

Boa `Context` is `!Send`, so every context lives on its own OS thread. The main
loop only exchanges messages with it.

## Per-agent context

`attach_js_runtime` (observer on `Insert<JsRuntimeHandle>`) starts a thread that
owns one `JsSession` for the agent's whole lifetime. The session keeps globals
between `eval_javascript` calls, which is why `eval_javascript_reset_context`
exists.

Channels (`JsChannels`, a resource):

| Direction | Type | Contents |
| --- | --- | --- |
| main → thread | `JsCommand::Run { request, task }` | `JsTask::Eval` / `Reset` |
| thread → main | `JsEvent::Started/Done` | start time, result string or error |
| thread → main | `HostRequest { agent, token, command }` | `execute(...)`, module reads |
| main → thread | `HostResponse { token, result }` | reply to a host request |

`JsChannels.next_request` is a single `AtomicU64` shared by every context, so
request ids are unique across agents (tool rounds key on them).

## Task kinds

```rust
enum JsTask {
    Eval { source, depth, root },   // depth/root are used by execute()
    Reset { symbol: Option<String> },
}
```

`depth` and `root` are stored in a `Cell<(u32, u64)>` inside the session and
read by the `execute` native function for every call it makes.

## The `execute` global

```js
execute(tool_name, args, background_after)
```

- `args` is converted with `to_json` (undefined → empty object; objects,
  arrays, primitives supported; symbols/BigInt/cycles are errors).
- `background_after` is read with `as_number()` on purpose: `to_json` maps
  `NaN`/`Infinity` to `null`, which would silently mean “use the default”.
  Non-numeric values throw. `undefined`/`null` mean “use the tool default”.
- The call is forwarded as `HostCommand::Tool { name, arguments, depth, root,
  background }`, and the thread **blocks** until the main loop answers.
- Nested calls never detach; `background_after` is rejected for them.

`print`/`console.log/info/warn/error` append to a per-session channel that is
drained after each evaluation.

## Host bridge

`HostBridge::call` sends a `HostRequest` with a fresh token and blocks on the
per-agent `HostResponse` channel. The main loop drains requests in
`poll_js_host_requests` and answers through `JsRuntimeHandle::respond`, which is
also how `NestedReply::Agent` delivers nested-tool results.

A context thread can therefore never deadlock the main loop: the main loop never
waits for a Boa heap.

## Module loader

`VirtualModuleLoader` has two modes:

- `ModuleRoot::Vfs` (agent contexts): the flat VFS. A specifier is a file name;
  `./foo.js` is accepted as a convenience. Source is fetched through
  `HostCommand::ReadFile`.
- `ModuleRoot::Tools` (script workers): resolves `./`, `../`, and bare names
  against an immutable `ToolSources` snapshot, then parses every module with
  `Source::from_bytes(...).with_path(...)`. The path is what makes nested
  relative imports resolve; a module parsed without it breaks its own imports.

Relative specifiers escape-check against the snapshot root; bare names resolve
to a `tool.json`/`library.json` entry, then `index.js`, then `main.js`, then a
`.js` suffix.

## Reset and symbol dropping

- `eval_javascript_reset_context` rebuilds the whole `Context` through
  `build_context_with` (fresh globals, same dispatch and module root).
- With `symbol`, it checks `typeof`, `delete globalThis.<name>`, then checks
  again. Lexical (`let`/`const`) and non-configurable bindings survive and are
  reported.

## Errors, limits, cancellation

- Evaluation errors become `"JavaScript error: …"` strings; async job errors
  become `"JavaScript async error: …"`.
- A panic inside a context task is caught (`catch_unwind`), reported, and the
  context is reset rather than poisoning the thread.
- `runtime_limits_mut().set_loop_iteration_limit(10_000_000)` is the only bound
  on runaway CPU. There is **no wall-clock timeout** by design; long work is
  expected to be detached (root calls only) and its result delivered as a
  message.
- Worker shutdown is a `JsCommand::Shutdown`; a context cannot be killed while
  evaluating, so removal waits for it to finish.

## Adding a native function

1. Build it with `NativeFunction::from_closure` (see `print`/`execute`).
2. Capture only `Send` data (`mpsc::Sender`, `Rc<dyn HostDispatch>` created
   inside the thread) — never a Boa value.
3. Register with `register_global_builtin_callable(name, arity, function)`.
   `arity` is metadata (the `length` property); calls with other argument counts
   still arrive, so use `args.get(i)` / `get_or_undefined`.
4. If it needs world state, add a `HostCommand` variant and handle it in
   `poll_js_host_requests` / `poll_script_host_requests`.

## Script workers

Script tools (`docs/tools-folder.md`) reuse the same `JsSession`, only with
`ModuleRoot::Tools` and a fresh context per call. Their host requests carry a
`Sender` instead of an agent token, and completions travel through
`ScriptRuntime`/`ScriptEvent` rather than `JsChannels`.
