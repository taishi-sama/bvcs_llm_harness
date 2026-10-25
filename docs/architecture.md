# Architecture

OpenCode-style harness: a Bevy app where every conversation is an entity, the
network lives on Tokio tasks, and every JavaScript heap lives on its own OS
thread. The Bevy main loop never blocks on HTTP, Boa, or the filesystem mirror.

## Threads and ownership

| Thread | Owns | Talks to the main loop via |
| --- | --- | --- |
| Bevy main loop | the entire ECS world: agents, tools, VFS, UI, tool rounds | — |
| Tokio tasks (`bevy_tokio_tasks`) | one HTTP stream per active turn | `StreamChunk` / `StreamEnded` / `StreamFailed` |
| One Boa thread per agent (`js_runtime`) | that agent's persistent `Context` | `JsEvent` (results), `HostRequest` (tool calls / file reads) |
| One thread per script tool call (`tools/script`) | a throwaway `Context` + module graph | `ScriptEvent`, `ScriptHostRequest` |

The rule is absolute: JavaScript heaps and HTTP response bodies are `!Send` or
blocking where it matters, so they are never touched by a Bevy system. The
main loop only drains channels.

## ECS shape

Entities:

- **Agent**: `Agent` (history + provider + turn state) + `AgentView` (UI entity
  handles) + `JsRuntimeHandle` (context thread). Exactly one agent carries
  `ActiveAgent` at a time.
- **Tool**: `ToolEntry` (+ `ScriptTool` for `.tools` tools, + markers
  `Advertised` / `DisabledTool` / `ModelOnly`).
- **UI**: panel, transcript log, cards, collapsible bodies, tabs, buttons.

Key resources:

| Resource | Purpose |
| --- | --- |
| `AppConfig` | provider, model, endpoint, token, session id |
| `AgentIndex` | name → entity, used by cross-agent tools |
| `AgentLayout` | tab/panel roots and the next agent number |
| `AgentRuntime` | active turns and their cancellation handles |
| `VirtualFs` | flat root files + the hierarchical `.tools` tree |
| `VfsStateDir` | directory for the Save/Load VFS buttons |
| `ToolsDir` | `./.tools` mirror root |
| `ToolSourceCache` | immutable `.tools` snapshot for script/module work |
| `ToolRounds` | rounds, pending/detached/nested calls, budgets |
| `AutoBackgroundPolicy` | global auto-background grace |
| `ScriptRuntime` | script worker channels and concurrency budget |
| `ToolRegistryStatus` | last rescan revision/report/broken count |
| `MirrorStatus` | mirror pending/error state |
| `TextCollapseConfig` | transcript collapse thresholds |
| `GlyphCacheLimit` | font atlas budget |
| `JsChannels` | global interpreter channels and request ids |

## Plugins and schedules

`src/main.rs` only assembles domain plugins:

```
ConfigPlugin → MessagesPlugin → VirtualFsPlugin → GlyphCachePlugin →
JsRuntimePlugin → AgentPlugin → ToolsPlugin → TranscriptPlugin → ShellPlugin →
HarnessBrpPlugin → BrpExtrasPlugin
```

Each plugin owns its resources, observers, messages, and systems. Cross-plugin
ordering lives in one place: `AppSet` in `src/sets.rs`, chained once by
`configure_update_sets`. Adding a phase means adding one enum variant, one line
in the chain, and `.in_set(...)` on the systems in that phase. `ShellPlugin`
creates the shared `AgentLayout` that `AgentPlugin` consumes; the BRP plugin
depends on resources owned by `AgentPlugin`/`MessagesPlugin`.

```
PreStartup:
  seed_tool_tree → register_static_tools → register_script_tools

Startup:
  setup (UI), log_glyph_cache_limit

Update:
  Input → Turn → ToolSources → PollEvents → HostRequests → ExecuteTools →
  AutoBackground → ApplyResults → ToolActivity → MirrorFlush →
  SendAgentMessage → ReconcileTools → Resume → Transcripts → AgentMaintenance

PostUpdate:
  auto_scroll (after UiSystems::Layout)
```

`Input` covers agent creation and user input, `Turn` decodes a stream, the tool
phases execute and commit one round, `Resume` starts the next request, and
`Transcripts`/`AgentMaintenance` update the UI after the state settles.

The tool phases run in the same frame as the turn phase that produced them, so a
finished response dispatches its tool calls immediately.

## Messages

`src/messages.rs` is the complete protocol:

- turn lifecycle: `StartAgentTurn`, `NewAgentRequested`, `CancelAgentRequest`,
  `RetryAgentTurn`
- transport: `StreamChunk`, `StreamEnded`, `StreamFailed`, `AgentStreamEvent`,
  `AgentTurnFinished`, `AgentFailed`
- tools: `RunToolCalls`, `ToolRoundFinished`, `TaskResult`
- agents: `SendAgentMessage`, `RemoveAgentRequest`, `RenameAgentRequest`

Requests flow as messages; results are usually written directly into ECS
components or channel-drained buffers to avoid double-buffering.

## Principles

1. **Snapshots per request.** The model-facing tool list and request body are
   built when a request starts; `.tools` sources are snapshotted per script
   call. `reconcile_script_tools` runs after tool calls and before
   `start_agent_turn`, so a registry edit made by a tool call is in the next
   request of the same turn. A script call in the same assistant batch still
   uses the source snapshot taken at the start of that frame.
2. **Single source of truth.** Tool metadata lives only in `definitions()`;
   script tools are entities reconciled from manifests; the `.tools` tree is
   the truth and disk is a best-effort mirror.
3. **No blocking the frame.** Every potentially long operation is a task/thread
   plus a channel. The main loop does bounded bookkeeping per frame.
4. **Errors are data.** Tool handlers return `(message, is_error)`, streams
   produce typed errors, and the model sees recoverable failures as tool
   results rather than panics.
5. **Determinism where it is cached.** Tool order is `(order, name)`; schemas
   sort parameters; history is committed in call order.
6. **Change detection first.** `refresh_transcripts` only visits agents whose
   `Agent` changed; per-frame UI systems compare before writing text/colors so
   unchanged frames do not dirty the UI.

## Module map

```
src/main.rs         plugin assembly, clear color
src/sets.rs         ordered AppSet phases for Update
src/agent.rs        Agent component, state, observers, input, cache diagnostics
src/agent/          turn, stream decoding, tool dispatch, rename/removal
src/brp.rs          harness/* BRP methods for chat control and inspection
src/config.rs       provider selection, AppConfig, ConfigPlugin
src/transport.rs    HTTP streaming + provider request bodies
src/messages.rs     Bevy message protocol
src/virtual_fs.rs   flat root VFS + .tools mount
src/vfs_path.rs     path parsing/validation
src/js_runtime.rs   Boa contexts, execute/module plumbing
src/tools.rs        tool execution, argument helpers, plugin
src/tools/          definitions, registry, catalog, world, schedule, fs,
                    agents, search, script, manifest, tree, mirror, rescan
src/transcript.rs   panels, tabs, collapsibles, text input
src/transcript/render.rs  cards, live updates, scrolling, stats
src/tool_display.rs display-only tool-call naming and result presentation
src/shell.rs        top-level layout and VFS Save/Load
crates/llm_api      provider adapters, streaming decoders, schemas, usage
crates/llm_history_model  Entry/history model shared with providers
```

See `docs/tool-calls.md` for dispatch and lifetimes, and the topic docs beside
it for the subsystems above.
