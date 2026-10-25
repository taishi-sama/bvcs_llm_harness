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

## Schedules

`main.rs` registers two chained `Update` groups, in this order:

```
PreStartup:
  seed_tool_tree → register_static_tools → register_script_tools

Startup:
  setup (UI), log_glyph_cache_limit

Update group A (turn/input path):
  create_requested_agents, sync_retry_buttons, cancel_agent_turn,
  retry_agent_turn, submit_input, deliver_pending_messages,
  decode_stream_chunks, handle_stream_events, handle_turn_finished

Update group B (async/tool path):
  refresh_tool_sources, poll_js_events, poll_script_events,
  handle_task_result, poll_js_host_requests, poll_script_host_requests,
  execute_tool_calls, auto_background_long_calls, apply_tool_results,
  update_tool_activity, flush_tool_mirror, handle_send_agent_message,
  reconcile_script_tools, start_agent_turn, handle_agent_failed,
  refresh_transcripts, rename_agents, remove_agents

PostUpdate:
  auto_scroll (after UI layout)
```

The two groups are chained as well, so a turn finishing in group A dispatches
its tools in group B in the same frame.

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

## Module map

```
src/main.rs         app wiring, schedules, brand of the UI
src/agent.rs        Agent component, turn loop, tool dispatch/completion
src/config.rs       provider selection and AppConfig
src/transport.rs    HTTP streaming + provider request bodies
src/messages.rs     Bevy message protocol
src/virtual_fs.rs   flat root VFS + .tools mount
src/vfs_path.rs     path parsing/validation
src/js_runtime.rs   Boa contexts, execute/module plumbing
src/tools.rs        tool definitions + inline dispatch
src/tools/          registry, catalog, world, schedule, fs, agents,
                    search, script, manifest, tree, mirror, rescan
src/transcript.rs   chat cards, collapsibles, tabs, stats, tool activity
src/tool_display.rs display-only tool-call naming and result presentation
src/shell.rs        top-level layout and VFS Save/Load
crates/llm_api      provider adapters, streaming decoders, schemas, usage
crates/llm_history_model  Entry/history model shared with providers
```

See `docs/tool-calls.md` for dispatch and lifetimes, and the topic docs beside
it for the subsystems above.
