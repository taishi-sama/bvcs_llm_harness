# Decisions and known gaps

This file records deliberate choices and the remaining rough edges, so a future
agent does not “fix” a decision by accident.

## Deliberate decisions

- **No wall-clock timeouts.** Script and interpreter work is not cancelled by a
  timer. Runaway CPU is bounded by Boa's 10M loop-iteration limit; long work is
  expected to be detached at the root-call level and delivered as a message.
  `tool.json` has no `timeout_ms`.
- **Only root calls detach.** `run_in_background` (JS tools) and
  `background_after` (the `execute` gateway) work at depth 0. Nested
  `execute(..., background_after)` is rejected. This keeps grace tied to
  `Started`, which only root calls track reliably.
- **`execute` is a model-facing gateway.** It is advertised and ModelOnly, and
  dispatches any tool with the same depth/budget/background rules as the
  JavaScript global. Direct tool calls remain possible; the gateway is not the
  only path.
- **Discovery is not access control.** `OnlyManual` tools and script tools are
  not advertised, but the model can name any registered tool. `search_tools`
  makes hidden tools discoverable. There is no per-agent allowlist yet.
- **`.tools` is VFS-first.** The in-memory tree is authoritative; disk is a
  best-effort mirror. Boot seeds with mirroring suppressed. External disk edits
  require `rescan_tools(source: "disk")`.
- **Script tools get a fresh context per call.** This makes calls isolated and
  parallel-safe, and means library module state never persists between calls.
  Persist through the VFS instead.
- **Built-ins beat script tools on name conflicts**, and invalid manifests are
  skipped with a warning rather than aborting registration.
- **Provider strict mode is not targeted.** Optional parameters are emitted as
  JSON Schema optional properties, which OpenAI *strict* function calling
  cannot represent. Compatible/llama.cpp endpoints are the target.

## Known gaps

### Execution

- Running Boa evaluations (agent or script) cannot be interrupted. Cancel/remove
  drops bookkeeping; the thread finishes and its result is discarded. A true
  cancellation would need a subprocess boundary.
- Script concurrency is capped at 32 live calls; beyond that, calls fail fast.
- Script host calls block their worker until the main loop answers. All current
  paths answer (errors included), but a new host command must do the same or
  the worker hangs.
- Detached tasks keep a budget reference until their result is routed. If a
  detached task's result is never produced (e.g. a thread that never returns),
  that budget entry leaks and its slot stays occupied.

### Registration and rescan

- There is no filesystem watcher. Disk edits are picked up at boot or with
  `rescan_tools(source: "disk")`.
- `rescan_tools(apply: false, source: "disk")` records the new revision so the
  automatic reconciler does not apply the dry run. A later unrelated edit
  triggers reconciliation of the earlier dry-run diff too.
- Reconciliation keys by tool name. A rename is a remove + add; a temporary
  name collision is reported and the second manifest is skipped.
- `order` ties are broken by name; a manifest cannot force a position against a
  built-in with the same order.

### `.tools` / VFS

- The mirror does not `fsync` the directory after rename. It is best-effort by
  design; `MirrorStatus.last_error` is the signal.
- Seeding caps files at 1 MiB and skips symlinks and `*.tmp`.
- Empty-directory pruning happens on delete flush; a directory left empty by
  other means stays on disk.
- `fs_ls` prints every `.tools` key. With a very large tree this is noisy; a
  directory-only listing may be preferred later.
- `.tools` data files are seeded and mirrored, but scripts reading them must use
  `fs_read` (there is no host filesystem access inside a script).

### Tooling and UI

- `ToolRegistryStatus` (registered/broken/report) is a resource only; it is not
  shown in the UI yet and is not `Reflect`ed for BRP.
- `CallContext.root` is carried but unused by handlers; it exists for future
  per-call accounting.
- `ToolEntry.tags`, `Found.availability`, `Found.advertised`, and
  `Found.model_only` are partially unused. `model_only` is enforced for
  JavaScript/script origins, but not for direct model calls (intended).
- No per-agent tool visibility or allowlist. Every agent sees the same
  advertised set.
- `execute` and `search_tools` are advertised; the catalog is otherwise
  trimmed. Adjust `catalog::policy` if schema size becomes a problem.
- The transcript presents `execute` as the tool it targets, but nested
  `execute()` calls made inside `run_javascript`/scripts are not shown as
  separate cards.

### Testing

- `execute_tool_calls` and `handle_host_tool` have no direct unit tests. They are
  exercised by app-level tests and by BRP-driven runs against a real model.
  A focused test needs an App with an agent, messages, and `ToolWorld`.
- Script runtime timeouts/panics are covered only indirectly (no watchdog
  exists now; panics are caught and reported).
- `tool_activity_readout_has_stable_wording` tests the label, not the counter
  composition; `running_for` has no dedicated test.

## Security posture

- Paths are validated centrally (`VfsPath`) and the mirror refuses symlinked
  parents. Nothing under `.tools` can reach a host path through a normal write.
- Scripts resolve imports only inside `.tools` and call tools through the
  harness; there is no host filesystem or network API in a script context.
- `read_file` (arbitrary host reads) remains removed; the VFS is the only file
  surface.
- `API_TOKEN` is never placed in a prompt or exposed through a tool result.
- `ModelOnly` prevents scripts from driving agent lifecycle or the interpreter;
  it does not restrict the model itself.
