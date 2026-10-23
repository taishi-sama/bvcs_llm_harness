# Testing and debugging

## Commands

```sh
cargo test                    # unit + App-level system tests
cargo clippy --all-targets    # must stay warning-free
cargo fmt --check
cargo run                     # the app (needs a provider endpoint)
```

The suite is fast (<1 s) because the expensive paths are channel- and
thread-bound, not part of unit tests.

## Test strategy

- Pure logic (path parsing, manifest scanning, resolver, schema, budgets,
  background parsing) is unit-tested next to the code.
- Systems are tested with a small `App`: insert the resources the system needs,
  run `app.update()`, assert on components/resources. See
  `tools/rescan.rs` (reconciliation), `tools/tree.rs` (seed/flush), and the
  registry smoke test in `src/tools.rs`.
- Boa behavior is tested through `JsSession::new_isolated` /
  `new_tools_isolated`, which use a refusing host. That covers imports, fresh
  contexts, `execute` and result rendering without a Bevy app.
- Test helpers live in the module they test; avoid shared fixtures that two
  subsystems can drift from.

When adding a tool, add at least:

1. `every_definition_has_a_handler` still passes (definition → execution).
2. the advertisement trim test reflects your policy choice.
3. a handler or App-level test for the happy path and one error path.

## Driving the app with BRP

The app registers `bevy_brp_extras`. With the MCP tools available:

1. `brp_launch({ target_name: "bvcs_llm_harness", profile: "debug", path: <repo> })`
   — it builds if the binary is stale and returns immediately.
2. Wait until the BRP port listens before sending input; a stale previous
   instance can hold 15702 and the new process will start without a listener.
   Check `brp_status` / `ss -ltn | grep 15702`, and kill the old PID if needed.
3. `brp_extras_screenshot({ path })` to see the UI.
4. `brp_extras_move_mouse({ position: [x, y] })` → `brp_extras_click_mouse` to
   focus the input (the input is near the bottom center).
5. `brp_extras_type_text({ text })` queues **one character per frame**. Wait a
   few seconds before `brp_extras_send_keys({ keys: ["Enter"] })`, or the
   prompt is submitted half-typed.
6. Wait for the local model (a 35B model commonly takes 60–90 s for a tool
   round), then screenshot again.
7. `brp_shutdown({ app_name: "bvcs_llm_harness" })` when done.

BRP launches log to `/tmp/bevy_brp_mcp_bvcs_llm_harness_port15702_*.log`; startup
registration lines (`seeded N tool file(s)`, `registered script tool …`) are
there. `RUST_LOG=info` in the app is the same stream.

Without MCP, the equivalent check is `cargo run` plus reading that log.

## Inspecting state

- `history_dumps.jsonl` (gitignored) records turn commits, request-build
  failures, prompt-cache regressions, and rollbacks, including
  `previous_history`.
- `brp_status`, `world_query`, `world_get_resources`, and
  `brp_extras_get_diagnostics` inspect the live app. Reflected resources are
  visible; most domain types are not `Reflect` yet, so logs are the fallback.
- Tool registry health is in `ToolRegistryStatus` (registered/broken/report).
  It is not surfaced in the UI yet; read it over BRP once it is reflected.

## Debugging a tool call

1. Reproduce the logic in a unit test (arguments → outcome).
2. For integration, launch the app and ask the model explicitly:
   *“Call execute with tool_name X and args …; show the exact result.”*
3. If the model bypasses `execute`, name the tool and print the result; the
   transcript shows the tool-call card and result.
4. Check the log for `tool manifest:` warnings when a `.tools` tool is missing.

## Common pitfalls

- **No listener on 15702** — a previous instance is still exiting; kill it and
  relaunch.
- **Typing swallowed** — the input was not focused or Enter arrived before the
  queue drained.
- **Tool not found after writing `.tools`** — registration is revision-gated
  from the in-memory tree; call `rescan_tools` (or wait for the next edit).
- **Script import fails** — check that every module is reachable from the
  referrer and that the entry is `.js`; the error lists the paths tried.
- **Mirror did not update** — flush is debounced (250 ms) and best-effort;
  check `MirrorStatus.last_error`.
