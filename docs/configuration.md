# Configuration

Configuration is environment-driven; there is no config file. All values are
read at process start (a few are read when a resource is created).

## Harness

| Variable | Default | Meaning |
| --- | --- | --- |
| `PROVIDER` | `responses` | `responses`, `completions`, or `anthropic` |
| `MODEL` | `qwen3.6-35b-a3b` | model name sent in requests |
| `ENDPOINT` | provider default | request URL override |
| `API_TOKEN` | empty | `Authorization: Bearer <token>` when non-empty |
| `SESSION_ID` | generated | `x-opencode-session` / prompt-cache key |
| `VFS_STATE_DIR` | `vfs_state` | directory for the VFS Save/Load buttons (flat root only) |
| `TOOLS_DIR` | `.tools` | on-disk mirror root for the `.tools` tree |
| `JS_AUTO_BACKGROUND_MS` | `5000` | grace before auto-detaching root calls that opt in; `0` disables |
| `COLLAPSE_MAX_CHARS` | `1000` | transcript collapse threshold (chars) |
| `COLLAPSE_MAX_LINES` | `10` | transcript collapse threshold (lines) |
| `FONT_ATLAS_MAX_MB` | `32` | shared glyph atlas budget; `0` disables the limiter |
| `FONT_ATLAS_CHECK_MS` | `250` | atlas size check interval; `0` every frame |
| `FONT_ATLAS_COOLDOWN_MS` | `1000` | minimum delay between atlas clears |
| `RUST_LOG` | Bevy default | log filtering, e.g. `RUST_LOG=info` |

Relative paths (`VFS_STATE_DIR`, `TOOLS_DIR`) resolve against the working
directory. Invalid numeric values fall back to defaults unless a section says
otherwise.

## Where values go

`AppConfig` is inserted at startup and copied into every agent created later
(`create_agent`). It holds provider, model, endpoint, token, and session id.
Provider changes require a restart; per-agent model switching is not exposed in
the UI.

`AutoBackgroundPolicy` is created from `JS_AUTO_BACKGROUND_MS` once. It is only
consulted for tools whose `ToolEntry.auto_background` is true (`run_javascript`
and `eval_javascript`); everything else never auto-detaches unless a root call
passes `run_in_background` / `background_after`.

`ToolsDir` is created from `TOOLS_DIR`, made absolute, and used by the boot
seed and the mirror. The tree lives in memory; the directory is a mirror.

`VfsStateDir` backs the Save/Load buttons. It never includes `.tools`.

## Runtime resources

See `docs/architecture.md` for the full table. The ones with knobs:

- `TextCollapseConfig` — `COLLAPSE_*`
- `GlyphCacheLimit` — `FONT_ATLAS_*`
- `AutoBackgroundPolicy` — `JS_AUTO_BACKGROUND_MS`
- `ToolsDir`, `VfsStateDir` — paths above
- `MirrorStatus`, `ToolRegistryStatus` — read-only health; no env knob

## Networking

The default endpoint is a local llama.cpp server on port 8080 with the
OpenAI-compatible paths. For OpenCode Go:

```sh
ENDPOINT=https://opencode.ai/inference/go/openai/v1/chat/completions \
PROVIDER=completions MODEL=deepseek-v4.1-flash API_TOKEN=... cargo run
```

The connector example uses the same variables.

## BRP

`bevy_brp_extras` is enabled with the default port 15702 (overridable by the
plugin's env var). It exposes world inspection and input simulation, which is
how the app is driven in tests — see `docs/testing-debugging.md`.

## Logging

`RUST_LOG=info` shows startup registration, mirror seeding, and rescan
summaries. Tool manifests that fail validation are logged as
`tool manifest: …` warnings. `history_dumps.jsonl` is written independently of
the log level.
