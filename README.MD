# DO NOT LOOK, THE PROJECT IS PUBLIC ONLY FOR SAKE OF SHARING IT WITH SOME PEOPLE


Semi-"vibecoded" harness for sake of learning, developing robust LLM provider API and trying various silly ideas.

Obviously, requires set-up Rust toolchain.

```bash
cargo run --release
```

## Environment variables

Names are case-sensitive. Invalid numeric values fall back to the default unless stated otherwise.

### Harness

Used by the main `bvcs_llm_harness` binary (`cargo run`).

| Variable | Default | Description |
| --- | --- | --- |
| `PROVIDER` | `responses` | Streaming wire API: `responses`, `completions`, or `anthropic`. Unknown values use `responses`. |
| `MODEL` | `qwen3.6-35b-a3b` | Model name sent in requests. |
| `ENDPOINT` | Provider-specific local URL | Overrides the server URL. Defaults: `http://127.0.0.1:8080/v1/responses`, `.../v1/chat/completions`, or `.../v1/messages`. |
| `API_TOKEN` | empty | If non-empty, sent as `Authorization: Bearer <token>`. |
| `SESSION_ID` | generated per process | Sent as the `x-opencode-session` header for request routing and prompt-cache stickiness. |
| `VFS_STATE_DIR` | `vfs_state` | Directory used by the VFS Save/Load buttons. Relative paths resolve against the current working directory. |
| `TOOLS_DIR` | `.tools` | On-disk mirror for the `.tools` script-tool tree. Relative paths resolve against the current working directory. |
| `JS_AUTO_BACKGROUND_MS` | `5000` | Grace period before a root tool call that opts in (JavaScript tools by default) is detached into the background. `0` disables automatic backgrounding. |
| `COLLAPSE_MAX_CHARS` | `1000` | Collapse a transcript body when it exceeds this many characters. Must be greater than zero. |
| `COLLAPSE_MAX_LINES` | `10` | Collapse a transcript body when it exceeds this many lines. Must be greater than zero. |
| `FONT_ATLAS_MAX_MB` | `32` | Maximum size of Bevy's shared font glyph atlas in MiB. When exceeded, the atlas is cleared and visible text is re-rasterized. `0` disables the limiter. |
| `FONT_ATLAS_CHECK_MS` | `250` | How often the font atlas size is checked, in milliseconds. `0` checks every frame. |
| `FONT_ATLAS_COOLDOWN_MS` | `1000` | Minimum delay between atlas clears, in milliseconds. `0` allows clearing on every check. |

### Connector example

Used by `crates/llm_connector_example`.

| Variable | Default | Description |
| --- | --- | --- |
| `PROVIDER` | `anthropic` | Streaming wire API: `anthropic`, `completions`, or `responses`. Unknown values use `anthropic`. |
| `MODEL` | `qwen3.6-35b-a3b` | Model name sent in requests. |
| `ENDPOINT` | Provider-specific local URL | Overrides the server URL, using the same defaults as the harness. |
| `API_TOKEN` | empty | If non-empty, sent as `Authorization: Bearer <token>`. |
| `SESSION_ID` | generated per process | Sent as the `x-opencode-session` header. |
| `PROMPT` | `Hi! List all available tools and try guess what they do.` | Initial user message sent by the example. |

### Standard environment

| Variable | Default | Description |
| --- | --- | --- |
| `RUST_LOG` | Bevy default | Controls log filtering through Bevy's `LogPlugin`, e.g. `RUST_LOG=info` or `RUST_LOG=bevy_render=warn`. |


### Critical


## Internal notes

- `docs/architecture.md` — threads, ECS shape, schedules, messages, principles
- `docs/agent-loop.md` — turn lifecycle, history, retries, pending messages
- `docs/tool-calls.md` — dispatch, request lifetimes, budgets, how to add a tool
- `docs/scheduling.md` — rounds, detached/nested calls, budgets, cancellation
- `docs/js-runtime.md` — Boa contexts, `execute`, module loading, reset
- `docs/tools-folder.md` — `.tools` manifests, imports, execution, mirror, rescan
- `docs/virtual-fs.md` — flat root + `.tools` mount, `fs_*` semantics, persistence
- `docs/providers.md` — `llm_api` adapters, streaming, schemas, validation
- `docs/configuration.md` — environment variables and defaults
- `docs/testing-debugging.md` — tests, BRP workflow, logs, pitfalls
- `docs/known-issues.md` — deliberate decisions and remaining gaps
