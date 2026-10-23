# AGENTS

Notes for coding agents working in this repository.

Before changing a subsystem, read its doc under `docs/` and update that doc in
the same change. The docs are part of the deliverable, not an afterthought.

## Which doc to read / update

| If you touch… | Read and update |
| --- | --- |
| threads, resources, schedules, messages, module layout | `docs/architecture.md` |
| turns, history, streaming, retries, pending messages | `docs/agent-loop.md` |
| tool dispatch, tool lifetime, budgets, adding a tool | `docs/tool-calls.md` |
| rounds, detached/nested calls, cancellation | `docs/scheduling.md` |
| Boa contexts, `execute`, module loading, reset | `docs/js-runtime.md` |
| `.tools` manifests, imports, mirror, rescan | `docs/tools-folder.md` |
| `fs_*`, path validation, Save/Load, `.tools` mount | `docs/virtual-fs.md` |
| `llm_api`, providers, streaming, schemas, validation | `docs/providers.md` |
| environment variables and defaults | `docs/configuration.md` (and the table in `README.MD`) |
| tests, BRP workflow, debugging | `docs/testing-debugging.md` |
| decisions, known gaps, security posture | `docs/known-issues.md` |

## Maintenance rules

1. A behavior, default, error message, or invariant change updates its doc in
   the same commit.
2. A new subsystem, env var, tool, or failure mode gets a short section in the
   most relevant doc — do not leave knowledge only in code or chat.
3. When you fix or accept a gap, update `docs/known-issues.md`.
4. Keep docs implementation-accurate: real file/function names, defaults, and
   limits. If code and docs disagree, resolve it immediately.
5. Cross-link instead of duplicating. `docs/tool-calls.md` is the anchor for
   dispatch; link to it rather than restating it.
6. Keep `README.MD` to the run instructions, environment table, and docs index.

## Before committing

```sh
cargo test
cargo clippy --all-targets
cargo fmt --check
```

Three invariants worth repeating: the Bevy main loop never blocks on HTTP or a
Boa heap; tool metadata has one source (`definitions()` in `src/tools.rs`); the
in-memory `.tools` tree is authoritative and disk is a best-effort mirror.
