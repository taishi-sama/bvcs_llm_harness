# `.tools`: script tools and libraries

`.tools` is a hierarchical source tree mounted beside the flat virtual
filesystem. It holds script tools (model-callable) and libraries (importable by
scripts). The in-memory tree is the source of truth; `./.tools` on disk is a
best-effort mirror.

## Layout

```
.tools/
  echo/
    tool.json          # manifest: model-facing definition
    main.js            # entry module, exports main(args)
    lib/util.js        # optional helpers, imported by main.js
  shared/
    library.json       # optional: marks a library and sets its entry
    index.js
  notes.md             # any other file: seeded and mirrored, importable only by path
```

A directory is a **tool** only if `tool.json` parses. A directory with
`library.json` is a library with a declared entry. Anything else is a plain
module directory resolved by convention.

## `tool.json`

```json
{
  "name": "echo",
  "description": "Echo text back, prefixed.",
  "parameters": {
    "text": { "type": "string", "description": "Text to echo", "required": true }
  },
  "entry": "main.js",
  "export": "main",
  "order": 500
}
```

| Field | Default | Notes |
| --- | --- | --- |
| `name` | required | must equal the directory name; unique among all tools |
| `description` | `Tool defined in .tools/<dir>` | model-facing |
| `parameters` | `{}` | `type` is `string` / `number` / `boolean`; `description` is required; `required` defaults `true` |
| `entry` | `main.js` | must be a `.js` file that exists in the tree |
| `export` | `main` | exported function the host calls |
| `order` | `500` | advertisement order; script tools start at `1000 + order` |

There is intentionally **no timeout field**: script calls are not cancelled by
wall clock. Runaway CPU is bounded by Boa's loop iteration limit, and long work
should be detached at the root-call level.

## `library.json`

```json
{ "entry": "index.js" }
```

Marks a directory as an importable library (and lets it use a non-standard
entry). Libraries never appear in the model's schema.

## Import resolution

Inside a script module:

- `./x.js`, `../lib/x.js` — relative to the importing module's path
- `tool2` — the target directory's manifest entry, then `index.js`, then
  `main.js`
- `lib` — same, for a library directory
- `flat` — `flat.js` as a last resort

Escapes (`..` past `.tools`, absolute paths, backslashes) are rejected. Every
module returned by the loader is parsed with its path, so nested relative
imports work at any depth.

## Execution

A call is exactly “reset the context, evaluate the entry module, call
`main(args)`”:

- a fresh Boa context per call (no globals or module state leak between calls)
- the same `print`/`console`, `execute(tool, args)`, result rendering, and error
  formatting as `eval_javascript`
- arguments are validated against the manifest before the context is created:
  missing required parameters and wrong declared types are errors; unknown keys
  are passed through
- `main` is called with one argument: the parsed JSON arguments object
- a returned promise is awaited; printed output is followed by the returned
  value, and a `throw` (or rejected promise) becomes an error result
- libraries are instantiated per call, so they cannot hold cross-call state.
  Persist data through the VFS tools instead.

Scripts may call any tool except `ModelOnly` ones (agent lifecycle and the
interpreter); nested calls always run synchronously and cannot detach. See
`docs/tool-calls.md` for depth/budget rules.

## Mirroring

`ToolsTree` is the truth. The mirror exists so tools survive restarts and can be
edited in a normal editor.

- **Boot**: `seed_tool_tree` imports `./.tools` (all regular files, ≤1 MiB,
  symlinks and `*.tmp` skipped) with mirroring suppressed.
- **Writes**: `.tools/...` writes through `fs_*` mark paths dirty; `driver`
  flushes them after a 250 ms debounce.
- **Atomic**: temp file in the same directory (`create_new`, so a planted
  symlink cannot be followed) then rename.
- **Safe**: every parent component is checked with `symlink_metadata`; the
  canonical parent must stay under the canonical root, otherwise the flush
  fails and keeps the path dirty.
- **Delete**: removes the file and prunes now-empty parent directories.
- **Best effort**: a failed flush never fails the VFS write. `MirrorStatus {
  pending, last_error }` exposes the condition.
- `.tools` is **not** part of `vfs_state` Save/Load; the mirror is its
  persistence.

Disk edits made outside the harness are invisible until an explicit rescan:
`rescan_tools(apply: true, source: "disk")` reseeds the tree and reconciles.

## Live registration and rescan

- `reconcile_script_tools` runs after tool calls and before the next request's
  schema is built. It does work only when `VirtualFs::tools().revision` changes
  (or a rescan requested a forced apply). It fingerprints each manifest
  (definition + entry + export + order), then spawns / updates / despawns tool
  entities and updates `ToolRegistryStatus`.
- `rescan_tools(apply = true, source = "vfs" | "disk")` returns the same diff and
  asks the automatic reconciler to apply it after tool calls, before the next
  request. The first number is how many valid script tools the scan found in
  `.tools`, not the live registry size. Because hot reload usually applied the
  change already, an empty diff is followed by the last automatic change.
- `apply: false` is a dry run; it records the revision so the automatic
  reconciler does not apply that diff, and a later dry run cancels an earlier
  pending apply in the same frame. Deletions stay registered (and advertised)
  until an explicit apply, so a call can fail with "missing tool source" in the
  meantime.
- Broken manifests never abort the scan: they are skipped and reported (invalid
  JSON, name/dir mismatch, missing or non-`.js` entry, duplicate name, collision
  with a built-in or a JavaScript host tool). JSON syntax errors carry a
  line/column; typed field errors mention that nested objects such as parameters
  are checked too.

## Example

```
.tools/demo/tool.json   {"name":"demo","description":"Demo"}
.tools/demo/main.js     export function main(args) { return "demo:" + (args.message ?? ""); }
```

Write both, call `rescan_tools`, then `execute(tool_name: "demo", args: {...})`.
