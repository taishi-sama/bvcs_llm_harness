# Virtual filesystem

Two mounts share one `fs_*` tool surface:

| Mount | Shape | Backing |
| --- | --- | --- |
| root | flat, exactly one path segment | in-memory `HashMap<String, String>` |
| `.tools/...` | hierarchical, `/`-separated | in-memory `ToolsTree`, mirrored to disk |

All paths from the model or scripts pass through `VfsPath::parse` before any
handler sees them.

## Path rules

- root: `notes.txt` only. `dir/file.txt`, absolute paths, `..`, `.`, empty
  segments, trailing slashes, backslashes, and control characters are rejected.
- `.tools`: `.tools/<segment>[/<segment>...]`; `.tools` alone is rejected;
  escapes (`..`) are rejected.
- `VfsPath::display()` renders the path back as the model wrote it (root name or
  `.tools/<relative>`), and that is what messages and grep output use.

## `fs_*` behaviour

| Tool | Notes |
| --- | --- |
| `fs_read` | whole file, or `offset`/`limit` byte range. Ranges must start and end on UTF-8 boundaries. |
| `fs_sed` | inclusive 1-based line range; `start_line` defaults 1, `end_line` to EOF; line numbers on by default. |
| `fs_ls` | lists root files with size/line count, then every `.tools` key with size/line count. |
| `fs_grep` | literal substring; `file:line:text` for hits, `file-line-text` for context, `--` between groups. Omit `path` to search the flat root **and** `.tools`. `context` capped at 100, `max_matches` default 200 / max 5000, optional case folding. |
| `fs_write` | create or overwrite; reports line count. `.tools` writes mark the mirror dirty. |
| `fs_edit_lines` | replace an inclusive 1-based range; empty content deletes the range. |
| `fs_edit_pattern` | replace text. Literal and exactly-once by default; `allow_multiple` replaces every match; `use_regex` treats `pattern` as a regular expression and expands `$1` / `${name}` captures in `replacement`. |
| `fs_delete` | remove a file; `.tools` deletions also prune empty directories in the mirror. |

Files are UTF-8 text. Writes are in-memory and immediate; persistence differs by
mount.

## Persistence

- **Root**: the Save/Load VFS buttons snapshot the flat root into `VfsStateDir`
  (default `vfs_state`, override with `VFS_STATE_DIR`). Loading replaces the
  root wholesale and ignores subdirectories; `.tools` is never included.
- **`.tools`**: the on-disk mirror (`./.tools`, override with `TOOLS_DIR`) is
  written from the tree after every change (debounced, atomic, best-effort) and
  read back at boot. See `docs/tools-folder.md`.

These are deliberately separate: `vfs_state` is session state, `.tools` is
project content.

## Design notes

- The flat root is a feature: the model cannot get lost in directories, and
  every name is exactly one token.
- `.tools` is the only hierarchy because imports and manifests need it.
- Reads never normalize into the mirror; the tree is authoritative, so a failed
  mirror write cannot change what a call sees.
- Empty files are valid; `read_range` on an empty file returns `""`.
- `edit_pattern` refusing ambiguous edits is intentional: it replaces the “edit
  the wrong occurrence” failure mode with a clear error. `allow_multiple` is the
  explicit opt-in for the replace-everything case.

## Where to look

| Concern | File |
| --- | --- |
| root + mount dispatch, all `fs_*` semantics | `src/virtual_fs.rs` |
| path parsing/validation | `src/vfs_path.rs` |
| `.tools` tree, dirty tracking, seed/flush | `src/tools/tree.rs` |
| Save/Load | `src/virtual_fs.rs` (`save_to_dir`, `load_from_dir`), `src/shell.rs` |
| Mirror scheduling/status | `src/tools/mirror.rs` |
