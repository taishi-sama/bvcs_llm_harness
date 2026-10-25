//! Shared in-memory flat filesystem.

use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};

use bevy::prelude::Resource;
use regex::Regex;

use crate::{
    tools::tree::ToolsTree,
    vfs_path::{Mount, VfsPath},
};

/// Snapshot folder used by the Save/Load VFS buttons.
///
/// Defaults to `vfs_state` in the current working directory. Set
/// `VFS_STATE_DIR` to use another folder; relative values are resolved against
/// the current working directory at startup.
#[derive(Resource, Clone, Debug)]
pub(crate) struct VfsStateDir {
    pub(crate) path: PathBuf,
}

impl Default for VfsStateDir {
    fn default() -> Self {
        let path = env::var_os("VFS_STATE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("vfs_state"));
        let path = if path.is_absolute() {
            path
        } else {
            env::current_dir()
                .map(|cwd| cwd.join(&path))
                .unwrap_or(path)
        };
        Self { path }
    }
}

/// Shared in-memory flat filesystem. Every agent reads and writes the same
/// files, and nothing touches the host filesystem.
#[derive(Resource, Default)]
pub(crate) struct VirtualFs {
    files: HashMap<String, String>,
    /// Validated, hierarchical `.tools` sources. See [`crate::tools::tree`].
    tools: ToolsTree,
}

fn split_virtual_file(content: &str) -> (Vec<String>, bool) {
    (
        content.lines().map(str::to_owned).collect(),
        content.ends_with('\n'),
    )
}

/// Directory (inside the snapshot root) that holds the manifest. Load ignores
/// subdirectories, so this file never becomes part of the virtual filesystem.
const SNAPSHOT_DIR: &str = ".vfs_state";
/// Manifest file written by [`VirtualFs::save_to_dir`]. It lists the files in
/// the snapshot so a later save can prune entries the VFS no longer has.
const SNAPSHOT_MANIFEST: &str = "manifest";
/// Manifest location used by the first version of the snapshot support. It is
/// skipped on Load so it cannot become a virtual file.
const LEGACY_SNAPSHOT_MANIFEST: &str = ".vfs_state_manifest";

fn snapshot_manifest_path(dir: &Path) -> PathBuf {
    dir.join(SNAPSHOT_DIR).join(SNAPSHOT_MANIFEST)
}

fn read_snapshot_manifest(dir: &Path) -> Vec<String> {
    let path = snapshot_manifest_path(dir);
    let Ok(metadata) = fs::symlink_metadata(&path) else {
        return Vec::new();
    };
    if !metadata.file_type().is_file() {
        return Vec::new();
    }
    fs::read_to_string(&path)
        .map(|text| {
            text.lines()
                .map(|line| line.trim_end_matches('\r'))
                .filter(|line| is_plain_name(line))
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn is_plain_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains('/')
        && !name.contains('\\')
        && !name.chars().any(char::is_control)
}

/// Atomic snapshot write. A symlink planted at the target or temp path is not
/// followed (the target is refused; the temp is removed and recreated).
fn write_snapshot_file(dir: &Path, name: &str, content: &[u8]) -> Result<(), String> {
    use std::io::Write as _;

    let target = dir.join(name);
    if let Ok(metadata) = fs::symlink_metadata(&target)
        && metadata.file_type().is_symlink()
    {
        return Err(format!(
            "refusing to write `{}`: it is a symlink",
            target.display()
        ));
    }

    let temp = dir.join(format!(".{name}.vfs-tmp"));
    let _ = fs::remove_file(&temp);
    let result = (|| -> Result<(), String> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|err| format!("could not create `{}`: {err}", temp.display()))?;
        file.write_all(content)
            .map_err(|err| format!("could not write `{}`: {err}", temp.display()))?;
        file.sync_all()
            .map_err(|err| format!("could not flush `{}`: {err}", temp.display()))?;
        drop(file);
        fs::rename(&temp, &target)
            .map_err(|err| format!("could not replace `{}`: {err}", target.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

impl VirtualFs {
    /// The `.tools` source tree (mounted, not part of the flat root).
    pub(crate) fn tools(&self) -> &ToolsTree {
        &self.tools
    }

    pub(crate) fn tools_mut(&mut self) -> &mut ToolsTree {
        &mut self.tools
    }

    /// Parse a caller-supplied path. All validation lives in [`VfsPath`]:
    /// the flat root stays directory-free, and `.tools` is the only mount.
    fn resolve(path: &str) -> Result<VfsPath, String> {
        VfsPath::parse(path.trim())
    }

    fn read_any(&self, path: &VfsPath) -> Result<String, String> {
        match path.mount() {
            Mount::Root => self.files.get(&path.relative()).cloned().ok_or_else(|| {
                format!(
                    "No such file: `{}`. Use fs_write to create it.",
                    path.display()
                )
            }),
            Mount::Tools => self.tools.read(&path.relative()).map_err(|_| {
                format!(
                    "No such file: `{}`. Use fs_write to create it.",
                    path.display()
                )
            }),
        }
    }

    fn write_any(&mut self, path: &VfsPath, content: String) {
        match path.mount() {
            Mount::Root => {
                self.files.insert(path.relative(), content);
            }
            Mount::Tools => self.tools.write(&path.relative(), content),
        }
    }

    fn delete_any(&mut self, path: &VfsPath) -> Result<usize, String> {
        match path.mount() {
            Mount::Root => self
                .files
                .remove(&path.relative())
                .map(|content| content.len())
                .ok_or_else(|| {
                    format!(
                        "No such file: `{}`. Use fs_ls to list files.",
                        path.display()
                    )
                }),
            Mount::Tools => self
                .tools
                .delete(&path.relative())
                .map_err(|err| format!("{err}. Use fs_ls to list files.")),
        }
    }

    pub(crate) fn list(&self) -> String {
        if self.files.is_empty() && self.tools.is_empty() {
            return "Virtual filesystem is empty.".to_owned();
        }

        let mut names: Vec<&String> = self.files.keys().collect();
        names.sort_unstable();
        let mut listing = if names.is_empty() {
            String::new()
        } else {
            let mut listing = format!("Files ({}):", names.len());
            for name in names {
                let content = &self.files[name];
                listing.push_str(&format!(
                    "\n{name} - {} bytes, {} line(s)",
                    content.len(),
                    content.lines().count()
                ));
            }
            listing
        };
        if !self.tools.is_empty() {
            if !listing.is_empty() {
                listing.push('\n');
            }
            let mount = crate::vfs_path::TOOLS_MOUNT;
            listing.push_str(&format!("{mount}/ ({} file(s)):", self.tools.len()));
            for (path, content) in self.tools.iter() {
                listing.push_str(&format!(
                    "\n{mount}/{path} - {} bytes, {} line(s)",
                    content.len(),
                    content.lines().count()
                ));
            }
        }
        listing
    }

    pub(crate) fn read(&self, path: &str) -> Result<String, String> {
        self.read_range(path, None, None)
    }

    /// Read a byte range. `offset` and `limit` are byte offsets into the UTF-8
    /// contents; omitted bounds read from the start / to the end.
    pub(crate) fn read_range(
        &self,
        path: &str,
        offset: Option<u64>,
        limit: Option<u64>,
    ) -> Result<String, String> {
        let path = Self::resolve(path)?;
        let name = path.display();
        let content = self.read_any(&path)?;

        let start = usize::try_from(offset.unwrap_or(0))
            .map_err(|_| "`offset` is too large for this platform".to_owned())?;
        if start > content.len() {
            return Err(format!(
                "`offset` ({start}) is past the end of `{name}` ({} bytes)",
                content.len()
            ));
        }
        if !content.is_char_boundary(start) {
            return Err(format!(
                "`offset` ({start}) is not on a UTF-8 character boundary in `{name}`"
            ));
        }

        let end = match limit {
            Some(limit) => {
                let limit = usize::try_from(limit)
                    .map_err(|_| "`limit` is too large for this platform".to_owned())?;
                start.saturating_add(limit).min(content.len())
            }
            None => content.len(),
        };
        if !content.is_char_boundary(end) {
            let range = if offset.is_some() {
                "`offset` + `limit`"
            } else {
                "`limit`"
            };
            return Err(format!(
                "{range} ends at byte {end}, which is not on a UTF-8 character boundary in `{name}`"
            ));
        }

        Ok(content[start..end].to_owned())
    }

    /// Search every file (or one file) for a literal substring.
    ///
    /// Returns grep-style lines: `file:line:text` for matches and
    /// `file-line-text` for context lines. A `--` line separates disjoint
    /// context groups.
    pub(crate) fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        context_lines: usize,
        ignore_case: bool,
        max_matches: usize,
    ) -> Result<String, String> {
        if pattern.is_empty() {
            return Err("pattern must not be empty".to_owned());
        }

        let context = context_lines.min(100);
        let max_matches = max_matches.max(1);
        let needle = if ignore_case {
            pattern.to_lowercase()
        } else {
            pattern.to_owned()
        };

        let mut files = match path {
            Some(path) => {
                let path = Self::resolve(path)?;
                vec![(path.display(), self.read_any(&path)?)]
            }
            None => {
                let mut files: Vec<(String, String)> = self
                    .files
                    .iter()
                    .map(|(name, content)| (name.clone(), content.clone()))
                    .collect();
                files.extend(self.tools.iter().map(|(name, content)| {
                    (
                        format!("{}/{name}", crate::vfs_path::TOOLS_MOUNT),
                        content.to_owned(),
                    )
                }));
                files
            }
        };
        files.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut output = String::new();
        let mut found = 0usize;
        let mut truncated = false;

        for (name, content) in files {
            let lines = content.lines().collect::<Vec<_>>();
            let mut matched = Vec::new();
            let mut limit_hit = false;

            for (index, line) in lines.iter().enumerate() {
                let is_match = if ignore_case {
                    line.to_lowercase().contains(&needle)
                } else {
                    line.contains(&needle)
                };
                if !is_match {
                    continue;
                }
                if found + matched.len() >= max_matches {
                    limit_hit = true;
                    truncated = true;
                    break;
                }
                matched.push(index);
            }

            if !matched.is_empty() {
                found += matched.len();
                let mut is_match = vec![false; lines.len()];
                for &index in &matched {
                    is_match[index] = true;
                }

                let mut windows = Vec::<(usize, usize)>::new();
                for &index in &matched {
                    let start = index.saturating_sub(context);
                    let end = (index + context + 1).min(lines.len());
                    if let Some(last) = windows.last_mut()
                        && start <= last.1
                    {
                        last.1 = last.1.max(end);
                        continue;
                    }
                    windows.push((start, end));
                }

                for (group, (start, end)) in windows.into_iter().enumerate() {
                    if group > 0 && context > 0 {
                        output.push_str("--\n");
                    }
                    for index in start..end {
                        let separator = if is_match[index] { ':' } else { '-' };
                        output.push_str(&format!(
                            "{name}{separator}{}{separator}{}\n",
                            index + 1,
                            lines[index]
                        ));
                    }
                }
            }

            if limit_hit {
                break;
            }
        }

        if found == 0 {
            return Ok(format!("No matches for `{pattern}`."));
        }
        if truncated {
            output.push_str(&format!(
                "... (truncated at {found} matches; limit {max_matches}, raise max_matches or narrow the pattern)\n"
            ));
        }

        Ok(output.trim_end_matches('\n').to_owned())
    }

    /// Print an inclusive, 1-based line range, similar to
    /// `sed -n 'START,ENDp'`.
    pub(crate) fn line_range(
        &self,
        path: &str,
        start_line: Option<u64>,
        end_line: Option<u64>,
        line_numbers: bool,
    ) -> Result<String, String> {
        let path = Self::resolve(path)?;
        let name = path.display();
        let content = self.read_any(&path)?;
        let lines = content.lines().collect::<Vec<_>>();
        if lines.is_empty() {
            return Ok(format!("`{name}` is empty."));
        }

        let start = usize::try_from(start_line.unwrap_or(1))
            .map_err(|_| "start_line is too large for this platform".to_owned())?;
        if start == 0 {
            return Err("start_line is 1-based and must be at least 1".to_owned());
        }
        if start > lines.len() {
            return Err(format!(
                "start_line ({start}) is past the end of `{name}` ({} line(s))",
                lines.len()
            ));
        }

        let end = match end_line {
            Some(end) => usize::try_from(end)
                .map_err(|_| "end_line is too large for this platform".to_owned())?,
            None => lines.len(),
        };
        if end < start {
            return Err(format!(
                "end_line ({end}) must be greater than or equal to start_line ({start})"
            ));
        }
        let end = end.min(lines.len());

        let mut output = String::new();
        for (offset, line) in lines[start - 1..end].iter().enumerate() {
            let number = start + offset;
            if line_numbers {
                output.push_str(&format!("{number}: {line}\n"));
            } else {
                output.push_str(line);
                output.push('\n');
            }
        }

        Ok(output.trim_end_matches('\n').to_owned())
    }

    /// Write every virtual file into `dir`, creating the directory if needed.
    /// Existing files with the same names are overwritten atomically.
    ///
    /// Each save records the names it wrote in `.vfs_state/manifest` and removes
    /// files listed by the previous save that are no longer in the VFS, so a
    /// snapshot does not resurrect a file it deleted. Files that no Save has
    /// recorded (including a legacy state directory) are never pruned; clean
    /// them manually if needed. Target-file symlinks are refused rather than
    /// followed.
    pub(crate) fn save_to_dir(&self, dir: &Path) -> Result<String, String> {
        fs::create_dir_all(dir)
            .map_err(|err| format!("could not create `{}`: {err}", dir.display()))?;

        let snapshot_dir = dir.join(SNAPSHOT_DIR);
        match fs::symlink_metadata(&snapshot_dir) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!(
                    "refusing to use `{}`: it is a symlink",
                    snapshot_dir.display()
                ));
            }
            Ok(metadata) if !metadata.is_dir() => {
                return Err(format!(
                    "`{}` exists and is not a directory",
                    snapshot_dir.display()
                ));
            }
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&snapshot_dir).map_err(|err| {
                    format!("could not create `{}`: {err}", snapshot_dir.display())
                })?;
            }
            Err(err) => {
                return Err(format!(
                    "could not inspect `{}`: {err}",
                    snapshot_dir.display()
                ));
            }
        }

        let previous = read_snapshot_manifest(dir);
        let mut names = self.files.keys().collect::<Vec<_>>();
        names.sort_unstable();
        for name in &names {
            write_snapshot_file(dir, name, self.files[*name].as_bytes())?;
        }

        for old in previous {
            if self.files.contains_key(&old) || !is_plain_name(&old) {
                continue;
            }
            let path = dir.join(&old);
            // Remove regular files only; a symlink is not ours to follow or
            // delete (and `remove_file` on one would just unlink it anyway).
            if let Ok(metadata) = fs::symlink_metadata(&path)
                && metadata.file_type().is_file()
            {
                fs::remove_file(&path)
                    .map_err(|err| format!("could not remove `{}`: {err}", path.display()))?;
            }
        }

        let manifest = names
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let manifest = format!("{manifest}\n");
        write_snapshot_file(&snapshot_dir, SNAPSHOT_MANIFEST, manifest.as_bytes())?;

        Ok(format!(
            "Saved {} file(s) to `{}`.",
            names.len(),
            dir.display()
        ))
    }

    /// Replace the virtual filesystem with every UTF-8 regular file found in
    /// `dir`. Subdirectories are ignored.
    pub(crate) fn load_from_dir(&mut self, dir: &Path) -> Result<String, String> {
        let entries = fs::read_dir(dir)
            .map_err(|err| format!("could not read `{}`: {err}", dir.display()))?;

        let mut files = HashMap::new();
        let mut skipped = 0usize;
        for entry in entries {
            let entry = entry
                .map_err(|err| format!("could not read an entry in `{}`: {err}", dir.display()))?;
            let file_type = entry
                .file_type()
                .map_err(|err| format!("could not inspect `{}`: {err}", entry.path().display()))?;
            if !file_type.is_file() {
                skipped += 1;
                continue;
            }

            let name = entry.file_name().into_string().map_err(|name| {
                format!(
                    "cannot load file with a non-UTF-8 name in `{}`: {:?}",
                    dir.display(),
                    name
                )
            })?;
            if name == LEGACY_SNAPSHOT_MANIFEST {
                continue;
            }
            let content = fs::read_to_string(entry.path()).map_err(|err| {
                format!(
                    "could not read `{}` as UTF-8 text: {err}",
                    entry.path().display()
                )
            })?;
            files.insert(name, content);
        }

        let count = files.len();
        self.files = files;

        let skipped = if skipped == 0 {
            String::new()
        } else {
            format!(" (skipped {skipped} non-file entr(y/ies))")
        };
        Ok(format!(
            "Loaded {count} file(s) from `{}`{skipped}.",
            dir.display()
        ))
    }

    pub(crate) fn write(&mut self, path: &str, content: String) -> Result<String, String> {
        let path = Self::resolve(path)?;
        let name = path.display();
        let lines = content.lines().count();
        self.write_any(&path, content);
        Ok(format!("Wrote `{name}` ({lines} lines)."))
    }

    pub(crate) fn delete(&mut self, path: &str) -> Result<String, String> {
        let path = Self::resolve(path)?;
        let name = path.display();
        let bytes = self.delete_any(&path)?;
        Ok(format!("Deleted `{name}` ({bytes} bytes)."))
    }

    /// Replaces an inclusive, 1-based line range. An empty `content` deletes
    /// the selected lines.
    pub(crate) fn edit_lines(
        &mut self,
        path: &str,
        start_line: u64,
        end_line: u64,
        content: String,
    ) -> Result<String, String> {
        let path = Self::resolve(path)?;
        let name = path.display();
        let original = self.read_any(&path)?;
        let (mut lines, trailing_newline) = split_virtual_file(&original);

        let start = usize::try_from(start_line)
            .map_err(|_| format!("invalid start_line `{start_line}`"))?;
        let end =
            usize::try_from(end_line).map_err(|_| format!("invalid end_line `{end_line}`"))?;
        if start == 0 {
            return Err("start_line is 1-based and must be at least 1".to_owned());
        }
        if end < start {
            return Err(format!(
                "end_line ({end_line}) must be greater than or equal to start_line ({start_line})"
            ));
        }
        if end > lines.len() {
            return Err(format!(
                "line range {start_line}-{end_line} is out of bounds; `{name}` has {} line(s)",
                lines.len()
            ));
        }

        let replacement: Vec<String> = if content.is_empty() {
            Vec::new()
        } else {
            content.lines().map(str::to_owned).collect()
        };
        lines.splice(start - 1..end, replacement);

        let mut updated = lines.join("\n");
        if trailing_newline && !updated.is_empty() {
            updated.push('\n');
        }
        let line_count = lines.len();
        self.write_any(&path, updated);
        Ok(format!(
            "Replaced lines {start_line}-{end_line} in `{name}`; file now has {line_count} line(s)."
        ))
    }

    /// Replaces a substring or regular expression.
    ///
    /// By default the pattern is literal and must match exactly once. Matching
    /// every occurrence requires `allow_multiple`. Regex replacements expand
    /// `$1`/`${name}` capture references.
    pub(crate) fn edit_pattern(
        &mut self,
        path: &str,
        pattern: &str,
        replacement: &str,
        allow_multiple: bool,
        use_regex: bool,
    ) -> Result<String, String> {
        let path = Self::resolve(path)?;
        let name = path.display();
        if pattern.is_empty() {
            return Err("pattern must not be empty".to_owned());
        }
        let original = self.read_any(&path)?;

        let regex = if use_regex {
            Some(
                Regex::new(pattern)
                    .map_err(|err| format!("invalid regular expression `{pattern}`: {err}"))?,
            )
        } else {
            None
        };
        let count = match &regex {
            Some(regex) => regex.find_iter(&original).count(),
            None => original.matches(pattern).count(),
        };
        if count == 0 {
            return Err(format!("Pattern not found in `{name}`."));
        }
        if count > 1 && !allow_multiple {
            return Err(format!(
                "Pattern matches {count} times in `{name}`; refusing to edit because exactly one match is required."
            ));
        }

        let updated = match &regex {
            Some(regex) if allow_multiple => regex.replace_all(&original, replacement).into_owned(),
            Some(regex) => regex.replacen(&original, 1, replacement).into_owned(),
            None if allow_multiple => original.replace(pattern, replacement),
            None => original.replacen(pattern, replacement, 1),
        };
        self.write_any(&path, updated);
        Ok(match count {
            1 => format!("Replaced the single match of the pattern in `{name}`."),
            count => format!("Replaced {count} matches of the pattern in `{name}`."),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_mount_reads_and_writes_outside_the_flat_root() {
        let mut fs = VirtualFs::default();
        fs.write(".tools/tool1/main.js", "export const x = 1;".into())
            .unwrap();
        assert_eq!(
            fs.read(".tools/tool1/main.js").unwrap(),
            "export const x = 1;"
        );
        assert!(fs.read("tool1/main.js").is_err());
        assert!(
            fs.list().contains(".tools/tool1/main.js - 19 bytes"),
            "{}",
            fs.list()
        );

        fs.edit_pattern(".tools/tool1/main.js", "1", "2", false, false)
            .unwrap();
        assert_eq!(
            fs.read(".tools/tool1/main.js").unwrap(),
            "export const x = 2;"
        );
        fs.delete(".tools/tool1/main.js").unwrap();
        assert!(fs.read(".tools/tool1/main.js").is_err());
    }

    #[test]
    fn tools_paths_are_validated() {
        let mut fs = VirtualFs::default();
        assert!(fs.write(".tools/../escape.js", "x".into()).is_err());
        assert!(fs.write(".tools", "x".into()).is_err());
        assert!(fs.write("dir/file.txt", "x".into()).is_err());
    }

    #[test]
    fn virtual_fs_supports_shared_file_lifecycle() {
        let mut fs = VirtualFs::default();
        assert!(fs.read("notes.txt").is_err());

        fs.write("notes.txt", "one\ntwo\nthree\n".into()).unwrap();
        assert_eq!(fs.read("notes.txt").unwrap(), "one\ntwo\nthree\n");
        let listing = fs.list();
        assert!(listing.contains("Files (1):"), "{listing}");
        assert!(
            listing.contains("notes.txt - 14 bytes, 3 line(s)"),
            "{listing}"
        );

        fs.edit_lines("notes.txt", 2, 2, "TWO".into()).unwrap();
        assert_eq!(fs.read("notes.txt").unwrap(), "one\nTWO\nthree\n");

        fs.edit_pattern("notes.txt", "TWO", "two", false, false)
            .unwrap();
        assert_eq!(fs.read("notes.txt").unwrap(), "one\ntwo\nthree\n");

        fs.delete("notes.txt").unwrap();
        assert!(fs.read("notes.txt").is_err());
        let err = fs.delete("notes.txt").unwrap_err();
        assert!(err.contains("Use fs_ls"), "{err}");
        assert_eq!(fs.list(), "Virtual filesystem is empty.");
    }

    #[test]
    fn virtual_fs_edits_are_unambiguous_and_bounded() {
        let mut fs = VirtualFs::default();
        fs.write("a.txt", "same same\nsecond".into()).unwrap();

        let err = fs
            .edit_pattern("a.txt", "same", "other", false, false)
            .expect_err("ambiguous pattern must fail");
        assert!(err.contains("2 times"), "{err}");

        assert!(
            fs.edit_pattern("a.txt", "missing", "x", false, false)
                .is_err()
        );
        assert!(fs.edit_lines("a.txt", 0, 1, "x".into()).is_err());
        assert!(fs.edit_lines("a.txt", 3, 3, "x".into()).is_err());
        assert!(fs.edit_lines("a.txt", 2, 1, "x".into()).is_err());
        assert!(fs.write("dir/file.txt", "x".into()).is_err());
        assert!(fs.read("..").is_err());
    }

    #[test]
    fn edit_pattern_can_replace_all_literal_matches() {
        let mut fs = VirtualFs::default();
        fs.write("m.txt", "same same same".into()).unwrap();

        let message = fs
            .edit_pattern("m.txt", "same", "other", true, false)
            .unwrap();
        assert_eq!(fs.read("m.txt").unwrap(), "other other other");
        assert!(message.contains("3 matches"), "{message}");
    }

    #[test]
    fn edit_pattern_supports_regular_expressions_and_captures() {
        let mut fs = VirtualFs::default();
        fs.write("r.txt", "name: Ada".into()).unwrap();

        fs.edit_pattern("r.txt", r"name: (\w+)", "name: ${1} Lovelace", false, true)
            .unwrap();
        assert_eq!(fs.read("r.txt").unwrap(), "name: Ada Lovelace");

        fs.write("n.txt", "a1 b2".into()).unwrap();
        let err = fs
            .edit_pattern("n.txt", r"\d", "0", false, true)
            .expect_err("ambiguous regex must fail");
        assert!(err.contains("2 times"), "{err}");

        let message = fs.edit_pattern("n.txt", r"\d", "0", true, true).unwrap();
        assert_eq!(fs.read("n.txt").unwrap(), "a0 b0");
        assert!(message.contains("2 matches"), "{message}");
    }

    #[test]
    fn edit_pattern_rejects_invalid_regex_and_treats_literals_literally() {
        let mut fs = VirtualFs::default();
        fs.write("lit.txt", "a.b".into()).unwrap();

        fs.edit_pattern("lit.txt", ".", "!", false, false).unwrap();
        assert_eq!(fs.read("lit.txt").unwrap(), "a!b");

        let err = fs
            .edit_pattern("lit.txt", "(", "x", false, true)
            .expect_err("invalid regex must fail");
        assert!(err.contains("invalid regular expression"), "{err}");
    }

    #[test]
    fn virtual_fs_line_edits_can_delete_lines() {
        let mut fs = VirtualFs::default();
        fs.write("d.txt", "a\nb\nc".into()).unwrap();

        fs.edit_lines("d.txt", 2, 2, String::new()).unwrap();
        assert_eq!(fs.read("d.txt").unwrap(), "a\nc");

        fs.edit_lines("d.txt", 1, 2, String::new()).unwrap();
        assert_eq!(fs.read("d.txt").unwrap(), "");
    }

    #[test]
    fn virtual_fs_read_range_uses_byte_offsets() {
        let mut fs = VirtualFs::default();
        fs.write("r.txt", "abcdef".into()).unwrap();

        assert_eq!(fs.read_range("r.txt", Some(2), Some(3)).unwrap(), "cde");
        assert_eq!(fs.read_range("r.txt", Some(4), None).unwrap(), "ef");
        assert_eq!(fs.read_range("r.txt", None, Some(2)).unwrap(), "ab");
        assert_eq!(fs.read_range("r.txt", Some(6), None).unwrap(), "");
        assert_eq!(fs.read_range("r.txt", Some(4), Some(100)).unwrap(), "ef");

        assert!(fs.read_range("r.txt", Some(7), None).is_err());
        assert!(fs.read_range("missing.txt", Some(0), None).is_err());
    }

    #[test]
    fn virtual_fs_read_range_respects_utf8_boundaries() {
        let mut fs = VirtualFs::default();
        fs.write("u.txt", "é!".into()).unwrap();

        assert_eq!(fs.read_range("u.txt", None, Some(2)).unwrap(), "é");
        assert!(fs.read_range("u.txt", Some(1), None).is_err());
        let err = fs.read_range("u.txt", None, Some(1)).unwrap_err();
        assert!(err.contains("`limit` ends at byte 1"), "{err}");
        assert!(!err.contains("offset"), "{err}");
    }

    #[test]
    fn virtual_fs_grep_reports_locations_and_context() {
        let mut fs = VirtualFs::default();
        fs.write("a.txt", "one\ntwo\nneedle here\nfour\nfive\n".into())
            .unwrap();
        fs.write("b.txt", "nothing\nNeedle\n".into()).unwrap();

        let output = fs.grep("needle", None, 1, false, 200).unwrap();
        assert!(output.contains("a.txt:3:needle here"), "{output}");
        assert!(output.contains("a.txt-2-two"), "{output}");
        assert!(output.contains("a.txt-4-four"), "{output}");
        assert!(!output.contains("b.txt"), "{output}");
        assert!(!output.contains("a.txt-1-one"), "{output}");
        assert!(!output.contains("a.txt-5-five"), "{output}");

        let insensitive = fs.grep("needle", None, 0, true, 200).unwrap();
        assert!(insensitive.contains("a.txt:3:needle here"), "{insensitive}");
        assert!(insensitive.contains("b.txt:2:Needle"), "{insensitive}");
        assert!(!insensitive.contains("a.txt-2-two"), "{insensitive}");

        assert_eq!(
            fs.grep("absent", None, 0, false, 200).unwrap(),
            "No matches for `absent`."
        );
    }

    #[test]
    fn virtual_fs_grep_merges_windows_and_marks_truncation() {
        let mut fs = VirtualFs::default();
        fs.write("m.txt", "a\nmatch\nb\nmatch\nc\n".into()).unwrap();

        let merged = fs.grep("match", None, 1, false, 200).unwrap();
        assert!(!merged.contains("--"), "{merged}");
        assert_eq!(merged.lines().count(), 5, "{merged}");

        let separate = fs.grep("match", None, 0, false, 200).unwrap();
        assert!(!separate.contains("--"), "{separate}");
        assert_eq!(separate.lines().count(), 2, "{separate}");

        let truncated = fs.grep("match", None, 0, false, 1).unwrap();
        assert!(truncated.contains("m.txt:2:match"), "{truncated}");
        assert!(!truncated.contains("m.txt:4:match"), "{truncated}");
        assert!(truncated.contains("truncated at 1 matches"), "{truncated}");
        assert!(truncated.contains("limit 1"), "{truncated}");
    }

    #[test]
    fn virtual_fs_line_range_prints_inclusive_lines() {
        let mut fs = VirtualFs::default();
        fs.write("s.txt", "one\ntwo\nthree\nfour\nfive\n".into())
            .unwrap();

        assert_eq!(
            fs.line_range("s.txt", Some(2), Some(4), true).unwrap(),
            "2: two\n3: three\n4: four"
        );
        assert_eq!(
            fs.line_range("s.txt", Some(2), Some(4), false).unwrap(),
            "two\nthree\nfour"
        );
        assert_eq!(
            fs.line_range("s.txt", None, None, false).unwrap(),
            "one\ntwo\nthree\nfour\nfive"
        );
        assert_eq!(
            fs.line_range("s.txt", Some(4), Some(100), false).unwrap(),
            "four\nfive"
        );

        assert!(fs.line_range("s.txt", Some(0), Some(1), true).is_err());
        assert!(fs.line_range("s.txt", Some(4), Some(2), true).is_err());
        assert!(fs.line_range("s.txt", Some(6), None, true).is_err());
    }

    #[test]
    fn virtual_fs_round_trips_through_a_directory() {
        let dir = std::env::temp_dir().join(format!("bvcs_vfs_state_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        let mut saved = VirtualFs::default();
        saved.write("a.txt", "alpha\n".into()).unwrap();
        saved.write("b.txt", "beta".into()).unwrap();
        let message = saved.save_to_dir(&dir).unwrap();
        assert!(message.contains("Saved 2 file(s)"), "{message}");

        let mut loaded = VirtualFs::default();
        loaded.write("stale.txt", "stale".into()).unwrap();
        let message = loaded.load_from_dir(&dir).unwrap();
        assert!(message.contains("Loaded 2 file(s)"), "{message}");
        assert_eq!(loaded.read("a.txt").unwrap(), "alpha\n");
        assert_eq!(loaded.read("b.txt").unwrap(), "beta");
        assert!(loaded.read("stale.txt").is_err());

        // A later save removes files written by the previous snapshot that the
        // VFS no longer has, so Load cannot resurrect deleted files.
        saved.delete("a.txt").unwrap();
        saved.save_to_dir(&dir).unwrap();
        assert!(!dir.join("a.txt").exists());
        let mut reloaded = VirtualFs::default();
        let message = reloaded.load_from_dir(&dir).unwrap();
        assert!(message.contains("Loaded 1 file(s)"), "{message}");
        assert_eq!(reloaded.read("b.txt").unwrap(), "beta");
        assert!(reloaded.read("a.txt").is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn snapshot_prunes_only_previous_files_and_ignores_traversal() {
        let dir =
            std::env::temp_dir().join(format!("bvcs_vfs_snapshot_safe_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("stale.txt"), "stale").unwrap();
        fs::write(dir.join("victim.txt"), "old").unwrap();
        fs::create_dir_all(dir.join(SNAPSHOT_DIR)).unwrap();
        fs::write(
            dir.join(SNAPSHOT_DIR).join(SNAPSHOT_MANIFEST),
            "../escape.txt\r\nvictim.txt\r\nsub/x.txt\r\n",
        )
        .unwrap();

        let mut vfs = VirtualFs::default();
        vfs.write("a.txt", "alpha".into()).unwrap();
        vfs.save_to_dir(&dir).unwrap();

        assert!(
            dir.join("stale.txt").exists(),
            "files never recorded in a manifest are left alone"
        );
        assert!(
            !dir.join("victim.txt").exists(),
            "a previous snapshot file is pruned"
        );
        assert!(dir.join("a.txt").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn snapshot_refuses_to_follow_symlinks() {
        use std::os::unix::fs::symlink;

        let dir =
            std::env::temp_dir().join(format!("bvcs_vfs_snapshot_link_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("target.txt");
        fs::write(&target, "keep").unwrap();
        symlink(&target, dir.join("a.txt")).unwrap();

        let mut vfs = VirtualFs::default();
        vfs.write("a.txt", "overwrite".into()).unwrap();
        let err = vfs.save_to_dir(&dir).unwrap_err();
        assert!(err.contains("symlink"), "{err}");
        assert_eq!(fs::read_to_string(&target).unwrap(), "keep");

        let _ = fs::remove_dir_all(&dir);
    }
}
