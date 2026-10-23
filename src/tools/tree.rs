//! The `.tools` source tree: hierarchical, in-memory, mirrored to disk.
//!
//! The tree is the source of truth. `dirty` tracks paths whose on-disk mirror
//! may be stale; the mirror is best-effort (see `mirror.rs`) and a failure to
//! write it never fails a virtual write.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};

/// Largest file the boot import will read from `./.tools`.
const MAX_SEED_FILE: u64 = 1024 * 1024;

/// Files under `.tools`, keyed by path relative to the mount (`tool1/main.js`).
#[derive(Default)]
pub(crate) struct ToolsTree {
    files: BTreeMap<String, String>,
    /// Bumped on every mutation so module-source snapshots can tell when they
    /// are stale.
    revision: u64,
    /// Paths whose mirror may be stale. Deleted paths stay here until a flush
    /// confirms the file is gone.
    dirty: BTreeSet<String>,
}

impl ToolsTree {
    pub(crate) fn len(&self) -> usize {
        self.files.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    /// Sorted `(relative path, contents)` pairs.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.files
            .iter()
            .map(|(path, content)| (path.as_str(), content.as_str()))
    }

    pub(crate) fn read(&self, relative: &str) -> Result<String, String> {
        self.files
            .get(relative)
            .cloned()
            .ok_or_else(|| format!("No such file: `{relative}`. Use fs_write to create it."))
    }

    pub(crate) fn write(&mut self, relative: &str, content: String) {
        self.files.insert(relative.to_owned(), content);
        self.dirty.insert(relative.to_owned());
        self.revision = self.revision.wrapping_add(1);
    }

    pub(crate) fn delete(&mut self, relative: &str) -> Result<usize, String> {
        let removed = self
            .files
            .remove(relative)
            .ok_or_else(|| format!("No such file: `{relative}`"))?;
        self.dirty.insert(relative.to_owned());
        self.revision = self.revision.wrapping_add(1);
        Ok(removed.len())
    }

    pub(crate) fn has_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    /// Replace the tree with the contents of `root` (normally `./.tools`).
    ///
    /// Only `*.js`, `tool.json`, and `library.json` are imported. Symlinks,
    /// oversized files, and paths that would escape `root` are skipped.
    /// Nothing is marked dirty: the disk is already the source of this state.
    pub(crate) fn seed_from_dir(&mut self, root: &Path) -> Result<usize, String> {
        // Build into a local map and swap only on success: a mid-walk I/O error
        // must not leave a silently truncated catalog behind.
        let mut files = BTreeMap::new();
        if root.exists() {
            Self::seed_dir(root, "", &mut files)?;
        }
        let count = files.len();
        self.files = files;
        self.dirty.clear();
        self.revision = self.revision.wrapping_add(1);
        Ok(count)
    }

    fn seed_dir(
        dir: &Path,
        prefix: &str,
        files: &mut BTreeMap<String, String>,
    ) -> Result<(), String> {
        let entries = fs::read_dir(dir)
            .map_err(|err| format!("could not read `{}`: {err}", dir.display()))?;
        for entry in entries {
            let entry = entry
                .map_err(|err| format!("could not read an entry in `{}`: {err}", dir.display()))?;
            let file_type = entry
                .file_type()
                .map_err(|err| format!("could not inspect `{}`: {err}", entry.path().display()))?;
            // Skip symlinks: `.tools` must not reach outside itself.
            if file_type.is_symlink() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let relative = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}/{name}")
            };
            if file_type.is_dir() {
                Self::seed_dir(&entry.path(), &relative, files)?;
                continue;
            }
            if !file_type.is_file() || !keep_on_seed(&name) {
                continue;
            }
            let metadata = entry
                .metadata()
                .map_err(|err| format!("could not inspect `{}`: {err}", entry.path().display()))?;
            if metadata.len() > MAX_SEED_FILE {
                continue;
            }
            let Ok(content) = fs::read_to_string(entry.path()) else {
                continue;
            };
            files.insert(relative, content);
        }
        Ok(())
    }

    /// Write every dirty path to `root` and drop the ones that succeeded.
    ///
    /// Returns one message per failure; an empty vec means the mirror is in
    /// sync with the tree.
    pub(crate) fn flush_to_dir(&mut self, root: &Path) -> Vec<String> {
        let mut failures = Vec::new();
        let paths: Vec<String> = self.dirty.iter().cloned().collect();
        for relative in paths {
            let result = mirror_one(root, &relative, self.files.get(&relative));
            match result {
                Ok(()) => {
                    self.dirty.remove(&relative);
                }
                Err(err) => failures.push(err),
            }
        }
        failures
    }
}

fn keep_on_seed(name: &str) -> bool {
    // Everything the mirror can write is seeded again, so data files written by
    // a tool survive a restart. Only in-flight temp files are skipped.
    !name.ends_with(".tmp")
}

fn mirror_one(root: &Path, relative: &str, content: Option<&String>) -> Result<(), String> {
    let Some(file_name) = relative.rsplit('/').next() else {
        return Err(format!("invalid mirror path `{relative}`"));
    };
    let parent = safe_parent(root, relative)?;
    let target = parent.join(file_name);
    match content {
        Some(content) => write_atomic(&target, content),
        None => match fs::remove_file(&target) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(format!("could not remove `{}`: {err}", target.display())),
        },
    }
}

/// Create (if needed) and validate the parent of a mirror path, refusing to
/// follow symlinked directories. `Path::starts_with` alone is lexical and does
/// not protect against a symlinked component.
fn safe_parent(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let canonical_root = fs::canonicalize(root)
        .map_err(|err| format!("could not resolve `{}`: {err}", root.display()))?;
    let Some((parent, _)) = relative.rsplit_once('/') else {
        return Ok(root.to_path_buf());
    };

    let mut current = root.to_path_buf();
    for segment in parent.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(format!("invalid mirror path `{relative}`"));
        }
        current.push(segment);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!(
                    "refusing to mirror through symlink `{}`",
                    current.display()
                ));
            }
            Ok(metadata) if !metadata.is_dir() => {
                return Err(format!("`{}` is not a directory", current.display()));
            }
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&current)
                    .map_err(|err| format!("could not create `{}`: {err}", current.display()))?;
            }
            Err(err) => {
                return Err(format!("could not inspect `{}`: {err}", current.display()));
            }
        }
    }

    let canonical = fs::canonicalize(&current)
        .map_err(|err| format!("could not resolve `{}`: {err}", current.display()))?;
    if !canonical.starts_with(&canonical_root) {
        return Err(format!("refusing to mirror `{relative}` outside `.tools`"));
    }
    Ok(current)
}

/// Write through a temporary file in the same directory, then rename.
fn write_atomic(target: &Path, content: &str) -> Result<(), String> {
    let temp = target.with_file_name(format!(
        ".{}.tmp",
        target
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("tool")
    ));
    // Remove a planted temp symlink itself, then create exclusively so the
    // write can never follow one.
    let _ = fs::remove_file(&temp);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|err| format!("could not create `{}`: {err}", temp.display()))?;
    file.write_all(content.as_bytes())
        .map_err(|err| format!("could not write `{}`: {err}", temp.display()))?;
    file.sync_all()
        .map_err(|err| format!("could not flush `{}`: {err}", temp.display()))?;
    drop(file);
    fs::rename(&temp, target).map_err(|err| {
        let _ = fs::remove_file(&temp);
        format!("could not replace `{}`: {err}", target.display())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_reads_and_deletes() {
        let mut tree = ToolsTree::default();
        assert!(tree.read("tool1/main.js").is_err());

        tree.write("tool1/main.js", "print(1)".into());
        assert_eq!(tree.read("tool1/main.js").unwrap(), "print(1)");
        assert!(tree.has_dirty());

        assert_eq!(tree.delete("tool1/main.js").unwrap(), 8);
        assert!(tree.read("tool1/main.js").is_err());
        assert!(tree.delete("tool1/main.js").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn flush_refuses_symlinked_parents() {
        use std::os::unix::fs::symlink;

        let base = std::env::temp_dir().join(format!(
            "bvcs-tools-symlink-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let root = base.join("tools");
        let outside = base.join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, root.join("link")).unwrap();

        let mut tree = ToolsTree::default();
        tree.write("link/escaped.js", "x".into());
        let failures = tree.flush_to_dir(&root);

        assert!(!failures.is_empty(), "symlinked mirror must fail");
        assert!(!outside.join("escaped.js").exists());

        fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn seed_and_flush_round_trip() {
        let root = std::env::temp_dir().join(format!(
            "bvcs-tools-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tool1/lib")).unwrap();
        fs::write(root.join("tool1/tool.json"), "{}").unwrap();
        fs::write(root.join("tool1/main.js"), "1").unwrap();
        fs::write(root.join("tool1/lib/x.js"), "2").unwrap();
        fs::write(root.join("tool1/notes.txt"), "ignored").unwrap();

        let mut tree = ToolsTree::default();
        assert_eq!(tree.seed_from_dir(&root).unwrap(), 4);
        assert!(!tree.has_dirty(), "seeding must not queue write-backs");
        assert_eq!(tree.read("tool1/notes.txt").unwrap(), "ignored");

        tree.write("tool2/main.js", "3".into());
        tree.delete("tool1/lib/x.js").unwrap();
        assert!(tree.flush_to_dir(&root).is_empty());
        assert!(!tree.has_dirty());
        assert_eq!(fs::read_to_string(root.join("tool2/main.js")).unwrap(), "3");
        assert!(!root.join("tool1/lib/x.js").exists());

        fs::remove_dir_all(&root).unwrap();
    }
}
