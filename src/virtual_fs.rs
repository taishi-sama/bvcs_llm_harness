//! Shared in-memory flat filesystem.

use std::collections::HashMap;

use bevy::prelude::Resource;

/// Shared in-memory flat filesystem. Every agent reads and writes the same
/// files, and nothing touches the host filesystem.
#[derive(Resource, Default)]
pub(crate) struct VirtualFs {
    files: HashMap<String, String>,
}

fn split_virtual_file(content: &str) -> (Vec<String>, bool) {
    (
        content.lines().map(str::to_owned).collect(),
        content.ends_with('\n'),
    )
}

impl VirtualFs {
    /// The virtual filesystem has no directories: a path is a plain file name.
    fn normalize_path(path: &str) -> Result<String, String> {
        let path = path.trim();
        if path.is_empty() {
            return Err("path must not be empty".to_owned());
        }
        if path.contains('/') || path.contains('\\') {
            return Err(format!(
                "invalid path `{path}`: the virtual filesystem is flat (no directories)"
            ));
        }
        if path == "." || path == ".." {
            return Err(format!("invalid file name `{path}`"));
        }
        Ok(path.to_owned())
    }

    pub(crate) fn list(&self) -> String {
        if self.files.is_empty() {
            return "Virtual filesystem is empty.".to_owned();
        }

        let mut names: Vec<&String> = self.files.keys().collect();
        names.sort_unstable();
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
    }

    pub(crate) fn read(&self, path: &str) -> Result<String, String> {
        self.read_range(path, None, None)
    }

    /// Read a byte range. `seek` and `len` are byte offsets into the UTF-8
    /// contents; omitted bounds read from the start / to the end.
    pub(crate) fn read_range(
        &self,
        path: &str,
        seek: Option<u64>,
        len: Option<u64>,
    ) -> Result<String, String> {
        let path = Self::normalize_path(path)?;
        let content = self
            .files
            .get(&path)
            .ok_or_else(|| format!("No such file: `{path}`. Use fs_write to create it."))?;

        let start = usize::try_from(seek.unwrap_or(0))
            .map_err(|_| "`seek` is too large for this platform".to_owned())?;
        if start > content.len() {
            return Err(format!(
                "`seek` ({start}) is past the end of `{path}` ({} bytes)",
                content.len()
            ));
        }
        if !content.is_char_boundary(start) {
            return Err(format!(
                "`seek` ({start}) is not on a UTF-8 character boundary in `{path}`"
            ));
        }

        let end = match len {
            Some(len) => {
                let len = usize::try_from(len)
                    .map_err(|_| "`len` is too large for this platform".to_owned())?;
                start.saturating_add(len).min(content.len())
            }
            None => content.len(),
        };
        if !content.is_char_boundary(end) {
            return Err(format!(
                "`seek` + `len` ends at byte {end}, which is not on a UTF-8 character boundary in `{path}`"
            ));
        }

        Ok(content[start..end].to_owned())
    }

    pub(crate) fn write(&mut self, path: &str, content: String) -> Result<String, String> {
        let path = Self::normalize_path(path)?;
        let lines = content.lines().count();
        self.files.insert(path.clone(), content);
        Ok(format!("Wrote `{path}` ({lines} lines)."))
    }

    pub(crate) fn delete(&mut self, path: &str) -> Result<String, String> {
        let path = Self::normalize_path(path)?;
        let content = self
            .files
            .remove(&path)
            .ok_or_else(|| format!("No such file: `{path}`"))?;
        Ok(format!("Deleted `{path}` ({} bytes).", content.len()))
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
        let path = Self::normalize_path(path)?;
        let original = self
            .files
            .get(&path)
            .cloned()
            .ok_or_else(|| format!("No such file: `{path}`"))?;
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
                "line range {start_line}-{end_line} is out of bounds; `{path}` has {} line(s)",
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
        self.files.insert(path.clone(), updated);
        Ok(format!(
            "Replaced lines {start_line}-{end_line} in `{path}`; file now has {line_count} line(s)."
        ))
    }

    /// Replaces a literal substring only when it occurs exactly once.
    pub(crate) fn edit_pattern(
        &mut self,
        path: &str,
        pattern: &str,
        replacement: &str,
    ) -> Result<String, String> {
        let path = Self::normalize_path(path)?;
        if pattern.is_empty() {
            return Err("pattern must not be empty".to_owned());
        }
        let original = self
            .files
            .get(&path)
            .cloned()
            .ok_or_else(|| format!("No such file: `{path}`"))?;

        match original.matches(pattern).count() {
            0 => Err(format!("Pattern not found in `{path}`.")),
            1 => {
                self.files
                    .insert(path.clone(), original.replacen(pattern, replacement, 1));
                Ok(format!(
                    "Replaced the single match of the pattern in `{path}`."
                ))
            }
            count => Err(format!(
                "Pattern matches {count} times in `{path}`; refusing to edit because exactly one match is required."
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        fs.edit_pattern("notes.txt", "TWO", "two").unwrap();
        assert_eq!(fs.read("notes.txt").unwrap(), "one\ntwo\nthree\n");

        fs.delete("notes.txt").unwrap();
        assert!(fs.read("notes.txt").is_err());
        assert_eq!(fs.list(), "Virtual filesystem is empty.");
    }

    #[test]
    fn virtual_fs_edits_are_unambiguous_and_bounded() {
        let mut fs = VirtualFs::default();
        fs.write("a.txt", "same same\nsecond".into()).unwrap();

        let err = fs
            .edit_pattern("a.txt", "same", "other")
            .expect_err("ambiguous pattern must fail");
        assert!(err.contains("2 times"), "{err}");

        assert!(fs.edit_pattern("a.txt", "missing", "x").is_err());
        assert!(fs.edit_lines("a.txt", 0, 1, "x".into()).is_err());
        assert!(fs.edit_lines("a.txt", 3, 3, "x".into()).is_err());
        assert!(fs.edit_lines("a.txt", 2, 1, "x".into()).is_err());
        assert!(fs.write("dir/file.txt", "x".into()).is_err());
        assert!(fs.read("..").is_err());
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
        assert!(fs.read_range("u.txt", None, Some(1)).is_err());
    }
}
