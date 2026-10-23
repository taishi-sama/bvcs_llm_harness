//! Parsed virtual-filesystem paths.
//!
//! The flat root keeps its "plain file names, no directories" invariant in
//! code: a root path has exactly one segment. `.tools` is a separate
//! hierarchical mount whose paths are also the relative keys used by the
//! script-tool module resolver and by the on-disk mirror.
//!
//! Everything that turns a model- or script-supplied string into a location
//! goes through [`VfsPath::parse`], so traversal, absolute paths, and empty
//! segments are rejected in one place before any handler sees them.

/// Mount point of the tool-source tree.
pub(crate) const TOOLS_MOUNT: &str = ".tools";

/// Which virtual filesystem a path belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mount {
    /// Flat shared filesystem: exactly one path segment.
    Root,
    /// Hierarchical tool sources: `.tools/<segment>[/<segment>...]`.
    Tools,
}

/// A validated virtual path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VfsPath {
    mount: Mount,
    segments: Vec<String>,
}

impl VfsPath {
    /// Parse and validate a path as it came from a model or a script.
    pub(crate) fn parse(raw: &str) -> Result<Self, String> {
        if raw.is_empty() {
            return Err("path is empty".to_owned());
        }
        if raw.starts_with('/') {
            return Err(format!("path `{raw}` must be relative"));
        }
        if raw.contains('\\') {
            return Err(format!("path `{raw}` must use `/` separators"));
        }
        if raw.chars().any(|ch| ch == '\0' || ch.is_control()) {
            return Err(format!("path `{raw}` contains control characters"));
        }

        let (mount, rest) = match raw.strip_prefix(TOOLS_MOUNT) {
            Some(rest) => {
                let Some(rest) = rest.strip_prefix('/') else {
                    return Err(format!(
                        "`{TOOLS_MOUNT}` is a mount point; use `{TOOLS_MOUNT}/<file>`"
                    ));
                };
                (Mount::Tools, rest)
            }
            None => (Mount::Root, raw),
        };

        if rest.is_empty() || rest.ends_with('/') {
            return Err(format!("path `{raw}` names no file"));
        }
        let mut segments = Vec::new();
        for segment in rest.split('/') {
            match segment {
                "" => return Err(format!("path `{raw}` has an empty segment")),
                "." | ".." => return Err(format!("path `{raw}` may not contain `{segment}`")),
                _ => segments.push(segment.to_owned()),
            }
        }

        if mount == Mount::Root && segments.len() != 1 {
            return Err(format!(
                "path `{raw}` must be a plain file name; `{TOOLS_MOUNT}/` is the only directory"
            ));
        }

        Ok(Self { mount, segments })
    }

    pub(crate) const fn mount(&self) -> Mount {
        self.mount
    }

    pub(crate) fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Path relative to its mount, joined with `/`.
    ///
    /// For [`Mount::Root`] this is the flat file name; for [`Mount::Tools`] it
    /// is the key used by the tool tree and the disk mirror.
    pub(crate) fn relative(&self) -> String {
        self.segments.join("/")
    }

    /// Last segment, which is never empty for a parsed path.
    pub(crate) fn file_name(&self) -> &str {
        self.segments
            .last()
            .map(String::as_str)
            .expect("parsed paths always have at least one segment")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_paths_stay_flat() {
        let path = VfsPath::parse("notes.txt").unwrap();
        assert_eq!(path.mount(), Mount::Root);
        assert_eq!(path.segments(), ["notes.txt"]);
        assert_eq!(path.relative(), "notes.txt");

        assert!(VfsPath::parse("dir/notes.txt").is_err());
        assert!(VfsPath::parse("a/b/c").is_err());
    }

    #[test]
    fn tools_paths_are_hierarchical() {
        let path = VfsPath::parse(".tools/tool1/lib/x.js").unwrap();
        assert_eq!(path.mount(), Mount::Tools);
        assert_eq!(path.segments(), ["tool1", "lib", "x.js"]);
        assert_eq!(path.relative(), "tool1/lib/x.js");
        assert_eq!(path.file_name(), "x.js");
    }

    #[test]
    fn rejects_escapes_and_malformed_paths() {
        for raw in [
            "",
            "/etc/passwd",
            "../secret",
            ".tools/../secret",
            ".tools/tool1/../../x",
            "a//b",
            "trailing/",
            ".tools/",
            ".tools",
            "back\\slash",
            "nul\0byte",
        ] {
            assert!(VfsPath::parse(raw).is_err(), "`{raw}` should be rejected");
        }
    }

    #[test]
    fn dots_inside_names_are_fine() {
        let path = VfsPath::parse(".tools/tool.1/main.js").unwrap();
        assert_eq!(path.relative(), "tool.1/main.js");
        assert_eq!(path.file_name(), "main.js");
    }
}
