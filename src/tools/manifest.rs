//! Script-tool manifests (`.tools/<dir>/tool.json`, `library.json`).
//!
//! Manifests are inert data: the advertisement is rebuilt on every model
//! request, so a broken script must not be able to break or slow request
//! building. A directory is a tool only when its `tool.json` parses; anything
//! else under `.tools` is an importable library.

use std::collections::{HashMap, HashSet};

use llm_api::generic_tool::{ParameterDescription, ParameterType, Tool};
use serde::Deserialize;

use crate::tools::{is_js_tool, tree::ToolsTree};

/// Default entry module for a tool directory.
const DEFAULT_ENTRY: &str = "main.js";
/// Default exported function the host calls.
const DEFAULT_EXPORT: &str = "main";
/// Advertisement order for script tools, after the built-ins.
const DEFAULT_ORDER: u32 = 500;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ManifestType {
    String,
    Number,
    Boolean,
}

impl From<ManifestType> for ParameterType {
    fn from(value: ManifestType) -> Self {
        match value {
            ManifestType::String => ParameterType::String,
            ManifestType::Number => ParameterType::Number,
            ManifestType::Boolean => ParameterType::Boolean,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawParameter {
    r#type: ManifestType,
    description: String,
    #[serde(default = "default_true")]
    required: bool,
}

const fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct RawTool {
    name: String,
    description: Option<String>,
    #[serde(default)]
    parameters: HashMap<String, RawParameter>,
    #[serde(default = "default_entry")]
    entry: String,
    #[serde(default = "default_export")]
    export: String,
    #[serde(default = "default_order")]
    order: u32,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

fn default_entry() -> String {
    DEFAULT_ENTRY.to_owned()
}

fn default_export() -> String {
    DEFAULT_EXPORT.to_owned()
}

const fn default_order() -> u32 {
    DEFAULT_ORDER
}

#[derive(Debug, Deserialize)]
struct RawLibrary {
    #[serde(default = "default_index")]
    entry: String,
}

fn default_index() -> String {
    "index.js".to_owned()
}

/// A validated script tool ready to become an entity.
pub(crate) struct ScriptToolSpec {
    pub name: String,
    pub definition: Tool,
    /// Path relative to `.tools`, e.g. `tool1/main.js`.
    pub entry: String,
    #[allow(dead_code, reason = "consumed by the script executor in phase 3")]
    pub export: String,
    pub order: u32,
    #[allow(dead_code, reason = "consumed by the script executor in phase 3")]
    pub timeout_ms: Option<u64>,
}

/// An importable library declared by `library.json`.
pub(crate) struct LibrarySpec {
    #[allow(dead_code, reason = "consumed by the module resolver in phase 3")]
    pub directory: String,
    /// Path relative to `.tools`.
    #[allow(dead_code, reason = "consumed by the module resolver in phase 3")]
    pub entry: String,
}

/// Scan the tree for tool manifests. Invalid manifests are reported and
/// skipped; they never abort the scan.
pub(crate) fn scan_tools(tree: &ToolsTree, warnings: &mut Vec<String>) -> Vec<ScriptToolSpec> {
    let mut reserved: HashSet<String> = crate::tools::definitions()
        .into_iter()
        .map(|tool| tool.name)
        .collect();
    let mut specs = Vec::new();

    let mut manifest_paths: Vec<&str> = tree
        .iter()
        .map(|(path, _)| path)
        .filter(|path| path.ends_with("/tool.json"))
        .collect();
    manifest_paths.sort_unstable();

    for manifest_path in manifest_paths {
        let directory = manifest_path.trim_end_matches("/tool.json");
        if directory.contains('/') {
            warnings.push(format!(
                "`{manifest_path}` is nested deeper than `.tools/<tool>/`; skipped"
            ));
            continue;
        }
        let Some((_, raw)) = tree.iter().find(|(path, _)| *path == manifest_path) else {
            continue;
        };
        let parsed: RawTool = match serde_json::from_str(raw) {
            Ok(parsed) => parsed,
            Err(err) => {
                warnings.push(format!("`{manifest_path}` is invalid: {err}"));
                continue;
            }
        };
        if parsed.name != directory {
            warnings.push(format!(
                "`{manifest_path}` names `{}` but lives in `{directory}`; names must match",
                parsed.name
            ));
            continue;
        }
        if reserved.contains(&parsed.name) {
            warnings.push(format!(
                "`{manifest_path}` reuses name `{}`; skipped",
                parsed.name
            ));
            continue;
        }
        let entry = format!("{directory}/{}", parsed.entry);
        if tree.read(&entry).is_err() {
            warnings.push(format!(
                "`{manifest_path}` points at missing entry `{}`; skipped",
                parsed.entry
            ));
            continue;
        }
        if is_js_tool(&parsed.name) {
            warnings.push(format!(
                "`{manifest_path}` collides with a JavaScript host tool; skipped"
            ));
            continue;
        }

        let parameters = parsed
            .parameters
            .into_iter()
            .map(|(name, raw)| {
                (
                    name,
                    ParameterDescription {
                        r#type: raw.r#type.into(),
                        description: raw.description,
                        required: raw.required,
                    },
                )
            })
            .collect();

        reserved.insert(parsed.name.clone());
        specs.push(ScriptToolSpec {
            definition: Tool {
                name: parsed.name,
                description: Some(
                    parsed
                        .description
                        .unwrap_or_else(|| format!("Tool defined in `.tools/{directory}`")),
                ),
                parameters,
            },
            name: directory.to_owned(),
            entry,
            export: parsed.export,
            order: parsed.order,
            timeout_ms: parsed.timeout_ms,
        });
    }

    specs.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.name.cmp(&b.name)));
    specs
}

/// Resolve each top-level directory to its entry module for the script-tool
/// module resolver. Manifests win; otherwise `index.js` then `main.js`.
pub(crate) fn entry_index(tree: &ToolsTree) -> HashMap<String, String> {
    let mut index = HashMap::new();

    for (path, content) in tree.iter() {
        if let Some(directory) = path.strip_suffix("/tool.json")
            && !directory.contains('/')
        {
            if let Ok(raw) = serde_json::from_str::<RawTool>(content) {
                let entry = format!("{directory}/{}", raw.entry);
                if tree.read(&entry).is_ok() {
                    index.insert(directory.to_owned(), entry);
                }
            }
            continue;
        }
        if let Some(directory) = path.strip_suffix("/library.json")
            && !directory.contains('/')
            && let Ok(raw) = serde_json::from_str::<RawLibrary>(content)
        {
            let entry = format!("{directory}/{}", raw.entry);
            if tree.read(&entry).is_ok() {
                index.insert(directory.to_owned(), entry);
            }
        }
    }

    let mut directories: Vec<String> = tree
        .iter()
        .filter_map(|(path, _)| {
            path.split_once('/')
                .map(|(directory, _)| directory.to_owned())
        })
        .collect();
    directories.sort_unstable();
    directories.dedup();
    for directory in directories {
        if index.contains_key(&directory) {
            continue;
        }
        for name in ["index.js", "main.js"] {
            let entry = format!("{directory}/{name}");
            if tree.read(&entry).is_ok() {
                index.insert(directory, entry);
                break;
            }
        }
    }

    index
}

/// Scan `library.json` manifests. Libraries have no entry in the model's tool
/// list; they are importable from tool scripts.
pub(crate) fn scan_libraries(tree: &ToolsTree, warnings: &mut Vec<String>) -> Vec<LibrarySpec> {
    let mut manifest_paths: Vec<&str> = tree
        .iter()
        .map(|(path, _)| path)
        .filter(|path| path.ends_with("/library.json"))
        .collect();
    manifest_paths.sort_unstable();

    let mut libraries = Vec::new();
    for manifest_path in manifest_paths {
        let directory = manifest_path.trim_end_matches("/library.json").to_owned();
        if directory.contains('/') {
            warnings.push(format!(
                "`{manifest_path}` is nested deeper than `.tools/<library>/`; skipped"
            ));
            continue;
        }
        let Some((_, raw)) = tree.iter().find(|(path, _)| *path == manifest_path) else {
            continue;
        };
        let entry = match serde_json::from_str::<RawLibrary>(raw) {
            Ok(parsed) => parsed.entry,
            Err(err) => {
                warnings.push(format!("`{manifest_path}` is invalid: {err}"));
                continue;
            }
        };
        let entry = format!("{directory}/{entry}");
        if tree.read(&entry).is_err() {
            warnings.push(format!(
                "`{manifest_path}` points at missing entry `{}`; skipped",
                directory
            ));
            continue;
        }
        libraries.push(LibrarySpec { directory, entry });
    }
    libraries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(tree: &mut ToolsTree, path: &str, content: &str) {
        tree.write(path, content.to_owned());
    }

    #[test]
    fn scans_valid_tools_and_skips_broken_manifests() {
        let mut tree = ToolsTree::default();
        write(
            &mut tree,
            "good/tool.json",
            r#"{
                "name": "good",
                "description": "Does good things",
                "parameters": {
                    "query": {"type": "string", "description": "What to look for"},
                    "limit": {"type": "number", "description": "Max", "required": false}
                }
            }"#,
        );
        write(&mut tree, "good/main.js", "export function main() {}");
        write(&mut tree, "broken/tool.json", "{ not json");
        write(&mut tree, "broken/main.js", "");
        write(&mut tree, "missing/tool.json", r#"{"name":"missing"}"#);
        write(&mut tree, "wrong/tool.json", r#"{"name":"other"}"#);
        write(&mut tree, "wrong/main.js", "");

        let mut warnings = Vec::new();
        let specs = scan_tools(&tree, &mut warnings);

        assert_eq!(specs.len(), 1);
        let spec = &specs[0];
        assert_eq!(spec.name, "good");
        assert_eq!(spec.entry, "good/main.js");
        assert_eq!(spec.export, "main");
        assert_eq!(spec.definition.name, "good");
        assert!(spec.definition.parameters["query"].required);
        assert!(!spec.definition.parameters["limit"].required);
        assert_eq!(warnings.len(), 3, "{warnings:?}");
    }

    #[test]
    fn scans_libraries() {
        let mut tree = ToolsTree::default();
        write(&mut tree, "lib/library.json", r#"{"entry":"index.js"}"#);
        write(&mut tree, "lib/index.js", "export const x = 1;");
        write(&mut tree, "bare/index.js", "export const y = 2;");

        let mut warnings = Vec::new();
        let libraries = scan_libraries(&tree, &mut warnings);

        assert!(warnings.is_empty());
        assert_eq!(libraries.len(), 1);
        assert_eq!(libraries[0].directory, "lib");
        assert_eq!(libraries[0].entry, "lib/index.js");
    }
}
