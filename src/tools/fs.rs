//! Filesystem tool handlers (`fs_*`).
//!
//! Each handler is a straight move of the matching arm from the legacy
//! executor: same parameter handling, defaults, bounds, and error strings.

use serde_json::Value;

use crate::tools::{
    DEFAULT_GREP_MATCHES, MAX_GREP_CONTEXT, MAX_GREP_MATCHES, bool_option, optional_u64,
    registry::{CallContext, ToolFn},
    required_u64,
    world::ToolWorld,
};

fn fs_read(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _call: &CallContext<'_>,
) -> (String, bool) {
    match arguments.get("path").and_then(Value::as_str) {
        Some(path) => match (
            optional_u64(arguments, "offset"),
            required_u64(arguments, "limit"),
        ) {
            (Ok(offset), Ok(limit)) => match world.fs.read_range(path, offset, Some(limit)) {
                Ok(content) => (content, false),
                Err(err) => (err, true),
            },
            (Err(err), _) | (_, Err(err)) => (err, true),
        },
        None => ("missing required string parameter `path`".to_owned(), true),
    }
}

fn fs_grep(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _call: &CallContext<'_>,
) -> (String, bool) {
    let Some(pattern) = arguments.get("pattern").and_then(Value::as_str) else {
        return (
            "missing required string parameter `pattern`".to_owned(),
            true,
        );
    };
    let path = arguments
        .get("path")
        .and_then(Value::as_str)
        .filter(|path| !path.is_empty());
    let context = match optional_u64(arguments, "context") {
        Ok(context) => context.unwrap_or(0).min(MAX_GREP_CONTEXT) as usize,
        Err(err) => return (err, true),
    };
    let max_matches = match optional_u64(arguments, "max_matches") {
        Ok(max_matches) => max_matches
            .unwrap_or(DEFAULT_GREP_MATCHES)
            .min(MAX_GREP_MATCHES) as usize,
        Err(err) => return (err, true),
    };
    match world.fs.grep(
        pattern,
        path,
        context,
        bool_option(arguments, "ignore_case", false),
        max_matches,
    ) {
        Ok(output) => (output, false),
        Err(err) => (err, true),
    }
}

fn fs_sed(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _call: &CallContext<'_>,
) -> (String, bool) {
    match arguments.get("path").and_then(Value::as_str) {
        Some(path) => {
            let start = match optional_u64(arguments, "start_line") {
                Ok(start) => start,
                Err(err) => return (err, true),
            };
            let end = match optional_u64(arguments, "end_line") {
                Ok(end) => end,
                Err(err) => return (err, true),
            };
            match world.fs.line_range(
                path,
                start,
                end,
                bool_option(arguments, "line_numbers", true),
            ) {
                Ok(output) => (output, false),
                Err(err) => (err, true),
            }
        }
        None => ("missing required string parameter `path`".to_owned(), true),
    }
}

fn fs_ls(
    world: &mut ToolWorld<'_, '_>,
    _arguments: &Value,
    _call: &CallContext<'_>,
) -> (String, bool) {
    (world.fs.list(), false)
}

fn fs_write(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _call: &CallContext<'_>,
) -> (String, bool) {
    match (
        arguments.get("path").and_then(Value::as_str),
        arguments.get("content").and_then(Value::as_str),
    ) {
        (Some(path), Some(content)) => match world.fs.write(path, content.to_owned()) {
            Ok(message) => (message, false),
            Err(err) => (err, true),
        },
        _ => (
            "missing required string parameters `path` and `content`".to_owned(),
            true,
        ),
    }
}

fn fs_edit_lines(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _call: &CallContext<'_>,
) -> (String, bool) {
    match (
        arguments.get("path").and_then(Value::as_str),
        arguments.get("start_line").and_then(Value::as_u64),
        arguments.get("end_line").and_then(Value::as_u64),
        arguments.get("content").and_then(Value::as_str),
    ) {
        (Some(path), Some(start), Some(end), Some(content)) => {
            match world.fs.edit_lines(path, start, end, content.to_owned()) {
                Ok(message) => (message, false),
                Err(err) => (err, true),
            }
        }
        _ => (
            "missing required parameters `path`, `start_line`, `end_line`, `content`".to_owned(),
            true,
        ),
    }
}

fn fs_edit_pattern(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _call: &CallContext<'_>,
) -> (String, bool) {
    match (
        arguments.get("path").and_then(Value::as_str),
        arguments.get("pattern").and_then(Value::as_str),
        arguments.get("replacement").and_then(Value::as_str),
    ) {
        (Some(path), Some(pattern), Some(replacement)) => {
            match world.fs.edit_pattern(path, pattern, replacement) {
                Ok(message) => (message, false),
                Err(err) => (err, true),
            }
        }
        _ => (
            "missing required string parameters `path`, `pattern`, `replacement`".to_owned(),
            true,
        ),
    }
}

fn fs_delete(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _call: &CallContext<'_>,
) -> (String, bool) {
    match arguments.get("path").and_then(Value::as_str) {
        Some(path) => match world.fs.delete(path) {
            Ok(message) => (message, false),
            Err(err) => (err, true),
        },
        None => ("missing required string parameter `path`".to_owned(), true),
    }
}

/// Handler lookup used by `catalog::execution`.
pub(crate) fn handler(name: &str) -> Option<ToolFn> {
    match name {
        "fs_read" => Some(fs_read),
        "fs_grep" => Some(fs_grep),
        "fs_sed" => Some(fs_sed),
        "fs_ls" => Some(fs_ls),
        "fs_write" => Some(fs_write),
        "fs_edit_lines" => Some(fs_edit_lines),
        "fs_edit_pattern" => Some(fs_edit_pattern),
        "fs_delete" => Some(fs_delete),
        _ => None,
    }
}
