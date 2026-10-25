//! `search_tools`: progressive disclosure for tools that are not advertised.
//!
//! Returns a compact one-line signature per match instead of a full JSON
//! schema: enough to call the tool, roughly a fifth of the tokens.

use llm_api::generic_tool::{ParameterType, Tool};
use regex::RegexBuilder;
use serde_json::Value;

use crate::tools::registry::ToolEntry;
use crate::tools::{is_truthy, world::ToolWorld};

const DEFAULT_MAX: u64 = 20;
const MAX_RESULTS: u64 = 100;
/// Bounds regex compile time on model-authored patterns.
const MAX_FILTER_LEN: usize = 256;

pub(crate) fn search_tools(
    world: &mut ToolWorld<'_, '_>,
    args: &Value,
    _call: &crate::tools::registry::CallContext<'_>,
) -> (String, bool) {
    let filter = args
        .get("filter")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if filter.len() > MAX_FILTER_LEN {
        return (
            format!("`filter` is too long (max {MAX_FILTER_LEN} characters)"),
            true,
        );
    }

    let requested_max = args.get("max").and_then(Value::as_u64);
    if requested_max == Some(0) {
        return ("`max` must be at least 1".to_owned(), true);
    }
    let max = requested_max.unwrap_or(DEFAULT_MAX).min(MAX_RESULTS) as usize;
    let descriptions = match args.get("search_descriptions") {
        None | Some(Value::Null) => true,
        value => is_truthy(value),
    };

    let regex = if filter.is_empty() {
        None
    } else {
        match RegexBuilder::new(filter).case_insensitive(true).build() {
            Ok(regex) => Some(regex),
            Err(err) => return (format!("invalid `filter` regular expression: {err}"), true),
        }
    };

    let mut hits: Vec<&ToolEntry> = world
        .tools
        .iter()
        .filter(|(_, _, _, disabled, _)| disabled.is_none())
        .filter(|(_, entry, _, _, _)| match regex.as_ref() {
            None => true,
            Some(regex) => {
                regex.is_match(&entry.definition.name)
                    || (descriptions
                        && entry
                            .definition
                            .description
                            .as_deref()
                            .is_some_and(|text| regex.is_match(text)))
            }
        })
        .map(|(_, entry, _, _, _)| entry)
        .collect();

    hits.sort_by(|a, b| {
        a.order
            .cmp(&b.order)
            .then_with(|| a.definition.name.cmp(&b.definition.name))
    });

    if hits.is_empty() {
        return (format!("No tools matched `{filter}`."), false);
    }

    let total = hits.len();
    let shown = total.min(max);
    let mut output = hits
        .into_iter()
        .take(shown)
        .map(|entry| signature(&entry.definition))
        .collect::<Vec<_>>()
        .join("\n");
    if total > shown {
        output.push_str(&format!("\n({shown} of {total} matches shown)"));
    }
    if let Some(requested) = requested_max
        && requested > MAX_RESULTS
    {
        output.push_str(&format!(
            "\n(`max` clamped from {requested} to {MAX_RESULTS})"
        ));
    }
    (output, false)
}

/// `name(required: type, optional?: type) — description`
fn signature(tool: &Tool) -> String {
    let mut names: Vec<&str> = tool.parameters.keys().map(String::as_str).collect();
    names.sort_unstable();
    let parameters = names
        .iter()
        .map(|name| {
            let parameter = &tool.parameters[*name];
            format!(
                "{name}{}: {}",
                if parameter.required { "" } else { "?" },
                type_name(&parameter.r#type)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{}({parameters}) — {}",
        tool.name,
        tool.description.as_deref().unwrap_or("")
    )
}

const fn type_name(parameter: &ParameterType) -> &'static str {
    match parameter {
        ParameterType::String => "string",
        ParameterType::Number => "number",
        ParameterType::Boolean => "boolean",
        ParameterType::Object => "object",
    }
}
