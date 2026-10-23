//! Agent lifecycle and history tool handlers.
//!
//! Extracted from the legacy match in `tools.rs`; `handler` is the dispatch
//! entry point used by `catalog::execution`.

use bevy::prelude::Entity;
use serde_json::Value;

use crate::{
    agent::{MAX_AGENTS, create_agent},
    messages::{RemoveAgentRequest, RenameAgentRequest, SendAgentMessage},
    tools::{
        entry_sender, entry_type, history_agent, history_window, initial_task_message,
        optional_i64, optional_u64,
        registry::{CallContext, ToolFn},
        world::ToolWorld,
    },
};

fn send_message_to_agent(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _source_entity: Entity,
    source_agent: &str,
) -> (String, bool) {
    let target_name = arguments.get("agent").and_then(Value::as_str);
    let message = arguments.get("message").and_then(Value::as_str);
    match (target_name, message) {
        (Some(target_name), Some(_)) if target_name == source_agent => {
            ("Cannot send a message to the same agent.".to_owned(), true)
        }
        (Some(target_name), Some(message)) => {
            if let Some(target) = world.index.by_name.get(target_name) {
                let target = *target;
                world.send_agent_message.write(SendAgentMessage {
                    target,
                    from: source_agent.to_owned(),
                    message: message.to_owned(),
                    verbatim: false,
                });
                (
                    format!(
                        "Queued message for `{target_name}`. Continue your task or end turn to await reply."
                    ),
                    false,
                )
            } else {
                let mut names = world.index.by_name.keys().cloned().collect::<Vec<_>>();
                names.sort();
                (
                    format!(
                        "No agent named `{target_name}`. Available agents: {}",
                        names.join(", ")
                    ),
                    true,
                )
            }
        }
        _ => (
            "missing required string parameters `agent` and `message`".to_owned(),
            true,
        ),
    }
}

fn list_agents(
    world: &mut ToolWorld<'_, '_>,
    _arguments: &Value,
    _source_entity: Entity,
    source_agent: &str,
) -> (String, bool) {
    let mut names = world.index.by_name.keys().cloned().collect::<Vec<_>>();
    names.sort();
    let available = names
        .into_iter()
        .map(|name| {
            if name == source_agent {
                format!("{name} (you)")
            } else {
                name
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    (format!("Available agents: {available}"), false)
}

/// Agent targeted by a history read. An omitted or blank `agent` means the
/// caller's own history, matching the optional schema parameter.
fn history_target_name<'a>(arguments: &'a Value, source_agent: &'a str) -> &'a str {
    arguments
        .get("agent")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or(source_agent)
}

fn get_agent_history(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _source_entity: Entity,
    source_agent: &str,
) -> (String, bool) {
    let target_name = history_target_name(arguments, source_agent);
    let target = match history_agent(target_name, &world.index, &world.agents) {
        Ok(target) => target,
        Err(err) => return (err, true),
    };
    let position = match optional_i64(arguments, "position") {
        Ok(position) => position,
        Err(err) => return (err, true),
    };
    let amount = match optional_u64(arguments, "amount") {
        Ok(amount) => amount,
        Err(err) => return (err, true),
    };

    let history = target.history();
    let (start, end) = history_window(history.len(), position, amount);
    let items = history[start..end]
        .iter()
        .enumerate()
        .map(|(offset, entry)| {
            serde_json::json!({
                "type": entry_type(entry),
                "sender": entry_sender(entry, target.name()),
                "message_index": start + offset,
            })
        })
        .collect::<Vec<_>>();

    match serde_json::to_string_pretty(&items) {
        Ok(json) => (json, false),
        Err(err) => (format!("could not serialize history: {err}"), true),
    }
}

fn get_agent_message(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _source_entity: Entity,
    source_agent: &str,
) -> (String, bool) {
    let target_name = history_target_name(arguments, source_agent);
    let target = match history_agent(target_name, &world.index, &world.agents) {
        Ok(target) => target,
        Err(err) => return (err, true),
    };
    let Some(message_index) = arguments.get("index").and_then(Value::as_u64) else {
        return (
            "missing required non-negative integer parameter `index`".to_owned(),
            true,
        );
    };

    let history = target.history();
    let Some(entry) = history.get(message_index as usize) else {
        return (
            format!(
                "index {message_index} is out of range; `{target_name}` has {} messages (0..{})",
                history.len(),
                history.len().saturating_sub(1)
            ),
            true,
        );
    };

    match serde_json::to_string_pretty(entry) {
        Ok(json) => (json, false),
        Err(err) => (format!("could not serialize message: {err}"), true),
    }
}

fn spawn_agent(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    _source_entity: Entity,
    source_agent: &str,
) -> (String, bool) {
    if world.index.by_name.len() >= MAX_AGENTS {
        return (
            format!("Agent limit reached ({MAX_AGENTS}); remove an agent first."),
            true,
        );
    }

    let requested_name = arguments
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty());
    let prompt = arguments
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty());

    if let Some(name) = requested_name
        && world.index.by_name.contains_key(name)
    {
        return (format!("An agent named `{name}` already exists."), true);
    }

    let mut number = world.layout.next_number;
    let name = if let Some(name) = requested_name {
        name.to_owned()
    } else {
        let mut candidate = format!("LLM Agent {number}");
        while world.index.by_name.contains_key(&candidate) {
            number += 1;
            candidate = format!("LLM Agent {number}");
        }
        candidate
    };
    world.layout.next_number = number + 1;

    let agent = create_agent(
        &mut world.commands,
        &world.layout,
        &world.config,
        name.clone(),
        number as i32,
    );
    world.index.by_name.insert(name.clone(), agent);

    if let Some(prompt) = prompt {
        // Delivered verbatim (no cross-agent envelope), so the new agent's
        // first user message starts exactly with this text.
        world.send_agent_message.write(SendAgentMessage {
            target: agent,
            from: source_agent.to_owned(),
            message: initial_task_message(source_agent, prompt),
            verbatim: true,
        });
        (
            format!("Spawned `{name}` in a new tab with its initial task."),
            false,
        )
    } else {
        (format!("Spawned `{name}` in a new tab."), false)
    }
}

fn rename_self(
    world: &mut ToolWorld<'_, '_>,
    arguments: &Value,
    source_entity: Entity,
    source_agent: &str,
) -> (String, bool) {
    let Some(new_name) = arguments
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
    else {
        return (
            "missing required non-empty string parameter `name`".to_owned(),
            true,
        );
    };

    if new_name == source_agent {
        return ("Name unchanged.".to_owned(), false);
    }
    if world.index.by_name.contains_key(new_name) {
        return (format!("An agent named `{new_name}` already exists."), true);
    }

    // Reserve the name immediately so two renames in one tool round
    // cannot both claim it.
    world.index.by_name.remove(source_agent);
    world
        .index
        .by_name
        .insert(new_name.to_owned(), source_entity);
    world.rename_agent.write(RenameAgentRequest {
        agent: source_entity,
        name: new_name.to_owned(),
    });
    (format!("Renamed this agent to `{new_name}`."), false)
}

fn end_agent(
    world: &mut ToolWorld<'_, '_>,
    _arguments: &Value,
    source_entity: Entity,
    _source_agent: &str,
) -> (String, bool) {
    world.remove_agent.write(RemoveAgentRequest {
        agent: source_entity,
    });
    ("Ending this agent...".to_owned(), false)
}

/// Handler lookup used by `catalog::execution`.
///
/// Each wrapper adapts the per-call [`CallContext`] to the source entity/name
/// pair the handlers were extracted with.
macro_rules! agent_tools {
    ($($name:literal => $wrapper:ident = $handler:ident),* $(,)?) => {
        $(
            fn $wrapper(
                world: &mut ToolWorld<'_, '_>,
                arguments: &Value,
                call: &CallContext<'_>,
            ) -> (String, bool) {
                $handler(world, arguments, call.source_entity, call.source_agent)
            }
        )*

        pub(crate) fn handler(name: &str) -> Option<ToolFn> {
            Some(match name {
                $($name => $wrapper,)*
                _ => return None,
            })
        }
    };
}

agent_tools! {
    "send_message_to_agent" => send_message_to_agent_tool = send_message_to_agent,
    "list_agents" => list_agents_tool = list_agents,
    "get_agent_history" => get_agent_history_tool = get_agent_history,
    "get_agent_message" => get_agent_message_tool = get_agent_message,
    "spawn_agent" => spawn_agent_tool = spawn_agent,
    "rename_self" => rename_self_tool = rename_self,
    "end_agent" => end_agent_tool = end_agent,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_history_agent_defaults_to_the_caller() {
        assert_eq!(
            history_target_name(&Value::Null, "LLM Agent 1"),
            "LLM Agent 1"
        );
        assert_eq!(
            history_target_name(&serde_json::json!({ "agent": "   " }), "LLM Agent 1"),
            "LLM Agent 1"
        );
        assert_eq!(
            history_target_name(&serde_json::json!({ "agent": "Worker" }), "LLM Agent 1"),
            "Worker"
        );
    }
}
