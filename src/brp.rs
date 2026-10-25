//! BRP endpoints for driving the harness from an MCP client.
//!
//! These methods intentionally reuse the same runtime paths as the UI and
//! tools: `activate_agent`, `SendAgentMessage`, and `history_view`.

use bevy::{ecs::system::IntoSystem, prelude::*};
use bevy_remote::{
    BrpError, BrpResult, RemoteMethodSystemId, RemoteMethods, RemotePlugin, error_codes,
};
use serde_json::{Value, json};

use crate::{
    agent::{ActiveAgent, Agent, AgentIndex, activate_agent},
    messages::SendAgentMessage,
    tools::{history_view, history_window, optional_i64, optional_u64, sorted_agent_names},
};

/// BRP methods for agent chat control and inspection.
pub(crate) struct HarnessBrpPlugin;

impl Plugin for HarnessBrpPlugin {
    fn build(&self, app: &mut App) {
        if app.is_plugin_added::<RemotePlugin>() {
            register_method(app, "harness/list_agents", list_agents);
            register_method(app, "harness/get_active_agent", get_active_agent);
            register_method(app, "harness/set_active_agent", set_active_agent);
            register_method(app, "harness/send_message", send_message);
            register_method(app, "harness/get_history", get_history);
        } else {
            app.add_plugins(
                RemotePlugin::default()
                    .with_method_main("harness/list_agents", list_agents)
                    .with_method_main("harness/get_active_agent", get_active_agent)
                    .with_method_main("harness/set_active_agent", set_active_agent)
                    .with_method_main("harness/send_message", send_message)
                    .with_method_main("harness/get_history", get_history),
            );
        }
    }
}

fn register_method<M>(
    app: &mut App,
    name: &'static str,
    handler: impl IntoSystem<In<Option<Value>>, BrpResult, M> + 'static,
) {
    let system = app.world_mut().register_system(handler);
    app.world_mut()
        .resource_mut::<RemoteMethods>()
        .insert(name, RemoteMethodSystemId::Instant(system));
}

fn invalid(message: impl Into<String>) -> BrpError {
    BrpError {
        code: error_codes::INVALID_PARAMS,
        message: message.into(),
        data: None,
    }
}

fn agent_json(entity: Entity, agent: &Agent, active: bool) -> Value {
    json!({
        "entity": entity.to_bits(),
        "name": agent.name(),
        "active": active,
        "pending": agent.is_pending(),
        "current_turn": agent.current_turn(),
        "history_len": agent.history().len(),
        "retry_message": agent.retry_message(),
    })
}

fn resolve_agent<'a>(
    params: &Value,
    index: &AgentIndex,
    active: &Query<Entity, With<ActiveAgent>>,
    agents: &'a Query<(Entity, &Agent)>,
) -> Result<(Entity, &'a Agent), BrpError> {
    let entity = if let Some(value) = params.get("agent") {
        let name = value
            .as_str()
            .ok_or_else(|| invalid("`agent` must be a string"))?;
        index.by_name.get(name).copied().ok_or_else(|| {
            invalid(format!(
                "no agent named `{name}`. Available agents: {}",
                sorted_agent_names(index).join(", ")
            ))
        })?
    } else if let Some(value) = params.get("entity") {
        let bits = value
            .as_u64()
            .ok_or_else(|| invalid("`entity` must be a non-negative integer"))?;
        agents
            .iter()
            .find(|(entity, _)| entity.to_bits() == bits)
            .map(|(entity, _)| entity)
            .ok_or_else(|| invalid(format!("no agent with entity {bits}")))?
    } else {
        active
            .single()
            .map_err(|_| invalid("no active agent (expected exactly one)"))?
    };
    agents
        .get(entity)
        .map(|(_, agent)| (entity, agent))
        .map_err(|_| invalid(format!("agent {entity} no longer exists")))
}

fn list_agents(
    _params: In<Option<Value>>,
    agents: Query<(Entity, &Agent)>,
    active: Query<Entity, With<ActiveAgent>>,
) -> BrpResult {
    let active_entity = active.single().ok();
    let mut rows = agents
        .iter()
        .map(|(entity, agent)| agent_json(entity, agent, Some(entity) == active_entity))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left["name"]
            .as_str()
            .unwrap_or("")
            .cmp(right["name"].as_str().unwrap_or(""))
    });
    Ok(Value::Array(rows))
}

fn get_active_agent(
    _params: In<Option<Value>>,
    agents: Query<(Entity, &Agent)>,
    active: Query<Entity, With<ActiveAgent>>,
) -> BrpResult {
    let Some(entity) = active.single().ok() else {
        return Ok(Value::Null);
    };
    let agent = agents
        .get(entity)
        .map_err(|_| invalid(format!("agent {entity} no longer exists")))?
        .1;
    Ok(agent_json(entity, agent, true))
}

fn set_active_agent(
    In(params): In<Option<Value>>,
    mut commands: Commands,
    index: Res<AgentIndex>,
    agents: Query<(Entity, &Agent)>,
    active: Query<Entity, With<ActiveAgent>>,
) -> BrpResult {
    let params = params.unwrap_or(Value::Null);
    let (entity, agent) = resolve_agent(&params, &index, &active, &agents)?;
    activate_agent(&mut commands, &active, Some(entity));
    Ok(agent_json(entity, agent, true))
}

fn send_message(
    In(params): In<Option<Value>>,
    index: Res<AgentIndex>,
    agents: Query<(Entity, &Agent)>,
    active: Query<Entity, With<ActiveAgent>>,
    mut messages: MessageWriter<SendAgentMessage>,
) -> BrpResult {
    let params = params.unwrap_or(Value::Null);
    let text = params
        .get("message")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| invalid("`message` must be a non-empty string"))?;
    let (entity, agent) = resolve_agent(&params, &index, &active, &agents)?;
    let from = params
        .get("from")
        .and_then(Value::as_str)
        .unwrap_or("brp")
        .to_owned();
    let verbatim = params
        .get("verbatim")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    messages.write(SendAgentMessage {
        target: entity,
        from,
        message: text.to_owned(),
        verbatim,
    });
    Ok(json!({
        "queued": true,
        "agent": agent.name(),
        "entity": entity.to_bits(),
        "verbatim": verbatim,
    }))
}

fn get_history(
    In(params): In<Option<Value>>,
    index: Res<AgentIndex>,
    agents: Query<(Entity, &Agent)>,
    active: Query<Entity, With<ActiveAgent>>,
) -> BrpResult {
    let params = params.unwrap_or(Value::Null);
    let (entity, agent) = resolve_agent(&params, &index, &active, &agents)?;
    let position = optional_i64(&params, "position").map_err(invalid)?;
    let amount = optional_u64(&params, "amount").map_err(invalid)?;
    let total = agent.history().len();
    let (start, end) = history_window(total, position, amount);

    Ok(json!({
        "agent": agent.name(),
        "entity": entity.to_bits(),
        "total": total,
        "start": start,
        "end": end,
        "entries": history_view(agent, position, amount, true),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::reflect::AppTypeRegistry;
    use bevy_remote::RemoteMethods;
    use std::any::TypeId;

    use crate::{
        agent::{Agent, AgentIndex, handle_send_agent_message},
        config::Provider,
        tools::{MirrorStatus, ToolRegistryStatus},
        transcript::{AgentView, TextCollapseConfig},
    };

    fn app_with_agents() -> (App, Entity, Entity) {
        let mut app = App::new();
        app.init_resource::<AgentIndex>()
            .add_message::<SendAgentMessage>()
            .add_systems(Update, handle_send_agent_message);

        let first = app
            .world_mut()
            .spawn(Agent::new(
                "Alpha".to_owned(),
                Provider::Responses,
                "model".to_owned(),
            ))
            .id();
        let second = app
            .world_mut()
            .spawn(Agent::new(
                "Beta".to_owned(),
                Provider::Responses,
                "model".to_owned(),
            ))
            .id();
        {
            let mut index = app.world_mut().resource_mut::<AgentIndex>();
            index.by_name.insert("Alpha".to_owned(), first);
            index.by_name.insert("Beta".to_owned(), second);
        }
        app.world_mut().entity_mut(first).insert(ActiveAgent);
        (app, first, second)
    }

    #[test]
    fn harness_methods_are_registered() {
        let mut app = App::new();
        app.add_plugins(HarnessBrpPlugin);

        let methods = app.world().resource::<RemoteMethods>();
        for name in [
            "harness/list_agents",
            "harness/get_active_agent",
            "harness/set_active_agent",
            "harness/send_message",
            "harness/get_history",
        ] {
            assert!(methods.get(name).is_some(), "missing `{name}`");
        }
    }

    #[test]
    fn list_agents_is_sorted_and_marks_the_active_agent() {
        let (mut app, first, _) = app_with_agents();
        let result = app
            .world_mut()
            .run_system_cached_with(list_agents, None::<Value>)
            .unwrap()
            .unwrap();

        assert_eq!(result[0]["name"], "Alpha");
        assert_eq!(result[1]["name"], "Beta");
        assert_eq!(result[0]["entity"], first.to_bits());
        assert_eq!(result[0]["active"], true);
        assert_eq!(result[1]["active"], false);
    }

    #[test]
    fn set_active_agent_moves_the_marker() {
        let (mut app, first, second) = app_with_agents();
        let result = app
            .world_mut()
            .run_system_cached_with(set_active_agent, Some(json!({ "agent": "Beta" })))
            .unwrap()
            .unwrap();
        app.world_mut().flush();

        assert_eq!(result["name"], "Beta");
        assert!(app.world().get::<ActiveAgent>(first).is_none());
        assert!(app.world().get::<ActiveAgent>(second).is_some());
    }

    #[test]
    fn send_message_rejects_a_non_string_agent() {
        let (mut app, _, _) = app_with_agents();
        let err = app
            .world_mut()
            .run_system_cached_with(
                send_message,
                Some(json!({ "agent": 5, "message": "hello" })),
            )
            .unwrap()
            .unwrap_err();

        assert_eq!(err.code, error_codes::INVALID_PARAMS);
    }

    #[test]
    fn send_message_queues_verbatim_text() {
        let (mut app, _, second) = app_with_agents();
        let result = app
            .world_mut()
            .run_system_cached_with(
                send_message,
                Some(json!({ "agent": "Beta", "message": "  hello  " })),
            )
            .unwrap()
            .unwrap();
        assert_eq!(result["queued"], true);
        app.update();

        let agent = app.world().get::<Agent>(second).unwrap();
        assert_eq!(agent.pending_message_count(), 1);
    }

    #[test]
    fn get_history_includes_raw_entries_and_window_bounds() {
        let (mut app, _, _) = app_with_agents();
        let result = app
            .world_mut()
            .run_system_cached_with(get_history, Some(json!({ "agent": "Alpha", "amount": 1 })))
            .unwrap()
            .unwrap();

        assert_eq!(result["total"], 1);
        assert_eq!(result["start"], 0);
        assert_eq!(result["end"], 1);
        assert_eq!(result["entries"][0]["type"], "system_prompt");
        assert!(result["entries"][0]["message"].is_object());
    }

    #[test]
    fn reflected_inspection_types_are_registered() {
        let app = App::new();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        for type_id in [
            TypeId::of::<ActiveAgent>(),
            TypeId::of::<AgentView>(),
            TypeId::of::<TextCollapseConfig>(),
            TypeId::of::<ToolRegistryStatus>(),
            TypeId::of::<MirrorStatus>(),
        ] {
            assert!(registry.get(type_id).is_some(), "type {type_id:?} missing");
        }
    }
}
