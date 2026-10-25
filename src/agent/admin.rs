//! Agent rename/removal and incoming message delivery.

use super::*;

pub(crate) fn rename_agents(
    mut requests: MessageReader<RenameAgentRequest>,
    mut agents: Query<(&mut Agent, &AgentView)>,
    mut index: ResMut<AgentIndex>,
    mut texts: Query<&mut Text>,
    mut commands: Commands,
) {
    for request in requests.read() {
        let Ok((mut agent, view)) = agents.get_mut(request.agent) else {
            continue;
        };
        if index
            .by_name
            .get(&request.name)
            .is_some_and(|existing| *existing != request.agent)
        {
            continue;
        }

        let old_name = std::mem::replace(&mut agent.name, request.name.clone());
        if index.by_name.get(&old_name) == Some(&request.agent) {
            index.by_name.remove(&old_name);
        }
        index.by_name.insert(request.name.clone(), request.agent);

        commands
            .entity(request.agent)
            .insert(Name::new(request.name.clone()));
        set_text(&mut texts, view.tab_label, request.name.clone());
        set_text(
            &mut texts,
            view.status,
            format!("Renamed to {}", request.name),
        );
    }
}

pub(crate) fn handle_send_agent_message(
    mut messages: MessageReader<SendAgentMessage>,
    mut agents: Query<&mut Agent>,
) {
    for message in messages.read() {
        if let Ok(mut agent) = agents.get_mut(message.target) {
            agent.pending_messages.push_back(if message.verbatim {
                QueuedMessage::Task(message.message.clone())
            } else {
                QueuedMessage::Agent {
                    from: message.from.clone(),
                    text: message.message.clone(),
                }
            });
        }
    }
}

pub(crate) fn handle_task_result(
    mut results: MessageReader<TaskResult>,
    mut agents: Query<&mut Agent>,
) {
    for result in results.read() {
        if let Ok(mut agent) = agents.get_mut(result.agent) {
            agent
                .pending_messages
                .push_back(QueuedMessage::BackgroundResult {
                    task_id: result.task_id,
                    result: result.result.clone(),
                    is_error: result.is_error,
                });
        }
    }
}

pub(crate) fn handle_agent_failed(
    mut failures: MessageReader<AgentFailed>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut rounds: ResMut<ToolRounds>,
    mut texts: Query<&mut Text>,
) {
    for failure in failures.read() {
        rounds.cancel_rounds(failure.agent);
        let Ok((mut agent, mut view)) = agents.get_mut(failure.agent) else {
            continue;
        };
        if agent.current_turn != Some(failure.turn_id) {
            continue;
        }

        error!(
            "agent request for {} failed: {}",
            agent.name, failure.message
        );
        if let Some(assistant) = view.assistant {
            set_text(
                &mut texts,
                assistant.answer_body,
                format!("Request failed: {}", failure.message),
            );
        }
        let reason = if failure.message.starts_with("HTTP ") {
            "endpoint_error"
        } else {
            "agent_error"
        };
        agent.retry_message = agent.rollback_pending_turn();
        log_history_dump(
            &agent,
            reason,
            serde_json::json!({ "error": &failure.message }),
        );
        agent.pending = false;
        agent.current_turn = None;
        agent.streaming_text.clear();
        agent.streaming_thinking.clear();
        view.scroll_to_bottom = true;
        set_text(&mut texts, view.status, "Request failed");
    }
}

pub(crate) fn remove_agents(
    mut requests: MessageReader<RemoveAgentRequest>,
    mut commands: Commands,
    tabs: Query<(Entity, &AgentTab)>,
    mut runtime: ResMut<AgentRuntime>,
    mut rounds: ResMut<ToolRounds>,
) {
    let mut seen = HashSet::new();
    for request in requests.read() {
        let agent = request.agent;
        if !seen.insert(agent) {
            continue;
        }

        // Stop any stream task and drop asynchronous tool work first.
        if let Some(active_turn) = runtime.active.remove(&agent) {
            let _ = active_turn.cancel.send(true);
        }
        rounds.remove_agent(agent);

        // The panel is despawned by the ActiveAgent remove observer (if this
        // agent owned it); tabs are separate roots and are removed here.
        for (tab, tab_agent) in &tabs {
            if tab_agent.agent == agent {
                commands.entity(tab).despawn();
            }
        }

        // Despawning the agent drops its JsRuntimeHandle, which triggers the
        // Discard observer and shuts down that agent's interpreter thread.
        commands.entity(agent).despawn();
    }
}
