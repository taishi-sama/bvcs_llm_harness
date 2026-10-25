//! Starting a turn and committing a finished turn.

use super::*;

#[allow(
    clippy::too_many_arguments,
    reason = "this is the central turn-start system"
)]
pub(crate) fn start_agent_turn(
    mut requests: MessageReader<StartAgentTurn>,
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut runtime: ResMut<AgentRuntime>,
    mut texts: Query<&mut Text>,
    collapse: Res<TextCollapseConfig>,
    tokio_runtime: Res<TokioTasksRuntime>,
    app_config: Res<AppConfig>,
    tools: AdvertisedTools,
    mut commands: Commands,
) {
    for request in requests.read() {
        if runtime.active.contains_key(&request.agent) {
            continue;
        }
        let Ok((mut agent, mut view)) = agents.get_mut(request.agent) else {
            continue;
        };

        agent.retry_message = None;
        agent.previous_request_body = agent.last_request_body.take();
        let config = agent.model_config(advertised_tools(&tools));
        let body = match request_body(agent.provider, &agent.history, &config) {
            Ok(body) => body,
            Err(err) => {
                error!("could not build request for {}: {err:#}", agent.name);
                agent.retry_message = agent.rollback_pending_turn();
                agent.pending = false;
                log_history_dump(
                    &agent,
                    "request_build_error",
                    serde_json::json!({ "error": format!("{err:#}") }),
                );
                set_text(
                    &mut texts,
                    view.status,
                    format!("Could not build request: {err:#}"),
                );
                continue;
            }
        };
        agent.last_request_body = Some(body.clone());

        let provider = agent.provider;
        let kind = config.kind.clone();
        let turn_id = runtime.next_turn_id;
        runtime.next_turn_id += 1;
        agent.current_turn = Some(turn_id);
        agent.pending = true;
        agent.streaming_text.clear();
        agent.streaming_thinking.clear();

        if view.transcript_visible {
            begin_live_assistant(&mut commands, &mut view, &agent, &collapse);
        } else {
            view.assistant = None;
        }
        view.scroll_to_bottom = true;
        set_text(&mut texts, view.status, "Connecting...");

        let agent_entity = request.agent;
        let endpoint = app_config.endpoint.clone();
        let token = app_config.bearer_token().map(str::to_owned);
        let session_id = app_config.session_id.clone();
        let (cancel_tx, cancel_rx) = watch::channel(false);
        tokio_runtime.spawn_background_task(move |ctx| async move {
            stream_transport(
                ctx,
                cancel_rx,
                agent_entity,
                turn_id,
                endpoint,
                token,
                session_id,
                body,
            )
            .await;
        });

        runtime.active.insert(
            request.agent,
            ActiveTurn {
                turn_id,
                kind: kind.clone(),
                decoder: Decoder::new(provider, kind.usage_provider()),
                draft: Vec::new(),
                stop_reason: None,
                cancel: cancel_tx,
            },
        );
    }
}

pub(crate) fn handle_turn_finished(
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut finished: MessageReader<AgentTurnFinished>,
    mut texts: Query<&mut Text>,
    mut collapsibles: Query<&mut CollapsibleText>,
    mut nodes: Query<&mut Node>,
    mut run_tools: MessageWriter<RunToolCalls>,
    mut commands: Commands,
) {
    for message in finished.read() {
        let Ok((mut agent, mut view)) = agents.get_mut(message.agent) else {
            continue;
        };
        if agent.current_turn != Some(message.turn_id) {
            continue;
        }

        if let Some(usage) = &message.usage {
            record_usage(&mut agent, usage);
            set_text(&mut texts, view.stats, stats_text(Some(usage)));
        }

        let answer = if !agent.streaming_text.is_empty() {
            agent.streaming_text.clone()
        } else if let Some(text) = response_display_text(&message.entries) {
            text
        } else if !agent.streaming_thinking.is_empty() {
            "".to_owned()
        } else {
            "(empty response)".to_owned()
        };
        let thinking = response_thinking_text(&message.entries).or_else(|| {
            (!agent.streaming_thinking.is_empty()).then(|| agent.streaming_thinking.clone())
        });

        if let Some(assistant) = view.assistant {
            set_text(&mut texts, assistant.answer_body, answer);
            if let Some(thinking) = thinking {
                set_display(&mut nodes, assistant.thinking_container, Display::Flex);
                set_collapsible_text(
                    &mut texts,
                    &mut collapsibles,
                    assistant.thinking_body,
                    thinking,
                );
            } else {
                set_display(&mut nodes, assistant.thinking_container, Display::None);
            }
        }

        let tool_call = match message.entries.last() {
            Some(Entry::ResponseToolCall { tool_call, .. }) => Some(tool_call.clone()),
            _ => None,
        };

        if message.entries.is_empty() {
            agent.retry_message = agent.rollback_pending_turn();
            agent.pending = false;
            agent.current_turn = None;
            if let Some(assistant) = view.assistant.take() {
                commands.entity(assistant.card).despawn();
            }
            set_text(&mut texts, view.status, "Empty response");
        } else {
            agent.extend_history("assistant_response", message.entries.clone());
            agent.streaming_text.clear();
            agent.streaming_thinking.clear();

            if let Some(tool_call) = tool_call {
                agent.pending = true;
                set_text(
                    &mut texts,
                    view.status,
                    format!(
                        "Running tools({})...",
                        tool_call
                            .iter()
                            .map(|x| x.tool_name.clone())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                );
                run_tools.write(RunToolCalls {
                    agent: message.agent,
                    turn_id: message.turn_id,
                    tool_call,
                });
            } else {
                agent.pending = false;
                agent.current_turn = None;
                agent.retry_message = None;
                let mut status = String::from("Ready");
                if let Some(reason) = message
                    .stop_reason
                    .as_deref()
                    .filter(|reason| *reason != "end_turn" && *reason != "stop")
                {
                    status.push_str(" | ");
                    status.push_str(reason);
                }
                set_text(&mut texts, view.status, status);
            }
        }

        view.scroll_to_bottom = true;
    }
}
