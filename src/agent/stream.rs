//! Transport stream decoding and UI event fan-out.

use super::*;

pub(crate) fn decode_stream_chunks(
    mut runtime: ResMut<AgentRuntime>,
    mut chunks: MessageReader<StreamChunk>,
    mut ended: MessageReader<StreamEnded>,
    mut transport_failed: MessageReader<StreamFailed>,
    mut stream_events: MessageWriter<AgentStreamEvent>,
    mut finished: MessageWriter<AgentTurnFinished>,
    mut failed: MessageWriter<AgentFailed>,
) {
    let mut decode_failure = None;

    for chunk in chunks.read() {
        let Some(active) = runtime.active.get_mut(&chunk.agent) else {
            continue;
        };
        if active.turn_id != chunk.turn_id {
            continue;
        }

        match active
            .decoder
            .push_bytes(RawStreamBytes::new(chunk.bytes.clone()), &mut active.draft)
        {
            Ok(events) => write_stream_events(&mut stream_events, chunk.agent, active, events),
            Err(err) => {
                decode_failure = Some((chunk.agent, active.turn_id, err.to_string()));
                break;
            }
        }
    }

    if let Some((agent, turn_id, message)) = decode_failure {
        runtime.active.remove(&agent);
        failed.write(AgentFailed {
            agent,
            turn_id,
            message,
        });
        return;
    }

    for end in ended.read() {
        if !runtime
            .active
            .get(&end.agent)
            .is_some_and(|active| active.turn_id == end.turn_id)
        {
            continue;
        }

        let mut active = runtime
            .active
            .remove(&end.agent)
            .expect("active turn checked above");
        match active.decoder.finish(&mut active.draft) {
            Ok(events) => write_stream_events(&mut stream_events, end.agent, &mut active, events),
            Err(err) => {
                failed.write(AgentFailed {
                    agent: end.agent,
                    turn_id: active.turn_id,
                    message: err.to_string(),
                });
                continue;
            }
        }

        let usage = active.decoder.usage().cloned();
        match finish_draft_for(
            Entry::Draft {
                blocks: active.draft,
            },
            &active.kind,
        ) {
            Ok(entries) => {
                finished.write(AgentTurnFinished {
                    agent: end.agent,
                    turn_id: active.turn_id,
                    entries,
                    usage,
                    stop_reason: active.stop_reason,
                });
            }
            Err(err) => {
                failed.write(AgentFailed {
                    agent: end.agent,
                    turn_id: active.turn_id,
                    message: err.to_string(),
                });
            }
        }
    }

    for fail in transport_failed.read() {
        if !runtime
            .active
            .get(&fail.agent)
            .is_some_and(|active| active.turn_id == fail.turn_id)
        {
            continue;
        }
        runtime.active.remove(&fail.agent);
        failed.write(AgentFailed {
            agent: fail.agent,
            turn_id: fail.turn_id,
            message: fail.message.clone(),
        });
    }
}

fn write_stream_events(
    stream_events: &mut MessageWriter<AgentStreamEvent>,
    agent: Entity,
    active: &mut ActiveTurn,
    events: Vec<StreamEvent>,
) {
    for event in events {
        if let StreamEvent::Completed { stop_reason } = &event {
            active.stop_reason = stop_reason.clone();
        }
        stream_events.write(AgentStreamEvent {
            agent,
            turn_id: active.turn_id,
            event,
        });
    }
}

pub(crate) fn handle_stream_events(
    mut agents: Query<(&mut Agent, &mut AgentView)>,
    mut stream_events: MessageReader<AgentStreamEvent>,
    mut texts: Query<&mut Text>,
    mut collapsibles: Query<&mut CollapsibleText>,
    mut nodes: Query<&mut Node>,
    collapse: Res<TextCollapseConfig>,
    mut commands: Commands,
) {
    for message in stream_events.read() {
        let Ok((mut agent, mut view)) = agents.get_mut(message.agent) else {
            continue;
        };
        if agent.current_turn != Some(message.turn_id) {
            continue;
        }

        match &message.event {
            StreamEvent::TextDelta(text) => {
                agent.streaming_text.push_str(text);
                ensure_live_assistant(&mut commands, &mut view, &agent, &collapse);
                if let Some(assistant) = view.assistant {
                    set_text(
                        &mut texts,
                        assistant.answer_body,
                        agent.streaming_text.clone(),
                    );
                }
                view.scroll_to_bottom = true;
            }
            StreamEvent::ThinkingDelta(text) => {
                agent.streaming_thinking.push_str(text);
                ensure_live_assistant(&mut commands, &mut view, &agent, &collapse);
                if let Some(assistant) = view.assistant {
                    set_display(&mut nodes, assistant.thinking_container, Display::Flex);
                    set_collapsible_text(
                        &mut texts,
                        &mut collapsibles,
                        assistant.thinking_body,
                        agent.streaming_thinking.clone(),
                    );
                }
                view.scroll_to_bottom = true;
                set_text(
                    &mut texts,
                    view.status,
                    format!(
                        "Thinking... ({} chars)",
                        agent.streaming_thinking.chars().count()
                    ),
                );
            }
            StreamEvent::ToolCallDelta {
                name: Some(name), ..
            } => {
                set_text(
                    &mut texts,
                    view.status,
                    format!("Preparing tool call `{name}`..."),
                );
            }
            StreamEvent::Completed { .. } => {
                set_text(&mut texts, view.status, "Finishing...");
            }
            StreamEvent::Usage(usage) => {
                set_text(&mut texts, view.stats, stats_text(Some(usage)));
            }
            _ => {}
        }
    }
}
