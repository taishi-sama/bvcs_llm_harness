//! Transcript card rendering, live updates, and scrolling.

use super::*;

/// Header and body entities for a collapsible text block.
pub(super) struct CollapsibleParts {
    pub(super) header: Entity,
    pub(super) body: Entity,
}

/// Spawn a text block with a clickable header. The body starts collapsed when
/// its contents exceed the configured limit.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_collapsible_text(
    commands: &mut Commands,
    label: &str,
    label_font_size: f32,
    label_color: Color,
    text: String,
    font_size: f32,
    text_color: Color,
    mode: CollapseMode,
    collapse: &TextCollapseConfig,
) -> CollapsibleParts {
    let mut hint = Entity::PLACEHOLDER;
    let header = commands
        .spawn((
            Node {
                width: percent(100.),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                column_gap: px(8.),
                ..default()
            },
            CollapsibleToggle {
                body: Entity::PLACEHOLDER,
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(label.to_owned()),
                TextFont {
                    font_size: FontSize::Px(label_font_size),
                    ..default()
                },
                TextColor(label_color),
            ));
            hint = parent
                .spawn((
                    Text::new(""),
                    TextFont {
                        font_size: FontSize::Px(12.),
                        ..default()
                    },
                    TextColor(Color::srgb(0.55, 0.75, 1.0)),
                    TextLayout::no_wrap(),
                ))
                .id();
        })
        .id();

    let state = CollapsibleText::new(text, collapse, mode, hint);
    let hint_text = state.hint_text();
    let visible = state.visible_text();
    let body = commands
        .spawn((
            Text::new(visible),
            TextFont {
                font_size: FontSize::Px(font_size),
                ..default()
            },
            TextColor(text_color),
            TextLayout {
                linebreak: LineBreak::WordOrCharacter,
                ..default()
            },
            Node {
                width: percent(100.),
                ..default()
            },
            state,
        ))
        .id();

    commands.entity(hint).insert(Text::new(hint_text));
    commands
        .entity(header)
        .insert(CollapsibleToggle { body })
        .observe(toggle_collapsible);

    CollapsibleParts { header, body }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_message_card(
    commands: &mut Commands,
    log: Entity,
    role: &str,
    role_color: Color,
    text: String,
    background: Color,
    collapsible: bool,
    collapse: &TextCollapseConfig,
) -> Entity {
    let card = commands
        .spawn((
            Node {
                width: percent(100.),
                flex_direction: FlexDirection::Column,
                row_gap: px(4.),
                padding: px(12.).all(),
                border_radius: BorderRadius::all(px(10.)),
                ..default()
            },
            BackgroundColor(background),
            TranscriptCard,
        ))
        .id();

    let body = if collapsible {
        let parts = spawn_collapsible_text(
            commands,
            role,
            13.,
            role_color,
            text,
            18.,
            Color::srgb(0.90, 0.92, 0.96),
            CollapseMode::Head,
            collapse,
        );
        commands
            .entity(card)
            .add_children(&[parts.header, parts.body]);
        parts.body
    } else {
        let mut body = Entity::PLACEHOLDER;
        commands.entity(card).with_children(|parent| {
            parent.spawn((
                Text::new(role),
                TextFont {
                    font_size: FontSize::Px(13.),
                    ..default()
                },
                TextColor(role_color),
            ));
            body = parent
                .spawn((
                    Text::new(text),
                    TextFont {
                        font_size: FontSize::Px(18.),
                        ..default()
                    },
                    TextColor(Color::srgb(0.90, 0.92, 0.96)),
                    TextLayout {
                        linebreak: LineBreak::WordOrCharacter,
                        ..default()
                    },
                    Node {
                        width: percent(100.),
                        ..default()
                    },
                    MessageBody,
                ))
                .id();
        });
        body
    };

    commands.entity(log).add_children(&[card]);
    body
}

pub(super) fn spawn_assistant_card(
    commands: &mut Commands,
    log: Entity,
    text: &str,
    thinking: Option<&str>,
    collapse: &TextCollapseConfig,
) -> AssistantView {
    let card = commands
        .spawn((
            Node {
                width: percent(100.),
                flex_direction: FlexDirection::Column,
                row_gap: px(8.),
                padding: px(12.).all(),
                border_radius: BorderRadius::all(px(10.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.10, 0.115, 0.15)),
            TranscriptCard,
        ))
        .id();

    let role = commands
        .spawn((
            Text::new("Assistant"),
            TextFont {
                font_size: FontSize::Px(13.),
                ..default()
            },
            TextColor(Color::srgb(0.55, 0.85, 0.70)),
        ))
        .id();

    let thinking_container = commands
        .spawn(Node {
            width: percent(100.),
            display: if thinking.is_some() {
                Display::Flex
            } else {
                Display::None
            },
            flex_direction: FlexDirection::Column,
            row_gap: px(3.),
            padding: px(9.).all(),
            border_radius: BorderRadius::all(px(7.)),
            ..default()
        })
        .insert(BackgroundColor(Color::srgb(0.085, 0.095, 0.125)))
        .id();

    let thinking_parts = spawn_collapsible_text(
        commands,
        "Thinking",
        12.,
        Color::srgb(0.62, 0.66, 0.76),
        thinking.map(str::to_owned).unwrap_or_default(),
        15.,
        Color::srgb(0.70, 0.74, 0.82),
        CollapseMode::HeadAndTail,
        collapse,
    );

    let answer = if !text.is_empty() {
        text.to_owned()
    } else if thinking.is_some() {
        "".to_owned()
    } else {
        "(empty response)".to_owned()
    };

    let answer_body = commands
        .spawn((
            Text::new(answer),
            TextFont {
                font_size: FontSize::Px(18.),
                ..default()
            },
            TextColor(Color::srgb(0.90, 0.92, 0.96)),
            TextLayout {
                linebreak: LineBreak::WordOrCharacter,
                ..default()
            },
            Node {
                width: percent(100.),
                ..default()
            },
            MessageBody,
        ))
        .id();

    commands
        .entity(thinking_container)
        .add_children(&[thinking_parts.header, thinking_parts.body]);
    commands
        .entity(card)
        .add_children(&[role, thinking_container, answer_body]);
    commands.entity(log).add_children(&[card]);

    AssistantView {
        card,
        thinking_container,
        thinking_body: thinking_parts.body,
        answer_body,
    }
}

pub(super) fn tool_result_summary(
    results: &[ToolCallResult],
    calls: &HashMap<String, PresentedCall>,
) -> String {
    results
        .iter()
        .map(|result| {
            let call = calls.get(&result.tool_use_id);
            let name = call
                .map(|call| call.name.as_str())
                .unwrap_or(&result.tool_use_id);
            let body = match call.map_or(ToolPresentation::Plain, |call| call.kind) {
                ToolPresentation::Plain => result.result.as_str(),
            };
            format!(
                "{name} [{}]\n{body}",
                if result.is_error { "error" } else { "ok" },
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub(super) fn tool_display_map(history: &[Entry]) -> HashMap<String, PresentedCall> {
    let mut calls = HashMap::new();
    for entry in history {
        if let Entry::ResponseToolCall { tool_call, .. } = entry {
            for call in tool_call {
                calls.insert(call.tool_use_id.clone(), present_call(call));
            }
        }
    }
    calls
}

fn render_entry(
    commands: &mut Commands,
    log: Entity,
    entry: &Entry,
    calls: &HashMap<String, PresentedCall>,
    collapse: &TextCollapseConfig,
) {
    match entry {
        Entry::SystemPrompt { content } | Entry::SystemMessage { content, .. } => {
            spawn_message_card(
                commands,
                log,
                "System",
                Color::srgb(0.55, 0.60, 0.70),
                content.clone(),
                Color::srgb(0.09, 0.10, 0.13),
                false,
                collapse,
            );
        }
        Entry::UserMessage { content } => {
            if let Some(text) = &content.text {
                spawn_message_card(
                    commands,
                    log,
                    "You",
                    Color::srgb(0.55, 0.75, 1.0),
                    text.clone(),
                    Color::srgb(0.12, 0.17, 0.26),
                    false,
                    collapse,
                );
            }
        }
        Entry::Response { .. } | Entry::ResponseToolCall { .. } => {
            let text = entry.response_joined_text().unwrap_or_default();
            let thinking = response_thinking_text(std::slice::from_ref(entry));
            if !text.is_empty() || thinking.is_some() {
                spawn_assistant_card(commands, log, &text, thinking.as_deref(), collapse);
            }

            if let Entry::ResponseToolCall { tool_call, .. } = entry
                && !tool_call.is_empty()
            {
                let names = tool_call
                    .iter()
                    .map(|call| format!("`{}`", present_call(call).name))
                    .collect::<Vec<_>>()
                    .join(", ");
                spawn_message_card(
                    commands,
                    log,
                    "Tool calls",
                    Color::srgb(0.95, 0.75, 0.40),
                    format!("Calling {names}"),
                    Color::srgb(0.13, 0.115, 0.09),
                    true,
                    collapse,
                );
            }
        }
        Entry::ToolCallResult { tool_call } => {
            spawn_message_card(
                commands,
                log,
                "Tool calls",
                Color::srgb(0.95, 0.75, 0.40),
                tool_result_summary(tool_call, calls),
                Color::srgb(0.13, 0.115, 0.09),
                true,
                collapse,
            );
        }
        Entry::Draft { .. } => {}
    }
}

/// Rebuild the transcript from the complete conversation history plus the live
/// streaming card, if any.
pub(crate) fn render_transcript(
    commands: &mut Commands,
    view: &mut AgentView,
    agent: &Agent,
    collapse: &TextCollapseConfig,
) {
    commands.entity(view.log).despawn_children();
    view.assistant = None;

    let history = agent.history();
    let calls = tool_display_map(history);
    for entry in history {
        render_entry(commands, view.log, entry, &calls, collapse);
    }

    if live_pending(agent) {
        view.assistant = Some(spawn_live_assistant(commands, view.log, agent, collapse));
    }

    view.transcript_visible = true;
    view.rendered_entries = history.len();
    view.scroll_to_bottom = true;
}

/// Start (or restart) the live assistant card for a new turn without rebuilding
/// the already-rendered history.
pub(crate) fn begin_live_assistant(
    commands: &mut Commands,
    view: &mut AgentView,
    agent: &Agent,
    collapse: &TextCollapseConfig,
) {
    if view.transcript_visible {
        view.assistant = Some(spawn_live_assistant(commands, view.log, agent, collapse));
    } else {
        view.assistant = None;
    }
    view.scroll_to_bottom = true;
}

fn spawn_live_assistant(
    commands: &mut Commands,
    log: Entity,
    agent: &Agent,
    collapse: &TextCollapseConfig,
) -> AssistantView {
    let thinking = (!agent.streaming_thinking().is_empty()).then_some(agent.streaming_thinking());
    let text = agent.streaming_text();
    let answer = if text.is_empty() && thinking.is_none() {
        "..."
    } else {
        text
    };
    spawn_assistant_card(commands, log, answer, thinking, collapse)
}

/// Whether a live streaming/placeholder card exists for the current turn.
fn live_pending(agent: &Agent) -> bool {
    agent.has_active_turn()
        || !agent.streaming_text().is_empty()
        || !agent.streaming_thinking().is_empty()
}

/// Render only the newest history entry when exactly one was appended. This is
/// the "draft popped" path: the live assistant card is replaced by the
/// finalized block without rebuilding the whole transcript.
pub(crate) fn refresh_transcripts(
    mut commands: Commands,
    mut views: Query<(&mut AgentView, &Agent), Changed<Agent>>,
    collapse: Res<TextCollapseConfig>,
) {
    for (mut view, agent) in &mut views {
        if !view.transcript_visible {
            continue;
        }

        let history_len = agent.history().len();
        if history_len < view.rendered_entries {
            // Rollback or reset: the transcript is no longer trustworthy.
            render_transcript(&mut commands, &mut view, agent, &collapse);
            continue;
        }
        if history_len > view.rendered_entries + 1 {
            render_transcript(&mut commands, &mut view, agent, &collapse);
            continue;
        }
        if history_len == view.rendered_entries + 1 {
            if let Some(assistant) = view.assistant.take() {
                commands.entity(assistant.card).despawn();
            }
            if let Some(entry) = agent.history().last() {
                let calls = tool_display_map(agent.history());
                render_entry(&mut commands, view.log, entry, &calls, &collapse);
            }
            view.rendered_entries = history_len;
            view.scroll_to_bottom = true;
        }
    }
}

/// Spawn the live card if streaming began while the transcript had no card
/// (for example after the tab was re-activated).
pub(crate) fn ensure_live_assistant(
    commands: &mut Commands,
    view: &mut AgentView,
    agent: &Agent,
    collapse: &TextCollapseConfig,
) {
    if view.transcript_visible && view.assistant.is_none() && live_pending(agent) {
        view.assistant = Some(spawn_live_assistant(commands, view.log, agent, collapse));
    }
}

pub(crate) fn response_display_text(entries: &[Entry]) -> Option<String> {
    let text = entries
        .iter()
        .filter_map(Entry::response_joined_text)
        .collect::<Vec<_>>()
        .join("\n\n");
    if !text.is_empty() {
        return Some(text);
    }

    for entry in entries {
        if let Entry::ResponseToolCall { tool_call, .. } = entry
            && !tool_call.is_empty()
        {
            return Some(format!(
                "Calling {}...",
                tool_call
                    .iter()
                    .map(|call| format!("`{}`", present_call(call).name))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }

    None
}

pub(crate) fn response_thinking_text(entries: &[Entry]) -> Option<String> {
    let thinking = entries
        .iter()
        .flat_map(|entry| entry.response_thinking_blocks())
        .cloned()
        .collect::<Vec<Thinking>>();
    thinking_joined(&thinking)
}

pub(crate) fn set_text(texts: &mut Query<&mut Text>, entity: Entity, value: impl Into<String>) {
    if let Ok(mut text) = texts.get_mut(entity) {
        text.0 = value.into();
    }
}

/// Update a collapsible body, re-applying its current collapsed/expanded state
/// and refreshing the header hint. Falls back to a plain text update for bodies
/// that are not collapsible.
pub(crate) fn set_collapsible_text(
    texts: &mut Query<&mut Text>,
    collapsibles: &mut Query<&mut CollapsibleText>,
    entity: Entity,
    value: impl Into<String>,
) {
    let value = value.into();
    let Ok(mut collapsible) = collapsibles.get_mut(entity) else {
        if let Ok(mut text) = texts.get_mut(entity) {
            text.0 = value;
        }
        return;
    };

    collapsible.full = value;
    let visible = collapsible.visible_text();
    let hint = collapsible.hint_text();
    let hint_entity = collapsible.hint;
    if let Ok(mut text) = texts.get_mut(entity) {
        text.0 = visible;
    }
    if let Ok(mut text) = texts.get_mut(hint_entity) {
        text.0 = hint;
    }
}

pub(crate) fn set_display(nodes: &mut Query<&mut Node>, entity: Entity, display: Display) {
    if let Ok(mut node) = nodes.get_mut(entity) {
        node.display = display;
    }
}

/// Marks the "Currently running toolcalls: N" readout.
#[derive(Component)]
pub(super) struct ToolActivityText;

/// Renders the in-flight tool-call count.
///
/// Inline tools finish within the frame, so this currently tracks interpreter
/// and detached calls; the per-root budget work will feed the same readout.
pub(crate) fn tool_activity_text(running: usize) -> String {
    format!("Currently running toolcalls: {running}")
}

/// Keep every visible agent's tool-activity readout in sync with [`ToolRounds`].
pub(crate) fn update_tool_activity(
    rounds: Res<ToolRounds>,
    views: Query<(Entity, &AgentView)>,
    mut texts: Query<&mut Text>,
) {
    for (agent, view) in &views {
        if view.tool_activity == Entity::PLACEHOLDER {
            continue;
        }
        let wanted = tool_activity_text(rounds.running_for(agent));
        let Ok(mut text) = texts.get_mut(view.tool_activity) else {
            continue;
        };
        if text.0 != wanted {
            text.0 = wanted;
        }
    }
}

/// Renders the fixed-width usage readout. Fields are padded so the text does
/// not shift as token counts grow (Bevy's default font is monospaced).
pub(crate) fn stats_text(usage: Option<&Usage>) -> String {
    fn field(value: Option<u64>) -> String {
        value.map_or_else(|| "-".to_owned(), |value| value.to_string())
    }

    let input = field(usage.and_then(|usage| usage.input_tokens));
    let output = field(usage.and_then(|usage| usage.output_tokens));
    let cached = field(usage.and_then(|usage| usage.cached_input_tokens));
    format!("in {input:>6}   out {output:>6}   cached {cached:>6}")
}

pub(crate) fn auto_scroll(
    mut views: Query<&mut AgentView>,
    mut logs: Query<(&mut ScrollPosition, &ComputedNode)>,
) {
    for mut view in &mut views {
        if !view.follow_tail || !view.scroll_to_bottom {
            continue;
        }
        let Ok((mut scroll_position, computed)) = logs.get_mut(view.log) else {
            continue;
        };

        let max_offset =
            (computed.content_size() - computed.size()) * computed.inverse_scale_factor();
        scroll_position.y = max_offset.y.max(0.);
        view.scroll_to_bottom = false;
    }
}
