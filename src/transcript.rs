//! Chat transcript display: cards, collapsibles, bounded window, and scrolling.
//!
//! This module owns everything the user sees inside an agent's chat log. Agent
//! state and the LLM loop stay in [`crate::agent`]; the display layer only reads
//! history through accessors and keeps at most [`MAX_TRANSCRIPT_CARDS`] cards
//! alive, so offscreen conversations cannot grow VRAM usage.

use std::{collections::HashMap, env};

use bevy::{
    input_focus::tab_navigation::TabIndex,
    picking::events::{PointerClick, PointerScroll},
    prelude::*,
    text::{EditableText, TextCursorStyle},
    ui_widgets::{Activate, Button, TextInput},
};
use llm_api::{usage::Usage, wire_common::thinking_joined};
use llm_history_model::history::{Entry, Thinking, ToolCallResult};

use crate::{
    agent::Agent,
    messages::{CancelAgentRequest, RemoveAgentRequest, RetryAgentTurn},
};

/// Maximum number of message cards kept alive per transcript. Rendering only
/// this window keeps offscreen text from consuming VRAM.
pub(crate) const MAX_TRANSCRIPT_CARDS: usize = 20;

/// Where the visible transcript window starts.
///
/// The log never keeps more than [`MAX_TRANSCRIPT_CARDS`] cards. Scrolling moves
/// the window one entry at a time instead of moving through a long smooth list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TranscriptAnchor {
    /// Show the newest entries at the bottom and keep the window pinned as the
    /// dialogue grows.
    FollowTail,
    /// Show this history entry as the first card at the top of the viewport.
    Entry(usize),
}

/// Truncation limits for long reasoning and tool-call bodies. A body collapses
/// when it exceeds either limit.
#[derive(Resource, Clone, Debug)]
pub(crate) struct TextCollapseConfig {
    pub(crate) max_chars: usize,
    pub(crate) max_lines: usize,
}

impl Default for TextCollapseConfig {
    fn default() -> Self {
        const DEFAULT_CHARS: usize = 1_000;
        const DEFAULT_LINES: usize = 10;
        Self {
            max_chars: env_limit("COLLAPSE_MAX_CHARS").unwrap_or(DEFAULT_CHARS),
            max_lines: env_limit("COLLAPSE_MAX_LINES").unwrap_or(DEFAULT_LINES),
        }
    }
}

fn env_limit(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

/// Entity handles for one agent's UI.
#[derive(Component)]
pub(crate) struct AgentView {
    pub(crate) panel: Entity,
    pub(crate) log: Entity,
    pub(crate) input: Entity,
    pub(crate) status: Entity,
    /// Fixed-width usage readout (input/output/cached tokens).
    pub(crate) stats: Entity,
    pub(crate) tab_label: Entity,
    pub(crate) assistant: Option<AssistantView>,
    /// Whether the transcript cards for this agent are currently spawned.
    pub(crate) transcript_visible: bool,
    /// Number of history entries the current transcript reflects.
    pub(crate) rendered_entries: usize,
    /// Set when the scroll container should jump to the bottom after layout.
    pub(crate) scroll_to_bottom: bool,
    /// Set when an entry-anchored window should start at the top after layout.
    pub(crate) scroll_to_top: bool,
    /// Start of the visible card window; [`TranscriptAnchor::FollowTail`] keeps
    /// it pinned to the newest dialogue.
    pub(crate) anchor: TranscriptAnchor,
}

impl AgentView {
    /// Pin the window to the newest entry again.
    pub(crate) fn follow_tail(&mut self) {
        self.anchor = TranscriptAnchor::FollowTail;
        self.scroll_to_bottom = true;
        self.scroll_to_top = false;
    }

    /// Whether the transcript currently follows newly appended dialogue.
    pub(crate) fn is_following_tail(&self) -> bool {
        matches!(self.anchor, TranscriptAnchor::FollowTail)
    }

    /// Keep the viewport pinned to the newest card only while following tail.
    /// Entry-anchored windows must stay at their selected top card.
    pub(crate) fn scroll_tail_if_following(&mut self) {
        if self.is_following_tail() {
            self.scroll_to_bottom = true;
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct AssistantView {
    pub(crate) card: Entity,
    pub(crate) thinking_container: Entity,
    pub(crate) thinking_body: Entity,
    pub(crate) answer_body: Entity,
}

/// The visible agent tab for [`Agent`].
#[derive(Component)]
pub(crate) struct AgentTab {
    pub(crate) agent: Entity,
}

/// Retry button belonging to one agent.
#[derive(Component)]
pub(crate) struct RetryAgentButton {
    pub(crate) agent: Entity,
}

/// Destructive button that removes the agent it belongs to.
#[derive(Component)]
pub(crate) struct KillAgentButton {
    pub(crate) agent: Entity,
}

#[derive(Component)]
pub(crate) struct MessageBody;

#[derive(Component)]
pub(crate) struct StatusText;

/// The scroll container that owns one agent's transcript cards.
#[derive(Component)]
pub(crate) struct TranscriptLog;

/// A removable transcript card root.
#[derive(Component)]
pub(crate) struct TranscriptCard;

/// How a collapsed body chooses the text it keeps visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CollapseMode {
    /// Show the leading lines (used for tool output).
    Head,
    /// Show the leading and trailing lines (used for live reasoning, so the
    /// newest reasoning stays visible while it streams in).
    HeadAndTail,
}

/// A text body that collapses behind a "Show more" header when it exceeds the
/// configured character or line limit.
#[derive(Component)]
pub(crate) struct CollapsibleText {
    full: String,
    collapsed: bool,
    max_chars: usize,
    max_lines: usize,
    mode: CollapseMode,
    /// Text entity in the header that shows the expand/collapse affordance.
    hint: Entity,
}

impl CollapsibleText {
    fn new(full: String, config: &TextCollapseConfig, mode: CollapseMode, hint: Entity) -> Self {
        Self {
            full,
            collapsed: true,
            max_chars: config.max_chars,
            max_lines: config.max_lines,
            mode,
            hint,
        }
    }

    fn exceeds_limit(&self) -> bool {
        self.full.chars().count() > self.max_chars || self.full.lines().count() > self.max_lines
    }

    /// Text that should currently be rendered. Long text starts truncated and
    /// is only shown in full after the user expands it.
    fn visible_text(&self) -> String {
        if self.collapsed && self.exceeds_limit() {
            match self.mode {
                CollapseMode::Head => collapsed_text(&self.full, self.max_chars, self.max_lines),
                CollapseMode::HeadAndTail => {
                    collapsed_head_tail_text(&self.full, self.max_chars, self.max_lines)
                }
            }
        } else {
            self.full.clone()
        }
    }

    fn hint_text(&self) -> String {
        if !self.exceeds_limit() {
            return String::new();
        }
        if self.collapsed {
            format!(
                "Show more ({} chars, {} lines)",
                self.full.chars().count(),
                self.full.lines().count()
            )
        } else {
            "Show less".to_owned()
        }
    }
}

/// Clickable header that toggles the [`CollapsibleText`] stored on `body`.
#[derive(Component)]
pub(crate) struct CollapsibleToggle {
    body: Entity,
}

/// Keep the first `max_lines` lines and then at most `max_chars` characters.
fn collapsed_text(full: &str, max_chars: usize, max_lines: usize) -> String {
    let line_limited = full.lines().take(max_lines).collect::<Vec<_>>().join("\n");
    let truncated = line_limited.chars().take(max_chars).collect::<String>();
    let hidden = full
        .chars()
        .count()
        .saturating_sub(truncated.chars().count());
    if hidden == 0 {
        truncated
    } else {
        format!("{truncated}\n... ({hidden} more characters)")
    }
}

/// Show the first half and the last half of the line budget, with an ellipsis
/// between them. Each side also gets half of the character budget, so the
/// collapsed block stays compact even when lines are very long.
fn collapsed_head_tail_text(full: &str, max_chars: usize, max_lines: usize) -> String {
    const SEPARATOR: &str = "\n...\n";
    let lines = full.lines().collect::<Vec<_>>();

    if lines.len() <= max_lines {
        // Only the character limit was exceeded. Split the single block into
        // leading and trailing character windows instead.
        return head_tail_chars(full, max_chars, SEPARATOR);
    }

    let head_lines = (max_lines / 2).max(1);
    let tail_lines = max_lines.saturating_sub(head_lines).max(1);
    let head_end = head_lines.min(lines.len());
    let tail_start = lines.len().saturating_sub(tail_lines).max(head_end);
    let head = lines[..head_end].join("\n");
    let tail = lines[tail_start..].join("\n");

    let budget = max_chars.saturating_sub(SEPARATOR.chars().count());
    let head_budget = budget / 2;
    let tail_budget = budget - head_budget;
    let head = head.chars().take(head_budget).collect::<String>();
    let tail = tail
        .chars()
        .rev()
        .take(tail_budget)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();

    format!("{head}{SEPARATOR}{tail}")
}

fn head_tail_chars(full: &str, max_chars: usize, separator: &str) -> String {
    let budget = max_chars.saturating_sub(separator.chars().count());
    let head_budget = budget / 2;
    let tail_budget = budget - head_budget;
    let head = full.chars().take(head_budget).collect::<String>();
    let tail = full
        .chars()
        .rev()
        .take(tail_budget)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!("{head}{separator}{tail}")
}

/// Toggles a collapsible body when its header is clicked. Pointer events bubble
/// from the header's text children, so `click.entity` is the header entity.
fn toggle_collapsible(
    click: On<PointerClick>,
    toggles: Query<&CollapsibleToggle>,
    mut bodies: Query<&mut CollapsibleText>,
    mut texts: Query<&mut Text>,
) {
    let Ok(toggle) = toggles.get(click.entity) else {
        return;
    };
    let Ok(mut body) = bodies.get_mut(toggle.body) else {
        return;
    };
    if !body.exceeds_limit() {
        return;
    }

    body.collapsed = !body.collapsed;
    let visible = body.visible_text();
    let hint = body.hint_text();
    let hint_entity = body.hint;
    if let Ok(mut text) = texts.get_mut(toggle.body) {
        text.0 = visible;
    }
    if let Ok(mut text) = texts.get_mut(hint_entity) {
        text.0 = hint;
    }
}

pub(crate) struct PanelEntities {
    pub(crate) panel: Entity,
    pub(crate) log: Entity,
    pub(crate) input: Entity,
    pub(crate) status: Entity,
    pub(crate) stats: Entity,
}

pub(crate) fn create_agent_panel(
    commands: &mut Commands,
    panels_root: Entity,
    agent: Entity,
) -> PanelEntities {
    let panel = commands
        .spawn((
            Node {
                width: percent(100.),
                height: percent(100.),
                min_height: px(0.),
                display: Display::None,
                flex_direction: FlexDirection::Column,
                row_gap: px(12.),
                ..default()
            },
            Visibility::Hidden,
        ))
        .id();

    let log = commands
        .spawn((
            Node {
                width: percent(100.),
                flex_basis: px(0.),
                flex_grow: 1.,
                min_height: px(0.),
                flex_direction: FlexDirection::Column,
                row_gap: px(10.),
                padding: px(4.).all(),
                overflow: Overflow::scroll_y(),
                ..default()
            },
            TranscriptLog,
        ))
        .id();

    // Discrete scrolling. The observer is attached to the scroll container so
    // wheel events that hit child message cards bubble up to it. Each notch
    // shifts the rendered window by one history entry instead of moving the
    // scroll offset smoothly through a long list.
    commands.entity(log).observe(
        move |on_scroll: On<PointerScroll>,
              mut views: Query<&mut AgentView>,
              agents: Query<&Agent>,
              children: Query<&Children>,
              nodes: Query<&Node>,
              computed: Query<&ComputedNode>,
              scroll_positions: Query<&ScrollPosition>,
              collapse: Res<TextCollapseConfig>,
              mut commands: Commands| {
            if on_scroll.y == 0. {
                return;
            }
            let Ok(mut view) = views.get_mut(agent) else {
                return;
            };
            let Ok(agent_ref) = agents.get(agent) else {
                return;
            };

            let visible_top = if view.is_following_tail() {
                visible_top_card(view.log, &children, &nodes, &computed, &scroll_positions)
            } else {
                0
            };
            let older = on_scroll.y > 0.;
            if scroll_transcript(&mut view, agent_ref, older, visible_top) {
                render_transcript_with_live(&mut commands, &mut view, agent_ref, &collapse, false);
            }
        },
    );

    let input = commands
        .spawn((
            Node {
                flex_grow: 1.,
                flex_basis: px(0.),
                min_width: px(0.),
                min_height: px(46.),
                padding: px(12.).all(),
                border: px(2.).all(),
                border_radius: BorderRadius::all(px(10.)),
                align_items: AlignItems::Center,
                ..default()
            },
            TextInput,
            EditableText {
                visible_lines: Some(1.),
                allow_newlines: false,
                ..default()
            },
            TextLayout::no_wrap(),
            TextFont {
                font_size: FontSize::Px(18.),
                ..default()
            },
            TextCursorStyle::default(),
            BackgroundColor(Color::srgb(0.10, 0.115, 0.15)),
            BorderColor::all(Color::srgb(0.24, 0.28, 0.36)),
            TabIndex(1),
        ))
        .id();

    let retry = commands
        .spawn((
            Button,
            RetryAgentButton { agent },
            Node {
                min_width: px(82.),
                height: px(46.),
                padding: UiRect::horizontal(px(14.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(2.).all(),
                border_radius: BorderRadius::all(px(10.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.11, 0.12, 0.14)),
            BorderColor::all(Color::srgb(0.22, 0.25, 0.30)),
            TabIndex(2),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Retry"),
                TextFont {
                    font_size: FontSize::Px(16.),
                    ..default()
                },
                TextColor(Color::srgb(0.82, 0.88, 0.96)),
            ));
        })
        .id();

    commands
        .entity(retry)
        .observe(move |_: On<Activate>, mut commands: Commands| {
            commands.write_message(RetryAgentTurn { agent });
        });

    let stop = commands
        .spawn((
            Button,
            Node {
                min_width: px(82.),
                height: px(46.),
                padding: UiRect::horizontal(px(14.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(2.).all(),
                border_radius: BorderRadius::all(px(10.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.22, 0.10, 0.11)),
            BorderColor::all(Color::srgb(0.55, 0.25, 0.27)),
            TabIndex(3),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Stop"),
                TextFont {
                    font_size: FontSize::Px(16.),
                    ..default()
                },
                TextColor(Color::srgb(0.98, 0.82, 0.82)),
            ));
        })
        .id();

    commands
        .entity(stop)
        .observe(move |_: On<Activate>, mut commands: Commands| {
            commands.write_message(CancelAgentRequest { agent });
        });

    let kill = commands
        .spawn((
            Button,
            KillAgentButton { agent },
            Node {
                min_width: px(82.),
                height: px(46.),
                padding: UiRect::horizontal(px(14.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(2.).all(),
                border_radius: BorderRadius::all(px(10.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.30, 0.07, 0.08)),
            BorderColor::all(Color::srgb(0.72, 0.20, 0.22)),
            TabIndex(5),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Kill"),
                TextFont {
                    font_size: FontSize::Px(16.),
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.78, 0.78)),
            ));
        })
        .id();

    commands.entity(kill).observe(
        |activate: On<Activate>, buttons: Query<&KillAgentButton>, mut commands: Commands| {
            if let Ok(button) = buttons.get(activate.entity) {
                commands.write_message(RemoveAgentRequest {
                    agent: button.agent,
                });
            }
        },
    );

    let scroll_to_bottom = commands
        .spawn((
            Button,
            Node {
                min_width: px(140.),
                height: px(28.),
                flex_shrink: 0.,
                padding: UiRect::horizontal(px(10.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(1.).all(),
                border_radius: BorderRadius::all(px(8.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.13, 0.16, 0.21)),
            BorderColor::all(Color::srgb(0.30, 0.36, 0.46)),
            TabIndex(4),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Scroll to bottom"),
                TextFont {
                    font_size: FontSize::Px(12.),
                    ..default()
                },
                TextColor(Color::srgb(0.82, 0.88, 0.96)),
            ));
        })
        .id();

    commands.entity(scroll_to_bottom).observe(
        move |_: On<Activate>,
              mut views: Query<&mut AgentView>,
              agents: Query<&Agent>,
              collapse: Res<TextCollapseConfig>,
              mut commands: Commands| {
            let Ok(mut view) = views.get_mut(agent) else {
                return;
            };
            let Ok(agent_ref) = agents.get(agent) else {
                return;
            };
            view.follow_tail();
            render_transcript(&mut commands, &mut view, agent_ref, &collapse);
        },
    );

    let input_row = commands
        .spawn(Node {
            width: percent(100.),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(8.),
            ..default()
        })
        .add_children(&[input, retry, stop, kill])
        .id();

    let status = commands
        .spawn((
            Text::new("Ready"),
            TextFont {
                font_size: FontSize::Px(13.),
                ..default()
            },
            TextColor(Color::srgb(0.55, 0.60, 0.70)),
            StatusText,
        ))
        .id();

    let mut stats = Entity::PLACEHOLDER;
    let stats_box = commands
        .spawn((
            Node {
                width: px(320.),
                height: px(28.),
                flex_shrink: 0.,
                padding: UiRect::horizontal(px(10.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(1.).all(),
                border_radius: BorderRadius::all(px(8.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.10, 0.12, 0.16)),
            BorderColor::all(Color::srgb(0.22, 0.26, 0.33)),
        ))
        .with_children(|parent| {
            stats = parent
                .spawn((
                    Text::new(stats_text(None)),
                    TextFont {
                        font_size: FontSize::Px(12.),
                        ..default()
                    },
                    TextColor(Color::srgb(0.62, 0.68, 0.78)),
                    TextLayout::no_wrap(),
                ))
                .id();
        })
        .id();

    let footer = commands
        .spawn(Node {
            width: percent(100.),
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: px(8.),
            ..default()
        })
        .add_children(&[status, stats_box, scroll_to_bottom])
        .id();

    commands
        .entity(panel)
        .add_children(&[log, input_row, footer]);
    commands.entity(panels_root).add_children(&[panel]);

    PanelEntities {
        panel,
        log,
        input,
        status,
        stats,
    }
}

pub(crate) fn create_agent_tab(
    commands: &mut Commands,
    tabs: Entity,
    agent: Entity,
    name: &str,
    tab_index: i32,
) -> (Entity, Entity) {
    let mut label = Entity::PLACEHOLDER;
    let tab = commands
        .spawn((
            Button,
            AgentTab { agent },
            Node {
                min_width: px(84.),
                height: px(34.),
                padding: UiRect::horizontal(px(14.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(1.).all(),
                border_radius: BorderRadius::all(px(8.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.12, 0.14, 0.18)),
            BorderColor::all(Color::srgb(0.24, 0.28, 0.36)),
            TabIndex(tab_index),
        ))
        .with_children(|parent| {
            label = parent
                .spawn((
                    Text::new(name.to_owned()),
                    TextFont {
                        font_size: FontSize::Px(14.),
                        ..default()
                    },
                    TextColor(Color::srgb(0.82, 0.87, 0.95)),
                ))
                .id();
        })
        .id();

    commands.entity(tabs).add_children(&[tab]);
    (tab, label)
}

/// Header and body entities for a collapsible text block.
struct CollapsibleParts {
    header: Entity,
    body: Entity,
}

/// Spawn a text block with a clickable header. The body starts collapsed when
/// its contents exceed the configured limit.
#[allow(clippy::too_many_arguments)]
fn spawn_collapsible_text(
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
pub(crate) fn spawn_message_card(
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

pub(crate) fn spawn_assistant_card(
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

fn tool_result_summary(results: &[ToolCallResult], tool_names: &HashMap<String, String>) -> String {
    results
        .iter()
        .map(|result| {
            let name = tool_names
                .get(&result.tool_use_id)
                .map(String::as_str)
                .unwrap_or(&result.tool_use_id);
            format!(
                "{name} [{}]\n{}",
                if result.is_error { "error" } else { "ok" },
                result.result
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn tool_name_map(history: &[Entry]) -> HashMap<String, String> {
    let mut names = HashMap::new();
    for entry in history {
        if let Entry::ResponseToolCall { tool_call, .. } = entry {
            for call in tool_call {
                names.insert(call.tool_use_id.clone(), call.tool_name.clone());
            }
        }
    }
    names
}

fn render_entry(
    commands: &mut Commands,
    log: Entity,
    entry: &Entry,
    tool_names: &HashMap<String, String>,
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
                    .map(|call| format!("`{}`", call.tool_name))
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
                tool_result_summary(tool_call, tool_names),
                Color::srgb(0.13, 0.115, 0.09),
                true,
                collapse,
            );
        }
        Entry::Draft { .. } => {}
    }
}

/// Rebuild the visible transcript window from the configured anchor.
pub(crate) fn render_transcript(
    commands: &mut Commands,
    view: &mut AgentView,
    agent: &Agent,
    collapse: &TextCollapseConfig,
) {
    render_transcript_with_live(commands, view, agent, collapse, false);
}

/// Start (or restart) the live assistant card for a new turn.
pub(crate) fn begin_live_assistant(
    commands: &mut Commands,
    view: &mut AgentView,
    agent: &Agent,
    collapse: &TextCollapseConfig,
) {
    render_transcript_with_live(commands, view, agent, collapse, true);
}

fn render_transcript_with_live(
    commands: &mut Commands,
    view: &mut AgentView,
    agent: &Agent,
    collapse: &TextCollapseConfig,
    force_live: bool,
) {
    commands.entity(view.log).despawn_children();
    view.assistant = None;

    let history = agent.history();
    let history_len = history.len();
    let following_tail = view.is_following_tail();
    let show_live = following_tail && live_pending(agent, force_live);
    // The live card shares the 20-card budget with history cards, so remove one
    // history card from the window while a response is streaming.
    let capacity = MAX_TRANSCRIPT_CARDS.saturating_sub(usize::from(show_live));
    let start = match view.anchor {
        TranscriptAnchor::FollowTail => history_len.saturating_sub(capacity),
        TranscriptAnchor::Entry(index) => {
            let clamped = index.min(history_len.saturating_sub(1));
            if clamped != index {
                view.anchor = TranscriptAnchor::Entry(clamped);
            }
            clamped
        }
    };
    let end = (start + capacity).min(history_len);
    let tool_names = tool_name_map(history);
    for entry in &history[start..end] {
        render_entry(commands, view.log, entry, &tool_names, collapse);
    }

    if show_live {
        let thinking =
            (!agent.streaming_thinking().is_empty()).then_some(agent.streaming_thinking());
        let text = agent.streaming_text();
        let answer = if text.is_empty() && thinking.is_none() {
            "..."
        } else {
            text
        };
        view.assistant = Some(spawn_assistant_card(
            commands, view.log, answer, thinking, collapse,
        ));
    }

    view.transcript_visible = true;
    view.rendered_entries = history_len;
    // Entry-anchored windows always start at the selected entry so scrolling
    // changes the top card by exactly one. FollowTail stays glued to the newest
    // card, even when fewer than 20 cards are rendered.
    view.scroll_to_bottom = following_tail;
    view.scroll_to_top = !following_tail;
}

/// Whether a live streaming/placeholder card exists for the current turn.
fn live_pending(agent: &Agent, force_live: bool) -> bool {
    force_live
        || agent.has_active_turn()
        || !agent.streaming_text().is_empty()
        || !agent.streaming_thinking().is_empty()
}

/// Whether the live card should be part of the visible window.
fn live_requested(view: &AgentView, agent: &Agent, force_live: bool) -> bool {
    view.is_following_tail() && live_pending(agent, force_live)
}

/// Rebuild the window when history changed. Scrolling itself re-renders in the
/// scroll observer, so this only handles appended or rolled-back entries.
pub(crate) fn refresh_transcripts(
    mut commands: Commands,
    mut views: Query<(&mut AgentView, &Agent)>,
    collapse: Res<TextCollapseConfig>,
) {
    for (mut view, agent) in &mut views {
        if !view.transcript_visible {
            continue;
        }
        if agent.history().len() != view.rendered_entries {
            render_transcript(&mut commands, &mut view, agent, &collapse);
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
    if view.transcript_visible && view.assistant.is_none() && live_requested(view, agent, false) {
        render_transcript_with_live(commands, view, agent, collapse, false);
    }
}

/// Move the transcript window by exactly one history entry.
fn scroll_transcript(view: &mut AgentView, agent: &Agent, older: bool, visible_top: usize) -> bool {
    let history_len = agent.history().len();
    let tail_capacity =
        MAX_TRANSCRIPT_CARDS.saturating_sub(usize::from(live_pending(agent, false)));
    let next = shifted_anchor(view.anchor, older, history_len, tail_capacity, visible_top);
    if next == view.anchor {
        return false;
    }
    view.anchor = next;
    true
}

/// Pure anchor transition used by both the wheel observer and tests.
fn shifted_anchor(
    anchor: TranscriptAnchor,
    older: bool,
    history_len: usize,
    tail_capacity: usize,
    visible_top: usize,
) -> TranscriptAnchor {
    if history_len == 0 {
        return TranscriptAnchor::Entry(0);
    }
    let last_index = history_len - 1;

    match (anchor, older) {
        (TranscriptAnchor::FollowTail, false) => TranscriptAnchor::FollowTail,
        (TranscriptAnchor::FollowTail, true) => {
            // `visible_top` is the card index inside the current tail window.
            // Step one entry above whatever the user can currently see, so the
            // first wheel notch also works when fewer than 20 cards exist.
            let tail_start = history_len.saturating_sub(tail_capacity);
            let top = (tail_start + visible_top).min(last_index);
            TranscriptAnchor::Entry(top.saturating_sub(1))
        }
        (TranscriptAnchor::Entry(index), true) => {
            let top = index.min(last_index);
            TranscriptAnchor::Entry(top.saturating_sub(1))
        }
        (TranscriptAnchor::Entry(index), false) => {
            let top = index.min(last_index);
            if top + 1 >= history_len {
                TranscriptAnchor::FollowTail
            } else {
                TranscriptAnchor::Entry(top + 1)
            }
        }
    }
}

/// Find the card currently at the top of the scroll viewport. The transcript
/// layout uses a fixed row gap, so card sizes are enough to map scroll offset
/// back to a card index without another layout pass.
fn visible_top_card(
    log: Entity,
    children: &Query<&Children>,
    nodes: &Query<&Node>,
    computed: &Query<&ComputedNode>,
    scroll_positions: &Query<&ScrollPosition>,
) -> usize {
    let Ok(cards) = children.get(log) else {
        return 0;
    };
    let Some(log_computed) = computed.get(log).ok() else {
        return 0;
    };
    let scroll_y = scroll_positions.get(log).map_or(0., |position| position.y);
    let inverse_scale = log_computed.inverse_scale_factor();
    let row_gap = nodes.get(log).ok().map_or(0., |node| match node.row_gap {
        Val::Px(gap) => gap,
        _ => 0.,
    });

    let mut top = 0;
    let mut offset = 0.;
    for (index, card) in cards.iter().enumerate() {
        if offset > scroll_y + 0.5 {
            break;
        }
        top = index;
        if let Ok(card) = computed.get(card) {
            offset += card.size().y * inverse_scale + row_gap;
        }
    }
    top
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
                    .map(|call| format!("`{}`", call.tool_name))
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
        if view.scroll_to_top {
            if let Ok((mut scroll_position, _)) = logs.get_mut(view.log) {
                scroll_position.y = 0.;
            }
            view.scroll_to_top = false;
            view.scroll_to_bottom = false;
            continue;
        }
        if !view.scroll_to_bottom {
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

#[cfg(test)]
mod transcript_tests {
    use super::*;
    use crate::config::Provider;
    use llm_history_model::history::MessageContent;

    fn agent_view(log: Entity) -> AgentView {
        AgentView {
            panel: Entity::PLACEHOLDER,
            log,
            input: Entity::PLACEHOLDER,
            status: Entity::PLACEHOLDER,
            stats: Entity::PLACEHOLDER,
            tab_label: Entity::PLACEHOLDER,
            assistant: None,
            transcript_visible: true,
            rendered_entries: 0,
            scroll_to_bottom: false,
            scroll_to_top: false,
            anchor: TranscriptAnchor::FollowTail,
        }
    }

    fn agent_with_messages(count: usize) -> Agent {
        let mut agent = Agent::new("Worker".to_owned(), Provider::Responses, "model".to_owned());
        for index in 0..count {
            agent.push_entry_for_test(Entry::UserMessage {
                content: MessageContent {
                    text: Some(format!("message {index}")),
                    image: None,
                },
            });
        }
        agent
    }

    fn card_body_text(world: &World, card: Entity) -> String {
        let children = world.get::<Children>(card).expect("card has children");
        children
            .iter()
            .filter_map(|child| world.get::<Text>(child))
            .last()
            .expect("card has a body")
            .0
            .clone()
    }

    fn rendered_bodies(world: &World, log: Entity) -> Vec<String> {
        world
            .get::<Children>(log)
            .expect("log has cards")
            .iter()
            .filter(|child| world.get::<TranscriptCard>(*child).is_some())
            .map(|card| card_body_text(world, card))
            .collect()
    }

    #[test]
    fn window_keeps_only_twenty_cards_and_follows_tail() {
        let mut world = World::new();
        let log = world.spawn((Node::default(), TranscriptLog)).id();
        let agent = agent_with_messages(50);
        let mut view = agent_view(log);
        {
            let mut commands = world.commands();
            render_transcript(
                &mut commands,
                &mut view,
                &agent,
                &TextCollapseConfig::default(),
            );
        }
        world.flush();

        let bodies = rendered_bodies(&world, log);
        assert_eq!(bodies.len(), MAX_TRANSCRIPT_CARDS);
        assert_eq!(bodies.first().unwrap(), "message 30");
        assert_eq!(bodies.last().unwrap(), "message 49");
    }

    #[test]
    fn anchored_window_renders_the_selected_slice() {
        let mut world = World::new();
        let log = world.spawn((Node::default(), TranscriptLog)).id();
        let agent = agent_with_messages(50);
        let mut view = agent_view(log);
        view.anchor = TranscriptAnchor::Entry(10);
        {
            let mut commands = world.commands();
            render_transcript(
                &mut commands,
                &mut view,
                &agent,
                &TextCollapseConfig::default(),
            );
        }
        world.flush();

        let bodies = rendered_bodies(&world, log);
        assert_eq!(bodies.len(), MAX_TRANSCRIPT_CARDS);
        assert_eq!(bodies.first().unwrap(), "message 9");
        assert_eq!(bodies.last().unwrap(), "message 28");
    }

    #[test]
    fn live_card_shares_the_twenty_card_budget() {
        let mut world = World::new();
        let log = world.spawn((Node::default(), TranscriptLog)).id();
        let mut agent = agent_with_messages(50);
        agent.mark_active_turn_for_test();
        let mut view = agent_view(log);
        {
            let mut commands = world.commands();
            render_transcript(
                &mut commands,
                &mut view,
                &agent,
                &TextCollapseConfig::default(),
            );
        }
        world.flush();

        let card_count = world.get::<Children>(log).unwrap().len();
        assert_eq!(card_count, MAX_TRANSCRIPT_CARDS);
        assert!(view.assistant.is_some());
        // 19 history cards plus the live assistant card.
        let bodies = rendered_bodies(&world, log);
        assert_eq!(bodies.len(), MAX_TRANSCRIPT_CARDS);
        assert_eq!(bodies.first().unwrap(), "message 31");
        assert_eq!(bodies.last().unwrap(), "...");
    }

    #[test]
    fn anchored_window_can_start_inside_a_short_dialogue() {
        let mut world = World::new();
        let log = world.spawn((Node::default(), TranscriptLog)).id();
        let agent = agent_with_messages(10);
        let mut view = agent_view(log);
        view.anchor = TranscriptAnchor::Entry(5);
        {
            let mut commands = world.commands();
            render_transcript(
                &mut commands,
                &mut view,
                &agent,
                &TextCollapseConfig::default(),
            );
        }
        world.flush();

        let bodies = rendered_bodies(&world, log);
        assert_eq!(bodies.len(), 6);
        assert_eq!(bodies.first().unwrap(), "message 4");
        assert_eq!(bodies.last().unwrap(), "message 9");
    }

    #[test]
    fn scrolling_moves_one_entry_and_returns_to_tail() {
        assert_eq!(
            shifted_anchor(TranscriptAnchor::FollowTail, true, 50, 20, 0),
            TranscriptAnchor::Entry(29)
        );
        assert_eq!(
            shifted_anchor(TranscriptAnchor::Entry(29), false, 50, 20, 0),
            TranscriptAnchor::Entry(30)
        );
        assert_eq!(
            shifted_anchor(TranscriptAnchor::Entry(49), false, 50, 20, 0),
            TranscriptAnchor::FollowTail
        );
        assert_eq!(
            shifted_anchor(TranscriptAnchor::Entry(10), true, 50, 20, 0),
            TranscriptAnchor::Entry(9)
        );
        assert_eq!(
            shifted_anchor(TranscriptAnchor::Entry(10), false, 50, 20, 0),
            TranscriptAnchor::Entry(11)
        );
        assert_eq!(
            shifted_anchor(TranscriptAnchor::Entry(0), true, 50, 20, 0),
            TranscriptAnchor::Entry(0)
        );
    }

    #[test]
    fn scrolling_works_when_the_dialogue_is_shorter_than_the_window() {
        // Ten history entries all fit in the 20-card window. Scrolling up must
        // still move one visible card instead of doing nothing.
        assert_eq!(
            shifted_anchor(TranscriptAnchor::FollowTail, true, 10, 20, 7),
            TranscriptAnchor::Entry(6)
        );
        assert_eq!(
            shifted_anchor(TranscriptAnchor::Entry(6), true, 10, 20, 0),
            TranscriptAnchor::Entry(5)
        );
        assert_eq!(
            shifted_anchor(TranscriptAnchor::Entry(8), false, 10, 20, 0),
            TranscriptAnchor::Entry(9)
        );
        assert_eq!(
            shifted_anchor(TranscriptAnchor::Entry(9), false, 10, 20, 0),
            TranscriptAnchor::FollowTail
        );
    }

    #[test]
    fn refresh_rebuilds_window_when_history_changes() {
        let mut app = App::new();
        let log = app.world_mut().spawn((Node::default(), TranscriptLog)).id();
        let agent = app.world_mut().spawn(agent_with_messages(3)).id();
        let mut view = agent_view(log);
        view.rendered_entries = 1;
        app.world_mut().entity_mut(agent).insert(view);
        app.add_systems(Update, refresh_transcripts);
        app.init_resource::<TextCollapseConfig>();
        app.update();

        assert_eq!(
            app.world()
                .get::<AgentView>(agent)
                .unwrap()
                .rendered_entries,
            4
        );
        assert_eq!(app.world().get::<Children>(log).unwrap().len(), 4);
    }
}

#[cfg(test)]
mod display_tests {
    use super::*;
    use std::time::Duration;

    fn config() -> TextCollapseConfig {
        TextCollapseConfig {
            max_chars: 1_000,
            max_lines: 10,
        }
    }

    fn numbered_lines(count: usize) -> String {
        (0..count)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn short_text_is_not_collapsed() {
        let state = CollapsibleText::new(
            "short".to_owned(),
            &config(),
            CollapseMode::HeadAndTail,
            Entity::PLACEHOLDER,
        );
        assert!(!state.exceeds_limit());
        assert_eq!(state.visible_text(), "short");
        assert_eq!(state.hint_text(), "");
    }

    #[test]
    fn tool_output_collapses_to_the_leading_lines() {
        let state = CollapsibleText::new(
            numbered_lines(20),
            &config(),
            CollapseMode::Head,
            Entity::PLACEHOLDER,
        );
        let visible = state.visible_text();
        assert!(visible.starts_with("line 0"), "{visible}");
        assert!(!visible.contains("line 10"), "{visible}");
        assert!(visible.contains("more characters"), "{visible}");
        assert!(state.hint_text().contains("Show more"));
    }

    #[test]
    fn reasoning_shows_the_first_and_last_lines() {
        let mut state = CollapsibleText::new(
            numbered_lines(20),
            &config(),
            CollapseMode::HeadAndTail,
            Entity::PLACEHOLDER,
        );
        let visible = state.visible_text();
        for index in 0..5 {
            assert!(visible.contains(&format!("line {index}")), "{visible}");
        }
        for index in 15..20 {
            assert!(visible.contains(&format!("line {index}")), "{visible}");
        }
        assert!(!visible.contains("line 7"), "{visible}");
        assert!(visible.contains("..."), "{visible}");
        assert!(state.hint_text().contains("Show more"));

        // New reasoning updates the trailing window while the body stays collapsed.
        state.full.push_str("\nline 20");
        let visible = state.visible_text();
        assert!(visible.contains("line 16"), "{visible}");
        assert!(visible.contains("line 20"), "{visible}");
        assert!(state.collapsed);
    }

    #[test]
    fn long_tool_result_card_starts_collapsed() {
        let mut world = World::new();
        let log = world.spawn_empty().id();
        let body = {
            let mut commands = world.commands();
            spawn_message_card(
                &mut commands,
                log,
                "Tool calls",
                Color::srgb(0.95, 0.75, 0.40),
                numbered_lines(20),
                Color::srgb(0.13, 0.115, 0.09),
                true,
                &config(),
            )
        };
        world.flush();

        let state = world
            .get::<CollapsibleText>(body)
            .expect("tool card body is collapsible");
        assert!(state.collapsed);
        let rendered = &world.get::<Text>(body).unwrap().0;
        assert!(rendered.starts_with("line 0"), "{rendered}");
        assert!(!rendered.contains("line 10"), "{rendered}");
    }

    #[test]
    fn clicking_a_collapsible_header_expands_the_body() {
        use bevy::{
            camera::NormalizedRenderTarget,
            picking::{
                backend::HitData,
                events::Pointer,
                pointer::{Location, PointerButton, PointerId},
            },
        };

        let mut world = World::new();
        let parts = {
            let mut commands = world.commands();
            spawn_collapsible_text(
                &mut commands,
                "Thinking",
                12.,
                Color::WHITE,
                numbered_lines(20),
                15.,
                Color::WHITE,
                CollapseMode::HeadAndTail,
                &config(),
            )
        };
        world.flush();

        let collapsed = world.get::<Text>(parts.body).unwrap().0.clone();
        assert!(!collapsed.contains("line 10"), "{collapsed}");

        let camera = world.spawn_empty().id();
        world.trigger(PointerClick {
            entity: parts.header,
            pointer: Pointer::new(
                PointerId::Mouse,
                Location {
                    position: Vec2::ZERO,
                    target: NormalizedRenderTarget::None {
                        width: 1,
                        height: 1,
                    },
                },
            ),
            button: PointerButton::Primary,
            hit: HitData::new(camera, 0., None, None),
            duration: Duration::ZERO,
            count: 1,
        });

        let expanded = world.get::<Text>(parts.body).unwrap().0.clone();
        assert_eq!(expanded, numbered_lines(20));
        assert!(!world.get::<CollapsibleText>(parts.body).unwrap().collapsed);
    }

    #[test]
    fn thinking_only_assistant_card_is_not_ellipsis() {
        let mut world = World::new();
        let log = world.spawn_empty().id();
        let assistant = {
            let mut commands = world.commands();
            spawn_assistant_card(
                &mut commands,
                log,
                "",
                Some("reasoning"),
                &TextCollapseConfig::default(),
            )
        };
        world.flush();

        assert_eq!(world.get::<Text>(assistant.answer_body).unwrap().0, "");
        assert_eq!(
            world.get::<Text>(assistant.thinking_body).unwrap().0,
            "reasoning"
        );
    }

    #[test]
    fn tool_results_show_the_called_tool_name() {
        let results = vec![ToolCallResult {
            result: "file contents".to_owned(),
            is_error: false,
            tool_use_id: "call-1".to_owned(),
        }];
        let names = HashMap::from([("call-1".to_owned(), "fs_read".to_owned())]);
        let summary = tool_result_summary(&results, &names);
        assert!(summary.contains("fs_read"), "{summary}");
    }
}
