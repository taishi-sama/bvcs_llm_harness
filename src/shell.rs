//! Top-level application shell (tabs and panels).

use bevy::{
    input_focus::tab_navigation::{TabGroup, TabIndex},
    prelude::*,
    ui_widgets::{Activate, Button},
};

use crate::{
    agent::{ActiveAgent, AgentLayout, create_agent},
    config::AppConfig,
    messages::NewAgentRequested,
};

pub(crate) fn setup(mut commands: Commands, config: Res<AppConfig>) {
    commands.spawn((Camera2d, IsDefaultUiCamera));

    let root = commands
        .spawn((
            Node {
                width: percent(100.),
                height: percent(100.),
                flex_direction: FlexDirection::Column,
                padding: px(18.).all(),
                row_gap: px(12.),
                ..default()
            },
            BackgroundColor(Color::srgb(0.055, 0.065, 0.085)),
        ))
        .id();

    let header = commands
        .spawn(Node {
            width: percent(100.),
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                Text::new("LLM Agents"),
                TextFont {
                    font_size: FontSize::Px(24.),
                    ..default()
                },
                TextColor(Color::srgb(0.92, 0.94, 0.98)),
            ));
            parent.spawn((
                Text::new(format!(
                    "{} | {} | {}",
                    config.provider.label(),
                    config.model_name,
                    config.endpoint
                )),
                TextFont {
                    font_size: FontSize::Px(13.),
                    ..default()
                },
                TextColor(Color::srgb(0.55, 0.60, 0.70)),
            ));
        })
        .id();

    let tabs = commands
        .spawn((
            Node {
                flex_grow: 1.,
                flex_direction: FlexDirection::Row,
                flex_wrap: FlexWrap::Wrap,
                align_items: AlignItems::Center,
                column_gap: px(8.),
                row_gap: px(6.),
                ..default()
            },
            TabGroup::new(0),
        ))
        .id();

    let plus = commands
        .spawn((
            Button,
            Node {
                min_width: px(38.),
                height: px(34.),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(1.).all(),
                border_radius: BorderRadius::all(px(8.)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.14, 0.17, 0.22)),
            BorderColor::all(Color::srgb(0.30, 0.36, 0.46)),
            TabIndex(0),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("+"),
                TextFont {
                    font_size: FontSize::Px(20.),
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.90, 0.98)),
            ));
        })
        .id();
    commands
        .entity(plus)
        .observe(|_: On<Activate>, mut commands: Commands| {
            commands.write_message(NewAgentRequested);
        });

    let tab_bar = commands
        .spawn(Node {
            width: percent(100.),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(8.),
            ..default()
        })
        .add_children(&[tabs, plus])
        .id();

    let panels = commands
        .spawn((
            Node {
                width: percent(100.),
                flex_basis: px(0.),
                flex_grow: 1.,
                min_height: px(0.),
                ..default()
            },
            TabGroup::new(1),
        ))
        .id();

    commands
        .entity(root)
        .add_children(&[header, tab_bar, panels]);

    let layout = AgentLayout {
        tabs,
        panels,
        next_number: 2,
    };
    let agent = create_agent(&mut commands, &layout, &config, "LLM Agent 1".to_owned(), 1);

    commands.insert_resource(layout);
    commands.insert_resource(ActiveAgent(Some(agent)));
}
