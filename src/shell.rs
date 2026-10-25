//! Top-level application shell (tabs and panels).

use bevy::{
    input_focus::tab_navigation::{TabGroup, TabIndex},
    picking::hover::Hovered,
    prelude::*,
    ui::Outline,
    ui_widgets::{Activate, Button},
};

use crate::{
    agent::{ActiveAgent, AgentLayout, create_agent},
    config::AppConfig,
    messages::NewAgentRequested,
    virtual_fs::{VfsStateDir, VirtualFs},
};

/// Top-level shell: window layout, static controls, and the first agent.
pub(crate) struct ShellPlugin;

impl Plugin for ShellPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup);
    }
}

fn spawn_action_button(
    commands: &mut Commands,
    label: &str,
    background: Color,
    border: Color,
) -> Entity {
    commands
        .spawn((
            Button,
            Hovered::default(),
            Outline::new(px(2.), px(2.), Color::NONE),
            Node {
                min_width: px(92.),
                height: px(30.),
                padding: UiRect::horizontal(px(10.)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: px(1.).all(),
                border_radius: BorderRadius::all(px(8.)),
                ..default()
            },
            BackgroundColor(background),
            BorderColor::all(border),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(label.to_owned()),
                TextFont {
                    font_size: FontSize::Px(13.),
                    ..default()
                },
                TextColor(Color::srgb(0.88, 0.92, 0.98)),
            ));
        })
        .id()
}

pub(crate) fn setup(mut commands: Commands, config: Res<AppConfig>, storage: Res<VfsStateDir>) {
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

    let title = commands
        .spawn((
            Text::new("LLM Agents"),
            TextFont {
                font_size: FontSize::Px(24.),
                ..default()
            },
            TextColor(Color::srgb(0.92, 0.94, 0.98)),
        ))
        .id();

    let save = spawn_action_button(
        &mut commands,
        "Save VFS",
        Color::srgb(0.10, 0.22, 0.15),
        Color::srgb(0.35, 0.66, 0.43),
    );
    let load = spawn_action_button(
        &mut commands,
        "Load VFS",
        Color::srgb(0.11, 0.17, 0.27),
        Color::srgb(0.36, 0.55, 0.82),
    );

    let vfs_status = commands
        .spawn((
            Text::new(format!("VFS: {}", storage.path.display())),
            TextFont {
                font_size: FontSize::Px(12.),
                ..default()
            },
            TextColor(Color::srgb(0.50, 0.56, 0.66)),
        ))
        .id();

    commands.entity(save).observe(
        move |_: On<Activate>,
              fs: Res<VirtualFs>,
              storage: Res<VfsStateDir>,
              mut texts: Query<&mut Text>| {
            let message = match fs.save_to_dir(&storage.path) {
                Ok(message) => message,
                Err(err) => format!("Save failed: {err}"),
            };
            if let Ok(mut text) = texts.get_mut(vfs_status) {
                text.0 = message;
            }
        },
    );

    commands.entity(load).observe(
        move |_: On<Activate>,
              mut fs: ResMut<VirtualFs>,
              storage: Res<VfsStateDir>,
              mut texts: Query<&mut Text>| {
            let message = match fs.load_from_dir(&storage.path) {
                Ok(message) => message,
                Err(err) => format!("Load failed: {err}"),
            };
            if let Ok(mut text) = texts.get_mut(vfs_status) {
                text.0 = message;
            }
        },
    );

    let title_row = commands
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(8.),
            ..default()
        })
        .add_children(&[title, save, load])
        .id();

    let title_block = commands
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Start,
            row_gap: px(4.),
            ..default()
        })
        .add_children(&[title_row, vfs_status])
        .id();

    let config_text = commands
        .spawn((
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
        .add_children(&[title_block, config_text])
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
            Hovered::default(),
            Outline::new(px(2.), px(2.), Color::NONE),
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
    commands.entity(agent).insert(ActiveAgent);
}
