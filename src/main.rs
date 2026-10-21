use bevy::prelude::*;
use bevy_tokio_tasks::{TaskContext, TokioTasksPlugin, TokioTasksRuntime};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(TokioTasksPlugin::default())
        .add_systems(Startup, spawn_endpoint)
        .run();
}

fn spawn_endpoint(runtime: ResMut<TokioTasksRuntime>, mut commands: Commands) {
} 