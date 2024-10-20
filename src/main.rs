use bevy::prelude::*;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};

mod components;
mod resources;
mod systems;
mod constants;

use systems::*;
use resources::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(LogDiagnosticsPlugin::default())
        .insert_resource(DebugMode(false))
        .init_resource::<WindowSize>()
        .add_systems(Startup, setup)
        .add_systems(Update, (
            update_boids,
            move_boids,
            update_window_size,
            toggle_debug_mode,
            draw_debug_cones,
        ))
        .run();
}