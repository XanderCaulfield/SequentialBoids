use bevy::prelude::*;

#[derive(Resource, Default)]
pub struct WindowSize {
    pub width: f32,
    pub height: f32,
}

#[derive(Resource)]
pub struct DebugMode(pub bool);