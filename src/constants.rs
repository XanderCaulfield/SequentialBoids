use std::f32::consts::PI;

pub const BOID_SPEED: f32 = 200.0;
pub const BOID_COUNT: usize = 20;
pub const PERCEPTION_RADIUS: f32 = 50.0;
pub const PERCEPTION_ANGLE: f32 = PI / 2.0; // 90 degrees in radians
pub const SEPARATION_RADIUS: f32 = 25.0;

pub const SEPARATION_WEIGHT: f32 = 5.0;
pub const ALIGNMENT_WEIGHT: f32 = 1.0;
pub const COHESION_WEIGHT: f32 = 1.0;