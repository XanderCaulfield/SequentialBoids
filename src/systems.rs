use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use rand::Rng;
use std::time::Instant;

use crate::components::*;
use crate::resources::*;
use crate::constants::*;

pub fn setup(
    mut commands: Commands,
    window_query: Query<&Window, With<PrimaryWindow>>,
) {
    let window = window_query.single();
    
    commands.insert_resource(WindowSize {
        width: window.width(),
        height: window.height(),
    });

    commands.spawn(Camera2dBundle::default());

    let mut rng = rand::thread_rng();

    for _ in 0..BOID_COUNT {
        let position = Vec2::new(
            rng.gen_range(-window.width()/2.0..window.width()/2.0),
            rng.gen_range(-window.height()/2.0..window.height()/2.0),
        );
        let velocity = Vec2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)).normalize() * BOID_SPEED;

        commands.spawn((
            SpriteBundle {
                sprite: Sprite {
                    color: Color::srgb(0.8, 0.8, 1.0), // Light blue color
                    custom_size: Some(Vec2::new(10.0, 10.0)), // Size of the boid
                    ..default()
                },
                transform: Transform::from_translation(position.extend(0.0)),
                ..default()
            },
            Boid { velocity },
        ));
    }
}

pub fn update_boids(mut query: Query<(&Transform, &mut Boid, &mut Sprite)>, time: Res<Time>) {
    let start_time = Instant::now();
    
    let boids: Vec<(Vec2, Vec2)> = query
        .iter()
        .map(|(transform, boid, _)| (transform.translation.truncate(), boid.velocity))
        .collect();

    for (transform, mut boid, mut sprite) in query.iter_mut() {
        let position = transform.translation.truncate();
        
        let mut separation = Vec2::ZERO;
        let mut alignment = Vec2::ZERO;
        let mut cohesion = Vec2::ZERO;
        let mut total = 0;
        let mut close_boids = 0;

        for &(other_pos, other_vel) in &boids {
            let to_other = other_pos - position;
            let distance = to_other.length();
            let angle = boid.velocity.angle_between(to_other);

            if distance > 0.0 && distance < PERCEPTION_RADIUS && angle.abs() < PERCEPTION_ANGLE / 2.0 {
                if distance < SEPARATION_RADIUS {
                    separation -= to_other.normalize() / distance;
                    close_boids += 1;
                }
                alignment += other_vel;
                cohesion += other_pos;
                total += 1;
            }
        }

        if total > 0 {
            separation = if close_boids > 0 { separation / close_boids as f32 } else { Vec2::ZERO };
            alignment = (alignment / total as f32) - boid.velocity;
            cohesion = (cohesion / total as f32) - position;

            let mut acceleration = Vec2::ZERO;
            acceleration += separation * SEPARATION_WEIGHT;
            acceleration += alignment * ALIGNMENT_WEIGHT;
            acceleration += cohesion * COHESION_WEIGHT;

            boid.velocity += acceleration * time.delta_seconds();

            if boid.velocity.length() > BOID_SPEED {
                boid.velocity = boid.velocity.normalize() * BOID_SPEED;
            }

            let pressure = (separation * SEPARATION_WEIGHT).length() / SEPARATION_WEIGHT;
            let pressure = pressure.clamp(0.0, 1.0);

            sprite.color = pressure_to_color(pressure);
        }
    }

    let duration = start_time.elapsed();
    println!("Update boids took: {:?}", duration);
}

pub fn move_boids(
    mut query: Query<(&mut Transform, &Boid)>,
    window_size: Res<WindowSize>,
    time: Res<Time>,
) {
    for (mut transform, boid) in query.iter_mut() {
        let mut position = transform.translation.truncate() + boid.velocity * time.delta_seconds();
        
        position.x = wrap(position.x, -window_size.width/2.0, window_size.width/2.0);
        position.y = wrap(position.y, -window_size.height/2.0, window_size.height/2.0);
        
        transform.translation = position.extend(transform.translation.z);
        
        let angle = boid.velocity.y.atan2(boid.velocity.x);
        transform.rotation = Quat::from_rotation_z(angle - std::f32::consts::PI / 2.0);
    }
}

pub fn update_window_size(
    mut window_size: ResMut<WindowSize>,
    window_query: Query<&Window, With<PrimaryWindow>>,
) {
    let window = window_query.single();
    window_size.width = window.width();
    window_size.height = window.height();
}

pub fn toggle_debug_mode(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut debug_mode: ResMut<DebugMode>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyD) {
        debug_mode.0 = !debug_mode.0;
        println!("Debug mode: {}", if debug_mode.0 { "ON" } else { "OFF" });
    }
}

pub fn draw_debug_cones(
    query: Query<(&Transform, &Boid)>,
    debug_mode: Res<DebugMode>,
    mut gizmos: Gizmos,
) {
    if debug_mode.0 {
        for (transform, boid) in query.iter() {
            let position = transform.translation.truncate();
            let direction = boid.velocity.normalize();

            let cone_point2 = position + direction.rotate(Vec2::from_angle(PERCEPTION_ANGLE / 2.0)) * PERCEPTION_RADIUS;
            let cone_point3 = position + direction.rotate(Vec2::from_angle(-PERCEPTION_ANGLE / 2.0)) * PERCEPTION_RADIUS;

            gizmos.line_2d(position, cone_point2, Color::srgba(0.0, 1.0, 0.0, 0.2));
            gizmos.line_2d(position, cone_point3, Color::srgba(0.0, 1.0, 0.0, 0.2));
            gizmos.line_2d(cone_point2, cone_point3, Color::srgba(0.0, 1.0, 0.0, 0.2));
        }
    }
}

fn wrap(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        max - (min - value) % (max - min)
    } else if value >= max {
        min + (value - min) % (max - min)
    } else {
        value
    }
}

fn pressure_to_color(pressure: f32) -> Color {
    let hue = (1.0 - pressure) * 120.0; // 120 for green, 0 for red
    let saturation = 1.0;
    let value = 0.8;
    Color::hsla(hue, saturation, value, 1.0)
}