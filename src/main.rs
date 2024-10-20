use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use rand::Rng;


const BOID_COUNT: usize = 2000;
const BOID_SPEED_LIMIT: f32 = 300.0;
const MAX_TURN_RATE: f32 = 30.0;
const MAX_RANDOM_TURN_INTERVAL: f32 = 3.0;
const TRAIL_LENGTH: usize = 50;

#[derive(Component)]
struct Boid {
    velocity: Vec2,
    trail: Vec<Vec2>,
    last_random_turn_time: f32,
    next_random_turn_interval: f32,
}

#[derive(Resource)]
struct SimulationParams {
    coherence: f32,
    separation: f32,
    alignment: f32,
    visual_range: f32,
    trace_paths: bool,
}

#[derive(Component)]
enum UIElement {
    CoherenceSlider,
    SeparationSlider,
    AlignmentSlider,
    VisualRangeSlider,
    ResetButton,
    TracePathsButton,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(SimulationParams {
            coherence: 0.05,
            separation: 0.05,
            alignment: 0.05,
            visual_range: 75.0,
            trace_paths: true,
        })
        .add_systems(Startup, (setup, setup_ui))
        .add_systems(Update, (
            update_boids,
            move_boids,
            update_sliders,
            handle_button_clicks,
            update_trails,
        ))
        .run();
}
fn spawn_boids(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
) {
    let mut rng = rand::thread_rng();
    let triangle = meshes.add(Mesh::from(RegularPolygon::new(5.0, 3)));
    let material = materials.add(ColorMaterial::from(Color::srgb(0.33, 0.55, 0.95)));

    for _ in 0..BOID_COUNT {
        let velocity = Vec2::new(rng.gen_range(-150.0..150.0), rng.gen_range(-150.0..150.0));
        let position = Vec2::new(rng.gen_range(-300.0..300.0), rng.gen_range(-300.0..300.0));
        let angle = velocity.y.atan2(velocity.x);

        commands.spawn((
            Boid {
                velocity,
                trail: Vec::new(),
                last_random_turn_time: 0.,
                next_random_turn_interval: (rng.gen_range(0.0..MAX_RANDOM_TURN_INTERVAL))
            },
            MaterialMesh2dBundle {
                mesh: triangle.clone().into(),
                material: material.clone(),
                transform: Transform::from_xyz(position.x, position.y, 0.0)
                    .with_rotation(Quat::from_rotation_z(angle - std::f32::consts::FRAC_PI_2)),
                ..default()
            },
        ));
    }
}

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<ColorMaterial>>) {
    commands.spawn(Camera2dBundle::default());
    spawn_boids(&mut commands, &mut meshes, &mut materials);
}

fn reset_boids(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    boid_query: &Query<Entity, With<Boid>>,
) {
    // Despawn existing boids
    for entity in boid_query.iter() {
        commands.entity(entity).despawn();
    }

    // Spawn new boids
    spawn_boids(commands, meshes, materials);
}


fn setup_ui(mut commands: Commands) {
    commands
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::FlexEnd,
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            parent
                .spawn(NodeBundle {
                    style: Style {
                        width: Val::Percent(100.0),
                        height: Val::Px(100.0),
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::SpaceEvenly,
                        ..default()
                    },
                    background_color: Color::srgba(0.1, 0.1, 0.1, 0.5).into(),
                    ..default()
                })
                .with_children(|parent| {
                    spawn_slider(parent, "Coherence", UIElement::CoherenceSlider);
                    spawn_slider(parent, "Separation", UIElement::SeparationSlider);
                    spawn_slider(parent, "Alignment", UIElement::AlignmentSlider);
                    spawn_slider(parent, "Visual Range", UIElement::VisualRangeSlider);
                    spawn_button(parent, "Reset", UIElement::ResetButton);
                    spawn_button(parent, "Trace Paths", UIElement::TracePathsButton);
                });
        });
}


fn spawn_slider(parent: &mut ChildBuilder, label: &str, ui_element: UIElement) {
    parent
        .spawn(NodeBundle {
            style: Style {
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            parent.spawn(TextBundle::from_section(
                label,
                TextStyle {
                    font_size: 16.0,
                    color: Color::WHITE,
                    ..default()
                },
            ));
            parent
                .spawn(NodeBundle {
                    style: Style {
                        width: Val::Px(200.0),
                        height: Val::Px(20.0),
                        ..default()
                    },
                    background_color: Color::srgb_u8(38, 38, 38).into(),
                    ..default()
                })
                .with_children(|parent| {
                    parent.spawn((
                        NodeBundle {
                            style: Style {
                                width: Val::Percent(50.0),
                                height: Val::Percent(100.0),
                                ..default()
                            },
                            background_color: Color::srgb_u8(102, 102, 102).into(),
                            ..default()
                        },
                        ui_element,
                        Interaction::default(),
                    ));
                });
        });
}

fn update_sliders(
    mut interaction_query: Query<(&Interaction, &UIElement, &mut Style, &Node, &GlobalTransform), (Changed<Interaction>, With<UIElement>)>,
    mut sim_params: ResMut<SimulationParams>,
    q_window: Query<&Window>,
) {
    let window = q_window.single();
    
    for (interaction, ui_element, mut style, node, transform) in interaction_query.iter_mut() {
        if let Interaction::Pressed = *interaction {
            if let Some(cursor_position) = window.cursor_position() {
                let node_width = node.size().x;
                let relative_x = (cursor_position.x - transform.translation().x) / node_width;
                let value = relative_x.clamp(0.0, 1.0);
                
                match ui_element {
                    UIElement::CoherenceSlider => sim_params.coherence = value * 0.02,
                    UIElement::SeparationSlider => sim_params.separation = value * 0.2,
                    UIElement::AlignmentSlider => sim_params.alignment = value * 0.1,
                    UIElement::VisualRangeSlider => sim_params.visual_range = value * 200.0,
                    _ => {}
                }
                
                // Update the slider handle width
                style.width = Val::Percent(value * 100.0);
            }
        }
    }
}

fn spawn_button(parent: &mut ChildBuilder, text: &str, ui_element: UIElement) {
    parent.spawn((
        ButtonBundle {
            style: Style {
                width: Val::Px(150.0),
                height: Val::Px(50.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            background_color: Color::srgb_u8(38, 38, 38).into(),
            ..default()
        },
        ui_element,
    ))
    .with_children(|parent| {
        parent.spawn(TextBundle::from_section(
            text,
            TextStyle {
                font_size: 20.0,
                color: Color::WHITE,
                ..default()
            },
        ));
    });
}


fn handle_button_clicks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    interaction_query: Query<(&Interaction, &UIElement, &Children), (Changed<Interaction>, With<UIElement>)>,
    mut sim_params: ResMut<SimulationParams>,
    boid_query: Query<Entity, With<Boid>>,
    mut text_query: Query<&mut Text>,
) {
    for (interaction, ui_element, children) in interaction_query.iter() {
        if let Interaction::Pressed = *interaction {
            match ui_element {
                UIElement::ResetButton => reset_boids(&mut commands, &mut meshes, &mut materials, &boid_query),
                UIElement::TracePathsButton => {
                    sim_params.trace_paths = !sim_params.trace_paths;
                    if let Some(child) = children.first() {
                        if let Ok(mut text) = text_query.get_mut(*child) {
                            text.sections[0].value = if sim_params.trace_paths {
                                "Hide Paths".to_string()
                            } else {
                                "Show Paths".to_string()
                            };
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
fn update_boids(
    mut query: Query<(&mut Transform, &mut Boid)>,
    params: Res<SimulationParams>,
    window_query: Query<&Window>,
    time: Res<Time>,
) {
    let window = window_query.single();
    let width = window.width();
    let height = window.height();

    let boids: Vec<(Vec2, Vec2)> = query
        .iter()
        .map(|(transform, boid)| (transform.translation.truncate(), boid.velocity))
        .collect();

    let mut rng = rand::thread_rng();

    for (mut transform, mut boid) in query.iter_mut() {
        let mut position = transform.translation.truncate();

        let mut center_of_mass = Vec2::ZERO;
        let mut avoid_vector = Vec2::ZERO;
        let mut average_velocity = Vec2::ZERO;
        let mut num_neighbors = 0;

        for (other_pos, other_vel) in &boids {
            let diff = *other_pos - position;
            let distance = diff.length();

            if distance < params.visual_range && distance > 0.0 {
                // Coherence
                center_of_mass += *other_pos;

                // Separation (increased effect for closer boids)
                if distance < params.visual_range / 2.0 {
                    avoid_vector -= diff.normalize() * (params.visual_range / (2.0 * distance.max(0.1)));
                }

                // Alignment
                average_velocity += *other_vel;

                num_neighbors += 1;
            }
        }

        if num_neighbors > 0 {
            center_of_mass /= num_neighbors as f32;
            average_velocity /= num_neighbors as f32;

            let coherence = (center_of_mass - position) * params.coherence;
            let separation = avoid_vector * params.separation;
            let alignment = (average_velocity - boid.velocity) * params.alignment;

            boid.velocity += coherence + separation + alignment;
        }

        // Apply random turning at individual random intervals
        if time.elapsed_seconds() - boid.last_random_turn_time > boid.next_random_turn_interval {
            let turn_angle = rng.gen_range(-MAX_TURN_RATE..MAX_TURN_RATE) * time.delta_seconds();
            let (sin, cos) = turn_angle.sin_cos();
            let new_velocity = Vec2::new(
                boid.velocity.x * cos - boid.velocity.y * sin,
                boid.velocity.x * sin + boid.velocity.y * cos
            );
            boid.velocity = new_velocity.normalize() * boid.velocity.length();
            
            boid.last_random_turn_time = time.elapsed_seconds();
            boid.next_random_turn_interval = rng.gen_range(0.0..MAX_RANDOM_TURN_INTERVAL);
        }

        // Maintain speed
        let current_speed = boid.velocity.length();
        if current_speed < BOID_SPEED_LIMIT / 2.0 {
            boid.velocity = boid.velocity.normalize() * (BOID_SPEED_LIMIT / 2.0);
        } else if current_speed > BOID_SPEED_LIMIT {
            boid.velocity = boid.velocity.normalize() * BOID_SPEED_LIMIT;
        }

        // Update position
        let delta_time = time.delta_seconds();
        position += boid.velocity * delta_time;

        // Wrap around screen edges
        position.x = (position.x + width) % width - width / 2.0;
        position.y = (position.y + height) % height - height / 2.0;

        transform.translation = position.extend(transform.translation.z);

        // Update rotation to face velocity direction
        if boid.velocity.length_squared() > 0.0 {
            let angle = boid.velocity.y.atan2(boid.velocity.x);
            transform.rotation = Quat::from_rotation_z(angle - std::f32::consts::FRAC_PI_2);
        }
    }
}

fn move_boids(
    mut query: Query<(&mut Transform, &mut Boid)>,
    time: Res<Time>,
    window_query: Query<&Window>,
) {
    let window = window_query.single();
    let width = window.width();
    let height = window.height();

    for (mut transform, mut boid) in query.iter_mut() {
        let old_position = transform.translation.truncate();
        let new_position = old_position + boid.velocity * time.delta_seconds();

        // Wrap around screen edges
        let wrapped_position = Vec2::new(
            (new_position.x + width) % width - width / 2.0,
            (new_position.y + height) % height - height / 2.0,
        );

        transform.translation = wrapped_position.extend(transform.translation.z);

        // Update trail
        if boid.trail.is_empty() || wrapped_position.distance(*boid.trail.last().unwrap()) > 5.0 {
            boid.trail.push(wrapped_position);
            if boid.trail.len() > TRAIL_LENGTH {
                boid.trail.remove(0);
            }
        }
    }
}

fn update_trails(mut gizmos: Gizmos, query: Query<&Boid>, sim_params: Res<SimulationParams>, window_query: Query<&Window>) {
    if sim_params.trace_paths {
        let window = window_query.single();
        let width = window.width();
        let height = window.height();

        for boid in query.iter() {
            for window in boid.trail.windows(2) {
                if let [start, end] = window {
                    let start_wrapped = Vec2::new(
                        (start.x + width) % width - width / 2.0,
                        (start.y + height) % height - height / 2.0,
                    );
                    let end_wrapped = Vec2::new(
                        (end.x + width) % width - width / 2.0,
                        (end.y + height) % height - height / 2.0,
                    );

                    // Check if the line crosses the screen edge
                    if (start_wrapped - end_wrapped).length() > width / 2.0 || (start_wrapped - end_wrapped).length() > height / 2.0 {
                        continue; // Skip drawing this line segment
                    }

                    gizmos.line_2d(start_wrapped, end_wrapped, Color::srgba(0.33, 0.55, 0.95, 0.1));
                }
            }
        }
    }
}
