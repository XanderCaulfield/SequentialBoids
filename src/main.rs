use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::keyboard::KeyboardInput;
use bevy::input::ButtonState;
use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use rand::Rng;

// Constants
const BOID_COUNT: usize = 2000;
const BOID_SPEED_LIMIT: f32 = 300.0;
const MAX_TURN_RATE: f32 = 30.0;
const MAX_RANDOM_TURN_INTERVAL: f32 = 3.0;
const TRAIL_LENGTH: usize = 50;

// The Boid itself.
#[derive(Component)]
struct Boid {
    velocity: Vec2,
    trail: Vec<Vec2>,
    last_random_turn_time: f32,
    next_random_turn_interval: f32,
}

// Parameters that influence the behaviour of the boids.
#[derive(Resource)]
struct SimulationParams {
    coherence: f32,
    separation: f32,
    alignment: f32,
    visual_range: f32,
    trace_paths: bool,
}

// User interface elements.
#[derive(Component)]
enum UIElement {
    CoherenceInput,
    SeparationInput,
    AlignmentInput,
    VisualRangeInput,
    ResetButton,
    TracePathsButton,
}
// FPS Counter
#[derive(Component)]
struct FpsText;

// Text Input Fields
#[derive(Component)]
struct TextInput {
    is_focused: bool,
    buffer: String,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .insert_resource(SimulationParams {
            coherence: 0.04,
            separation: 0.2,
            alignment: 0.03,
            visual_range: 100.0,
            trace_paths: true,
        })
        .add_systems(Startup, (setup, setup_ui))
        .add_systems(Update, (
            update_boids,
            move_boids,
            handle_text_input,
            update_fps_text,
            update_text_inputs,
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

            // Add FPS counter to the top left of the screen

            parent.spawn(NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    top: Val::Px(10.0),
                    left: Val::Px(10.0),
                    ..default()
                },
                ..default()
            })
            .with_children(|parent| {
                parent.spawn((
                    TextBundle::from_section(
                        "FPS: --",
                        TextStyle {
                            font_size: 20.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    ),
                    FpsText,
                ));
            });


            parent.spawn(NodeBundle {
                    style: Style {
                        width: Val::Percent(100.0),
                        height: Val::Px(100.0),
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::SpaceEvenly,
                        padding: UiRect::all(Val::Px(10.0)),
                        ..default()
                    },
                    background_color: Color::srgba(0.1, 0.1, 0.1, 0.5).into(),
                    ..default()
                })
                .with_children(|parent| {
                    spawn_text_input(parent, "Coherence", "0.05", UIElement::CoherenceInput);
                    spawn_text_input(parent, "Separation", "0.1", UIElement::SeparationInput);
                    spawn_text_input(parent, "Alignment", "0.05", UIElement::AlignmentInput);
                    spawn_text_input(parent, "Visual Range", "75.0", UIElement::VisualRangeInput);
                    spawn_button(parent, "Reset", UIElement::ResetButton);
                    spawn_button(parent, "Trace Paths", UIElement::TracePathsButton);
                });
        });
}

fn update_fps_text(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
) {
    if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
        if let Some(value) = fps.smoothed() {
            for mut text in query.iter_mut() {
                text.sections[0].value = format!("FPS: {:.1}", value);
            }
        }
    }
}

fn spawn_text_input(parent: &mut ChildBuilder, label: &str, default_value: &str, ui_element: UIElement) {
    parent
        .spawn(NodeBundle {
            style: Style {
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Start,
                margin: UiRect {
                    top: Val::ZERO,
                    bottom: Val::Px(5.0),
                    left: Val::ZERO,
                    right: Val::ZERO,
                },
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            // Label
            parent.spawn(TextBundle::from_section(
                label,
                TextStyle {
                    font_size: 16.0,
                    color: Color::WHITE,
                    ..default()
                },
            ));
            
            // Text input box
            parent.spawn((
                TextBundle {
                    text: Text::from_section(
                        default_value.to_string(),
                        TextStyle {
                            font_size: 16.0,
                            color: Color::BLACK,
                            ..default()
                        },
                    ),
                    style: Style {
                        width: Val::Px(80.0),
                        height: Val::Px(30.0),
                        padding: UiRect::all(Val::Px(5.0)),
                        ..default()
                    },
                    background_color: Color::WHITE.into(),
                    ..default()
                },
                ui_element,
                TextInput {
                    is_focused: false,
                    buffer: default_value.to_string(),
                },
                Interaction::default(),
            ));
        });
}

fn handle_text_input(
    mut text_query: Query<(&mut TextInput, &mut Text, &mut BackgroundColor, &Interaction, &UIElement)>,
    mut sim_params: ResMut<SimulationParams>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut keyboard_events: EventReader<KeyboardInput>,
) {
    // Handle focus changes
    if mouse.just_pressed(MouseButton::Left) {
        for (mut input, _, mut bg_color, interaction, _) in text_query.iter_mut() {
            input.is_focused = matches!(interaction, Interaction::Pressed);
            *bg_color = if input.is_focused {
                Color::srgb(0.9, 0.9, 1.0).into()
            } else {
                Color::WHITE.into()
            };
        }
    }

    // Handle keyboard input
    for event in keyboard_events.read() {
        if event.state == ButtonState::Pressed {
            // Find the focused text input
            for (mut input, mut text, _, _, ui_element) in text_query.iter_mut() {
                if input.is_focused {
                    match event.key_code {
                        KeyCode::Backspace => {
                            input.buffer.pop();
                            text.sections[0].value = input.buffer.clone();
                        }
                        KeyCode::Enter => {
                            if let Ok(value) = input.buffer.parse::<f32>() {
                                // Update simulation parameters based on the UI element type
                                match ui_element {
                                    UIElement::CoherenceInput => sim_params.coherence = value,
                                    UIElement::SeparationInput => sim_params.separation = value,
                                    UIElement::AlignmentInput => sim_params.alignment = value,
                                    UIElement::VisualRangeInput => sim_params.visual_range = value,
                                    _ => {}
                                }
                            }
                            input.is_focused = false;
                        }
                        // Handle numeric input and decimal point
                        key_code => {
                            if let Some(char) = key_code_to_char(key_code) {
                                if char.is_ascii_digit() || char == '.' {
                                    input.buffer.push(char);
                                    text.sections[0].value = input.buffer.clone();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn key_code_to_char(key_code: KeyCode) -> Option<char> {
    match key_code {
        KeyCode::Digit0 | KeyCode::Numpad0 => Some('0'),
        KeyCode::Digit1 | KeyCode::Numpad1 => Some('1'),
        KeyCode::Digit2 | KeyCode::Numpad2 => Some('2'),
        KeyCode::Digit3 | KeyCode::Numpad3 => Some('3'),
        KeyCode::Digit4 | KeyCode::Numpad4 => Some('4'),
        KeyCode::Digit5 | KeyCode::Numpad5 => Some('5'),
        KeyCode::Digit6 | KeyCode::Numpad6 => Some('6'),
        KeyCode::Digit7 | KeyCode::Numpad7 => Some('7'),
        KeyCode::Digit8 | KeyCode::Numpad8 => Some('8'),
        KeyCode::Digit9 | KeyCode::Numpad9 => Some('9'),
        KeyCode::Period | KeyCode::NumpadDecimal => Some('.'),
        _ => None,
    }
}

fn update_text_inputs(
    mut text_query: Query<(&mut Text, &UIElement, &Interaction), (Changed<Interaction>, With<UIElement>)>,
    mut sim_params: ResMut<SimulationParams>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    for (text, ui_element, interaction) in text_query.iter_mut() {
        if let Interaction::Pressed = *interaction {
            // Handle text input focus if needed
            continue;
        }

        // Only process if Enter is pressed
        if !keyboard.just_pressed(KeyCode::Enter) {
            continue;
        }

        let current_value = &text.sections[0].value;
        if let Ok(value) = current_value.parse::<f32>() {
            match ui_element {
                UIElement::CoherenceInput => sim_params.coherence = value,
                UIElement::SeparationInput => sim_params.separation = value,
                UIElement::AlignmentInput => sim_params.alignment = value,
                UIElement::VisualRangeInput => sim_params.visual_range = value,
                _ => {}
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
                // Add catch-all for input elements since they don't need button click handling
                UIElement::CoherenceInput | UIElement::SeparationInput | 
                UIElement::AlignmentInput | UIElement::VisualRangeInput => {}
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
