use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::time::{Timer, TimerMode};
use bevy::input::keyboard::KeyboardInput;
use bevy::input::ButtonState;
use bevy::prelude::*;
use bevy::sprite::{Anchor, MaterialMesh2dBundle};
use rand::Rng;
use noise::{NoiseFn, Perlin};
use std::collections::HashMap;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;


// ==================== Constants ====================
const BOID_COUNT: usize = 2000;
const BOID_SPEED_LIMIT: f32 = 300.0;
const TRAIL_LENGTH: usize = 25;
const GRID_CELL_SIZE: f32 = 60.0;
const FIXED_TIME_STEP: f32 = 1.0 / 60.0;


// Noise generation constants
const NOISE_SCALE: f64 = 0.01;
const NOISE_STRENGTH: f32 = 2.0;
const NOISE_INFLUENCE: f32 = 0.2;

// ==================== Components ====================
#[derive(Component)]
struct Boid {
    velocity: Vec2,
    trail: Vec<Vec2>,
    noise_seed: u64,    // u64 for deterministic seeding
    frame_count: u64,   // Frame counter for deterministic noise
}

#[derive(Component)]
struct GridPosition {
    cell: IVec2,
}

#[derive(Component)]
struct GridCellText;

#[derive(Component)]
struct FpsText;

#[derive(Component)]
struct TextInput {
    is_focused: bool,
    buffer: String,
    cursor_visible: bool,
    cursor_timer: Timer,
    cursor_position: usize,
}

// ==================== Resources ====================
#[derive(Resource)]
struct SpatialGrid {
    cells: HashMap<IVec2, Vec<Entity>>,
}

#[derive(Resource)]
struct SimulationState {
    frame: u64,
    boid_states: Vec<BoidState>,
}

#[derive(Clone, Debug)]
struct BoidState {
    position: Vec2,
    velocity: Vec2,
}

#[derive(Resource)]
struct DebugConfig {
    show_grid: bool,
}

#[derive(Resource)]
struct SimulationRng(ChaCha8Rng);

#[derive(Resource)]
struct SimulationParams {
    coherence: f32,
    separation: f32,
    alignment: f32,
    visual_range: f32,
    trace_paths: bool,
}

// ==================== UI Elements ====================
#[derive(Component)]
enum UIElement {
    CoherenceInput,
    SeparationInput,
    AlignmentInput,
    VisualRangeInput,
    ResetButton,
    TracePathsButton,
}

#[derive(Clone)]
struct TextState {
    buffer: String,
    cursor_position: usize,
    cursor_visible: bool,
}

// ==================== Default Implementations ====================

impl Default for Boid {
    fn default() -> Self {
        Self {
            velocity: Vec2::ZERO,
            trail: Vec::with_capacity(TRAIL_LENGTH),
            noise_seed: 0,
            frame_count: 0,
        }
    }
}

impl Default for SpatialGrid {
    fn default() -> Self {
        Self {
            cells: HashMap::new(),
        }
    }
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            show_grid: false,
        }
    }
}

impl SpatialGrid {
    fn get_nearby_entities(&self, cell: IVec2) -> Vec<Entity> { 
        let mut nearby = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                let neighbor_cell = cell + IVec2::new(dx, dy);
                if let Some(entities) = self.cells.get(&neighbor_cell) {
                    nearby.extend(entities);
                }
            }
        }
        nearby
    }
    
    fn world_to_cell(position: Vec2) -> IVec2 { 
        IVec2::new(
            ((position.x + GRID_CELL_SIZE / 2.0) / GRID_CELL_SIZE).floor() as i32,
            ((position.y + GRID_CELL_SIZE / 2.0) / GRID_CELL_SIZE).floor() as i32,
        )
    }
}

impl Default for SimulationRng {
    fn default() -> Self {
        // Use a fixed seed for reproducibility
        SimulationRng(ChaCha8Rng::seed_from_u64(42))
    }
}

// ==================== Main App Setup ====================
fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .insert_resource(SimulationRng::default())
        .insert_resource(SimulationState {
            frame: 0,
            boid_states: Vec::new(),
        })
        .insert_resource(SimulationParams {
            coherence: 0.015,
            separation: 0.25,
            alignment: 0.125,
            visual_range: 60.0,
            trace_paths: false,
        })
        .insert_resource(DebugConfig::default())
        .add_systems(Startup, (setup, setup_ui))
        .add_systems(Update, (
            record_simulation_state,

            // Grid systems
            update_spatial_grid,
            update_boid_movement,
            draw_spatial_grid,
            toggle_grid_visibility,
            
            // Boid movement systems
            // move_boids,
            cleanup_trails_on_toggle,
            update_trails,

            
            // UI systems
            handle_text_input,
            update_fps_text,
            update_text_inputs,
            update_cursor,
            handle_button_clicks,
        ))
        .run();
}

// ==================== Setup Systems ====================
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut rng: ResMut<SimulationRng>,
) {
    commands.spawn(Camera2dBundle::default());
    commands.insert_resource(SpatialGrid::default());
    spawn_boids(&mut commands, &mut meshes, &mut materials, &mut rng);
}

fn spawn_boids(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    rng: &mut ResMut<SimulationRng>,
) {
    let triangle = meshes.add(Mesh::from(RegularPolygon::new(5.0, 3)));
    let material = materials.add(ColorMaterial::from(Color::srgb(0.33, 0.55, 0.95)));

    for i in 0..BOID_COUNT {
        let velocity = Vec2::new(
            rng.0.gen_range(-150.0..150.0),
            rng.0.gen_range(-150.0..150.0)
        );
        let position = Vec2::new(
            rng.0.gen_range(-300.0..300.0),
            rng.0.gen_range(-300.0..300.0)
        );
        let angle = velocity.y.atan2(velocity.x);

        let mut trail = Vec::with_capacity(TRAIL_LENGTH);
        trail.push(position);

        commands.spawn((
            Boid {
                velocity,
                trail,
                noise_seed: i as u64,
                frame_count: 0,
            },
            MaterialMesh2dBundle {
                mesh: triangle.clone().into(),
                material: material.clone(),
                transform: Transform::from_xyz(position.x, position.y, 0.0)
                    .with_rotation(Quat::from_rotation_z(angle - std::f32::consts::FRAC_PI_2)),
                ..default()
            },
            GridPosition {
                cell: SpatialGrid::world_to_cell(position),
            },
        ));
    }
}


fn reset_boids(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    boid_query: &Query<Entity, With<Boid>>,
    grid_text_query: &Query<Entity, With<GridCellText>>,
    rng: &mut ResMut<SimulationRng>,
) {
    // Clean up grid text entities first
    for entity in grid_text_query.iter() {
        commands.entity(entity).despawn_recursive();
    }

    // Then clean up boids
    for entity in boid_query.iter() {
        commands.entity(entity).despawn_recursive();
    }

    // Spawn new boids
    spawn_boids(commands, meshes, materials, rng);
}

// ==================== UI Setup and Systems ====================

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
            // FPS Counter
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

            // Control Panel
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
                spawn_text_input(parent, "Coherence", "0.015", UIElement::CoherenceInput);
                spawn_text_input(parent, "Separation", "0.25", UIElement::SeparationInput);
                spawn_text_input(parent, "Alignment", "0.125", UIElement::AlignmentInput);
                spawn_text_input(parent, "Visual Range", "60.0", UIElement::VisualRangeInput);
                spawn_button(parent, "Reset", UIElement::ResetButton);
                spawn_button(parent, "Trace Paths", UIElement::TracePathsButton);
            });
        });
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
                    cursor_visible: true,
                    cursor_timer: Timer::from_seconds(0.5, TimerMode::Repeating),
                    cursor_position: default_value.len(),
                },
                Interaction::default(),
            ));
        });
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

// Clean up trails when toggling trace paths
fn cleanup_trails_on_toggle(
    mut query: Query<&mut Boid>,
    sim_params: Res<SimulationParams>,
    sim_params_prev: Local<bool>,
) {
    if *sim_params_prev != sim_params.trace_paths {
        if !sim_params.trace_paths {
            for mut boid in query.iter_mut() {
                boid.trail.clear();
            }
        }
    }
}

// ==================== Input Handling Systems ====================
fn handle_text_input(
    mut text_query: Query<(&mut TextInput, &mut Text, &mut BackgroundColor, &Interaction, &UIElement)>,
    mut sim_params: ResMut<SimulationParams>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut keyboard_events: EventReader<KeyboardInput>,
) {
    // Handle focus changes
    if mouse.just_pressed(MouseButton::Left) {
        for (mut input, mut text, mut bg_color, interaction, _) in text_query.iter_mut() {
            let was_focused = input.is_focused;
            input.is_focused = matches!(interaction, Interaction::Pressed);
            
            if input.is_focused != was_focused {
                let state = TextState {
                    buffer: input.buffer.clone(),
                    cursor_position: input.buffer.len(),
                    cursor_visible: input.is_focused,
                };
                
                update_text_state(&mut input, &mut text, state);
            }
            
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
            for (mut input, mut text, _, _, ui_element) in text_query.iter_mut() {
                if !input.is_focused {
                    continue;
                }

                let mut state = TextState {
                    buffer: input.buffer.clone(),
                    cursor_position: input.cursor_position,
                    cursor_visible: true,
                };

                let mut handled = true;
                match event.key_code {
                    KeyCode::ArrowLeft => {
                        if state.cursor_position > 0 {
                            state.cursor_position -= 1;
                        }
                    }
                    KeyCode::ArrowRight => {
                        if state.cursor_position < state.buffer.len() {
                            state.cursor_position += 1;
                        }
                    }
                    KeyCode::Backspace => {
                        if state.cursor_position > 0 {
                            state.buffer.remove(state.cursor_position - 1);
                            state.cursor_position -= 1;
                        }
                    }
                    KeyCode::Delete => {
                        if state.cursor_position < state.buffer.len() {
                            state.buffer.remove(state.cursor_position);
                        }
                    }
                    KeyCode::Enter => {
                        if let Ok(value) = state.buffer.parse::<f32>() {
                            match ui_element {
                                UIElement::CoherenceInput => sim_params.coherence = value,
                                UIElement::SeparationInput => sim_params.separation = value,
                                UIElement::AlignmentInput => sim_params.alignment = value,
                                UIElement::VisualRangeInput => sim_params.visual_range = value,
                                _ => {}
                            }
                        }
                        state.cursor_visible = false;
                        input.is_focused = false;
                    }
                    key_code => {
                        if let Some(char) = key_code_to_char(key_code) {
                            if char.is_ascii_digit() || char == '.' {
                                state.buffer.insert(state.cursor_position, char);
                                state.cursor_position += 1;
                            }
                        } else {
                            handled = false;
                        }
                    }
                }

                if handled {
                    update_text_state(&mut input, &mut text, state);
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

fn update_text_state(input: &mut TextInput, text: &mut Text, state: TextState) {
    input.buffer = state.buffer.clone();
    input.cursor_position = state.cursor_position;
    input.cursor_visible = state.cursor_visible;
    
    text.sections[0].value = if input.is_focused && input.cursor_visible {
        let mut display = input.buffer.clone();
        display.insert(input.cursor_position, '|');
        display
    } else {
        input.buffer.clone()
    };
}

fn update_cursor(
    time: Res<Time>,
    mut query: Query<(&mut TextInput, &mut Text)>,
) {
    for (mut input, mut text) in query.iter_mut() {
        if input.is_focused {
            input.cursor_timer.tick(time.delta());
            
            if input.cursor_timer.just_finished() {
                input.cursor_visible = !input.cursor_visible;
                
                let buffer = input.buffer.clone();
                let cursor_pos = input.cursor_position;
                let show_cursor = input.cursor_visible && input.is_focused;
                text.sections[0].value = update_text_display(&buffer, cursor_pos, show_cursor);
            }
        }
    }
}

fn update_text_display(buffer: &str, cursor_position: usize, show_cursor: bool) -> String {
    if show_cursor {
        let mut display = buffer.to_string();
        display.insert(cursor_position, '|');
        display
    } else {
        buffer.to_string()
    }
}

fn handle_button_clicks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut rng: ResMut<SimulationRng>,
    interaction_query: Query<(&Interaction, &UIElement, &Children), (Changed<Interaction>, With<UIElement>)>,
    mut sim_params: ResMut<SimulationParams>,
    boid_query: Query<Entity, With<Boid>>,
    grid_text_query: Query<Entity, With<GridCellText>>,  // Add this
    mut text_query: Query<&mut Text>,
) {
    for (interaction, ui_element, children) in interaction_query.iter() {
        if let Interaction::Pressed = *interaction {
            match ui_element {
                UIElement::ResetButton => {
                    reset_boids(
                        &mut commands,
                        &mut meshes,
                        &mut materials,
                        &boid_query,
                        &grid_text_query,  // Add this
                        &mut rng,
                    );
                },
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

// ==================== Grid Systems ====================
fn update_spatial_grid(
    mut grid: ResMut<SpatialGrid>,
    query: Query<(Entity, &Transform, &GridPosition), With<Boid>>,
    mut commands: Commands,
) {
    grid.cells.clear();
    
    for (entity, transform, grid_pos) in query.iter() {
        let position = transform.translation.truncate();
        let new_cell = SpatialGrid::world_to_cell(position);
        
        if new_cell != grid_pos.cell {
            commands.entity(entity).insert(GridPosition { cell: new_cell });
        }
        
        grid.cells.entry(new_cell).or_default().push(entity);
    }
}

fn draw_spatial_grid(
    mut commands: Commands,
    mut gizmos: Gizmos,
    grid: Res<SpatialGrid>,
    debug_config: Res<DebugConfig>,
    window_query: Query<&Window>,
    text_query: Query<Entity, With<GridCellText>>,
) {
    // Always clean up existing grid text entities first
    for entity in text_query.iter() {
        commands.entity(entity).despawn();
    }

    // If grid is not visible, just return after cleanup
    if !debug_config.show_grid {
        return;
    }

    let window = window_query.single();
    let width = window.width();
    let height = window.height();
    let half_width = width / 2.0;
    let half_height = height / 2.0;

    // Calculate grid boundaries
    let start_cell_x = ((-half_width) / GRID_CELL_SIZE).floor() as i32;
    let end_cell_x = ((half_width) / GRID_CELL_SIZE).ceil() as i32;
    let start_cell_y = ((-half_height) / GRID_CELL_SIZE).floor() as i32;
    let end_cell_y = ((half_height) / GRID_CELL_SIZE).ceil() as i32;

    // Draw background grid
    for x in start_cell_x..=end_cell_x {
        for y in start_cell_y..=end_cell_y {
            let cell_pos = Vec2::new(
                (x as f32 * GRID_CELL_SIZE) + (GRID_CELL_SIZE / 2.0),
                (y as f32 * GRID_CELL_SIZE) + (GRID_CELL_SIZE / 2.0),
            );
            
            gizmos.rect_2d(
                cell_pos,
                0.0,
                Vec2::new(GRID_CELL_SIZE, GRID_CELL_SIZE),
                Color::srgb(0.2, 0.2, 0.2).with_alpha(0.1),
            );
        }
    }

    // Draw occupied cells and their counts
    for (&cell, entities) in grid.cells.iter() {
        let cell_pos = Vec2::new(
            (cell.x as f32 * GRID_CELL_SIZE) + (GRID_CELL_SIZE / 2.0),
            (cell.y as f32 * GRID_CELL_SIZE) + (GRID_CELL_SIZE / 2.0),
        );

        if cell_pos.x >= -half_width - GRID_CELL_SIZE && 
           cell_pos.x <= half_width + GRID_CELL_SIZE &&
           cell_pos.y >= -half_height - GRID_CELL_SIZE && 
           cell_pos.y <= half_height + GRID_CELL_SIZE {
            
            let density = (entities.len() as f32) / (BOID_COUNT as f32);
            let base_color = Color::srgba(1.0, 0.0, 0.0, density.min(0.5));
            
            gizmos.rect_2d(
                cell_pos,
                0.0,
                Vec2::new(GRID_CELL_SIZE, GRID_CELL_SIZE),
                base_color,
            );

            gizmos.rect_2d(
                cell_pos,
                0.0,
                Vec2::new(GRID_CELL_SIZE, GRID_CELL_SIZE),
                Color::WHITE.with_alpha(0.2),
            );

            if !entities.is_empty() {
                commands.spawn((
                    Text2dBundle {
                        text: Text::from_section(
                            entities.len().to_string(),
                            TextStyle {
                                font_size: 16.0,
                                color: Color::WHITE,
                                ..default()
                            },
                        ),
                        transform: Transform::from_xyz(cell_pos.x, cell_pos.y, 1.0),
                        text_anchor: Anchor::Center,
                        ..default()
                    },
                    GridCellText,
                ));
            }
        }
    }
}

fn toggle_grid_visibility(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut debug_config: ResMut<DebugConfig>,
) {
    if keyboard.just_pressed(KeyCode::KeyG) {
        debug_config.show_grid = !debug_config.show_grid;
    }
}


// ==================== Boid Movement Systems ====================

fn get_deterministic_noise(perlin: &Perlin, boid: &Boid, frame: u64) -> f32 {
    // Create two different noise patterns for more interesting movement
    let noise1 = perlin.get([
        (frame as f64) * (NOISE_SCALE as f64),
        (boid.noise_seed as f64) * 0.1,
        0.0,
        0.0,
    ]) as f32;
    
    let noise2 = perlin.get([
        (boid.noise_seed as f64) * 0.1,
        (frame as f64) * (NOISE_SCALE as f64),
        0.0,
        0.0,
    ]) as f32;
    
    (noise1 + noise2) * 0.5
}

fn update_boid_movement(
    mut query: Query<(Entity, &mut Transform, &mut Boid, &GridPosition)>,
    grid: Res<SpatialGrid>,
    params: Res<SimulationParams>,
    window_query: Query<&Window>,
) {
    let window = window_query.single();
    let width = window.width();
    let height = window.height();
    
    let perlin = Perlin::new(1);
    
    // Cache current positions and velocities
    let boids_data: HashMap<Entity, (Vec2, Vec2)> = query
        .iter()
        .map(|(entity, transform, boid, _)| {
            (entity, (transform.translation.truncate(), boid.velocity))
        })
        .collect();
    
    for (entity, mut transform, mut boid, grid_pos) in query.iter_mut() {
        let current_position = transform.translation.truncate();
        let nearby_entities = (&*grid).get_nearby_entities(grid_pos.cell);
        
        // Calculate flocking forces
        let mut center_of_mass = Vec2::ZERO;
        let mut avoid_vector = Vec2::ZERO;
        let mut average_velocity = Vec2::ZERO;
        let mut num_neighbors = 0;
        
        for &other_entity in &nearby_entities {
            if other_entity == entity {
                continue;
            }
            
            if let Some((other_pos, other_vel)) = boids_data.get(&other_entity) {
                let diff = *other_pos - current_position;
                let distance = diff.length();
                
                if distance < params.visual_range && distance > 0.0 {
                    center_of_mass += *other_pos;
                    
                    if distance < params.visual_range / 2.0 {
                        avoid_vector -= diff.normalize() * (params.visual_range / (2.0 * distance.max(0.1)));
                    }
                    
                    average_velocity += *other_vel;
                    num_neighbors += 1;
                }
            }
        }

        // Apply flocking rules if we have neighbors
        if num_neighbors > 0 {
            center_of_mass /= num_neighbors as f32;
            average_velocity /= num_neighbors as f32;
            
            let coherence = (center_of_mass - current_position) * params.coherence;
            let separation = avoid_vector * params.separation;
            let alignment = (average_velocity - boid.velocity) * params.alignment;
            
            boid.velocity += coherence + separation + alignment;
        }

        // Apply noise influence
        boid.frame_count += 1;
        let noise_value = get_deterministic_noise(&perlin, &boid, boid.frame_count);
        let turn_angle = noise_value * NOISE_STRENGTH * FIXED_TIME_STEP * NOISE_INFLUENCE;
        let (sin, cos) = turn_angle.sin_cos();
        let noise_direction = Vec2::new(
            boid.velocity.x * cos - boid.velocity.y * sin,
            boid.velocity.x * sin + boid.velocity.y * cos
        ).normalize();
        
        boid.velocity = boid.velocity.lerp(noise_direction * boid.velocity.length(), NOISE_INFLUENCE);
        
        // Apply speed limits
        let speed = boid.velocity.length();
        if speed < BOID_SPEED_LIMIT / 2.0 {
            boid.velocity = boid.velocity.normalize() * (BOID_SPEED_LIMIT / 2.0);
        } else if speed > BOID_SPEED_LIMIT {
            boid.velocity = boid.velocity.normalize() * BOID_SPEED_LIMIT;
        }
        
        // Update position with fixed time step
        let new_position = current_position + boid.velocity * FIXED_TIME_STEP;
        
        // Handle screen wrapping
        let wrapped_position = Vec2::new(
            (new_position.x + width / 2.0).rem_euclid(width) - width / 2.0,
            (new_position.y + height / 2.0).rem_euclid(height) - height / 2.0
        );
        
        // Update transform
        transform.translation = wrapped_position.extend(transform.translation.z);
        
        // Update rotation
        if boid.velocity.length_squared() > 0.0 {
            let angle = boid.velocity.y.atan2(boid.velocity.x);
            transform.rotation = Quat::from_rotation_z(angle - std::f32::consts::FRAC_PI_2);
        }

        // Update trail
        if boid.trail.is_empty() {
            boid.trail.push(wrapped_position);
        } else {
            let last_pos = *boid.trail.last().unwrap();
            let distance = (wrapped_position - last_pos).length();
            
            if distance > 5.0 {
                boid.trail.push(wrapped_position);
                if boid.trail.len() > TRAIL_LENGTH {
                    boid.trail.remove(0);
                }
            }
        }
    }
}

fn update_trails(
    mut gizmos: Gizmos,
    query: Query<&Boid>,
    sim_params: Res<SimulationParams>,
    window_query: Query<&Window>,
) {
    if !sim_params.trace_paths {
        return;
    }

    let window = window_query.single();
    let width = window.width();
    let height = window.height();

    for boid in query.iter() {
        if boid.trail.len() < 2 {
            continue;
        }

        for i in 0..boid.trail.len() - 1 {
            let p1 = boid.trail[i];
            let p2 = boid.trail[i + 1];

            // Calculate the shortest path between points considering screen wrapping
            let dx = (p2.x - p1.x).abs();
            let dy = (p2.y - p1.y).abs();
            
            let wrapped_dx = if dx > width / 2.0 {
                if p2.x > p1.x {
                    p2.x - width - p1.x
                } else {
                    p2.x + width - p1.x
                }
            } else {
                p2.x - p1.x
            };

            let wrapped_dy = if dy > height / 2.0 {
                if p2.y > p1.y {
                    p2.y - height - p1.y
                } else {
                    p2.y + height - p1.y
                }
            } else {
                p2.y - p1.y
            };

            // Draw the line if it's a reasonable length
            if wrapped_dx.abs() < width / 2.0 && wrapped_dy.abs() < height / 2.0 {
                gizmos.line_2d(
                    p1,
                    p1 + Vec2::new(wrapped_dx, wrapped_dy),
                    Color::srgba(0.33, 0.55, 0.95, 0.1),
                );
            }
        }
    }
}

fn record_simulation_state(
    query: Query<(&Transform, &Boid)>,
    mut sim_state: ResMut<SimulationState>,
) {
    sim_state.frame += 1;
    sim_state.boid_states.clear();
    
    for (transform, boid) in query.iter() {
        sim_state.boid_states.push(BoidState {
            position: transform.translation.truncate(),
            velocity: boid.velocity,
        });
    }
}

// ==================== Helper Functions ====================
fn update_text_inputs(
    mut text_query: Query<(&mut Text, &UIElement, &Interaction), (Changed<Interaction>, With<UIElement>)>,
    mut sim_params: ResMut<SimulationParams>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    for (text, ui_element, interaction) in text_query.iter_mut() {
        if let Interaction::Pressed = *interaction {
            continue;
        }

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