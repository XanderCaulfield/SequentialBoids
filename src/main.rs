use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::time::{Timer, TimerMode};
use bevy::input::keyboard::KeyboardInput;
use bevy::input::ButtonState;
use bevy::prelude::*;
use bevy::sprite::{Anchor, MaterialMesh2dBundle};
use rand::Rng;
use noise::{NoiseFn, Perlin};
use std::collections::HashMap;

// Constants
const BOID_COUNT: usize = 2000;
const BOID_SPEED_LIMIT: f32 = 300.0;
const TRAIL_LENGTH: usize = 25;
const GRID_CELL_SIZE: f32 = 60.0;


// Noise gen for random movement
const NOISE_SCALE: f32 = 0.1;    // Adjusts how quickly the noise pattern changes.
const NOISE_STRENGTH: f32 = 100.0;  // Adjusts how strong the random turning force is.
const TIME_SCALE: f32 = 0.02;       // How quickly the noise pattern evolves over time

// The Almighty Boid
#[derive(Component)]
struct Boid {
    velocity: Vec2,
    trail: Vec<Vec2>,
    noise_offset_x: f32,
    noise_offset_y: f32,
    noise_offset_z: f32,    // For time-based variation.
    noise_seed: f32,        // For per-boid variation.
}

#[derive(Component)]
struct GridPosition {
    cell: IVec2,
}

#[derive(Resource)]
struct SpatialGrid {
    cells: HashMap<IVec2, Vec<Entity>>,
}

impl Default for SpatialGrid {
    fn default() -> Self {
        Self {
            cells: HashMap::new(),
        }
    }
}

impl SpatialGrid {
    // Get nearby entities within the visual range
    fn get_nearby_entities(&self, cell: IVec2) -> Vec<Entity> {
        let mut nearby = Vec::new();
        
        // Check current cell and all adjacent cells
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
    
    // Convert world position to grid cell
    fn world_to_cell(position: Vec2) -> IVec2 {
        IVec2::new(
            (position.x / GRID_CELL_SIZE).floor() as i32,
            (position.y / GRID_CELL_SIZE).floor() as i32,
        )
    }
}

#[derive(Resource)]
struct DebugConfig {
    show_grid: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            show_grid: false,
        }
    }
}

#[derive(Component)]
struct GridCellText;

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
    cursor_visible: bool,
    cursor_timer: Timer,
    cursor_position: usize,
}

#[derive(Clone)]
struct TextState {
    buffer: String,
    cursor_position: usize,
    cursor_visible: bool,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
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
            update_spatial_grid,
            update_boids_with_grid,
            move_boids,
            handle_text_input,
            update_fps_text,
            update_text_inputs,
            update_cursor,
            handle_button_clicks,
            update_trails,
            toggle_grid_visibility,
            draw_spatial_grid,
            cleanup_grid_text,   
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

        // Add GridPosition component during spawn
        commands.spawn((
            Boid {
                velocity,
                trail: Vec::new(),
                noise_offset_x: rng.gen_range(-1000.0..1000.0),
                noise_offset_y: rng.gen_range(-1000.0..1000.0),
                noise_offset_z: rng.gen_range(-1000.0..1000.0),
                noise_seed: rng.gen_range(0.0..100.0),
            },
            MaterialMesh2dBundle {
                mesh: triangle.clone().into(),
                material: material.clone(),
                transform: Transform::from_xyz(position.x, position.y, 0.0)
                    .with_rotation(Quat::from_rotation_z(angle - std::f32::consts::FRAC_PI_2)),
                ..default()
            },
            GridPosition {
                cell: SpatialGrid::world_to_cell(Vec2::new(position.x, position.y)),
            },
        ));
    }
}


fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<ColorMaterial>>) {
    // Initialize the camera
    commands.spawn(Camera2dBundle::default());
    
    // Initialize the spatial grid resource
    commands.insert_resource(SpatialGrid::default());
    
    // Spawn boids
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

    // Spawn new boids with grid positions
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
                    spawn_text_input(parent, "Coherence", "0.015", UIElement::CoherenceInput);      // Updated
                    spawn_text_input(parent, "Separation", "0.25", UIElement::SeparationInput);     // Updated
                    spawn_text_input(parent, "Alignment", "0.125", UIElement::AlignmentInput);      // Updated
                    spawn_text_input(parent, "Visual Range", "60.0", UIElement::VisualRangeInput); // Updated
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
                    cursor_visible: true,
                    cursor_timer: Timer::from_seconds(0.5, TimerMode::Repeating),
                    cursor_position: default_value.len(), // Start cursor at end of text
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
        for (mut input, mut text, mut bg_color, interaction, _) in text_query.iter_mut() {
            let was_focused = input.is_focused;
            input.is_focused = matches!(interaction, Interaction::Pressed);
            
            if input.is_focused != was_focused {
                // Create a separate state struct to avoid borrowing conflicts
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

                // Create a state that we can modify without borrowing conflicts
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

fn update_cursor(
    time: Res<Time>,
    mut query: Query<(&mut TextInput, &mut Text)>,
) {
    for (mut input, mut text) in query.iter_mut() {
        if input.is_focused {
            input.cursor_timer.tick(time.delta());
            
            if input.cursor_timer.just_finished() {
                input.cursor_visible = !input.cursor_visible;
                
                // Gather values before updating text
                let buffer = input.buffer.clone();
                let cursor_pos = input.cursor_position;
                let show_cursor = input.cursor_visible && input.is_focused;
                text.sections[0].value = update_text_display(&buffer, cursor_pos, show_cursor);
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

fn update_text_display(buffer: &str, cursor_position: usize, show_cursor: bool) -> String {
    if show_cursor {
        let mut display = buffer.to_string();
        display.insert(cursor_position, '|');
        display
    } else {
        buffer.to_string()
    }
}

fn update_text_state(input: &mut TextInput, text: &mut Text, state: TextState) {
    input.buffer = state.buffer.clone();
    input.cursor_position = state.cursor_position;
    input.cursor_visible = state.cursor_visible;
    
    // Update display text
    text.sections[0].value = if input.is_focused && input.cursor_visible {
        let mut display = input.buffer.clone();
        display.insert(input.cursor_position, '|');
        display
    } else {
        input.buffer.clone()
    };
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

fn update_spatial_grid(
    mut grid: ResMut<SpatialGrid>,
    query: Query<(Entity, &Transform, &GridPosition), With<Boid>>,
    mut commands: Commands,
) {
    // Clear the previous grid
    grid.cells.clear();
    
    // Update grid positions
    for (entity, transform, grid_pos) in query.iter() {
        let position = transform.translation.truncate();
        let new_cell = SpatialGrid::world_to_cell(position);
        
        // If cell has changed, update the component
        if new_cell != grid_pos.cell {
            commands.entity(entity).insert(GridPosition { cell: new_cell });
        }
        
        // Add entity to the grid
        grid.cells.entry(new_cell).or_default().push(entity);
    }
}

fn update_boids_with_grid(
    mut query: Query<(Entity, &mut Transform, &mut Boid, &GridPosition)>,
    grid: Res<SpatialGrid>,
    params: Res<SimulationParams>,
    window_query: Query<&Window>,
    time: Res<Time>,
) {
    let window = window_query.single();
    let width = window.width();
    let height = window.height();
    
    let perlin = Perlin::new(1);
    let current_time = time.elapsed_seconds() * TIME_SCALE;
    
    // Store positions and velocities for current frame
    let boids_data: HashMap<Entity, (Vec2, Vec2)> = query
        .iter()
        .map(|(entity, transform, boid, _)| {
            (entity, (transform.translation.truncate(), boid.velocity))
        })
        .collect();
    
    for (entity, mut transform, mut boid, grid_pos) in query.iter_mut() {
        let mut position = transform.translation.truncate();
        
        // Get nearby boids using spatial grid
        let nearby_entities = grid.get_nearby_entities(grid_pos.cell);
        
        let mut center_of_mass = Vec2::ZERO;
        let mut avoid_vector = Vec2::ZERO;
        let mut average_velocity = Vec2::ZERO;
        let mut num_neighbors = 0;
        
        // Process only nearby boids
        for &other_entity in &nearby_entities {
            if other_entity == entity {
                continue;
            }
            
            if let Some((other_pos, other_vel)) = boids_data.get(&other_entity) {
                let diff = *other_pos - position;
                let distance = diff.length();
                
                if distance < params.visual_range && distance > 0.0 {
                    // Coherence
                    center_of_mass += *other_pos;
                    
                    // Separation
                    if distance < params.visual_range / 2.0 {
                        avoid_vector -= diff.normalize() * (params.visual_range / (2.0 * distance.max(0.1)));
                    }
                    
                    // Alignment
                    average_velocity += *other_vel;
                    
                    num_neighbors += 1;
                }
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
        
        boid.noise_offset_x += NOISE_SCALE * time.delta_seconds();
        boid.noise_offset_y += NOISE_SCALE * time.delta_seconds();
        boid.noise_offset_z = current_time;
        
        let noise_value = perlin.get([
            boid.noise_offset_x as f64,
            boid.noise_offset_y as f64,
            boid.noise_offset_z as f64,
            boid.noise_seed as f64,
        ]) as f32;
        
        let turn_angle = noise_value * NOISE_STRENGTH * time.delta_seconds();
        let (sin, cos) = turn_angle.sin_cos();
        let slight_adjustment = Vec2::new(
            boid.velocity.x * cos - boid.velocity.y * sin,
            boid.velocity.x * sin + boid.velocity.y * cos
        ).normalize();
        
        boid.velocity = boid.velocity.lerp(slight_adjustment * boid.velocity.length(), 0.01);
        
        // Speed limiting
        let current_speed = boid.velocity.length();
        if current_speed < BOID_SPEED_LIMIT / 2.0 {
            boid.velocity = boid.velocity.normalize() * (BOID_SPEED_LIMIT / 2.0);
        } else if current_speed > BOID_SPEED_LIMIT {
            boid.velocity = boid.velocity.normalize() * BOID_SPEED_LIMIT;
        }
        
        // Position update and wrapping
        let delta_time = time.delta_seconds();
        position += boid.velocity * delta_time;
        
        position.x = (position.x + width) % width - width / 2.0;
        position.y = (position.y + height) % height - height / 2.0;
        
        transform.translation = position.extend(transform.translation.z);
        
        // Update rotation
        if boid.velocity.length_squared() > 0.0 {
            let angle = boid.velocity.y.atan2(boid.velocity.x);
            transform.rotation = Quat::from_rotation_z(angle - std::f32::consts::FRAC_PI_2);
        }
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
    // First, clean up all existing text entities that still exist
    for entity in text_query.iter() {
        commands.entity(entity).despawn_recursive();
    }

    if !debug_config.show_grid {
        return;
    }

    let window = window_query.single();
    let width = window.width();
    let height = window.height();

    // Calculate visible area in world coordinates
    let half_width = width / 2.0;
    let half_height = height / 2.0;

    // Calculate grid start and end points
    let start_cell_x = ((-half_width) / GRID_CELL_SIZE).floor() as i32;
    let end_cell_x = ((half_width) / GRID_CELL_SIZE).ceil() as i32;
    let start_cell_y = ((-half_height) / GRID_CELL_SIZE).floor() as i32;
    let end_cell_y = ((half_height) / GRID_CELL_SIZE).ceil() as i32;

    // Draw background grid
    for x in start_cell_x..=end_cell_x {
        for y in start_cell_y..=end_cell_y {
            let cell_pos = Vec2::new(
                x as f32 * GRID_CELL_SIZE,
                y as f32 * GRID_CELL_SIZE
            );
            
            // Draw faint grid for empty cells
            gizmos.rect_2d(
                cell_pos,
                0.0,
                Vec2::new(GRID_CELL_SIZE, GRID_CELL_SIZE),
                Color::srgb(0.2, 0.2, 0.2).with_alpha(0.1),
            );
        }
    }

    // Draw occupied grid cells
    for (&cell, entities) in grid.cells.iter() {
        let cell_pos = Vec2::new(
            cell.x as f32 * GRID_CELL_SIZE,
            cell.y as f32 * GRID_CELL_SIZE
        );

        // Only draw if cell is in visible area
        if cell_pos.x >= -half_width - GRID_CELL_SIZE && 
           cell_pos.x <= half_width + GRID_CELL_SIZE &&
           cell_pos.y >= -half_height - GRID_CELL_SIZE && 
           cell_pos.y <= half_height + GRID_CELL_SIZE {
            
            // Calculate cell color based on boid density
            let density = (entities.len() as f32) / (BOID_COUNT as f32);
            let base_color = Color::srgba(1.0, 0.0, 0.0, density.min(0.5));
            
            // Draw filled cell
            gizmos.rect_2d(
                cell_pos,
                0.0,
                Vec2::new(GRID_CELL_SIZE, GRID_CELL_SIZE),
                base_color,
            );

            // Draw cell outline
            gizmos.rect_2d(
                cell_pos,
                0.0,
                Vec2::new(GRID_CELL_SIZE, GRID_CELL_SIZE),
                Color::WHITE.with_alpha(0.2),
            );

            // Spawn text showing number of boids in cell
            if entities.len() > 0 {
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
        println!("Grid visibility: {}", if debug_config.show_grid { "on" } else { "off" }); // Optional debug print
    }
}

// System to cleanup old text when toggling grid visibility
fn cleanup_grid_text(
    mut commands: Commands,
    text_query: Query<Entity, With<GridCellText>>,
    debug_config: Res<DebugConfig>,
) {
    if !debug_config.show_grid {
        // Only attempt to despawn entities that still exist
        for entity in text_query.iter() {
            commands.entity(entity).despawn_recursive();
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
