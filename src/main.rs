use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    sprite::{Anchor, MaterialMesh2dBundle},
    time::{Timer, TimerMode, Fixed},
    input::{
        keyboard::KeyboardInput,
        ButtonState,
    },
    app::AppExit,
};
use clap::Command;
use noise::{NoiseFn, Perlin};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Serialize, Deserialize};
use std::{
    collections::{HashMap, VecDeque},
    error::Error,
    fs::File,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

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

// ==================== Core Components ====================

#[derive(Component, Clone, Debug, Serialize, Deserialize)]
pub struct Boid {
    #[serde(with = "vec2_serde")]
    velocity: Vec2,
    #[serde(skip)]  // Don't serialize trail for benchmarking
    trail: Vec<Vec2>,
    noise_seed: u64,
    frame_count: u64,
}

#[derive(Component)]
pub struct GridPosition {
    cell: IVec2,
}

#[derive(Component)]
pub struct GridCellText;

#[derive(Component)]
pub struct FpsText;

// =======================================================
// ==================== UI COMPONENTS ====================
// =======================================================

#[derive(Component)]
pub struct TextInput {
    is_focused: bool,
    buffer: String,
    cursor_visible: bool,
    cursor_timer: Timer,
    cursor_position: usize,
}

#[derive(Component)]
pub enum UIElement {
    CoherenceInput,
    SeparationInput,
    AlignmentInput,
    VisualRangeInput,
    ResetButton,
    TracePathsButton,
}

// ===================================================
// ==================== RESOURCES ====================
// ===================================================

#[derive(Resource)]
pub struct SpatialGrid {
    cells: HashMap<IVec2, Vec<Entity>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoidState {
    #[serde(with = "vec2_serde")]
    position: Vec2,
    #[serde(with = "vec2_serde")]
    velocity: Vec2,
}

#[derive(Resource)]
pub struct DebugConfig {
    show_grid: bool,
}

#[derive(Resource)]
pub struct SimulationRng(ChaCha8Rng);

#[derive(Resource, Clone, Serialize, Deserialize)]
pub struct SimulationParams {
    coherence: f32,
    separation: f32,
    alignment: f32,
    visual_range: f32,
    trace_paths: bool,
}

// ============================================================
// ==================== Benchmarking Types ====================
// ============================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SimulationMode {
    Interactive,
    Benchmark,
    Validate,
    Record,
}

#[derive(Resource, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub mode: SimulationMode,
    pub output_path: Option<PathBuf>,
    pub reference_path: Option<PathBuf>,
    #[serde(skip)]
    pub fixed_time: Option<Time<Fixed>>,
    #[serde(with = "duration_serde")]
    pub benchmark_duration: Option<Duration>,
    pub parameter_schedule_path: Option<PathBuf>,
    pub rng_seed: u64,
    pub noise_seed: u32,
}

#[derive(Resource)]
pub struct PerformanceMetrics {
    frame_times: VecDeque<Duration>,
    physics_times: VecDeque<Duration>,
    spatial_grid_times: VecDeque<Duration>,
    movement_times: VecDeque<Duration>,
    current_frame: MetricsFrame,
    max_samples: usize,
}

#[derive(Default)]
struct MetricsFrame {
    frame_start: Option<Instant>,
    physics_start: Option<Instant>,
    spatial_start: Option<Instant>,
    movement_start: Option<Instant>,
}

#[derive(Resource)]
pub struct DeterminismValidator {
    parameter_schedule: Vec<ParameterChange>,
    snapshots: Vec<StateSnapshot>,
    current_step: u64,
    validation_mode: ValidationMode,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ParameterChange {
    step: u64,                      // Simulation step when the change should occur
    parameter: ParameterType,
    value: f32,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum ParameterType {
    Coherence,
    Separation,
    Alignment,
    VisualRange,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    step: u64,
    boid_states: Vec<BoidState>,
    parameters: SimulationParams,
}

#[derive(Clone, PartialEq)]
pub enum ValidationMode {
    Recording,
    Validating,
    Disabled,
}

// ===============================================================
// ==================== Serialization Helpers ====================
// ===============================================================

mod vec2_serde {
    use bevy::math::Vec2;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(vec: &Vec2, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeTuple;
        let mut tup = serializer.serialize_tuple(2)?;
        tup.serialize_element(&vec.x)?;
        tup.serialize_element(&vec.y)?;
        tup.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec2, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (x, y) = <(f32, f32)>::deserialize(deserializer)?;
        Ok(Vec2::new(x, y))
    }
}

mod duration_serde {
    use serde::{self, Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match duration {
            Some(d) => serializer.serialize_some(&d.as_secs_f64()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt_secs: Option<f64> = Option::deserialize(deserializer)?;
        Ok(opt_secs.map(|secs| Duration::from_secs_f64(secs)))
    }
}

// =================================================================
// ==================== Default Implementations ====================
// =================================================================

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

impl Default for SimulationRng {
    fn default() -> Self {
        SimulationRng(ChaCha8Rng::seed_from_u64(42))
    }
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            coherence: 0.015,
            separation: 0.25,
            alignment: 0.125,
            visual_range: 60.0,
            trace_paths: false,
        }
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            mode: SimulationMode::Interactive,
            output_path: None,
            reference_path: None,
            fixed_time: Some(Time::<Fixed>::default()),
            benchmark_duration: None,
            parameter_schedule_path: None,
            rng_seed: 42,                       // Default Seed
            noise_seed: 1,                      // Default Seed
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            frame_times: VecDeque::with_capacity(1000),
            physics_times: VecDeque::with_capacity(1000),
            spatial_grid_times: VecDeque::with_capacity(1000),
            movement_times: VecDeque::with_capacity(1000),
            current_frame: MetricsFrame::default(),
            max_samples: 1000,
        }
    }
}

// ==================================================================
// ==================== Resource Implementations ====================
// ==================================================================

impl SimulationConfig {
    pub fn from_args() -> Self {
        let matches = Command::new("boids")
            .arg(clap::Arg::new("mode")
                .long("mode")
                .value_parser(["interactive", "benchmark", "validate", "record"])
                .default_value("interactive"))
            .arg(clap::Arg::new("output")
                .long("output")
                .value_name("FILE"))
            .arg(clap::Arg::new("reference")
                .long("reference")
                .value_name("FILE"))
            .arg(clap::Arg::new("duration")
                .long("duration")
                .value_name("SECONDS"))
            .arg(clap::Arg::new("parameter-schedule")
                .long("parameter-schedule")
                .value_name("FILE"))
            .arg(clap::Arg::new("rng-seed")
                .long("rng-seed")
                .value_name("SEED")
                .value_parser(clap::value_parser!(u64))
                .default_value("42"))
            .arg(clap::Arg::new("noise-seed")
                .long("noise-seed")
                .value_name("SEED")
                .value_parser(clap::value_parser!(u32))
                .default_value("1"))
            .get_matches();

        let mode = match matches.get_one::<String>("mode").unwrap().as_str() {
            "interactive" => SimulationMode::Interactive,
            "benchmark" => SimulationMode::Benchmark,
            "validate" => SimulationMode::Validate,
            "record" => SimulationMode::Record,
            _ => SimulationMode::Interactive,
        };

        Self {
            mode,
            output_path: matches.get_one::<String>("output").map(PathBuf::from),
            reference_path: matches.get_one::<String>("reference").map(PathBuf::from),
            fixed_time: Some(Time::<Fixed>::default()),
            benchmark_duration: matches.get_one::<String>("duration")
                .and_then(|d| d.parse::<u64>().ok())
                .map(|secs| Duration::from_secs(secs)),
            parameter_schedule_path: matches.get_one::<String>("parameter-schedule").map(PathBuf::from),
            rng_seed: *matches.get_one::<u64>("rng-seed").unwrap(),
            noise_seed: *matches.get_one::<u32>("noise-seed").unwrap(),
        }
    }
}

impl SimulationRng {
    pub fn new(seed: u64) -> Self {
        SimulationRng(ChaCha8Rng::seed_from_u64(seed))
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

impl DeterminismValidator {
    pub fn new(config: &SimulationConfig) -> Self {
        let validation_mode = match config.mode {
            SimulationMode::Record => ValidationMode::Recording,
            SimulationMode::Validate => ValidationMode::Validating,
            _ => ValidationMode::Disabled,
        };

        let parameter_schedule = if let Some(path) = &config.parameter_schedule_path {
            Self::load_parameter_schedule(path).unwrap_or_default()
        } else {
            Vec::new()
        };

        Self {
            parameter_schedule,
            snapshots: Vec::new(),
            current_step: 0,
            validation_mode,
        }
    }

    pub fn save_reference(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &self.snapshots)?;
        Ok(())
    }

    pub fn load_reference(&mut self, path: &Path) -> Result<(), Box<dyn Error>> {
        let file = File::open(path)?;
        self.snapshots = serde_json::from_reader(file)?;
        Ok(())
    }

    fn load_parameter_schedule(path: &Path) -> Result<Vec<ParameterChange>, Box<dyn Error>> {
        let file = File::open(path)?;
        let schedule: Vec<ParameterChange> = serde_json::from_reader(file)?;
        Ok(schedule)
    }

    pub fn validate_state(&self, step: u64, current_states: &[BoidState], params: &SimulationParams) -> bool {
        if let Some(snapshot) = self.snapshots.iter().find(|s| s.step == step) {
            // Validates number of boids matches
            if snapshot.boid_states.len() != current_states.len() {
                return false;
            }
    
            // Validates parameters match
            if !self.params_match(&snapshot.parameters, params) {
                return false;
            }
    
            // Validates each boid's state
            for (recorded, current) in snapshot.boid_states.iter().zip(current_states.iter()) {
                if !self.states_match(recorded, current) {
                    return false;
                }
            }
            true
        } else {
            true // No snapshot to compare against
        }
    }

    fn params_match(&self, recorded: &SimulationParams, current: &SimulationParams) -> bool {
        const EPSILON: f32 = 1e-6;
        (recorded.coherence - current.coherence).abs() < EPSILON &&
        (recorded.separation - current.separation).abs() < EPSILON &&
        (recorded.alignment - current.alignment).abs() < EPSILON &&
        (recorded.visual_range - current.visual_range).abs() < EPSILON
    }

    fn states_match(&self, recorded: &BoidState, current: &BoidState) -> bool {
        const EPSILON: f32 = 1e-6;
        (recorded.position - current.position).length_squared() < EPSILON &&
        (recorded.velocity - current.velocity).length_squared() < EPSILON
    }
}

impl PerformanceMetrics {
    pub fn begin_frame(&mut self) {
        self.current_frame.frame_start = Some(Instant::now());
    }

    pub fn end_frame(&mut self) {
        if let Some(start) = self.current_frame.frame_start.take() {
            let duration = start.elapsed();
            if self.frame_times.len() >= self.max_samples {
                self.frame_times.pop_front();
            }
            self.frame_times.push_back(duration);
        }
    }

    pub fn begin_physics(&mut self) {
        self.current_frame.physics_start = Some(Instant::now());
    }

    pub fn end_physics(&mut self) {
        if let Some(start) = self.current_frame.physics_start.take() {
            let duration = start.elapsed();
            if self.physics_times.len() >= self.max_samples {
                self.physics_times.pop_front();
            }
            self.physics_times.push_back(duration);
        }
    }

    pub fn begin_spatial(&mut self) {
        self.current_frame.spatial_start = Some(Instant::now());
    }

    pub fn end_spatial(&mut self) {
        if let Some(start) = self.current_frame.spatial_start.take() {
            let duration = start.elapsed();
            if self.spatial_grid_times.len() >= self.max_samples {
                self.spatial_grid_times.pop_front();
            }
            self.spatial_grid_times.push_back(duration);
        }
    }

    pub fn begin_movement(&mut self) {
        self.current_frame.movement_start = Some(Instant::now());
    }

    pub fn end_movement(&mut self) {
        if let Some(start) = self.current_frame.movement_start.take() {
            let duration = start.elapsed();
            if self.movement_times.len() >= self.max_samples {
                self.movement_times.pop_front();
            }
            self.movement_times.push_back(duration);
        }
    }

    pub fn save_benchmark_results(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        #[derive(Serialize)]
        struct BenchmarkResults {
            frame_times: Vec<f64>,
            physics_times: Vec<f64>,
            spatial_times: Vec<f64>,
            movement_times: Vec<f64>,
        }

        let results = BenchmarkResults {
            frame_times: self.frame_times.iter().map(|d| d.as_secs_f64()).collect(),
            physics_times: self.physics_times.iter().map(|d| d.as_secs_f64()).collect(),
            spatial_times: self.spatial_grid_times.iter().map(|d| d.as_secs_f64()).collect(),
            movement_times: self.movement_times.iter().map(|d| d.as_secs_f64()).collect(),
        };

        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &results)?;
        Ok(())
    }
}

// =======================================================
// ==================== SETUP SYSTEMS ====================
// =======================================================

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

fn apply_parameter_schedule(
    validator: Res<DeterminismValidator>,
    mut sim_params: ResMut<SimulationParams>,
) {
    // Only apply parameter changes in benchmark or validation modes
    if validator.validation_mode == ValidationMode::Disabled {
        return;
    }

    let current_step = validator.current_step;
    
    for change in &validator.parameter_schedule {
        if change.step == current_step {
            // Store old value for logging
            let old_value = match change.parameter {
                ParameterType::Coherence => sim_params.coherence,
                ParameterType::Separation => sim_params.separation,
                ParameterType::Alignment => sim_params.alignment,
                ParameterType::VisualRange => sim_params.visual_range,
            };
            
            // Apply change
            match change.parameter {
                ParameterType::Coherence => sim_params.coherence = change.value,
                ParameterType::Separation => sim_params.separation = change.value,
                ParameterType::Alignment => sim_params.alignment = change.value,
                ParameterType::VisualRange => sim_params.visual_range = change.value,
            }
            
            // Log the change with before/after values
            info!("Step {}: Parameter {:?} changed from {} to {}", 
                  current_step, change.parameter, old_value, change.value);
            
            // Log current state of all parameters
            info!("Current parameters: Coherence={}, Separation={}, Alignment={}, VisualRange={}", 
                  sim_params.coherence, sim_params.separation, sim_params.alignment, sim_params.visual_range);
        }
    }
}

// ==============================================================
// ==================== CORE PHYSICS SYSTEMS ====================
// ==============================================================

fn fixed_timestep_physics(
    fixed_time: Res<Time<Fixed>>,
    mut metrics: ResMut<PerformanceMetrics>,
    mut validator: ResMut<DeterminismValidator>,
    mut query: Query<(Entity, &mut Transform, &mut Boid, &GridPosition)>,
    grid: Res<SpatialGrid>,
    params: Res<SimulationParams>,
    window_query: Query<&Window>,
    config: Res<SimulationConfig>,
) {
    metrics.begin_physics();
    
    let dt = fixed_time.delta_seconds();
    let validation_mode = validator.validation_mode.clone();
    let current_step = validator.current_step;
    
    update_physics_step(
        dt,
        &mut validator,
        &mut query,
        &grid,
        &params,
        &window_query,
        &mut metrics,
        &config,
    );

    metrics.end_physics();

    // Handle validation and recording after physics update
    let current_states: Vec<BoidState> = query.iter().map(|(_, transform, boid, _)| {
        BoidState {
            position: transform.translation.truncate(),
            velocity: boid.velocity,
        }
    }).collect();

    match validation_mode {
        ValidationMode::Validating => {
            if !validator.validate_state(current_step, &current_states, &params) {
                error!("State validation failed at step {}", current_step);
            }
            validator.current_step += 1;
        },
        ValidationMode::Recording => {
            // Only record every 1000th frame to keep file size manageable
            if current_step % 1000 == 0 {
                validator.snapshots.push(StateSnapshot {
                    step: current_step,
                    boid_states: current_states,
                    parameters: params.clone(),
                });
            }
            validator.current_step += 1;
        },
        ValidationMode::Disabled => {}
    }
}

fn update_physics_step(
    dt: f32,
    validator: &mut DeterminismValidator,
    query: &mut Query<(Entity, &mut Transform, &mut Boid, &GridPosition)>,
    grid: &SpatialGrid,
    params: &SimulationParams,
    window_query: &Query<&Window>,
    metrics: &mut PerformanceMetrics,
    config: &SimulationConfig,
) {
    let window = window_query.single();
    let width = window.width();
    let height = window.height();
    
    let perlin = Perlin::new(config.noise_seed);
    
    // Cache current positions and velocities
    let boids_data: HashMap<Entity, (Vec2, Vec2)> = query
        .iter()
        .map(|(entity, transform, boid, _)| {
            (entity, (transform.translation.truncate(), boid.velocity))
        })
        .collect();
    
    // Start timing movement calculations
    metrics.begin_movement();
    
    for (entity, mut transform, mut boid, grid_pos) in query.iter_mut() {
        let current_position = transform.translation.truncate();
        let nearby_entities = grid.get_nearby_entities(grid_pos.cell);
        
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
        let turn_angle = noise_value * NOISE_STRENGTH * dt * NOISE_INFLUENCE;
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
        
        // Update position with dt
        let new_position = current_position + boid.velocity * dt;
        
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

        // Update trail if enabled
        if params.trace_paths {
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

    metrics.end_movement();

    // State validation for benchmarking modes
    if validator.validation_mode == ValidationMode::Validating {
        let current_states: Vec<BoidState> = query.iter().map(|(_, transform, boid, _)| {
            BoidState {
                position: transform.translation.truncate(),
                velocity: boid.velocity,
            }
        }).collect();

        if !validator.validate_state(validator.current_step, &current_states, params) {
            error!("State validation failed at step {}", validator.current_step);
        }
        
        // Increment the step counter
        validator.current_step += 1;
    }
}

fn get_deterministic_noise(perlin: &Perlin, boid: &Boid, frame: u64) -> f32 {
    let noise1 = perlin.get([
        (frame as f64) * NOISE_SCALE,
        (boid.noise_seed as f64) * 0.1,
        0.0,
        0.0,
    ]) as f32;
    
    let noise2 = perlin.get([
        (boid.noise_seed as f64) * 0.1,
        (frame as f64) * NOISE_SCALE,
        0.0,
        0.0,
    ]) as f32;
    
    (noise1 + noise2) * 0.5
}

fn update_spatial_grid(
    mut grid: ResMut<SpatialGrid>,
    query: Query<(Entity, &Transform, &GridPosition), With<Boid>>,
    mut commands: Commands,
    mut metrics: ResMut<PerformanceMetrics>,
) {
    metrics.begin_spatial();
    
    grid.cells.clear();
    
    for (entity, transform, grid_pos) in query.iter() {
        let position = transform.translation.truncate();
        let new_cell = SpatialGrid::world_to_cell(position);
        
        if new_cell != grid_pos.cell {
            commands.entity(entity).insert(GridPosition { cell: new_cell });
        }
        
        grid.cells.entry(new_cell).or_default().push(entity);
    }

    metrics.end_spatial();
}

fn handle_non_interactive_modes(
    config: Res<SimulationConfig>,
    metrics: Res<PerformanceMetrics>,
    validator: Res<DeterminismValidator>,
    time: Res<Time>,
    mut app_exit: EventWriter<AppExit>,
) {
    match config.mode {
        SimulationMode::Interactive => return,
        SimulationMode::Benchmark => {
            if let Some(duration) = config.benchmark_duration {
                if time.elapsed() >= duration {
                    if let Some(path) = &config.output_path {
                        metrics.save_benchmark_results(path)
                            .expect("Failed to save benchmark results");
                    }
                    app_exit.send(AppExit::Success);
                }
            }
        }
        SimulationMode::Record => {
            if let Some(duration) = config.benchmark_duration {
                if time.elapsed() >= duration {
                    if let Some(path) = &config.output_path {
                        validator.save_reference(path)
                            .expect("Failed to save reference data");
                    }
                    app_exit.send(AppExit::Success);
                }
            }
        }
        SimulationMode::Validate => {
            if let Some(duration) = config.benchmark_duration {
                if time.elapsed() >= duration {
                    app_exit.send(AppExit::Success);
                }
            }
        }
    }
}

// ====================================================
// ==================== UI SYSTEMS ====================
// ====================================================

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
                update_text_state_safe(&mut input, &mut text, state);
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
                    KeyCode::Enter | KeyCode::NumpadEnter => {
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
                    update_text_state_safe(&mut input, &mut text, state);
                }
            }
        }
    }
}

#[derive(Clone)]
struct TextState {
    buffer: String,
    cursor_position: usize,
    cursor_visible: bool,
}

fn update_text_state_safe(input: &mut TextInput, text: &mut Text, state: TextState) {
    input.buffer = state.buffer.clone();
    input.cursor_position = state.cursor_position;
    input.cursor_visible = state.cursor_visible;
    
    let display = if input.is_focused && input.cursor_visible {
        let mut display = input.buffer.clone();
        display.insert(input.cursor_position, '|');
        display
    } else {
        input.buffer.clone()
    };
    
    text.sections[0].value = display;
}

fn update_cursor(
    time: Res<Time>,
    mut query: Query<(&mut TextInput, &mut Text)>,
) {
    for (mut input, mut text) in query.iter_mut() {
        if input.is_focused {
            input.cursor_timer.tick(time.delta());
            
            if input.cursor_timer.just_finished() {
                let state = TextState {
                    buffer: input.buffer.clone(),
                    cursor_position: input.cursor_position,
                    cursor_visible: !input.cursor_visible,
                };
                update_text_state_safe(&mut input, &mut text, state);
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

fn handle_button_clicks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut rng: ResMut<SimulationRng>,
    interaction_query: Query<(&Interaction, &UIElement, &Children), (Changed<Interaction>, With<UIElement>)>,
    mut sim_params: ResMut<SimulationParams>,
    boid_query: Query<Entity, With<Boid>>,
    grid_text_query: Query<Entity, With<GridCellText>>,
    mut text_query: Query<&mut Text>,
) {
    for (interaction, ui_element, children) in interaction_query.iter() {
        if let Interaction::Pressed = *interaction {
            match ui_element {
                UIElement::ResetButton => {
                    // Clean up grid text entities
                    for entity in grid_text_query.iter() {
                        commands.entity(entity).despawn_recursive();
                    }
                    
                    // Clean up boids
                    for entity in boid_query.iter() {
                        commands.entity(entity).despawn_recursive();
                    }
                    
                    // Spawn new boids
                    spawn_boids(&mut commands, &mut meshes, &mut materials, &mut rng);
                }
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

// ===============================================================
// ==================== VISUALISATION SYSTEMS ====================
// ===============================================================

fn draw_spatial_grid(
    mut commands: Commands,
    mut gizmos: Gizmos,
    grid: Res<SpatialGrid>,
    debug_config: Res<DebugConfig>,
    window_query: Query<&Window>,
    text_query: Query<Entity, With<GridCellText>>,
) {
    // Clean up existing grid text entities
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
                Color::srgba(0.2, 0.2, 0.2, 0.1),
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
                Color::srgba(1.0, 1.0, 1.0, 0.2),
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

fn toggle_grid_visibility(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut debug_config: ResMut<DebugConfig>,
) {
    if keyboard.just_pressed(KeyCode::KeyG) {
        debug_config.show_grid = !debug_config.show_grid;
    }
}

fn cleanup_trails_on_toggle(
    mut query: Query<&mut Boid>,
    sim_params: Res<SimulationParams>,
    mut prev_trace_paths: Local<bool>,
) {
    if *prev_trace_paths != sim_params.trace_paths {
        if !sim_params.trace_paths {
            for mut boid in query.iter_mut() {
                boid.trail.clear();
            }
        }
        *prev_trace_paths = sim_params.trace_paths;
    }
}

// ================================================================
// ==================== MAIN APP CONFIGURATION ====================
// ================================================================

fn main() {
    let config = SimulationConfig::from_args();
    
    let mut app = App::new();
    
    // Basic setup for all modes
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Boids Simulation".to_string(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(config.clone())
        .insert_resource(Time::<Fixed>::default())
        .insert_resource(PerformanceMetrics::default())
        .insert_resource(DeterminismValidator::new(&config))
        .insert_resource(SimulationRng::new(config.rng_seed))
        .insert_resource(SimulationParams::default())
        .insert_resource(SpatialGrid::default())
        .insert_resource(DebugConfig::default())
        .insert_resource(Time::<Fixed>::from_seconds(FIXED_TIME_STEP.into()));

    // Common systems for all modes
    app.add_systems(Startup, setup)
        .add_systems(FixedUpdate, (
            apply_parameter_schedule,
            fixed_timestep_physics,
            update_spatial_grid,
        ));

    // Only add interactive elements in interactive mode
    if matches!(config.mode, SimulationMode::Interactive) {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Startup, setup_ui)
            .add_systems(Update, (
                // UI Systems
                handle_text_input,
                update_fps_text,
                update_cursor,
                handle_button_clicks,
                
                // Visualization Systems
                draw_spatial_grid,
                toggle_grid_visibility,
                update_trails,
                cleanup_trails_on_toggle,
            ).chain());
    }

    // Add benchmark/validation systems for non-interactive modes
    if !matches!(config.mode, SimulationMode::Interactive) {
        app.add_systems(Update, (
            handle_non_interactive_modes,
            apply_parameter_schedule,
        ));
    }

    app.run();
}