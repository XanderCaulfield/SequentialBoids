use bevy::{
    app::AppExit,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    sprite::{Anchor, MaterialMesh2dBundle},
    time::{Timer, TimerMode, Fixed},
    input::{
        keyboard::KeyboardInput,
        ButtonState,
    },
    
};
use clap::Command;
use noise::{NoiseFn, Perlin};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Serialize, Deserialize};
use std::{
    collections::{BTreeMap, VecDeque},
    error::Error,
    fs::File,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

// ==================== Constants ====================

const BOID_COUNT: usize = 200;
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
    id: u64,
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

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
struct GridKey(i32, i32);

impl From<IVec2> for GridKey {
    fn from(vec: IVec2) -> Self {
        GridKey(vec.x, vec.y)
    }
}

impl From<GridKey> for IVec2 {
    fn from(key: GridKey) -> Self {
        IVec2::new(key.0, key.1)
    }
}

impl GridKey {
    fn x(&self) -> i32 { self.0 }
    fn y(&self) -> i32 { self.1 }
}

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

/// Spatial partitioning grid for efficient neighbor lookups
/// Uses a BTreeMap to maintain deterministic ordering of cells
#[derive(Resource)]
pub struct SpatialGrid {
    cells: BTreeMap<GridKey, Vec<(Entity, u64)>>,    // Store (Entity, BoidID) pairs
}

/// Represents the current state of a boid for validation purposes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoidState {
    #[serde(with = "vec2_serde")]
    position: Vec2,
    #[serde(with = "vec2_serde")]
    velocity: Vec2,
}

/// Configuration for debug visualization features
#[derive(Resource)]
pub struct DebugConfig {
    show_grid: bool,
}

// Used to prevent overlapping logging in console output
#[derive(Resource)]
struct LoggingState {
    is_logging: bool,
}

/// Deterministic random number generator using ChaCha8
/// This ensures consistent behavior across runs with the same seed
#[derive(Resource)]
pub struct SimulationRng(ChaCha8Rng);

/// Core simulation parameters that control boid behavior
#[derive(Resource, Clone, Serialize, Deserialize)]
pub struct SimulationParams {
    coherence: f32,      // How strongly boids are attracted to the center of mass
    separation: f32,     // How strongly boids avoid each other
    alignment: f32,      // How strongly boids align their velocity with neighbors
    visual_range: f32,   // How far boids can see
    trace_paths: bool,   // Whether to show movement trails
}

/// Defines the different modes the simulation can run in
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SimulationMode {
    Interactive,    // Normal interactive mode with UI
    Benchmark,      // Performance testing mode
    Validate,       // Validation against reference data
    Record,         // Record reference data
}

/// Main configuration for the simulation, including runtime parameters
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

/// Tracks performance metrics for benchmarking and analysis
#[derive(Resource)]
pub struct PerformanceMetrics {
    frame_times: VecDeque<Duration>,      // Track complete frame durations
    physics_times: VecDeque<Duration>,    // Track physics step durations
    spatial_grid_times: VecDeque<Duration>, // Track spatial partitioning durations
    movement_times: VecDeque<Duration>,   // Track boid movement calculation durations
    current_frame: MetricsFrame,
    max_samples: usize,
}

/// Helper struct for tracking timing within a single frame
#[derive(Default)]
struct MetricsFrame {
    frame_start: Option<Instant>,
    physics_start: Option<Instant>,
    spatial_start: Option<Instant>,
    movement_start: Option<Instant>,
}

/// Handles validation of simulation determinism
#[derive(Resource)]
pub struct DeterminismValidator {
    parameter_schedule: Vec<ParameterChange>,
    snapshots: Vec<StateSnapshot>,
    current_step: u64,
    validation_mode: ValidationMode,
}

/// Represents a scheduled change in simulation parameters
#[derive(Clone, Serialize, Deserialize)]
pub struct ParameterChange {
    step: u64,                      // Simulation step when the change should occur
    parameter: ParameterType,
    value: f32,
}

/// Enum for different parameter types that can be changed
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum ParameterType {
    Coherence,
    Separation,
    Alignment,
    VisualRange,
}

/// Snapshot of simulation state for validation
#[derive(Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    step: u64,
    boid_states: Vec<BoidState>,
    parameters: SimulationParams,
}

/// Current validation mode of the simulation
#[derive(Clone, PartialEq)]
pub enum ValidationMode {
    Recording,    // Recording reference data
    Validating,   // Validating against reference data
    Disabled,     // Validation disabled
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
    enum TimingSet {
        FrameTiming,
    }

// ===============================================================
// ==================== Serialization Helpers ====================
// ===============================================================

/// Serialization helpers for Vec2 type
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

/// Serialization helpers for Duration type
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
// ================== Determinism Validation =======================
// =================================================================

impl DeterminismValidator {
    pub fn from_config(config: &SimulationConfig) -> Self {
        let validation_mode = match config.mode {
            SimulationMode::Validate => ValidationMode::Validating,
            SimulationMode::Record => ValidationMode::Recording,
            _ => ValidationMode::Disabled,
        };

        let parameter_schedule = if let Some(ref path) = config.parameter_schedule_path {
            // First check if the file exists
            if !path.exists() {
                error!("Parameter schedule file not found: {}", path.display());
                Vec::new()
            } else {
                Self::load_parameter_schedule(path)
                    .unwrap_or_else(|e| {
                        error!("Failed to load parameter schedule: {}", e);
                        Vec::new()
                    })
            }
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

    pub fn validate_state(&self, step: u64, current_states: &[BoidState], params: &SimulationParams) -> bool {
        if let Some(snapshot) = self.snapshots.iter().find(|s| s.step == step) {
            let mut all_valid = true;
            
            // 1. Validate parameters
            if !self.validate_parameters(&snapshot.parameters, params) {
                error!("Parameter mismatch at step {}:", step);
                error!("  Expected: Coherence={}, Separation={}, Alignment={}, VisualRange={}", 
                    snapshot.parameters.coherence,
                    snapshot.parameters.separation,
                    snapshot.parameters.alignment,
                    snapshot.parameters.visual_range);
                error!("  Actual:   Coherence={}, Separation={}, Alignment={}, VisualRange={}", 
                    params.coherence,
                    params.separation,
                    params.alignment,
                    params.visual_range);
                all_valid = false;
            }

            // 2. Validate boid count
            if snapshot.boid_states.len() != current_states.len() {
                error!("Boid count mismatch at step {}", step);
                error!("  Expected: {} boids", snapshot.boid_states.len());
                error!("  Actual:   {} boids", current_states.len());
                all_valid = false;
            }

            // 3. Validate individual boid states
            for (i, (expected, actual)) in snapshot.boid_states.iter().zip(current_states.iter()).enumerate() {
                if !self.validate_boid_state(expected, actual) {
                    error!("Boid {} state mismatch at step {}:", i, step);
                    error!("  Expected: pos={:?}, vel={:?}", expected.position, expected.velocity);
                    error!("  Actual:   pos={:?}, vel={:?}", actual.position, actual.velocity);
                    all_valid = false;
                    // Break after first few mismatches to avoid console spam
                    if i > 5 {
                        error!("... and more mismatches (truncated)");
                        break;
                    }
                }
            }

            if all_valid {
                // Only log validation success periodically to avoid spam
                if step % 1000 == 0 {
                    info!("✓ Validation passed for step {}", step);
                }
            }

            all_valid
        } else {
            // No reference data for this step is fine
            true
        }
    }

    fn validate_parameters(&self, expected: &SimulationParams, actual: &SimulationParams) -> bool {
        const EPSILON: f32 = 1e-6;
        
        (expected.coherence - actual.coherence).abs() < EPSILON &&
        (expected.separation - actual.separation).abs() < EPSILON &&
        (expected.alignment - actual.alignment).abs() < EPSILON &&
        (expected.visual_range - actual.visual_range).abs() < EPSILON
    }

    fn validate_boid_state(&self, expected: &BoidState, actual: &BoidState) -> bool {
        const EPSILON: f32 = 1e-5;
        
        (expected.position - actual.position).length() < EPSILON &&
        (expected.velocity - actual.velocity).length() < EPSILON
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
        let file = File::open(path).map_err(|e| {
            format!("Failed to open parameter schedule file '{}': {}", 
                path.display(), e)
        })?;
        
        serde_json::from_reader(file).map_err(|e| {
            format!("Failed to parse parameter schedule from '{}': {}", 
                path.display(), e).into()
        })
    }
}

// =================================================================
// ================== Performance Metrics ==========================
// =================================================================

fn track_frame_times(
    mut metrics: ResMut<PerformanceMetrics>,
) {
    metrics.end_frame();  // End previous frame
    metrics.begin_frame(); // Start new frame
}

impl PerformanceMetrics {

    pub fn begin_frame(&mut self) {
        self.current_frame.frame_start = Some(Instant::now());
    }

    pub fn end_frame(&mut self) {
        if let Some(start) = self.current_frame.frame_start {
            self.frame_times.push_back(start.elapsed());
            if self.frame_times.len() > self.max_samples {
                self.frame_times.pop_front();
            }
        }
    }

    pub fn begin_physics(&mut self) {
        self.current_frame.frame_start = Some(Instant::now());
        self.current_frame.physics_start = Some(Instant::now());
    }

    pub fn end_physics(&mut self) {
        if let Some(start) = self.current_frame.physics_start {
            self.physics_times.push_back(start.elapsed());
            if self.physics_times.len() > self.max_samples {
                self.physics_times.pop_front();
            }
        }
    }

    pub fn begin_movement(&mut self) {
        self.current_frame.movement_start = Some(Instant::now());
    }

    pub fn end_movement(&mut self) {
        if let Some(start) = self.current_frame.movement_start {
            self.movement_times.push_back(start.elapsed());
            if self.movement_times.len() > self.max_samples {
                self.movement_times.pop_front();
            }
        }
    }

    pub fn begin_spatial(&mut self) {
        self.current_frame.spatial_start = Some(Instant::now());
    }

    pub fn end_spatial(&mut self) {
        if let Some(start) = self.current_frame.spatial_start {
            self.spatial_grid_times.push_back(start.elapsed());
            if self.spatial_grid_times.len() > self.max_samples {
                self.spatial_grid_times.pop_front();
            }
        }
    }

    pub fn save_benchmark_results(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let mut file = File::create(path)?;
        
        let results = serde_json::json!({
            "frame_times": self.frame_times.iter()
                .map(|d| d.as_secs_f64() * 1000.0)
                .collect::<Vec<f64>>(),
            "physics_times": self.physics_times.iter()
                .map(|d| d.as_secs_f64() * 1000.0)
                .collect::<Vec<f64>>(),
            "spatial_times": self.spatial_grid_times.iter()
                .map(|d| d.as_secs_f64() * 1000.0)
                .collect::<Vec<f64>>(),
            "movement_times": self.movement_times.iter()
                .map(|d| d.as_secs_f64() * 1000.0)
                .collect::<Vec<f64>>(),
            "statistics": {
                "frame_time_avg": self.frame_times.iter()
                    .map(|d| d.as_secs_f64() * 1000.0)
                    .sum::<f64>() / self.frame_times.len() as f64,
                "physics_time_avg": self.physics_times.iter()
                    .map(|d| d.as_secs_f64() * 1000.0)
                    .sum::<f64>() / self.physics_times.len() as f64,
                "spatial_time_avg": self.spatial_grid_times.iter()
                    .map(|d| d.as_secs_f64() * 1000.0)
                    .sum::<f64>() / self.spatial_grid_times.len() as f64,
                "movement_time_avg": self.movement_times.iter()
                    .map(|d| d.as_secs_f64() * 1000.0)
                    .sum::<f64>() / self.movement_times.len() as f64,
            }
        });

        serde_json::to_writer_pretty(&mut file, &results)?;
        Ok(())
    }
}

// =================================================================
// ==================== Default Implementations ====================
// =================================================================

impl Default for Boid {
    fn default() -> Self {
        Self {
            id: 0,
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
            cells: BTreeMap::new(),
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

impl Default for LoggingState {
    fn default() -> Self {
        Self {
            is_logging: false,
        }
    }
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            coherence: 0.015,    // Default attraction strength
            separation: 0.25,     // Default separation strength
            alignment: 0.125,     // Default alignment strength
            visual_range: 60.0,   // Default vision range
            trace_paths: false,   // Trails disabled by default
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
            rng_seed: 42,        // Default RNG seed
            noise_seed: 1,       // Default noise seed
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

impl SpatialGrid {
    fn world_to_cell(position: Vec2) -> IVec2 {
        IVec2::new(
            (position.x / GRID_CELL_SIZE).floor() as i32,
            (position.y / GRID_CELL_SIZE).floor() as i32,
        )
    }

    // Method to calculate if cells are in range based on visual range
    fn is_in_range(cell1: IVec2, cell2: IVec2, visual_range: f32) -> bool {
        let diff = cell2 - cell1;
        // Calculate how many cells correspond to the visual range
        // Add 1 to account for partial cells and use ceiling to ensure we don't miss any potential neighbors
        let max_cell_distance = (visual_range / GRID_CELL_SIZE).ceil() as i32;
        diff.x.abs() <= max_cell_distance && diff.y.abs() <= max_cell_distance
    }
}

impl SimulationRng {
    pub fn new(seed: u64) -> Self {
        Self(ChaCha8Rng::seed_from_u64(seed))
    }
}

impl SimulationConfig {
    pub fn from_args() -> Self {
        let matches = Command::new("Boids Simulation")
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
            .arg(clap::Arg::new("params")
                .long("params")
                .value_name("FILE"))
            .arg(clap::Arg::new("seed")
                .long("seed")
                .value_name("NUMBER")
                .default_value("42"))
            .get_matches();

            let parameter_schedule_path = matches.get_one::<String>("params")
            .map(|p| {
                let path = PathBuf::from(p);
                if !path.exists() {
                    warn!("Parameter schedule file not found: {}", path.display());
                }
                path
            });


        let mode = match matches.get_one::<String>("mode").map(|s| s.as_str()) {
            Some("benchmark") => SimulationMode::Benchmark,
            Some("validate") => SimulationMode::Validate,
            Some("record") => SimulationMode::Record,
            _ => SimulationMode::Interactive,
        };

        Self {
            mode,
            output_path: matches.get_one::<String>("output").map(PathBuf::from),
            reference_path: matches.get_one::<String>("reference").map(PathBuf::from),
            fixed_time: Some(Time::<Fixed>::default()),
            benchmark_duration: matches.get_one::<String>("duration")
                .and_then(|s| s.parse::<f64>().ok())
                .map(Duration::from_secs_f64),
            parameter_schedule_path,
            rng_seed: matches.get_one::<String>("seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(42),
            noise_seed: 1,
        }
    }
}

// =======================================================
// ==================== SETUP SYSTEMS ====================
// =======================================================

/// Initial setup of the simulation environment
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

/// Spawns initial boid entities with deterministic positions and velocities
fn spawn_boids(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    rng: &mut ResMut<SimulationRng>,
) {
    let triangle = meshes.add(Mesh::from(RegularPolygon::new(5.0, 3)));
    let material = materials.add(ColorMaterial::from(Color::srgb(0.33, 0.55, 0.95)));

    for i in 0..BOID_COUNT {
        // Generate deterministic initial conditions using the seeded RNG
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
                id: i as u64,
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

/// Applies scheduled parameter changes at appropriate simulation steps
fn apply_parameter_schedule(
    validator: Res<DeterminismValidator>,
    mut sim_params: ResMut<SimulationParams>,
    mut logging_state: ResMut<LoggingState>,
) {
    if validator.validation_mode == ValidationMode::Disabled {
        return;
    }

    // Skip if we're already logging
    if logging_state.is_logging {
        return;
    }

    let current_step = validator.current_step;
    let mut change_applied = false;
    
    for change in &validator.parameter_schedule {
        if change.step == current_step && !change_applied {
            // Store old value for logging
            let old_value = match change.parameter {
                ParameterType::Coherence => sim_params.coherence,
                ParameterType::Separation => sim_params.separation,
                ParameterType::Alignment => sim_params.alignment,
                ParameterType::VisualRange => sim_params.visual_range,
            };
            
            // Only apply and log if there's an actual change
            if (match change.parameter {
                ParameterType::Coherence => sim_params.coherence,
                ParameterType::Separation => sim_params.separation,
                ParameterType::Alignment => sim_params.alignment,
                ParameterType::VisualRange => sim_params.visual_range,
            } - change.value).abs() > f32::EPSILON {
                // Apply change
                match change.parameter {
                    ParameterType::Coherence => sim_params.coherence = change.value,
                    ParameterType::Separation => sim_params.separation = change.value,
                    ParameterType::Alignment => sim_params.alignment = change.value,
                    ParameterType::VisualRange => sim_params.visual_range = change.value,
                }
                
                logging_state.is_logging = true;

                // Make parameter changes very visible in console
                info!("╔════════════════════════════════════════════════════════════");
                info!("║ PARAMETER CHANGE at step {}", current_step);
                info!("║ {:?}: {:.3} -> {:.3}", change.parameter, old_value, change.value);
                info!("║ Current parameters:");
                info!("║   Coherence:    {:.3}", sim_params.coherence);
                info!("║   Separation:   {:.3}", sim_params.separation);
                info!("║   Alignment:    {:.3}", sim_params.alignment);
                info!("║   VisualRange:  {:.3}", sim_params.visual_range);
                info!("╚════════════════════════════════════════════════════════════");
                
                logging_state.is_logging = false;
                change_applied = true;
            }
        }
    }
}
// ==============================================================
// ==================== CORE PHYSICS SYSTEMS ====================
// ==============================================================

/// Main physics system that runs at a fixed timestep
/// This is crucial for deterministic behavior as it ensures consistent simulation steps
/// regardless of frame rate
fn fixed_timestep_physics(
    fixed_time: Res<Time<Fixed>>,
    mut metrics: ResMut<PerformanceMetrics>,
    mut validator: ResMut<DeterminismValidator>,
    mut query: Query<(Entity, &mut Transform, &mut Boid, &GridPosition)>,
    grid: Res<SpatialGrid>,
    params: Res<SimulationParams>,
    config: Res<SimulationConfig>,
    window_query: Query<&Window>,
) {
    metrics.begin_physics();
    
    let dt = fixed_time.delta_seconds();
    let validation_mode = validator.validation_mode.clone();
    let current_step = validator.current_step;
    
    let window = window_query.single();
    let width = window.width();
    let height = window.height();
    
    // 1. Collect all current states in a deterministic order
    let mut boid_states: Vec<(Entity, BoidState, &Boid, &GridPosition)> = query
        .iter()
        .map(|(entity, transform, boid, grid_pos)| {
            (entity, 
             BoidState {
                position: transform.translation.truncate(),
                velocity: boid.velocity,
             },
             boid,
             grid_pos)
        })
        .collect();
    
    // 2. Sort by entity ID for deterministic ordering
    boid_states.sort_by_key(|(_, _, boid, _)| boid.id);
    
    // 3. Calculate updates using spatial grid for efficient neighbor lookup
    metrics.begin_movement();
    let updates: Vec<(Entity, Vec2, Vec2)> = boid_states
        .iter()
        .map(|(entity, state, boid, grid_pos)| {
            let mut neighbors: Vec<&BoidState> = Vec::new();
            
            // Calculate cell range based on visual range
            let cell_range = (params.visual_range / GRID_CELL_SIZE).ceil() as i32;
            
            // Check cells within the calculated range
            for dy in -cell_range..=cell_range {
                for dx in -cell_range..=cell_range {
                    let check_cell = grid_pos.cell + IVec2::new(dx, dy);
                    
                    if !SpatialGrid::is_in_range(grid_pos.cell, check_cell, params.visual_range) {
                        continue;
                    }
                    
                    if let Some(cell_entities) = grid.cells.get(&GridKey::from(check_cell)) {
                        // Add states for boids in this cell
                        for &(other_entity, _) in cell_entities {
                            if other_entity != *entity {
                                if let Some(other_state) = boid_states.iter()
                                    .find(|(e, _, _, _)| *e == other_entity)
                                    .map(|(_, state, _, _)| state) {
                                    
                                    // Final precise distance check
                                    let diff = other_state.position - state.position;
                                    if diff.length() <= params.visual_range {
                                        neighbors.push(other_state);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            neighbors.sort_by(|a, b| {
                a.position.x.partial_cmp(&b.position.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.position.y.partial_cmp(&b.position.y)
                        .unwrap_or(std::cmp::Ordering::Equal))
            });
            
            let (new_velocity, new_position) = 
                calculate_boid_update(state, boid, &neighbors, &params, dt, &config);

            // Apply screen wrapping
            let wrapped_position = Vec2::new(
                (new_position.x + width / 2.0).rem_euclid(width) - width / 2.0,
                (new_position.y + height / 2.0).rem_euclid(height) - height / 2.0
            );
            
            (*entity, new_velocity, wrapped_position)
        })
        .collect();
    metrics.end_movement();

    // 4. Apply updates in deterministic order
    for (entity, new_velocity, new_position) in updates {
        if let Ok((_, mut transform, mut boid, _)) = query.get_mut(entity) {
            boid.velocity = new_velocity;
            transform.translation = new_position.extend(transform.translation.z);
            boid.frame_count += 1;
            
            // Update trail
            if params.trace_paths {
                boid.trail.push(new_position);
                if boid.trail.len() > TRAIL_LENGTH {
                    boid.trail.remove(0);
                }
            }
            
            // Update rotation deterministically
            if boid.velocity.length_squared() > 0.0 {
                let angle = boid.velocity.y.atan2(boid.velocity.x);
                transform.rotation = Quat::from_rotation_z(angle - std::f32::consts::FRAC_PI_2);
            }
        }
    }

    metrics.end_physics();

    // 5. Validation handling
    let final_states: Vec<BoidState> = query.iter().map(|(_, transform, boid, _)| {
        BoidState {
            position: transform.translation.truncate(),
            velocity: boid.velocity,
        }
    }).collect();

    match validation_mode {
        ValidationMode::Validating => {
            if !validator.validate_state(current_step, &final_states, &params) {
                error!("State validation failed at step {}", current_step);
            }
            validator.current_step += 1;
        },
        ValidationMode::Recording => {
            if current_step % 1000 == 0 {  // Record every 1000th step
                validator.snapshots.push(StateSnapshot {
                    step: current_step,
                    boid_states: final_states,
                    parameters: params.clone(),
                });
            }
            validator.current_step += 1;
        },
        ValidationMode::Disabled => {}
    }
}

/// Calculates the new velocity and position for a single boid
/// All inputs are processed in a deterministic order
fn calculate_boid_update(
    state: &BoidState,
    boid: &Boid,
    neighbors: &[&BoidState],
    params: &SimulationParams,
    dt: f32,
    config: &SimulationConfig,
) -> (Vec2, Vec2) {
    let mut center_of_mass = Vec2::ZERO;
    let mut avoid_vector = Vec2::ZERO;
    let mut average_velocity = Vec2::ZERO;
    let mut num_neighbors = 0;

    // Process neighbors in their sorted order
    for neighbor in neighbors {
        let diff = neighbor.position - state.position;
        let distance = diff.length();
        
        if distance < params.visual_range && distance > 0.0 {
            // Accumulate in deterministic order
            center_of_mass += neighbor.position;
            
            if distance < params.visual_range / 2.0 {
                avoid_vector -= diff.normalize() * (params.visual_range / (2.0 * distance.max(0.1)));
            }
            
            average_velocity += neighbor.velocity;
            num_neighbors += 1;
        }
    }

    let mut new_velocity = state.velocity;

    if num_neighbors > 0 {
        // Calculate forces in deterministic order
        center_of_mass /= num_neighbors as f32;
        average_velocity /= num_neighbors as f32;
        
        // Apply forces in fixed order for determinism
        let coherence = (center_of_mass - state.position) * params.coherence;
        let separation = avoid_vector * params.separation;
        let alignment = (average_velocity - state.velocity) * params.alignment;
        
        new_velocity += coherence;
        new_velocity += separation;
        new_velocity += alignment;
    }

    // Apply noise influence deterministically using boid ID
    let perlin = Perlin::new(config.noise_seed);
    let noise_value = get_deterministic_noise(&perlin, boid, boid.frame_count);
    let turn_angle = noise_value * NOISE_STRENGTH * dt * NOISE_INFLUENCE;
    let (sin, cos) = turn_angle.sin_cos();
    let noise_direction = Vec2::new(
        new_velocity.x * cos - new_velocity.y * sin,
        new_velocity.x * sin + new_velocity.y * cos
    ).normalize();
    
    new_velocity = new_velocity.lerp(noise_direction * new_velocity.length(), NOISE_INFLUENCE);

    // Apply speed limits deterministically
    let speed = new_velocity.length();
    if speed < BOID_SPEED_LIMIT / 2.0 {
        new_velocity = new_velocity.normalize() * (BOID_SPEED_LIMIT / 2.0);
    } else if speed > BOID_SPEED_LIMIT {
        new_velocity = new_velocity.normalize() * BOID_SPEED_LIMIT;
    }

    // Calculate new position (wrapping will be applied in the physics system)
    let new_position = state.position + new_velocity * dt;

    (new_velocity, new_position)
}

/// Generates deterministic noise value for a boid's movement
/// Uses the boid's ID and frame count to ensure reproducibility
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

/// Updates the spatial grid for efficient neighbor lookups
/// Maintains deterministic ordering of entities within cells
fn update_spatial_grid(
    mut grid: ResMut<SpatialGrid>,
    query: Query<(Entity, &Transform, &GridPosition, &Boid)>,
    mut commands: Commands,
    mut metrics: ResMut<PerformanceMetrics>,
) {
    metrics.begin_spatial();
    
    grid.cells.clear();
    
    for (entity, transform, grid_pos, boid) in query.iter() {
        let position = transform.translation.truncate();
        let new_cell = SpatialGrid::world_to_cell(position);
        
        if new_cell != grid_pos.cell {
            commands.entity(entity).insert(GridPosition { cell: new_cell });
        }
        
        grid.cells.entry(GridKey::from(new_cell))
            .or_default()
            .push((entity, boid.id));
    }

    // Sort entities within each cell by boid ID for deterministic ordering
    for entities in grid.cells.values_mut() {
        entities.sort_by_key(|&(_, id)| id);
    }

    metrics.end_spatial();
}

// ====================================================
// ==================== UI SYSTEMS ====================
// ====================================================

/// Sets up the user interface elements including parameter controls and debug info
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
            // FPS Counter in top-left corner
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

            // Control Panel at bottom
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

/// Helper function to create a text input field with label
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

/// Helper function to create a button with text
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

/// Updates the FPS counter text
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

/// Handles text input for parameter adjustment fields
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

                match event.key_code {
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
                        if let Some(c) = key_code_to_char(key_code) {
                            state.buffer.insert(state.cursor_position, c);
                            state.cursor_position += 1;
                        }
                    }
                }

                update_text_state_safe(&mut input, &mut text, state);
            }
        }
    }
}

/// Helper struct for managing text input state
#[derive(Clone)]
struct TextState {
    buffer: String,
    cursor_position: usize,
    cursor_visible: bool,
}

/// Safely updates the text input state maintaining cursor visibility
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

/// Updates the cursor blink state for text inputs
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

// =======================================================
// ==================== CONTROL SYSTEMS ====================
// =======================================================

/// Handles button interactions including reset and trace path toggles
/// Ensures deterministic reset behavior by reinitializing RNG with original seed
fn handle_button_clicks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut rng: ResMut<SimulationRng>,
    config: Res<SimulationConfig>,
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
                    // Clean up existing entities
                    for entity in grid_text_query.iter() {
                        commands.entity(entity).despawn_recursive();
                    }
                    
                    for entity in boid_query.iter() {
                        commands.entity(entity).despawn_recursive();
                    }
                    
                    // Reset RNG to initial seed for deterministic behavior
                    *rng = SimulationRng::new(config.rng_seed);
                    
                    // Spawn new boids with reset RNG
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

/// Handles the escape key for application exit
fn handle_escape(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut exit: EventWriter<AppExit>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}

// ===============================================================
// ==================== VISUALIZATION SYSTEMS ====================
// ===============================================================

/// Renders movement trails for boids when enabled
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

        let alpha_step = 1.0 / (boid.trail.len() as f32);
        
        for i in 0..boid.trail.len() - 1 {
            let p1 = boid.trail[i];
            let p2 = boid.trail[i + 1];
            let alpha = alpha_step * (i as f32);

            // Calculate shortest path considering screen wrapping
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

            if wrapped_dx.abs() < width / 2.0 && wrapped_dy.abs() < height / 2.0 {
                gizmos.line_2d(
                    p1,
                    p1 + Vec2::new(wrapped_dx, wrapped_dy),
                    Color::srgba(0.33, 0.55, 0.95, alpha),
                );
            }
        }
    }
}

/// Renders the spatial grid for debugging when enabled
fn draw_spatial_grid(
    mut commands: Commands,
    mut gizmos: Gizmos,
    grid: Res<SpatialGrid>,
    debug_config: Res<DebugConfig>,
    text_query: Query<Entity, With<GridCellText>>,
    window_query: Query<&Window>,
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
    for (cell, entities) in grid.cells.iter() {
        let cell_pos = Vec2::new(
            (cell.x() as f32 * GRID_CELL_SIZE) + (GRID_CELL_SIZE / 2.0),
            (cell.y() as f32 * GRID_CELL_SIZE) + (GRID_CELL_SIZE / 2.0),
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

/// Toggles grid visibility for debugging
fn toggle_grid_visibility(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut debug_config: ResMut<DebugConfig>,
) {
    if keyboard.just_pressed(KeyCode::KeyG) {
        debug_config.show_grid = !debug_config.show_grid;
    }
}

/// Cleans up trails when trace paths is toggled off
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

/// Handles non-interactive simulation modes (benchmark, validate, record)
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
                        info!("Saving benchmark results after {}s", time.elapsed().as_secs_f32());
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
                        info!("Saving reference data after {}s", time.elapsed().as_secs_f32());
                        validator.save_reference(path)
                            .expect("Failed to save reference data");
                    }
                    info!("Recording complete - exiting");
                    app_exit.send(AppExit::Success);
                }
            } else {
                warn!("No duration set for recording mode");
            }
        }
        SimulationMode::Validate => {
            if let Some(duration) = config.benchmark_duration {
                if time.elapsed() >= duration {
                    info!("Validation complete after {}s", time.elapsed().as_secs_f32());
                    app_exit.send(AppExit::Success);
                }
            }
        }
    }
}

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
        .insert_resource(LoggingState::default())
        .insert_resource(Time::<Fixed>::default())
        .insert_resource(PerformanceMetrics::default())
        .insert_resource(DeterminismValidator::from_config(&config))
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
        ).chain());

    // Add frame timing system to run last
    app.add_systems(Update, track_frame_times.in_set(TimingSet::FrameTiming))
        .configure_sets(Update, (TimingSet::FrameTiming,).after(handle_non_interactive_modes));

    // Mode-specific setup
    match config.mode {
        SimulationMode::Interactive => {
            // Add interactive mode systems
            app.add_plugins(FrameTimeDiagnosticsPlugin::default())
                .add_systems(Startup, setup_ui)
                .add_systems(Update, (
                    // UI Systems
                    handle_text_input,
                    update_fps_text,
                    update_cursor,
                    handle_button_clicks,
                    handle_escape,
                    
                    // Visualization Systems
                    draw_spatial_grid,
                    toggle_grid_visibility,
                    update_trails,
                    cleanup_trails_on_toggle,
                ).chain());
        }
        _ => {
            // Add non-interactive mode systems
            app.add_systems(Update, (
                handle_non_interactive_modes,
                apply_parameter_schedule,
            ));

            // Load validation data if in validation mode
            if let SimulationMode::Validate = config.mode {
                if let Some(ref path) = config.reference_path {
                    let mut validator = app.world_mut().resource_mut::<DeterminismValidator>();
                    validator.load_reference(path)
                        .expect("Failed to load reference data");
                }
            }
        }
    }

    // Run the app
    app.run();
}

// Helper function to convert keycode to character
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