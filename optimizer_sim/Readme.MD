# Sally Magnetic Wall-Climbing Robot – Simulation & Optimization

This codebase implements a MuJoCo-based physics simulation and Bayesian optimization pipeline for **Sally**, a magnetic wall-climbing robot. The goal is to tune simulation and control parameters so that Sally can reliably adhere to and drive along a vertical steel wall.

## Overview

Sally is a four-wheeled robot whose wheels contain permanent magnets that attract to a ferromagnetic wall. The simulation models magnetic adhesion forces, wheel contact dynamics, and actuator control to evaluate robot performance across three operating modes. A Bayesian optimizer searches a 16-dimensional parameter space to minimize a mode-specific cost function.

## Operating Modes

The system supports three distinct modes, configured in `config.py`:

**Hold** – All wheels are locked at a fixed position (zero velocity target). The cost function penalizes any movement at all; the optimizer tries to find parameters that keep Sally perfectly stationary on the wall with zero slip.

**Drive Sideways** – Wheels are oriented sideways (pivot angle = 0) and given a velocity target of 2 m/s. The cost function measures how closely the robot's average lateral (Y-axis) velocity matches the target.

**Drive Up** – Wheels are oriented to drive vertically (pivot angle = -π/2) with a 2 m/s target. The cost function measures deviation of the robot's average vertical (Z-axis) velocity from the target.

Velocity control is implemented as a position ramp: the actuator target is set to `ω_target × t`, so the PD controller continuously chases a moving setpoint, producing approximately constant angular velocity.

## File Descriptions

### `config.py`

Defines the mode configurations and derived constants. Each mode specifies the actuator control strategy (position hold vs. velocity ramp), target values, cost function name, simulation duration, settle time, and pivot joint angle. Velocity targets in m/s are converted to rad/s using the known wheel radius (25 mm).

Key constants:
- `N_CALLS`: Number of Bayesian optimization iterations (default: 20)
- `WHEEL_RADIUS`: 0.025 m (25 mm)
- Default mode can be set via `DEFAULT_MODE`

### `sim_optimizer.py`

Core simulation engine. Key responsibilities:

- **Magnetic force model** – Uses a dipole-dipole approximation to compute attraction force as a function of distance between each wheel's sampling spheres (96 total, 24 per wheel) and the wall. Forces are clamped to a configurable per-wheel maximum (`max_force_per_wheel`).
- **Parameter application** – Accepts a dictionary of tunable parameters and applies them to the MuJoCo model before simulation: wall friction coefficients, solver settings (`solref`, `solimp`, `noslip_iterations`), rocker joint stiffness/damping, wheel actuator PD gains, magnetic field strength, and force limits.
- **Initial conditions** – The robot is placed 35 mm from the wall at a height of 0.35 m, rotated to face the wall surface (quaternion: [-0.707, 0, 0.707, 0]).
- **Simulation loop** – At each timestep (1 kHz), the engine:
  - Clears external forces
  - Computes magnetic forces for all sampling spheres within `max_magnetic_distance`
  - Applies forces to wheel bodies
  - Sets actuator commands based on the active mode
  - Steps the physics
  - Checks for instability (non-finite accelerations, solver failures, iteration limits)
- **Trajectory output** – Returns a list of timestamped state snapshots (position, velocity, quaternion) or `None` if the simulation diverges.
- **Visualization mode** – Can delegate to `viewer.py` for interactive 3D visualization.

### `tune_params.py`

Bayesian optimization driver using `scikit-optimize` (`gp_minimize`). Defines:

- **Search space** – 16 parameters spanning:
  - Magnetic properties: remanence (`Br`), cutoff distance (`max_magnetic_distance`), max force per wheel
  - Solver tuning: `solref` (time constant, damping ratio), `solimp` (dmin, dmax, width), `noslip_iterations`
  - Wall friction: sliding, torsional, rolling
  - Rocker joint dynamics: stiffness, damping
  - Wheel actuator gains: `kp`, `kv`
  
- **Cost functions** – Three mode-specific functions:
  - `cost_minimize_slip`: Total Euclidean path length after settle time (target = 0)
  - `cost_drive_side`: Absolute error between average Y velocity and target
  - `cost_drive_up`: Absolute error between average Z velocity and target
  
- **Optimization loop** – Runs `N_CALLS` iterations (default 20) of Gaussian Process–based optimization with:
  - `n_initial_points`: N_CALLS // 5 random exploration points
  - Results are sorted by cost and saved to CSV (`optimization_results_{mode}.csv`)
  - Each run gets a unique 8-character ID for tracking
  
- **Auto-visualization** – After optimization completes, automatically launches the best parameter set in the interactive viewer.

Usage:
```bash
python tune_params.py --mode hold
python tune_params.py --mode drive_sideways
python tune_params.py --mode drive_up
```

### `viewer.py`

Interactive 3D visualization using MuJoCo's passive viewer. Features:

- **Force arrows** – Blue arrows rendered at each magnetic contact point, scaled proportionally to force magnitude (2 mm per 10 N).
- **Playback controls** – Press Enter to start, Space to pause/resume.
- **Camera tracking** – Automatically follows the robot body (trackbodyid=1) with an adjustable orbit camera (distance=8.0, azimuth=45°, elevation=-45°).
- **Real-time telemetry** – Prints COM velocity every 0.5s (X, Y, Z components and total speed).
- **CSV replay** – Can load any ranked result from a previous optimization CSV and replay it with full visualization.
- **Near/far plane settings** – Adjusted for close-up viewing (znear=0.01, zfar=100.0).

Usage:
```bash
# View best result from hold mode optimization
python viewer.py --mode hold --rank 1 --duration 10.0

# View second-best result
python viewer.py --mode drive_sideways --rank 2
```

## Parameter Space Summary

| Parameter | Range | Scale | Description |
|-----------|-------|-------|-------------|
| `Br` | 1.332 – 1.628 T | uniform | Magnet remanence (±10% of 1.48 T) |
| `solref_timeconst` | 0.0001 – 0.0008 | uniform | Contact time constant |
| `solref_dampratio` | 10 – 50 | uniform | Contact damping ratio |
| `solimp_dmin` | 0.8 – 0.99 | uniform | Impedance min distance |
| `solimp_dmax` | 0.9 – 1.0 | uniform | Impedance max distance |
| `solimp_width` | 1e-4 – 1e-2 | log-uniform | Impedance transition width |
| `sliding_friction` | 0.9 – 1.0 | uniform | Coulomb friction coefficient |
| `torsional_friction` | 1e-5 – 0.1 | log-uniform | Torsional friction |
| `rolling_friction` | 1e-5 – 0.1 | log-uniform | Rolling friction |
| `rocker_stiffness` | 100 – 1000 | uniform | Rocker joint spring stiffness (N/m) |
| `rocker_damping` | 0.1 – 5.0 | log-uniform | Rocker joint damping (N·s/m) |
| `wheel_kp` | 1 – 50 | log-uniform | Wheel actuator proportional gain |
| `wheel_kv` | 0.1 – 10 | log-uniform | Wheel actuator derivative gain |
| `max_magnetic_distance` | 5 – 100 mm | log-uniform | Magnetic force cutoff distance |
| `noslip_iterations` | 5 – 50 | integer | Solver no-slip iterations |
| `max_force_per_wheel` | 20 – 1000 N | log-uniform | Maximum magnetic force per wheel |

## Best Known Parameters

The codebase includes optimized default parameters for hold mode in `sim_optimizer.py`:

```python
default_params = {
        'ground_friction': [0.945545, 0.1, 0.000824],
        'solref': [0.000452, 37.916003],
        'solimp': [0.827705, 1.0, 0.003718, 0.5, 1.0],
        'noslip_iterations': 28,
        'rocker_stiffness': 23.830724,
        'rocker_damping': 0.349339,
        'wheel_kp': 3.109834,
        'wheel_kv': 0.958800,
        'Br': 1.551777,
        'max_magnetic_distance': 0.008674,
    }
```

These parameters can be used as a warm start (`x0`) for optimization in other modes.

## Workflow

1. **Run optimization** for a specific mode:
   ```bash
   python tune_params.py --mode hold
   ```
   This runs 20 iterations and saves results to `optimization_results_hold.csv`, then automatically launches the viewer with the best parameters.

2. **Review results** in the CSV file, sorted by cost (lowest first).

3. **Visualize specific results**:
   ```bash
   python viewer.py --mode hold --rank 1 --duration 10.0
   ```

4. **Test parameters directly** by modifying `default_params` in `sim_optimizer.py` and running:
   ```bash
   python sim_optimizer.py
   ```

## Key Implementation Details

- **Magnetic force calculation**: Dipole-dipole model with distance^4 falloff
- **Sampling resolution**: 24 spheres per wheel for fine-grained contact detection
- **Simulation timestep**: 1 kHz (1e-3 s)
- **Pivot joint locking**: Pivot angles are set and locked at 1000 N/m stiffness to enforce mode geometry
- **Instability detection**: Checks for non-finite accelerations, solver iteration limits, and invalid trajectory data
- **Settle time**: 0.2s initial period excluded from cost calculation to allow transients to decay

## Dependencies

- MuJoCo (with Python bindings)
- NumPy
- SciPy
- scikit-optimize (`skopt`)
- imageio (optional, for recording – currently unused)

## Troubleshooting

**Simulation fails with "unstable" errors:**
- Reduce `rocker_stiffness` or increase `rocker_damping`
- Increase `solref_timeconst` for softer contacts
- Reduce `max_force_per_wheel` if magnetic forces are too aggressive

**Robot falls off wall:**
- Increase `Br` (magnetic strength)
- Increase `max_magnetic_distance` (engagement range)
- Increase `sliding_friction`

**Poor tracking in drive modes:**
- Tune `wheel_kp` and `wheel_kv` gains
- Adjust `rocker_stiffness` for better contact distribution
- Check `max_force_per_wheel` is sufficient

**Optimization converges slowly:**
- Use `x0` parameter in `gp_minimize` to start from known good parameters
- Increase `n_initial_points` for more exploration
- Increase `N_CALLS` for longer search