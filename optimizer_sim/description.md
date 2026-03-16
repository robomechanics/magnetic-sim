# Sally Magnetic Wall-Climbing Robot – Simulation & Optimization

This codebase implements a MuJoCo-based physics simulation and Bayesian optimization pipeline for **Sally**, a magnetic wall-climbing robot. The goal is to tune simulation and control parameters so that Sally can reliably adhere to and drive along a vertical steel wall.

**Current Status:** Stable climbing behavior achieved for both `drive_up` (vertical) and `drive_sideways` (lateral) modes.

---

## Overview

Sally is a four-wheeled robot whose wheels contain permanent magnets that attract to a ferromagnetic wall. The simulation models magnetic adhesion forces, wheel contact dynamics, and actuator control to evaluate robot performance across three operating modes. A Bayesian optimizer searches a 16-dimensional parameter space to minimize a mode-specific cost function.

**Robot Mass:** 8.3 kg total (frame ~1.42 kg + rocker/wheel assemblies + 1.9 kg pXRF sensor payload distributed across frame body).

---

## Operating Modes

The system supports three distinct modes, configured in `config.py`:

**Hold** – All wheels are locked at a fixed position (zero velocity target). The cost function penalizes any movement at all; the optimizer tries to find parameters that keep Sally perfectly stationary on the wall with zero slip.

**Drive Sideways** – Wheels are oriented sideways (pivot angle = 0) and given a velocity target of 1 m/s. The cost function measures deviation of the robot's average lateral (Y-axis) velocity from the target, with smaller penalties for unwanted X and Z displacement.

**Drive Up** – Wheels are oriented to drive vertically (pivot angle = -π/2) with a 2 m/s target. The cost function measures deviation of the robot's average vertical (Z-axis) velocity from the target, with smaller penalties for unwanted X and Y displacement.

---

## Actuator Architecture

**Pivot joints** use `<position>` actuators (kp=20, kv=10 in XML) — these lock each wheel's orientation to the mode-specific `pivot_angle`.

**Wheel drive joints** use **`<intvelocity>`** (integrated-velocity) actuators. This is the MuJoCo-recommended approach for velocity tracking — unlike pure `velocity` actuators, `intvelocity` actuators track both position and velocity as part of the activation state, giving significantly more stable velocity control.

The control signal `data.ctrl[act_id]` is set to `actuator_target_rads` (the desired wheel angular velocity in rad/s). The optimizer tunes the `wheel_kv` damping gain at runtime via `model.actuator_gainprm[act_id, 0]`. The `wheel_kp` term does not apply to `intvelocity` actuators and is not tuned.

---

## File Descriptions

### `config.py`

Central configuration hub for all mode settings, default parameters, and the Bayesian optimization search space. Key responsibilities:

- **Mode configurations** – Each mode specifies actuator strategy, target velocity, cost function name, simulation duration, settle time, and pivot joint angle.
- **`DEFAULT_PARAMS`** – Best-known parameter set (currently optimized for hold and sideways modes). Used as a warm-start `x0` for optimization runs.
- **`SEARCH_SPACE`** – 16-dimensional `skopt` search space covering magnetic, solver, friction, joint dynamics, and control parameters.
- **Derived constants** – Velocity targets in m/s are automatically converted to rad/s using `WHEEL_RADIUS = 0.025 m`.

Key constants:
- `N_CALLS`: Number of Bayesian optimization iterations (default: 20)
- `WHEEL_RADIUS`: 0.025 m (25 mm)
- `DEFAULT_MODE`: Active mode for direct simulation runs

### `sim_optimizer.py`

Core simulation engine. Key responsibilities:

- **Magnetic force model** – Uses a dipole-dipole approximation to compute attraction force as a function of distance between each wheel's sampling spheres (96 total, 24 per wheel) and the wall. Forces are clamped to a configurable per-wheel maximum (`max_force_per_wheel`).
- **Parameter application** – Accepts a dictionary of tunable parameters and applies them to the MuJoCo model before simulation: wall friction coefficients, solver settings (`solref`, `solimp`, `noslip_iterations`), rocker joint stiffness/damping, wheel actuator damping gain (`wheel_kv`), magnetic field strength, and force limits.
- **Initial conditions** – The robot is placed 35 mm from the wall at a height of 0.35 m, rotated to face the wall surface (quaternion: [-0.707, 0, 0.707, 0]).
- **Simulation loop** – At each timestep (1 kHz), the engine:
  - Clears external forces
  - Computes magnetic forces for all sampling spheres within `max_magnetic_distance`
  - Applies forces to wheel bodies
  - Sets actuator commands to `actuator_target_rads` (constant velocity target)
  - Steps the physics
  - Checks for instability (non-finite accelerations, solver failures, excessive velocities)
- **Trajectory output** – Returns a list of timestamped state snapshots (position, velocity, quaternion) or `None` if the simulation diverges.
- **Visualization mode** – Can delegate to `viewer.py` for interactive 3D visualization.

### `tune_params.py`

Bayesian optimization driver using `scikit-optimize` (`gp_minimize`). Key responsibilities:

- **Objective function** – Assembles `sim_params` from the flat optimizer parameter vector, runs a headless simulation, then calls the mode-specific cost function.
- **Cost functions** – Three mode-specific functions:
  - `cost_minimize_slip`: Total Euclidean path length after settle time (target = 0 movement)
  - `cost_drive_side`: Absolute error between average Y velocity and target, plus weighted X/Z displacement penalties and a Y distance penalty
  - `cost_drive_up`: Absolute error between average Z velocity and target, plus weighted X/Y displacement penalties
- **Optimization loop** – Runs `N_CALLS` iterations of Gaussian Process–based optimization with `n_initial_points = N_CALLS // 5` random exploration points.
- **Warm start** – A `best_hold_x0` vector is constructed from `DEFAULT_PARAMS` for use as `x0` in `gp_minimize` (uncomment to enable).
- **Results** – Sorted by cost and saved to `optimization_results_{mode}.csv`. Each run gets a unique 8-character ID.
- **Auto-visualization** – After optimization, automatically launches the best parameter set in the interactive viewer.

Usage:
```bash
python tune_params.py --mode hold
python tune_params.py --mode drive_sideways
python tune_params.py --mode drive_up
```

### `viewer.py`

Interactive 3D visualization using MuJoCo's passive viewer. Features:

- **Force arrows** – Blue arrows rendered at each magnetic contact point, scaled proportionally to force magnitude (2 mm per 10 N).
- **Direction arrow** – Red arrow showing the robot's forward (Y-axis) direction.
- **Playback controls** – Press Enter to start, Space to pause/resume.
- **Camera tracking** – Automatically follows the robot body (trackbodyid=1) with an adjustable orbit camera (distance=8.0, azimuth=45°, elevation=-45°).
- **Real-time telemetry** – Prints COM velocity every 0.5s (X, Y, Z components and total speed).
- **CSV replay** – Loads any ranked result from a previous optimization CSV and replays it with full visualization.
- **Solver settings** – Matches headless simulation settings (iterations=100, tolerance=1e-8) to ensure visual consistency.

Usage:
```bash
# View best result from hold mode optimization
python viewer.py --mode hold --rank 1 --duration 10.0

# View second-best drive_up result
python viewer.py --mode drive_up --rank 2
```

---

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
| `wheel_kv` | 0.1 – 10 | log-uniform | Wheel `intvelocity` actuator damping gain |
| `max_magnetic_distance` | 5 – 100 mm | log-uniform | Magnetic force cutoff distance |
| `noslip_iterations` | 5 – 30 | integer | Solver no-slip iterations |
| `max_force_per_wheel` | 100 – 300 N | uniform | Maximum magnetic force per wheel |

> **Note:** `wheel_kp` is not tuned — it does not apply to `intvelocity` actuators. Only `wheel_kv` (the damping gain) is relevant and is set via `model.actuator_gainprm[act_id, 0]`.

---

## Best Known Parameters

Defined in `config.py` as `DEFAULT_PARAMS`. These were optimized for hold and sideways modes and serve as a warm-start for further optimization runs:

```python
DEFAULT_PARAMS = {
    'ground_friction': [0.923048, 0.000546, 0.000241],
    'solref': [0.000349, 30.867833],
    'solimp': [0.870259, 0.980192, 0.000135, 0.5, 1.0],
    'noslip_iterations': 12,
    'rocker_stiffness': 991.791538,
    'rocker_damping': 0.279167,
    'wheel_kp': 1.655694,       # legacy field; not applied to intvelocity
    'wheel_kv': 4.849150,
    'Br': 1.490629,
    'max_magnetic_distance': 0.009070,
    'max_force_per_wheel': 53.821678,
}
```

---

## Workflow

1. **Run optimization** for a specific mode:
   ```bash
   python tune_params.py --mode drive_up
   ```
   This runs 20 iterations and saves results to `optimization_results_drive_up.csv`, then automatically launches the viewer with the best parameters.

2. **Review results** in the CSV file, sorted by cost (lowest first).

3. **Visualize specific results**:
   ```bash
   python viewer.py --mode drive_up --rank 1 --duration 10.0
   ```

4. **Test parameters directly** by modifying `DEFAULT_PARAMS` in `config.py` and running:
   ```bash
   python sim_optimizer.py
   ```

---

## Key Implementation Details

- **Actuator type**: `intvelocity` for wheel drive joints; `position` for pivot joints
- **Velocity control**: `data.ctrl[act_id] = actuator_target_rads` (constant target, not a ramp)
- **Magnetic force calculation**: Dipole-dipole model with distance^4 falloff
- **Sampling resolution**: 24 spheres per wheel for fine-grained contact detection
- **Simulation timestep**: 1 kHz (1e-3 s), fixed
- **Pivot joint locking**: Pivot angles set and locked at 1000 N/m stiffness to enforce mode geometry (overrides the `rocker_stiffness` parameter for pivot joints only)
- **Instability detection**: Checks for non-finite accelerations, solver iteration limits, excessive velocities (>100 m/s), and invalid trajectory data
- **Settle time**: 0.2s initial period excluded from cost calculation to allow transients to decay
- **Robot mass**: 8.3 kg total, with a 1.9 kg pXRF sensor payload added as a non-colliding geom on the frame body

---

## Dependencies

- MuJoCo (with Python bindings)
- NumPy
- SciPy
- scikit-optimize (`skopt`)

---

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
- Tune `wheel_kv` gain (the only active gain for `intvelocity` actuators)
- Adjust `rocker_stiffness` for better contact distribution
- Check `max_force_per_wheel` is sufficient for the robot's 8.3 kg mass

**Optimization converges slowly:**
- Uncomment `x0=best_hold_x0` in `tune_params.py` to start from known good parameters
- Increase `n_initial_points` for more exploration
- Increase `N_CALLS` for longer search