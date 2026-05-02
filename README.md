# magnetic-sim

**Physics-based simulation suite for Sally — a quadrupedal magnetic wall-climbing robot developed at the CMU Robomechanics Lab.**

This repository documents the full simulation development arc, from early wheeled-platform magnetic adhesion modeling through to legged locomotion with inverse kinematics, floor-to-wall transition sequences, and CMA-ES physics parameter optimization. Each subdirectory represents a distinct research phase, and all modules share a common MuJoCo-based physics backend.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Module Guide](#module-guide)
  - [sally\_sim — Wheeled Platform Baseline](#sally_sim--wheeled-platform-baseline)
  - [optimizer\_sim — Bayesian Physics Optimization (Wheeled)](#optimizer_sim--bayesian-physics-optimization-wheeled)
  - [mwc\_sim — Standalone EM Characterization (Legacy)](#mwc_sim--standalone-em-characterization-legacy)
  - [legged\_sim — Legged Locomotion & Floor-to-Wall Transition](#legged_sim--legged-locomotion--floor-to-wall-transition)
  - [legged\_sim/optimizer — Combined Floor / Wall / Pull-off Optimization](#legged_simoptimizer--combined-floor--wall--pull-off-optimization)
- [Configuration Reference](#configuration-reference)
- [Architecture Notes](#architecture-notes)

---

## Project Overview

Sally is a quadrupedal robot designed to traverse ferromagnetic surfaces — including the floor-to-wall and wall-to-ceiling transitions required in industrial inspection tasks. Each of her four legs terminates in an electromagnet foot that can adhere to steel surfaces. The simulation stack in this repository was built to:

1. **Characterize magnetic adhesion** using a dipole-dipole force model calibrated against real pull-off and wrench test data.
2. **Identify physics parameters** (friction, contact solver settings, magnetic field strength) via black-box optimization so that simulation behavior matches physical hardware.
3. **Plan and execute legged locomotion** across surface transitions using differential inverse kinematics (IK), PID joint control, and a phase-based sequence runner.

The modules are largely independent but share common conventions for magnetic force modeling, MuJoCo XML structure, and CMA-ES optimization infrastructure. The active research module is `legged_sim/`, which contains both the legged simulation (`sim.py`) and a unified physics-parameter optimizer (`legged_sim/optimizer/`) that supersedes the standalone wheeled and electromagnet-only optimizers.

---

## Repository Structure

```
magnetic_sim/
│
├── sally_sim/                    # Phase 1 — wheeled Sally MuJoCo baseline
│   ├── robot_sally_patched.xml   # Full MJCF for wheeled Sally (4-wheel omnidirectional)
│   ├── sim_sally_magnet_wall.py  # Interactive viewer: magnetic force + drive
│   └── viewer.py                 # Bare scene viewer (no control logic)
│
├── optimizer_sim/                # Phase 2 — Bayesian optimization for wheeled platform
│   ├── config.py                 # Mode definitions (hold / drive_sideways / drive_up)
│   ├── sim_optimizer.py          # Headless MuJoCo sim + cost evaluation
│   ├── tune_params.py            # scikit-optimize GP minimize entry point
│   └── viewer.py                 # Visualization of optimization results
│
├── mwc_sim/                      # Phase 3 — standalone EM characterization (legacy)
│   ├── pulloff_*.py              # Pull-off test sim, optimizer, viewer
│   └── wrench_*.py               # Wrench/peel test sim, optimizer, viewer
│
└── legged_sim/                   # Phase 4 — Legged locomotion (primary active module)
    ├── mwc_mjcf/
    │   ├── scene.xml             # World scene: floor, wall, lighting, camera
    │   ├── robot.xml             # Sally MJCF (regenerated at runtime via bake_joint_angles)
    │   └── robot_original.xml    # Reference MJCF — never written to
    ├── config.py                 # Physics constants, bake angles, magnet body names
    ├── sequences.py              # Typed IKTarget / PhaseContext / SequenceRunner (no MuJoCo deps)
    ├── sim.py                    # Main sim entry point: IK, PID, magnetics, viz, dual-foot f2w
    ├── viewer.py                 # Marker-drawing helpers and joint-vis builder
    │
    └── optimizer/                # Combined floor-lift + wall-hold + pull-off optimizer
        ├── combined_config.py    # Single-source-of-truth config + cost functions
        ├── combined_optmizer.py  # CMA-ES driver (parallel pool, weighted multi-sim cost)
        ├── sim_opt_sim.py        # Headless FL-lift sim (floor adhesion under stance)
        ├── sim_wallopt_sim.py    # Headless f2w sim (FL planted on wall, stance drift)
        ├── sim_pulloff_sim.py    # Headless pull-off sim (vertical detach force)
        └── viewer.py             # Replay viewer for floor / wall / pulloff with force plots
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `mujoco >= 3.0` | Physics engine and viewer |
| `mink` | Differential IK with `FrameTask` / SE3 targets |
| `numpy` | Numerical computing throughout |
| `cma` | CMA-ES black-box optimizer (combined optimizer + `mwc_sim`) |
| `scikit-optimize` | Gaussian-process Bayesian optimizer (`optimizer_sim`) |
| `scipy` | Rotation utilities |
| `matplotlib` | Force / displacement / per-leg plots |
| `quadprog` | QP solver backend used by `mink.solve_ik` |

---

## Installation

It is recommended to use a conda or virtualenv environment.

```bash
# Clone the repository
git clone https://github.com/NSSRS/magnetic_sim.git
cd magnetic_sim

# Install core dependencies
pip install mujoco mink numpy scipy matplotlib scikit-optimize cma quadprog

# Verify MuJoCo installation
python -c "import mujoco; print(mujoco.__version__)"
```

> **Note:** `mink` requires `mujoco >= 3.0`. If you encounter IK solver errors, confirm your MuJoCo version with `python -c "import mujoco; print(mujoco.__version__)"`.

> **WSL Users:** If running under Windows Subsystem for Linux, the MuJoCo passive viewer requires a working X11/Wayland display forwarding setup (e.g. VcXsrv or WSLg). Headless modes (`--headless` for `sim.py`, the optimizer itself) work without display configuration.

---

## Module Guide

---

### `sally_sim` — Wheeled Platform Baseline

**Purpose:** The starting point for the project. Loads the original wheeled Sally MJCF (converted from CAD), applies a dipole-dipole magnetic adhesion model at the wheel contact faces, and provides an interactive viewer for manual testing of adhesion behavior and simple motor drive.

**Key file:** `robot_sally_patched.xml` — defines the full kinematic chain of the four-wheeled omnidirectional Sally platform, including rocker-bogie suspension, pivot joints, wheel actuators, and 24 sampling spheres per wheel.

**Key file:** `sim_sally_magnet_wall.py` — interactive simulation that applies magnetic forces from each wheel toward a vertical steel wall geom. Supports keyboard-toggled drive (`s`) and pause (`space`). Force arrows are rendered per-wheel at each timestep.

**Magnetic force model** (shared across all modules):

```
F = (3μ₀m²) / (2π(2d)⁴)
```

where `m = (Br × V) / μ₀` is the magnetic dipole moment, `Br` is the remanence field strength, `V` is the magnet volume, and `d` is the closest-point distance from the sampling sphere to the steel plate geom.

---

### `optimizer_sim` — Bayesian Physics Optimization (Wheeled)

**Purpose:** Identifies physics parameters for the wheeled Sally simulation — friction coefficients, solver settings, control gains, and magnetic parameters — by running a Gaussian-process Bayesian optimization loop against three behavioral targets: hold (zero slip), drive sideways, and drive upward.

**Key files:** `config.py` (modes, search space), `sim_optimizer.py` (headless sim + cost), `tune_params.py` (`gp_minimize` entry point), `viewer.py` (replay).

**Cost functions:**

| Mode | Cost |
|---|---|
| `hold` | Total 3D displacement after settle (minimize slip) |
| `drive_sideways` | Y-velocity error + off-axis (X, Z) drift penalty |
| `drive_up` | Z-velocity error + off-axis (X, Y) drift penalty |

```bash
cd optimizer_sim
python tune_params.py --mode hold
python tune_params.py --mode drive_sideways
python tune_params.py --mode drive_up
python viewer.py --rank 1 --mode drive_up
```

---

### `mwc_sim` — Standalone EM Characterization (Legacy)

**Purpose:** Original standalone characterization of the EML63mm-12 electromagnet via two CMA-ES optimizers — one for vertical pull-off force, one for lever-arm wrench/peel. **Superseded by `legged_sim/optimizer/`,** which folds pull-off into a combined cost alongside floor-lift and wall-hold sims and uses the same 14-dimensional physics search space.

The directory is preserved for reference. The pull-off goal force (956.37 N, EML63mm-12 holding 215 lbs) and the dipole-dipole sampling-sphere geometry derived from this work are reused directly in the legged optimizer.

```bash
cd mwc_sim
python pulloff_sim.py
python pulloff_optimizer.py
python wrench_optimizer.py --warm-start-from results/<run_dir>
```

---

### `legged_sim` — Legged Locomotion & Floor-to-Wall Transition

**Purpose:** The primary active research module. Implements IK-controlled legged locomotion for the four-legged Sally design, including a five-phase floor-to-wall foot placement sequence and dual-leg sequential placement (FL then FR). This module contains the most recent and complete simulation architecture and serves as the substrate for the combined optimizer in the `optimizer/` subdirectory.

#### Robot Design (`mwc_mjcf/robot.xml`)

Sally has four legs (FL, FR, BL, BR), each with the following joint chain:

| Joint | Type | Range | Description |
|---|---|---|---|
| `hip_pitch_<leg>` | Hinge (Z-axis) | ±45° | Horizontal swing of entire leg |
| `knee_<leg>` | Hinge (diagonal XY) | ±90° | First link bend |
| `wrist_<leg>` | Hinge (diagonal XY) | ±90° | Second link bend |
| `ee_<leg>` | Hinge | ±90° | End-effector pitch |
| `ee2_<leg>` | Hinge (passive, spring+damper) | ±90° | EE compliance joint |
| `em_z_<leg>` | Hinge (passive) | ±180° | Electromagnet yaw compliance |

The electromagnet body on each foot contains 8 sampling spheres in a ring at radius 22 mm, used for the dipole-dipole force calculation. The scene defines a 3 m × 1 m steel floor plate and a 1 m × 1 m steel wall plate with an inner face at X = 0.500 m, forming a clean 90° floor-to-wall corner. Plate top/inner faces are exposed as absolute setpoints (`FLOOR_Z = 0.0`, `WALL_X = 0.5`) consumed by the optimizer cost functions.

`robot.xml` is **regenerated at runtime** by `bake_joint_angles()` from `robot_original.xml` (the clean reference, never written to). The optimizer's worker processes share the same `robot.xml` path, and a per-process `fcntl.LOCK_EX` file lock around bake + model load serialises the rewrite.

#### Typed Sequence Interface (`sequences.py`)

All trajectory logic is isolated in `sequences.py`, which has zero MuJoCo imports. The interface is now built around two dataclasses:

- **`IKTarget`** — complete IK specification for one step: `position`, optional `face_axis` (world direction that EE local −Y should point toward), and per-task costs (`position_cost`, `orientation_cost`). Convenience constructors: `IKTarget.position_only(...)` and `IKTarget.with_orientation(...)`. Replaces the older 4-tuple return.
- **`PhaseContext`** — typed context injected into every phase callback: `foot`, `ee_home`, refreshable `ee_pos_fn` / `hip_pivot_fn` closures, optional `wall_dist_fn`, magnet enable/disable callables, and a free `shared` dict for cross-phase state (used e.g. for `f2w_measure → f2w_reach` to pass the measured contact target).

A `Phase` carries a `name`, `duration`, `step(t_rel, ctx) -> IKTarget` callable, an optional `on_enter(ctx)` hook, and an `unbounded` flag. `SequenceRunner` drives a list of phases by simulation time, enters phases via `on_enter`, advances when `t_rel >= duration` (unless `unbounded`), and exposes `force_complete()` for handoffs (used by the dual-foot mode to freeze FL at its wall contact while FR begins its own sequence).

**Available sequences:**

| Sequence | Description |
|---|---|
| `orient` | Lift → swing −45° → hold with EE facing global −X |
| `f2w` | Full FL floor-to-wall: lift → swing −45° → orient → measure → reach |
| `f2w_fr` | FR mirror of `f2w` (swing **+45°**), used internally for dual-foot placement |

**Floor-to-wall phases (`f2w`):**

1. **LIFT** (3.0 s) — Quintic-smooth raise EE by 10 cm; disables swing foot magnet on entry.
2. **SWING** (1.5 s) — Arc EE around the hip pivot by −45° (CW from above) in the XY plane.
3. **F2W_ORIENT** (3.0 s) — Translate EE to a waypoint 40 cm above `ee_home` Z and rotate to face the wall (`face_axis = −wall_normal`). Uses `ee_home` Z rather than swing-landing Z so the orient height stays consistent if the body has dropped.
4. **F2W_MEASURE** (instant) — Computes EE→wall distance analytically via dot-product projection against the known `wall_face_pos`. Falls back to `mj_ray` if no wall pose is given. Writes `ctx.shared['f2w_reach_target']` with a foot standoff (8 mm) and additional safety clearance (20 mm) applied. This replaces the older `mj_ray`-only approach, which was unreliable when the EE rotated past the wall normal.
5. **F2W_REACH** (2.0 s, unbounded) — Smooth-step EE from current position to `f2w_reach_target`; enables foot magnet once motion completes. Phase holds forever once reached.

#### IK Architecture (`sim.py — IKSolver`)

Whole-body differential IK is solved using `mink` with one `FrameTask` per foot, a `FrameTask` for `main_frame`, a `PostureTask` for joint-limit regularization, and a `ConfigurationLimit`.

- The swing foot consumes an `IKTarget` directly: position, face axis (if present), and costs are applied to the foot's `FrameTask`. `mink` bakes some weights at construction, so the solver lazily rebuilds the swing-foot task when `orientation_cost` changes.
- Stance feet on the **floor** are locked position-only (`stance_targets[foot]`).
- Stance feet on the **wall** are locked SE3 — both position and rotation — via `lock_stance_orientation(data, foot)`, which snapshots the full 4×4 frame at handoff time. The wall-foot task is rebuilt with `orientation_cost = 30`.
- During orient phases (`face_axis is not None`), `body_task.orientation_cost` is set to 0 to avoid competing with the EE orientation task.
- Orientation rotation is computed once per phase via Rodrigues' formula and cached in `_ori_target_rot`; it is invalidated when `face_axis` becomes `None` (e.g. when transitioning to a free-position phase).
- IK is solved every `IK_EVERY_N = 10` physics steps, i.e. every 5 ms at `TIMESTEP = 0.0005` (200 Hz IK rate, 2 kHz physics rate).

#### PID Control (`PIDController`)

Joint-space PID converts IK position targets to motor torques at every physics step:

```
τ = Kp·e + Ki·∫e dt + Kd·ė
Kp=500, Ki=200, Kd=30, I_clamp=±100
```

#### Dual-Foot Mode (`--sequence f2w`)

When `f2w` is selected, `sim.py` runs FL first. Once FL's `F2W_REACH` phase has dwelt for `DUAL_REACH_DWELL = 3.0 s`, FL is declared planted:

1. `runner.force_complete()` freezes FL at its contact target.
2. `ik.lock_stance_orientation(data, "FL")` snapshots FL's full SE3 as a stance lock.
3. The swing-foot pointers (`swing_foot_ref`, `hip_jid_ref`, `swing_mag_bid_ref`) are rotated to FR.
4. A fresh `SequenceRunner` is started for FR using `f2w_fr` (swing **+45°** to mirror FL).
5. FR is added to `wall_feet` once **its** F2W_REACH has dwelt for `DUAL_REACH_DWELL`.

The mutable swing-foot state is held in single-element lists so the closures passed into `PhaseContext` always read the live foot.

#### Per-Step Surface Penetration Telemetry

`_read_surface_penetration` measures EE intrusion using **relative displacement** rather than `mj_geomDistance`, so the readout is unaffected by collision-margin settings:

- **Floor feet** — drop below `stance_targets[foot][2]` is reported as floor penetration; `_CONTACT_LIFT_THRESH = 8 mm` defines lost contact.
- **Wall feet** — advance past `wall_baselines[foot][0]` (snapshotted X at handoff) is reported as wall penetration.

#### Running the Simulation

```bash
cd legged_sim

# GUI — default sequence (set in config.SEQUENCE, currently "f2w")
python sim.py

# GUI — explicit sequence selection
python sim.py --sequence orient
python sim.py --sequence f2w        # dual-foot FL→FR is automatic when f2w selected

# Headless — useful for batch testing
python sim.py --headless --sequence f2w --duration 25.0

# Disable IK (free-fall settle only, for physics debugging)
python sim.py --no-ik

# Enable magnetic forces during the live sim
python sim.py --sequence f2w --magnets
```

**Viewer controls:** `Space` toggles pause/run.

**Telemetry (printed every 1 s):**

```
t= 8.2  [████████░░░░] FL/F2W_REACH  72.3%  pos_err= 4.1mm  (Δx=+2.1 Δy=-3.2 Δz=+1.8)  ang_to_goal=  3.4°  stance=BL/1.2mm  mag=ON
         FL [SWING]  contact=○  airborne
         FR [FLOOR]  contact=●  pen= 0.42mm
         BL [FLOOR]  contact=●  pen= 0.18mm
         BR [FLOOR]  contact=●  pen= 0.21mm  ◄
```

---

### `legged_sim/optimizer` — Combined Floor / Wall / Pull-off Optimization

**Purpose:** Unified CMA-ES optimization over the same 14-dimensional physics-parameter search space against three independent headless simulations. Replaces the separate `sim_opt_config.py`, `sim_wallopt_config.py`, and `sim_pulloff_config.py` config files (and their separate optimizers) with a single source of truth.

The combined cost is a fixed weighted sum:

```
combined_cost = 0.4 · floor_cost  +  0.4 · wall_cost  +  0.2 · pulloff_cost
```

#### Three Sub-Simulations

Each sub-sim is a standalone headless runner that returns scalar penalty terms; the optimizer assembles them into per-sim costs via the corresponding `*_calculate_cost` function in `combined_config.py`.

**1. Floor lift (`sim_opt_sim.run_headless_floor`)** — Robot settles on the floor, FL foot lifts to a fixed height, all other feet must remain planted. Returns `(stance_norm, stance_floor_pen, zero_contact_frac)`.

| Term | Weight | Meaning |
|---|---|---|
| `mean_norm` | 30 % | Mean ‖drift‖ of FR/BL/BR EE from their settled baselines |
| `mean_neg_z` | 30 % | Mean `max(0, FLOOR_Z − ee_z)` of FR/BL/BR (Z drop below floor surface) |
| `zero_contact_frac` | 40 % | Fraction of hold steps where any stance foot lost magnetic contact |

**2. Wall hold (`sim_wallopt_sim.run_headless_wall`)** — Robot settles, runs the full `f2w` sequence to plant FL on the wall, holds for 3 s. Returns `(stance_norm, stance_into_x, zero_contact_frac)`.

| Term | Weight | Meaning |
|---|---|---|
| `fl_norm` | 30 % | Mean ‖drift‖ of FR/BL/BR from settled baselines |
| `fl_into_x` | 30 % | FL EM body X penetration past planted baseline (positive = pushed further into wall) |
| `zero_contact_frac` | 40 % | Fraction of hold steps where any stance foot lost magnetic contact |

**3. Pull-off (`sim_pulloff_sim.run_headless_lift`)** — Single FL magnet on floor, pulled vertically with a linear force ramp at `PULL_RATE_OPT = 40 N/s`. Returns the applied force at the moment a sustained drop in `f_mag` below `DETACH_FORCE_FRAC × max_force_per_wheel` begins. Cost is one-sided shortfall, normalised:

```
cost = max(0, max_force_per_wheel − pulloff_force) / max_force_per_wheel
cost = 9999  if pulloff_force == 0  (no adhesion)
```

The pull-off sim runs a two-stage settle (gravity-only, then gravity + magnetics) with a PID holding the robot in its baked pose throughout — without it the body sags during settle and the FL wheel lifts off before adhesion can engage.

#### `combined_config.py` — Single Source of Truth

Every sub-sim and the optimizer import shared constants from this file: `MU_0`, `MAGNET_VOLUME`, `MAGNET_BODY_NAMES`, `TIMESTEP`, `SETTLE_TIME`, the 14-dim `space` (a list of `Dim` dataclasses replacing `skopt.space.Real`), `point_to_params()`, the three cost functions, and the magnet array geometry.

The file also **rewrites `robot.xml` on every import** by re-baking the magnet array Y offset (`ARRAY_EM_Y_M = -0.010075 m`, placing the array 10 mm from the far end of the EE enclosure). In a parallel optimization run, every worker process executes this on import; safety relies on the `fcntl.LOCK_EX` lock inside `_setup_model`, not on the import-time write itself.

#### `combined_optmizer.py` — CMA-ES Driver

(The filename has a typo — it's `combined_optmizer.py`, not `combined_optimizer.py`.)

- Backend: `cma.CMAEvolutionStrategy` with `popsize = BATCH_SIZE = 16`, `sigma0 = 0.3`, log-uniform / uniform priors handled by a per-dim transform (`is_log` flags map between log-space internal coordinates and real-space parameters).
- Parallelism: `multiprocessing.Pool` with `start_method="spawn"`; pool size = `min(cpu_count, BATCH_SIZE)`. Each worker imports `combined_config` (triggering the `robot.xml` rewrite) and runs all three sub-sims sequentially. Crashes in any sub-sim are caught and converted to a fully-penalised cost (`zero_contact_frac = 1.0` or `pulloff_force = 0.0`).
- Output per run (`results/<timestamp>_<suffix>/`):
  - `optimization_results.csv` — every evaluation
  - `optimization_bests.csv` — only new-best rows, with timestamp + elapsed minutes + eval count
  - `cmaes_state.pkl` — full `CMAEvolutionStrategy` for resume
  - `best_params.json` — best parameters as a flat dict for the viewer
  - Snapshots of `combined_config.py`, `sim_opt_sim.py`, `sim_wallopt_sim.py`, `sim_pulloff_sim.py` for full reproducibility

#### Running Optimization

```bash
cd legged_sim/optimizer

# Default: 300 evals, batch 16, fresh CMA-ES
python combined_optmizer.py

# Custom budget + run tag
python combined_optmizer.py --n-calls 500 --suffix wall_focus

# Resume an interrupted run from its last batch
python combined_optmizer.py --resume-from results/20260101T120000_wall_focus

# Warm-start from another run's best params (loads the last row of optimization_bests.csv)
python combined_optmizer.py --warm-start-from results/20260101T120000_wall_focus
```

After optimization completes, the script automatically launches `viewer.py` with `--mode all`, which replays the floor-lift sim, then the wall-hold sim, then the pull-off sim — each followed by a per-leg force plot (matplotlib, blocking until the window is closed).

#### Viewer Modes

```bash
# Standalone — pick which sub-sim to replay
python viewer.py --params results/<run>/best_params.json --mode floor
python viewer.py --params results/<run>/best_params.json --mode wall
python viewer.py --params results/<run>/best_params.json --mode pulloff
python viewer.py --params results/<run>/best_params.json --mode both   # floor + wall
python viewer.py --params results/<run>/best_params.json --mode all    # floor + wall + pulloff
```

In the MuJoCo window: `Space` / `Enter` toggle pause, `→` single-steps while paused. The pull-off viewer renders blue arrows for per-sphere magnetic attraction and a red arrow for the applied pull force.

#### Search Space (14 dims)

Identical across all three sub-sims, with one carve-out: the **pull-off-only** standalone optimizer (in `mwc_sim`) historically used `[300, 1200]` for `max_force_per_wheel`; the combined optimizer uses `[800, 1500]`. Override via `PULLOFF_SPACE` in `combined_config.py` if needed.

| Dim | Range | Prior |
|---|---|---|
| `sliding_friction` | [0.01, 2.0] | log-uniform |
| `torsional_friction` | [1e-6, 10.0] | log-uniform |
| `rolling_friction` | [1e-6, 1e-3] | log-uniform |
| `solref_timeconst` | [1e-5, 5.0] | log-uniform |
| `solimp_dmin` | [0.001, 0.999] | uniform |
| `solimp_width` | [1e-7, 1.0] | log-uniform |
| `solimp_midpoint` | [0.01, 0.99] | uniform |
| `solimp_power` | [1.0, 10.0] | uniform |
| `noslip_iterations` | [0, 100] | uniform |
| `noslip_tolerance` | [1e-6, 1e-3] | log-uniform |
| `margin` | [0.0, 0.02] | uniform |
| `Br` | [0.5, 2.0] | log-uniform |
| `max_magnetic_distance` | [0.001, 0.2] | log-uniform |
| `max_force_per_wheel` | [800, 1500] | log-uniform |

`solref_dampratio` is fixed at 10.0; `solimp_dmax` is fixed at 0.9999.

---

## Configuration Reference

### `legged_sim/config.py`

| Parameter | Default | Description |
|---|---|---|
| `TIMESTEP` | `0.0005` | Physics timestep (s), 2 kHz |
| `SETTLE_TIME` | `2.0` | Gravity-only settling before IK starts |
| `SIM_DURATION` | `30.0` | Headless run cap |
| `KNEE_BAKE_DEG` / `WRIST_BAKE_DEG` / `EE_BAKE_DEG` | per-leg dicts | Pre-baked joint angles for initial pose |
| `SEQUENCE` | `"f2w"` | Default sequence if none specified via CLI |
| `PARAMS` | dict (hybrid optimizer best) | Active solver / friction / magnet preset |

The default `PARAMS` is the latest combined-optimizer best (floor + wall + pull-off, 800 N min adhesion floor): `Br = 1.88`, `max_magnetic_distance = 57.5 mm`, `max_force_per_wheel = 800.7 N`, `noslip_iterations = 49`. Earlier presets (commented out) are kept in the file for reference.

### `legged_sim/sim.py` constants

| Constant | Value | Description |
|---|---|---|
| `IK_EVERY_N` | `10` | Physics steps between IK solves (5 ms at 2 kHz physics) |
| `IK_DAMPING` | `1e-3` | Levenberg-Marquardt damping in mink |
| `DUAL_REACH_DWELL` | `3.0 s` | FL dwell time in F2W_REACH before FR starts |
| `PID_KP / KI / KD` | `500 / 200 / 30` | PID gains |
| `_CONTACT_LIFT_THRESH` | `8 mm` | EE Z lift above settle baseline = lost floor contact |

### `legged_sim/optimizer/combined_config.py` constants

| Constant | Value | Description |
|---|---|---|
| `N_CALLS` | `300` | Default optimization budget |
| `BATCH_SIZE` | `16` | CMA-ES population size |
| `CMAES_SIGMA0` | `0.3` | Initial step size in normalised coords |
| `OPTIMIZER_RANDOM_STATE` | `42` | CMA-ES seed |
| `FLOOR_Z` / `WALL_X` | `0.0` / `0.5 m` | Absolute scene setpoints used by cost functions |
| `GOAL_FORCE` | `956.37 N` | EML63mm-12 datasheet hold (used by standalone pull-off optimizer) |
| `PULL_RATE_OPT` | `40 N/s` | Ramp rate during optimization (covers full max_force_per_wheel range) |
| `DETACH_HOLD` | `1.0 s` | Sustained drop required before declaring detachment |
| `DETACH_FORCE_FRAC` | `0.5` | Fraction of `max_force_per_wheel` below which a sustained drop is detachment |

---

## Architecture Notes

**Separation of concerns.** `sequences.py` has zero MuJoCo imports. All trajectory logic — quintic interpolation, phase factories, wall-distance hooks, smooth-step targets — lives there as pure Python operating on `IKTarget` and `PhaseContext`. `sim.py` owns the physics loop, IK solver, PID controller, magnetic force application, dual-foot handoff, and visualization. `legged_sim/optimizer/sim_*_sim.py` and `legged_sim/optimizer/viewer.py` reimplement the minimum subset of `sim.py` needed for headless cost evaluation; they share `sequences.py` directly with the live sim, so any phase-logic change applies uniformly.

**Magnetic force application.** Forces are applied each timestep via `apply_mag()`. Per sampling sphere, the closest plate (floor or wall) is found via `mj_geomDistance`, the distance is fed through the dipole-dipole formula, and the contribution is accumulated. Per-magnet totals are clipped to `max_force_per_wheel` before being written to `data.xfrc_applied`. Off-magnets (`off_mids` set) skip the calculation entirely — used during swing phases and during the pull-off ramp.

**Geometry baking.** `bake_joint_angles()` in `config.py` pre-rotates link endpoints via Rodrigues into `robot.xml` from the clean `robot_original.xml` reference. This sets initial joint angles without relying on MuJoCo keyframes. The optimizer additionally rewrites the magnet array sphere positions on every `combined_config` import (placing the array 10 mm from the EE far end), and serialises the bake + load via `fcntl.LOCK_EX` so multiprocessing workers don't race.

**CMA-ES checkpointing.** `combined_optmizer.py` pickles the full `CMAEvolutionStrategy` to `results/<run>/cmaes_state.pkl` after every batch. `--resume-from <dir>` reloads this exactly, so an interrupted 300-eval run can be continued without re-warming the covariance matrix. `--warm-start-from <dir>` is different: it seeds `x0` with the last row of `optimization_bests.csv` but starts a fresh CMA-ES instance.

**IK target lifecycle.** The active `IKTarget` is held on `sim.py` between solves. When `face_axis` is `None`, `IKSolver._ori_target_rot` is invalidated so the next phase that introduces an orientation constraint computes a fresh Rodrigues rotation. When the swing foot's `orientation_cost` changes, `mink`'s `FrameTask` is rebuilt — necessary because some weights are baked at construction time.

**Stance orientation lock.** Wall-planted feet require the orientation lock; a pure position lock leaves the EE rotationally free, and the magnetic + body weight torque slowly rotates the foot off the wall normal. `lock_stance_orientation(data, foot)` snapshots `(R, t)` from `data.xmat` / `data.xpos` and the corresponding stance task is rebuilt with `orientation_cost = 30`.

---

Research focus: contact-aware control, physics-consistent simulation, and locomotion planning for multi-surface magnetic wall-climbing robots.
