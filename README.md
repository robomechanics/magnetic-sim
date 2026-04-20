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
  - [mwc\_sim — Electromagnet Characterization](#mwc_sim--electromagnet-characterization)
  - [legged\_sim — Legged Locomotion & Floor-to-Wall Transition](#legged_sim--legged-locomotion--floor-to-wall-transition)
- [How to Run](#how-to-run)
- [Configuration Reference](#configuration-reference)
- [Architecture Notes](#architecture-notes)
- [Known Issues & Limitations](#known-issues--limitations)

---

## Project Overview

Sally is a quadrupedal robot designed to traverse ferromagnetic surfaces — including the floor-to-wall and wall-to-ceiling transitions required in industrial inspection tasks. Each of her four legs terminates in an electromagnet foot that can adhere to steel surfaces. The simulation stack in this repository was built to:

1. **Characterize magnetic adhesion** using a dipole-dipole force model calibrated against real pull-off and wrench test data.
2. **Identify physics parameters** (friction, contact solver settings, magnetic field strength) via black-box optimization so that simulation behavior matches physical hardware.
3. **Plan and execute legged locomotion** across surface transitions using differential inverse kinematics (IK), PID joint control, and a phase-based sequence runner.

The four modules are largely independent but share common conventions for magnetic force modeling, MuJoCo XML structure, and CMA-ES optimization infrastructure.

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
├── mwc_sim/                      # Phase 3 — EE electromagnet characterization
│   ├── pulloff_config.py         # Pull-off test parameters & CMA-ES search space
│   ├── pulloff_sim.py            # Headless pull-off simulation
│   ├── pulloff_optimizer.py      # CMA-ES optimizer for pull-off force matching
│   ├── pulloff_viewer.py         # Interactive pull-off viewer with force arrows
│   ├── wrench_config.py          # Wrench/peel test parameters & CMA-ES search space
│   ├── wrench_sim.py             # Headless wrench simulation (lever-arm peel)
│   ├── wrench_optimizer.py       # CMA-ES optimizer for wrench moment matching
│   └── wrench_viewer.py          # Interactive wrench viewer
│
└── legged_sim/                   # Phase 4 — Legged locomotion (primary active module)
    ├── mwc_mjcf/
    │   ├── scene.xml             # World scene: floor, wall, lighting, camera
    │   └── robot.xml             # Sally MJCF: 4 legs × (hip_pitch, knee, wrist, EE, em_z)
    ├── config.py                 # Physics constants, bake angles, magnet body names
    ├── sequences.py              # Phase factories and SequenceRunner (no MuJoCo imports)
    ├── sim.py                    # Main simulation entry point: IK, PID, magnetics, viz
    └── viewer.py                 # Crawl gait viewer (legacy wheeled-style viewer)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `mujoco >= 3.0` | Physics engine and viewer |
| `mink` | Differential IK with `FrameTask` / SE3 targets |
| `numpy` | Numerical computing throughout |
| `cma` | CMA-ES black-box optimizer (`mwc_sim` optimizers) |
| `scikit-optimize` | Gaussian-process Bayesian optimizer (`optimizer_sim`) |
| `scipy` | Rotation utilities in `optimizer_sim/viewer.py` |
| `matplotlib` | Pull-off and wrench result plots |
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

> **WSL Users:** If running under Windows Subsystem for Linux, the MuJoCo passive viewer requires a working X11/Wayland display forwarding setup (e.g. VcXsrv or WSLg). Headless modes (`--headless`) work without display configuration.

---

## Module Guide

---

### `sally_sim` — Wheeled Platform Baseline

**Purpose:** The starting point for the project. Loads the original wheeled Sally MJCF (converted from CAD), applies a dipole-dipole magnetic adhesion model at the wheel contact faces, and provides an interactive viewer for manual testing of adhesion behavior and simple motor drive.

**Key file:** `robot_sally_patched.xml`
Defines the full kinematic chain of the four-wheeled omnidirectional Sally platform, including rocker-bogie suspension, pivot joints, wheel actuators, and 24 sampling spheres per wheel for magnetic force evaluation.

**Key file:** `sim_sally_magnet_wall.py`
Interactive simulation that applies magnetic forces from each wheel toward a vertical steel wall geom. Supports keyboard-toggled drive (`s`) and pause (`space`). Force arrows are rendered per-wheel at each timestep.

**Magnetic force model** (shared across all modules):

```
F = (3μ₀m²) / (2π(2d)⁴)
```

where `m = (Br × V) / μ₀` is the magnetic dipole moment, `Br` is the remanence field strength, `V` is the magnet volume, and `d` is the closest-point distance from the sampling sphere to the steel plate geom.

---

### `optimizer_sim` — Bayesian Physics Optimization (Wheeled)

**Purpose:** Identifies physics parameters for the wheeled Sally simulation — friction coefficients, solver settings, control gains, and magnetic parameters — by running a Gaussian-process Bayesian optimization loop against three behavioral targets: hold (zero slip), drive sideways, and drive upward.

**Key files:**

- `config.py` — Defines the three operating modes, their cost functions, and the 16-dimensional scikit-optimize search space.
- `sim_optimizer.py` — Headless MuJoCo simulation that accepts a parameter dict, runs for the configured duration, and returns a trajectory. Instability is caught and penalized.
- `tune_params.py` — Entry point. Runs `gp_minimize` over the search space, collects per-call results, saves a ranked CSV, and launches the viewer with the best-found parameters.
- `viewer.py` — Replays any result from `optimization_results_<mode>.csv` with full visualization, including magnetic force arrows, robot heading arrow, and a post-simulation performance summary.

**Cost functions:**

| Mode | Cost |
|---|---|
| `hold` | Total 3D displacement after settle (minimize slip) |
| `drive_sideways` | Y-velocity error + off-axis (X, Z) drift penalty |
| `drive_up` | Z-velocity error + off-axis (X, Y) drift penalty |

**Running optimization:**

```bash
cd optimizer_sim
python tune_params.py --mode hold
python tune_params.py --mode drive_sideways
python tune_params.py --mode drive_up
```

**Replaying a result:**

```bash
python viewer.py --rank 1 --mode drive_up
```

**Optimized parameter presets** from prior runs are stored in `config.py` under `DEFAULT_PARAMS` and can be used as warm-start seeds.

---

### `mwc_sim` — Electromagnet Characterization

**Purpose:** Characterizes the EML63mm-24 electromagnet used on Sally's end-effectors by matching two physical test modes — vertical pull-off and lever-arm wrench/peel — against simulation using CMA-ES optimization over 14 physics parameters.

This module was used to identify the `Br`, `max_magnetic_distance`, and contact solver settings that best reproduce the measured 956 N holding force and the peel moment derived from a lever-arm geometry.

**Two test modes:**

#### Pull-Off Test (`pulloff_*`)

A magnet body is placed on a steel plate, allowed to settle under gravity and magnetic attraction, then subjected to a linearly ramping upward force. The pull-off force is defined as the peak applied force at detachment. The optimizer minimizes the normalized shortfall from the target pull-off force.

```bash
cd mwc_sim
python pulloff_sim.py                    # headless + plot
python pulloff_sim.py --pull-rate 50     # custom ramp rate
python pulloff_optimizer.py              # CMA-ES optimization (200 evals, batch 20)
python pulloff_optimizer.py --resume-from results/<run_dir>
```

#### Wrench / Peel Test (`wrench_*`)

A horizontal force is applied at the tip of a rigid lever arm attached to the magnet, generating both a shear force and a peeling moment at the interface. Separate toggles (`APPLY_FORCE`, `APPLY_MOMENT` in `wrench_config.py`) allow testing pure shear, pure peel, or the full combined wrench.

The cost function (70% shortfall from target, 30% XY drift before detachment) prevents the optimizer from finding parameters that hold in pull-off but allow sliding under shear.

```bash
python wrench_sim.py
python wrench_optimizer.py
python wrench_optimizer.py --warm-start-from results/<run_dir>
```

**Optimized presets** are stored in `wrench_config.py` under `PARAM_PRESETS`:

| Preset | Description |
|---|---|
| `hold` | Minimizes slip under static load |
| `drive_sideways` | Matches sideways drive behavior |
| `drive_up` | Matches upward drive behavior |
| `pull_off` | Matches vertical pull-off detachment force |
| `combined` | Balanced preset across multiple modes |

**Resuming and warm-starting:**

All optimizers support checkpoint resumption and warm-starting from a prior best CSV:

```bash
python wrench_optimizer.py --resume-from results/20250101T120000_run
python wrench_optimizer.py --warm-start-from results/20250101T120000_run
```

---

### `legged_sim` — Legged Locomotion & Floor-to-Wall Transition

**Purpose:** The primary active research module. Implements IK-controlled legged locomotion for the four-legged Sally design, including a five-phase floor-to-wall foot placement sequence. This module contains the most recent and complete simulation architecture.

#### Robot Design (`mwc_mjcf/robot.xml`)

Sally has four legs (FL, FR, BL, BR), each with four joints:

| Joint | Type | Range | Description |
|---|---|---|---|
| `hip_pitch_<leg>` | Hinge (Z-axis) | ±45° | Horizontal swing of entire leg |
| `knee_<leg>` | Hinge (diagonal XY) | ±90° | First link bend |
| `wrist_<leg>` | Hinge (diagonal XY) | ±90° | Second link bend |
| `ee_<leg>` | Hinge | ±90° | End-effector pitch |
| `ee2_<leg>` | Hinge (passive, spring+damper) | ±90° | EE compliance joint |
| `em_z_<leg>` | Hinge (passive) | ±180° | Electromagnet yaw compliance |

The electromagnet body on each foot contains 8 sampling spheres arranged in a ring at radius 22 mm, used for the dipole-dipole force calculation. All legs are colored distinctly for debugging: FL=Red, FR=Green, BL=Blue, BR=Black.

The scene defines a 3 m × 1 m steel floor plate and a 1 m × 1 m steel wall plate with an inner face at X = 0.500 m, forming a clean 90° floor-to-wall corner.

#### IK Architecture (`sim.py` — `IKSolver`)

Whole-body differential IK is solved using `mink` with one `FrameTask` per foot plus a `FrameTask` for the main frame body and a `PostureTask` for joint-limit regularization.

- **Swing foot** tracks a moving SE3 target computed by the active sequence phase.
- **Stance feet** are locked to their settle-time positions. Once a foot lands on the wall, its full SE3 (position + orientation) is locked using `lock_stance_orientation()` to prevent rotational drift.
- **Orientation control** during f2w phases uses `face_axis` to drive EE local −Y toward the wall normal. The rotation target is computed via Rodrigues' formula at phase entry and held fixed throughout the phase.
- IK is solved every `IK_EVERY_N = 10` physics steps (i.e., every 20 ms at `dt=0.002`) to reduce compute load.

#### PID Control (`PIDController`)

Joint-space PID converts IK position targets to motor torques at every physics step:

```
τ = Kp × e + Ki × ∫e dt + Kd × ė
Kp=500, Ki=200, Kd=30, I_clamp=100
```

#### Sequence Architecture (`sequences.py`)

All trajectory logic is isolated in `sequences.py`, which has no MuJoCo imports. It defines pure Python `Phase` objects with callable `target_pos` and `face_axis` fields, and a `SequenceRunner` that drives a list of phases by wall-clock simulation time.

**Available sequences:**

| Sequence | Description |
|---|---|
| `orient` | Lift → swing −45° → hold with EE facing −X |
| `f2w` | Full FL floor-to-wall: lift → swing → orient → measure → reach |
| `f2w_fr` | FR mirror of `f2w` (swing +45°), used internally for dual-foot mode |

**Floor-to-wall phases (f2w):**

1. **LIFT** — Raise EE by 10 cm; disables swing foot magnet on entry.
2. **SWING** — Arc EE around the hip pivot by −45° (quintic interpolation).
3. **F2W_ORIENT** — Translate EE to a raised waypoint above the swing landing and rotate to face the wall (`face_axis = −wall_normal`).
4. **F2W_MEASURE** — Zero-duration phase. Computes EE→wall distance analytically via dot-product projection against the known wall face position. Writes `f2w_reach_target` into the runner context with foot standoff and safety clearance applied.
5. **F2W_REACH** — Interpolates EE from current position to `f2w_reach_target`; enables the foot magnet when motion completes. Phase is `unbounded` (holds forever once reached).

**Dual-foot mode (`--sequence f2w`):**

When the `f2w` sequence is selected, `sim.py` runs FL first. Once FL's F2W_REACH phase has dwelt for `DUAL_REACH_DWELL = 3.0 s`, FL is declared planted: its SE3 is locked as a stance foot, and a fresh `SequenceRunner` is started for FR using the `f2w_fr` sub-sequence. This enables sequential floor-to-wall placement for both front feet.

#### Running the Simulation

```bash
cd legged_sim

# GUI — default orient sequence (FL swing, hold with EE facing −X)
python sim.py

# GUI — floor-to-wall sequence (FL foot → wall)
python sim.py --sequence f2w

# GUI — dual foot floor-to-wall (FL then FR)
python sim.py --sequence f2w   # dual mode is automatic when f2w selected

# Headless — useful for batch testing
python sim.py --headless --sequence f2w --duration 25.0

# Disable IK (free-fall settle only, for physics debugging)
python sim.py --no-ik

# Enable magnetic forces
python sim.py --sequence f2w --magnets
```

**Viewer controls:**

- `Space` — toggle pause / run

**Telemetry (printed every 1 s):**

```
t= 8.2  [████████░░░░] FL/F2W_RE  72.3%  pos_err= 4.1mm  (Δx=+2.1 Δy=-3.2 Δz=+1.8)  ang_to_goal=  3.4°  stance=BL/1.2mm  mag=ON
```

---

## Configuration Reference

### `legged_sim/config.py`

| Parameter | Default | Description |
|---|---|---|
| `TIMESTEP` | `0.002` | Physics timestep (s) |
| `JOINT_DAMPING` | `1.0` | Default joint damping (Nm·s/rad) |
| `KNEE_BAKE_DEG` | per-leg dict | Pre-baked knee angles for initial pose |
| `WRIST_BAKE_DEG` | per-leg dict | Pre-baked wrist angles |
| `EE_BAKE_DEG` | per-leg dict | Pre-baked EE angles |
| `SEQUENCE` | `"orient"` | Default sequence if none specified via CLI |
| `PARAMS` | dict | Contact solver settings applied at startup |

### `legged_sim/sim.py` constants

| Constant | Value | Description |
|---|---|---|
| `SETTLE_TIME` | `2.0 s` | Duration of gravity-only settling before IK starts |
| `IK_EVERY_N` | `10` | Physics steps between IK solves |
| `IK_DAMPING` | `1e-3` | Levenberg-Marquardt damping in mink |
| `DUAL_REACH_DWELL` | `3.0 s` | FL dwell time in F2W_REACH before FR starts |
| `PID_KP/KI/KD` | `500/200/30` | PID gains |

---

## Architecture Notes

**Separation of concerns:** `sequences.py` is deliberately free of MuJoCo imports. All trajectory logic (quintic interpolation, phase factories, wall distance computation hooks) lives there. `sim.py` owns the physics loop, IK solver, PID controller, magnetic force application, and visualization. This makes trajectory logic unit-testable without a running simulation.

**Magnetic force application:** Forces are applied at each timestep via `apply_mag()` using the dipole-dipole model evaluated at each sampling sphere. Per-magnet force vectors are clipped to `max_force_per_wheel` before application to prevent unphysical adhesion values.

**Geometry baking:** `bake_joint_angles()` in `config.py` pre-rotates MJCF link endpoints via Rodrigues rotation into `robot.xml` at startup, setting initial joint angles without relying on MuJoCo keyframes. This ensures the robot spawns in a stable crouching pose rather than the zero-angle configuration.

**CMA-ES checkpointing:** Both `pulloff_optimizer.py` and `wrench_optimizer.py` checkpoint the full `CMAEvolutionStrategy` object to `results/<run>/cmaes_state.pkl` after every batch. Runs can be resumed exactly from where they left off using `--resume-from`.

---

## Known Issues & Limitations

- **Wall reachability check:** `sim.py` will call `sys.exit(1)` if the wall ray cast indicates the target contact point is outside the FK-computed maximum reach of the swing foot. Move the robot spawn position closer to the wall if this occurs.
- **IK orientation locking:** The `_ori_target_rot` cache in `IKSolver` is reset between FL and FR phases (`ik._ori_target_rot = None` in `_start_runner_for`). If you add new sequences, ensure this reset is called when switching feet.
- **`mwc_sim` XML assets:** The wrench and pull-off scenes reference `mwc_mjcf/` asset paths. Ensure you run these scripts from within the `mwc_sim/` directory or update `SCENE_XML` in the respective config files.
- **`franka_test` submodule removed:** The `franka_test/` directory (which contained a `mujoco_menagerie` git submodule) has been removed from the main branch. If you need it, check the `MWC_steven` branch.
- **Headless on WSL:** The passive viewer (`mujoco.viewer.launch_passive`) requires display forwarding under WSL. Use `--headless` for display-free environments.

---

## Authors

Nathan S. — CMU Robomechanics Lab, under Prof. Aaron M. Johnson.

Research focus: contact-aware control, physics-consistent simulation, and locomotion planning for multi-surface magnetic wall-climbing robots.
