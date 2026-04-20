# Sally Magnetic Adhesion Simulation Suite

MuJoCo-based simulations for characterizing the magnetic adhesion of Sally — a four-wheel omnidirectional wall-climbing robot. Three test modes are provided: a **pull-off test** (normal detachment force), a **wrench/peel test** (combined shear force + bending moment), and a **combined optimizer** that tunes parameters against both simultaneously.

---

## Overview

Magnetic forces are not natively supported in MuJoCo, so this suite implements a **dipole-dipole force model** applied via an array of non-colliding sampling spheres embedded inside the physical magnet body. Each sphere computes its closest-point distance to the steel plate and applies a proportional attractive force along that direction. Together they approximate the distributed magnetic adhesion of a real permanent magnet.

```
F = (3 μ₀ m²) / (2π (2d)⁴)       m = Br · V_magnet / μ₀
```

Where `d` is the closest-point distance from each sampling sphere to the plate surface, `Br` is the magnet remanence (Tesla), and `V_magnet` is the physical magnet volume. The resulting force vector is applied to the magnet body via `xfrc_applied`.

All three pipelines share the same 14-dimensional parameter space and CMA-ES optimization framework. Each has its own config file as the single source of truth for its parameters, cost function, and presets. The combined optimizer imports from both configs to co-optimize a single parameter set against both test modes.

---

## File Structure

```
.
├── XML/
│   └── flat_plate_pull.xml          # Pull-off scene: magnet + plate MJCF
├── mwc_mjcf/
│   └── scene.xml                    # Wrench scene: top-level MJCF with includes
│
├── ── Pull-Off Pipeline ─────────────────────────────────────────────────────
├── pulloff_config.py                # Single source of truth for pull-off params
├── pulloff_sim.py                   # Headless pull-off simulation + plotting
├── pulloff_viewer.py                # Interactive viewer for pull-off test
├── pulloff_optimizer.py             # CMA-ES optimizer for pull-off sim
│
├── ── Wrench Pipeline ───────────────────────────────────────────────────────
├── wrench_config.py                 # Single source of truth for wrench params
├── wrench_sim.py                    # Headless wrench/peel simulation + plotting
├── wrench_viewer.py                 # Interactive viewer for wrench/peel test
├── wrench_optimizer.py              # CMA-ES optimizer for wrench sim
│
├── ── Combined Pipeline ─────────────────────────────────────────────────────
├── combined_optimizer.py            # CMA-ES optimizer over both sims jointly
│
└── results/
    ├── pulloff_results.png          # Output plot from pulloff_sim.py
    ├── wrench_results.png           # Output plot from wrench_sim.py
    └── <timestamp>_<suffix>/        # Per-run optimizer output directories
        ├── optimization_results.csv
        ├── optimization_bests.csv
        ├── cmaes_state.pkl
        └── *_config.py              # Config snapshot(s) copied at run start
```

---

## XML Scene Files

### `XML/flat_plate_pull.xml` — Pull-Off Scene

This is the primary MJCF scene for the pull-off test. It defines the physical world: the steel plate, the magnet body, the sampling spheres, and the bracket mesh.

**`<option>`** sets `timestep="0.001"` with an implicit integrator and standard gravity. This timestep is small enough to resolve the fast contact dynamics during magnetic snap-on/off.

**`<default>`** sets `condim="6"` globally, enabling full 3D friction (normal + two tangential + torsional + two rolling) on all geoms by default.

**`<worldbody>`** contains two bodies:

**`plate` body** — A static 500×500×10 mm steel box (`plate_geom`). Its top surface sits at z=0. It is the collision target for all `mj_geomDistance` calls in the magnetic force computation. Friction and solver parameters on this geom are overwritten at runtime by `setup_model()`.

**`magnet` body** — The full magnet + bracket assembly, positioned so its bottom face is flush with the plate top at z=0 (body origin at z=0.017 m, cylinder offset −0.0025 m). It carries a `<freejoint name="magnet_free"/>` giving it full 6-DOF motion. It contains:

- **`magnet_cyl`** — Cylinder of radius 31.5 mm, half-height 14.5 mm, density 7850 kg/m³ (steel). Rendered red.
- **`bracket`** — STL mesh (`1102_PF_Magnet_Bracket.STL`) at 1:1000 scale, density 2700 kg/m³ (aluminum). Rendered semi-transparent blue.
- **Sampling spheres (8 spheres, bottom ring)** — Non-colliding (`contype="0" conaffinity="0"`, `mass=1e-6 kg`) in a ring of radius 28.5 mm at z=−17 mm. Their world positions are used to query `mj_geomDistance` each timestep and the computed attractive force is applied to the magnet body.

  > Two additional rings (middle at z=0, top at z=+10.5 mm, 8 spheres each) are defined but commented out. Uncommenting enables a 3-layer, 24-point force field.

---

### `mwc_mjcf/scene.xml` — Wrench Test Top-Level Scene

This is the top-level MJCF file for the wrench/peel test. It is a thin wrapper that configures global visuals and includes the magnet+plate fragment.

**`<compiler>`** sets `angle="radian"` and `meshdir="assets"`.

**`<option>`** sets gravity to `0 0 -9.81` with an implicit integrator.

**`<visual>`** configures headlight and fog range (`znear=0.01`, `zfar=10.0`). Blue-to-white gradient skybox in `<asset>`.

**`<statistic center="0 0 0.3" extent="1.0"/>`** centers the viewer camera at startup.

**`<include file="flat_plate_pull.xml"/>`** pulls in the magnet + plate world body definitions. The wrench scene shares the same physical geometry as the pull-off scene; all behavioral differences are in how Python applies forces at runtime.

> `wrench_sim.py`'s `setup_model()` looks for body name `1103___pp___aws_pem_215lbs__eml63mm_24` and site name `stick_tip`. A `ValueError` is raised at startup if either is missing.

---

---

# Pull-Off Pipeline

Normal detachment force characterization. Pull force is applied vertically at the magnet COM and ramps until the magnet lifts off the plate. Detachment is detected by vertical COM displacement exceeding a threshold.

---

## `pulloff_config.py` — Pull-Off Single Source of Truth

**Purpose:** Centralizes all pull-off simulation parameters, CMA-ES settings, cost function, parameter space, and preset configurations. `pulloff_sim.py`, `pulloff_viewer.py`, and `pulloff_optimizer.py` all import from here.

**Key constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `TIMESTEP` | 0.0005 s | Simulation timestep (2000 Hz) |
| `PULL_RATE` | 20.0 N/s | Default ramp rate for single-run sim |
| `PULL_RATE_OPT` | 40.0 N/s | Fixed ramp rate used during optimization |
| `SETTLE_TIME` | 2.0 s | Two-phase settle (gravity 0→1 s, magnetic 1→2 s) |
| `SIM_DURATION` | 30.0 s | Hard stop |
| `DETACH_DIST` | 10.0 mm | Vertical COM displacement threshold for detachment |
| `DETACH_HOLD` | 1.0 s | Displacement must stay above threshold this long |
| `GOAL_FORCE` | 956.37 N | Target pull-off force for optimization cost |
| `N_CALLS` | 200 | Total candidate evaluations per optimizer run |
| `BATCH_SIZE` | 20 | CMA-ES population size |
| `CMAES_SIGMA0` | 0.3 | Initial CMA-ES step size |
| `ACTIVE_PRESET` | `'pull_off'` | Which `PARAM_PRESETS` entry to load as `PARAMS` |

**Cost function — `calculate_cost(pulloff_force) → dict`:**

Cost is purely the normalized shortfall from `GOAL_FORCE`. No XY drift term — pull-off is a pure vertical test.
- **No engagement** (`pulloff_force == 0`): sentinel cost `9999.0`.
- **Underperformance**: `shortfall = max(0, (GOAL - achieved) / GOAL)`.
- **Goal met** (`achieved ≥ GOAL`): zero cost.

**Parameter search space (14 dims):** Identical to the wrench config — see the wrench pipeline section for the full table. `solimp_dmax` fixed at 0.9999; `solref_dampratio` fixed at 10.0.

**`point_to_params(point) → dict`:** Converts a raw CMA-ES point to the `PARAMS` dict consumed by `setup_model()`. Identical structure to `wrench_config.point_to_params`.

**Preset configurations (`PARAM_PRESETS`):**

| Preset | `Br` (T) | `max_magnetic_distance` (m) | `max_force_per_wheel` (N) |
|--------|----------|-----------------------------|---------------------------|
| `hold` | 1.332 | 0.01141 | 139.3 |
| `drive_sideways` | 1.476 | 0.02974 | 264.2 |
| `drive_up` | 1.599 | 0.01839 | 205.3 |
| `pull_off` | 1.327 | 0.07062 | 917.2 |

---

## `pulloff_sim.py` — Headless Pull-Off Simulation

**Purpose:** Runs a complete pull-off test without a viewer, detects detachment by vertical COM displacement, saves a results plot, then launches the interactive viewer.

**Critical functions:**

`mag_force(dist, Br) → float`
Dipole-dipole attractive force (N) for a single sampling sphere. Formula: `F = (3μ₀m²) / (2π(2d)⁴)` where `m = Br·V/μ₀`.

`setup_model(params) → (model, data, plate_id, magnet_id, sphere_gids)`
Loads `SCENE_XML`, applies `params` overrides (friction, solref, solimp, noslip, margin), resolves `plate_geom` and magnet body IDs. Builds `sphere_gids` by filtering for `mjGEOM_SPHERE` geoms on the magnet body. Takes `params` as an explicit argument — no fallback — so the optimizer can inject arbitrary candidates.

`apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto, params) → float`
Iterates over sampling spheres, computes dipole force, clips total assembly force vector against `max_force_per_wheel`, accumulates into `xfrc_applied`. Returns `fvec_total[2]` (Fz, negative = attraction toward plate).

`run_headless(pull_rate, params) → (records, pulloff_force)`
Three-phase loop:
- **Phase 1** (`t < SETTLE_TIME/2`): gravity only — magnet falls onto plate.
- **Phase 2** (`t < SETTLE_TIME`): magnetic forces — magnet snaps and settles.
- **Phase 3** (ramp): linearly increasing upward force at magnet COM (`f_pull = pull_rate × t_ramp`). Detachment declared when `z_disp > DETACH_DIST` sustained for `DETACH_HOLD` seconds.

Both `pull_rate` and `params` are explicit arguments — no fallback.

`plot(records, pulloff_force, pull_rate, params)`
Two-panel figure: (left) Force vs. Time; (right) Force vs. Displacement zoomed to the detach region. Saved to `results/pulloff_results.png`.

**Entry point:** `parse args → run_headless() → pulloff_viewer.run_viewer() → plot()`

---

## `pulloff_viewer.py` — Interactive Pull-Off Viewer

**Purpose:** Re-runs the pull-off simulation in real time inside the MuJoCo passive viewer with per-sphere force arrow overlays.

**Arrow overlays:**
- **Blue** — magnetic attraction force at each sampling sphere, scaled by `MAG_ARROW_SCALE`.
- **Red** — applied upward pull force at magnet COM, scaled by `FORCE_ARROW_SCALE`.

**Key constants (from `pulloff_config.py`):**

| Constant | Default | Description |
|----------|---------|-------------|
| `REAL_TIME_FACTOR` | 2.0 | Wall-clock speed relative to sim time |
| `ARROW_RADIUS` | 0.004 m | Visual shaft radius of force arrows |
| `MAG_ARROW_SCALE` | 0.001 m/N | Blue magnetic arrow length scaling |
| `FORCE_ARROW_SCALE` | 0.005 m/10N | Red pull arrow length scaling |
| `TELEMETRY_INTERVAL` | 0.1 s | Terminal print interval |

`run_viewer(pull_rate)`
Mirrors the headless sim: magnetic forces enabled after `SETTLE_TIME/2`; pull force and displacement tracking start at `SETTLE_TIME`. Detachment detected by `z_disp > DETACH_DIST` for `DETACH_HOLD` seconds. Telemetry prints phase, `F_pull`, `F_mag`, and `z_disp` each interval.

---

## `pulloff_optimizer.py` — CMA-ES Pull-Off Optimizer

**Purpose:** Searches the 14-dimensional parameter space to maximize pull-off force. Cost is purely the normalized shortfall from `GOAL_FORCE` (no drift term).

**Usage:**
```bash
python pulloff_optimizer.py
python pulloff_optimizer.py --n-calls 500 --suffix my_run
python pulloff_optimizer.py --resume-from results/20250101T120000_my_run
python pulloff_optimizer.py --warm-start-from results/20250101T120000_my_run
```

**CSV columns (`optimization_results.csv`):** `id`, `cost`, `elapsed_min`, `pulloff_force`, `detach_cost`, `achieved`, `goal`, `shortfall`, + 14 parameter columns.

**Worker — `_evaluate_one_candidate(args)`:**
Imports `pulloff_sim.run_headless` and `pulloff_config.calculate_cost` inside the worker (spawn-safe). On crash, returns `pulloff_force=0` (sentinel cost). Returns `(point_index, cost_data, wall_time)`.

**Post-optimization replay:**
Best params injected into `pulloff_config.PARAMS` at runtime, confirmed via `run_headless`, then displayed in `pulloff_viewer.run_viewer`.

Config snapshot saved: `pulloff_config.py` copied into run directory at start.

---

---

# Wrench Pipeline

Combined shear force + bending moment characterization. Horizontal force is applied at the tip of a rigid lever arm attached to the magnet, generating a contact wrench at the magnet-plate interface. Detachment is detected when magnetic force drops to near zero.

---

## `wrench_config.py` — Wrench Single Source of Truth

**Purpose:** Centralizes all wrench simulation parameters, optimization settings, cost function, parameter space, and preset configurations. `wrench_sim.py`, `wrench_viewer.py`, `wrench_optimizer.py`, and `combined_optimizer.py` all import from here.

**Key constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `TIMESTEP` | 0.0005 s | Simulation timestep (2000 Hz) |
| `PULL_RATE` | 40.0 N/s | Default ramp rate for single-run sim |
| `PULL_RATE_OPT` | 40.0 N/s | Fixed ramp rate used during optimization |
| `SETTLE_TIME` | 1.0 s | Two-phase settle (gravity 0→0.5 s, magnetic 0.5→1 s) |
| `SIM_DURATION` | 40.0 s | Hard stop |
| `DETACH_HOLD` | 0.5 s | Time `f_mag` must stay below threshold to confirm detachment |
| `DETACH_THRESHOLD` | 0.01 N | Magnetic force below this counts as detached |
| `APPLY_FORCE` | `True` | Toggle horizontal shear force at stick tip |
| `APPLY_MOMENT` | `False` | Toggle resulting torque at magnet COM |
| `GOAL_FORCE` | 956.37 N | Target detachment force for optimization cost |
| `PEEL_R` | 0.057 m | Moment arm radius |
| `GOAL_WRENCH` | `GOAL_FORCE × PEEL_R` | Target detachment moment (N·m) |
| `N_CALLS` | 200 | Total candidate evaluations per optimizer run |
| `BATCH_SIZE` | 20 | CMA-ES population size |
| `CMAES_SIGMA0` | 0.3 | Initial CMA-ES step size |
| `ACTIVE_PRESET` | `'pull_off'` | Which `PARAM_PRESETS` entry to load as `PARAMS` |

**Cost weights:**

| Weight | Value | Penalty |
|--------|-------|---------|
| `COST_WEIGHT_DETACH` | 0.30 | Shortfall from `GOAL_FORCE` or `GOAL_WRENCH` |
| `COST_WEIGHT_XY_DRIFT` | 0.70 | Normalized XY displacement of magnet COM before detachment |

**Cost function — `calculate_cost(detach_force, detach_moment, xy_drift) → dict`:**

Three cases:
- **No engagement** (both zero): sentinel cost `9999.0` — magnet never attached.
- **Underperformance** (`achieved < goal`): one-sided shortfall `max(0, (goal − achieved) / goal)`.
- **Goal met** (`achieved ≥ goal`): zero shortfall; XY drift penalty only.

XY drift normalized against `DRIFT_REFERENCE_M = 0.01 m`, clamped to `[0, 1]`. Active metric selected by `APPLY_FORCE` / `APPLY_MOMENT` toggles.

**Parameter search space (14 dims):**

| Dimension | Range | Prior |
|-----------|-------|-------|
| `sliding_friction` | [0.01, 2.0] | log-uniform |
| `torsional_friction` | [1e-6, 10.0] | log-uniform |
| `rolling_friction` | [1e-6, 1e-3] | log-uniform |
| `solref_timeconst` | [1e-5, 1.0] | log-uniform |
| `solimp_dmin` | [0.001, 0.999] | uniform |
| `solimp_width` | [1e-7, 1.0] | log-uniform |
| `solimp_midpoint` | [0.01, 0.99] | uniform |
| `solimp_power` | [2.0, 7.0] | uniform |
| `noslip_iterations` | [0, 60] | uniform |
| `noslip_tolerance` | [1e-6, 1e-3] | log-uniform |
| `margin` | [0.0, 0.005] | uniform |
| `Br` | [0.5, 2.0] | log-uniform |
| `max_magnetic_distance` | [0.012, 0.1] | log-uniform |
| `max_force_per_wheel` | [300, 1200] | log-uniform |

Fixed: `solimp_dmax = 0.9999`, `solref_dampratio = 10.0`.

**Preset configurations (`PARAM_PRESETS`):**

| Preset | `Br` (T) | `max_magnetic_distance` (m) | `max_force_per_wheel` (N) | Use case |
|--------|----------|-----------------------------|---------------------------|----------|
| `hold` | 1.332 | 0.01141 | 139.3 | Stationary hold |
| `drive_sideways` | 1.476 | 0.02974 | 264.2 | Lateral locomotion |
| `drive_up` | 1.599 | 0.01839 | 205.3 | Climbing upward |
| `pull_off` | 1.327 | 0.07062 | 917.2 | Normal detachment |
| `combined` | 1.206 | 0.01451 | 912.3 | Combined force + moment |

---

## `wrench_sim.py` — Headless Wrench/Peel Simulation

**Purpose:** Runs a wrench/peel test headless, applies a horizontal force at the tip of a rigid lever arm, records force/moment/magnetic attraction per step, detects peel-off, saves a plot, then launches the viewer.

**Critical functions:**

`setup_model(params) → (model, data, plate_id, magnet_id, sphere_gids, tip_site_id)`
Loads `SCENE_XML`, applies `params` overrides, resolves IDs for `plate_geom`, the magnet body, and the `stick_tip` site. Returns `tip_site_id` — its world position is queried each step to compute the moment arm. Raises `ValueError` if any named element is missing.

`apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto, params) → float`
Same structure as in `pulloff_sim`. Clips total assembly force magnitude against `max_force_per_wheel` (not per-sphere). Returns total force magnitude — drops to near zero at detachment.

`apply_wrench_force(model, data, magnet_id, f_pull, tip_site_id)`
Applies horizontal force in +X at the stick tip. Controlled by `APPLY_FORCE` / `APPLY_MOMENT` toggles:
- `force_vec = [f_pull, 0, 0]`
- `r = [0, 0, PEEL_R]` — fixed moment arm.
- `moment = cross(r, force_vec)` — torque about magnet center.
- Accumulates into `xfrc_applied[:3]` and/or `[3:]` depending on toggles.

`run_headless(pull_rate, params) → (records, detach_force, detach_moment, xy_drift)`
Three-phase loop:
- **Phase 1** (`t < SETTLE_TIME/2`): gravity only.
- **Phase 2** (`t < SETTLE_TIME`): magnetic settle.
- **Phase 3** (ramp): `apply_mag` + `apply_wrench_force` each step. XY drift accumulated from settle position. Detachment declared when `f_mag < DETACH_THRESHOLD` sustained for `DETACH_HOLD` seconds.

Both `pull_rate` and `params` are explicit arguments — no fallback.

`plot(records, detach_force, pull_rate, params)`
Two-panel figure: (left) Force & Moment vs. Time; (right) Magnetic Force vs. Applied Force. Saved to `results/wrench_results.png`.

**Entry point:** `parse args → run_headless() → wrench_viewer.run_viewer() → plot()`

---

## `wrench_viewer.py` — Interactive Wrench/Peel Viewer

**Purpose:** Re-runs the wrench simulation in real time with force arrow overlays. Called automatically by `wrench_sim.py` and `wrench_optimizer.py`, or standalone.

**Arrow overlays:**
- **Blue** — magnetic attraction at each sampling sphere.
- **Red** — applied horizontal force at stick tip (only drawn if `APPLY_FORCE = True`).
- **Green** — torque axis at stick tip (only drawn if `APPLY_MOMENT = True`).

Labels and telemetry reflect the active toggles truthfully.

**Key constants (from `wrench_config.py`):**
- `REAL_TIME_FACTOR = 0.8` — 20% slow-motion to observe peel dynamics.

`run_viewer(pull_rate)`
Mirrors the headless sim. Magnetic forces enabled after `SETTLE_TIME/2`. Pull force arrow drawn at `data.site_xpos[tip_site_id]`. Telemetry prints phase, `F_pull`, `tau`, and `F_mag` each `TELEMETRY_INTERVAL` seconds.

---

## `wrench_optimizer.py` — CMA-ES Wrench Optimizer

**Purpose:** Searches the 14-dimensional parameter space to maximize detachment force (or moment) while minimizing XY drift.

**Usage:**
```bash
python wrench_optimizer.py
python wrench_optimizer.py --n-calls 500 --suffix my_run
python wrench_optimizer.py --resume-from results/20250101T120000_my_run
python wrench_optimizer.py --warm-start-from results/20250101T120000_my_run
```

**CSV columns (`optimization_results.csv`):** `id`, `cost`, `elapsed_min`, `detach_force`, `detach_moment`, `xy_drift`, `detach_cost`, `drift_cost`, `achieved`, `goal`, `shortfall`, + 14 parameter columns.

**Worker — `_evaluate_one_candidate(args)`:**
Imports `wrench_sim.run_headless` and `wrench_config.calculate_cost` inside the worker (spawn-safe). On crash, returns zero force/moment/drift (sentinel cost). Returns `(point_index, cost_data, wall_time)`.

**Post-optimization replay:**
Best params injected into `wrench_config.PARAMS` at runtime, confirmed headless, then displayed in `wrench_viewer.run_viewer`.

Config snapshot saved: `wrench_config.py` copied into run directory at start.

---

---

# Combined Pipeline

Co-optimizes a single 14-dimensional parameter set against both the pull-off and wrench simulations simultaneously. Each candidate is evaluated on both sims sequentially in each worker process; the combined cost is a weighted sum of the individual costs.

---

## `combined_optimizer.py` — CMA-ES Combined Optimizer

**Purpose:** Finds parameters that simultaneously maximize pull-off force and wrench detachment performance, using the shared 14-dim space from `wrench_config` and both cost functions from their respective configs.

**Usage:**
```bash
python combined_optimizer.py
python combined_optimizer.py --n-calls 500 --suffix my_run
python combined_optimizer.py --resume-from results/20250101T120000_my_run
python combined_optimizer.py --warm-start-from results/20250101T120000_my_run
```

**Combined cost:**

```
total_cost = COST_WEIGHT_PULLOFF × pulloff_cost
           + COST_WEIGHT_WRENCH  × wrench_cost
```

| Weight | Default | Description |
|--------|---------|-------------|
| `COST_WEIGHT_PULLOFF` | 0.5 | Weight on pull-off shortfall cost |
| `COST_WEIGHT_WRENCH` | 0.5 | Weight on wrench detach + drift cost |

If either sim returns the failure sentinel (`≥ 9999`), the combined cost is set to `9999` directly — no partial credit.

**Pull rates:**

| Constant | Default | Description |
|----------|---------|-------------|
| `PULL_RATE_PULLOFF` | 40.0 N/s | Ramp rate used for pull-off sim during optimization |
| `PULL_RATE_WRENCH` | 40.0 N/s | Ramp rate used for wrench sim during optimization |

**Imports:** Parameter space and `point_to_params` from `wrench_config` (identical to `pulloff_config`). Optimizer settings (`N_CALLS`, `BATCH_SIZE`, `CMAES_SIGMA0`, `OPTIMIZER_RANDOM_STATE`) from `wrench_config`. Both `calculate_cost` functions imported with aliases (`wrench_calculate_cost`, `pulloff_calculate_cost`).

**Worker — `_evaluate_one_candidate(args)`:**
Receives `(point_index, point, pull_rate_pulloff, pull_rate_wrench)`. Runs pull-off sim first, then wrench sim sequentially within the same worker. Both use the same `params = point_to_params(point)`. Each crash caught independently. Combined cost assembled from both result dicts.

**CSV columns (`optimization_results.csv`):** `id`, `cost`, `elapsed_min`, `pulloff_force`, `pulloff_shortfall`, `pulloff_cost`, `wrench_achieved`, `wrench_shortfall`, `wrench_detach_cost`, `wrench_drift_cost`, `xy_drift`, + 14 parameter columns.

**Post-optimization replay:**
Best params injected into both `pulloff_config.PARAMS` and `wrench_config.PARAMS` at runtime. Headless confirmation for both sims. Then `pulloff_viewer.run_viewer` launches, followed by `wrench_viewer.run_viewer`.

Config snapshots saved: both `pulloff_config.py` and `wrench_config.py` copied into run directory at start.

---

---

# Shared CMA-ES Framework

All three optimizers share the same CMA-ES ask/tell architecture:

**`_cmaes_space_info() → (x0, lower, upper, is_log)`**
Extracts bounds from the `space` list. Log-uniform dimensions are mapped to log₁₀ space so CMA-ES operates in a uniform-scaled coordinate system.

**`_cmaes_to_real(x_internal, is_log) → list`**
Converts internal CMA-ES point back to real-space values (applies `10^x` for log dims). Called inside `ask()`.

**`_create_cmaes_optimizer(x0_override, es_override) → (ask, tell, es)`**
Builds the ask/tell interface around `cma.CMAEvolutionStrategy`.
- `ask()` calls `es.ask()` for the full population, stores internal points in `ask._last_internal`, returns real-space points.
- `tell(points, costs)` passes `ask._last_internal` back to `es.tell()` — required for correct covariance matrix updates (re-encoding real-space points would silently corrupt the update).
- `es_override` → resume: existing ES object reused directly. `x0_override` → warm-start: initial mean set to best known point encoded in internal space.

**`_run_optimization(...)` main loop:**
1. `ask()` → full population of `BATCH_SIZE` points.
2. Submit all to `pool.imap_unordered` (parallel evaluation).
3. Sort results by `point_index` to restore population order.
4. `tell(points, costs)` → CMA-ES covariance update.
5. Append all results to CSV; update best tracker; print batch summary.
6. Pickle `es` to `cmaes_state.pkl` after every batch.

**CLI arguments (all three optimizers):**

| Argument | Description |
|----------|-------------|
| `--suffix` / `-s` | Appended to timestamped run directory name |
| `--n-calls` | Override `N_CALLS` from config |
| `--resume-from` | Resume from `cmaes_state.pkl` (exact optimizer state) |
| `--warm-start-from` | Warm-start from last row of `optimization_bests.csv` (new optimizer, seeded mean) |

`--resume-from` and `--warm-start-from` are mutually exclusive.

**Parallelism:** Pool size is `min(os.cpu_count(), BATCH_SIZE)`. `multiprocessing.set_start_method("spawn")` avoids fork-based instability with MuJoCo's internal state.

---

## Function Call Graph

```
pulloff_sim.py __main__
├── run_headless(pull_rate, params)
│   ├── setup_model(params)
│   └── [step loop]
│       ├── apply_mag(..., params)
│       │   ├── mujoco.mj_geomDistance()
│       │   └── mag_force()
│       └── mujoco.mj_step()
├── pulloff_viewer.run_viewer()
│   ├── setup_model(PARAMS)
│   └── [viewer loop]
│       ├── mujoco.mj_geomDistance() + mag_force()
│       ├── add_arrow()
│       └── mujoco.mj_step()
└── plot()

pulloff_optimizer.py __main__
├── _run_optimization()
│   ├── _create_cmaes_optimizer()
│   └── [batch loop]
│       ├── ask()                             ← es.ask() + _cmaes_to_real()
│       ├── pool.imap_unordered(_evaluate_one_candidate)
│       │   └── [worker] pulloff_sim.run_headless(pull_rate, params)
│       │       └── pulloff_config.calculate_cost(pulloff_force)
│       ├── tell(points, costs)               ← es.tell(internal_points, costs)
│       └── pickle(es) → cmaes_state.pkl
├── pulloff_sim.run_headless(PULL_RATE_OPT, best_params)   ← replay
└── pulloff_viewer.run_viewer(PULL_RATE_OPT)

wrench_sim.py __main__
├── run_headless(pull_rate, params)
│   ├── setup_model(params)
│   └── [step loop]
│       ├── apply_mag(..., params)
│       ├── apply_wrench_force()
│       └── mujoco.mj_step()
├── wrench_viewer.run_viewer()
│   ├── setup_model(PARAMS)
│   └── [viewer loop]
│       ├── mujoco.mj_geomDistance() + mag_force()
│       ├── apply_wrench_force()
│       ├── add_arrow()
│       └── mujoco.mj_step()
└── plot()

wrench_optimizer.py __main__
├── _run_optimization()
│   ├── _create_cmaes_optimizer()
│   └── [batch loop]
│       ├── ask()
│       ├── pool.imap_unordered(_evaluate_one_candidate)
│       │   └── [worker] wrench_sim.run_headless(pull_rate, params)
│       │       └── wrench_config.calculate_cost(detach_force, detach_moment, xy_drift)
│       ├── tell(points, costs)
│       └── pickle(es) → cmaes_state.pkl
├── wrench_sim.run_headless(PULL_RATE_OPT, best_params)    ← replay
└── wrench_viewer.run_viewer(PULL_RATE_OPT)

combined_optimizer.py __main__
├── _run_optimization()
│   ├── _create_cmaes_optimizer()
│   └── [batch loop]
│       ├── ask()
│       ├── pool.imap_unordered(_evaluate_one_candidate)
│       │   └── [worker]
│       │       ├── pulloff_sim.run_headless(PULL_RATE_PULLOFF, params)
│       │       │   └── pulloff_config.calculate_cost(pulloff_force)
│       │       └── wrench_sim.run_headless(PULL_RATE_WRENCH, params)
│       │           └── wrench_config.calculate_cost(detach_force, detach_moment, xy_drift)
│       ├── tell(points, costs)               ← weighted sum cost
│       └── pickle(es) → cmaes_state.pkl
├── pulloff_sim.run_headless(PULL_RATE_PULLOFF, best_params)  ← replay
├── wrench_sim.run_headless(PULL_RATE_WRENCH,  best_params)   ← replay
├── pulloff_viewer.run_viewer(PULL_RATE_PULLOFF)
└── wrench_viewer.run_viewer(PULL_RATE_WRENCH)
```

---

## Detachment Detection

| Test | Criterion | Hold duration |
|------|-----------|---------------|
| Pull-off | Vertical displacement > 10 mm | 1.0 s |
| Wrench/peel | Magnetic force < 0.01 N | 0.5 s |

The peak applied force recorded before the hold criterion is met is reported as the **detachment force**.

---

## Usage

```bash
# Pull-off test (headless + viewer + plot)
python pulloff_sim.py
python pulloff_sim.py --pull-rate 50

# Wrench/peel test (headless + viewer + plot)
python wrench_sim.py
python wrench_sim.py --pull-rate 20

# Viewer only (standalone)
python pulloff_viewer.py --pull-rate 30
python wrench_viewer.py --pull-rate 20

# Pull-off CMA-ES optimizer
python pulloff_optimizer.py
python pulloff_optimizer.py --n-calls 500 --suffix my_run
python pulloff_optimizer.py --resume-from results/20250101T120000_my_run
python pulloff_optimizer.py --warm-start-from results/20250101T120000_my_run

# Wrench CMA-ES optimizer
python wrench_optimizer.py
python wrench_optimizer.py --n-calls 500 --suffix my_run
python wrench_optimizer.py --resume-from results/20250101T120000_my_run
python wrench_optimizer.py --warm-start-from results/20250101T120000_my_run

# Combined CMA-ES optimizer (pull-off + wrench jointly)
python combined_optimizer.py
python combined_optimizer.py --n-calls 500 --suffix my_run
python combined_optimizer.py --resume-from results/20250101T120000_my_run
python combined_optimizer.py --warm-start-from results/20250101T120000_my_run
```

**Viewer controls:**

| Key | Action |
|-----|--------|
| `ENTER` | Start simulation |
| `SPACE` | Pause / Resume |

---

## Dependencies

```bash
pip install mujoco numpy matplotlib cma scikit-optimize
```

Requires MuJoCo ≥ 3.0.