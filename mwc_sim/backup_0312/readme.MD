# Sally Magnetic Adhesion Simulation Suite

MuJoCo-based simulations for characterizing the magnetic adhesion of Sally — a four-wheel omnidirectional wall-climbing robot. Two test modes are provided: a **pull-off test** (normal detachment force) and a **wrench/peel test** (combined shear force + bending moment).

---

## Overview

Magnetic forces are not natively supported in MuJoCo, so this suite implements a **dipole-dipole force model** applied via an array of non-colliding sampling spheres embedded inside the physical magnet body. Each sphere computes its closest-point distance to the steel plate and applies a proportional attractive force along that direction. Together they approximate the distributed magnetic adhesion of a real permanent magnet.

```
F = (3 μ₀ m²) / (2π (2d)⁴)       m = Br · V_magnet / μ₀
```

Where `d` is the closest-point distance from each sampling sphere to the plate surface, `Br` is the magnet remanence (Tesla), and `V_magnet` is the physical magnet volume. The resulting force vector is applied to the magnet body via `xfrc_applied`.

Simulation parameters (`Br`, max force per sphere, friction, solver settings) are tuned via Bayesian optimization against physical hardware experiments for three operating modes: **Hold**, **Drive Sideways**, and **Drive Up**.

---

## File Structure

```
.
├── XML/
│   └── flat_plate_pull.xml      # Pull-off scene: magnet + plate MJCF
├── mwc_mjcf/
│   └── scene.xml                # Wrench scene: top-level MJCF with includes
├── pulloff_sim.py               # Headless pull-off simulation + plotting
├── pulloff_viewer.py            # Interactive viewer for pull-off test
├── wrench_sim.py                # Headless wrench/peel simulation + plotting
├── wrench_viewer.py             # Interactive viewer for wrench/peel test
└── results/
    └── wrench_results.png       # Output plot (auto-generated)
```

---

## XML Scene Files

### `XML/flat_plate_pull.xml` — Pull-Off Scene

This is the primary MJCF scene for the pull-off test. It defines the physical world: the steel plate, the magnet body, the sampling spheres, and the bracket mesh.

**`<option>`** sets `timestep="0.001"` with an implicit integrator and standard gravity. This timestep is small enough to resolve the fast contact dynamics during magnetic snap-on/off.

**`<default>`** sets `condim="6"` globally, enabling full 3D friction (normal + two tangential + torsional + two rolling) on all geoms by default.

**`<worldbody>`** contains two bodies:

**`plate` body** — A static 500×500×10 mm steel box (`plate_geom`). Its top surface sits at z=0. It is the collision target for all `mj_geomDistance` calls in the magnetic force computation. Friction and solver parameters on this geom are overwritten at runtime by `setup_model()`.

**`magnet` body** — The full magnet + bracket assembly, positioned so its bottom face is flush with the plate top at z=0 (body origin at z=0.017 m, cylinder offset −0.0025 m). It carries a `<freejoint name="magnet_free"/>` giving it full 6-DOF motion so it can translate and rotate freely when forces are applied. It contains:

- **`magnet_cyl`** — The physical magnet geometry: a cylinder of radius 31.5 mm, half-height 14.5 mm, density 7850 kg/m³ (steel). Rendered red.
- **`bracket`** — An STL mesh of the magnet bracket (`1102_PF_Magnet_Bracket.STL`) imported at 1:1000 scale (mm → m) and rotated into place with euler angles. Rendered semi-transparent blue. Density 2700 kg/m³ (aluminum). `contype`/`conaffinity` are not disabled — set explicitly if collisions cause instability.
- **Sampling spheres (8 spheres, bottom ring)** — Non-colliding spheres (`contype="0" conaffinity="0"`, `mass=1e-6 kg`) arranged in a ring of radius 28.5 mm at z=−17 mm, which is the bottom face of the magnet cylinder. These are the force calculation points: their world positions are used to query `mj_geomDistance` to the plate each timestep, and the computed attractive force is applied to the magnet body. They are rendered as blue dots in the viewer.

  > Two additional rings (middle at z=0, top at z=+10.5 mm, 8 spheres each) are defined but commented out. Uncommenting them enables a 3-layer, 24-point force field for higher spatial resolution at the cost of computational overhead.

---

### `mwc_mjcf/scene.xml` — Wrench Test Top-Level Scene

This is the top-level MJCF file for the wrench/peel test. It is a thin wrapper that configures global visuals and includes the magnet+plate fragment.

**`<compiler>`** sets `angle="radian"` and `meshdir="assets"`. All euler angles in included files are interpreted as radians.

**`<option>`** sets gravity to `0 0 -9.81` with an implicit integrator.

**`<visual>`** configures headlight diffuse/ambient/specular values and fog range (`znear=0.01`, `zfar=10.0`). A blue-to-white gradient skybox (`rgb1="0.6 0.8 1.0"`) is defined in `<asset>`.

**`<statistic center="0 0 0.3" extent="1.0"/>`** centers the viewer camera on the scene at startup so the magnet is in frame.

**`<include file="flat_plate_pull.xml"/>`** pulls in the magnet + plate world body definitions. The wrench scene shares the same physical geometry as the pull-off scene; all behavioral differences are in how Python applies forces at runtime.

> `wrench_sim.py`'s `setup_model()` looks for body name `1103___pp___aws_pem_215lbs__eml63mm_24` and site name `stick_tip`, which must be defined either in the included fragment or added directly to this scene. A `ValueError` is raised at startup if either is missing.

---

## Python Files

### `pulloff_sim.py` — Headless Pull-Off Simulation

**Purpose:** Runs a complete pull-off test without a viewer, collects per-step force and displacement records, detects detachment, prints the pull-off force, saves a results plot, then launches the interactive viewer.

**Key constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `PULL_RATE` | 20.0 N/s | Force ramp rate; overridable via `--pull-rate` |
| `SETTLE_TIME` | 0.3 s | Duration of magnetic settle phase before ramp |
| `DETACH_DIST` | 10.0 mm | Vertical displacement threshold for detachment |
| `DETACH_HOLD` | 1.0 s | Displacement must stay above threshold this long |
| `SIM_DURATION` | 100.0 s | Hard stop for the simulation loop |
| `MAGNET_VOLUME` | π·(0.0315²)·0.025 m³ | Physical magnet volume for dipole moment |
| `PARAMS` | dict | All tunable physics parameters applied at runtime |

**Critical functions:**

`mag_force(dist, Br) → float`
Computes the dipole-dipole attractive force (N) for a single sampling sphere at closest-point distance `dist` from the plate. Pure math, no MuJoCo calls. Formula: `F = (3μ₀m²) / (2π(2d)⁴)` where `m = Br·V/μ₀`. Called inside `apply_mag()` and independently in the viewer loop.

`setup_model(xmlpath) → (model, data, plate_id, magnet_id, joint_id, sphere_gids)`
Loads the MJCF via `mujoco.MjModel.from_xml_path()`. Applies all `PARAMS` overrides: plate friction, `opt.o_solref`, `opt.o_solimp`, `opt.noslip_iterations`. Resolves body/geom/joint integer IDs by name using `mj_name2id`. Builds `sphere_gids` by filtering all geoms on the magnet body to those of type `mjGEOM_SPHERE`. These sphere IDs are the only geoms used for force calculations downstream.

`apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto) → float`
Core magnetic force function. Iterates over all sampling sphere geom IDs. For each:
1. Calls `mujoco.mj_geomDistance(model, data, gid, plate_id, 50.0, fromto)` — writes the closest-point segment into `fromto[0:3]` (sphere surface point) and `fromto[3:6]` (plate surface point), returns scalar distance.
2. Skips if `dist ≤ 0` or `dist > max_magnetic_distance`.
3. Calls `mag_force(dist, Br)` and clamps result to `max_force_per_wheel`.
4. Computes force direction as the normalized `fromto` direction vector.
5. Accumulates force into `data.xfrc_applied[magnet_id, :3]`.
Returns total Fz (used to track magnetic attraction in the downward direction).

`run_headless(pull_rate) → (records, pulloff_force)`
Main simulation loop. Two phases:
- **Settle phase** (`t < SETTLE_TIME`): `apply_mag()` called each step, no pull force. Magnet snaps to plate and equilibrates.
- **Ramp phase** (`t ≥ SETTLE_TIME`): linearly increasing upward force (`f_pull = pull_rate × t_ramp`) added to `xfrc_applied[magnet_id, 2]` each step. Displacement `z_disp` tracked via `data.qpos[jnt_qposadr[joint_id]]`. Detachment declared when `z_disp > DETACH_DIST` for `DETACH_HOLD` seconds.
Appends `{t, f_pull, f_mag, z_disp}` dict to `records` each step.

`smooth(data, window=200) → np.ndarray`
Box-filter (moving average) smoother applied to raw per-step signals before plotting. Window of 200 steps = 0.2 s of smoothing at dt=0.001.

`plot(records, pulloff_force, pull_rate)`
Generates a two-panel matplotlib figure: (left) Force vs. Time with pull ramp, smoothed magnetic attraction, and detachment marker; (right) Force vs. Displacement zoomed to the detach region. Saved to `results/wrench_results.png`.

**Entry point (`__main__`):**
```
parse args → run_headless() → pulloff_viewer.run_viewer() → plot()
```

---

### `pulloff_viewer.py` — Interactive Pull-Off Viewer

**Purpose:** Re-runs the pull-off simulation in real time inside the MuJoCo passive viewer with per-sphere force arrow overlays. Called automatically by `pulloff_sim.py` after the headless run completes, or run standalone.

**Imports from `pulloff_sim`:** `XMLPATH`, `PARAMS`, `SETTLE_TIME`, `SIM_DURATION`, `DETACH_DIST`, `DETACH_HOLD`, `setup_model`, `apply_mag`, `mag_force`. This ensures the viewer uses identical model setup and force logic to the headless run — changing `PARAMS` in `pulloff_sim.py` affects both.

**Key constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `REAL_TIME_FACTOR` | 2.0 | Wall-clock speed relative to sim time |
| `ARROW_RADIUS` | 0.004 m | Visual shaft radius of force arrows |
| `key_state` | `{"paused": True}` | Shared pause flag toggled by keyboard |

**Critical functions:**

`key_callback(keycode)`
MuJoCo passive viewer callback. Toggles `key_state["paused"]` on `SPACE` (keycode 32) or `ENTER` (keycode 257). Registered at viewer launch.

`add_arrow(scene, start, end, rgba)`
Adds a visual arrow to `v.user_scn` using `mjv_initGeom` (sets type, size, pos, mat) followed by `mjv_connector` (orients the arrow from `start` to `end`). Guards against overflow with `scene.ngeom < scene.maxgeom`. Called once per sphere (blue) and once for the pull force (red) each step.

`run_viewer(pull_rate)`
Main viewer loop inside `mujoco.viewer.launch_passive()` context. Each step:
1. If paused: calls `v.sync()` and sleeps 20 ms.
2. Clears `xfrc_applied` and resets `user_scn.ngeom = 0`.
3. Iterates over sphere geoms: computes distance, force, and direction (mirroring `apply_mag` inline); applies force to `xfrc_applied`; draws a blue arrow at each sphere position scaled to force magnitude.
4. After `SETTLE_TIME`: computes ramp force, adds it to `xfrc_applied[magnet_id, 2]`, draws a red upward arrow at magnet center.
5. Calls `mj_step`, checks detachment (displacement + hold timer), prints telemetry every 0.1 s sim-time.
6. Sleeps `max(0, dt_wall - elapsed)` to maintain `REAL_TIME_FACTOR`.

> The viewer re-runs the full simulation from scratch. It does not replay headless results. Behavior should be identical but floating-point order of operations may produce minor differences.

---

### `wrench_sim.py` — Headless Wrench/Peel Simulation

**Purpose:** Runs a wrench/peel test headless, where a horizontal force applied at the end of a rigid lever arm generates a bending moment at the magnet-plate interface. Records force, moment, and magnetic attraction per step, detects peel-off, saves a plot, then launches the viewer.

**Key constants:**

| Constant | Default | Description |
|----------|---------|-------------|
| `PULL_RATE` | 20.0 N/s | Force ramp rate |
| `SETTLE_TIME` | 1.0 s | Longer settle than pull-off — two-phase settle |
| `DETACH_HOLD` | 0.5 s | Time magnetic force must be ~0 to confirm detachment |
| `SIM_DURATION` | 50.0 s | Hard stop |
| `STICK_TIP_LOCAL` | `[0, 0, 0.0825]` m | Lever arm tip in magnet body frame (82.5 mm along local Z) |
| `PARAMS` | dict | Three pre-tuned configs; Sideways active by default |

**Critical functions:**

`mag_force(dist, Br) → float`
Identical dipole-dipole formula as in `pulloff_sim.py`.

`setup_model() → (model, data, plate_id, magnet_id, sphere_gids, tip_site_id)`
Loads `SCENE_XML`, applies `PARAMS` overrides, resolves IDs for `plate_geom`, the magnet body (`1103___pp___aws_pem_215lbs__eml63mm_24`), and the `stick_tip` site. Returns `tip_site_id` in addition to the standard outputs — the site's world position is queried each step to compute the moment arm. Raises `ValueError` if any named element is not found.

`apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto) → float`
Same structure as in `pulloff_sim.py` but returns total force **magnitude** (sum of per-sphere scalar forces) rather than Fz. This is used to determine whether any magnetic contact still exists — when it drops to near zero, the magnet has peeled away.

`apply_wrench_force(data, magnet_id, f_pull, tip_site_id)`
Applies a horizontal force in +X at the stick tip site's current world position. Steps:
1. `tip_world = data.site_xpos[tip_site_id]` — the site's world-frame position updates each step as the magnet rotates.
2. `force = [f_pull, 0, 0]`
3. `r = tip_world - data.xpos[magnet_id]` — moment arm in world frame.
4. `moment = cross(r, force)` — resultant torque about magnet center.
5. Accumulates both into `data.xfrc_applied[magnet_id, :3]` and `[3:]`. Using the site position (not a fixed offset) correctly accounts for the magnet tilting as it begins to peel.

`run_headless(pull_rate) → (records, detach_force)`
Three-phase simulation loop:
- **Phase 1** (`t < SETTLE_TIME/2`): gravity only — magnet falls onto the plate under gravity.
- **Phase 2** (`t < SETTLE_TIME`): magnetic forces applied — magnet snaps and settles.
- **Phase 3** (ramp): `apply_mag` + `apply_wrench_force` each step. Records `{t, f_pull, moment, f_mag}`. Detachment declared when `f_mag < 0.01 N` sustained for `DETACH_HOLD` seconds.

`plot(records, detach_force, pull_rate)`
Two-panel figure: (left) Force & Moment vs. Time with all three signals; (right) Magnetic Force vs. Applied Force over time. Saved to `results/wrench_results.png`.

**Entry point (`__main__`):**
```
parse args → run_headless() → wrench_viewer.run_viewer() → plot()
```

---

### `wrench_viewer.py` — Interactive Wrench/Peel Viewer

**Purpose:** Re-runs the wrench simulation in real time with force arrow overlays in the MuJoCo passive viewer. Called automatically by `wrench_sim.py`, or run standalone.

**Imports from `wrench_sim`:** `SCENE_XML`, `PARAMS`, `SETTLE_TIME`, `SIM_DURATION`, `DETACH_HOLD`, `STICK_TIP_LOCAL`, `PULL_RATE`, `setup_model`, `apply_mag`, `mag_force`, `apply_wrench_force`. Changing anything in `wrench_sim.py` automatically affects the viewer.

**Key constants:**
- `REAL_TIME_FACTOR = 0.8` — Slight slow-motion (20% slower than real time) to observe peel dynamics clearly.

**Critical functions:**

`key_callback(keycode)`, `add_arrow(scene, start, end, rgba)`
Identical in structure to `pulloff_viewer.py`.

`run_viewer(pull_rate)`
Main viewer loop. Structure mirrors `pulloff_viewer.run_viewer()` with two differences:
1. Magnetic forces are enabled after `SETTLE_TIME/2` (not at t=0) to match the two-phase settle in `wrench_sim.py`.
2. The pull force arrow is drawn at `data.site_xpos[tip_site_id]` pointing in +X, reflecting where the wrench force is actually applied.
Telemetry prints phase, `F_pull`, and `F_mag` every 0.1 s.

---

## Function Call Graph

```
pulloff_sim.py __main__
├── run_headless()
│   ├── setup_model()
│   │   └── mujoco.MjModel.from_xml_path()
│   └── [step loop]
│       ├── apply_mag()
│       │   ├── mujoco.mj_geomDistance()
│       │   └── mag_force()
│       └── mujoco.mj_step()
├── pulloff_viewer.run_viewer()       ← imports pulloff_viewer
│   ├── setup_model()
│   └── [viewer loop]
│       ├── mujoco.mj_geomDistance()
│       ├── mag_force()
│       ├── add_arrow()
│       └── mujoco.mj_step()
└── plot()

wrench_sim.py __main__
├── run_headless()
│   ├── setup_model()
│   │   └── mujoco.MjModel.from_xml_path()
│   └── [step loop]
│       ├── apply_mag()
│       │   ├── mujoco.mj_geomDistance()
│       │   └── mag_force()
│       ├── apply_wrench_force()
│       └── mujoco.mj_step()
├── wrench_viewer.run_viewer()        ← imports wrench_viewer
│   ├── setup_model()
│   └── [viewer loop]
│       ├── mujoco.mj_geomDistance()
│       ├── mag_force()
│       ├── apply_wrench_force()
│       ├── add_arrow()
│       └── mujoco.mj_step()
└── plot()
```

---

## Physics Parameters

Three pre-tuned `PARAMS` configurations exist in `wrench_sim.py`. Uncomment the desired block and comment out the others:

| Config | `Br` (T) | `max_magnetic_distance` (m) | `max_force_per_wheel` (N) | Use case |
|--------|----------|-----------------------------|---------------------------|----------|
| **Sideways** (default) | 1.476 | 0.02974 | 264.2 | Driving sideways on vertical wall |
| **Up** | 1.628 | 0.01319 | 300.0 | Climbing upward |
| **Hold** | 1.332 | 0.01141 | 139.3 | Stationary / holding position |

All configurations share the same parameter keys:

| Parameter | Description |
|-----------|-------------|
| `Br` | Magnet remanence (T). Scales the dipole moment `m = Br·V/μ₀`. |
| `max_magnetic_distance` | Distance cutoff (m) beyond which no force is applied. |
| `max_force_per_wheel` | Per-sphere force clamp (N). Prevents unphysical forces at very small distances. |
| `ground_friction` | Plate friction `[sliding, torsional, rolling]`. Applied to `plate_geom` at runtime. |
| `solref` | MuJoCo constraint solver reference `[timeconst, dampratio]`. Applied to `opt.o_solref`. |
| `solimp` | MuJoCo solver impedance `[dmin, dmax, width, midpoint, power]`. Applied to `opt.o_solimp`. |
| `noslip_iterations` | Number of noslip solver iterations. Higher = stiffer contact, slower sim. |

---

## Detachment Detection

| Test | Criterion | Hold duration |
|------|-----------|---------------|
| Pull-off | Vertical displacement > 10 mm | 1.0 s |
| Wrench/peel | Magnetic force < 0.01 N (all spheres out of range) | 0.5 s |

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
```

**Viewer controls:**

| Key | Action |
|-----|--------|
| `ENTER` | Start simulation |
| `SPACE` | Pause / Resume |

---

## Dependencies

```bash
pip install mujoco numpy matplotlib
```

Requires MuJoCo ≥ 3.0.