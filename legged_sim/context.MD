# `legged_sim` — Context Reference

Focused on **XML positioning**, **IK solver**, **time steps**, and **sequence position/orient commands**. Pulled from `scene.xml`, `robot.xml`, `config.py`, `sim.py`, `sequences.py`.

---

## Geometry Redesign — Link Lengths Updated ✓

Robot redesign applied. All four files updated. Current physical dimensions:

| Link | **Current length** | Notes |
|---|---|---|
| Hip | **106.94 mm** | Elevation angle 50°, 45° XY diagonal |
| Knee | **158.27 mm** | Same XY-diagonal direction as before |
| Wrist | **200.56 mm** | Same XY-diagonal direction as before |
| EE | **48.66 mm** | Along local +X only |
| Electromagnet | unchanged | EM body `pos` updated with EE link; standoff preserved at 0.000075 m |

**What was updated:**
- `robot_original.xml` — all `fromto` endpoints and child body `pos` for hip/knee/wrist/EE/EM
- `robot.xml` — same changes (bake angles all 0, so identical to source)
- `config.py` — `KNEE_GEOM_ZERO`, `WRIST_GEOM_ZERO`, `EE_GEOM_ZERO`, `EM_POS_ZERO`

**Still pending after geometry update:**
- > **TODO (sequences.py):** Re-tune `lift_phase(height=0.10)` and `f2w_orient_phase(raise_z=0.40)` after geometry update. Longer hip/knee links raise EE home height and grow the swing arc radius.

---

## 1. XML Positioning

### Scene geometry (`scene.xml`)
World frame: +X = front, +Y = left, +Z = up.

| Geom | `pos` (m) | `size` (half-extents, m) | Inner face | Notes |
|---|---|---|---|---|
| `floor` | `-1.0  0  -0.05` | `1.5  0.5  0.05` | Top at **Z = 0** | X spans **[-2.5, +0.5]**; far edge flush with wall |
| `wall`  | `+0.55 0  +0.5`  | `0.05 0.5 0.5`   | Inner at **X = +0.500** | Outer face at X = +0.600 |

Both plates: `condim=6`, `friction="0.8 0.005 0.0001"` (overridden at runtime by `PARAMS['ground_friction']`).

### Robot spawn (`robot.xml` + `sim.py:setup_model`)
- `main_frame` body declared at `pos="-2.0 0 0.45"` with a `freejoint` named `root`.
- After `bake_joint_angles` and `mj_forward`, **`sim.py` adds +2.1 to the freejoint X qpos**, so actual spawn is **(+0.1, 0, 0.45)** — front of robot ~0.40 m from wall inner face.
- Frame is `0.260 × 0.260 × 0.046` m; corner mount points are **(±0.0965, ±0.092, +0.023)** on top face.

### Joint pivot placement — critical invariant

Every joint pivot sits at the **origin of its child body**, set by the child body's `pos=` attribute.
The link geom is `fromto="0 0 0  Δx Δy Δz"`, where the endpoint `Δx Δy Δz` is always set to the **same value** as the child body's `pos`. This means:

> **Extending or shortening a link requires updating both values identically and atomically:**
> ```
> geom  fromto="0 0 0  NEW_X NEW_Y NEW_Z"     ← visual stretch
> body  pos="NEW_X NEW_Y NEW_Z"               ← joint pivot follows the new tip
> ```
> If only `fromto` is updated, the geom stretches but the joint pivot stays at the old position — kinematics and visuals are out of sync. If only `pos` is updated, the joint moves but the cylinder geom still points to the old location.

Joint `axis=` vectors, joint `range=`, body `quat=`, and `KNEE/WRIST/EE_AXIS` in `config.py` are all **length-independent** — they define orientation in the local body frame and do not change when link lengths change.

### Leg chain — **current values** (example `FL`, all positions in parent-body local frame)

Body-origin-at-joint convention; cylinder `fromto="0 0 0  Δx Δy Δz"` spans joint→next joint.

```
main_frame
└── hip_link_FL          pos=( 0.0965,  0.092,   0.023)   joint hip_pitch_FL  axis Z   range ±45°
    │   hip_geom_FL      fromto="0 0 0  0.04861 0.04861 0.08192"   ← hip link cylinder
    │
    └── knee_link_FL     pos=( 0.04861,  0.04861,  0.08192)  ← must equal hip fromto tip; knee joint pivot here
                         joint knee_FL  axis (+0.707,−0.707,0)  ±130°
        │   knee_geom_FL fromto="0 0 0  0.10279 0.10279 -0.06258"  ← knee link cylinder
        │
        └── wrist_link_FL pos=( 0.10279,  0.10279, −0.06258) ← must equal knee fromto tip; wrist joint pivot here
                           joint wrist_FL  axis (+0.707,−0.707,0)  ±57°
            │   wrist_geom_FL fromto="0 0 0  0.07091 0.07091 -0.17369"  ← wrist link cylinder
            │
            └── EE_FL    pos=( 0.07091,  0.07091, −0.17369)  ← must equal wrist fromto tip; ee joint pivot here
                         quat=(0.7071,0,0.7071,0)
                         joint ee_FL  axis (0,−0.707,+0.707)  ±90°
                         joint ee2_FL axis (0,+0.707,+0.707)  ±90°  (passive)
                │   ee_link_geom_FL fromto="0 0 0  0.04866 0 0"  ← EE link cylinder
                │
                └── electromagnet_FL pos=( 0.04874, 0, 0)  ← EE tip + standoff (0.000075 m)
                                     quat=(0.5, 0.5, 0.5, −0.5)
                                     joint em_z_FL axis (0,−1,0)  ±180°  (passive)
```

### All-leg position table (current, post-redesign)

#### Hip link — 106.94 mm, 50° elevation, 45° XY diagonal

| Leg | `hip_geom` `fromto` endpoint = `knee_link` `pos` |
|---|---|
| FL | `( 0.04861,  0.04861,  0.08192)` |
| FR | `( 0.04861, -0.04861,  0.08192)` |
| BL | `(-0.04861,  0.04861,  0.08192)` |
| BR | `(-0.04861, -0.04861,  0.08192)` |

#### Knee link — 158.27 mm

| Leg | `knee_geom` `fromto` endpoint = `wrist_link` `pos` |
|---|---|
| FL | `( 0.10279,  0.10279, -0.06258)` |
| FR | `( 0.10279, -0.10279, -0.06258)` |
| BL | `(-0.10279,  0.10279, -0.06258)` |
| BR | `(-0.10279, -0.10279, -0.06258)` |

#### Wrist link — 200.56 mm

| Leg | `wrist_geom` `fromto` endpoint = `EE_*` `pos` |
|---|---|
| FL | `( 0.07091,  0.07091, -0.17369)` |
| FR | `( 0.07091, -0.07091, -0.17369)` |
| BL | `(-0.07091,  0.07091, -0.17369)` |
| BR | `(-0.07091, -0.07091, -0.17369)` |

#### EE link — 48.66 mm, along EE local +X

| All legs | `ee_link_geom` `fromto` endpoint | `electromagnet` `pos` |
|---|---|---|
| FL/FR/BL/BR | `(0.04866, 0, 0)` | `(0.04874, 0, 0)` |

> **Note on electromagnet pos:** Standoff = 0.000075 m preserved from old design (0.04866 + 0.000075 = 0.04874). EM quat, mesh, mass, and contact spheres are all unchanged.

---

### `config.py` — bake-relevant dicts (current values)

`bake_joint_angles()` regex-matches these exact strings against `robot_original.xml`. If they drift out of sync, baking will silently fail to substitute.

```python
KNEE_GEOM_ZERO = {
    'FL': np.array([ 0.10279,  0.10279, -0.06258]),
    'FR': np.array([ 0.10279, -0.10279, -0.06258]),
    'BL': np.array([-0.10279,  0.10279, -0.06258]),
    'BR': np.array([-0.10279, -0.10279, -0.06258]),
}
WRIST_GEOM_ZERO = {
    'FL': np.array([ 0.07091,  0.07091, -0.17369]),
    'FR': np.array([ 0.07091, -0.07091, -0.17369]),
    'BL': np.array([-0.07091,  0.07091, -0.17369]),
    'BR': np.array([-0.07091, -0.07091, -0.17369]),
}
EE_GEOM_ZERO = {leg: np.array([0.04866, 0.0, 0.0]) for leg in ('FL', 'FR', 'BL', 'BR')}
EM_POS_ZERO  = {leg: np.array([0.04874, 0.0, 0.0]) for leg in ('FL', 'FR', 'BL', 'BR')}
```

> **Hip link** (`hip_geom` fromto) is **not baked** by `bake_joint_angles()` — it is updated directly in both `robot_original.xml` and `robot.xml`.

---

### Bake offsets (`config.py`)
`bake_joint_angles` rewrites `robot_original.xml` → `robot.xml`:
- `KNEE_BAKE_DEG = WRIST_BAKE_DEG = EE_BAKE_DEG = 0.0` for all legs **currently** — the rotation/quat math is wired up but no offsets applied.
- Uses Rodrigues for geom `fromto`/body `pos`; `_axis_angle_to_rot ∘ _quat_to_rot ∘ _rot_to_quat` for EM quat.

### Actuators (`robot.xml`)
12 hip/knee/wrist motors + 4 `ee_*` motors = **16 motors total**, all `ctrlrange="-100 100"` (torque). The `ee2_*` and `em_z_*` joints have **no actuator** — passive with stiffness/damping in XML.

### Mass / dynamics overrides at load (`sim.py:setup_model`)
Ignores XML `<default><joint damping="1.0" armature="0.01"/>`:
```
model.dof_damping[:]  = 2.0
model.dof_armature[:] = 0.01
```
Friction on floor/wall geoms set from `PARAMS['ground_friction']`. Solver params (`o_solref`, `o_solimp`, `noslip_iterations`, `noslip_tolerance`, `o_margin`) all written from `PARAMS`.

---

## 2. Time Steps

| Constant | Value | Where | Purpose |
|---|---|---|---|
| `TIMESTEP` | **0.0005 s** (2 kHz) | `config.py` → `model.opt.timestep` | Physics step |
| `scene.xml` `<option timestep>` | 0.002 | scene.xml | **Overridden** by `setup_model` |
| `SETTLE_TIME` | 2.0 s | `config.py` → `sim.py` | Settling phase before sequence starts |
| `SIM_DURATION` | 30.0 s | `config.py` (unused) | `main()` uses `--duration 20.0` default |
| `IK_EVERY_N` | **10** | `sim.py` | IK solved every 10 physics steps → **200 Hz IK** |
| `n_iter` (per IK call) | 10 | `IKSolver.solve` default | Inner Levenberg–Marquardt iterations |
| `dt` passed to mink | `IK_EVERY_N * TIMESTEP` = **0.005 s** | `sim.py` line 571 | Integration window for `solve_ik` |
| `frame_dt` | 1/60 s | `sim.py` viewer loop | Render rate |
| `steps_per_frame` | `int(frame_dt / TIMESTEP)` = **33** | `sim.py` line 644 | Physics steps per render frame |
| `DUAL_REACH_DWELL` | 3.0 s | `sim.py` | Time in `F2W_REACH` before FL is "planted" and FR launches |
| `last_print` interval | **0.5 s** (sequence loop) / **1.0 s** (main loop) | `sim.py` lines 564, 656 | Telemetry printed every 0.5 s during settle/sequence; every 1.0 s in the outer viewer loop. `TELEMETRY_INTERVAL=0.5` from config is not referenced — hardcoded in sim.py. |
| `_CONTACT_LIFT_THRESH` | 0.008 m | `sim.py` line 321 | EE >8 mm above settle baseline = "lost contact" |

---

## 3. IK Solver (`sim.py:IKSolver`, mink)

### Architecture
Whole-body IK, single solver instance, 16 actuators. Tasks composed every tick:

```
tasks = [body_task, foot_FL, foot_FR, foot_BL, foot_BR, posture_task]
limits = [ConfigurationLimit(model)]
mink.solve_ik(config, tasks, dt=0.005, solver="quadprog",
              damping=1e-3, limits=limits)
```
10 iterations; each iteration calls `config.update(qpos)` → `solve_ik` → `config.integrate`.

### Task definitions
| Task | Frame | `position_cost` | `orientation_cost` | Notes |
|---|---|---|---|---|
| `body_task` | `main_frame` (body) | 50.0 | **50.0**, dropped to **0** when `target.face_axis is not None` | Competes with EE orient task; zeroed during orient phases |
| `foot_tasks[swing]` | `electromagnet_{foot}` | from `IKTarget.position_cost` (10.0 default) | from `IKTarget.orientation_cost` (0.0 default; **50.0** for orient phases) | **Reconstructed** when `orientation_cost` changes (mink bakes at construction) |
| `foot_tasks[stance, floor]` | `electromagnet_{foot}` | 50.0 | 0 (position-only) | Target = `stance_targets[foot]` snapshot at settle |
| `foot_tasks[stance, wall]` | `electromagnet_{foot}` | 50.0 | **30.0** | Full SE3 lock from `stance_se3[foot]` (set by `lock_stance_orientation`) |
| `posture_task` | — | `cost=0.01` | — | Anchors to `qpos.copy()` at settle |
| `_joint_task` (lazy) | per-DOF | `IKTarget.joint_cost` (scalar) | — | `mink.PostureTask` built on-demand by `_ensure_joint_task`; keyed by `(frozenset(dof_adrs), joint_cost)` — rebuilt only when the active joint set or cost changes. Active when `IKTarget.joint_targets` is non-empty. Skipped entirely when `ctrl_override` is set. |
| `config_limit` | `ConfigurationLimit` | — | — | Joint range limits |

`IK_DAMPING = 1e-3` used for **both** mink LM damping and the `solve_ik` damping kwarg.

### Orientation handling — `_compute_ori_target`
`face_axis` = world-frame direction that **EE local −Y** must point toward (EE local +Y is the magnet contact face).
- Rodrigues rotation: minimal rotation aligning current `−R[:,1]` to `face_axis`, applied on the **left** to current rot matrix.
- Edge case: if `sin_a < 1e-6` and `cos_a < 0` (180° flip), uses `−I + 2·n·nᵀ` reflection.
- **Cached in `self._ori_target_rot`** on first appearance, reused for the rest of the orient/measure/reach phases.
- Reset to `None` whenever `target.face_axis is None` (e.g., position-only phases) and on every `_start_runner_for(...)` (new foot).

### Stance lifecycle
- `record_stance(data)`: snapshots EE positions for all 4 feet → `stance_targets`; snapshots body SE3 → `body_task` target; snapshots `qpos` → `posture_task` target. Called at end-of-settle and again when FL hands off to FR.
- `lock_stance_orientation(data, foot)`: snapshots full 4×4 SE3 of an EE → `stance_se3[foot]`. Called only at FL→FR handoff for `FL`. From this moment, FL's foot task uses position **and** orientation cost 50/30.

### Solve output
Returns `ctrl` array (length `model.nu = 16`): `ctrl[i] = ik_qpos[jnt_qposadr[ctrl_jids[i]]]`. These are **joint position targets**, fed to the PID, not torques. The torques are computed by `PIDController.compute` (KP=500, KI=200, KD=30, I clamp=100).

**`ctrl_override` bypass:** When `IKTarget.ctrl_override` is set, `sim.py` skips `ik.solve()` entirely. Instead, named joint values are written directly into `ctrl_targets`; all other joints hold their previous `ctrl_targets`. The same PID then drives to the overridden values. This path is active for `joint_phase`-based sequences (e.g. `"f2w_test"`).

---

## 4. Sequence Position & Orient Commands (`sequences.py`)

### Datatypes
- `IKTarget(position, position_cost=10.0, face_axis=None, orientation_cost=0.0, joint_targets=None, joint_cost=5.0, ctrl_override=None)`
  - `joint_targets`: `Dict[str, float]` mapping joint name → target qpos (rad). Passed to a lazily-built mink `PostureTask` keyed by `(frozenset(dof_adrs), joint_cost)`. Rebuilt only when the active joint set or `joint_cost` changes.
  - `joint_cost`: scalar cost applied uniformly to every DOF in `joint_targets`. Tune relative to `position_cost` to balance Cartesian vs joint-space tracking.
  - `ctrl_override`: `Dict[str, float]` — when set, `sim.py` **skips the QP solve entirely** and writes these joint values directly as PID position references. All other joints hold their last `ctrl_targets`. Used by `joint_phase`. The IK framework is otherwise untouched.
- `PhaseContext(foot, ee_home, ee_pos_fn, hip_pivot_fn, wall_dist_fn, magnet_disable_fn, magnet_enable_fn, joint_qpos_fn, shared)`
  - `ee_pos_fn`/`hip_pivot_fn` are **refreshed every `step()` call** in `SequenceRunner.step`
  - `joint_qpos_fn`: `Callable[[str], float]` — returns current `qpos` for a named joint; wired by `_start_runner_for` to `ik.joint_qpos(data, jname)`. Used by `lift_phase` and `joint_phase` to capture on-entry joint values for smooth-step interpolation.
- `Phase(name, duration, step, on_enter=None, unbounded=False)` — `unbounded=True` ⇒ holds past `duration`, never auto-advances
- Smoothing: `_smooth(t, T, q0, q1)` = quintic smoothstep `s³(6s²−15s+10)` with `s = clip(t/T, 0, 1)`.

### Phase factories — what each emits

| Factory | Position target | Orientation (`face_axis`) | Magnet | `duration` / unbounded |
|---|---|---|---|---|
| `lift_phase(height, duration, joint_targets=None, joint_cost=5.0, position_cost=10.0)` | `start + smooth·[0,0,height]` from EE-pos-on-entry | None | **disable on enter** | 3.0 / False |
| `swing_phase(angle_deg, duration, face_axis)` | Rotate `(start − pivot)` by `smooth·angle_rad` about Z, add `pivot` | optional | — | bounded unless `unbounded=True` |
| `reach_phase(dx, duration, face_axis)` | `start + [smooth·dx, 0, 0]` (global +X extend) | optional | **enable on enter** | bounded unless `unbounded=True` |
| `hold_phase(face_axis, position_cost=8, orientation_cost=50, mag_enable_delay)` | `start` (frozen on entry) | optional | enable when `t >= mag_enable_delay` | **0.0, unbounded=True** |
| `f2w_orient_phase(raise_z, wall_normal, duration, position_cost=20, orientation_cost=30)` | `[land.x, land.y, ee_home.z + raise_z]` | `-wall_normal` | — | 3.0 |
| `f2w_measure_phase(wall_normal, wall_face_pos, foot_standoff=0.008, wall_offset=0.02, position_cost=8, orientation_cost=50)` | hold current EE pos | `-wall_normal` | — | **0.0** (instant) |
| `f2w_reach_phase(wall_normal, duration, position_cost=8, orientation_cost=50)` | `_smooth(t, T, start, ctx.shared['f2w_reach_target'])` | `-wall_normal` | enable when `t >= duration` | 2.0, **unbounded=True** |
| `joint_phase(joint_targets, duration=3.0, height=0.0, disable_magnet, enable_magnet, unbounded)` | EE dummy (carried for telemetry only — **IK QP skipped**) | None | optional | 3.0 / configurable |

**`joint_phase` detail:** Sets `IKTarget.ctrl_override = smooth-stepped joint dict`. `sim.py` detects `ctrl_override` every IK tick and writes values directly to `ctrl_targets`, bypassing `ik.solve()` entirely. Start values per joint are captured from live `qpos` on entry via `ctx.joint_qpos_fn`. Use for validating joint signs/angles without IK interference; once pose looks correct, switch to `lift_phase(joint_targets=...)` for the blended IK+joint version.

**`lift_phase` with `joint_targets`:** When `joint_targets` is provided, each joint is quintic-smooth-stepped from its on-entry live value to the target alongside the EE lift. A low `position_cost` (e.g. 1.0) lets the joint task dominate; the EE target acts as a soft hint.

**Convention for orientation:** `wall_normal` points **from EE toward wall**; `face_axis = −wall_normal` so IK drives EE local −Y to face away from the wall, i.e. EE local +Y (the magnet contact face) presses **into** the wall.

### `f2w_measure_phase` distance computation (analytic, preferred)
```
ray_dir = wall_normal (normalized)
dist    = (wall_face_pos − ee_pos) · ray_dir       # dot-product projection
eff     = dist − foot_standoff − wall_offset
if eff <= 0:  eff = dist − foot_standoff           # inside safety zone
if eff <= 0.001:  eff = 0.001                      # clamp to 1 mm
ctx.shared['f2w_reach_target'] = ee_pos + eff·ray_dir
```
Falls back to `ctx.wall_dist_fn` (which uses `mj_ray`) if `wall_face_pos is None`. The `mj_ray` path also runs a max-reach feasibility check and `sys.exit(1)` if unreachable.

### Concrete sequence dictionaries

#### `"orient"` — single-foot orient demo
```
1. lift_phase(height=0.10, duration=3.0)
2. swing_phase(angle_deg=-45, duration=1.5)
3. hold_phase(face_axis=[-1, 0, 0], mag_enable_delay=5.0)
   pos_cost=8, ori_cost=50  (defaults)
```

#### `"f2w"` — FL floor-to-wall (5 phases)
```
1. lift_phase(height=0.10, duration=3.0)
   - DISABLE FL magnet on entry
2. swing_phase(angle_deg=-45, duration=1.5)        # CCW from above (FL toward +X wall)
3. f2w_orient_phase(raise_z=0.40, wall_normal=[+1,0,0], duration=3.0,
                    position_cost=20, orientation_cost=30)
   - target = [land.x, land.y, ee_home.z + 0.40]   # uses ee_home.z, NOT swing-landing.z
   - face_axis = [-1, 0, 0]                        # EE -Y points -X
4. f2w_measure_phase(wall_normal=[+1,0,0], wall_face_pos=[0.500, 0, 0.5],
                     foot_standoff=0.008, wall_offset=0.02,
                     position_cost=8, orientation_cost=50)
   - duration=0; writes ctx.shared['f2w_reach_target']
5. f2w_reach_phase(wall_normal=[+1,0,0], duration=2.0,
                   position_cost=8, orientation_cost=50)
   - smooth-step start → ctx.shared['f2w_reach_target']
   - face_axis = [-1, 0, 0]
   - ENABLE FL magnet at t >= 2.0
   - unbounded → holds at reach_target after duration
```

> **TODO (sequences.py):** Re-tune `lift_phase(height=0.10)` and `f2w_orient_phase(raise_z=0.40)` for the new geometry. Longer hip/knee links raise the settled EE home height and grow the swing arc radius; both parameters were calibrated for the old link lengths.

#### `"f2w_test"` — FL direct joint-position tuning (no IK)
```
1. joint_phase(joint_targets=contracted_pose('FL'), duration=3.0,
               disable_magnet=True, unbounded=True)
   - Bypasses IK entirely; writes ctrl_override directly to PID
   - Starts from live qpos, quintic-smooth-steps to CONTRACTED_POSE_FL
   - Holds final angles indefinitely after 3 s
```
> **Workflow:** Validate joint signs/angles here first (pure PID, no EE constraint fights). Once the pose looks correct in the viewer, copy angles into `config.py` and switch the sequence to `lift_phase(joint_targets=contracted_pose('FL'), ...)` for the blended IK+joint version. Active sequence: **`SEQUENCE = "f2w_test"`** in `config.py`.

---

#### `"f2w_test"` — Full 3-Step Floor-to-Wall Plan (FL leg)

The single-`joint_phase` `"f2w_test"` above is a stepping stone. The full plan splits the FL transition into **3 sequential steps**, each using a different **control mode**. Two of the three control modes are already implemented; one is **TODO**. This subsection is the spec.

##### Three control modes

| # | Name | Status | Mechanism | Existing analog |
|---|---|---|---|---|
| **Control 1** | IK setpoints **with relaxed per-phase joint limits** | ✅ **implemented** | `IKTarget.control1(jnt_range_override=...)` — `IKSolver.solve()` temporarily widens `model.jnt_range` to build a cached `ConfigurationLimit`, restores defaults immediately. Dispatch via `ControlMode.C1`. | `IKTarget.control1(...)` |
| **Control 2** | Pure IK | ✅ implemented | `IKTarget.control2(...)` drives mink with default limits. | `lift_phase`, `swing_phase`, `f2w_orient_phase`, `f2w_reach_phase` |
| **Control 3** | Pure joint-angle position control (no IK) | ✅ implemented | `IKTarget.control3(...)` sets `ctrl_override`; `sim_step` dispatches on `ControlMode.C3`, skips IK solve entirely. | `joint_phase` |

###### Why Control 1 is needed

The XML `range=` attributes on each joint are tuned for **stable nominal poses** (hip ±45°, knee ±130°, wrist ±57°). During Step 2 / Step 3 of the f2w transition, the FL leg has to pass through configurations that are blocked by these defaults — knee/wrist need to swing past their nominal limits to clear the body and wall. Hard-relaxing the limits in the XML would compromise stability everywhere else in the sim; the limits must be **scoped to the active phase**.

###### Implementation (done — Option 2)

`IKTarget.jnt_range_override: Dict[str, Tuple[float, float]]` field added. In `IKSolver.solve()`, when set: temporarily mutates `model.jnt_range` for the listed joints, constructs a new `ConfigurationLimit`, restores defaults immediately. Cache-keyed by `frozenset` — only rebuilt when the dict changes. Verified via `test_T1.py` (6/6 structural checks passed). Visual confirmation deferred to T6/T7.

##### Coordinate convention (applies to all 3 steps)

- **Joint angles are commanded in body-local (parent-link) frame** — the convention used by MuJoCo `qpos` and the existing `joint_targets` dict.
- **When the user spec references world-frame attitudes** ("EE parallel with floor", "main frame tilt about world Y", "wrist above horizontal in world frame"), the implementer must translate that world-frame attitude into either:
  - A `face_axis` for the EE orient task (Control 2 path) — converted internally by `IKSolver._compute_ori_target`, **or**
  - A body-frame joint angle for `ctrl_override` (Control 3 path) — typically requires solving a small forward-kinematics problem by hand, or running a one-shot IK in the viewer and reading back `qpos`.
- Joint axes (body frame) for FL — same convention applies mirrored for FR/BL/BR:
  - `hip_pitch_FL` axis = body **Z** (right-hand rule: positive = CCW viewed from +Z)
  - `knee_FL` axis = `(+0.707, −0.707, 0)` (body X-minus-Y diagonal)
  - `wrist_FL` axis = `(+0.707, −0.707, 0)` (same diagonal as knee)
  - `ee_FL` axis = `(0, −0.707, +0.707)`
- All "CCW/CW" descriptions below follow the right-hand rule about the joint's declared axis. **Sign conventions for CCW vs CW must be confirmed empirically in the viewer** — see TODOs in each step.

---

##### Step 1 — Contract joints inward  (Control 3 only, FL)

###### Goal
Pull the FL leg in toward the body so it does not collide with the wall during the subsequent swing-and-reach.

###### Motions (qualitative)

| Joint | Target motion | Notes |
|---|---|---|
| `hip_pitch_FL` | Swing **outward CCW by ~45°** about body Z | "Outward" relative to body center |
| `knee_FL` | Swing **up CCW** into a back-bend about (+0.707,−0.707,0) | Knee link folds up toward the hip |
| `wrist_FL` | Swing **down CW** about (+0.707,−0.707,0) | Wrist link folds back toward the body |
| `ee_FL` | Set so the EE plate is **parallel with the floor** (world XY plane) | One concrete body-frame angle |

###### Control mode
**Control 3** (already implemented). All four joints driven by a single `joint_phase` — no IK active for FL during this step.

###### Implementation checklist

1. **`CONTRACTED_POSE_FL` in `config.py`** — confirmed and written (this is `STEP1_POSE_FL`):
   ```python
   CONTRACTED_POSE_FL = {
       'hip_pitch_FL': +np.radians(45),   # ✓ +45° swings leg toward +Y (away from wall)
       'knee_FL':      +np.radians(90),   # ✓ +90° folds knee tip upward (back-bend)
       'wrist_FL':     -np.radians(90),   # ✓ −90° folds wrist tip down and back toward body
       'ee_FL':         0.0,              # ✓ 0° → EE local +Y = world −Z (face-down)
   }
   ```
   > Face-down chosen — magnet is disabled during Step 1 so floor adhesion is not a concern.
2. **`SEQUENCES['f2w_test']`** — already wired via `joint_phase(joint_targets=contracted_pose('FL'), disable_magnet=True, unbounded=True)`. No changes to `sequences.py` needed.
3. Magnet disabled on entry. ✓

###### Open items

~~None — Step 1 is complete. Validated in viewer (test_T4.py): no self-collision during 3.0 s sweep, leg tucks cleanly toward +Y.~~

---

##### Step 2 — Extend leg + tilt main frame  (Control 1 on FL, Control 2 on body)

###### Goal
Unfold the contracted FL leg and swing it forward to a pre-reach pose; **simultaneously**, tilt the robot body 30° CW about world Y to bias the FL hip toward the wall, extending FL's effective reach.

###### Motions

###### FL leg — Control 1 (IK with relaxed joint limits)

| Joint | Target motion | Notes |
|---|---|---|
| `hip_pitch_FL` | Swing **inward CW by ~30°** about body Z | Opposite to Step 1's outward swing |
| `knee_FL` | Swing **CW away from body** about (+0.707,−0.707,0) | Unfolds the back-bend |
| `wrist_FL` | Swing **CCW** about (+0.707,−0.707,0) so EE stays **above horizontal in world frame** | "Above horizontal" = wrist tip Z > knee tip Z (world Z) |
| `ee_FL` | EE plate **near parallel with world XY plane**, within ±20° | Soft constraint — `orientation_cost` lower than reach phase is fine |

The FL EE position target for this phase is the pre-reach pose: somewhere above and forward of the eventual wall contact point. Driven by IK with **expanded** joint range so the solver can reach this pose without limit-saturation.

> **TODO: NEED CLEARING** — concrete FL EE position target for the pre-reach pose. Recommend deriving from the current `f2w_orient_phase` formula `[land.x, land.y, ee_home.z + 0.40]` with a forward bias (e.g. `land.x + 0.10`).

###### Main frame — Control 2 (pure IK)

- **Target body SE3:** the snapshot from `record_stance` (settled pose), post-multiplied by a 30° CW rotation about world **Y**:
  - "30° CW about Y" with right-hand rule = `Ry(−30°)`. Tilts the front of the robot **down toward the wall** (the FL direction).
  - In code: `tilted_R = body_R0 @ Ry(−30°)` (or world-frame: `Ry(−30°) @ body_R0` depending on which side the rotation is intended; **TODO: NEED CLEARING** which composition order matches "tilting toward FL").
- **Constraint:** FR, BL, BR feet stay planted at their stance positions (already enforced by the existing stance `foot_task`s with `position_cost=50`). The body tilt is therefore realized **through the FR, BL, BR stance leg joints** — the IK solves for those joint angles to satisfy both "feet planted" and "body tilted".
- **Critical: the body-task command must NOT actuate the FL leg.** FL is driven by Control 1 to its own EE target — body task should not steal FL DOFs to satisfy itself.

###### Implementation checklist

1. **Body-task target override** — currently `IKSolver.record_stance` snapshots body SE3 once at settle and never updates it. Need a sequence-driven override:
   - **Recommended (Option A):** add `body_target_se3: Optional[mink.SE3] = None` to `IKTarget`. In `IKSolver._compute_tasks`, when set, call `body_task.set_target(...)`; otherwise leave the stance snapshot. Symmetric with how `face_axis` overrides orientation.
   - **Option B:** add `IKSolver.set_body_target(se3)` called from the phase factory. Simpler but breaks the IKTarget-driven flow.
   - > **TODO: NEED CLEARING** — confirm Option A.
2. **Decoupling FL from the body task** — mink's `FrameTask` (used for body) computes a Jacobian over **all** DOFs by default, so any joint can be recruited to satisfy the body task. Options to prevent FL recruitment:
   - **Two-pass IK:** first pass solves FL `foot_task` only and freezes FL `qpos`; second pass solves body + stance feet with FL DOFs held fixed. Adds latency at 200 Hz IK rate — likely **infeasible**.
   - **Mask FL DOFs via a high-cost FL posture sub-task** during Step 2 — pin FL's joint angles (the Control-1-commanded values) with a `mink.PostureTask` whose cost is high enough that mink barely moves them to satisfy the body task. This piggybacks on the existing `_joint_task` machinery (`joint_targets` / `joint_cost`).
   - **Cost-imbalance approximation:** make the FL `foot_task` cost much higher than the body task cost (e.g. 200 vs 50). FL will be "almost decoupled" — the body task can only use FL marginally. Cheapest, but inexact.
   - > **TODO: NEED CLEARING** — Aaron's preference. Recommend trying the cost-imbalance approximation first, then escalating to the high-cost FL posture sub-task if FL still drifts when the body tilts.
3. **New phase factory: `extend_and_tilt_phase(...)`** in `sequences.py`. Emits an `IKTarget` carrying:
   - `position` = FL EE pre-reach target
   - `jnt_range_override` = relaxed limits dict for FL joints (Control 1)
   - `body_target_se3` = settled body pose × `Ry(−30°)` (Control 2)
   - `joint_targets` + `joint_cost` if using the FL-pin approach to decoupling (above)
4. **Define `STEP2_FL_LIMITS`** in `config.py` — which FL joints relax, and to what range. Start by widening knee/wrist by ±30° beyond their XML defaults; tune empirically.

###### Open items

- > **TODO: NEED CLEARING** — Whether the body tilt and the FL extension run on the **same timeline** (both smooth-stepped together over the same `duration`), or **sequentially** (tilt first, hold tilt, then extend FL). Same-timeline is simpler; sequential may avoid transient over-extension.
- > **TODO: NEED CLEARING** — Whether `face_axis` should also be set for FL EE orientation in this phase (currently the spec says "near parallel ±20°" — soft, may not warrant a separate orient task).

---

##### Step 3 — Reach forward  (Control 1 on FL, body held)

###### Goal
With the body tilted (held from Step 2) and FL leg pre-extended, reach the FL EE horizontally toward the wall and engage the magnet.

###### Motions

- FL EE moves horizontally (in world frame) toward wall inner face at `X = +0.500 m`.
- Body remains at the Step-2 tilt — held, not driven.
- Stance feet (FR, BL, BR) remain planted (existing stance-task behavior).

###### Control mode
- **Control 1** for FL: IK with relaxed joint limits to reach forward without limit-saturation.
- Body task: held at Step-2 target (no further tilting commanded). The new `body_target_se3` field should persist from Step 2.

This step is essentially a **modified `f2w_reach_phase` with `jnt_range_override` populated** plus a held body target.

###### Implementation checklist

1. **Reuse the f2w pipeline** — keep `f2w_measure_phase` as-is (it just writes `ctx.shared['f2w_reach_target']`), then call a new `f2w_reach_relaxed_phase` variant with `jnt_range_override`.
2. **Magnet enable timing** — keep existing pattern: enable when `t >= duration` (i.e., at contact moment).
3. **Body target hold** — emit the same `body_target_se3` from Step 2 in this phase's `IKTarget`. The new sequencing layer (Option A above) will keep it active.
4. **Joint range tightening near contact** (optional) — as the EE approaches the wall, optionally re-tighten FL joint ranges to default to ensure the contact pose is a stable nominal configuration.
   - > **TODO: NEED CLEARING** — whether this hardening is needed for stability. If not needed, leave relaxed throughout reach.

---

##### Combined sequence sketch — `sequences.py:SEQUENCES['f2w_test']`

```python
SEQUENCES['f2w_test'] = [
    # ── Step 1: contract FL inward (Control 3 — pure joint angles, no IK) ──
    joint_phase(
        joint_targets=STEP1_POSE_FL,           # TODO: define in config.py
        duration=2.0,
        disable_magnet=True,
        unbounded=False,
    ),

    # ── Step 2: extend FL + tilt body (Control 1 + Control 2) ──
    # TODO: phase factory does not exist yet — to be authored in sequences.py
    extend_and_tilt_phase(
        ee_target=STEP2_FL_EE_POS,             # TODO: define in config.py
        jnt_range_override=STEP2_FL_LIMITS,    # TODO: define in config.py — relaxed FL ranges
        body_rot_axis=[0, 1, 0],               # world Y
        body_rot_deg=-30.0,                    # CW about world Y (Ry(-30°))
        duration=3.0,
    ),

    # ── Step 3: reach FL forward, body held (Control 1) ──
    f2w_measure_phase(
        wall_normal=[+1, 0, 0],
        wall_face_pos=[0.500, 0, 0.5],
        foot_standoff=0.008,
        wall_offset=0.02,
    ),
    f2w_reach_relaxed_phase(                   # TODO: variant of f2w_reach_phase with jnt_range_override
        wall_normal=[+1, 0, 0],
        jnt_range_override=STEP3_FL_LIMITS,    # TODO: define in config.py
        duration=2.0,
        unbounded=True,
    ),
]
```

##### Cross-cutting TODO summary

| # | Item | File(s) | Blocking? |
|---|---|---|---|
| ~~T1~~ | ~~Per-phase joint-limit override (Control 1 infrastructure)~~ | ~~`sim.py:IKSolver`, `sequences.py:IKTarget`~~ | ✓ **Done** — `ControlMode` enum + `jnt_range_override` + named constructors; `sim_step` dispatch updated |
| ~~T2~~ | ~~Body-task target override mechanism (`body_target_se3` on `IKTarget`)~~ | ~~`sim.py:IKSolver`, `sequences.py:IKTarget`~~ | ✓ **Done** — `_settled_body_T` stored; `body_target_se3` overrides per-tick; verified 9/9 + visual |
| ~~T3~~ | ~~FL-decoupling strategy for body task~~ | ~~design decision~~ | ✓ **Done** — Option 1 (cost imbalance): FL `pos_cost=200`, body/stance `cost=50`. Verified: body tilt executes, FL stays planted. Wire into `extend_and_tilt_phase` at T7. |
| ~~T4~~ | ~~`STEP1_POSE_FL` joint angle values + signs~~ | ~~`config.py`, viewer-driven tuning~~ | ✓ **Done** — written as `CONTRACTED_POSE_FL`; `f2w_test` already wired via `joint_phase`; validated in viewer |
| T5 | `STEP2_FL_EE_POS` pre-reach target | `config.py`, derive from `f2w_orient` formula | **Yes** — Step 2 |
| T6 | `STEP2_FL_LIMITS`, `STEP3_FL_LIMITS` relaxed-range dicts | `config.py` | **Yes** — Steps 2 & 3 |
| T7 | New phase factories: `extend_and_tilt_phase`, `f2w_reach_relaxed_phase` | `sequences.py` | **Yes** — Steps 2 & 3 |
| ~~T8~~ | ~~EE "parallel with floor" disambiguation (face-up vs face-down)~~ | ~~design decision~~ | ✓ **Done** — face-down, `ee_FL = 0°` |
| ~~T9~~ | ~~Sign of "CCW/CW" for each FL joint relative to its declared axis~~ | ~~`config.py` comments + viewer test~~ | ✓ **Done** — all four joints confirmed in viewer |
| ~~T10~~ | ~~Body rotation composition order~~ | ~~sequence factory~~ | ✓ **Done** — `Ry(−30°) @ body_R0` (world-frame left-multiply) confirmed in viewer |
| T11 | Step 2 timeline: simultaneous tilt+extend vs sequential | design decision | Step 2 |

##### Recommended implementation order

1. ~~**T4, T8, T9**~~ ✓ **Done.**
2. **T1** — implement `jnt_range_override` (Option 2: new `IKTarget` field). Smallest IK change; unblocks T6, T7.
3. **T2** — implement `body_target_se3` (Option A: new `IKTarget` field). Symmetric with T1.
4. **T3, T11, T10** — design decisions; collect input from Aaron / Nadan in one pass. Implement chosen decoupling strategy.
5. **T5, T6, T7** — author the new phase factories and config dicts. Wire into `SEQUENCES['f2w_test']`.
6. End-to-end test in the viewer; iterate on `STEP*_LIMITS` and EE targets until the transition completes without collision or limit saturation.

---

#### `"f2w_fr"` — FR mirror, launched after FL plants
Identical to `"f2w"` **except** `swing_phase(angle_deg=+45, …)` (CW from above for FR).
`wall_normal`, `face_axis`, `wall_face_pos` are **the same** — both feet are heading to the same +X wall.

### Dual-leg orchestration (`sim.py:main`)
```
runner = SequenceRunner(SEQUENCES["f2w"])     # FL first
…
After FL F2W_REACH dwells DUAL_REACH_DWELL = 3.0 s:
    runner.force_complete()                   # FL frozen at last target
    ik.record_stance(data)                    # snapshot all 4 EE positions
    ik.lock_stance_orientation(data, "FL")    # FL gets full SE3 lock (orient_cost=30)
    swing_foot_ref → "FR"
    hip_jid_ref    → hip_pitch_FR
    swing_mag_bid  → electromagnet_FR
    swing_off       = {electromagnet_FR}      # FL re-enabled, FR disabled
    _start_runner_for("FR", "f2w_fr", t)
```
After FR enters `F2W_REACH` and dwells 3.0 s, `wall_feet` adds `"FR"` and `wall_baselines["FR"]` is captured for penetration tracking. `wall_baselines[foot][0]` (X-coord) is the reference for "wall penetration" = `ee.x − baseline_x`.

### Live-state plumbing notes
- `SequenceRunner.step()` overwrites `_ctx.ee_pos_fn` and `_ctx.hip_pivot_fn` every call so closures in phase factories always read **current** swing foot / hip pivot. Foot identity, `ee_home`, magnet fns, and `shared` persist for the duration of one runner.
- `_start_runner_for` resets `ik._ori_target_rot = None` so the cached orientation rotation is rebuilt for the new foot's first orient/measure/reach phase.
- `force_complete()` sets `done=True` but `_last_target` is preserved → subsequent `step()` calls return the frozen IKTarget. IK then treats FL as a stance foot via `stance_se3["FL"]`.

---

## Quick numeric sanity check

| Quantity | **Current value** |
|---|---|
| Hip link length | **106.94 mm** |
| Hip elevation angle | **50°** |
| Knee link length | **158.27 mm** |
| Wrist link length | **200.56 mm** |
| EE link length | **48.66 mm** |
| Robot front-of-frame at spawn (X) | +0.1 + 0.130 = **+0.23 m** (frame unchanged) |
| Wall inner face (X) | **+0.500 m** |
| Distance, frame-front to wall | **~0.27 m** |
| Spawn Z (`main_frame`) | **0.45 m** |
| X spawn offset (`sim.py`) | **+2.1** ✓ tuned for new link lengths |
| `f2w_orient` target Z | `ee_home.z + 0.40 m` — **needs re-tune** |
| `f2w_reach` net travel | `dist − 0.008 − 0.02` (formula unchanged) |
| Magnetic engage range | `PARAMS['max_magnetic_distance'] = 0.0575 m` |
| Per-wheel force cap | `PARAMS['max_force_per_wheel'] = 800.7 N` |
| `Br` | 1.883 T |
| EM contact sphere positions (local EM frame) | ±0.022 m ring (unchanged) |