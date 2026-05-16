# `f2w_test` Full 3-Step Implementation Checklist

Recommended order: T4/T8/T9 → T1 → T2 → T3/T11/T10 → T5/T6/T7 → end-to-end test

## Current status
- ✅ T8, T9, T4 complete — Step 1 fully done and wired.
- ✅ T1 complete — verified via `test_T1.py` (6/6) + C3 regression. Visual confirmation deferred to T6/T7.
- ✅ T2 complete — verified 9/9 structural + visual.
- ✅ T10 resolved — `Ry(−30°) @ body_R0` confirmed via viewer.
- ✅ T3 resolved — cost weighting (FL pos_cost=200) confirmed via test_T3.py.
- ✅ T10 resolved — Ry(−30°) @ body_R0 confirmed.
- 🔜 Next: T11 (simultaneous vs sequential, design decision), then T5/T6/T7.

## Test files (`legged_sim/test_files/`)

| File | Purpose |
|---|---|
| `test_T9.py` | Joint sign/direction confirmation — sets individual FL joints to test angles, freezes pose, draws world axes + EE frame |
| `test_orient.py` | (pre-existing) Orient sequence test |
| `test_vert_lift.py` | (pre-existing) Vertical lift test |
| `test_viz_joints.py` | (pre-existing) Joint visualization |
| `test_T4.py` | Contracted pose sweep — animates FL joints from settled pose to `STEP1_POSE_FL` over 3 s; watch for self-collision |

---

### T8 — EE face-up vs face-down disambiguation
- [x] **Decision: face-down** (EE local +Y = world −Z)
- Magnet disabled during Step 1 (`disable_magnet=True`), so accidental floor adhesion is not a concern
- The `ee_FL` target angle that achieves this is determined empirically in T9

---

### T9 — Confirm joint sign/direction for each FL joint in viewer
Procedure: set one joint at a time in `STEP1_POSE_FL`, run `"f2w_test"`, observe, confirm or flip sign.

| Joint | Declared axis | Expected motion | Target angle | Sign confirmed? |
|---|---|---|---|---|
| `hip_pitch_FL` | body Z | Outward swing (away from body center) | `+45°` | [x] |
| `knee_FL` | `(+0.707,−0.707,0)` | Up, back-bend toward hip | `+90°` | [x] |
| `wrist_FL` | `(+0.707,−0.707,0)` | Down/back toward body | `−90°` | [x] |
| `ee_FL` | `(0,−0.707,+0.707)` | EE local +Y → world −Z (face-down) | `0°` | [x] |

---

### T4 — Define `STEP1_POSE_FL` in `config.py`
```python
STEP1_POSE_FL = {
    'hip_pitch_FL': math.radians(+45.0),
    'knee_FL':      math.radians(+90.0),
    'wrist_FL':     math.radians(-90.0),
    'ee_FL':        math.radians(  0.0),
}
```
- [x] Signs confirmed for all four joints (T9)
- [x] Magnitudes validated in viewer — no self-collision, correct final pose
- [x] Step 1 duration 3.0 s confirmed sufficient
- [x] Values written into `config.py` as `CONTRACTED_POSE_FL` (note: same as `STEP1_POSE_FL` in context — existing name kept)
- [x] Already wired in `sequences.py` via `joint_phase(joint_targets=contracted_pose('FL'), disable_magnet=True, unbounded=True)` — no changes needed

---

## Step 2 — Extend FL + tilt body (Control 1 on FL, Control 2 on body)

### T1 — Per-phase joint-limit override infrastructure ✅ DONE
- [x] `ControlMode` enum (C1/C2/C3) added to `sequences.py` — explicit, swappable, identifiable
- [x] `control_mode: ControlMode` and `jnt_range_override: Dict[str, Tuple[float,float]]` added to `IKTarget`
- [x] Named constructors `IKTarget.control1/2/3(...)` added — every phase declares its mode explicitly
- [x] Existing constructors (`position_only`, `with_orientation`, `with_joint_targets`) updated to set `ControlMode.C2`
- [x] `joint_phase` updated to use `IKTarget.control3(...)`
- [x] `IKSolver.__init__`: `_default_jnt_range`, `_range_override_key`, `_override_config_limit` added
- [x] `IKSolver.solve()`: C1 path rebuilds `ConfigurationLimit` with widened ranges (cached by `frozenset`), restores model immediately after
- [x] `sim_step` dispatch: `if ik_target.control_mode == ControlMode.C3` replaces old `ctrl_override` check; C1/C2 comment block added
- [x] **Verified** — `test_T1.py`: all 6 structural checks passed (fields, restore, cache hit, rebuild, C2 isolation)
- [x] **C3 regression** — `f2w_test` sequence runs correctly with new dispatch
- [ ] ⚠️ Visual confirmation deferred — full C1 visual check comes at T6/T7 when Step 2 runs end-to-end and widened limits are tuned

---

### T2 — Body-task target override (`body_target_se3`) ✅ DONE
- [x] `body_target_se3: Optional[np.ndarray]` (4×4 SE3) added to `IKTarget`
- [x] `control1()` and `control2()` constructors accept `body_target_se3`
- [x] `IKSolver.__init__`: `_settled_body_T` added
- [x] `IKSolver.record_stance()`: stores `_settled_body_T = T.copy()`
- [x] `IKSolver.solve()`: sets `body_task.set_target(body_target_se3)` when set; restores `_settled_body_T` when None
- [x] **Verified** — `test_T2.py`: 9/9 structural + visual confirmed (body tilts, stance feet stay planted)
- [x] **T10 also resolved via this visual** — see below

---

### T3 — FL decoupling strategy for body task ✅ RESOLVED
- [x] **Decision: Option 1 — cost imbalance** (`FL pos_cost=200` vs body/stance `cost=50`)
- [x] Verified numerically + visually via `test_T3.py`:
  - Low cost (10): FL joints drift noticeably as solver recruits them for body tilt
  - High cost (200): FL stays planted, body tilt executes cleanly through stance legs
  - Body tilt angle similar in both trials — high FL cost does not suppress body tilt
- [x] Rationale confirmed: body tilt and FL EE task are kinematically decoupled — body tilt can only route through stance joints (FR/BL/BR), not FL. Cost weighting is sufficient; HQP not needed.
- [ ] Wire `pos_cost=200` into `extend_and_tilt_phase` factory (T7)

---

### T10 — Body rotation composition order ✅ RESOLVED (via T2 visual)
- [x] **Decision: `Ry(−30°) @ body_R0`** — world-frame left-multiply
- Verified in viewer: `Ry(−30°)` tilts the front of the robot toward +Z (wall face)
- In code: `tilt_T[:3,:3] = _ry(-30)[:3,:3] @ settled_T[:3,:3]`

---

### T11 — Step 2 timeline: simultaneous vs sequential (design decision)
- [ ] Decision needed from Aaron/Nadan
  - **Simultaneous**: tilt + FL extend smooth-stepped together over same `duration`. Simpler.
  - **Sequential**: tilt first, hold, then extend FL. May avoid transient over-extension.
- [ ] Implement chosen strategy in `extend_and_tilt_phase`

---

### T5 — Define `STEP2_FL_EE_POS` in `config.py`
- [ ] Derive from `f2w_orient_phase` formula: `[land.x + 0.10, land.y, ee_home.z + raise_z]`
  - `land.x` = X of eventual wall contact point
  - `raise_z` from re-tuned `f2w_orient_phase` (see geometry TODO below)
- [ ] Validate in viewer that pre-reach pose is reachable without limit saturation (before T6 limits are relaxed)

---

### T6 — Define `STEP2_FL_LIMITS` and `STEP3_FL_LIMITS` in `config.py`
- [ ] Start with knee/wrist widened ±30° beyond XML defaults:
  - `knee_FL` XML default: `±130°` → start at `±160°`
  - `wrist_FL` XML default: `±57°` → start at `±87°`
- [ ] Tune empirically until Step 2 reaches `STEP2_FL_EE_POS` without saturation
- [ ] Tune `STEP3_FL_LIMITS` separately for Step 3 reach (may need different ranges)

---

### T7 — New phase factories in `sequences.py`
- [ ] `extend_and_tilt_phase(ee_target, jnt_range_override, body_rot_axis, body_rot_deg, duration)`
  - Emits `IKTarget` with `position`, `jnt_range_override`, `body_target_se3`
  - Computes `body_target_se3 = settled_pose × Ry(body_rot_deg)`
  - Incorporates T10 composition order and T11 timeline decision
- [ ] `f2w_reach_relaxed_phase(wall_normal, jnt_range_override, duration, unbounded=True)`
  - Variant of `f2w_reach_phase` with `jnt_range_override` populated
  - Body target held from Step 2 (persists via `body_target_se3` in IKTarget)
  - Open question: whether to re-tighten FL joint limits as EE approaches wall (probably not needed — leave relaxed)
- [ ] Wire both into `SEQUENCES['f2w_test']`

---

## Step 3 — Reach FL forward, body held (Control 1)

Reuses `f2w_measure_phase` (no changes) + new `f2w_reach_relaxed_phase` (T7).
Magnet enabled at `t >= duration` per existing pattern.
Covered by T7 above.

---

## Geometry / Tuning (independent of step order)

- [ ] Re-tune `lift_phase(height=0.10)` for new link lengths (existing `"f2w"` sequence)
- [ ] Re-tune `f2w_orient_phase(raise_z=0.40)` for new link lengths (existing `"f2w"` sequence)
  - Longer hip/knee links raise the settled EE home height (`ee_home.z`) and grow swing arc radius

---

## End-to-end

- [ ] Full `"f2w_test"` 3-step sequence runs in viewer without collision or limit saturation
- [ ] FL EE contacts wall at `X = +0.500 m` and magnet engages
- [ ] Stance feet (FR, BL, BR) stay planted throughout
- [ ] Hand off to FR sequence (`"f2w_fr"`) works correctly after FL plants