"""
test_vert_lift.py — interactive viewer for vertical wall + FL lift test.

Spawns the robot on the vertical wall (via scene_vert_opt.xml keyframe),
settles with all 4 magnets ON, then lifts FL by LIFT_DZ while turning its
magnet OFF — identical to the floor lift test but on the wall.

Tests whether the 3 remaining stance magnets can resist gravity (now acting
sideways, −Z in world frame) while FL is in the air.

Controls:
  SPACE / ENTER — pause / resume
  RIGHT ARROW   — step one frame while paused

Prints per-foot drift summary when the viewer is closed.
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer as mjviewer

# ── Paths ─────────────────────────────────────────────────────────────────────
# test_vert_lift.py lives in legged_sim/test_files/
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
LEGGED_DIR    = os.path.abspath(os.path.join(_THIS_DIR, ".."))
OPTIMIZER_DIR = os.path.join(LEGGED_DIR, "optimizer")
SCENE_XML     = os.path.join(LEGGED_DIR, "mwc_mjcf", "scene_vert_opt.xml")

for _p in (LEGGED_DIR, OPTIMIZER_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    MU_0, MAGNET_VOLUME, MAGNET_BODY_NAMES,
    TIMESTEP,
    KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG,
)

# Re-use IK and PID from optimizer/sim_opt_sim
from sim_opt_sim import _IK, _PID, _mag_force, _apply_mag, _se3_from_pos

# ── Test params ───────────────────────────────────────────────────────────────

PARAMS = {
    'ground_friction':       [0.567206, 0.086353, 4.85595e-05],
    'solref':                [0.00428172, 10.0],
    'solimp':                [0.991827, 0.9999, 0.00691039, 0.269361, 3.12149],
    'noslip_iterations':     30,
    'noslip_tolerance':      1.06284e-06,
    'margin':                0.00181254,
    'Br':                    1.64657,
    'max_magnetic_distance': 0.0963125,
    'max_force_per_wheel':   800.426,
}

# ── Timing ────────────────────────────────────────────────────────────────────

FEET       = ('FL', 'FR', 'BL', 'BR')
SWING_FOOT = 'FL'

SETTLE_TIME = 2.0   # s — all magnets ON, hold bake pose on wall
LIFT_HOLD   = 3.0   # s — FL lifted, FL magnet OFF, stance magnets ON
LIFT_DZ     = 0.10  # m — how high to lift FL (in robot-frame Z = world −X on wall)

IK_DAMPING  = 1e-3
IK_EVERY_N  = 10

PID_KP      = 500.0
PID_KI      = 200.0
PID_KD      = 30.0
PID_I_CLAMP = 100.0

WALL_SPAWN_KEY = "wall_spawn"

# ── Key state ─────────────────────────────────────────────────────────────────

_key_state = {"paused": True, "step_once": False}

def _key_callback(keycode):
    if keycode in (32, 257):
        _key_state["paused"] = not _key_state["paused"]
    elif keycode == 262:
        _key_state["step_once"] = True

# ── Model setup ───────────────────────────────────────────────────────────────

def _setup_model(params):
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, WALL_SPAWN_KEY)
    if key_id == -1:
        raise RuntimeError(f"Keyframe '{WALL_SPAWN_KEY}' not found in scene_vert_opt.xml")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    for leg in FEET:
        for jname, bake_dict in [
            (f'knee_{leg}',  KNEE_BAKE_DEG),
            (f'wrist_{leg}', WRIST_BAKE_DEG),
            (f'ee_{leg}',    EE_BAKE_DEG),
        ]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                data.qpos[model.jnt_qposadr[jid]] = np.radians(bake_dict[leg])

    mujoco.mj_forward(model, data)

    plate_ids = set()
    for name in ("floor", "wall"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        plate_ids.add(gid)
        model.geom_friction[gid] = params['ground_friction']

    model.opt.o_solref          = params['solref']
    model.opt.o_solimp          = params['solimp']
    model.opt.noslip_iterations = params['noslip_iterations']
    model.opt.noslip_tolerance  = params['noslip_tolerance']
    model.opt.o_margin          = params['margin']
    model.dof_damping[:]        = 2.0
    model.dof_armature[:]       = 0.01

    magnet_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in MAGNET_BODY_NAMES
    ]
    sphere_gids = {
        mid: [
            gid for gid in range(model.ngeom)
            if model.geom_bodyid[gid] == mid
            and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
        ]
        for mid in magnet_ids
    }

    return model, data, plate_ids, magnet_ids, sphere_gids

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    params = PARAMS
    model, data, plate_ids, magnet_ids, sphere_gids = _setup_model(params)

    swing_mag_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{SWING_FOOT}")

    STANCE_FEET = [f for f in FEET if f != SWING_FOOT]
    stance_ee_bids = {
        foot: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{foot}")
        for foot in STANCE_FEET
    }
    all_ee_bids = {
        foot: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{foot}")
        for foot in FEET
    }

    ik  = _IK(model)
    pid = _PID(model)

    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]

    settled         = False
    ee_baseline     = {}
    ik_step_counter = 0
    target_pos      = np.zeros(3)
    drift_samples   = {foot: [] for foot in STANCE_FEET}

    total_time = SETTLE_TIME + LIFT_HOLD

    print("Vertical wall + FL lift test")
    print(f"  Settle: {SETTLE_TIME}s (all magnets ON)")
    print(f"  Lift:   {LIFT_HOLD}s  (FL magnet OFF, FL raised {LIFT_DZ*100:.0f}cm)")
    print("Press SPACE to start, RIGHT ARROW to step while paused.")

    with mjviewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
        viewer.cam.lookat[:] = [0.3, 0.0, 0.5]
        viewer.cam.distance  = 1.2
        viewer.cam.elevation = -10
        viewer.cam.azimuth   = 160

        step_start = time.perf_counter()

        while viewer.is_running():
            # ── Paused ────────────────────────────────────────────────────
            if _key_state["paused"] and not _key_state["step_once"]:
                viewer.sync()
                time.sleep(0.02)
                continue
            _key_state["step_once"] = False

            # ── Phase 1: settle ───────────────────────────────────────────
            if data.time < SETTLE_TIME:
                data.xfrc_applied[:] = 0
                _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params)
                data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
                mujoco.mj_step(model, data)

            # ── Transition: record stance + compute lift target ────────────
            elif not settled:
                settled = True
                ik.record_stance(data)
                ee_home    = ik.ee_pos(data, SWING_FOOT).copy()
                # On the wall, the robot's local Z maps to world -X, so lifting
                # FL means moving in robot-local +Z → world -X direction.
                # IK handles this automatically since targets are in world frame;
                # we just lift by LIFT_DZ in world Z to match floor convention.
                target_pos = ee_home + np.array([0.0, 0.0, LIFT_DZ])
                for foot, bid in stance_ee_bids.items():
                    ee_baseline[foot] = data.xpos[bid].copy()
                print(f"\n[t={data.time:.2f}s] Settle complete — lifting {SWING_FOOT}")
                print(f"  FL target: {target_pos}")

            # ── Phase 2: lift hold ────────────────────────────────────────
            elif data.time < total_time:
                data.xfrc_applied[:] = 0
                _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                           params, off_mids={swing_mag_bid})

                ik_step_counter += 1
                if ik_step_counter >= IK_EVERY_N:
                    ik_step_counter = 0
                    ctrl_targets = ik.solve(target_pos, data, IK_EVERY_N * TIMESTEP)

                data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
                mujoco.mj_step(model, data)

                for foot, bid in stance_ee_bids.items():
                    drift_samples[foot].append(
                        np.abs(data.xpos[bid] - ee_baseline[foot]).copy()
                    )

            viewer.sync()

            elapsed   = time.perf_counter() - step_start
            remaining = TIMESTEP - elapsed
            if remaining > 0:
                time.sleep(remaining)
            step_start = time.perf_counter()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Stance foot drift during lift (from settle baseline) ──")
    print(f"  {'foot':<6}  {'x (mm)':>8}  {'y (mm)':>8}  {'z (mm)':>8}")
    for foot in STANCE_FEET:
        samples = np.array(drift_samples[foot])
        if samples.size == 0:
            print(f"  {foot:<6}  (no samples)")
            continue
        mean = samples.mean(axis=0) * 1000
        print(f"  {foot:<6}  {mean[0]:>8.2f}  {mean[1]:>8.2f}  {mean[2]:>8.2f}")


if __name__ == "__main__":
    run()