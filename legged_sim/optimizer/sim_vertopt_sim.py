"""
sim_vertopt_sim.py — headless wall FL-lift runner for the vert optimizer.

Runs:
  Phase 1: settle (2 s, all magnets ON, all joints hold bake pose on wall)
  Phase 2: lift FL 10 cm and hold (3 s, FL magnet OFF, stance magnets ON)

Mirrors sim_opt_sim.py exactly but the robot is spawned on the vertical wall.
Gravity now acts sideways (-Z world), so the 3 stance magnets must resist
both pull-off and gravity-direction slip while FL is in the air.

Slip metric (returned as mean_abs_x/y/z, lower = better):
  X — mean |X drift| of FR/BL/BR EEs (pull-off from wall).
  Y — mean |Y drift| of FR/BL/BR EEs (lateral slip).
  Z — mean |Z drift| of FR/BL/BR EEs (gravity-direction drop).
  Sampled over [SETTLE_TIME + LIFT_MEASURE_START, total_time].

These feed into calculate_cost() in sim_vertopt_config.py:
  total = 0.50 * X  +  0.50 * Z  +  0.00 * Y
"""

import os
import sys
import numpy as np
import mujoco
import mink

# ── Paths ────────────────────────────────────────────────────────────────────
# sim_opt_sim.py lives in legged_sim/optimizer/, so ".." is legged_sim/ itself.

LEGGED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCENE_XML  = os.path.join(LEGGED_DIR, "mwc_mjcf", "scene_vert_opt.xml")

if LEGGED_DIR not in sys.path:
    sys.path.insert(0, LEGGED_DIR)

from config import (
    MU_0, MAGNET_VOLUME, MAGNET_BODY_NAMES,
    TIMESTEP,
    KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG,
)

# Reuse IK solver from floor sim — works identically on wall
from sim_opt_sim import _IK

# ── Constants ─────────────────────────────────────────────────────────────────

FEET       = ('FL', 'FR', 'BL', 'BR')
SWING_FOOT = 'FL'

SETTLE_TIME        = 2.0   # s — all magnets ON, hold bake pose
LIFT_HOLD          = 3.0   # s — FL lifted, FL magnet OFF, stance magnets ON
LIFT_MEASURE_START = 1.0   # s into lift before sampling (skip foot-travel transient)
LIFT_DZ            = 0.10  # m — how high to lift FL

IK_DAMPING  = 1e-3
IK_EVERY_N  = 10    # IK at 200 Hz, physics at 2000 Hz

# Wall-spawn keyframe name (must match scene_vert_opt.xml)
WALL_SPAWN_KEY = "wall_spawn"

PID_KP      = 500.0
PID_KI      = 200.0
PID_KD      = 30.0
PID_I_CLAMP = 100.0


# ── Magnetic force ─────────────────────────────────────────────────────────────

def _mag_force(dist: float, Br: float) -> float:
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m ** 2) / (2 * np.pi * (2 * dist) ** 4)


def _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params, off_mids=None):
    """Apply magnetic attraction from active magnets to the nearest steel plate."""
    if off_mids is None:
        off_mids = set()
    _fromto = np.zeros(6)
    for mid in magnet_ids:
        if mid in off_mids:
            continue
        fvec = np.zeros(3)
        for gid in sphere_gids[mid]:
            best_dist, best_fromto = np.inf, None
            for pid in plate_ids:
                d = mujoco.mj_geomDistance(model, data, gid, pid, 50.0, _fromto)
                if d < best_dist:
                    best_dist, best_fromto = d, _fromto.copy()
            if best_dist <= 0 or best_dist > params['max_magnetic_distance']:
                continue
            n    = best_fromto[3:6] - best_fromto[0:3]
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            fvec += _mag_force(best_dist, params['Br']) * (n / norm)
        total_mag = np.linalg.norm(fvec)
        if total_mag > params['max_force_per_wheel']:
            fvec *= params['max_force_per_wheel'] / total_mag
        data.xfrc_applied[mid, :3] += fvec


# ── Model setup ───────────────────────────────────────────────────────────────

def _setup_model(params: dict):
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP

    # Load wall-spawn keyframe (sets freejoint to wall position/orientation).
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, WALL_SPAWN_KEY)
    if key_id == -1:
        raise RuntimeError(
            f"Keyframe '{WALL_SPAWN_KEY}' not found in scene.xml. "
            "Add a <key name='wall_spawn' ...> element."
        )
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Overwrite joint angles with baked geometry values (same as floor sim).
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

    # Apply tuneable physics params to both floor and wall geoms.
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

    model.dof_damping[:]  = 2.0
    model.dof_armature[:] = 0.01

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


# ── PID controller ─────────────────────────────────────────────────────────────

class _PID:
    def __init__(self, model):
        self.nu        = model.nu
        self.ctrl_jids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.integral  = np.zeros(model.nu)
        self.prev_err  = np.zeros(model.nu)

    def compute(self, model, data, targets, dt):
        torques = np.zeros(self.nu)
        for i, jid in enumerate(self.ctrl_jids):
            qidx = model.jnt_qposadr[jid]
            err  = targets[i] - data.qpos[qidx]
            self.integral[i] = np.clip(
                self.integral[i] + err * dt, -PID_I_CLAMP, PID_I_CLAMP)
            derr = (err - self.prev_err[i]) / dt if dt > 0 else 0.0
            self.prev_err[i] = err
            torques[i] = PID_KP * err + PID_KI * self.integral[i] + PID_KD * derr
        return torques


# ── Headless wall FL-lift runner ──────────────────────────────────────────────

def run_headless_lift(params: dict) -> tuple[float, float, float]:
    """
    Spawn on wall, settle, then lift FL — identical to floor sim but on wall.

    Phases:
      1. Settle (0 → SETTLE_TIME): all magnets ON, PID holds bake pose.
      2. Lift   (SETTLE_TIME → total): FL raised LIFT_DZ, FL magnet OFF,
                 IK holds target, stance magnets ON.

    Returns:
        (mean_abs_x, mean_abs_y, mean_abs_z) of FR/BL/BR EEs:
          X — pull-off from wall.
          Y — lateral slip (not penalised).
          Z — gravity-direction drop (most critical on vertical wall).
        Sampled over [SETTLE_TIME + LIFT_MEASURE_START, total_time].
    """
    model, data, plate_ids, magnet_ids, sphere_gids = _setup_model(params)

    swing_mag_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{SWING_FOOT}")

    STANCE_FEET = [f for f in FEET if f != SWING_FOOT]
    stance_ee_bids = {
        foot: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{foot}")
        for foot in STANCE_FEET
    }

    # # ── Debug: verify magnet IDs and spawn gap ────────────────────────────────
    # print(f"[DEBUG] swing_mag_bid     = {swing_mag_bid}"
    #       f"  {'OK' if swing_mag_bid in magnet_ids else '*** NOT in magnet_ids — FL magnet will NOT turn off! ***'}")
    # print(f"[DEBUG] magnet_ids        = {magnet_ids}")
    # print(f"[DEBUG] stance_ee_bids    = {stance_ee_bids}")
    # fl_x   = data.xpos[swing_mag_bid][0] if swing_mag_bid != -1 else float('nan')
    # wall_x = 0.500  # inner face of wall geom from scene_vert_opt.xml
    # gap    = wall_x - fl_x
    # print(f"[DEBUG] FL ee world X     = {fl_x:.4f}m  |  wall face X = {wall_x:.4f}m  |  gap = {gap:.4f}m")
    # print(f"[DEBUG] max_magnetic_dist = {params['max_magnetic_distance']:.4f}m"
    #       f"  {'OK' if params['max_magnetic_distance'] >= gap else '*** TOO SHORT — magnets cannot reach wall from spawn! ***'}")
    # # ─────────────────────────────────────────────────────────────────────────

    ik  = _IK(model)
    pid = _PID(model)

    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]

    settled           = False
    z_settle_baseline = {}
    xy_lift_baseline  = {}
    target_pos        = np.zeros(3)
    ik_step_counter   = 0
    xy_drift_samples  = []
    z_drift_samples   = []

    total_time    = SETTLE_TIME + LIFT_HOLD
    measure_start = SETTLE_TIME + LIFT_MEASURE_START

    while data.time < total_time:
        t = data.time

        # ── Phase 1: settle ───────────────────────────────────────────────────
        if t < SETTLE_TIME:
            data.xfrc_applied[:] = 0
            _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params)
            data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
            mujoco.mj_step(model, data)
            continue

        # ── One-time snapshot at end of settle ────────────────────────────────
        if not settled:
            settled = True
            ik.record_stance(data)
            ee_home    = ik.ee_pos(data, SWING_FOOT).copy()
            target_pos = ee_home + np.array([-LIFT_DZ, 0.0, 0.0])  # pull off wall in -X (wall normal)
            for foot, bid in stance_ee_bids.items():
                pos = data.xpos[bid].copy()
                z_settle_baseline[foot] = pos[2]
                xy_lift_baseline[foot]  = pos[:2].copy()

        # ── Phase 2: lift hold ────────────────────────────────────────────────
        data.xfrc_applied[:] = 0
        _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                   params, off_mids={swing_mag_bid})

        ik_step_counter += 1
        if ik_step_counter >= IK_EVERY_N:
            ik_step_counter = 0
            ctrl_targets = ik.solve(target_pos, data, IK_EVERY_N * TIMESTEP)

        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

        if data.time < measure_start:
            continue

        z_drifts = np.array([
            abs(data.xpos[bid][2] - z_settle_baseline[foot])
            for foot, bid in stance_ee_bids.items()
        ])
        z_drift_samples.append(z_drifts.mean())

        xy_drifts = np.array([
            np.abs(data.xpos[bid][:2] - xy_lift_baseline[foot])
            for foot, bid in stance_ee_bids.items()
        ])
        xy_drift_samples.append(xy_drifts.mean(axis=0))

    if not z_drift_samples:
        return 0.0, 0.0, 0.0

    z_arr  = np.array(z_drift_samples)
    xy_arr = np.array(xy_drift_samples)
    return float(xy_arr.mean(axis=0)[0]), float(xy_arr.mean(axis=0)[1]), float(z_arr.mean())


# ── Interactive viewer ────────────────────────────────────────────────────────

_key_state = {"paused": True, "step_once": False}

def _key_callback(keycode):
    if keycode in (32, 257):   # SPACE or ENTER — toggle pause
        _key_state["paused"] = not _key_state["paused"]
    elif keycode == 262:        # RIGHT ARROW — step one frame
        _key_state["step_once"] = True


def run_with_viewer(params: dict) -> None:
    """
    Re-run the wall FL-lift scenario with a live MuJoCo viewer.

    Controls:
      SPACE / ENTER — pause / resume
      RIGHT ARROW   — step one physics frame while paused
    """
    import time
    import mujoco.viewer as mjviewer

    _key_state["paused"]    = True
    _key_state["step_once"] = False

    model, data, plate_ids, magnet_ids, sphere_gids = _setup_model(params)

    swing_mag_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{SWING_FOOT}")

    STANCE_FEET = [f for f in FEET if f != SWING_FOOT]
    stance_ee_bids = {
        foot: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{foot}")
        for foot in STANCE_FEET
    }

    ik  = _IK(model)
    pid = _PID(model)

    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]

    settled           = False
    z_settle_baseline = {}
    xy_lift_baseline  = {}
    target_pos        = np.zeros(3)
    ik_step_counter   = 0
    xy_drift_samples  = []
    z_drift_samples   = []

    total_time    = SETTLE_TIME + LIFT_HOLD
    measure_start = SETTLE_TIME + LIFT_MEASURE_START

    print("Wall FL-lift viewer — press SPACE to start, RIGHT ARROW to step while paused.")

    with mjviewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
        viewer.cam.lookat[:] = [0.3, 0.0, 0.5]
        viewer.cam.distance  = 1.2
        viewer.cam.elevation = -10
        viewer.cam.azimuth   = 160

        step_start = time.perf_counter()

        while viewer.is_running():
            if _key_state["paused"] and not _key_state["step_once"]:
                viewer.sync()
                time.sleep(0.02)
                continue
            _key_state["step_once"] = False

            if data.time < SETTLE_TIME:
                data.xfrc_applied[:] = 0
                _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params)
                data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
                mujoco.mj_step(model, data)

            elif data.time < total_time:
                if not settled:
                    settled = True
                    ik.record_stance(data)
                    ee_home    = ik.ee_pos(data, SWING_FOOT).copy()
                    target_pos = ee_home + np.array([-LIFT_DZ, 0.0, 0.0])  # pull off wall in -X (wall normal)
                    for foot, bid in stance_ee_bids.items():
                        pos = data.xpos[bid].copy()
                        z_settle_baseline[foot] = pos[2]
                        xy_lift_baseline[foot]  = pos[:2].copy()

                data.xfrc_applied[:] = 0
                _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                           params, off_mids={swing_mag_bid})

                ik_step_counter += 1
                if ik_step_counter >= IK_EVERY_N:
                    ik_step_counter = 0
                    ctrl_targets = ik.solve(target_pos, data, IK_EVERY_N * TIMESTEP)

                data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
                mujoco.mj_step(model, data)

                if data.time >= measure_start:
                    z_drifts = np.array([
                        abs(data.xpos[bid][2] - z_settle_baseline[foot])
                        for foot, bid in stance_ee_bids.items()
                    ])
                    z_drift_samples.append(z_drifts.mean())
                    xy_drifts = np.array([
                        np.abs(data.xpos[bid][:2] - xy_lift_baseline[foot])
                        for foot, bid in stance_ee_bids.items()
                    ])
                    xy_drift_samples.append(xy_drifts.mean(axis=0))

            viewer.sync()
            elapsed   = time.perf_counter() - step_start
            remaining = TIMESTEP - elapsed
            if remaining > 0:
                time.sleep(remaining)
            step_start = time.perf_counter()

        if z_drift_samples:
            z_arr  = np.array(z_drift_samples)
            xy_arr = np.array(xy_drift_samples)
            print(f"Viewer drift — x: {xy_arr.mean(axis=0)[0]*1000:.2f}mm  "
                  f"y: {xy_arr.mean(axis=0)[1]*1000:.2f}mm  "
                  f"z: {z_arr.mean()*1000:.2f}mm")


if __name__ == "__main__":
    import sys
    import argparse
    import json

    sys.path.insert(0, os.path.dirname(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--view", metavar="PARAMS_JSON",
                        help="Launch viewer using params from this JSON file")
    args = parser.parse_args()

    if args.view:
        with open(args.view) as f:
            params = json.load(f)
        params['ground_friction']   = list(params['ground_friction'])
        params['solref']            = list(params['solref'])
        params['solimp']            = list(params['solimp'])
        params['noslip_iterations'] = int(params['noslip_iterations'])
        print(f"Launching viewer with params from {args.view}")
        run_with_viewer(params)
    else:
        from sim_vertopt_config import PARAMS
        print("Running single wall-hold trial with default PARAMS...")
        x, y, z = run_headless_lift(PARAMS)
        print(f"mean drift — x: {x*1000:.2f}mm  y: {y*1000:.2f}mm  z: {z*1000:.2f}mm")
        print(f"(x=pull-off, y=lateral, z=gravity-direction slip)")