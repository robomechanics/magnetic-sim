"""
sim_opt_sim.py — headless wall-hold runner for the sim optimizer.

Runs:
  Phase 1: settle (2 s, all magnets ON, all joints hold bake pose)
  Phase 2: hold   (3 s, all magnets ON, all joints hold bake pose)

The robot is spawned on the vertical wall (inner face at X=0.500) via the
"wall_spawn" keyframe defined in scene_vert_opt.xml.  No foot is lifted; no IK is
run.  The PID simply holds the post-bake joint angles for the full 5 s.

Slip metric (returned as mean_abs_x/y/z, lower = better):
  X — mean absolute X drift of all 4 EE bodies from their settle position
      (positive = pulling away from wall; magnets resist this).
  Y — mean absolute lateral drift of all 4 EE bodies (side-to-side slip).
  Z — mean absolute vertical drift of all 4 EE bodies (gravity-direction
      slip; most critical for wall climbing).

These feed directly into calculate_cost() in sim_vertopt_config.py:
  total = 0.50 * Z  +  0.25 * X  +  0.25 * Y
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

# ── Constants ─────────────────────────────────────────────────────────────────

FEET = ('FL', 'FR', 'BL', 'BR')

SETTLE_TIME         = 2.0   # s — all magnets ON, hold bake pose
HOLD_TIME           = 3.0   # s — continue holding; sample drift here
HOLD_MEASURE_START  = 0.5   # s into hold before sampling (skip transient)

# Wall-spawn keyframe name (must match scene.xml)
WALL_SPAWN_KEY = "wall_spawn"

PID_KP      = 500.0
PID_KI      = 200.0
PID_KD      = 30.0
PID_I_CLAMP = 100.0


# ── Magnetic force ─────────────────────────────────────────────────────────────

def _mag_force(dist: float, Br: float) -> float:
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m ** 2) / (2 * np.pi * (2 * dist) ** 4)


def _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params):
    """Apply magnetic attraction from all magnets to the nearest steel plate."""
    _fromto = np.zeros(6)
    for mid in magnet_ids:
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


# ── Headless wall-hold runner ─────────────────────────────────────────────────

def run_headless_lift(params: dict) -> tuple[float, float, float]:
    """
    Spawn on wall, all magnets ON throughout, hold joint positions.

    Baseline is captured at t=0 (spawn pose after mj_forward) so the full
    drop during the settle transient is included in the cost — not hidden.
    Sampling runs from the very first step through to total_time.

    Returns:
        (mean_abs_x, mean_abs_y, mean_abs_z)
          X — mean |X drift| of all 4 EE bodies (wall penetration / pull-off).
          Y — mean |Y drift| of all 4 EE bodies (lateral slip, not penalised).
          Z — mean |Z drift| of all 4 EE bodies (gravity-direction drop).
        Sampled over every timestep from t=0 to total_time.
    """
    model, data, plate_ids, magnet_ids, sphere_gids = _setup_model(params)

    # Resolve EE body IDs for all four feet.
    ee_bids = {
        foot: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{foot}")
        for foot in FEET
    }

    pid = _PID(model)

    # PID targets = post-bake joint angles (held for entire simulation).
    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]

    # Baseline at spawn — capture before any physics steps.
    ee_baseline = {
        foot: data.xpos[bid].copy()
        for foot, bid in ee_bids.items()
    }

    xyz_samples = []   # list of arrays shape (4, 3), one per timestep
    total_time  = SETTLE_TIME + HOLD_TIME

    while data.time < total_time:
        # All magnets ON for the entire simulation.
        data.xfrc_applied[:] = 0
        _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params)
        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

        # Sample drift from spawn position every step.
        drift = np.array([
            np.abs(data.xpos[bid] - ee_baseline[foot])
            for foot, bid in ee_bids.items()
        ])  # shape (4, 3)
        xyz_samples.append(drift)

    if not xyz_samples:
        return 0.0, 0.0, 0.0

    arr = np.array(xyz_samples)            # shape (N, 4, 3)
    mean_per_axis = arr.mean(axis=(0, 1))  # shape (3,)  mean over time and feet
    return float(mean_per_axis[0]), float(mean_per_axis[1]), float(mean_per_axis[2])


# ── Interactive viewer ────────────────────────────────────────────────────────

_key_state = {"paused": True, "step_once": False}

def _key_callback(keycode):
    if keycode in (32, 257):   # SPACE or ENTER — toggle pause
        _key_state["paused"] = not _key_state["paused"]
    elif keycode == 262:        # RIGHT ARROW — step one frame
        _key_state["step_once"] = True


def run_with_viewer(params: dict) -> None:
    """
    Re-run the wall-hold scenario with a live MuJoCo viewer.

    Controls:
      SPACE / ENTER — pause / resume
      RIGHT ARROW   — step one physics frame while paused
      (starts paused so you can inspect the spawn pose first)

    The viewer stays open after the sim ends; close the window to exit.
    """
    import time
    import mujoco.viewer as mjviewer

    _key_state["paused"]    = True
    _key_state["step_once"] = False

    model, data, plate_ids, magnet_ids, sphere_gids = _setup_model(params)

    ee_bids = {
        foot: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{foot}")
        for foot in FEET
    }

    pid = _PID(model)
    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]

    # Baseline at spawn — before any physics steps.
    ee_baseline = {
        foot: data.xpos[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{foot}")].copy()
        for foot in FEET
    }
    xyz_samples = []
    total_time  = SETTLE_TIME + HOLD_TIME

    print("Viewer ready — press SPACE to start, RIGHT ARROW to step while paused.")

    with mjviewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
        viewer.cam.lookat[:] = [0.3, 0.0, 0.5]
        viewer.cam.distance  = 1.2
        viewer.cam.elevation = -10
        viewer.cam.azimuth   = 160

        step_start = time.perf_counter()

        while viewer.is_running():
            # ── Paused: just sync and idle ─────────────────────────────────
            if _key_state["paused"] and not _key_state["step_once"]:
                viewer.sync()
                time.sleep(0.02)
                continue
            _key_state["step_once"] = False

            # ── Physics step ───────────────────────────────────────────────
            if data.time < total_time:
                data.xfrc_applied[:] = 0
                _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params)
                data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
                mujoco.mj_step(model, data)

                # Sample drift from spawn position every step.
                drift = np.array([
                    np.abs(data.xpos[bid] - ee_baseline[foot])
                    for foot, bid in ee_bids.items()
                ])
                xyz_samples.append(drift)

            viewer.sync()

            # Pace to real time
            elapsed   = time.perf_counter() - step_start
            remaining = TIMESTEP - elapsed
            if remaining > 0:
                time.sleep(remaining)
            step_start = time.perf_counter()

        if xyz_samples:
            arr = np.array(xyz_samples)
            mean_per_axis = arr.mean(axis=(0, 1))
            print(f"Viewer drift — x: {mean_per_axis[0]*1000:.2f}mm  "
                  f"y: {mean_per_axis[1]*1000:.2f}mm  "
                  f"z: {mean_per_axis[2]*1000:.2f}mm")


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