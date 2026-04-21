"""
sim_opt_sim.py — headless lift-only runner for the sim optimizer.

Runs:
  Phase 1: settle (2 s, all magnets ON)
  Phase 2: lift FL 10 cm and hold (3 s, FL magnet OFF, stance magnets ON)
  [spin and drop phases are commented out / not executed]

Returns mean absolute body XYZ drift during the lift hold, relative to the
body position recorded at settle time.
"""

import os
import sys
import numpy as np
import mujoco
import mink

# ── Paths ────────────────────────────────────────────────────────────────────
# sim_opt_sim.py lives in legged_sim/optimizer/, so ".." is legged_sim/ itself.

LEGGED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCENE_XML  = os.path.join(LEGGED_DIR, "mwc_mjcf", "scene.xml")

if LEGGED_DIR not in sys.path:
    sys.path.insert(0, LEGGED_DIR)

from config import (
    MU_0, MAGNET_VOLUME, MAGNET_BODY_NAMES,
    TIMESTEP,
    KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG,
)

# ── Constants ─────────────────────────────────────────────────────────────────

FEET       = ('FL', 'FR', 'BL', 'BR')
SWING_FOOT = "FL"

SETTLE_TIME = 2.0   # s
LIFT_HOLD         = 3.0   # s — hold lifted pose, no spin/drop
LIFT_MEASURE_START = 1.0   # s into lift before sampling — skip foot-travel transient
LIFT_DZ     = 0.10  # m — how high to lift FL

IK_DAMPING  = 1e-3
IK_EVERY_N  = 10    # IK at 200 Hz, physics at 2000 Hz

PID_KP      = 500.0
PID_KI      = 200.0
PID_KD      = 30.0
PID_I_CLAMP = 100.0
# ── Magnetic force ────────────────────────────────────────────────────────────

def _mag_force(dist: float, Br: float) -> float:
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m ** 2) / (2 * np.pi * (2 * dist) ** 4)
def _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params, off_mids=None):
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
    mujoco.mj_resetData(model, data)

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

    model.opt.o_solref         = params['solref']
    model.opt.o_solimp         = params['solimp']
    model.opt.noslip_iterations = params['noslip_iterations']
    model.opt.noslip_tolerance  = params['noslip_tolerance']
    model.opt.o_margin          = params['margin']

    model.dof_damping[:] = 2.0
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
# ── PID controller ────────────────────────────────────────────────────────────

class _PID:
    def __init__(self, model):
        self.nu       = model.nu
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
# ── IK solver (minimal — same as sim.py) ─────────────────────────────────────

def _se3_from_pos(pos):
    T = np.eye(4)
    T[:3, 3] = pos
    return mink.SE3.from_matrix(T)
class _IK:
    def __init__(self, model):
        self.model  = model
        self.config = mink.Configuration(model)

        self.foot_tasks = {}
        self.ee_bids    = {}
        for foot in FEET:
            frame = f"electromagnet_{foot}"
            self.ee_bids[foot] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, frame)
            self.foot_tasks[foot] = mink.FrameTask(
                frame_name=frame, frame_type="body",
                position_cost=10.0, orientation_cost=0.0,
                lm_damping=IK_DAMPING,
            )

        self.body_task = mink.FrameTask(
            frame_name="main_frame", frame_type="body",
            position_cost=50.0, orientation_cost=50.0,
            lm_damping=IK_DAMPING,
        )
        self.posture_task = mink.PostureTask(
            model=model, cost=0.01, lm_damping=IK_DAMPING,
        )
        self.config_limit = mink.ConfigurationLimit(model)

        frozen_dofs = []
        for i in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jname and (jname.startswith('ee2_') or jname.startswith('em_z_')):
                frozen_dofs.append(model.jnt_dofadr[i])
        self.freeze_passive = mink.DofFreezingTask(
            model=model, dof_indices=frozen_dofs,
        ) if frozen_dofs else None

        self.ctrl_jids     = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.stance_targets = {}
        self.body_target    = None

    def ee_pos(self, data, foot):
        return data.xpos[self.ee_bids[foot]].copy()

    def record_stance(self, data):
        for foot in FEET:
            self.stance_targets[foot] = self.ee_pos(data, foot)
        self.posture_task.set_target(data.qpos.copy())

        body_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")
        body_pos = data.xpos[body_bid].copy()
        body_mat = data.xmat[body_bid].reshape(3, 3).copy()
        T = np.eye(4)
        T[:3, :3] = body_mat
        T[:3, 3]  = body_pos
        self.body_target = mink.SE3.from_matrix(T)
        self.body_task.set_target(self.body_target)

    def solve(self, swing_target, phys_data, dt, n_iter=10):
        ik_qpos = phys_data.qpos.copy()

        self.foot_tasks[SWING_FOOT].set_target(_se3_from_pos(swing_target))
        self.foot_tasks[SWING_FOOT].position_cost = 10.0
        for foot in FEET:
            if foot != SWING_FOOT:
                self.foot_tasks[foot].set_target(
                    _se3_from_pos(self.stance_targets[foot]))
                self.foot_tasks[foot].position_cost = 50.0

        tasks = [self.body_task] + [self.foot_tasks[f] for f in FEET] + [self.posture_task]
        if self.freeze_passive:
            tasks.append(self.freeze_passive)

        for _ in range(n_iter):
            self.config.update(ik_qpos)
            velocity = mink.solve_ik(
                self.config, tasks, dt,
                solver="quadprog", damping=IK_DAMPING,
                limits=[self.config_limit],
            )
            ik_qpos = self.config.integrate(velocity, dt)

        ctrl_targets = np.zeros(self.model.nu)
        for i, jid in enumerate(self.ctrl_jids):
            ctrl_targets[i] = ik_qpos[self.model.jnt_qposadr[jid]]
        return ctrl_targets
# ── Headless lift runner ──────────────────────────────────────────────────────

def run_headless_lift(params: dict) -> tuple[float, float, float]:
    """
    Run settle + lift-only phase with the given params.

    Phases executed:
      1. Settle  (0 → SETTLE_TIME): all magnets ON, PID hold
      2. Lift    (SETTLE_TIME → SETTLE_TIME + LIFT_HOLD): FL lifted LIFT_DZ,
                  FL magnet OFF, stance magnets ON
      [spin and drop phases are NOT executed]

    Returns:
        (mean_abs_x, mean_abs_y, mean_abs_z) where:
          XY — mean absolute lateral drift of FR/BL/BR electromagnets,
               sampled over [SETTLE_TIME+LIFT_MEASURE_START, total_time],
               relative to their XY positions at lift-start.
          Z  — mean absolute Z deviation of FR/BL/BR electromagnets over
               the same window, relative to their Z at END of settle
               (4 feet down, all magnets ON). Captures penetration change
               from redistributed magnetic load when FL lifts off.
               Sampling starts after LIFT_MEASURE_START to skip the
               foot-travel transient (foot still rising toward target).
    """
    model, data, plate_ids, magnet_ids, sphere_gids = _setup_model(params)

    swing_mag_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{SWING_FOOT}")

    # Stance feet whose electromagnet positions we measure
    STANCE_FEET = [f for f in FEET if f != SWING_FOOT]  # FR, BL, BR
    stance_ee_bids = {
        foot: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{foot}")
        for foot in STANCE_FEET
    }

    ik  = _IK(model)
    pid = _PID(model)

    # Initial ctrl targets from default qpos
    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]

    settled           = False
    z_settle_baseline = {}  # foot -> Z at END of settle (4 feet down, all magnets ON)
    xy_lift_baseline  = {}  # foot -> XY at moment FL magnet turns OFF
    target_pos        = np.zeros(3)
    ik_step_counter   = 0
    xy_drift_samples  = []
    z_drift_samples   = []

    total_time    = SETTLE_TIME + LIFT_HOLD
    measure_start = SETTLE_TIME + LIFT_MEASURE_START  # skip foot-travel transient

    while data.time < total_time:
        t = data.time

        # ── Phase 1: settle (all magnets ON) ──────────────────────────────────
        if t < SETTLE_TIME:
            data.xfrc_applied[:] = 0
            _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params)
            data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
            mujoco.mj_step(model, data)
            continue

        # ── One-time snapshot at end of settle ────────────────────────────────
        # Taken before FL magnet turns OFF so Z baseline reflects 4-foot
        # magnetic configuration (full penetration at rest).
        if not settled:
            settled = True
            ik.record_stance(data)
            ee_home    = ik.ee_pos(data, SWING_FOOT).copy()
            target_pos = ee_home + np.array([0.0, 0.0, LIFT_DZ])
            for foot, bid in stance_ee_bids.items():
                pos = data.xpos[bid].copy()
                z_settle_baseline[foot] = pos[2]   # Z with 4 feet + all magnets
                xy_lift_baseline[foot]  = pos[:2].copy()

        # ── Phase 2: lift hold (FL magnet OFF, stance magnets ON) ─────────────
        data.xfrc_applied[:] = 0
        _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                   params, off_mids={swing_mag_bid})

        ik_step_counter += 1
        if ik_step_counter >= IK_EVERY_N:
            ik_step_counter = 0
            ctrl_targets = ik.solve(target_pos, data, IK_EVERY_N * TIMESTEP)

        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

        # Only sample after foot has had time to reach lifted position
        if data.time < measure_start:
            continue

        # Z: how much do stance feet deviate from their settle-phase Z?
        #    Captures penetration change caused by redistributed magnetic load.
        z_drifts = np.array([
            abs(data.xpos[bid][2] - z_settle_baseline[foot])
            for foot, bid in stance_ee_bids.items()
        ])
        z_drift_samples.append(z_drifts.mean())

        # XY: lateral drift of stance feet from lift-start position
        xy_drifts = np.array([
            np.abs(data.xpos[bid][:2] - xy_lift_baseline[foot])
            for foot, bid in stance_ee_bids.items()
        ])
        xy_drift_samples.append(xy_drifts.mean(axis=0))

    if not z_drift_samples:
        return 0.0, 0.0, 0.0

    z_arr  = np.array(z_drift_samples)   # shape (N,)
    xy_arr = np.array(xy_drift_samples)  # shape (N, 2)
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
    Re-run the floor lift scenario with a live MuJoCo viewer.

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

    total_time    = SETTLE_TIME + LIFT_HOLD
    measure_start = SETTLE_TIME + LIFT_MEASURE_START

    print("Viewer ready — press SPACE to start, RIGHT ARROW to step while paused.")

    with mjviewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.1]
        viewer.cam.distance  = 1.2
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 45

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
                    target_pos = ee_home + np.array([0.0, 0.0, LIFT_DZ])
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

            viewer.sync()
            elapsed   = time.perf_counter() - step_start
            remaining = TIMESTEP - elapsed
            if remaining > 0:
                time.sleep(remaining)
            step_start = time.perf_counter()


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
        print(f"Launching floor-lift viewer with params from {args.view}")
        run_with_viewer(params)
    else:
        from sim_opt_config import PARAMS
        print("Running single trial with default PARAMS...")
        x, y, z = run_headless_lift(PARAMS)
        print(f"mean drift — x: {x*1000:.2f}mm  y: {y*1000:.2f}mm  z: {z*1000:.2f}mm")