"""
sim_wallopt_sim.py — headless f2w runner for the FL wall-adhesion optimizer.

Phases executed:
  1. Settle   (SETTLE_TIME s)      — all magnets ON, robot drops onto floor
  2. FL f2w   (sequence "f2w")     — FL lifts, swings, orients, reaches wall
  3. FL plant                      — FL declared planted after DUAL_REACH_DWELL s in F2W_REACH;
                                     FL baseline position captured here
  4. FR f2w   (sequence "f2w_fr")  — FR executes its mirror sequence
     Sampling starts FR_MEASURE_DELAY s after FR runner starts (FR is in early LIFT).
     Sampling ends when FR has been in F2W_REACH for FR_REACH_DWELL s, at which
     point FR is also planted and the trial is done.

Reward signal (minimise):
  - FL EE XYZ drift from planted baseline (65% of cost)
  - BL/BR EE mean XYZ drift from their settle-time positions (35% of cost)
"""

import os
import sys
import numpy as np
import mujoco
import mink

# ── Paths ─────────────────────────────────────────────────────────────────────

OPTIMIZER_DIR = os.path.abspath(os.path.dirname(__file__))
LEGGED_DIR    = os.path.abspath(os.path.join(OPTIMIZER_DIR, ".."))
SCENE_XML     = os.path.join(LEGGED_DIR, "mwc_mjcf", "scene.xml")

for _p in (LEGGED_DIR, OPTIMIZER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    MU_0, MAGNET_VOLUME, MAGNET_BODY_NAMES,
    TIMESTEP,
    KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG,
    bake_joint_angles,
)
from sequences import SEQUENCES, SequenceRunner

# ── Constants ─────────────────────────────────────────────────────────────────

FEET             = ('FL', 'FR', 'BL', 'BR')
STANCE_FEET      = ('BL', 'BR')   # floor feet whose drift we also penalise

SETTLE_TIME      = 2.0   # s — all magnets ON
DUAL_REACH_DWELL = 3.0   # s — FL must hold F2W_REACH this long before declared planted
FR_MEASURE_DELAY = 0.5   # s after FR runner starts before sampling begins
                          #   (FR is in early LIFT; short delay skips the step-off transient)
FR_REACH_DWELL   = 2.0   # s FR must be in F2W_REACH before trial ends (clean stop condition)
                          #   F2W_REACH is unbounded so runner.done never fires; we use dwell instead

IK_DAMPING  = 1e-3
IK_EVERY_N  = 10    # IK solve every N physics steps

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
        tot = np.linalg.norm(fvec)
        if tot > params['max_force_per_wheel']:
            fvec *= params['max_force_per_wheel'] / tot
        data.xfrc_applied[mid, :3] += fvec


# ── Model setup ───────────────────────────────────────────────────────────────

def _setup_model(params: dict):
    bake_joint_angles(os.path.join(LEGGED_DIR, "mwc_mjcf", "robot.xml"))
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

    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root_jid != -1:
        data.qpos[model.jnt_qposadr[root_jid]] += 2.05
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
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in MAGNET_BODY_NAMES
    ]
    sphere_gids = {
        mid: [gid for gid in range(model.ngeom)
              if model.geom_bodyid[gid] == mid
              and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE]
        for mid in magnet_ids
    }
    return model, data, plate_ids, magnet_ids, sphere_gids


# ── PID controller ────────────────────────────────────────────────────────────

class _PID:
    def __init__(self, model):
        self.nu        = model.nu
        self.ctrl_jids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.integral  = np.zeros(model.nu)
        self.prev_err  = np.zeros(model.nu)

    def compute(self, model, data, targets, dt):
        torques = np.zeros(self.nu)
        for i, jid in enumerate(self.ctrl_jids):
            err = targets[i] - data.qpos[model.jnt_qposadr[jid]]
            self.integral[i] = np.clip(self.integral[i] + err * dt, -PID_I_CLAMP, PID_I_CLAMP)
            derr = (err - self.prev_err[i]) / dt if dt > 0 else 0.0
            self.prev_err[i] = err
            torques[i] = PID_KP * err + PID_KI * self.integral[i] + PID_KD * derr
        return torques


# ── IK solver (mirrors sim.py IKSolver with stance_se3 orientation locking) ───

def _se3_from_pos(pos):
    T = np.eye(4); T[:3, 3] = pos
    return mink.SE3.from_matrix(T)


class _IK:
    """Whole-body IK matching sim.py IKSolver, including wall-foot orientation lock."""

    def __init__(self, model):
        self.model  = model
        self.config = mink.Configuration(model)

        self.ee_bids    = {}
        self.foot_tasks = {}
        for foot in FEET:
            frame = f"electromagnet_{foot}"
            self.ee_bids[foot] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame)
            self.foot_tasks[foot] = mink.FrameTask(
                frame_name=frame, frame_type="body",
                position_cost=10.0, orientation_cost=0.0, lm_damping=IK_DAMPING)

        self.body_task = mink.FrameTask(
            frame_name="main_frame", frame_type="body",
            position_cost=50.0, orientation_cost=50.0, lm_damping=IK_DAMPING)
        self.posture_task = mink.PostureTask(model=model, cost=0.01, lm_damping=IK_DAMPING)
        self.config_limit = mink.ConfigurationLimit(model)
        self.ctrl_jids    = [model.actuator_trnid[i, 0] for i in range(model.nu)]

        self.stance_targets  = {}
        self.stance_se3      = {}   # foot → 4×4 SE3 for wall-mounted feet (orientation held)
        self._ori_target_rot = None

    def ee_pos(self, data, foot) -> np.ndarray:
        return data.xpos[self.ee_bids[foot]].copy()

    def record_stance(self, data):
        for foot in FEET:
            self.stance_targets[foot] = self.ee_pos(data, foot)
        self.posture_task.set_target(data.qpos.copy())
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")
        T = np.eye(4)
        T[:3, :3] = data.xmat[bid].reshape(3, 3)
        T[:3,  3] = data.xpos[bid]
        self.body_task.set_target(mink.SE3.from_matrix(T))

    def lock_stance_orientation(self, data, foot):
        """Snapshot full SE3 for a wall foot so IK holds its orientation."""
        bid = self.ee_bids[foot]
        T = np.eye(4)
        T[:3, :3] = data.xmat[bid].reshape(3, 3)
        T[:3,  3] = data.xpos[bid]
        self.stance_se3[foot] = T

    def solve(self, swing_target, phys_data, dt, swing_foot: str,
              face_axis=None, position_cost=None, orientation_cost=None,
              n_iter=10) -> np.ndarray:
        ik_qpos  = phys_data.qpos.copy()
        pos_cost = position_cost    if position_cost    is not None else 10.0
        ori_cost = orientation_cost if orientation_cost is not None else 0.0

        if face_axis is not None:
            if self._ori_target_rot is None:
                goal = np.asarray(face_axis, float); goal /= np.linalg.norm(goal)
                cur  = phys_data.xmat[self.ee_bids[swing_foot]].reshape(3, 3).copy()
                cur_neg_y = -cur[:, 1]
                axis  = np.cross(cur_neg_y, goal)
                sin_a = np.linalg.norm(axis)
                cos_a = float(np.dot(cur_neg_y, goal))
                if sin_a < 1e-6:
                    self._ori_target_rot = cur.copy() if cos_a >= 0 else (
                        (-np.eye(3) + 2 * np.outer(cur_neg_y, cur_neg_y)) @ cur)
                else:
                    axis /= sin_a
                    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
                    self._ori_target_rot = (np.eye(3) + sin_a*K + (1.-cos_a)*(K@K)) @ cur
            T = np.eye(4)
            T[:3, :3] = self._ori_target_rot
            T[:3,  3] = swing_target
            target_se3 = mink.SE3.from_matrix(T)
        else:
            self._ori_target_rot = None
            target_se3 = _se3_from_pos(swing_target)

        if self.foot_tasks[swing_foot].orientation_cost != ori_cost:
            self.foot_tasks[swing_foot] = mink.FrameTask(
                frame_name=f"electromagnet_{swing_foot}", frame_type="body",
                position_cost=pos_cost, orientation_cost=ori_cost, lm_damping=IK_DAMPING)
        self.foot_tasks[swing_foot].set_target(target_se3)
        self.foot_tasks[swing_foot].position_cost = pos_cost
        self.body_task.orientation_cost = 0.0 if face_axis is not None else 50.0

        for foot in FEET:
            if foot != swing_foot:
                if foot in self.stance_se3:
                    se3_tgt = mink.SE3.from_matrix(self.stance_se3[foot])
                    if self.foot_tasks[foot].orientation_cost != 30.0:
                        self.foot_tasks[foot] = mink.FrameTask(
                            frame_name=f"electromagnet_{foot}", frame_type="body",
                            position_cost=50.0, orientation_cost=30.0, lm_damping=IK_DAMPING)
                    self.foot_tasks[foot].set_target(se3_tgt)
                else:
                    self.foot_tasks[foot].set_target(_se3_from_pos(self.stance_targets[foot]))
                    self.foot_tasks[foot].position_cost = 50.0

        tasks = [self.body_task] + [self.foot_tasks[f] for f in FEET] + [self.posture_task]

        for _ in range(n_iter):
            self.config.update(ik_qpos)
            vel     = mink.solve_ik(self.config, tasks, dt,
                                    solver="quadprog", damping=IK_DAMPING,
                                    limits=[self.config_limit])
            ik_qpos = self.config.integrate(vel, dt)

        ctrl = np.zeros(self.model.nu)
        for i, jid in enumerate(self.ctrl_jids):
            ctrl[i] = ik_qpos[self.model.jnt_qposadr[jid]]
        return ctrl


# ── Main headless runner ──────────────────────────────────────────────────────

def run_headless_wall(params: dict) -> tuple[float, float, float, float]:
    """
    Run settle → FL f2w → FR f2w_fr headlessly.

    Sampling window:
      Start: FR_MEASURE_DELAY s after FR runner starts (FR in early LIFT)
      End:   FR has been in F2W_REACH for FR_REACH_DWELL s  ← clean stop, no runaway

    Returns:
        (fl_x, fl_y, fl_z, stance_xyz) all in metres.
        fl_*:      mean absolute XYZ drift of FL EE from its planted baseline.
        stance_xyz: mean absolute XYZ drift of BL+BR EE from their settle positions.
        Returns (1, 1, 1, 1) on failure (FL never reached wall, etc.).
    """
    model, data, plate_ids, magnet_ids, sphere_gids = _setup_model(params)
    ik  = _IK(model)
    pid = _PID(model)

    fl_mag_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "electromagnet_FL")
    fr_mag_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "electromagnet_FR")
    fl_hip_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_pitch_FL")
    fr_hip_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_pitch_FR")

    _wall_gid      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    _ray_geomgroup = np.zeros(6, np.uint8)
    _geomid_out    = np.array([-1], dtype=np.int32)

    def _wall_dist_fn(ee_bid, hip_jid, ray_dir_override=None):
        pos     = data.xpos[ee_bid].copy()
        ray_dir = (np.array(ray_dir_override, float) if ray_dir_override is not None
                   else -data.xmat[ee_bid].reshape(3, 3)[:, 1])
        ray_dir /= np.linalg.norm(ray_dir)
        _geomid_out[0] = -1
        dist = mujoco.mj_ray(model, data, pos, ray_dir, _ray_geomgroup, 1, ee_bid, _geomid_out)
        if dist < 0 or int(_geomid_out[0]) != _wall_gid:
            return np.inf
        return float(dist)

    ctrl_targets = np.array([
        data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
        for i in range(model.nu)
    ])

    # ── Mutable swing-foot state ───────────────────────────────────────────────
    swing_foot_ref    = ["FL"]
    hip_jid_ref       = [fl_hip_jid]
    swing_mag_bid_ref = [fl_mag_bid]
    swing_off         = {fl_mag_bid}

    runner      = SequenceRunner(SEQUENCES["f2w"])
    fr_started  = [False]
    fr_start_t  = [None]   # sim time when FR runner begins
    fl_baseline = [None]   # FL EE XYZ at plant time

    # BL/BR baselines — captured at settle time
    stance_baselines: dict[str, np.ndarray] = {}

    # Sample accumulators
    fl_drift_samples:     list[np.ndarray] = []   # (3,) abs XYZ each step
    stance_drift_samples: list[float]      = []   # scalar mean abs XYZ each step

    def _start_runner_for(foot: str, seq_name: str, t: float):
        nonlocal runner
        runner = SequenceRunner(SEQUENCES[seq_name])
        ik._ori_target_rot = None
        ee_home = ik.ee_pos(data, foot).copy()
        runner.start(t, {
            'foot':              foot,
            'ee_home':           ee_home,
            'ee_pos_fn':         lambda: ik.ee_pos(data, swing_foot_ref[0]),
            'hip_pivot_fn':      lambda: data.xanchor[hip_jid_ref[0]].copy(),
            'wall_dist_fn':      lambda ray_dir_override=None: _wall_dist_fn(
                                     ik.ee_bids[swing_foot_ref[0]],
                                     hip_jid_ref[0], ray_dir_override),
            'magnet_disable_fn': lambda: swing_off.add(swing_mag_bid_ref[0]),
            'magnet_enable_fn':  lambda: swing_off.discard(swing_mag_bid_ref[0]),
        })

    settled    = False
    target_pos = np.zeros(3)
    face_axis  = None
    ik_counter = [0]

    # Budget: settle + FL f2w (~12s) + dwell + FR f2w (~12s) + reach dwell
    max_time = SETTLE_TIME + 12.0 + DUAL_REACH_DWELL + 12.0 + FR_REACH_DWELL + 2.0

    while data.time < max_time:
        t = data.time

        # ── Settle ────────────────────────────────────────────────────────────
        if t < SETTLE_TIME:
            data.xfrc_applied[:] = 0
            _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params)
            data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
            mujoco.mj_step(model, data)
            continue

        # ── Post-settle init (once) ────────────────────────────────────────────
        if not settled:
            settled = True
            ik.record_stance(data)
            target_pos = ik.ee_pos(data, "FL").copy()
            _start_runner_for("FL", "f2w", t)
            # Capture BL/BR baselines at settle — these are the "correct" positions
            for foot in STANCE_FEET:
                stance_baselines[foot] = ik.ee_pos(data, foot).copy()

        # ── Magnets ───────────────────────────────────────────────────────────
        data.xfrc_applied[:] = 0
        _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params,
                   off_mids=swing_off)

        # ── FL→FR transition ──────────────────────────────────────────────────
        if not fr_started[0]:
            ph = runner.current_phase
            fl_planted = (
                runner.done
                or (ph is not None
                    and ph.name == "F2W_REACH"
                    and (t - runner.phase_t0) >= DUAL_REACH_DWELL)
            )
            if fl_planted:
                fr_started[0] = True
                fr_start_t[0] = t
                runner.force_complete()
                ik.record_stance(data)
                ik.lock_stance_orientation(data, "FL")
                fl_baseline[0] = ik.ee_pos(data, "FL").copy()
                swing_foot_ref[0]    = "FR"
                hip_jid_ref[0]       = fr_hip_jid
                swing_mag_bid_ref[0] = fr_mag_bid
                swing_off.clear()
                swing_off.add(fr_mag_bid)
                _start_runner_for("FR", "f2w_fr", t)

        # ── IK solve ──────────────────────────────────────────────────────────
        ik_counter[0] += 1
        if ik_counter[0] >= IK_EVERY_N:
            ik_counter[0] = 0
            sf = swing_foot_ref[0]
            target_pos, face_axis, pos_cost, ori_cost = runner.step(t, {
                'ee_pos_fn':    lambda: ik.ee_pos(data, swing_foot_ref[0]),
                'hip_pivot_fn': lambda: data.xanchor[hip_jid_ref[0]].copy(),
            })
            ctrl_targets = ik.solve(
                target_pos, data, IK_EVERY_N * TIMESTEP,
                swing_foot=sf, face_axis=face_axis,
                position_cost=pos_cost, orientation_cost=ori_cost,
            )

        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

        # ── Sampling: from FR_MEASURE_DELAY after FR starts ───────────────────
        # This captures the ENTIRE FR swing, orient, and reach — all the moments
        # that perturb the body and stress FL's wall hold.
        if fr_started[0] and t >= fr_start_t[0] + FR_MEASURE_DELAY:
            # FL drift from its planted baseline
            fl_now = ik.ee_pos(data, "FL")
            fl_drift_samples.append(np.abs(fl_now - fl_baseline[0]))

            # BL/BR drift from their settle-time positions
            stance_drifts = [
                np.abs(ik.ee_pos(data, foot) - stance_baselines[foot]).mean()
                for foot in STANCE_FEET
            ]
            stance_drift_samples.append(float(np.mean(stance_drifts)))

        # ── Stop condition: FR has dwelt in F2W_REACH long enough ─────────────
        # runner.done never fires (F2W_REACH is unbounded), so we use dwell time.
        if fr_started[0]:
            ph = runner.current_phase
            if (ph is not None
                    and ph.name == "F2W_REACH"
                    and (t - runner.phase_t0) >= FR_REACH_DWELL):
                break

    if not fl_drift_samples:
        return 1.0, 1.0, 1.0, 1.0

    fl_arr = np.array(fl_drift_samples)        # (N, 3)
    fl_mean = fl_arr.mean(axis=0)              # mean abs drift per axis
    stance_mean = float(np.mean(stance_drift_samples))

    return float(fl_mean[0]), float(fl_mean[1]), float(fl_mean[2]), stance_mean


if __name__ == "__main__":
    from sim_wallopt_config import PARAMS
    print("Running single wall-adhesion trial with default PARAMS...")
    fl_x, fl_y, fl_z, stance = run_headless_wall(PARAMS)
    print(f"FL wall drift  — x: {fl_x*1000:.2f}mm  y: {fl_y*1000:.2f}mm  z: {fl_z*1000:.2f}mm")
    print(f"BL/BR stance   — mean: {stance*1000:.2f}mm")