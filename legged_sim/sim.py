"""
MWC sim: robot on ground, IK-controlled FL foot swing sequence.
Trajectory logic lives in sequences.py; this file owns sim, IK, PID, and viz.

Usage:
    python sim.py                          # GUI, basic sequence
    python sim.py --sequence orient        # GUI, orient sequence
    python sim.py --headless --sequence basic
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import mink
from scipy.optimize import minimize, differential_evolution

# ── paths ──────────────────────────────────────────────────────────────
LEGGED_DIR = os.path.join(os.path.dirname(__file__), "..", "legged_sim")
SCENE_XML  = os.path.join(LEGGED_DIR, "mwc_mjcf", "scene.xml")
sys.path.insert(0, LEGGED_DIR)

from config import (
    MU_0, MAGNET_VOLUME, MAGNET_BODY_NAMES, PARAMS,
    TIMESTEP, JOINT_DAMPING, JOINT_ARMATURE,
    KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG,
    bake_joint_angles, SEQUENCE as DEFAULT_SEQUENCE,
)
from sequences import SEQUENCES, SequenceRunner

# ── sim constants ───────────────────────────────────────────────────────
FEET        = ('FL', 'FR', 'BL', 'BR')
SWING_FOOT  = "FL"
IK_DAMPING  = 1e-3
SETTLE_TIME = 2.0
IK_EVERY_N  = 10

PID_KP, PID_KI, PID_KD, PID_I_CLAMP = 500.0, 200.0, 30.0, 100.0


# ── magnetic force ─────────────────────────────────────────────────────
def mag_force(dist, Br):
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)

def apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, off_mids=None):
    if off_mids is None:
        off_mids = set()
    _ft = np.zeros(6)
    for mid in magnet_ids:
        if mid in off_mids:
            continue
        fvec = np.zeros(3)
        for gid in sphere_gids[mid]:
            best_d, best_ft = np.inf, None
            for pid in plate_ids:
                d = mujoco.mj_geomDistance(model, data, gid, pid, 50.0, _ft)
                if d < best_d:
                    best_d, best_ft = d, _ft.copy()
            if best_d <= 0 or best_d > PARAMS['max_magnetic_distance']:
                continue
            n = best_ft[3:6] - best_ft[0:3]
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            fvec += mag_force(best_d, PARAMS['Br']) * (n / norm)
        tot = np.linalg.norm(fvec)
        if tot > PARAMS['max_force_per_wheel']:
            fvec *= PARAMS['max_force_per_wheel'] / tot
        data.xfrc_applied[mid, :3] += fvec


# ── model setup ────────────────────────────────────────────────────────
def setup_model():
    bake_joint_angles(os.path.join(LEGGED_DIR, "mwc_mjcf", "robot.xml"))
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP
    mujoco.mj_resetData(model, data)

    for leg in FEET:
        for jname, bake in [(f'knee_{leg}',  KNEE_BAKE_DEG),
                            (f'wrist_{leg}', WRIST_BAKE_DEG),
                            (f'ee_{leg}',    EE_BAKE_DEG)]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                data.qpos[model.jnt_qposadr[jid]] = np.radians(bake[leg])
    mujoco.mj_forward(model, data)

    # shift robot: net +2.50 in qadr+0 (was +2.0, moved 0.50 in +X toward wall at X=0.5)
    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root_jid != -1:
        qadr = model.jnt_qposadr[root_jid]
        data.qpos[qadr + 0] += 2.05
        mujoco.mj_forward(model, data)

    plate_ids = set()
    for name in ("floor", "wall"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        plate_ids.add(gid)
        model.geom_friction[gid] = PARAMS['ground_friction']
    for i in range(model.ngeom):
        if i not in plate_ids:
            model.geom_rgba[i, 3] = 0.5

    model.opt.o_solref          = PARAMS['solref']
    model.opt.o_solimp          = PARAMS['solimp']
    model.opt.noslip_iterations = PARAMS['noslip_iterations']
    model.opt.noslip_tolerance  = PARAMS['noslip_tolerance']
    model.opt.o_margin          = PARAMS['margin']
    model.dof_damping[:]  = 2.0
    model.dof_armature[:] = 0.01

    magnet_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                  for n in MAGNET_BODY_NAMES]
    sphere_gids = {
        mid: [gid for gid in range(model.ngeom)
              if model.geom_bodyid[gid] == mid
              and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE]
        for mid in magnet_ids
    }
    return model, data, plate_ids, magnet_ids, sphere_gids


# ── SE3 helpers ────────────────────────────────────────────────────────
def _se3_pos(pos):
    T = np.eye(4); T[:3, 3] = pos
    return mink.SE3.from_matrix(T)

def _se3_pos_rot(pos, face_axis, current_rot=None):
    # Aligns EE local -Y to face_axis using the minimal rotation from current_rot.
    # If current_rot is None, falls back to Gram-Schmidt (used only on first call).
    #
    # Why minimal rotation: mink FrameTask minimises full-SO3 error (all 3 axes).
    # Gram-Schmidt picks arbitrary X/Z which mink then fights to reach, causing
    # poor convergence. Minimal rotation keeps X/Z close to their current values
    # so the only non-trivial error is the -Y axis we actually care about.
    goal = np.asarray(face_axis, float)
    goal /= np.linalg.norm(goal)

    if current_rot is not None:
        cur_neg_y = -current_rot[:, 1]          # current EE local -Y in world frame
        axis      = np.cross(cur_neg_y, goal)
        sin_a     = np.linalg.norm(axis)
        cos_a     = float(np.dot(cur_neg_y, goal))

        if sin_a < 1e-6:
            # already aligned or exactly anti-aligned
            if cos_a >= 0:
                R = current_rot.copy()
            else:
                # 180° rotation around any axis perpendicular to cur_neg_y
                perp  = np.array([1., 0., 0.]) if abs(cur_neg_y[0]) < 0.9 else np.array([0., 1., 0.])
                perp -= np.dot(perp, cur_neg_y) * cur_neg_y
                perp /= np.linalg.norm(perp)
                K     = np.array([[     0, -perp[2],  perp[1]],
                                  [ perp[2],      0, -perp[0]],
                                  [-perp[1],  perp[0],      0]])
                R = (-np.eye(3) + 2 * np.outer(perp, perp)) @ current_rot
        else:
            axis /= sin_a
            K     = np.array([[    0, -axis[2],  axis[1]],
                              [ axis[2],     0, -axis[0]],
                              [-axis[1],  axis[0],     0]])
            R_delta = np.eye(3) + sin_a * K + (1. - cos_a) * (K @ K)
            R = R_delta @ current_rot
    else:
        # Gram-Schmidt fallback (first call before any current_rot is available)
        neg_y = goal
        y     = -neg_y
        up    = np.array([0., 0., 1.]) if abs(y[2]) < 0.9 else np.array([0., 1., 0.])
        x     = np.cross(up, y); x /= np.linalg.norm(x)
        z     = np.cross(x, y)
        R     = np.column_stack([x, y, z])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = pos
    return mink.SE3.from_matrix(T)


# ── IK solver ──────────────────────────────────────────────────────────
class IKSolver:
    def __init__(self, model):
        self.model  = model
        self.config = mink.Configuration(model)
        self.foot_tasks, self.ee_bids = {}, {}
        for foot in FEET:
            frame = f"electromagnet_{foot}"
            self.ee_bids[foot] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, frame)
            self.foot_tasks[foot] = mink.FrameTask(
                frame_name=frame, frame_type="body",
                position_cost=10.0, orientation_cost=0.0,
                lm_damping=IK_DAMPING)
        self.body_task = mink.FrameTask(
            frame_name="main_frame", frame_type="body",
            position_cost=50.0, orientation_cost=50.0,
            lm_damping=IK_DAMPING)
        self.posture_task = mink.PostureTask(
            model=model, cost=0.01, lm_damping=IK_DAMPING)
        self.config_limit = mink.ConfigurationLimit(model)

        frozen_dofs = []  # ee2_ and em_z_ free; stiffness set in XML
        self.freeze_passive = None
        print(f"[ik] frozen passive DOFs: {frozen_dofs}")

        self.ctrl_jids      = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.stance_targets = {}
        self._ik_qpos       = None
        self._warn_counter  = 0          # throttle: print warning every N solves
        self._hold_start_t  = None       # sim time when HOLD phase began
        self._hold_start_ctrl = None     # ctrl values at HOLD entry (interpolation start)
        self.HOLD_DURATION  = 3.0        # seconds to blend from swing pose to seed pose

    def ee_pos(self, data, foot):
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
        print("[ik] stance targets recorded:")
        for f, p in self.stance_targets.items():
            tag = " (swing)" if f == SWING_FOOT else ""
            print(f"  {f}: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]{tag}")

    def solve_orientation_seed(self, phys_data, face_axis, landing_pos=None):
        """
        One-shot global FK search for FL joint angles that best align EE local -Y
        with face_axis, while staying near landing_pos (swing end position).
        Called once at HOLD entry to warm-start mink's _ik_qpos.

        Strategy:
          Pass 1 — differential_evolution for global exploration (no gradient needed).
          Pass 2 — L-BFGS-B local polish from DE's best solution.

        Returns a full qpos array seeded with the best found FL joint angles.
        """
        goal    = np.asarray(face_axis, float)
        goal   /= np.linalg.norm(goal)
        ee_bid  = self.ee_bids[SWING_FOOT]
        scratch = mujoco.MjData(self.model)

        # collect FL actuated joints: (jid, qposadr)
        fl_joints = []
        for i in range(self.model.nu):
            jid   = self.model.actuator_trnid[i, 0]
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname and SWING_FOOT in jname:
                fl_joints.append((jid, self.model.jnt_qposadr[jid]))

        bounds = []
        for jid, _ in fl_joints:
            if self.model.jnt_limited[jid]:
                bounds.append(tuple(self.model.jnt_range[jid]))
            else:
                bounds.append((-np.pi, np.pi))

        # snapshot EE position at HOLD entry as the position anchor
        anchor = landing_pos if landing_pos is not None else phys_data.xpos[ee_bid].copy()

        def objective(x):
            scratch.qpos[:] = phys_data.qpos[:]
            for k, (_, qadr) in enumerate(fl_joints):
                scratch.qpos[qadr] = x[k]
            mujoco.mj_kinematics(self.model, scratch)
            neg_y   = -scratch.xmat[ee_bid].reshape(3, 3)[:, 1]
            ee_pos  = scratch.xpos[ee_bid].copy()

            ori_err = np.degrees(np.arccos(np.clip(np.dot(neg_y, goal), -1., 1.)))
            pos_err = np.linalg.norm(ee_pos - anchor) * 100  # scale to ~degree units

            return ori_err + pos_err

        x0 = np.array([phys_data.qpos[qadr] for _, qadr in fl_joints])
        print(f"[ik] HOLD entry — warm-start FK search (initial ang={objective(x0):.1f}°)...")

        de  = differential_evolution(objective, bounds, maxiter=300, tol=1e-4,
                                     seed=42, popsize=12, workers=1, x0=x0)
        res = minimize(objective, de.x, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})

        print(f"[ik] warm-start done — seed ang={res.fun:.1f}°  "
              f"(DE: {de.fun:.1f}°  polish: {res.fun:.1f}°)")

        seed_qpos = phys_data.qpos.copy()
        for k, (_, qadr) in enumerate(fl_joints):
            seed_qpos[qadr] = res.x[k]
        return seed_qpos

    def solve(self, swing_target, phys_data, dt, n_iter=10, face_axis=None,
              position_cost=None, orientation_cost=None, sim_t=None):
        if face_axis is not None:
            if self._ik_qpos is None:
                # warm-start: global FK search finds exact orientation solution
                self._ik_qpos = self.solve_orientation_seed(
                    phys_data, face_axis, landing_pos=swing_target)
                # snapshot ctrl values right at HOLD entry for interpolation
                self._hold_start_t = sim_t
                self._hold_start_ctrl = np.array([
                    phys_data.qpos[self.model.jnt_qposadr[jid]]
                    for jid in self.ctrl_jids])

            # quintic smooth-step interpolation from swing landing pose → seed pose
            seed_ctrl = np.array([
                self._ik_qpos[self.model.jnt_qposadr[jid]]
                for jid in self.ctrl_jids])

            if self._hold_start_t is not None and sim_t is not None:
                t_rel = sim_t - self._hold_start_t
                s = np.clip(t_rel / self.HOLD_DURATION, 0.0, 1.0)
                s = s**3 * (6*s**2 - 15*s + 10)   # quintic ease
            else:
                s = 1.0  # no time info → snap immediately

            ctrl = self._hold_start_ctrl + s * (seed_ctrl - self._hold_start_ctrl)
            return ctrl
        else:
            ik_qpos = phys_data.qpos.copy()
            self._ik_qpos = None          # reset so warm-start fires on next face_axis entry
            self._hold_start_t    = None  # reset interpolation state
            self._hold_start_ctrl = None
            self.foot_tasks[SWING_FOOT].set_target(_se3_pos(swing_target))
            self.foot_tasks[SWING_FOOT].position_cost    = 10.0
            self.foot_tasks[SWING_FOOT].orientation_cost = 0.0

        for foot in FEET:
            if foot != SWING_FOOT:
                self.foot_tasks[foot].set_target(_se3_pos(self.stance_targets[foot]))
                self.foot_tasks[foot].position_cost = 50.0

        tasks = ([self.body_task]
                 + [self.foot_tasks[f] for f in FEET]
                 + [self.posture_task]
                 + ([self.freeze_passive] if self.freeze_passive else []))

        for _ in range(n_iter):
            self.config.update(ik_qpos)
            vel     = mink.solve_ik(self.config, tasks, dt,
                                    solver="quadprog", damping=IK_DAMPING,
                                    limits=[self.config_limit])
            ik_qpos = self.config.integrate(vel, dt)

        # convergence check
        # NOTE: ori_res measures full SO3 error (all 3 axes) and will read large
        # even when the Y axis is close, because Gram-Schmidt assigns arbitrary X/Z
        # to the target frame. ang_diff is the true metric: angle between EE local
        # -Y and the face_axis goal. Gate the warning on ang_diff, not ori_res.
        if face_axis is not None:
            self.config.update(ik_qpos)
            T        = self.foot_tasks[SWING_FOOT].compute_error(self.config)
            pos_res  = np.linalg.norm(T[:3]) * 1000
            ori_res  = np.degrees(np.linalg.norm(T[3:]))  # full SO3, kept for reference only
            ee_neg_y = -self.config.data.xmat[self.ee_bids[SWING_FOOT]].reshape(3, 3)[:, 1]
            fa_norm  = np.asarray(face_axis) / np.linalg.norm(face_axis)
            ang_diff = np.degrees(np.arccos(np.clip(np.dot(ee_neg_y, fa_norm), -1., 1.)))
            self._warn_counter += 1
            if ang_diff > 5.0 and self._warn_counter % 50 == 0:
                joint_strs = []
                for i, jid in enumerate(self.ctrl_jids):
                    jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    if jname and SWING_FOOT in jname:
                        val = np.degrees(ik_qpos[self.model.jnt_qposadr[jid]])
                        joint_strs.append(f"{jname.replace('_'+SWING_FOOT,'')}={val:.1f}°")
                print(f"[ik] ⚠ ori not converged:  "
                      f"ang={ang_diff:.1f}°  pos_res={pos_res:.1f}mm  "
                      f"joints: {' '.join(joint_strs)}")

        ctrl = np.zeros(self.model.nu)
        for i, jid in enumerate(self.ctrl_jids):
            ctrl[i] = ik_qpos[self.model.jnt_qposadr[jid]]
        if face_axis is not None:
            self._ik_qpos = ik_qpos   # persist for next face_axis step
        return ctrl


# ── PID controller ─────────────────────────────────────────────────────
class PIDController:
    def __init__(self, model):
        self.nu        = model.nu
        self.ctrl_jids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.integral  = np.zeros(model.nu)
        self.prev_err  = np.zeros(model.nu)

    def compute(self, model, data, targets, dt):
        torques = np.zeros(self.nu)
        for i, jid in enumerate(self.ctrl_jids):
            err = targets[i] - data.qpos[model.jnt_qposadr[jid]]
            self.integral[i] = np.clip(
                self.integral[i] + err * dt, -PID_I_CLAMP, PID_I_CLAMP)
            derr = (err - self.prev_err[i]) / dt if dt > 0 else 0.
            self.prev_err[i] = err
            torques[i] = PID_KP * err + PID_KI * self.integral[i] + PID_KD * derr
        return torques


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", choices=list(SEQUENCES.keys()),
                        default=DEFAULT_SEQUENCE)
    parser.add_argument("--headless",  action="store_true")
    parser.add_argument("--duration",  type=float, default=20.0)
    parser.add_argument("--magnets",   action="store_true")
    parser.add_argument("--no-ik",     action="store_true")
    args = parser.parse_args()

    model, data, plate_ids, magnet_ids, sphere_gids = setup_model()

    ik  = IKSolver(model)
    pid = PIDController(model)

    hip_jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"hip_pitch_{SWING_FOOT}")

    # ── wall distance raycast + FK reachability (used by "f2w" sequence) ─
    _wall_gid      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    _ray_geomgroup = np.zeros(6, np.uint8)          # all geom groups enabled
    _geomid_out    = np.array([-1], dtype=np.int32) # mj_ray writes hit geom id here

    # Pre-compute FL max reach from hip by FK corner-sweep of all joint limits.
    # Runs once at startup; used to gate the f2w sequence at settle time.
    def _compute_fl_max_reach():
        from itertools import product as iproduct
        scratch   = mujoco.MjData(model)
        ee_bid    = ik.ee_bids[SWING_FOOT]
        fl_joints = [
            (model.actuator_trnid[i, 0],
             model.jnt_qposadr[model.actuator_trnid[i, 0]])
            for i in range(model.nu)
            if SWING_FOOT in (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                  model.actuator_trnid[i, 0]) or "")
        ]
        limit_vals = []
        for jid, _ in fl_joints:
            lo, hi = (model.jnt_range[jid] if model.jnt_limited[jid]
                      else np.array([-np.pi, np.pi]))
            limit_vals.append([float(lo), float(hi)])
        best = 0.0
        for combo in iproduct(*limit_vals):
            scratch.qpos[:] = data.qpos[:]
            for k, (_, qadr) in enumerate(fl_joints):
                scratch.qpos[qadr] = combo[k]
            mujoco.mj_kinematics(model, scratch)
            d = np.linalg.norm(scratch.xpos[ee_bid] - scratch.xanchor[hip_jid])
            if d > best:
                best = d
        return best

    _fl_max_reach = _compute_fl_max_reach()
    print(f"[f2w] FL max reach from hip: {_fl_max_reach * 1000:.1f} mm")

    def _wall_dist_fn(ray_dir_override=None):
        """
        Ray from EE origin toward the wall.
        Returns distance in metres to the wall surface, or np.inf on miss.
        Exits with a clear message if the wall is beyond FK max reach.

        ray_dir_override: optional world-frame unit vec to use as ray direction.
            When None, fires along EE local -Y (requires IK convergence).
            Pass wall_normal directly from f2w_measure_phase to make measurement
            independent of IK convergence state.
        """
        ee_bid  = ik.ee_bids[SWING_FOOT]
        pos     = data.xpos[ee_bid].copy()        # ray origin: EE body centre

        if ray_dir_override is not None:
            ray_dir = np.array(ray_dir_override, float)
        else:
            # Fall back to EE local -Y — only reliable after IK has converged
            ee_rot  = data.xmat[ee_bid].reshape(3, 3)
            ray_dir = -ee_rot[:, 1]
        ray_dir /= np.linalg.norm(ray_dir)

        # correct mj_ray signature: geomid is an output int32 array, returns float
        _geomid_out[0] = -1
        dist = mujoco.mj_ray(
            model, data, pos, ray_dir,
            _ray_geomgroup, 1, ee_bid, _geomid_out)
        hit_gid = int(_geomid_out[0])

        if dist < 0 or hit_gid != _wall_gid:
            # TOCHECK: ray missed — wall may be behind EE or a chassis body is
            # occluding it. Check robot spawn orientation and wall placement.
            print("[f2w] ⚠  Wall ray missed — check robot orientation / wall placement")
            return np.inf

        # ── FK reachability check ───────────────────────────────────────
        # Project where the ray hits the wall, then check if that point is
        # within the FL leg's max reach from the hip pivot.
        wall_contact_pt = pos + dist * ray_dir
        hip_pos         = data.xanchor[hip_jid].copy()
        dist_from_hip   = np.linalg.norm(wall_contact_pt - hip_pos)

        if dist_from_hip > _fl_max_reach:
            print(f"\n[f2w] ✗  Wall contact point is UNREACHABLE.")
            print(f"[f2w]    Hip → wall contact : {dist_from_hip * 1000:.1f} mm")
            print(f"[f2w]    FL max reach       : {_fl_max_reach  * 1000:.1f} mm")
            print(f"[f2w]    EE → wall surface  : {dist * 1000:.1f} mm")
            print("[f2w]    → Move the robot spawn position closer to the wall and restart.")
            sys.exit(1)
        # ───────────────────────────────────────────────────────────────

        return float(dist)
    # ─────────────────────────────────────────────────────────────────────
    swing_mag_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{SWING_FOOT}")
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")

    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]

    runner     = SequenceRunner(SEQUENCES[args.sequence])
    settled    = False
    target_pos = np.zeros(3)
    face_axis  = None
    last_print = 0.0
    ik_counter = [0]
    BAR        = 12

    print(f"sequence={args.sequence}  "
          f"phases={len(SEQUENCES[args.sequence])}  "
          f"magnets={'on' if args.magnets else 'off'}")

    def sim_step():
        nonlocal ctrl_targets, settled, target_pos, face_axis, last_print
        t = data.time

        data.xfrc_applied[:] = 0
        if args.magnets and settled:
            apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                      off_mids={swing_mag_bid})

        # settle
        if t < SETTLE_TIME:
            data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
            mujoco.mj_step(model, data)
            if t - last_print >= 0.5:
                last_print = t
                pct = t / SETTLE_TIME
                bar = "█" * int(pct * BAR) + "░" * (BAR - int(pct * BAR))
                print(f"t={t:5.1f}  [{bar}] SETTLE {pct*100:5.1f}%")
            return

        # snapshot + start runner once
        if not settled:
            settled = True
            if not args.no_ik:
                ik.record_stance(data)
                ee_home    = ik.ee_pos(data, SWING_FOOT).copy()
                target_pos = ee_home.copy()
                # Print EE local +X at settle so we know starting face direction
                ee_x0 = data.xmat[ik.ee_bids[SWING_FOOT]].reshape(3, 3)[:, 0]
                print(f"[diag] EE local +X at settle (world frame): {ee_x0.round(3)}")
                runner.start(t, {
                    'ee_home':      ee_home,
                    'ee_pos_fn':    lambda: ik.ee_pos(data, SWING_FOOT),
                    'hip_pivot_fn': lambda: data.xanchor[hip_jid].copy(),
                    'wall_dist_fn': _wall_dist_fn,
                })

        # trajectory + IK
        if not args.no_ik:
            ik_counter[0] += 1
            if ik_counter[0] >= IK_EVERY_N:
                ik_counter[0] = 0
                target_pos, face_axis, pos_cost, ori_cost = runner.step(t, {
                    'ee_pos_fn':    lambda: ik.ee_pos(data, SWING_FOOT),
                    'hip_pivot_fn': lambda: data.xanchor[hip_jid].copy(),
                })
                ctrl_targets = ik.solve(
                    target_pos, data, IK_EVERY_N * TIMESTEP,
                    face_axis=face_axis,
                    position_cost=pos_cost,
                    orientation_cost=ori_cost,
                    sim_t=t)

        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

        # telemetry
        if t - last_print >= 1.0:
            last_print = t

            if args.no_ik:
                print(f"t={t:5.1f}  body_z={data.xpos[body_id, 2]:.4f}")
                return

            phase_name, pct = runner.progress(t)
            bar = "█" * int(pct * BAR) + "░" * (BAR - int(pct * BAR))

            actual  = ik.ee_pos(data, SWING_FOOT)
            if target_pos is not None:
                diff    = (actual - target_pos) * 1000
                pos_err = np.linalg.norm(diff)
                pos_str = f"  pos_err={pos_err:5.1f}mm  (Δx={diff[0]:+5.1f} Δy={diff[1]:+5.1f} Δz={diff[2]:+5.1f})"
            else:
                pos_str = "  pos=unconstrained"

            ori_str = ""
            if face_axis is not None:
                ee_neg_y    = -data.xmat[ik.ee_bids[SWING_FOOT]].reshape(3, 3)[:, 1]
                fa_norm     = face_axis / np.linalg.norm(face_axis)
                ang_to_goal = np.degrees(np.arccos(
                    np.clip(np.dot(ee_neg_y, fa_norm), -1., 1.)))
                ori_str = f"  ang_to_goal={ang_to_goal:5.1f}°"

            worst_drift, worst_foot = 0., ""
            for foot in FEET:
                if foot == SWING_FOOT:
                    continue
                d = np.linalg.norm(
                    ik.ee_pos(data, foot) - ik.stance_targets[foot]) * 1000
                if d > worst_drift:
                    worst_drift, worst_foot = d, foot

            print(f"t={t:5.1f}  [{bar}] {phase_name:<6} {pct*100:5.1f}%"
                  f"{pos_str}"
                  f"{ori_str}"
                  f"  stance={worst_foot}/{worst_drift:.1f}mm")

    # ── visualization ──────────────────────────────────────────────────
    JOINT_COLORS = {
        'hip_pitch': [1.0, 1.0, 0.0, 0.9],
        'knee':      [0.0, 1.0, 1.0, 0.9],
        'wrist':     [1.0, 0.0, 1.0, 0.9],
        'ee2':       [0.2, 1.0, 0.5, 0.9],
        'em_z':      [0.5, 0.5, 1.0, 0.9],
        'ee':        [1.0, 0.5, 0.0, 0.9],
    }
    joint_vis = []
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not jname or jname == "root":
            continue
        color = [0.5, 0.5, 0.5, 0.8]
        for prefix, c in JOINT_COLORS.items():
            if jname.startswith(prefix):
                color = c
                break
        joint_vis.append((i, model.jnt_bodyid[i], model.jnt_axis[i].copy(), color))

    def draw_markers(viewer):
        scn = viewer._user_scn
        scn.ngeom = 0
        AXIS_LEN, AXIS_RAD = 0.09, 0.009

        for jid, bid, local_axis, color in joint_vis:
            if scn.ngeom >= scn.maxgeom:
                break
            body_rot = data.xmat[bid].reshape(3, 3)
            waxis = body_rot @ local_axis
            waxis /= np.linalg.norm(waxis)
            z = waxis
            x = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
            x = x - np.dot(x, z) * z;  x /= np.linalg.norm(x)
            y = np.cross(z, x)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                [AXIS_RAD, AXIS_LEN / 2, 0],
                                data.xanchor[jid],
                                np.column_stack([x, y, z]).flatten(), color)
            scn.ngeom += 1

        MAG_LEN, MAG_RAD = 0.05, 0.005
        for foot in FEET:
            bid = ik.ee_bids[foot]
            pos = data.xpos[bid].copy()
            pos[2] += MAG_LEN / 2 + 0.01
            color = [1.0, 0.1, 0.1, 0.9] if (settled and foot == SWING_FOOT) \
                    else [0.1, 1.0, 0.1, 0.9]
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                    [MAG_RAD, MAG_LEN / 2, 0],
                                    pos, np.eye(3).flatten(), color)
                scn.ngeom += 1

        if args.no_ik or not settled:
            return

        if target_pos is not None:
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0],
                                    target_pos, np.eye(3).flatten(), [0.2, 1.0, 0.2, 0.5])
                scn.ngeom += 1

            ALEN, ARAD = 0.04, 0.002
            for offset, rot, rgba in [
                (np.array([ALEN/2, 0, 0]),
                 np.array([[0,0,1],[0,1,0],[-1,0,0]], float), [1,.2,.2,.8]),
                (np.array([0, ALEN/2, 0]),
                 np.array([[1,0,0],[0,0,1],[0,-1,0]], float), [.2,1,.2,.8]),
                (np.array([0, 0, ALEN/2]), np.eye(3),         [.2,.2,1,.8]),
            ]:
                if scn.ngeom < scn.maxgeom:
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                        [ARAD, ALEN/2, 0],
                                        target_pos + offset, rot.flatten(), rgba)
                    scn.ngeom += 1

        for foot in FEET:
            if foot == SWING_FOOT or foot not in ik.stance_targets:
                continue
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.005, 0, 0],
                                    ik.stance_targets[foot],
                                    np.eye(3).flatten(), [1.0, 1.0, 0.2, 0.5])
                scn.ngeom += 1

        # EE local frame triad (X=red, Y=green, Z=blue); goal: local -Y → global +X
        ee_bid  = ik.ee_bids[SWING_FOOT]
        ee_pos  = data.xpos[ee_bid].copy()
        ee_rot  = data.xmat[ee_bid].reshape(3, 3)
        ELEN, ERAD = 0.04, 0.002
        for col, rgba in [(0, [1.,.2,.2,.9]),   # local X → red
                          (1, [.2,1.,.2,.9]),   # local Y → green
                          (2, [.2,.2,1.,.9])]:  # local Z → blue
            axis = ee_rot[:, col]
            # capsule frame: Z-axis of capsule must align with 'axis'
            cz = axis
            cx = np.array([1,0,0]) if abs(cz[0]) < 0.9 else np.array([0,1,0])
            cx = cx - np.dot(cx, cz) * cz; cx /= np.linalg.norm(cx)
            cy = np.cross(cz, cx)
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                    [ERAD, ELEN/2, 0],
                                    ee_pos + axis * ELEN/2,
                                    np.column_stack([cx, cy, cz]).flatten(), rgba)
                scn.ngeom += 1

    # ── run loop ───────────────────────────────────────────────────────
    if args.headless:
        while data.time < args.duration:
            sim_step()
    else:
        paused = [True]

        def key_callback(keycode):
            if keycode == 32:
                paused[0] = not paused[0]
                print(f"{'PAUSED' if paused[0] else 'RUNNING'}  t={data.time:.3f}")

        with mujoco.viewer.launch_passive(
                model, data, key_callback=key_callback) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.3]
            viewer.cam.distance  = 1.8
            viewer.cam.azimuth   = 135
            viewer.cam.elevation = -20
            print("PAUSED — press Space to start")

            frame_dt = 1.0 / 60
            steps_per_frame = max(1, int(frame_dt / model.opt.timestep))

            while viewer.is_running():
                frame_start = time.perf_counter()
                if not paused[0]:
                    for _ in range(steps_per_frame):
                        sim_step()
                draw_markers(viewer)
                viewer.sync()
                target = frame_start + frame_dt
                while time.perf_counter() < target:
                    pass


if __name__ == "__main__":
    main()