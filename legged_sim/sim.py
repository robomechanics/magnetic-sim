"""
MWC sim: robot on ground, IK-controlled FL foot swing sequence.
Trajectory logic lives in sequences.py; this file owns sim, IK, PID, and viz.

Usage:
    python sim.py                    # GUI, orient sequence (default)
    python sim.py --sequence f2w     # GUI, floor-to-wall sequence
    python sim.py --headless --sequence orient
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import mink

# ── paths ───────────────────────────────────────────────────────────────────
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


# ── constants ────────────────────────────────────────────────────────────────

FEET        = ('FL', 'FR', 'BL', 'BR')
SWING_FOOT  = "FL"
SETTLE_TIME = 2.0
IK_DAMPING  = 1e-3
IK_EVERY_N  = 10      # physics steps between IK solves

PID_KP, PID_KI, PID_KD, PID_I_CLAMP = 500.0, 200.0, 30.0, 100.0

_BAR_WIDTH = 12
_JOINT_COLORS = {
    'hip_pitch': [1.0, 1.0, 0.0, 0.9],
    'knee':      [0.0, 1.0, 1.0, 0.9],
    'wrist':     [1.0, 0.0, 1.0, 0.9],
    'ee2':       [0.2, 1.0, 0.5, 0.9],
    'em_z':      [0.5, 0.5, 1.0, 0.9],
    'ee':        [1.0, 0.5, 0.0, 0.9],
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _se3_pos(pos):
    T = np.eye(4); T[:3, 3] = pos
    return mink.SE3.from_matrix(T)

def _mag_force(dist, Br):
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)

def _build_joint_vis(model):
    """Pre-compute (jid, bid, local_axis, rgba) for draw_markers."""
    out = []
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not jname or jname == "root":
            continue
        color = next((c for p, c in _JOINT_COLORS.items() if jname.startswith(p)),
                     [0.5, 0.5, 0.5, 0.8])
        out.append((i, model.jnt_bodyid[i], model.jnt_axis[i].copy(), color))
    return out


# ── physics ──────────────────────────────────────────────────────────────────

def apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, off_mids=frozenset()):
    """Apply dipole-dipole magnetic forces for all active magnets."""
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
            fvec += _mag_force(best_d, PARAMS['Br']) * (n / norm)
        tot = np.linalg.norm(fvec)
        if tot > PARAMS['max_force_per_wheel']:
            fvec *= PARAMS['max_force_per_wheel'] / tot
        data.xfrc_applied[mid, :3] += fvec


def setup_model():
    """Load, configure, and return (model, data, plate_ids, magnet_ids, sphere_gids)."""
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

    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root_jid != -1:
        data.qpos[model.jnt_qposadr[root_jid]] += 2.05
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

    magnet_ids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                   for n in MAGNET_BODY_NAMES]
    sphere_gids = {
        mid: [gid for gid in range(model.ngeom)
              if model.geom_bodyid[gid] == mid
              and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE]
        for mid in magnet_ids
    }
    return model, data, plate_ids, magnet_ids, sphere_gids


# ── IK ───────────────────────────────────────────────────────────────────────

class IKSolver:
    """Whole-body IK: stance feet locked, swing foot tracks trajectory target."""

    def __init__(self, model):
        self.model  = model
        self.config = mink.Configuration(model)

        self.ee_bids    = {}
        self.foot_tasks = {}
        for foot in FEET:
            frame = f"electromagnet_{foot}"
            self.ee_bids[foot]    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame)
            self.foot_tasks[foot] = mink.FrameTask(
                frame_name=frame, frame_type="body",
                position_cost=10.0, orientation_cost=0.0, lm_damping=IK_DAMPING)

        self.body_task    = mink.FrameTask(frame_name="main_frame", frame_type="body",
                                           position_cost=50.0, orientation_cost=50.0,
                                           lm_damping=IK_DAMPING)
        self.posture_task = mink.PostureTask(model=model, cost=0.01, lm_damping=IK_DAMPING)
        self.config_limit = mink.ConfigurationLimit(model)
        self.ctrl_jids    = [model.actuator_trnid[i, 0] for i in range(model.nu)]

        self.stance_targets  = {}
        self._ori_target_rot = None  # cached rotation for active orientation phase

    def ee_pos(self, data, foot) -> np.ndarray:
        return data.xpos[self.ee_bids[foot]].copy()

    def record_stance(self, data):
        """Snapshot EE positions and body pose as stance references."""
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

    def solve(self, swing_target, phys_data, dt, n_iter=10,
              face_axis=None, position_cost=None, orientation_cost=None) -> np.ndarray:
        """Run IK and return joint position targets (ctrl array)."""
        ik_qpos  = phys_data.qpos.copy()
        pos_cost = position_cost    if position_cost    is not None else 10.0
        ori_cost = orientation_cost if orientation_cost is not None else 0.0

        if face_axis is not None:
            if self._ori_target_rot is None:
                self._ori_target_rot = self._compute_ori_target(
                    phys_data.xmat[self.ee_bids[SWING_FOOT]].reshape(3, 3).copy(), face_axis)
                print(f"[ik] orientation locked — EE -Y -> {(-self._ori_target_rot[:, 1]).round(3)}")
            T = np.eye(4)
            T[:3, :3] = self._ori_target_rot
            T[:3,  3] = swing_target
            target_se3 = mink.SE3.from_matrix(T)
        else:
            self._ori_target_rot = None
            target_se3 = _se3_pos(swing_target)

        # Rebuild swing-foot task when orientation cost changes (mink bakes at construction).
        if self.foot_tasks[SWING_FOOT].orientation_cost != ori_cost:
            self.foot_tasks[SWING_FOOT] = mink.FrameTask(
                frame_name=f"electromagnet_{SWING_FOOT}", frame_type="body",
                position_cost=pos_cost, orientation_cost=ori_cost, lm_damping=IK_DAMPING)
        self.foot_tasks[SWING_FOOT].set_target(target_se3)
        self.foot_tasks[SWING_FOOT].position_cost = pos_cost

        # Body orientation competes with EE; zero it during orientation phases.
        self.body_task.orientation_cost = 0.0 if face_axis is not None else 50.0

        for foot in FEET:
            if foot != SWING_FOOT:
                self.foot_tasks[foot].set_target(_se3_pos(self.stance_targets[foot]))
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

    @staticmethod
    def _compute_ori_target(current_rot: np.ndarray, face_axis) -> np.ndarray:
        """Minimal rotation aligning EE local -Y to face_axis (Rodrigues)."""
        goal = np.asarray(face_axis, float); goal /= np.linalg.norm(goal)
        cur_neg_y = -current_rot[:, 1]
        axis  = np.cross(cur_neg_y, goal)
        sin_a = np.linalg.norm(axis)
        cos_a = float(np.dot(cur_neg_y, goal))
        if sin_a < 1e-6:
            return current_rot.copy() if cos_a >= 0 else (
                (-np.eye(3) + 2 * np.outer(cur_neg_y, cur_neg_y)) @ current_rot)
        axis /= sin_a
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return (np.eye(3) + sin_a * K + (1. - cos_a) * (K @ K)) @ current_rot


# ── control ──────────────────────────────────────────────────────────────────

class PIDController:
    """Joint-space PID: position targets → torque commands."""

    def __init__(self, model):
        self.nu        = model.nu
        self.ctrl_jids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.integral  = np.zeros(model.nu)
        self.prev_err  = np.zeros(model.nu)

    def compute(self, model, data, targets, dt) -> np.ndarray:
        torques = np.zeros(self.nu)
        for i, jid in enumerate(self.ctrl_jids):
            err = targets[i] - data.qpos[model.jnt_qposadr[jid]]
            self.integral[i] = np.clip(self.integral[i] + err * dt, -PID_I_CLAMP, PID_I_CLAMP)
            derr = (err - self.prev_err[i]) / dt if dt > 0 else 0.
            self.prev_err[i] = err
            torques[i] = PID_KP * err + PID_KI * self.integral[i] + PID_KD * derr
        return torques


# ── visualization ─────────────────────────────────────────────────────────────

def _add_capsule(scn, size, pos, rot_flat, rgba):
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, size, pos, rot_flat, rgba)
        scn.ngeom += 1

def _add_sphere(scn, radius, pos, rgba):
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            [radius, 0, 0], pos, np.eye(3).flatten(), rgba)
        scn.ngeom += 1

def draw_markers(viewer, model, data, ik, joint_vis,
                 args, settled, target_pos, face_axis, swing_off, swing_mag_bid):
    """Render joint axes, magnet indicators, IK targets, and EE frame."""
    scn = viewer._user_scn
    scn.ngeom = 0

    # joint rotation axes
    for jid, bid, local_axis, color in joint_vis:
        z = data.xmat[bid].reshape(3, 3) @ local_axis
        z /= np.linalg.norm(z)
        x = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
        x = x - np.dot(x, z) * z;  x /= np.linalg.norm(x)
        _add_capsule(scn, [0.009, 0.045, 0], data.xanchor[jid],
                     np.column_stack([x, np.cross(z, x), z]).flatten(), color)

    # magnet status (red = swing/off, green = stance/on)
    for foot in FEET:
        pos = data.xpos[ik.ee_bids[foot]].copy(); pos[2] += 0.035
        color = [1.0, 0.1, 0.1, 0.9] if (settled and foot == SWING_FOOT) else [0.1, 1.0, 0.1, 0.9]
        _add_capsule(scn, [0.005, 0.025, 0], pos, np.eye(3).flatten(), color)

    if args.no_ik or not settled:
        return

    # IK target sphere + XYZ triad
    if target_pos is not None:
        _add_sphere(scn, 0.008, target_pos, [0.2, 1.0, 0.2, 0.5])
        for offset, rot, rgba in [
            (np.array([0.02, 0, 0]),  np.array([[0,0,1],[0,1,0],[-1,0,0]], float), [1,.2,.2,.8]),
            (np.array([0, 0.02, 0]),  np.array([[1,0,0],[0,0,1],[0,-1,0]], float), [.2,1,.2,.8]),
            (np.array([0, 0, 0.02]),  np.eye(3),                                   [.2,.2,1,.8]),
        ]:
            _add_capsule(scn, [0.002, 0.02, 0],
                         target_pos + offset, rot.flatten(), rgba)

    # stance target spheres
    for foot in FEET:
        if foot != SWING_FOOT and foot in ik.stance_targets:
            _add_sphere(scn, 0.005, ik.stance_targets[foot], [1.0, 1.0, 0.2, 0.5])

    # EE local frame triad (X=red, Y=green, Z=blue)
    ee_pos = data.xpos[ik.ee_bids[SWING_FOOT]].copy()
    ee_rot = data.xmat[ik.ee_bids[SWING_FOOT]].reshape(3, 3)
    for col, rgba in [(0, [1.,.2,.2,.9]), (1, [.2,1.,.2,.9]), (2, [.2,.2,1.,.9])]:
        cz = ee_rot[:, col]
        cx = np.array([1,0,0]) if abs(cz[0]) < 0.9 else np.array([0,1,0])
        cx = cx - np.dot(cx, cz) * cz;  cx /= np.linalg.norm(cx)
        _add_capsule(scn, [0.002, 0.02, 0], ee_pos + cz * 0.02,
                     np.column_stack([cx, np.cross(cz, cx), cz]).flatten(), rgba)


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", choices=list(SEQUENCES.keys()), default=DEFAULT_SEQUENCE)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--magnets",  action="store_true")
    parser.add_argument("--no-ik",    action="store_true")
    args = parser.parse_args()

    # ── setup ─────────────────────────────────────────────────────────────
    model, data, plate_ids, magnet_ids, sphere_gids = setup_model()
    ik  = IKSolver(model)
    pid = PIDController(model)

    hip_jid       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"hip_pitch_{SWING_FOOT}")
    swing_mag_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  f"electromagnet_{SWING_FOOT}")
    body_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "main_frame")
    joint_vis     = _build_joint_vis(model)

    # ── f2w helpers ────────────────────────────────────────────────────────
    _wall_gid      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    _ray_geomgroup = np.zeros(6, np.uint8)
    _geomid_out    = np.array([-1], dtype=np.int32)

    def _compute_fl_max_reach() -> float:
        """FK corner-sweep to find the maximum reach of the FL foot from its hip."""
        from itertools import product as iproduct
        scratch   = mujoco.MjData(model)
        ee_bid    = ik.ee_bids[SWING_FOOT]
        fl_joints = [(model.actuator_trnid[i, 0], model.jnt_qposadr[model.actuator_trnid[i, 0]])
                     for i in range(model.nu)
                     if SWING_FOOT in (mujoco.mj_id2name(
                         model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0]) or "")]
        limit_vals = [[float(lo), float(hi)] if model.jnt_limited[jid] else [-np.pi, np.pi]
                      for jid, _ in fl_joints for lo, hi in [model.jnt_range[jid]]]
        best = 0.0
        for combo in iproduct(*limit_vals):
            scratch.qpos[:] = data.qpos[:]
            for k, (_, qadr) in enumerate(fl_joints):
                scratch.qpos[qadr] = combo[k]
            mujoco.mj_kinematics(model, scratch)
            d = np.linalg.norm(scratch.xpos[ee_bid] - scratch.xanchor[hip_jid])
            if d > best: best = d
        return best

    _fl_max_reach = _compute_fl_max_reach()
    print(f"[f2w] FL max reach from hip: {_fl_max_reach * 1000:.1f} mm")

    def _wall_dist_fn(ray_dir_override=None) -> float:
        """Ray from EE toward wall; returns distance (m) or np.inf on miss."""
        ee_bid  = ik.ee_bids[SWING_FOOT]
        pos     = data.xpos[ee_bid].copy()
        ray_dir = (np.array(ray_dir_override, float) if ray_dir_override is not None
                   else -data.xmat[ee_bid].reshape(3, 3)[:, 1])
        ray_dir /= np.linalg.norm(ray_dir)

        _geomid_out[0] = -1
        dist    = mujoco.mj_ray(model, data, pos, ray_dir, _ray_geomgroup, 1, ee_bid, _geomid_out)
        hit_gid = int(_geomid_out[0])

        if dist < 0 or hit_gid != _wall_gid:
            print("[f2w] ⚠  Wall ray missed — check robot orientation / wall placement")
            return np.inf

        if np.linalg.norm(pos + dist * ray_dir - data.xanchor[hip_jid]) > _fl_max_reach:
            print(f"[f2w] ✗  Wall UNREACHABLE — move robot closer and restart.")
            sys.exit(1)

        return float(dist)

    # ── sim state ──────────────────────────────────────────────────────────
    ctrl_targets = np.array([data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
                              for i in range(model.nu)])
    runner     = SequenceRunner(SEQUENCES[args.sequence])
    swing_off  = {swing_mag_bid}
    settled    = False
    target_pos = np.zeros(3)
    face_axis  = None
    last_print = 0.0
    ik_counter = [0]

    print(f"sequence={args.sequence}  phases={len(SEQUENCES[args.sequence])}  "
          f"magnets={'on' if args.magnets else 'off'}")

    # ── sim step ───────────────────────────────────────────────────────────
    def sim_step():
        nonlocal ctrl_targets, settled, target_pos, face_axis, last_print
        t = data.time

        data.xfrc_applied[:] = 0
        if args.magnets and settled:
            apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, off_mids=swing_off)

        if t < SETTLE_TIME:
            data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
            mujoco.mj_step(model, data)
            if t - last_print >= 0.5:
                last_print = t
                pct = t / SETTLE_TIME
                bar = "█" * int(pct * _BAR_WIDTH) + "░" * (_BAR_WIDTH - int(pct * _BAR_WIDTH))
                print(f"t={t:5.1f}  [{bar}] SETTLE {pct*100:5.1f}%")
            return

        if not settled:
            settled = True
            if not args.no_ik:
                ik.record_stance(data)
                ee_home    = ik.ee_pos(data, SWING_FOOT).copy()
                target_pos = ee_home.copy()
                print(f"[diag] EE local +X at settle: "
                      f"{data.xmat[ik.ee_bids[SWING_FOOT]].reshape(3,3)[:,0].round(3)}")
                runner.start(t, {
                    'ee_home':           ee_home,
                    'ee_pos_fn':         lambda: ik.ee_pos(data, SWING_FOOT),
                    'hip_pivot_fn':      lambda: data.xanchor[hip_jid].copy(),
                    'wall_dist_fn':      _wall_dist_fn,
                    'magnet_disable_fn': lambda: swing_off.add(swing_mag_bid),
                    'magnet_enable_fn':  lambda: swing_off.discard(swing_mag_bid),
                })

        if not args.no_ik:
            ik_counter[0] += 1
            if ik_counter[0] >= IK_EVERY_N:
                ik_counter[0] = 0
                target_pos, face_axis, pos_cost, ori_cost = runner.step(t, {
                    'ee_pos_fn':    lambda: ik.ee_pos(data, SWING_FOOT),
                    'hip_pivot_fn': lambda: data.xanchor[hip_jid].copy(),
                })
                ctrl_targets = ik.solve(target_pos, data, IK_EVERY_N * TIMESTEP,
                                        face_axis=face_axis,
                                        position_cost=pos_cost, orientation_cost=ori_cost)

        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

        if t - last_print < 1.0:
            return
        last_print = t

        if args.no_ik:
            print(f"t={t:5.1f}  body_z={data.xpos[body_id, 2]:.4f}")
            return

        phase_name, pct = runner.progress(t)
        bar = "█" * int(pct * _BAR_WIDTH) + "░" * (_BAR_WIDTH - int(pct * _BAR_WIDTH))

        actual  = ik.ee_pos(data, SWING_FOOT)
        diff    = (actual - target_pos) * 1000
        pos_str = (f"  pos_err={np.linalg.norm(diff):5.1f}mm"
                   f"  (Δx={diff[0]:+5.1f} Δy={diff[1]:+5.1f} Δz={diff[2]:+5.1f})"
                   if target_pos is not None else "  pos=unconstrained")

        ori_str = ""
        if face_axis is not None:
            ee_neg_y = -data.xmat[ik.ee_bids[SWING_FOOT]].reshape(3, 3)[:, 1]
            ang      = np.degrees(np.arccos(np.clip(
                np.dot(ee_neg_y, face_axis / np.linalg.norm(face_axis)), -1., 1.)))
            ori_str  = f"  ang_to_goal={ang:5.1f}°"

        worst_d, worst_f = 0., ""
        for foot in FEET:
            if foot == SWING_FOOT: continue
            d = np.linalg.norm(ik.ee_pos(data, foot) - ik.stance_targets[foot]) * 1000
            if d > worst_d: worst_d, worst_f = d, foot

        print(f"t={t:5.1f}  [{bar}] {phase_name:<6} {pct*100:5.1f}%"
              f"{pos_str}{ori_str}"
              f"  stance={worst_f}/{worst_d:.1f}mm"
              f"  mag={'OFF' if swing_mag_bid in swing_off else 'ON '}")

    # ── run loop ───────────────────────────────────────────────────────────
    if args.headless:
        while data.time < args.duration:
            sim_step()
        return

    paused = [True]

    def key_callback(keycode):
        if keycode == 32:
            paused[0] = not paused[0]
            print(f"{'PAUSED' if paused[0] else 'RUNNING'}  t={data.time:.3f}")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.3]
        viewer.cam.distance  = 1.8
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -20
        print("PAUSED — press Space to start")

        frame_dt        = 1.0 / 60.0
        steps_per_frame = max(1, int(frame_dt / model.opt.timestep))

        while viewer.is_running():
            frame_start = time.perf_counter()
            if not paused[0]:
                for _ in range(steps_per_frame):
                    sim_step()
            draw_markers(viewer, model, data, ik, joint_vis,
                         args, settled, target_pos, face_axis, swing_off, swing_mag_bid)
            viewer.sync()
            elapsed = time.perf_counter() - frame_start
            if (remaining := frame_dt - elapsed) > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()