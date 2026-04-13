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

def _se3_pos_rot(pos, face_axis):
    # Aligns EE local -Y to face_axis (world-frame unit vector).
    neg_y = np.asarray(face_axis, float); neg_y /= np.linalg.norm(neg_y)
    y = -neg_y
    up = np.array([0., 0., 1.]) if abs(y[2]) < 0.9 else np.array([0., 1., 0.])
    x = np.cross(up, y); x /= np.linalg.norm(x)
    z = np.cross(x, y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, pos
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

    def solve(self, swing_target, phys_data, dt, n_iter=10, face_axis=None,
              position_cost=None, orientation_cost=None):
        if face_axis is not None:
            if self._ik_qpos is None:
                self._ik_qpos = phys_data.qpos.copy()
            ik_qpos = self._ik_qpos
            # use dummy zero position when unconstrained (position_cost=0 makes it irrelevant)
            pos = swing_target if swing_target is not None else np.zeros(3)
            self.foot_tasks[SWING_FOOT].set_target(_se3_pos_rot(pos, face_axis))
            self.foot_tasks[SWING_FOOT].position_cost    = position_cost    if position_cost    is not None else 10.0
            self.foot_tasks[SWING_FOOT].orientation_cost = orientation_cost if orientation_cost is not None else 10.0
            n_iter = 50
        else:
            ik_qpos = phys_data.qpos.copy()
            self._ik_qpos = None
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
        if face_axis is not None:
            self.config.update(ik_qpos)
            T        = self.foot_tasks[SWING_FOOT].compute_error(self.config)
            pos_res  = np.linalg.norm(T[:3]) * 1000
            ori_res  = np.degrees(np.linalg.norm(T[3:]))
            if ori_res > 5.0:
                ee_neg_y = -self.config.data.xmat[self.ee_bids[SWING_FOOT]].reshape(3, 3)[:, 1]
                fa_norm  = np.asarray(face_axis) / np.linalg.norm(face_axis)
                ang_diff = np.degrees(np.arccos(np.clip(np.dot(ee_neg_y, fa_norm), -1., 1.)))
                joint_strs = []
                for i, jid in enumerate(self.ctrl_jids):
                    jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    if jname and SWING_FOOT in jname:
                        joint_strs.append(
                            f"{jname}={np.degrees(ik_qpos[self.model.jnt_qposadr[jid]]):.1f}°")
                print(f"[ik] WARNING: orientation not converged — "
                      f"pos_res={pos_res:.1f}mm  ori_res={ori_res:.1f}°  "
                      f"ang_to_goal={ang_diff:.1f}°  "
                      f"ee_neg_y=[{ee_neg_y[0]:+.3f},{ee_neg_y[1]:+.3f},{ee_neg_y[2]:+.3f}]  "
                      f"goal=[{fa_norm[0]:+.3f},{fa_norm[1]:+.3f},{fa_norm[2]:+.3f}]  "
                      f"pos_cost={self.foot_tasks[SWING_FOOT].position_cost}  "
                      f"ori_cost={self.foot_tasks[SWING_FOOT].orientation_cost}  "
                      + "  ".join(joint_strs))

        ctrl = np.zeros(self.model.nu)
        for i, jid in enumerate(self.ctrl_jids):
            ctrl[i] = ik_qpos[self.model.jnt_qposadr[jid]]
        self._ik_qpos = ik_qpos
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
                    orientation_cost=ori_cost)

        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

        # telemetry
        if t - last_print >= 0.5:
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
                ee_neg_y = -data.xmat[ik.ee_bids[SWING_FOOT]].reshape(3, 3)[:, 1]
                fa_norm  = face_axis / np.linalg.norm(face_axis)
                ori_err  = np.degrees(np.arccos(
                    np.clip(np.dot(ee_neg_y, fa_norm), -1., 1.)))
                ori_str = f"  ori_err={ori_err:5.1f}° goal=0.0°"

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
            viewer.cam.lookat[:] = [-2.0, 0.0, 0.3]
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