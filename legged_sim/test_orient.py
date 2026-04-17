"""
test_orient.py — Hardcoded orientation feasibility test for FL EE -Y → global -X.

Bypasses mink entirely. After settling:
  1. Runs a one-shot scipy L-BFGS-B minimization over FL joint angles using
     pure forward kinematics (mj_kinematics, no physics step) to find the
     joint configuration that best aligns EE local -Y with global -X.
  2. Holds that pose via PID and monitors the live angle error.

Goal: determine whether the orientation is kinematically achievable before
      trusting the IK solver to find it.

Position is fully unconstrained throughout — only orientation is optimized.

Usage:
    python test_orient.py
    python test_orient.py --headless
"""

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import minimize, differential_evolution

LEGGED_DIR = os.path.join(os.path.dirname(__file__), "..", "legged_sim")
SCENE_XML  = os.path.join(LEGGED_DIR, "mwc_mjcf", "scene.xml")
sys.path.insert(0, LEGGED_DIR)

from config import (
    MU_0, MAGNET_VOLUME, MAGNET_BODY_NAMES,
    PARAMS,
    TIMESTEP,
    KNEE_BAKE_DEG,
    WRIST_BAKE_DEG,
    EE_BAKE_DEG,
    bake_joint_angles,
)

# ── constants ───────────────────────────────────────────────────────────
FEET           = ("FL", "FR", "BL", "BR")
SWING_FOOT     = "FL"
SETTLE_TIME    = 2.0
GOAL_FACE_AXIS = np.array([-1.0, 0.0, 0.0])  # global -X; EE local -Y should point here

PID_KP, PID_KI, PID_KD, PID_I_CLAMP = 500.0, 200.0, 30.0, 100.0


# ── magnetic force ───────────────────────────────────────────────────────
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
            n    = best_ft[3:6] - best_ft[0:3]
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            fvec += mag_force(best_d, PARAMS['Br']) * (n / norm)
        tot = np.linalg.norm(fvec)
        if tot > PARAMS['max_force_per_wheel']:
            fvec *= PARAMS['max_force_per_wheel'] / tot
        data.xfrc_applied[mid, :3] += fvec


# ── model setup ─────────────────────────────────────────────────────────
def setup_model():
    bake_joint_angles(os.path.join(LEGGED_DIR, "mwc_mjcf", "robot.xml"))
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP
    mujoco.mj_resetData(model, data)

    for leg in FEET:
        for jname, bake in [
            (f"knee_{leg}",  KNEE_BAKE_DEG),
            (f"wrist_{leg}", WRIST_BAKE_DEG),
            (f"ee_{leg}",    EE_BAKE_DEG),
        ]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                data.qpos[model.jnt_qposadr[jid]] = np.radians(bake[leg])
    mujoco.mj_forward(model, data)

    model.opt.o_solref          = PARAMS["solref"]
    model.opt.o_solimp          = PARAMS["solimp"]
    model.opt.noslip_iterations = PARAMS["noslip_iterations"]
    model.opt.noslip_tolerance  = PARAMS["noslip_tolerance"]
    model.opt.o_margin          = PARAMS["margin"]
    model.dof_damping[:]        = 2.0
    model.dof_armature[:]       = 0.01

    plate_ids = set()
    for name in ("floor", "wall"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid != -1:
            plate_ids.add(gid)
            model.geom_friction[gid] = PARAMS['ground_friction']

    magnet_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                  for n in MAGNET_BODY_NAMES]
    sphere_gids = {
        mid: [gid for gid in range(model.ngeom)
              if model.geom_bodyid[gid] == mid
              and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE]
        for mid in magnet_ids
    }
    return model, data, plate_ids, magnet_ids, sphere_gids


# ── FL joint helpers ─────────────────────────────────────────────────────
def find_fl_actuated_joints(model):
    """
    Returns list of (joint_name, jid, qposadr, actuator_idx) for every
    actuator whose joint name contains SWING_FOOT ("FL").
    """
    fl = []
    for i in range(model.nu):
        jid   = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname and SWING_FOOT in jname:
            fl.append((jname, jid, model.jnt_qposadr[jid], i))
    return fl


# ── one-shot orientation solver ──────────────────────────────────────────
def solve_orientation(model, data_settled, fl_joints, ee_bid):
    """
    Minimizes the angle between EE local -Y and GOAL_FACE_AXIS over FL joint
    angles using pure forward kinematics on a scratch MjData copy.

    No position constraint — only the orientation objective is minimised.

    Strategy:
      Pass 1 — differential_evolution for global exploration (coarse).
      Pass 2 — L-BFGS-B local polish from the best found solution.

    Returns (best_qpos, best_ang_deg).
    """
    scratch = mujoco.MjData(model)

    def fk_neg_y(x):
        """FK-only evaluation. Returns EE local -Y in world frame."""
        scratch.qpos[:] = data_settled.qpos[:]
        for k, (_, _, qadr, _) in enumerate(fl_joints):
            scratch.qpos[qadr] = x[k]
        mujoco.mj_kinematics(model, scratch)
        return -scratch.xmat[ee_bid].reshape(3, 3)[:, 1]

    def objective(x):
        neg_y = fk_neg_y(x)
        dot   = np.clip(np.dot(neg_y, GOAL_FACE_AXIS), -1.0, 1.0)
        return np.degrees(np.arccos(dot))  # degrees; 0° = perfect alignment

    # joint bounds (radians)
    bounds = []
    for _, jid, _, _ in fl_joints:
        if model.jnt_limited[jid]:
            bounds.append((model.jnt_range[jid, 0], model.jnt_range[jid, 1]))
        else:
            bounds.append((-np.pi, np.pi))

    x0 = np.array([data_settled.qpos[qadr] for _, _, qadr, _ in fl_joints])

    # ── pass 1: global search ──────────────────────────────────────────
    print("[orient] Pass 1 — differential_evolution (global)...")
    de_result = differential_evolution(
        objective, bounds,
        maxiter=400, tol=1e-4, seed=42,
        popsize=12, mutation=(0.5, 1.5), recombination=0.7,
        workers=1,
    )
    print(f"[orient] DE best: {de_result.fun:.2f}°")

    # ── pass 2: local polish ───────────────────────────────────────────
    print("[orient] Pass 2 — L-BFGS-B local polish...")
    result = minimize(
        objective, de_result.x, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    best_x   = result.x
    best_ang = objective(best_x)

    # ── diagnostics ───────────────────────────────────────────────────
    neg_y = fk_neg_y(best_x)
    print(f"\n[orient] ── Optimization result ──────────────────────────")
    print(f"[orient]   ang_to_goal  : {best_ang:.3f}°  "
          f"({'✓ ACHIEVABLE (<5°)' if best_ang < 5 else '✗ BEST EFFORT'})")
    print(f"[orient]   EE local -Y  : [{neg_y[0]:+.4f},  {neg_y[1]:+.4f},  {neg_y[2]:+.4f}]")
    print(f"[orient]   goal  (+X)   : [{GOAL_FACE_AXIS[0]:+.4f},  {GOAL_FACE_AXIS[1]:+.4f},  "
          f"{GOAL_FACE_AXIS[2]:+.4f}]")
    print(f"[orient]   scipy success: {result.success}  ({result.message})")
    print(f"[orient] Target joint angles:")
    for k, (jname, _, _, _) in enumerate(fl_joints):
        lo_deg = np.degrees(bounds[k][0])
        hi_deg = np.degrees(bounds[k][1])
        val_deg = np.degrees(best_x[k])
        at_limit = " ← AT LIMIT" if (abs(val_deg - lo_deg) < 0.5 or
                                      abs(val_deg - hi_deg) < 0.5) else ""
        print(f"    {jname:32s} = {val_deg:+7.2f}°  "
              f"(range [{lo_deg:+.0f}°, {hi_deg:+.0f}°]){at_limit}")
    print()

    best_qpos = data_settled.qpos.copy()
    for k, (_, _, qadr, _) in enumerate(fl_joints):
        best_qpos[qadr] = best_x[k]
    return best_qpos, best_ang


# ── PID ──────────────────────────────────────────────────────────────────
class PIDController:
    def __init__(self, model):
        self.nu        = model.nu
        self.ctrl_jids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.integral  = np.zeros(model.nu)
        self.prev_err  = np.zeros(model.nu)

    def compute(self, model, data, targets, dt):
        torques = np.zeros(self.nu)
        for i, jid in enumerate(self.ctrl_jids):
            err              = targets[i] - data.qpos[model.jnt_qposadr[jid]]
            self.integral[i] = np.clip(
                self.integral[i] + err * dt, -PID_I_CLAMP, PID_I_CLAMP)
            derr             = (err - self.prev_err[i]) / dt if dt > 0 else 0.0
            self.prev_err[i] = err
            torques[i]       = PID_KP * err + PID_KI * self.integral[i] + PID_KD * derr
        return torques


# ── main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless",  action="store_true")
    parser.add_argument("--duration",  type=float, default=15.0)
    parser.add_argument("--magnets",   action="store_true")
    args = parser.parse_args()

    model, data, plate_ids, magnet_ids, sphere_gids = setup_model()
    pid = PIDController(model)

    ee_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{SWING_FOOT}")
    swing_mag_bid = ee_bid  # FL electromagnet is both EE tracker and magnet to disable

    # initial ctrl_targets from baked qpos
    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]

    settled    = False
    last_print = 0.0
    BAR        = 12

    def sim_step():
        nonlocal ctrl_targets, settled, last_print

        t = data.time
        data.xfrc_applied[:] = 0
        if args.magnets and settled:
            apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                      off_mids={swing_mag_bid})

        # ── settle phase ───────────────────────────────────────────────
        if t < SETTLE_TIME:
            data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
            mujoco.mj_step(model, data)
            if t - last_print >= 0.5:
                last_print = t
                pct = t / SETTLE_TIME
                bar = "█" * int(pct * BAR) + "░" * (BAR - int(pct * BAR))
                print(f"t={t:5.1f}  [{bar}] SETTLE {pct*100:5.1f}%")
            return

        # ── one-shot solve (runs once at t == SETTLE_TIME) ─────────────
        if not settled:
            settled = True
            fl_joints = find_fl_actuated_joints(model)
            print(f"[orient] FL joints: {[j[0] for j in fl_joints]}")

            ee_neg_y0 = -data.xmat[ee_bid].reshape(3, 3)[:, 1]
            ang0      = np.degrees(np.arccos(
                np.clip(np.dot(ee_neg_y0, GOAL_FACE_AXIS), -1.0, 1.0)))
            print(f"[orient] EE local -Y at settle : {ee_neg_y0.round(3)}")
            print(f"[orient] Initial ang_to_goal   : {ang0:.2f}°\n")

            orient_qpos, best_ang = solve_orientation(model, data, fl_joints, ee_bid)

            # Build ctrl_targets from the solved qpos.
            # Non-FL joints keep their current settled values.
            for i in range(model.nu):
                jid          = model.actuator_trnid[i, 0]
                ctrl_targets[i] = orient_qpos[model.jnt_qposadr[jid]]

            print(f"[orient] Holding solution. Monitoring EE -Y live...\n")

        # ── hold and monitor ───────────────────────────────────────────
        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

        if t - last_print >= 0.5:
            last_print = t
            ee_neg_y = -data.xmat[ee_bid].reshape(3, 3)[:, 1]
            ang      = np.degrees(np.arccos(
                np.clip(np.dot(ee_neg_y, GOAL_FACE_AXIS), -1.0, 1.0)))
            status   = "✓" if ang < 5.0 else ("~" if ang < 15.0 else "✗")
            print(f"t={t:5.1f}  {status} ang_to_goal={ang:6.2f}°  "
                  f"EE_neg_y=[{ee_neg_y[0]:+.3f}, {ee_neg_y[1]:+.3f}, {ee_neg_y[2]:+.3f}]")

    # ── viewer / headless ──────────────────────────────────────────────
    if args.headless:
        while data.time < args.duration:
            sim_step()
        return

    paused = [True]

    def key_callback(keycode):
        if keycode == 32:
            paused[0] = not paused[0]
            print(f"{'PAUSED' if paused[0] else 'RUNNING'}  t={data.time:.3f}")

    def draw_ee_frame(viewer):
        """Draw EE local frame triad: X=red, Y=green, Z=blue.
        Goal: green (local +Y) should point toward global -X,
              i.e. local -Y (anti-green) points toward global +X.
        """
        scn = viewer._user_scn
        scn.ngeom = 0
        ee_pos = data.xpos[ee_bid].copy()
        ee_rot = data.xmat[ee_bid].reshape(3, 3)
        ELEN, ERAD = 0.05, 0.003
        colors = [[1., .2, .2, .9], [.2, 1., .2, .9], [.2, .2, 1., .9]]
        for col, rgba in enumerate(colors):
            axis = ee_rot[:, col]
            cz   = axis
            cx   = (np.array([1, 0, 0]) if abs(cz[0]) < 0.9
                    else np.array([0, 1, 0]))
            cx   = cx - np.dot(cx, cz) * cz
            cx  /= np.linalg.norm(cx)
            cy   = np.cross(cz, cx)
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                    [ERAD, ELEN / 2, 0],
                    ee_pos + axis * ELEN / 2,
                    np.column_stack([cx, cy, cz]).flatten(), rgba)
                scn.ngeom += 1
        # goal direction arrow (cyan, along global +X from EE pos)
        goal = GOAL_FACE_AXIS
        cz   = goal
        cx   = np.array([0, 1, 0]) if abs(cz[1]) < 0.9 else np.array([1, 0, 0])
        cx   = cx - np.dot(cx, cz) * cz; cx /= np.linalg.norm(cx)
        cy   = np.cross(cz, cx)
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                [0.002, ELEN / 2, 0],
                ee_pos + goal * ELEN / 2,
                np.column_stack([cx, cy, cz]).flatten(),
                [0., 1., 1., 0.6])
            scn.ngeom += 1

    with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [-2.0, 0.0, 0.3]
        viewer.cam.distance  = 1.8
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -20
        print("PAUSED — press Space to start")
        print(f"Goal: FL EE local -Y (anti-green arrow) → global -X (cyan arrow)\n")

        frame_dt        = 1.0 / 60
        steps_per_frame = max(1, int(frame_dt / model.opt.timestep))

        while viewer.is_running():
            frame_start = time.perf_counter()
            if not paused[0]:
                for _ in range(steps_per_frame):
                    sim_step()
            draw_ee_frame(viewer)
            viewer.sync()
            target = frame_start + frame_dt
            while time.perf_counter() < target:
                pass


if __name__ == "__main__":
    main()