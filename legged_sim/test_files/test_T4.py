"""
test_T4.py — Contracted pose sweep for FL leg (T4).

Animates all four FL joints from the settled pose to STEP1_POSE_FL over
DURATION seconds using the same quintic smooth-step as joint_phase.
Watch the FL (RED) leg for self-collision with the body, wall, or floor,
and for any interference with the other stance legs.

After the sweep completes the pose is held indefinitely so you can inspect
the final contracted configuration. Press R to replay, Ctrl+C to quit.

Place in legged_sim/test_files/ and run from legged_sim/:
    python test_files/test_T4.py
    python test_files/test_T4.py --duration 5.0   # slower sweep
    python test_files/test_T4.py --hip 45 --knee 80 --wrist -80 --ee 0  # alternate angles
"""

import argparse
import math
import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR   = os.path.join(_HERE, "..")
LEGGED_DIR = os.path.join(_SIM_DIR, "..", "legged_sim")
sys.path.insert(0, _SIM_DIR)
sys.path.insert(0, LEGGED_DIR)

from sim import setup_model, PIDController, PID_KP, PID_KI, PID_KD, PID_I_CLAMP, TIMESTEP, SETTLE_TIME

# ── default STEP1_POSE_FL angles (confirmed via test_T9.py) ───────────────────
DEFAULT_HIP   =  +45.0
DEFAULT_KNEE  =  +90.0
DEFAULT_WRIST =  -90.0
DEFAULT_EE    =    0.0

FL_JOINTS = ['hip_pitch_FL', 'knee_FL', 'wrist_FL', 'ee_FL']


def _quintic(t, T):
    """Quintic smooth-step — matches joint_phase in sequences.py."""
    s = float(np.clip(t / T, 0.0, 1.0))
    return s ** 3 * (6 * s ** 2 - 15 * s + 10)


def _get_jid(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint '{name}' not found.")
    return jid


def _draw_ee_frame(scn, data, ee_bid):
    """Draw EE local X (red), Y (green = magnet face), Z (blue) stubs."""
    ee_pos = data.xpos[ee_bid].copy()
    ee_rot = data.xmat[ee_bid].reshape(3, 3)
    for col, rgba in enumerate([[1.,.2,.2,1.],[.2,1.,.2,1.],[.2,.2,1.,1.]]):
        if scn.ngeom >= scn.maxgeom:
            break
        cz = ee_rot[:, col]
        cx = np.array([1,0,0]) if abs(cz[0]) < 0.9 else np.array([0,1,0])
        cx = cx - np.dot(cx, cz) * cz; cx /= np.linalg.norm(cx)
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
            [0.003, 0.03, 0], ee_pos + cz * 0.03,
            np.column_stack([cx, np.cross(cz, cx), cz]).flatten(),
            np.array(rgba, dtype=np.float32))
        scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description="T4 contraction sweep test.")
    parser.add_argument('--duration', type=float, default=3.0,
                        help="Sweep duration in seconds (default 3.0, matches joint_phase).")
    parser.add_argument('--hip',   type=float, default=DEFAULT_HIP)
    parser.add_argument('--knee',  type=float, default=DEFAULT_KNEE)
    parser.add_argument('--wrist', type=float, default=DEFAULT_WRIST)
    parser.add_argument('--ee',    type=float, default=DEFAULT_EE)
    args = parser.parse_args()

    targets_deg = {
        'hip_pitch_FL': args.hip,
        'knee_FL':      args.knee,
        'wrist_FL':     args.wrist,
        'ee_FL':        args.ee,
    }

    print()
    print("=" * 60)
    print("  T4 — Contraction sweep (FL, RED leg)")
    print(f"  Duration: {args.duration:.1f} s  (matches joint_phase default)")
    print()
    print("  Target angles:")
    for jname, deg in targets_deg.items():
        print(f"    {jname:<20} → {deg:+.1f}°")
    print()
    print("  WATCH FOR:")
    print("    - Any part of the FL leg clipping through the frame body")
    print("    - FL leg hitting the floor or wall during the sweep")
    print("    - FL leg interfering with FR / BL / BR legs")
    print("    - Final pose: leg should be visibly tucked away from +X wall")
    print()
    print("  Press R to replay from the beginning.")
    print("  Ctrl+C to quit.")
    print("=" * 60)
    print()

    # ── load model ────────────────────────────────────────────────────────────
    model, data, plate_ids, magnet_ids, sphere_gids = setup_model()
    pid = PIDController(model)

    ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "electromagnet_FL")

    # Get joint ids and qpos addresses
    jids  = {n: _get_jid(model, n) for n in FL_JOINTS}
    qadrs = {n: model.jnt_qposadr[jids[n]] for n in FL_JOINTS}

    # ctrl_targets: baked joint qpos, same as sim.py line 503
    ctrl_targets = np.array([data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
                              for i in range(model.nu)])

    # ── settle: PID + no magnets, matches sim.py settle phase exactly ─────────
    settle_steps = int(SETTLE_TIME / model.opt.timestep)
    print(f"Settling for {SETTLE_TIME:.1f} s (PID, no magnets)...")
    for i in range(settle_steps):
        data.xfrc_applied[:] = 0
        data.ctrl[:] = pid.compute(model, data, ctrl_targets, model.opt.timestep)
        mujoco.mj_step(model, data)
        if i % int(settle_steps / 4) == 0:
            pct = int(100 * i / settle_steps)
            print(f"  {pct}%...")
    print("Settled. Starting sweep...")

    # Print where the robot actually landed so we can verify
    body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")
    body_pos = data.xpos[body_id].copy()
    print(f"  Body position after settle: {body_pos.round(3)}")

    # Capture start angles from settled qpos
    start_deg = {n: math.degrees(data.qpos[qadrs[n]]) for n in FL_JOINTS}
    targets_rad = {n: math.radians(targets_deg[n]) for n in FL_JOINTS}

    sweep_start = time.perf_counter()
    replaying   = [False]

    def _reset_sweep():
        # Reset to settled qpos and restart timer
        for n in FL_JOINTS:
            data.qpos[qadrs[n]] = math.radians(start_deg[n])
        mujoco.mj_forward(model, data)
        sweep_start_ref[0] = time.perf_counter()

    sweep_start_ref = [sweep_start]

    def key_callback(keycode):
        if keycode == ord('R') or keycode == ord('r'):
            print("Replaying sweep...")
            _reset_sweep()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = body_pos          # point at actual settled body, not hardcoded origin
        viewer.cam.distance  = 1.4
        viewer.cam.azimuth   = 110
        viewer.cam.elevation = -15

        frame_dt = 1.0 / 60.0
        last_pct = -1

        while viewer.is_running():
            t_sweep = time.perf_counter() - sweep_start_ref[0]
            alpha   = _quintic(t_sweep, args.duration)

            # Apply interpolated joint angles
            for n in FL_JOINTS:
                q0 = math.radians(start_deg[n])
                q1 = targets_rad[n]
                data.qpos[qadrs[n]] = q0 + alpha * (q1 - q0)
            mujoco.mj_forward(model, data)

            # Progress printout
            pct = int(alpha * 100)
            if pct != last_pct and pct % 10 == 0:
                last_pct = pct
                if pct < 100:
                    print(f"  Sweep: {pct:3d}%")
                else:
                    print("  Sweep complete — holding final pose. Press R to replay.")

            # Draw EE frame
            scn = viewer._user_scn
            scn.ngeom = 0
            _draw_ee_frame(scn, data, ee_bid)
            viewer.sync()
            time.sleep(frame_dt)


if __name__ == "__main__":
    main()