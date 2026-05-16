"""
test_t9.py — Joint sign/direction confirmation for FL Step 1 (T9).

Place this file next to sim.py and run from that directory.

Usage:
    python test_t9.py --joint hip           # hip_pitch_FL = +45°
    python test_t9.py --joint knee          # knee_FL = +90°
    python test_t9.py --joint wrist         # wrist_FL = −90°
    python test_t9.py --joint ee            # ee_FL sweep — see below

For the ee test, first fill in --hip / --knee / --wrist with the angles you
confirmed for those joints, so the upstream pose is correct before testing ee:
    python test_t9.py --joint ee --hip 45 --knee 90 --wrist -90 --ee 90

The pose is frozen each frame (qpos pinned) so gravity has no effect.
Observe the FL leg (RED) and confirm the console description matches what you see.
Press Ctrl+C to quit.
"""

import argparse
import math
import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# ── paths ────────────────────────────────────────────────────────────────────
# test_t9.py lives in test_files/, which sits next to sim.py.
_HERE      = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR   = os.path.join(_HERE, "..")          # directory containing sim.py
LEGGED_DIR = os.path.join(_SIM_DIR, "..", "legged_sim")
sys.path.insert(0, _SIM_DIR)
sys.path.insert(0, LEGGED_DIR)

from sim import setup_model   # reuse existing model-load / bake / spawn logic

# ─────────────────────────────────────────────────────────────────────────────
# What each joint test should look like, and what it means if it's wrong.
# ─────────────────────────────────────────────────────────────────────────────
EXPECT = {
    'hip': dict(
        joint='hip_pitch_FL',
        default_deg=45.0,
        look_for=(
            "The FL (RED) knee mount should swing toward +Y — the LEFT side of\n"
            "the robot, away from the wall face at +X.\n"
            "Looking from above, the hip link rotates CCW (counter-clockwise).\n"
            "\n"
            "WRONG if: the knee swings toward −Y (rightward / toward the wall).\n"
            "FIX:      use −45° instead."
        ),
    ),
    'knee': dict(
        joint='knee_FL',
        default_deg=90.0,
        look_for=(
            "The FL (RED) wrist mount (knee tip) should lift UPWARD (+Z).\n"
            "The knee link folds back toward the hip in a back-bend — like the\n"
            "leg folding up underneath you.\n"
            "\n"
            "WRONG if: the wrist mount drops downward (−Z).\n"
            "FIX:      use −90° instead."
        ),
    ),
    'wrist': dict(
        joint='wrist_FL',
        default_deg=-90.0,
        look_for=(
            "The FL (RED) EE mount (wrist tip) should fold DOWN and INWARD,\n"
            "curling back toward the main frame.\n"
            "The wrist link should not extend outward — it should tuck under.\n"
            "\n"
            "WRONG if: the EE mount extends away from the body.\n"
            "FIX:      use +90° instead."
        ),
    ),
    'ee': dict(
        joint='ee_FL',
        default_deg=90.0,
        look_for=(
            "The magnet face (EE local +Y) should point straight DOWN (world −Z).\n"
            "In the viewer, the GREEN axis stub on the EE body represents local +Y.\n"
            "It should point toward the floor.\n"
            "\n"
            "WRONG if: green stub points up (+Z), or sideways.\n"
            "FIX:      try −90°, then ±180° if neither works.\n"
            "NOTE:     this depends on the upstream hip/knee/wrist pose, so make\n"
            "          sure --hip / --knee / --wrist are set to their confirmed values."
        ),
    ),
}

# ─────────────────────────────────────────────────────────────────────────────

def _set_joint(model, data, name: str, deg: float):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint '{name}' not found in model.")
    data.qpos[model.jnt_qposadr[jid]] = math.radians(deg)


def _draw_world_axes(scn):
    """Draw world X (red), Y (green), Z (blue) axes at the origin."""
    for direction, rgba in [
        ([1,0,0], [1.,.2,.2,1.]),
        ([0,1,0], [.2,1.,.2,1.]),
        ([0,0,1], [.2,.2,1.,1.]),
    ]:
        if scn.ngeom >= scn.maxgeom:
            break
        cz = np.array(direction, float)
        cx = np.array([0,0,1]) if abs(cz[2]) < 0.9 else np.array([1,0,0])
        cx = cx - np.dot(cx, cz) * cz; cx /= np.linalg.norm(cx)
        tip      = cz * 0.15
        rot_flat = np.column_stack([cx, np.cross(cz, cx), cz]).flatten()
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            [0.005, 0.075, 0], tip, rot_flat, np.array(rgba, dtype=np.float32))
        scn.ngeom += 1


def _draw_ee_frame(scn, data, ee_bid):
    """Draw EE local X (red), Y (green), Z (blue) stubs so magnet face is visible."""
    ee_pos = data.xpos[ee_bid].copy()
    ee_rot = data.xmat[ee_bid].reshape(3, 3)
    colors = [[1., .2, .2, 1.], [.2, 1., .2, 1.], [.2, .2, 1., 1.]]
    for col, rgba in enumerate(colors):
        if scn.ngeom >= scn.maxgeom:
            break
        cz = ee_rot[:, col]
        cx = np.array([1, 0, 0]) if abs(cz[0]) < 0.9 else np.array([0, 1, 0])
        cx = cx - np.dot(cx, cz) * cz; cx /= np.linalg.norm(cx)
        size     = [0.003, 0.03, 0]
        tip_pos  = ee_pos + cz * 0.03
        rot_flat = np.column_stack([cx, np.cross(cz, cx), cz]).flatten()
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            size, tip_pos, rot_flat, np.array(rgba, dtype=np.float32))
        scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description="T9 joint sign test for FL leg.")
    parser.add_argument('--joint',  choices=['hip', 'knee', 'wrist', 'ee'], required=True)
    parser.add_argument('--hip',    type=float, default=0.0,
                        help="hip_pitch_FL angle (deg). Default 0.")
    parser.add_argument('--knee',   type=float, default=0.0,
                        help="knee_FL angle (deg). Default 0.")
    parser.add_argument('--wrist',  type=float, default=0.0,
                        help="wrist_FL angle (deg). Default 0.")
    parser.add_argument('--ee',     type=float, default=None,
                        help="ee_FL angle (deg). Only used when --joint ee.")
    args = parser.parse_args()

    info   = EXPECT[args.joint]
    tested = info['joint']

    # Determine the angle for the joint under test.
    angle_map = {
        'hip':   args.hip   if args.joint != 'hip'   else (args.hip   or info['default_deg']),
        'knee':  args.knee  if args.joint != 'knee'  else (args.knee  or info['default_deg']),
        'wrist': args.wrist if args.joint != 'wrist' else (args.wrist or info['default_deg']),
        'ee':    (args.ee if args.ee is not None else info['default_deg']),
    }
    # For the joint under test, use its default if no override.
    if args.joint == 'hip'   and args.hip   == 0.0: angle_map['hip']   = info['default_deg']
    if args.joint == 'knee'  and args.knee  == 0.0: angle_map['knee']  = info['default_deg']
    if args.joint == 'wrist' and args.wrist == 0.0: angle_map['wrist'] = info['default_deg']

    # ── load model ────────────────────────────────────────────────────────────
    model, data, *_ = setup_model()

    ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "electromagnet_FL")

    # ── apply all FL joint angles ─────────────────────────────────────────────
    joints_to_set = {
        'hip_pitch_FL': angle_map['hip'],
        'knee_FL':      angle_map['knee'],
        'wrist_FL':     angle_map['wrist'],
        'ee_FL':        angle_map['ee'],
    }
    for jname, deg in joints_to_set.items():
        _set_joint(model, data, jname, deg)

    mujoco.mj_forward(model, data)

    # Snapshot qpos so we can freeze it each frame.
    frozen_qpos = data.qpos.copy()

    # ── print instructions ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  JOINT TEST: {tested}")
    print(f"  Angles set:")
    for jname, deg in joints_to_set.items():
        marker = " ← UNDER TEST" if jname == tested else ""
        print(f"    {jname:<20} = {deg:+.1f}°{marker}")
    print()
    print("  WHAT TO LOOK FOR:")
    for line in info['look_for'].splitlines():
        print(f"    {line}")
    print()
    print("  FL leg is RED. Press Ctrl+C to quit.")
    print("=" * 60)
    print()

    # ── open viewer ───────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.15, 0.1, 0.4]
        viewer.cam.distance  = 1.2
        viewer.cam.azimuth   = 135   # looking at FL corner from front-left
        viewer.cam.elevation = -15

        while viewer.is_running():
            # Pin qpos every frame so gravity / contacts don't move the pose.
            data.qpos[:] = frozen_qpos
            mujoco.mj_forward(model, data)

            # Draw EE local frame so magnet face direction is visible.
            scn = viewer._user_scn
            scn.ngeom = 0
            _draw_world_axes(scn)
            _draw_ee_frame(scn, data, ee_bid)

            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()