"""
test_T2.py — Verify T2: body_target_se3 override in IKSolver.

Two parts:
  1. Structural checks (no viewer) — always runs.
  2. Visual check (--visual) — settles with magnets, then lets you toggle between:
       T = Ry(−30°): front of robot tilts toward −Z (floor direction)
       Y = Ry(+30°): front of robot tilts toward +Z (ceiling direction)
       R = reset to settled pose
     Report which one looks like "front tilts toward wall" — that resolves T10.

Place in legged_sim/test_files/ and run from legged_sim/:
    python test_files/test_T2.py
    python test_files/test_T2.py --visual
"""

import argparse
import math
import sys
import os
import numpy as np
import mujoco
import mujoco.viewer

_HERE      = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR   = os.path.join(_HERE, "..")
LEGGED_DIR = os.path.join(_SIM_DIR, "..", "legged_sim")
sys.path.insert(0, _SIM_DIR)
sys.path.insert(0, LEGGED_DIR)

from sim import (setup_model, apply_mag, IKSolver, PIDController,
                 TIMESTEP, SETTLE_TIME, IK_EVERY_N, SWING_FOOT)
from sequences import IKTarget, ControlMode

PASS        = "  ✅ PASS"
FAIL        = "  ❌ FAIL"
MAG_LOCK_T  = 1.0   # seconds of post-settle mag lock


# ── helpers ───────────────────────────────────────────────────────────────────

def _settle(model, data, pid):
    """Gravity drop + PID, no magnets — matches sim.py settle."""
    ctrl_targets = np.array([data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
                              for i in range(model.nu)])
    for _ in range(int(SETTLE_TIME / model.opt.timestep)):
        data.xfrc_applied[:] = 0
        data.ctrl[:] = pid.compute(model, data, ctrl_targets, model.opt.timestep)
        mujoco.mj_step(model, data)
    return ctrl_targets


def _mag_lock(model, data, pid, ctrl_targets, plate_ids, magnet_ids, sphere_gids):
    """All magnets on + PID — lets robot fully stabilise before visual."""
    for _ in range(int(MAG_LOCK_T / model.opt.timestep)):
        data.xfrc_applied[:] = 0
        apply_mag(model, data, sphere_gids, plate_ids, magnet_ids)
        data.ctrl[:] = pid.compute(model, data, ctrl_targets, model.opt.timestep)
        mujoco.mj_step(model, data)


def _ry(deg: float) -> np.ndarray:
    """4×4 rotation matrix about world Y."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    T = np.eye(4)
    T[0, 0] =  c;  T[0, 2] = s
    T[2, 0] = -s;  T[2, 2] = c
    return T


def _tilt(settled_T: np.ndarray, deg: float) -> np.ndarray:
    """World-frame Ry(deg) left-multiplied onto settled body SE3."""
    out = settled_T.copy()
    out[:3, :3] = _ry(deg)[:3, :3] @ settled_T[:3, :3]
    return out


# ── structural checks ─────────────────────────────────────────────────────────

def run_structural(model, data, ik):
    print("\n" + "=" * 60)
    print("  test_T2.py — structural checks")
    print("=" * 60)

    dt      = IK_EVERY_N * TIMESTEP
    ee_home = ik.ee_pos(data, SWING_FOOT).copy()
    tilt_T  = _tilt(ik._settled_body_T, -30)

    results = []

    def chk(label, val):
        results.append(val)
        print(f"  {label}: {val}  {PASS if val else FAIL}")

    print("\n── Fields")
    c2 = IKTarget.control2(pos=ee_home)
    c1 = IKTarget.control1(pos=ee_home, jnt_range_override={"knee_FL": (-1.6, 1.6)})
    c3 = IKTarget.control3(pos=ee_home, joint_targets={"knee_FL": 0.5})
    chk("control2 body_target_se3 is None",  c2.body_target_se3 is None)
    chk("control1 body_target_se3 is None",  c1.body_target_se3 is None)
    chk("control3 body_target_se3 is None",  c3.body_target_se3 is None)

    print("\n── _settled_body_T")
    chk("_settled_body_T not None",          ik._settled_body_T is not None)
    chk("_settled_body_T shape (4,4)",       ik._settled_body_T.shape == (4, 4))

    print("\n── Constructors accept body_target_se3")
    c2t = IKTarget.control2(pos=ee_home, body_target_se3=tilt_T)
    c1t = IKTarget.control1(pos=ee_home, jnt_range_override={"knee_FL": (-1.6, 1.6)},
                            body_target_se3=tilt_T)
    chk("control2 body_target_se3 set",      c2t.body_target_se3 is not None)
    chk("control1 body_target_se3 set",      c1t.body_target_se3 is not None)

    print("\n── solve() no crash")
    for label, tgt in [("None", c2), ("tilt", c2t)]:
        try:
            ik.solve(tgt, data, dt, swing_foot=SWING_FOOT)
            chk(f"solve(body_target_se3={label}) no crash", True)
        except Exception as e:
            chk(f"solve(body_target_se3={label}) no crash", False)
            print(f"    Exception: {e}")

    n = sum(results)
    print(f"\n{'='*60}")
    print(f"  Result: {n}/{len(results)} passed  {'✅' if n==len(results) else '❌'}")
    print(f"{'='*60}")
    return n == len(results)


# ── visual check ──────────────────────────────────────────────────────────────

def run_visual(model, data, ik, pid, ctrl_targets,
               plate_ids, magnet_ids, sphere_gids):

    body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")
    body_pos = data.xpos[body_id].copy()
    fl_mid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "electromagnet_FL")
    swing_off = frozenset([fl_mid])

    ee_home   = ik.ee_pos(data, SWING_FOOT).copy()
    settled_T = ik._settled_body_T.copy()

    tgts = {
        "reset" : IKTarget.control2(pos=ee_home),
        "tilt-" : IKTarget.control2(pos=ee_home, body_target_se3=_tilt(settled_T, -30)),
        "tilt+" : IKTarget.control2(pos=ee_home, body_target_se3=_tilt(settled_T, +30)),
    }
    mode      = ["reset"]
    ik_ctr    = [0]
    live_ctrl = ctrl_targets.copy()

    print("\n── VISUAL CHECK")
    print("  Keys:")
    print("    T = Ry(−30°) — front tilts toward FLOOR (−Z)")
    print("    Y = Ry(+30°) — front tilts toward CEILING (+Z)")
    print("    R = reset to settled")
    print()
    print("  ⚠️  Note which (T or Y) tilts the front toward the +X wall.")
    print("      That resolves T10 (body rotation composition order).")
    print(f"\n  Body at: {body_pos.round(3)}")

    def key_callback(keycode):
        if keycode in (ord('T'), ord('t')):
            mode[0] = "tilt-"; print("  → Ry(−30°): front → floor.")
        elif keycode in (ord('Y'), ord('y')):
            mode[0] = "tilt+"; print("  → Ry(+30°): front → ceiling.")
        elif keycode in (ord('R'), ord('r')):
            mode[0] = "reset"; print("  → Reset.")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = body_pos
        viewer.cam.distance  = 1.6
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -10

        while viewer.is_running():
            data.xfrc_applied[:] = 0
            apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                      off_mids=swing_off)

            ik_ctr[0] += 1
            if ik_ctr[0] >= IK_EVERY_N:
                ik_ctr[0] = 0
                live_ctrl[:] = ik.solve(tgts[mode[0]], data,
                                        IK_EVERY_N * TIMESTEP,
                                        swing_foot=SWING_FOOT)

            data.ctrl[:] = pid.compute(model, data, live_ctrl, model.opt.timestep)
            mujoco.mj_step(model, data)
            viewer.sync()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true")
    args = parser.parse_args()

    model, data, plate_ids, magnet_ids, sphere_gids = setup_model()
    pid = PIDController(model)
    ik  = IKSolver(model)

    print("Phase 1: settle (gravity + PID, no magnets)...")
    ctrl_targets = _settle(model, data, pid)
    print("Phase 2: magnetic lock (all magnets on + PID)...")
    _mag_lock(model, data, pid, ctrl_targets, plate_ids, magnet_ids, sphere_gids)
    ik.record_stance(data)
    print("Stable.\n")

    run_structural(model, data, ik)

    if args.visual:
        run_visual(model, data, ik, pid, ctrl_targets,
                   plate_ids, magnet_ids, sphere_gids)
    else:
        print("\n  Run with --visual to open the viewer.")


if __name__ == "__main__":
    main()