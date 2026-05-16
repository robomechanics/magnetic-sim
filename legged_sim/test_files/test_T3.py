"""
test_T3.py — Verify T3: cost weighting decouples FL EE from body tilt.

Two parts:
  1. Numerical — runs physics for 2 s with each cost setting and measures:
       - Body tilt angle achieved (°)
       - FL joint drift from settled (°)
     Expected: body tilt similar in both; FL drift much lower with high cost.

  2. Visual (--visual) — toggle between low/high FL cost while body tilt
     is commanded. Console prints body tilt + FL drift each second.
       L = low FL cost  (pos_cost=10)
       H = high FL cost (pos_cost=200)
       R = reset to settled

Place in legged_sim/test_files/ and run from legged_sim/:
    python test_files/test_T3.py
    python test_files/test_T3.py --visual
"""

import argparse
import math
import sys
import os
import time
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
MAG_LOCK_T  = 1.0
TILT_DEG    = -30.0   # confirmed in T2: Ry(-30°) tilts front toward wall
TRIAL_TIME  = 2.0     # seconds of physics per cost trial

FL_JOINTS   = ['hip_pitch_FL', 'knee_FL', 'wrist_FL', 'ee_FL']

LOW_COST    = 10
HIGH_COST   = 200


# ── helpers ───────────────────────────────────────────────────────────────────

def _settle(model, data, pid):
    ctrl_targets = np.array([data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
                              for i in range(model.nu)])
    for _ in range(int(SETTLE_TIME / model.opt.timestep)):
        data.xfrc_applied[:] = 0
        data.ctrl[:] = pid.compute(model, data, ctrl_targets, model.opt.timestep)
        mujoco.mj_step(model, data)
    return ctrl_targets


def _mag_lock(model, data, pid, ctrl_targets, plate_ids, magnet_ids, sphere_gids):
    fl_mid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "electromagnet_FL")
    swing_off = frozenset([fl_mid])
    for _ in range(int(MAG_LOCK_T / model.opt.timestep)):
        data.xfrc_applied[:] = 0
        apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, off_mids=swing_off)
        data.ctrl[:] = pid.compute(model, data, ctrl_targets, model.opt.timestep)
        mujoco.mj_step(model, data)


def _ry(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    T = np.eye(4)
    T[0,0]=c; T[0,2]=s; T[2,0]=-s; T[2,2]=c
    return T


def _tilt_T(settled_T, deg):
    out = settled_T.copy()
    out[:3,:3] = _ry(deg)[:3,:3] @ settled_T[:3,:3]
    return out


def _body_xmat(data, model):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")
    return data.xmat[bid].reshape(3,3).copy()


def _tilt_angle_deg(current_R, settled_R):
    """Angle (°) between settled and current body X-axis in world frame."""
    settled_x = settled_R[:, 0]
    current_x = current_R[:, 0]
    dot = np.clip(np.dot(settled_x, current_x), -1., 1.)
    return math.degrees(math.acos(dot))


def _fl_drift_deg(model, ik, ctrl_before, ctrl_after):
    """Max FL joint drift (°) between two ctrl arrays."""
    max_drift = 0.0
    for jname in FL_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        qadr = model.jnt_qposadr[jid]
        for i, cjid in enumerate(ik.ctrl_jids):
            if model.jnt_qposadr[cjid] == qadr:
                drift = abs(math.degrees(ctrl_after[i] - ctrl_before[i]))
                max_drift = max(max_drift, drift)
                break
    return max_drift


def _snapshot(data):
    return data.qpos.copy(), data.qvel.copy()


def _restore(data, snap):
    data.qpos[:], data.qvel[:] = snap
    # zero accelerations / forces — mj_forward will recompute
    data.qacc[:] = 0
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0


def _run_trial(model, data, ik, pid, ctrl_settled,
               plate_ids, magnet_ids, sphere_gids,
               fl_cost, tilt_T, snap):
    """Restore state, run TRIAL_TIME of physics with the given FL cost + body tilt."""
    _restore(data, snap)
    mujoco.mj_forward(model, data)

    fl_mid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "electromagnet_FL")
    swing_off = frozenset([fl_mid])
    ee_home   = ik.ee_pos(data, SWING_FOOT).copy()

    target    = IKTarget.control2(pos=ee_home, body_target_se3=tilt_T,
                                  pos_cost=fl_cost)
    live_ctrl = ctrl_settled.copy()
    ik_ctr    = [0]
    steps     = int(TRIAL_TIME / model.opt.timestep)

    for _ in range(steps):
        data.xfrc_applied[:] = 0
        apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, off_mids=swing_off)
        ik_ctr[0] += 1
        if ik_ctr[0] >= IK_EVERY_N:
            ik_ctr[0] = 0
            live_ctrl[:] = ik.solve(target, data,
                                    IK_EVERY_N * TIMESTEP, swing_foot=SWING_FOOT)
        data.ctrl[:] = pid.compute(model, data, live_ctrl, TIMESTEP)
        mujoco.mj_step(model, data)

    return live_ctrl.copy()


# ── numerical test ────────────────────────────────────────────────────────────

def run_numerical(model, data, ik, pid, ctrl_settled,
                  plate_ids, magnet_ids, sphere_gids,
                  settled_R, settled_T):

    print("\n" + "=" * 65)
    print("  test_T3.py — numerical: FL cost weighting")
    print("=" * 65)
    print(f"  Commanding body tilt: Ry({TILT_DEG}°)")
    print(f"  Trial duration: {TRIAL_TIME:.1f} s each")
    print(f"  Measuring: body tilt angle (°) + max FL joint drift (°)")
    print()

    snap = _snapshot(data)
    tilt = _tilt_T(settled_T, TILT_DEG)

    results = {}
    for cost_label, cost in [("LOW  (cost=10) ", LOW_COST),
                               ("HIGH (cost=200)", HIGH_COST)]:
        print(f"  Running {cost_label}...", flush=True)
        ctrl_after = _run_trial(model, data, ik, pid, ctrl_settled,
                                plate_ids, magnet_ids, sphere_gids,
                                cost, tilt, snap)
        current_R  = _body_xmat(data, model)
        tilt_deg   = _tilt_angle_deg(current_R, settled_R)
        fl_drift   = _fl_drift_deg(model, ik, ctrl_settled, ctrl_after)
        results[cost_label] = (tilt_deg, fl_drift)
        print(f"    body tilt achieved: {tilt_deg:5.1f}°  |  FL drift: {fl_drift:5.1f}°")

    print()
    low_tilt,  low_drift  = results["LOW  (cost=10) "]
    high_tilt, high_drift = results["HIGH (cost=200)"]

    # Pass conditions:
    # 1. Both tilt at least 5° (body task executes in both cases)
    # 2. High cost FL drift is at least 50% less than low cost drift
    both_tilt  = low_tilt  > 5.0 and high_tilt > 5.0
    tilt_close = abs(high_tilt - low_tilt) < 15.0   # within 15° of each other
    drift_reduced = (high_drift < low_drift * 0.5) if low_drift > 0.1 else True

    print(f"  Body tilt > 5° in both trials:             {both_tilt}  {PASS if both_tilt else FAIL}")
    print(f"  Body tilt similar (within 15°):            {tilt_close}  {PASS if tilt_close else FAIL}")
    print(f"  FL drift reduced ≥50% at high cost:        {drift_reduced}  {PASS if drift_reduced else FAIL}")

    all_pass = both_tilt and tilt_close and drift_reduced
    print(f"\n  {'✅ T3 numerical passed.' if all_pass else '❌ Some checks failed — see above.'}")
    print("=" * 65)
    return all_pass


# ── visual test ───────────────────────────────────────────────────────────────

def run_visual(model, data, ik, pid, ctrl_settled,
               plate_ids, magnet_ids, sphere_gids,
               settled_R, settled_T):

    body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")
    body_pos  = data.xpos[body_id].copy()
    fl_mid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "electromagnet_FL")
    swing_off = frozenset([fl_mid])

    ee_home = ik.ee_pos(data, SWING_FOOT).copy()
    tilt    = _tilt_T(settled_T, TILT_DEG)

    snap       = _snapshot(data)
    mode       = ["reset"]     # "low", "high", "reset"
    ik_ctr     = [0]
    live_ctrl  = ctrl_settled.copy()
    last_print = [0.0]

    targets = {
        "low"  : IKTarget.control2(pos=ee_home, body_target_se3=tilt, pos_cost=LOW_COST),
        "high" : IKTarget.control2(pos=ee_home, body_target_se3=tilt, pos_cost=HIGH_COST),
        "reset": IKTarget.control2(pos=ee_home),
    }

    print("\n── VISUAL CHECK")
    print("  Keys:")
    print(f"    L = low FL cost  (pos_cost={LOW_COST})  — body tilts, FL may drift")
    print(f"    H = high FL cost (pos_cost={HIGH_COST}) — body tilts, FL stays put")
    print(f"    R = reset to settled pose")
    print()
    print("  Console prints body tilt (°) + FL drift (°) each second.")
    print("  Goal: similar body tilt, much lower FL drift with H.\n")

    def key_callback(keycode):
        if keycode in (ord('L'), ord('l')):
            mode[0] = "low"
            _restore(data, snap)
            live_ctrl[:] = ctrl_settled
            mujoco.mj_forward(model, data)
            print(f"  → Low FL cost ({LOW_COST}). Watch FL drift.")
        elif keycode in (ord('H'), ord('h')):
            mode[0] = "high"
            _restore(data, snap)
            live_ctrl[:] = ctrl_settled
            mujoco.mj_forward(model, data)
            print(f"  → High FL cost ({HIGH_COST}). FL should stay planted.")
        elif keycode in (ord('R'), ord('r')):
            mode[0] = "reset"
            _restore(data, snap)
            live_ctrl[:] = ctrl_settled
            mujoco.mj_forward(model, data)
            print("  → Reset.")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = body_pos
        viewer.cam.distance  = 1.6
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -10

        while viewer.is_running():
            data.xfrc_applied[:] = 0

            if mode[0] != "reset":
                apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                          off_mids=swing_off)
            else:
                apply_mag(model, data, sphere_gids, plate_ids, magnet_ids)

            ik_ctr[0] += 1
            if ik_ctr[0] >= IK_EVERY_N:
                ik_ctr[0] = 0
                live_ctrl[:] = ik.solve(targets[mode[0]], data,
                                        IK_EVERY_N * TIMESTEP,
                                        swing_foot=SWING_FOOT)

            data.ctrl[:] = pid.compute(model, data, live_ctrl, TIMESTEP)
            mujoco.mj_step(model, data)

            # ── telemetry ──
            now = data.time
            if now - last_print[0] >= 1.0:
                last_print[0] = now
                current_R  = _body_xmat(data, model)
                tilt_deg   = _tilt_angle_deg(current_R, settled_R)
                fl_drift   = _fl_drift_deg(model, ik, ctrl_settled, live_ctrl)
                cost_label = {"low": f"LOW ({LOW_COST})", "high": f"HIGH ({HIGH_COST})",
                              "reset": "RESET"}[mode[0]]
                print(f"  [{cost_label:12s}]  body_tilt={tilt_deg:5.1f}°  "
                      f"fl_drift={fl_drift:5.1f}°  t={now:.1f}s")

            viewer.sync()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true")
    args = parser.parse_args()

    model, data, plate_ids, magnet_ids, sphere_gids = setup_model()
    pid = PIDController(model)
    ik  = IKSolver(model)

    print("Phase 1: settle...")
    ctrl_settled = _settle(model, data, pid)
    print("Phase 2: magnetic lock...")
    _mag_lock(model, data, pid, ctrl_settled, plate_ids, magnet_ids, sphere_gids)
    ik.record_stance(data)

    settled_R = _body_xmat(data, model)
    settled_T = ik._settled_body_T.copy()
    print("Stable.\n")

    run_numerical(model, data, ik, pid, ctrl_settled,
                  plate_ids, magnet_ids, sphere_gids,
                  settled_R, settled_T)

    if args.visual:
        run_visual(model, data, ik, pid, ctrl_settled,
                   plate_ids, magnet_ids, sphere_gids,
                   settled_R, settled_T)
    else:
        print("\n  Run with --visual to open the viewer.")


if __name__ == "__main__":
    main()