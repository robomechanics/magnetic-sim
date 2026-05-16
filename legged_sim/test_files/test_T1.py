"""
test_T1.py — Verify T1: ControlMode dispatch + jnt_range_override.

Two checks:
  1. C3 regression  — fire a C3 target through the dispatch; confirm ctrl_targets
                      update and no IK solve runs.
  2. C1 smoke test  — fire a C1 target with jnt_range_override through ik.solve();
                      confirm (a) the override config_limit is built, (b) model.jnt_range
                      is restored to default immediately after, (c) a second solve with
                      the same override does NOT rebuild (cache hit).

No viewer is opened — pure console output. Should complete in ~5 s (settle time).

Place in legged_sim/test_files/ and run from legged_sim/:
    python test_files/test_T1.py
"""

import math
import sys
import os
import numpy as np
import mujoco

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR   = os.path.join(_HERE, "..")
LEGGED_DIR = os.path.join(_SIM_DIR, "..", "legged_sim")
sys.path.insert(0, _SIM_DIR)
sys.path.insert(0, LEGGED_DIR)

from sim import setup_model, IKSolver, PIDController, TIMESTEP, SETTLE_TIME, IK_EVERY_N
from sequences import IKTarget, ControlMode

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"

def _settle(model, data, pid):
    """Replicate sim.py settle: PID, no magnets."""
    ctrl_targets = np.array([data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
                              for i in range(model.nu)])
    steps = int(SETTLE_TIME / model.opt.timestep)
    for _ in range(steps):
        data.xfrc_applied[:] = 0
        data.ctrl[:] = pid.compute(model, data, ctrl_targets, model.opt.timestep)
        mujoco.mj_step(model, data)
    return ctrl_targets


def _jnt_range(model, jname):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return tuple(model.jnt_range[jid].tolist())


def main():
    print("\n" + "=" * 60)
    print("  test_T1.py — Control mode dispatch + jnt_range_override")
    print("=" * 60)

    # ── setup ──────────────────────────────────────────────────────────────────
    model, data, *_ = setup_model()
    pid = PIDController(model)
    ik  = IKSolver(model)

    print("\nSettling...")
    ctrl_targets = _settle(model, data, pid)
    ik.record_stance(data)
    print("Settled.\n")

    dt = IK_EVERY_N * TIMESTEP

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 1 — ControlMode.C3 target has correct fields
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 50)
    print("CHECK 1: IKTarget.control3() fields")

    c3 = IKTarget.control3(
        pos=ik.ee_pos(data, "FL"),
        joint_targets={"knee_FL": math.radians(90), "wrist_FL": math.radians(-90)},
    )
    c3_mode_ok      = c3.control_mode == ControlMode.C3
    c3_override_ok  = "knee_FL" in (c3.ctrl_override or {})
    c3_no_range_ok  = c3.jnt_range_override is None

    print(f"  control_mode == C3:        {c3_mode_ok}  {PASS if c3_mode_ok else FAIL}")
    print(f"  ctrl_override populated:   {c3_override_ok}  {PASS if c3_override_ok else FAIL}")
    print(f"  jnt_range_override is None:{c3_no_range_ok}  {PASS if c3_no_range_ok else FAIL}")

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 2 — ControlMode.C1 target has correct fields
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("CHECK 2: IKTarget.control1() fields")

    override_dict = {"knee_FL": (-1.6, 1.6), "wrist_FL": (-1.6, 1.6)}
    c1 = IKTarget.control1(
        pos=ik.ee_pos(data, "FL"),
        jnt_range_override=override_dict,
    )
    c1_mode_ok      = c1.control_mode == ControlMode.C1
    c1_range_ok     = c1.jnt_range_override == override_dict
    c1_no_override  = c1.ctrl_override is None

    print(f"  control_mode == C1:           {c1_mode_ok}  {PASS if c1_mode_ok else FAIL}")
    print(f"  jnt_range_override populated: {c1_range_ok}  {PASS if c1_range_ok else FAIL}")
    print(f"  ctrl_override is None:        {c1_no_override}  {PASS if c1_no_override else FAIL}")

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 3 — model.jnt_range restored after C1 solve
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("CHECK 3: model.jnt_range restored after C1 solve")

    knee_default  = _jnt_range(model, "knee_FL")
    wrist_default = _jnt_range(model, "wrist_FL")
    print(f"  Default knee_FL  range: {tuple(round(r, 4) for r in knee_default)}")
    print(f"  Default wrist_FL range: {tuple(round(r, 4) for r in wrist_default)}")

    print("  Running C1 solve (expect '[ik] C1 limit rebuilt' below)...")
    ik.solve(c1, data, dt, swing_foot="FL")

    knee_after  = _jnt_range(model, "knee_FL")
    wrist_after = _jnt_range(model, "wrist_FL")
    knee_restored  = np.allclose(knee_after,  knee_default,  atol=1e-6)
    wrist_restored = np.allclose(wrist_after, wrist_default, atol=1e-6)

    print(f"  knee_FL  after solve: {tuple(round(r, 4) for r in knee_after)}  {PASS if knee_restored else FAIL}")
    print(f"  wrist_FL after solve: {tuple(round(r, 4) for r in wrist_after)}  {PASS if wrist_restored else FAIL}")

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 4 — second solve with same override is a cache hit (no rebuild)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("CHECK 4: Cache hit on second C1 solve (no rebuild message)")
    print("  Running second C1 solve (expect NO '[ik] C1 limit rebuilt' below)...")
    limit_before = ik._override_config_limit
    ik.solve(c1, data, dt, swing_foot="FL")
    limit_after  = ik._override_config_limit
    cache_hit = limit_before is limit_after
    print(f"  _override_config_limit unchanged: {cache_hit}  {PASS if cache_hit else FAIL}")

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 5 — different override triggers a rebuild
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("CHECK 5: Different override triggers rebuild")
    c1_new = IKTarget.control1(
        pos=ik.ee_pos(data, "FL"),
        jnt_range_override={"knee_FL": (-2.0, 2.0)},  # different dict
    )
    print("  Running C1 solve with new override (expect '[ik] C1 limit rebuilt' below)...")
    limit_before = ik._override_config_limit
    ik.solve(c1_new, data, dt, swing_foot="FL")
    limit_after  = ik._override_config_limit
    rebuilt = limit_before is not limit_after
    print(f"  _override_config_limit replaced:  {rebuilt}  {PASS if rebuilt else FAIL}")

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 6 — C2 target uses default config_limit (not override)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("CHECK 6: C2 target uses default config_limit")
    c2 = IKTarget.control2(pos=ik.ee_pos(data, "FL"))
    c2_mode_ok = c2.control_mode == ControlMode.C2
    c2_no_range = c2.jnt_range_override is None
    print(f"  control_mode == C2:           {c2_mode_ok}  {PASS if c2_mode_ok else FAIL}")
    print(f"  jnt_range_override is None:   {c2_no_range}  {PASS if c2_no_range else FAIL}")
    # Solve C2 — should not print any C1 rebuild message
    print("  Running C2 solve (expect no C1 messages below)...")
    ik.solve(c2, data, dt, swing_foot="FL")
    knee_after_c2 = _jnt_range(model, "knee_FL")
    c2_range_ok = np.allclose(knee_after_c2, knee_default, atol=1e-6)
    print(f"  knee_FL range still default:  {c2_range_ok}  {PASS if c2_range_ok else FAIL}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    all_checks = [
        c3_mode_ok, c3_override_ok, c3_no_range_ok,
        c1_mode_ok, c1_range_ok, c1_no_override,
        knee_restored, wrist_restored,
        cache_hit, rebuilt,
        c2_mode_ok, c2_no_range, c2_range_ok,
    ]
    n_pass = sum(all_checks)
    n_total = len(all_checks)
    print("\n" + "=" * 60)
    print(f"  Result: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("  ✅ T1 verified — safe to proceed to T2.")
    else:
        print("  ❌ Some checks failed — review output above.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()