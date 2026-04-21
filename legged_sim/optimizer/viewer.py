"""
viewer.py — Optimizer viewer for Sally's floor-lift and wall-hold simulations.

Launched by combined_optimizer.py after optimization finishes.
Runs the floor-lift sim first (close window → wall-hold sim), then wall-hold.
After each MuJoCo window closes, a per-leg magnetic force plot is displayed
(static, blocking until you close it).

Usage (standalone):
    python viewer.py --params results/.../best_params.json --mode floor
    python viewer.py --params results/.../best_params.json --mode wall
    python viewer.py --params results/.../best_params.json --mode both   ← default

Controls in MuJoCo window:
    SPACE / ENTER  — start / pause
    RIGHT ARROW    — single-step while paused
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer as mjviewer

# ── Path setup ────────────────────────────────────────────────────────────────

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_LEGGED_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (_THIS_DIR, _LEGGED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Matplotlib ────────────────────────────────────────────────────────────────

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("[viewer] matplotlib not found — force plots disabled.")

# ── Key handler (shared) ──────────────────────────────────────────────────────

_key_state = {"paused": True, "step_once": False}


def _key_callback(keycode):
    if keycode in (32, 257):        # SPACE or ENTER
        _key_state["paused"] = not _key_state["paused"]
    elif keycode == 262:            # RIGHT ARROW
        _key_state["step_once"] = True


# ── Force recorder ─────────────────────────────────────────────────────────────

class ForceRecorder:
    """
    Accumulates per-leg magnetic force magnitude samples during the sim loop.
    Call .record() each step, then .show_plot() after the viewer closes.
    """

    def __init__(self, leg_names: list[str]):
        self.leg_names = leg_names
        self._times:  list[float]            = []
        self._forces: dict[str, list[float]] = {leg: [] for leg in leg_names}

    def record(self, t: float, forces: dict[str, float]) -> None:
        self._times.append(t)
        for leg in self.leg_names:
            self._forces[leg].append(abs(forces.get(leg, 0.0)))

    def show_plot(self, title: str = "Per-Leg Magnetic Force") -> None:
        """Show a static matplotlib figure; blocks until the user closes it."""
        if not _HAS_MPL:
            return
        if not self._times:
            print("[viewer] No force data collected — skipping plot.")
            return

        t_arr  = np.array(self._times)
        n_legs = len(self.leg_names)

        fig = plt.figure(figsize=(11, 2.6 * n_legs))
        gs  = gridspec.GridSpec(n_legs, 1, hspace=0.5)
        fig.suptitle(title, fontsize=13, fontweight="bold")

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
        for i, leg in enumerate(self.leg_names):
            ax    = fig.add_subplot(gs[i])
            f_arr = np.array(self._forces[leg])
            color = colors[i % len(colors)]

            ax.plot(t_arr, f_arr, color=color, linewidth=1.2)
            ax.fill_between(t_arr, 0, f_arr, alpha=0.15, color=color)
            ax.set_ylabel("Force (N)", fontsize=9)
            ax.set_title(leg, fontsize=10, loc="left", pad=3)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xlim(t_arr[0], t_arr[-1])
            ymax = f_arr.max() if f_arr.max() > 0 else 1.0
            ax.set_ylim(0, ymax * 1.15)

            peak_idx = int(np.argmax(f_arr))
            ax.annotate(
                f"peak {f_arr[peak_idx]:.2f} N",
                xy=(t_arr[peak_idx], f_arr[peak_idx]),
                xytext=(8, -14), textcoords="offset points",
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

        fig.axes[-1].set_xlabel("Simulation time (s)", fontsize=10)
        plt.show()


# ── Per-leg force computation ─────────────────────────────────────────────────

def _compute_leg_forces(
    model, data,
    magnet_ids: list,
    sphere_gids: dict,
    plate_ids,
    mag_force_fn,
    params: dict,
    leg_names: list[str],
    off_mids: set = None,
) -> dict[str, float]:
    """
    Compute per-leg force magnitude by re-scanning the same geometry that
    _apply_mag already processed this step. Zero for any leg in off_mids.
    """
    if off_mids is None:
        off_mids = set()

    _ft    = np.zeros(6)
    result = {}

    for i, mid in enumerate(magnet_ids):
        leg = leg_names[i] if i < len(leg_names) else f"Leg{i}"
        if mid in off_mids:
            result[leg] = 0.0
            continue

        fvec = np.zeros(3)
        for gid in sphere_gids[mid]:
            best_dist, best_ft = np.inf, None
            for pid in plate_ids:
                d = mujoco.mj_geomDistance(model, data, gid, pid, 50.0, _ft)
                if d < best_dist:
                    best_dist = d
                    best_ft   = _ft.copy()

            max_d = params.get("max_magnetic_distance", 0.05)
            if best_dist <= 0 or best_dist > max_d or best_ft is None:
                continue
            n    = best_ft[3:6] - best_ft[0:3]
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            fvec += mag_force_fn(best_dist, params["Br"]) * (n / norm)

        result[leg] = float(np.linalg.norm(fvec))

    return result


# ── Floor-lift viewer ─────────────────────────────────────────────────────────

def run_floor(params: dict) -> None:
    """Replay the floor-lift scenario with a live viewer + force recording."""
    from sim_opt_sim import (
        _setup_model,
        _apply_mag,
        _mag_force,
        _IK, _PID,
        FEET, SWING_FOOT,
        SETTLE_TIME, LIFT_HOLD, LIFT_DZ,
        IK_EVERY_N,
    )
    from config import MAGNET_BODY_NAMES, TIMESTEP

    _key_state["paused"]    = True
    _key_state["step_once"] = False

    model, data, plate_ids, magnet_ids, sphere_gids = _setup_model(params)

    leg_names = [n.split("_")[-1] for n in MAGNET_BODY_NAMES]

    swing_mag_bid  = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{SWING_FOOT}")
    STANCE_FEET    = [f for f in FEET if f != SWING_FOOT]
    stance_ee_bids = {
        foot: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{foot}")
        for foot in STANCE_FEET
    }

    ik  = _IK(model)
    pid = _PID(model)

    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]

    settled         = False
    z_settle_base   = {}
    xy_lift_base    = {}
    target_pos      = np.zeros(3)
    ik_step_counter = 0
    total_time      = SETTLE_TIME + LIFT_HOLD

    recorder = ForceRecorder(leg_names)

    print("\n[viewer] Floor-lift simulation — press SPACE to start.")

    with mjviewer.launch_passive(model, data, key_callback=_key_callback) as v:
        v.cam.lookat[:] = [0.0, 0.0, 0.1]
        v.cam.distance  = 1.2
        v.cam.elevation = -20
        v.cam.azimuth   = 45

        step_start = time.perf_counter()

        while v.is_running():
            if _key_state["paused"] and not _key_state["step_once"]:
                v.sync()
                time.sleep(0.02)
                continue
            _key_state["step_once"] = False

            # ── Settle: all magnets ON ─────────────────────────────────────
            if data.time < SETTLE_TIME:
                data.xfrc_applied[:] = 0
                _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params)
                data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)

                recorder.record(data.time, _compute_leg_forces(
                    model, data, magnet_ids, sphere_gids, plate_ids,
                    _mag_force, params, leg_names))

                mujoco.mj_step(model, data)

            # ── Lift: FL magnet OFF, stance ON ────────────────────────────
            elif data.time < total_time:
                if not settled:
                    settled = True
                    ik.record_stance(data)
                    ee_home    = ik.ee_pos(data, SWING_FOOT).copy()
                    target_pos = ee_home + np.array([0.0, 0.0, LIFT_DZ])
                    for foot, bid in stance_ee_bids.items():
                        pos = data.xpos[bid].copy()
                        z_settle_base[foot] = pos[2]
                        xy_lift_base[foot]  = pos[:2].copy()

                data.xfrc_applied[:] = 0
                off = {swing_mag_bid}
                _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                           params, off_mids=off)

                ik_step_counter += 1
                if ik_step_counter >= IK_EVERY_N:
                    ik_step_counter = 0
                    ctrl_targets = ik.solve(target_pos, data, IK_EVERY_N * TIMESTEP)

                data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)

                recorder.record(data.time, _compute_leg_forces(
                    model, data, magnet_ids, sphere_gids, plate_ids,
                    _mag_force, params, leg_names, off_mids=off))

                mujoco.mj_step(model, data)

            # ── Sim done — keep window open for inspection ─────────────────
            else:
                v.sync()
                time.sleep(0.02)
                continue

            v.sync()
            elapsed = time.perf_counter() - step_start
            if TIMESTEP - elapsed > 0:
                time.sleep(TIMESTEP - elapsed)
            step_start = time.perf_counter()

    recorder.show_plot("Floor-Lift · Per-Leg Magnetic Force")


# ── Wall-hold viewer ──────────────────────────────────────────────────────────

def run_wall(params: dict) -> None:
    """Replay the wall-hold scenario with a live viewer + force recording."""
    from sim_vertopt_sim import (
        _setup_model,
        _apply_mag,
        _mag_force,
        _PID,
        SETTLE_TIME,
    )
    from config import MAGNET_BODY_NAMES, TIMESTEP

    # Grab HOLD_TIME with a sensible fallback in case the name differs.
    try:
        from sim_vertopt_sim import HOLD_TIME
        total_time = SETTLE_TIME + HOLD_TIME
    except ImportError:
        try:
            from sim_vertopt_sim import WALL_HOLD
            total_time = SETTLE_TIME + WALL_HOLD
        except ImportError:
            total_time = SETTLE_TIME + 3.0

    _key_state["paused"]    = True
    _key_state["step_once"] = False

    model, data, plate_ids, magnet_ids, sphere_gids = _setup_model(params)

    leg_names = [n.split("_")[-1] for n in MAGNET_BODY_NAMES]

    pid = _PID(model)

    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]

    recorder = ForceRecorder(leg_names)

    print("\n[viewer] Wall-hold simulation — press SPACE to start.")

    with mjviewer.launch_passive(model, data, key_callback=_key_callback) as v:
        v.cam.lookat[:] = [0.0, 0.0, 0.1]
        v.cam.distance  = 1.2
        v.cam.elevation = -20
        v.cam.azimuth   = 45

        step_start = time.perf_counter()

        while v.is_running():
            if _key_state["paused"] and not _key_state["step_once"]:
                v.sync()
                time.sleep(0.02)
                continue
            _key_state["step_once"] = False

            if data.time < total_time:
                data.xfrc_applied[:] = 0
                _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params)
                data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)

                recorder.record(data.time, _compute_leg_forces(
                    model, data, magnet_ids, sphere_gids, plate_ids,
                    _mag_force, params, leg_names))

                mujoco.mj_step(model, data)
            else:
                v.sync()
                time.sleep(0.02)
                continue

            v.sync()
            elapsed = time.perf_counter() - step_start
            if TIMESTEP - elapsed > 0:
                time.sleep(TIMESTEP - elapsed)
            step_start = time.perf_counter()

    recorder.show_plot("Wall-Hold · Per-Leg Magnetic Force")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optimizer viewer — floor/wall sim with per-leg force plots."
    )
    parser.add_argument("--params", "-p", required=True,
                        help="Path to best_params.json")
    parser.add_argument("--mode", "-m", default="both",
                        choices=["floor", "wall", "both"],
                        help="Which simulation to view (default: both)")
    args = parser.parse_args()

    params_path = os.path.abspath(args.params)
    if not os.path.exists(params_path):
        sys.exit(f"ERROR: params file not found: {params_path}")

    with open(params_path) as f:
        params = json.load(f)

    # JSON arrays come back as plain lists — ensure correct types.
    for key in ("ground_friction", "solref", "solimp"):
        if key in params:
            params[key] = list(params[key])
    if "noslip_iterations" in params:
        params["noslip_iterations"] = int(params["noslip_iterations"])

    print(f"[viewer] Loaded params from {params_path}")

    if args.mode in ("floor", "both"):
        run_floor(params)

    if args.mode in ("wall", "both"):
        print("\n[viewer] Floor viewer closed — launching wall-hold viewer.")
        run_wall(params)

    print("[viewer] Done.")


if __name__ == "__main__":
    main()