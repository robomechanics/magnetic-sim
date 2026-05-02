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


# ── Arrow helper (used by run_pulloff) ────────────────────────────────────────

def _add_arrow(scene, start: np.ndarray, end: np.ndarray,
               rgba: tuple, radius: float = 0.004) -> None:
    """Draw a single ARROW geom into *scene*. No-op if scene is full."""
    if scene.ngeom >= scene.maxgeom:
        return
    length = np.linalg.norm(end - start)
    if length < 1e-6:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([radius, radius, length]),
        pos=np.zeros(3), mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        radius, start, end,
    )
    scene.ngeom += 1


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

        magnitude = np.linalg.norm(fvec)
        max_f = params.get('max_force_per_wheel', 1e10)
        if magnitude > max_f:
            magnitude = max_f
        result[leg] = float(magnitude)

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
        v.opt.geomgroup[3]                                    = 0  # hide collision meshes (group 3) — fixes "4 links" replacing body
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]            = 0  # hide joint axes — fixes glare
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE]        = 0  # hide xfrc_applied arrows — unintended force arrows

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
                    ctrl_targets = ik.solve(target_pos, data, IK_EVERY_N * TIMESTEP, SWING_FOOT)

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


# ── Wall FL-lift viewer ───────────────────────────────────────────────────────

def run_wall(params: dict) -> None:
    """Replay the wall FL-lift scenario with a live viewer + force recording."""
    from sim_vertopt_sim import (
        _setup_model,
        _apply_mag,
        _mag_force,
        _PID,
        FEET, SWING_FOOT,
        SETTLE_TIME, LIFT_HOLD, LIFT_DZ,
        IK_EVERY_N,
    )
    # _IK lives in sim_opt_sim (sim_vertopt_sim re-imports it from there)
    from sim_opt_sim import _IK
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

    print("\n[viewer] Wall FL-lift simulation — press SPACE to start.")

    with mjviewer.launch_passive(model, data, key_callback=_key_callback) as v:
        # Match the camera angle from sim_vertopt_sim.run_with_viewer
        v.cam.lookat[:] = [0.3, 0.0, 0.5]
        v.cam.distance  = 1.2
        v.cam.elevation = -10
        v.cam.azimuth   = 160

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

            # ── Lift: FL pulled off wall in -X, FL magnet OFF, stance ON ──
            elif data.time < total_time:
                if not settled:
                    settled = True
                    ik.record_stance(data)
                    ee_home    = ik.ee_pos(data, SWING_FOOT).copy()
                    # Wall sim: pull off wall in -X (wall normal), not +Z
                    target_pos = ee_home + np.array([-LIFT_DZ, 0.0, 0.0])
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
                    ctrl_targets = ik.solve(target_pos, data, IK_EVERY_N * TIMESTEP, SWING_FOOT)

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

    recorder.show_plot("Wall FL-Lift · Per-Leg Magnetic Force")


# ── Pull-off viewer ───────────────────────────────────────────────────────────

def run_pulloff(params: dict) -> None:
    """Replay the pull-off scenario with a live viewer + force/displacement plot.

    Blue arrows — magnetic attraction force at each sampling sphere.
    Red arrow   — applied upward pull force at magnet COM.

    Uses force-based detachment detection (same logic as sim_pulloff_sim):
    sustained drop below DETACH_FORCE_FRAC * max_force_per_wheel for
    DETACH_HOLD seconds.  After the viewer window closes, the force vs
    time / displacement plot from sim_pulloff_sim.plot() is shown.
    """
    from sim_pulloff_sim import (
        setup_model, apply_mag, mag_force,
        DETACH_FORCE_FRAC,
        _PID, PID_KP, PID_KI, PID_KD,
        plot as pulloff_plot,
    )
    from sim_pulloff_config import (
        SETTLE_TIME, SIM_DURATION, PULL_RATE_OPT, DETACH_HOLD,
        REAL_TIME_FACTOR, ARROW_RADIUS,
        MAG_ARROW_SCALE, FORCE_ARROW_SCALE, TELEMETRY_INTERVAL,
        ACTIVE_PRESET,
    )

    _key_state["paused"]    = True
    _key_state["step_once"] = False

    model, data, plate_id, magnet_id, sphere_gids = setup_model(params)
    fromto   = np.zeros(6)
    dt_sim   = float(model.opt.timestep)
    dt_wall  = dt_sim / REAL_TIME_FACTOR

    pid = _PID(model)
    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]

    pull_rate        = PULL_RATE_OPT
    detach_threshold = DETACH_FORCE_FRAC * params["max_force_per_wheel"]

    ramp_started  = False
    ramp_t0       = 0.0
    z0            = 0.0
    engaged       = False
    detach_start  = None   # sim time when sustained drop began
    pulloff_force = 0.0
    detach_time   = None   # ramp time (s) at confirmed detachment
    separated     = False
    last_print    = -1.0
    records       = []

    print(f"\n[viewer] Pull-off simulation — press SPACE to start.")
    print(f"         max_force_per_wheel = {params['max_force_per_wheel']:.1f} N  "
          f"| detach_threshold = {detach_threshold:.1f} N  "
          f"| pull_rate = {pull_rate:.0f} N/s")

    with mjviewer.launch_passive(model, data, key_callback=_key_callback) as v:
        v.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.distance  = 0.25
        v.cam.azimuth   = 45
        v.cam.elevation = -20
        v.cam.lookat[:] = [0.0, 0.0, 0.02]

        step_start = time.perf_counter()

        while v.is_running() and data.time < SIM_DURATION:
            if _key_state["paused"] and not _key_state["step_once"]:
                v.sync(); time.sleep(0.02); continue
            _key_state["step_once"] = False

            v.user_scn.ngeom  = 0
            data.xfrc_applied[:] = 0.0

            # ── Phase 1 settle: gravity only (0 → SETTLE_TIME/2) ──────────
            # ── Phase 2 settle: magnetic snap   (SETTLE_TIME/2 → SETTLE_TIME)
            f_mag = 0.0
            if data.time >= SETTLE_TIME / 2:
                fvec_total = np.zeros(3)
                sphere_draws = []
                for gid in sphere_gids:
                    dist = mujoco.mj_geomDistance(
                        model, data, gid, plate_id, 50.0, fromto)
                    if dist <= 0 or dist > params["max_magnetic_distance"]:
                        continue
                    f  = mag_force(dist, params["Br"])
                    n  = fromto[3:6] - fromto[0:3]
                    nm = np.linalg.norm(n)
                    if nm < 1e-10:
                        continue
                    nd = n / nm
                    sphere_draws.append((f, nd, data.geom_xpos[gid].copy()))
                    fvec_total += f * nd

                total_mag = np.linalg.norm(fvec_total)
                scale = (min(1.0, params["max_force_per_wheel"] / total_mag)
                         if total_mag > 1e-10 else 1.0)
                data.xfrc_applied[magnet_id, :3] += fvec_total * scale
                f_mag = -fvec_total[2] * scale   # positive = toward plate

                for f, nd, sp in sphere_draws:
                    alen = max(0.002, MAG_ARROW_SCALE * f * scale)
                    _add_arrow(v.user_scn, sp, sp + alen * nd,
                               (0.1, 0.4, 0.9, 0.9), ARROW_RADIUS)

            # ── Ramp: linear pull from SETTLE_TIME onward ────────────────
            f_pull = 0.0
            if data.time >= SETTLE_TIME:
                if not ramp_started:
                    ramp_started = True
                    ramp_t0      = data.time
                    z0           = data.xpos[magnet_id][2]
                t_ramp = data.time - ramp_t0
                f_pull = pull_rate * t_ramp
                data.xfrc_applied[magnet_id, 2] += f_pull
                com  = data.xpos[magnet_id].copy()
                alen = max(0.005, FORCE_ARROW_SCALE * f_pull / 10.0)
                _add_arrow(v.user_scn, com, com + np.array([0.0, 0.0, alen]),
                           (0.9, 0.1, 0.1, 0.9), ARROW_RADIUS)

            # PID holds robot body in baked pose throughout all phases.
            data.ctrl[:] = pid.compute(model, data, ctrl_targets, dt_sim)

            mujoco.mj_step(model, data)

            z_disp = (data.xpos[magnet_id][2] - z0) * 1000 if ramp_started else 0.0

            if ramp_started:
                records.append({
                    "t":      data.time - ramp_t0,
                    "f_pull": f_pull,
                    "f_mag":  f_mag,
                    "z_disp": z_disp,
                })

            # ── Force-based detachment detection ──────────────────────────
            if ramp_started and not separated:
                if not engaged:
                    if f_mag > detach_threshold:
                        engaged = True
                else:
                    if f_mag < detach_threshold:
                        if detach_start is None:
                            detach_start  = data.time
                            pulloff_force = f_pull   # load at onset of failure
                        elif data.time - detach_start >= DETACH_HOLD:
                            separated   = True
                            detach_time = data.time - ramp_t0
                            print(
                                f"*** DETACHED | pull-off: {pulloff_force:.1f} N "
                                f"(target = {params['max_force_per_wheel']:.1f} N) "
                                f"| t = {detach_time:.3f}s ***"
                            )
                    else:
                        detach_start = None   # transient dip — reset clock

            # ── Telemetry ─────────────────────────────────────────────────
            if data.time - last_print >= TELEMETRY_INTERVAL:
                last_print = data.time
                phase = ("SEPARATED" if separated
                         else ("RAMP" if ramp_started else "SETTLE"))
                print(f"  t={data.time:.2f}s [{phase}]  "
                      f"F_pull={f_pull:.1f} N  F_mag={f_mag:.1f} N  "
                      f"z={z_disp:.2f} mm")

            v.sync()
            elapsed = time.perf_counter() - step_start
            sleep   = dt_wall - elapsed
            if sleep > 0:
                time.sleep(sleep)
            step_start = time.perf_counter()

    if records:
        pulloff_plot(records, pulloff_force, detach_time, pull_rate, params)
    else:
        print("[viewer] No ramp data collected — sim ended before SETTLE_TIME.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optimizer viewer — floor/wall/pull-off sim with per-leg force plots."
    )
    parser.add_argument("--params", "-p", required=True,
                        help="Path to best_params.json")
    parser.add_argument("--mode", "-m", default="all",
                        choices=["floor", "wall", "pulloff", "both", "all"],
                        help="Which simulation(s) to view "
                             "(both=floor+wall, all=floor+wall+pulloff, default: both)")
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

    if args.mode in ("floor", "both", "all"):
        run_floor(params)

    if args.mode in ("wall", "both", "all"):
        print("\n[viewer] Floor viewer closed — launching wall-hold viewer.")
        run_wall(params)

    if args.mode in ("pulloff", "all"):
        print("\n[viewer] Launching pull-off viewer.")
        run_pulloff(params)

    print("[viewer] Done.")


if __name__ == "__main__":
    main()