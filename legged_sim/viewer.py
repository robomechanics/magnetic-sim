"""
viewer.py — Interactive viewer for Sally's crawl gait.

Arrows:
  Blue — magnetic attraction force at each sampling sphere (per wheel)
  Green — foot target positions (stance=dark, swing=bright)

Controls: SPACE or ENTER to start/pause, RIGHT ARROW to step once.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from config import (
    PARAMS, SETTLE_TIME, SIM_DURATION, TIMESTEP,
    REAL_TIME_FACTOR, ARROW_RADIUS,
    MAG_ARROW_SCALE, TELEMETRY_INTERVAL,
    MAGNET_BODY_NAMES, MAG_ENABLED,
    SWING_DURATION, SWING_LIFT_HEIGHT,
    DEMAGNETIZE_HOLD, MAGNETIZE_HOLD,
)
from sim import (
    setup_model, mag_force, apply_mag,
    _build_em_body_ids, _read_contacts,
    SURFACES,
)
from trajectory import TrajectoryPlanner, GaitPhase, GAIT_ORDER
from ik import MinkRobot, EE_FRAME_NAME

key_state = {"paused": True, "step_once": False}


def key_callback(keycode):
    if keycode in (32, 257):   # SPACE or ENTER
        key_state["paused"] = not key_state["paused"]
    elif keycode == 262:        # RIGHT ARROW
        key_state["step_once"] = True


def add_arrow(scene, start, end, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    length = np.linalg.norm(end - start)
    if length < 1e-6:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([ARROW_RADIUS, ARROW_RADIUS, length]),
        pos=np.zeros(3), mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        ARROW_RADIUS, start, end,
    )
    scene.ngeom += 1


def add_sphere(scene, pos, radius, rgba):
    """Draw a small sphere marker at pos."""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius]),
        pos=pos.copy(), mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _print_telemetry(t, phase_label, f_mag_z, planner, contact_states,
                     foot_targets, magnet_states, robot=None):
    """Print a compact multi-line telemetry block."""
    # ── Header ────────────────────────────────────────────────────────────────
    gait_phase  = planner.phase.value if planner else "—"
    swing_foot  = planner.swing_foot  if planner else "—"
    phase_t     = f"{planner._phase_t:.2f}s" if planner else "—"

    print(f"\nt={t:.2f}s  [{phase_label}]  "
          f"gait={gait_phase}  swing={swing_foot}  phase_t={phase_t}  "
          f"F_mag={-f_mag_z:.1f}N")

    # ── Per-foot status ───────────────────────────────────────────────────────
    if planner and foot_targets:
        target_map = {ft.foot: ft for ft in foot_targets}
        for foot in GAIT_ORDER:
            ft      = target_map.get(foot)
            contact = contact_states.get(foot, False) if contact_states else False
            magnet  = magnet_states.get(foot, True)   if magnet_states  else True

            is_swing = (foot == swing_foot)
            role     = "SWING " if is_swing else "stance"

            if ft is not None:
                pos_str = f"({ft.pos_world[0]:+.3f}, {ft.pos_world[1]:+.3f}, {ft.pos_world[2]:+.3f})"
                w_str   = f"w_pos={ft.weight_pos:.1f}"
            else:
                pos_str = "—"
                w_str   = "—"

            contact_sym = "●" if contact else "○"
            magnet_sym  = "M" if magnet  else "m"

            # ── Tracking error ────────────────────────────────────────────────
            if robot is not None and ft is not None:
                actual  = robot.ee_pos_world(foot)
                err     = actual - ft.pos_world
                err_mag = np.linalg.norm(err)
                actual_str = f"({actual[0]:+.3f}, {actual[1]:+.3f}, {actual[2]:+.3f})"
                track_str  = f"  err={err_mag*1000:.1f}mm  actual={actual_str}"
                # Flag large errors
                flag = "  *** BAD TRACK ***" if err_mag > 0.05 else ""
            else:
                track_str = ""
                flag      = ""

            print(f"  {foot} [{role}]  contact={contact_sym}  mag={magnet_sym}  "
                  f"target={pos_str}  {w_str}{track_str}{flag}")


def run_viewer():
    model, data, plate_ids, magnet_ids, sphere_gids = setup_model(PARAMS)

    dt_sim  = float(model.opt.timestep)
    dt_wall = dt_sim / REAL_TIME_FACTOR

    model.vis.map.znear            = 0.001
    model.vis.map.zfar             = 10.0
    model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.diffuse[:] = [1.0, 1.0, 1.0]

    last_print   = -1.0
    planner      = None
    robot        = None
    em_body_ids  = None
    foot_targets = None
    contact_states  = {f: False for f in GAIT_ORDER}
    magnet_states   = {f: True  for f in GAIT_ORDER}

    print("Press ENTER or SPACE to start.")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as v:
        v.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.distance  = 1.5
        v.cam.azimuth   = 45
        v.cam.elevation = -20
        v.cam.lookat[:] = [0.0, 0.0, 0.45]
        v.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        while v.is_running() and data.time < SIM_DURATION:
            if key_state["paused"] and not key_state["step_once"]:
                v.sync(); time.sleep(0.02); continue
            key_state["step_once"] = False

            step_start       = time.perf_counter()
            v.user_scn.ngeom = 0
            data.xfrc_applied[:] = 0.0

            total_fz = 0.0

            # ── Settle phases (1 & 2): all magnets on, no gait ───────────────
            if data.time < SETTLE_TIME:
                if data.time >= SETTLE_TIME / 2 and MAG_ENABLED:
                    total_fz = apply_mag(
                        model, data, sphere_gids, plate_ids, magnet_ids,
                        PARAMS, magnet_states=None,
                    )

            # ── Gait phase: init planner+robot once at SETTLE_TIME ───────────
            else:
                if planner is None:
                    mujoco.mj_forward(model, data)
                    robot       = MinkRobot(model, data)
                    em_body_ids = _build_em_body_ids(model)
                    init_pos    = {foot: robot.ee_pos_world(foot) for foot in GAIT_ORDER}
                    planner     = TrajectoryPlanner(
                        surfaces         = SURFACES,
                        initial_foot_pos = init_pos,
                        body_R_world     = robot.body_R_world,
                        body_pos_world   = robot.body_pos_world,
                        walk_dir         = np.array([1.0, 0.0, 0.0]),   # +X = forward
                        lift_height      = SWING_LIFT_HEIGHT,
                        swing_duration   = SWING_DURATION,
                        demagnetize_hold = DEMAGNETIZE_HOLD,
                        magnetize_hold   = MAGNETIZE_HOLD,
                    )
                    print(f"[viewer] Gait planner initialised at t={data.time:.3f}s")

                contact_states = _read_contacts(model, data, plate_ids, em_body_ids)
                foot_targets   = planner.step(TIMESTEP, contact_states)
                magnet_states  = planner.magnet_states()

                robot.solve_ik(foot_targets, TIMESTEP)

                if MAG_ENABLED:
                    total_fz = apply_mag(
                        model, data, sphere_gids, plate_ids, magnet_ids,
                        PARAMS, magnet_states=magnet_states,
                    )

            # ── Magnetic force arrows (blue) ──────────────────────────────────
            if data.time >= SETTLE_TIME / 2 and MAG_ENABLED:
                _ft = np.zeros(6)
                for mid in magnet_ids:
                    fvec      = np.zeros(3)
                    fvec_list = []
                    for gid in sphere_gids[mid]:
                        best_dist, best_ft = np.inf, None
                        for pid in plate_ids:
                            d = mujoco.mj_geomDistance(model, data, gid, pid, 50.0, _ft)
                            if d < best_dist:
                                best_dist = d
                                best_ft   = _ft.copy()
                        dist   = best_dist
                        fromto = best_ft if best_ft is not None else np.zeros(6)
                        if dist <= 0 or dist > PARAMS['max_magnetic_distance']:
                            continue
                        f    = mag_force(dist, PARAMS['Br'])
                        n    = fromto[3:6] - fromto[0:3]
                        norm = np.linalg.norm(n)
                        if norm < 1e-10:
                            continue
                        nd = n / norm
                        fvec_list.append((f, nd, data.geom_xpos[gid].copy()))
                        fvec += f * nd

                    total_mag = np.linalg.norm(fvec)
                    scale = (min(1.0, PARAMS['max_force_per_wheel'] / total_mag)
                             if total_mag > 1e-10 else 1.0)

                    for f, nd, sp in fvec_list:
                        arrow_len = max(0.002, MAG_ARROW_SCALE * f * scale)
                        add_arrow(v.user_scn, sp, sp + arrow_len * nd,
                                  (0.1, 0.4, 0.9, 0.9))

            # ── Trajectory visualization ──────────────────────────────────────
            if planner and foot_targets:
                target_map = {ft.foot: ft for ft in foot_targets}
                for foot in GAIT_ORDER:
                    ft       = target_map[foot]
                    is_swing = (foot == planner.swing_foot)
                    leg      = planner._legs[foot]

                    # Current confirmed contact position — white dot.
                    add_sphere(v.user_scn, leg.contact_pos, 0.010,
                               (1.0, 1.0, 1.0, 0.8))

                    if is_swing:
                        # Planned landing target — red circle (larger).
                        if planner._swing_p_end is not None:
                            add_sphere(v.user_scn, planner._swing_p_end, 0.018,
                                       (1.0, 0.1, 0.1, 0.9))
                        # Swing departure point — orange dot.
                        if planner._swing_p_start is not None:
                            add_sphere(v.user_scn, planner._swing_p_start, 0.012,
                                       (1.0, 0.5, 0.0, 0.8))
                        # Swing arc — sample 10 points along the spline.
                        if (planner._swing_p_start is not None and
                                planner._swing_p_end is not None and
                                planner._swing_n_hat is not None):
                            for k in range(1, 10):
                                t_s = k / 10.0
                                arc_pos = planner._swing_pos(t_s)
                                add_sphere(v.user_scn, arc_pos, 0.005,
                                           (1.0, 0.6, 0.2, 0.7))
                        # Current IK target — yellow dot.
                        add_sphere(v.user_scn, ft.pos_world, 0.012,
                                   (1.0, 0.9, 0.0, 0.9))
                    else:
                        # Stance IK target — green dot.
                        add_sphere(v.user_scn, ft.pos_world, 0.010,
                                   (0.1, 0.9, 0.1, 0.6))

            mujoco.mj_step(model, data)

            # ── Telemetry ─────────────────────────────────────────────────────
            if data.time - last_print >= TELEMETRY_INTERVAL:
                last_print  = data.time
                phase_label = "SETTLE" if data.time < SETTLE_TIME else "CRAWL"
                _print_telemetry(
                    data.time, phase_label, total_fz,
                    planner, contact_states, foot_targets, magnet_states,
                    robot=robot,
                )

            v.sync()
            elapsed = time.perf_counter() - step_start
            sleep   = dt_wall - elapsed
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    run_viewer()