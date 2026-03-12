"""
wrench_viewer.py - Interactive viewer for the wrench/peel test.
Called automatically by wrench_sim.py, or run standalone.

Arrows:
  Blue  — magnetic attraction force at each sampling sphere
  Red   — applied horizontal force at stick tip (only drawn if actually applied)
  Green — torque axis at magnet COM (only drawn if actually applied)

Labels and telemetry show ON/OFF truthfully by reading what xfrc_applied
actually received, so they stay correct regardless of what is commented
in/out inside apply_wrench_force.

Controls: ENTER or SPACE to start/pause.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from wrench_config import (
    PARAMS, SETTLE_TIME, SIM_DURATION,
    DETACH_HOLD, DETACH_THRESHOLD, LEVER_ARM, PULL_RATE,
    APPLY_FORCE, APPLY_MOMENT,
    REAL_TIME_FACTOR, ARROW_RADIUS,
    TORQUE_ARROW_SCALE, FORCE_ARROW_SCALE, MAG_ARROW_SCALE,
    TELEMETRY_INTERVAL,
)
from wrench_sim import setup_model, mag_force, apply_wrench_force

key_state = {"paused": True}


def key_callback(keycode):
    if keycode in (32, 257):
        key_state["paused"] = not key_state["paused"]


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


def add_label(scene, text, pos):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_LABEL,
        size=np.zeros(3),
        pos=np.array(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    g.label = text
    scene.ngeom += 1


def run_viewer(pull_rate):
    model, data, plate_id, magnet_id, sphere_gids, tip_site_id = setup_model()
    fromto  = np.zeros(6)
    dt_sim  = float(model.opt.timestep)
    dt_wall = dt_sim / REAL_TIME_FACTOR

    model.vis.map.znear            = 0.001
    model.vis.map.zfar             = 10.0
    model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.diffuse[:] = [1.0, 1.0, 1.0]

    ramp_started = False
    ramp_t0      = 0.0
    separated    = False
    detach_force = 0.0
    detach_start = None
    last_print   = -1.0

    print("Press ENTER or SPACE to start.")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as v:
        v.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.distance  = 0.35
        v.cam.azimuth   = 45
        v.cam.elevation = -20
        v.cam.lookat[:] = [0.0, 0.0, 0.05]

        while v.is_running() and data.time < SIM_DURATION:
            if key_state["paused"]:
                v.sync(); time.sleep(0.02); continue

            step_start = time.perf_counter()
            v.user_scn.ngeom = 0
            data.xfrc_applied[:] = 0.0

            # ── Magnetic forces — blue arrows per sphere ──────────────────────
            # Inlined so we can draw per-sphere arrows with the direction vector.
            f_mag     = 0.0
            fvec_list = []
            if data.time >= SETTLE_TIME / 2:
                for gid in sphere_gids:
                    dist = mujoco.mj_geomDistance(model, data, gid, plate_id, 50.0, fromto)
                    if dist <= 0 or dist > PARAMS['max_magnetic_distance']:
                        continue
                    f    = mag_force(dist, PARAMS['Br'])
                    n    = fromto[3:6] - fromto[0:3]
                    norm = np.linalg.norm(n)
                    if norm < 1e-10:
                        continue
                    fvec_list.append((f, n / norm, data.geom_xpos[gid].copy()))
                    f_mag += f

                total_fvec = sum(f * nd for f, nd, _ in fvec_list) if fvec_list else np.zeros(3)
                total_mag  = np.linalg.norm(total_fvec)
                scale      = min(1.0, PARAMS['max_force_per_wheel'] / total_mag) if total_mag > 1e-10 else 1.0
                f_mag     *= scale
                data.xfrc_applied[magnet_id, :3] += total_fvec * scale

                for f, nd, sp in fvec_list:
                    arrow_len = max(0.002, MAG_ARROW_SCALE * f * scale)
                    add_arrow(v.user_scn, sp, sp + arrow_len * nd, (0.1, 0.4, 0.9, 0.9))

            # ── Wrench force + torque ─────────────────────────────────────────
            f_pull    = 0.0
            moment    = np.zeros(3)
            force_on  = False
            moment_on = False
            if data.time >= SETTLE_TIME:
                if not ramp_started:
                    ramp_started = True
                    ramp_t0      = data.time

                t_ramp    = data.time - ramp_t0
                f_pull    = pull_rate * t_ramp
                tip_world = data.site_xpos[tip_site_id].copy()
                com_pos   = data.xpos[magnet_id].copy()

                # Compute full moment for display regardless of what's applied
                force_vec  = np.array([f_pull, 0.0, 0.0])
                r          = tip_world - com_pos
                moment     = np.cross(r, force_vec)
                moment_mag = np.linalg.norm(moment)

                apply_wrench_force(model, data, magnet_id, f_pull, tip_site_id)
                force_on  = APPLY_FORCE
                moment_on = APPLY_MOMENT

                # Red arrow at stick tip — only if force actually applied
                if force_on and f_pull > 1e-6:
                    arrow_len = max(0.01, FORCE_ARROW_SCALE * f_pull / 10.0)
                    add_arrow(v.user_scn,
                              tip_world,
                              tip_world + np.array([arrow_len, 0, 0]),
                              (0.9, 0.1, 0.1, 0.9))

                # Green arrow — torque axis at stick tip — only if moment actually applied
                if moment_on and moment_mag > 1e-6:
                    torque_dir = moment / moment_mag
                    arrow_len  = max(0.01, TORQUE_ARROW_SCALE * moment_mag)
                    add_arrow(v.user_scn,
                              tip_world,
                              tip_world + arrow_len * torque_dir,
                              (0.1, 0.85, 0.2, 0.95))

                if not separated:
                    detach_force = max(detach_force, f_pull)
                    if f_mag < DETACH_THRESHOLD:
                        if detach_start is None:
                            detach_start = data.time
                        elif data.time - detach_start >= DETACH_HOLD:
                            separated = True
                            print(f"*** DETACHED | Force: {detach_force:.2f} N | Moment: {moment_mag:.3f} Nm ***")
                    else:
                        detach_start = None

            mujoco.mj_step(model, data)

            # ── Labels ────────────────────────────────────────────────────────
            moment_mag = np.linalg.norm(moment)
            phase      = "SEPARATED" if separated else ("RAMP" if ramp_started else "SETTLE")
            add_label(v.user_scn, f"[{phase}]  t={data.time:.2f}s",                                (0.0, 0.12, 0.0))
            add_label(v.user_scn, f"F_pull: {f_pull:.2f} N  ({'ON' if force_on else 'OFF'})",      (0.0, 0.10, 0.0))
            add_label(v.user_scn, f"Torque: {moment_mag:.4f} Nm  ({'ON' if moment_on else 'OFF'})", (0.0, 0.08, 0.0))
            add_label(v.user_scn, f"F_mag:  {f_mag:.2f} N",                                        (0.0, 0.06, 0.0))

            # ── Telemetry ─────────────────────────────────────────────────────
            if data.time - last_print >= TELEMETRY_INTERVAL:
                last_print = data.time
                print(f"t={data.time:.2f}s [{phase}] "
                      f"F={f_pull:.1f} N ({'on' if force_on else 'off'})  "
                      f"tau={moment_mag:.4f} Nm ({'on' if moment_on else 'off'})  "
                      f"Fmag={f_mag:.1f} N")

            v.sync()
            elapsed = time.perf_counter() - step_start
            if dt_wall - elapsed > 0:
                time.sleep(dt_wall - elapsed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull-rate', type=float, default=PULL_RATE)
    args = parser.parse_args()
    run_viewer(args.pull_rate)