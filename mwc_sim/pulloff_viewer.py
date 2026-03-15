"""
pulloff_viewer.py - Interactive viewer for the pull-off test.
Called automatically by pulloff_sim.py, or run standalone.

Pull force applied at magnet body COM (+Z). Displacement tracked as
change in magnet COM Z from ramp start.

Arrows:
  Blue — magnetic attraction force at each sampling sphere
  Red  — applied upward pull force at magnet COM

Controls: ENTER or SPACE to start/pause.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from pulloff_config import (
    PARAMS, SETTLE_TIME, SIM_DURATION,
    DETACH_DIST, DETACH_HOLD, PULL_RATE,
    REAL_TIME_FACTOR, ARROW_RADIUS,
    MAG_ARROW_SCALE, FORCE_ARROW_SCALE,
    TELEMETRY_INTERVAL,
)
from pulloff_sim import setup_model, apply_mag, mag_force

key_state = {"paused": True}


def key_callback(keycode):
    if keycode in (32, 257):   # SPACE or ENTER
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


def run_viewer(pull_rate=PULL_RATE):
    from pulloff_config import PARAMS
    model, data, plate_id, magnet_id, sphere_gids = setup_model(PARAMS)

    fromto  = np.zeros(6)
    dt_sim  = float(model.opt.timestep)
    dt_wall = dt_sim / REAL_TIME_FACTOR

    model.vis.map.znear            = 0.001
    model.vis.map.zfar             = 10.0
    model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.diffuse[:] = [1.0, 1.0, 1.0]

    ramp_started  = False
    ramp_t0       = 0.0
    z0            = 0.0
    separated     = False
    pulloff_force = 0.0
    lift_start    = None
    last_print    = -1.0

    print("Press ENTER or SPACE to start.")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as v:
        v.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.distance  = 0.25
        v.cam.azimuth   = 45
        v.cam.elevation = -20
        v.cam.lookat[:] = [0.0, 0.0, 0.02]

        while v.is_running() and data.time < SIM_DURATION:
            if key_state["paused"]:
                v.sync(); time.sleep(0.02); continue

            step_start = time.perf_counter()
            v.user_scn.ngeom = 0
            data.xfrc_applied[:] = 0.0

            # ── Magnetic forces — blue arrows per sphere ──────────────────────
            fvec_total = np.zeros(3)
            fvec_list  = []
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
                    nd = n / norm
                    fvec_list.append((f, nd, data.geom_xpos[gid].copy()))
                    fvec_total += f * nd

                # Clip total assembly force, not per-sphere
                total_mag = np.linalg.norm(fvec_total)
                scale     = min(1.0, PARAMS['max_force_per_wheel'] / total_mag) if total_mag > 1e-10 else 1.0
                data.xfrc_applied[magnet_id, :3] += fvec_total * scale

                for f, nd, sp in fvec_list:
                    arrow_len = max(0.002, MAG_ARROW_SCALE * f * scale)
                    add_arrow(v.user_scn, sp, sp + arrow_len * nd, (0.1, 0.4, 0.9, 0.9))

            f_mag_z = (fvec_total * scale)[2] if (fvec_total.any() and data.time >= SETTLE_TIME / 2) else 0.0

            # ── Pull force ramp — red arrow upward from COM ───────────────────
            f_pull = 0.0
            if data.time >= SETTLE_TIME:
                if not ramp_started:
                    ramp_started = True
                    ramp_t0      = data.time
                    z0           = data.xpos[magnet_id][2]

                t_ramp = data.time - ramp_t0
                f_pull = pull_rate * t_ramp
                data.xfrc_applied[magnet_id, 2] += f_pull   # upward at COM

                com_pos   = data.xpos[magnet_id].copy()
                arrow_len = max(0.005, FORCE_ARROW_SCALE * f_pull / 10.0)
                add_arrow(v.user_scn, com_pos, com_pos + np.array([0, 0, arrow_len]), (0.9, 0.1, 0.1, 0.9))

                if not separated:
                    pulloff_force = max(pulloff_force, f_pull)

            mujoco.mj_step(model, data)

            # ── Detachment check ──────────────────────────────────────────────
            z_disp_mm = (data.xpos[magnet_id][2] - z0) * 1000   # mm
            if ramp_started and not separated:
                if z_disp_mm > DETACH_DIST:
                    if lift_start is None:
                        lift_start = data.time
                    elif data.time - lift_start >= DETACH_HOLD:
                        separated = True
                        print(f"*** DETACHED | Peak: {pulloff_force:.2f} N | disp: {z_disp_mm:.3f} mm ***")
                else:
                    lift_start = None

            # ── Telemetry ─────────────────────────────────────────────────────
            if data.time - last_print >= TELEMETRY_INTERVAL:
                last_print = data.time
                phase = "SEPARATED" if separated else ("RAMP" if ramp_started else "SETTLE")
                print(
                    f"t={data.time:.2f}s [{phase}] "
                    f"F_pull={f_pull:.1f} N  "
                    f"F_mag={-f_mag_z:.1f} N  "
                    f"z_disp={z_disp_mm:.2f} mm"
                )

            v.sync()
            elapsed = time.perf_counter() - step_start
            sleep   = dt_wall - elapsed
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull-rate', type=float, default=PULL_RATE)
    args = parser.parse_args()
    run_viewer(args.pull_rate)