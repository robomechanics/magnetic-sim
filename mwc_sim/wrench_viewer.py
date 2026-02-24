"""
wrench_viewer.py - Interactive viewer for the wrench/peel test.
Called automatically by wrench_sim.py, or run standalone.

Controls: ENTER or SPACE to start, SPACE to pause/resume.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from wrench_sim import (
    XMLPATH, PARAMS, SETTLE_TIME, SIM_DURATION,
    DETACH_ANGLE, DETACH_DIST, DETACH_HOLD,
    STICK_TIP_LOCAL,
    setup_model, apply_mag, mag_force,
    apply_wrench_force, get_tilt_and_disp,
)

REAL_TIME_FACTOR = 0.8

ARROW_RADIUS = 0.004
key_state    = {"paused": True}


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


def run_viewer(pull_rate):
    model, data, plate_id, magnet_id, joint_id, sphere_gids = setup_model()
    fromto  = np.zeros(6)
    dt_sim  = float(model.opt.timestep)
    dt_wall = dt_sim / REAL_TIME_FACTOR

    model.vis.map.znear = 0.001
    model.vis.map.zfar  = 10.0
    model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.diffuse[:] = [1.0, 1.0, 1.0]

    ramp_started  = False
    ramp_t0       = 0.0
    pos0          = None
    separated     = False
    detach_force  = 0.0
    detach_start  = None
    last_print    = -1.0

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

            # Magnetic forces — blue arrows
            f_mag_z    = 0.0
            for gid in sphere_gids:
                dist = mujoco.mj_geomDistance(model, data, gid, plate_id, 50.0, fromto)
                if dist <= 0 or dist > PARAMS['max_magnetic_distance']:
                    continue
                f    = np.clip(mag_force(dist, PARAMS['Br']), 0.0, PARAMS['max_force_per_wheel'])
                n    = fromto[3:6] - fromto[0:3]
                norm = np.linalg.norm(n)
                if norm < 1e-10:
                    continue
                fvec = f * (n / norm)
                data.xfrc_applied[magnet_id, :3] += fvec
                f_mag_z += fvec[2]

                sphere_pos = data.geom_xpos[gid].copy()
                arrow_len  = max(0.002, 0.001 * f)
                add_arrow(v.user_scn, sphere_pos, sphere_pos + arrow_len * (n / norm), (0.1, 0.4, 0.9, 0.9))

            # Pull force ramp — red arrow at stick tip, pointing +X
            f_pull = 0.0
            if data.time >= SETTLE_TIME:
                if not ramp_started:
                    ramp_started = True
                    ramp_t0      = data.time
                    pos0         = data.xpos[magnet_id].copy()

                t_ramp = data.time - ramp_t0
                f_pull = pull_rate * t_ramp
                apply_wrench_force(data, magnet_id, model, f_pull)

                # Draw red arrow at stick tip in +X direction
                xmat      = data.xmat[magnet_id].reshape(3, 3)
                tip_world = data.xpos[magnet_id] + xmat @ STICK_TIP_LOCAL
                arrow_len = max(0.01, 0.005 * f_pull / 10.0)
                add_arrow(v.user_scn, tip_world,
                          tip_world + np.array([arrow_len, 0, 0]),
                          (0.9, 0.1, 0.1, 0.9))

                if not separated:
                    detach_force = max(detach_force, f_pull)

            mujoco.mj_step(model, data)

            # Detachment check
            if pos0 is not None:
                tilt_deg, disp_mm = get_tilt_and_disp(model, data, magnet_id, pos0)
                if ramp_started and not separated:
                    if tilt_deg > DETACH_ANGLE and disp_mm > DETACH_DIST:
                        if detach_start is None:
                            detach_start = data.time
                        elif data.time - detach_start >= DETACH_HOLD:
                            separated = True
                            print(f"*** DETACHED | Force: {detach_force:.2f} N | Tilt: {tilt_deg:.1f}° | Disp: {disp_mm:.2f} mm ***")
                    else:
                        detach_start = None
            else:
                tilt_deg, disp_mm = 0.0, 0.0

            # Telemetry every 0.1s
            if data.time - last_print >= 0.1:
                last_print = data.time
                phase = "SEPARATED" if separated else ("RAMP" if ramp_started else "SETTLE")
                print(f"t={data.time:.2f}s [{phase}] F={f_pull:.1f} N  Fmag={-f_mag_z:.1f} N  tilt={tilt_deg:.1f}°  disp={disp_mm:.2f} mm")

            v.sync()
            elapsed = time.perf_counter() - step_start
            sleep   = dt_wall - elapsed
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    import argparse
    from wrench_sim import PULL_RATE
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull-rate', type=float, default=PULL_RATE)
    args = parser.parse_args()
    run_viewer(args.pull_rate)