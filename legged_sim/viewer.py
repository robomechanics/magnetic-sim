"""
viewer.py — Interactive viewer for the adhesion hold test.

Arrows:
  Blue — magnetic attraction force at each sampling sphere (per wheel)

Controls: ENTER or SPACE to start/pause.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from config import (
    PARAMS, SETTLE_TIME, SIM_DURATION,
    REAL_TIME_FACTOR, ARROW_RADIUS,
    MAG_ARROW_SCALE, TELEMETRY_INTERVAL,
    MAGNET_BODY_NAMES, MAG_ENABLED,
)
from sim import setup_model, mag_force

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


def run_viewer():
    model, data, plate_id, magnet_ids, sphere_gids = setup_model(PARAMS)

    fromto  = np.zeros(6)
    dt_sim  = float(model.opt.timestep)
    dt_wall = dt_sim / REAL_TIME_FACTOR

    model.vis.map.znear            = 0.001
    model.vis.map.zfar             = 10.0
    model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.diffuse[:] = [1.0, 1.0, 1.0]

    last_print = -1.0

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

            step_start = time.perf_counter()
            v.user_scn.ngeom = 0
            data.xfrc_applied[:] = 0.0

            total_fz = 0.0

            # ── Magnetic forces — blue arrows per sphere, per wheel ────────────
            if data.time >= SETTLE_TIME / 2 and MAG_ENABLED:
                for mid in magnet_ids:
                    fvec = np.zeros(3)
                    fvec_list = []
                    for gid in sphere_gids[mid]:
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
                        fvec += f * nd

                    total_mag = np.linalg.norm(fvec)
                    scale     = min(1.0, PARAMS['max_force_per_wheel'] / total_mag) if total_mag > 1e-10 else 1.0
                    data.xfrc_applied[mid, :3] += fvec * scale
                    total_fz += (fvec * scale)[2]

                    for f, nd, sp in fvec_list:
                        arrow_len = max(0.002, MAG_ARROW_SCALE * f * scale)
                        add_arrow(v.user_scn, sp, sp + arrow_len * nd, (0.1, 0.4, 0.9, 0.9))

            mujoco.mj_step(model, data)

            # ── Telemetry ─────────────────────────────────────────────────────
            if data.time - last_print >= TELEMETRY_INTERVAL:
                last_print = data.time
                phase = "SETTLE" if data.time < SETTLE_TIME else "HOLD"
                
                # Joint names and positions
                joint_names = [
                    "hip_pitch_BL", "knee_BL", "wrist_BL", "ee_BL",
                    "hip_pitch_FL", "knee_FL", "wrist_FL", "ee_FL",
                    "hip_pitch_BR", "knee_BR", "wrist_BR", "ee_BR",
                    "hip_pitch_FR", "knee_FR", "wrist_FR", "ee_FR",
                ]
                joint_str_parts = []
                for name in joint_names:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid == -1:
                        joint_str_parts.append(f"{name}=?")
                        continue
                    q = data.qpos[model.jnt_qposadr[jid]]
                    joint_str_parts.append(f"{name}={q:+.1f}°")
                joint_str = "  ".join(joint_str_parts)
                
                print(
                    f"t={data.time:.2f}s [{phase}]  "
                    f"F_mag={-total_fz:.1f}N\n"
                    f"  {joint_str}"
                )

            v.sync()
            elapsed = time.perf_counter() - step_start
            sleep   = dt_wall - elapsed
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    run_viewer()