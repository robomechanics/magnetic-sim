"""
viewer.py

PURE visualization entry point.

Rules:
- NO parameter overrides
- NO spawn overrides
- NO control overrides
- NO physics divergence
- Uses identical simulation path as optimization
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import select
from pathlib import Path

from sim_sally_magnet_wall import (
    build_model,
    reset_data,
    apply_params,
    calculate_magnetic_force,
    DEFAULT_CONFIG,
)

# =============================
# Parameters to inspect (ONLY overrides optimizer passes in)
# =============================
INSPECT_PARAMS = {}

# Visual arrows
ARROW_RADIUS = 0.005
ARROW_COLOR = (0, 0, 1, 1)

key_state = {"paused": False}


# =============================================================================
# HELPERS
# =============================================================================
def add_visual_arrow(scene, from_point, to_point):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([ARROW_RADIUS, ARROW_RADIUS, np.linalg.norm(to_point - from_point)]),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(ARROW_COLOR, dtype=np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        ARROW_RADIUS,
        from_point,
        to_point,
    )
    scene.ngeom += 1


def check_keypress():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.readline().strip().lower()
    return None


def key_callback(keycode):
    if keycode == 32:  # space
        key_state["paused"] = not key_state["paused"]


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Get mode from INSPECT_PARAMS
    mode = INSPECT_PARAMS.get("mode", "sideways")
    rollout_mode = INSPECT_PARAMS.get("rollout_mode", "hold")
    # Build config STRICTLY from DEFAULT_CONFIG
    config = dict(DEFAULT_CONFIG)
    config["mode"] = mode

    # Build model
    model, ids = build_model(config)

    # Apply parameter overrides (optimizer / replay only)
    apply_params(model, INSPECT_PARAMS, config)

    # Reset state (single source of truth)
    data = reset_data(model, config)

    dt = model.opt.timestep
    sim_duration = config.get("sim_duration", 5.0)
    n_steps = int(sim_duration / dt)

    # Magnetic constants STRICTLY from config
    Br = config["Br"]
    V = config["magnet_volume"]
    MU_0 = config["MU_0"]
    max_force = config["max_total_force"] / len(ids["wheel_body_ids"])
    max_dist = config["max_magnetic_distance"]

    fromto = np.zeros(6)

    print("\n" + "=" * 60)
    print("Viewer (PURE PLAYER)")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"INSPECT_PARAMS: {INSPECT_PARAMS}")
    print(f"Duration: {sim_duration}s")
    print("=" * 60)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        step = 0
        while viewer.is_running() and step < n_steps:
            key = check_keypress()
            if key == "p":
                key_state["paused"] = not key_state["paused"]

            viewer.user_scn.ngeom = 0

            for bid in ids["wheel_body_ids"]:
                data.xfrc_applied[bid, :3] = 0.0

            for gid, bid in zip(ids["wheel_geom_ids"], ids["wheel_body_ids"]):
                dist = mujoco.mj_geomDistance(
                    model, data, gid, ids["wall_id"], 50, fromto
                )

                if dist < 0 or dist > max_dist:
                    continue

                n = fromto[3:6] - fromto[0:3]
                norm = np.linalg.norm(n)
                if norm < 1e-9:
                    continue

                n_hat = n / norm
                fmag = np.clip(
                    calculate_magnetic_force(dist, Br, V, MU_0),
                    0.0,
                    max_force,
                )

                data.xfrc_applied[bid, :3] = fmag * n_hat
                
                # Blue arrow for magnetic force
                add_visual_arrow(
                    viewer.user_scn,
                    fromto[0:3],
                    fromto[0:3] - 0.25 * n_hat,
                )
            
            # Visualize applied torque (if any) - RED arrows
            if rollout_mode == "drive":
                for i, (aid, bid) in enumerate(zip(ids["wheel_actuator_ids"], ids["wheel_body_ids"])):
                    # Get wheel center position
                    wheel_pos = data.xpos[bid].copy()
                    
                    # Get joint info
                    joint_id = model.actuator_trnid[aid][0]
                    joint_axis = model.jnt_axis[joint_id].copy()
                    dof_adr = model.jnt_dofadr[joint_id]
                    
                    # Get actual velocity and scale for visibility
                    wheel_vel = data.qvel[dof_adr]
                    vel_scale = 0.01  # Scale factor
                    arrow_end = wheel_pos + joint_axis * wheel_vel * vel_scale
                    
                    # Draw RED torque arrow
                    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_ARROW,
                            size=np.array([ARROW_RADIUS, ARROW_RADIUS, abs(wheel_vel * vel_scale)]),
                            pos=np.zeros(3),
                            mat=np.eye(3).flatten(),
                            rgba=np.array([1, 0, 0, 1], dtype=np.float32),  # RED
                        )
                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_ARROW,
                            ARROW_RADIUS,
                            wheel_pos,
                            arrow_end,
                        )
                        viewer.user_scn.ngeom += 1

            if not key_state["paused"]:
                mujoco.mj_step(model, data)
                step += 1

            viewer.sync()
            time.sleep(dt)

if __name__ == "__main__":
    main()
