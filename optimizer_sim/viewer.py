"""
viewer.py

Visualization / sanity-check entry point.

Purpose:
- Visually inspect a rollout using the same simulation logic as optimization
- Verify that tuned parameters actually change robot behavior
- No optimization, no logging, no physics duplication
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import select

from sim_sally_magnet_wall import (
    build_model, 
    reset_data, 
    apply_params, 
    calculate_magnetic_force, 
    DEFAULT_CONFIG,
    run_xml_generator
)


# =============================
# Optional: parameters to inspect
# (can be overridden by optimizer)
# =============================
INSPECT_PARAMS = {
    # Example overrides (can be empty)
    # "Br": 1.2,
    # "o_solref": [3e-4, 20],
}


# =============================
# Simulation settings
# =============================
SIM_DURATION = 20.0
FIXED_TORQUE = 0.5

# Visual arrows
ARROW_RADIUS = 0.005
ARROW_COLOR = (0, 0, 1, 1)


# =============================================================================
# HELPERS
# =============================================================================
def add_visual_arrow(scene, from_point, to_point, radius=0.005, rgba=(0, 0, 1, 1)):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([radius, radius, np.linalg.norm(to_point - from_point)]),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        radius,
        from_point,
        to_point,
    )
    scene.ngeom += 1


key_state = {"paused": False}


def check_keypress():
    # Non-blocking check for any key typed in terminal
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.readline().strip().lower()
    return None


def key_callback(keycode):
    # ASCII 32 = space
    if keycode == 32:
        key_state["paused"] = not key_state["paused"]


# =============================================================================
# MAIN
# =============================================================================
def main():
    from pathlib import Path
    from sim_sally_magnet_wall import generate_scene_with_robot
    
    # Regenerate XML with correct mode
    mode = INSPECT_PARAMS.get("mode", "sideways")
    patched_robot_file = run_xml_generator(mode=mode)
    
    # Generate temporary scene that includes the patched robot
    temp_scene_file = generate_scene_with_robot(patched_robot_file)
    
    try:
        # Update config to use temporary scene
        config = dict(DEFAULT_CONFIG)
        config["xml_file"] = temp_scene_file
        
        # Create config with position offset
        config["robot_position_offset"] = [-0.2, 0.0, 0.0]  # 0.2m up the wall
        
        # Build model using the temporary scene
        model, ids = build_model(config)
        apply_params(model, INSPECT_PARAMS, config)
        data = reset_data(model, config)

        dt = model.opt.timestep
        n_steps = int(SIM_DURATION / dt)

        # Magnetic parameters (same logic as rollout)
        Br = INSPECT_PARAMS.get("Br", DEFAULT_CONFIG["Br"])
        V = DEFAULT_CONFIG["magnet_volume"]
        MU_0 = DEFAULT_CONFIG["MU_0"]
        max_force = DEFAULT_CONFIG["max_total_force"] / len(ids["wheel_body_ids"])

        fromto = np.zeros(6)

        print("\n" + "="*60)
        print("Launching viewer for rollout inspection...")
        print("="*60)
        print(f"Mode: {mode}")
        print(f"Parameters: {INSPECT_PARAMS}")
        print(f"Duration: {SIM_DURATION}s")
        print(f"Fixed torque: {FIXED_TORQUE} N·m")
        print("\nControls:")
        print("  [Space] - Pause/Resume")
        print("  [Esc]   - Quit")
        print("="*60 + "\n")

        sim_time = 0.0
        step = 0

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer._paused = False
            key_state["paused"] = False

            while viewer.is_running() and step < n_steps:
                # Check terminal input
                key = check_keypress()
                if key == "p":
                    key_state["paused"] = not key_state["paused"]
                    print(f"[{'PAUSED' if key_state['paused'] else 'RUNNING'}]")

                # Clear arrows
                viewer.user_scn.ngeom = 0

                # Clear applied forces
                for bid in ids["wheel_body_ids"]:
                    data.xfrc_applied[bid, :3] = 0.0

                total_force = np.zeros(3)
                wheel_forces = {}

                # Magnetic force computation (same as rollout)
                for name, gid, bid in zip(ids["wheel_names"], ids["wheel_geom_ids"], ids["wheel_body_ids"]):
                    dist = mujoco.mj_geomDistance(
                        model, data, gid, ids["wall_id"], 50, fromto
                    )
                    
                    if dist < 0:
                        wheel_forces[name] = 0.0
                        continue

                    n = fromto[3:6] - fromto[0:3]
                    norm = np.linalg.norm(n)
                    if norm < 1e-9:
                        wheel_forces[name] = 0.0
                        continue

                    n_hat = n / norm

                    fmag = calculate_magnetic_force(dist, Br, V, MU_0)
                    fmag = np.clip(fmag, 0.0, max_force)
                    wheel_forces[name] = float(fmag)

                    fvec = fmag * n_hat
                    data.xfrc_applied[bid, :3] = fvec
                    total_force += fvec

                    # Visual arrow
                    add_visual_arrow(
                        viewer.user_scn,
                        fromto[0:3],
                        fromto[0:3] + (-0.05) * n_hat,
                        radius=ARROW_RADIUS,
                        rgba=ARROW_COLOR,
                    )

                # Fixed actuation
                for aid in ids["wheel_actuator_ids"]:
                    data.ctrl[aid] = FIXED_TORQUE

                # Step simulation
                if not key_state["paused"]:
                    mujoco.mj_step(model, data)
                    sim_time += dt
                    step += 1

                    # Print status every 100 steps
                    if step % 100 == 0:
                        print(
                            f"t={sim_time:.3f}s | F={np.linalg.norm(total_force):.2f}N | "
                            f"pos=[{data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f}]"
                        )

                viewer.sync()

                if not key_state["paused"]:
                    time.sleep(dt)

        print("\nViewer closed.")
    
    finally:
        # Clean up temporary files
        try:
            Path(temp_scene_file).unlink()
            Path(patched_robot_file).unlink()
            print(f"[CLEANUP] Removed temporary XML files")
        except Exception as e:
            pass

if __name__ == "__main__":
    main()