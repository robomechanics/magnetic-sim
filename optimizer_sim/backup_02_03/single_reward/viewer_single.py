"""
viewer_single.py

Interactive LIVE viewer for magnetic wall-climbing robot (single mode).
Similar to viewer.py but for the single-mode optimization system.
"""

from __future__ import annotations

import sys
import time
import json
import select
from pathlib import Path
from typing import Dict, Any, Optional

import mujoco
import mujoco.viewer
import numpy as np

from sim_sally_magnet_wall_single import (
    build_model,
    reset_data,
    apply_params,
    calculate_magnetic_force,
    DEFAULT_CONFIG,
)


# =============================================================================
# GLOBAL STATE
# =============================================================================

INSPECT_PARAMS: Optional[Dict[str, Any]] = None
SIM_DURATION: float = 5.0
SETTLE_TIME: float = 1.0

# Visual settings
ARROW_RADIUS = 0.005
ARROW_COLOR = (0, 0, 1, 1)

key_state = {"paused": False}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def add_visual_arrow(scene, from_point, to_point):
    """Add a visual arrow to the scene."""
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
    """Check for keyboard input (non-blocking)."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.readline().strip().lower()
    return None


def key_callback(keycode):
    """Handle keyboard callbacks from viewer."""
    if keycode == 32:  # space
        key_state["paused"] = not key_state["paused"]


# =============================================================================
# MAIN VIEWER
# =============================================================================

def main():
    """Launch interactive MuJoCo viewer with LIVE simulation."""
    
    # Get mode from INSPECT_PARAMS (defaults to sideways)
    mode = "sideways"
    if INSPECT_PARAMS:
        mode = INSPECT_PARAMS.get("mode", "sideways")
    
    print("\n" + "=" * 60)
    print("LIVE VIEWER - HOLD MODE")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Duration: {SIM_DURATION}s")
    if INSPECT_PARAMS:
        print(f"Parameter overrides: {len(INSPECT_PARAMS)} params")
    print("=" * 60)
    
    # Build config from DEFAULT_CONFIG
    config = dict(DEFAULT_CONFIG)
    config["mode"] = mode
    
    # Generate XML file or use pre-generated one
    # Check if a pre-generated file exists, otherwise generate one
    from sim_sally_magnet_wall_single import run_xml_generator, generate_scene_with_robot
    
    base_robot_file = f"robot_sally_patched_{mode}.xml"
    if not Path(base_robot_file).exists():
        print(f"Generating XML file for mode '{mode}'...")
        base_robot_file = run_xml_generator(mode=mode)
        print(f"Generated: {base_robot_file}")
    else:
        print(f"Using existing XML: {base_robot_file}")
    
    # Generate scene file that includes the robot
    scene_file = generate_scene_with_robot(base_robot_file)
    config["xml_file"] = scene_file
    
    # Build model
    print("Building model...")
    model, ids = build_model(config)
    print(f"Model built: {model.nq} DOFs, {model.nbody} bodies")
    
    # Apply parameter overrides (from optimizer)
    if INSPECT_PARAMS:
        print("Applying parameter overrides...")
        apply_params(model, INSPECT_PARAMS, config)
    
    # Reset state
    print("Resetting simulation state...")
    data = reset_data(model, config)
    
    # Simulation parameters
    dt = model.opt.timestep
    n_steps = int(SIM_DURATION / dt)
    
    # Magnetic constants from config
    Br = config["Br"]
    V = config["magnet_volume"]
    MU_0 = config["MU_0"]
    max_force = config["max_total_force"] / len(ids["wheel_body_ids"])
    max_dist = config["max_magnetic_distance"]
    
    fromto = np.zeros(6)
    
    print("\nControls:")
    print("  SPACE: Pause/Resume")
    print("  P: Toggle pause")
    print("  ESC: Exit")
    print("\nStarting viewer...\n")
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        step = 0
        
        while viewer.is_running() and step < n_steps:
            # Check for keyboard input
            key = check_keypress()
            if key == "p":
                key_state["paused"] = not key_state["paused"]
                print("Paused" if key_state["paused"] else "Playing")
            
            # Clear previous visual geometry
            viewer.user_scn.ngeom = 0
            
            # Reset applied forces
            for bid in ids["wheel_body_ids"]:
                data.xfrc_applied[bid, :3] = 0.0
            
            # Apply magnetic forces to each wheel
            for gid, bid in zip(ids["wheel_geom_ids"], ids["wheel_body_ids"]):
                # Calculate distance to wall
                dist = mujoco.mj_geomDistance(
                    model, data, gid, ids["wall_id"], 50, fromto
                )
                
                if dist < 0 or dist > max_dist:
                    continue
                
                # Calculate normal vector
                n = fromto[3:6] - fromto[0:3]
                norm = np.linalg.norm(n)
                if norm < 1e-9:
                    continue
                
                n_hat = n / norm
                
                # Calculate magnetic force
                fmag = np.clip(
                    calculate_magnetic_force(dist, Br, V, MU_0),
                    0.0,
                    max_force,
                )
                
                # Apply force
                data.xfrc_applied[bid, :3] = fmag * n_hat
                
                # Visualize magnetic force with blue arrow
                add_visual_arrow(
                    viewer.user_scn,
                    fromto[0:3],
                    fromto[0:3] - 0.25 * n_hat,
                )
            
            # Step simulation if not paused
            if not key_state["paused"]:
                mujoco.mj_step(model, data)
                step += 1
            
            # Update viewer
            viewer.sync()
            time.sleep(dt)
    
    print("\nSimulation complete!")
    print(f"Final position: {data.qpos[:3]}")
    
    # Cleanup temporary scene file
    try:
        if Path(scene_file).exists():
            Path(scene_file).unlink()
            print(f"Cleaned up: {scene_file}")
    except Exception as e:
        print(f"Warning: Could not cleanup {scene_file}: {e}")


def view_best_trial(exp_name: str = "single_hold"):
    """View the best trial from an optimization run."""
    
    # Load best parameters
    best_path = Path("logs") / exp_name / "best.json"
    
    if not best_path.exists():
        print(f"Best parameters not found: {best_path}")
        return
    
    with best_path.open("r", encoding="utf-8") as f:
        best = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Viewing best trial {best['trial_id']}")
    print(f"Reward: {best['reward']:.6f}")
    print(f"{'='*60}")
    
    # Convert to viewer format
    from optimizer_single import build_rollout_params
    params = build_rollout_params(best["params"])
    params["mode"] = "sideways"  # Ensure mode is set
    
    global INSPECT_PARAMS
    INSPECT_PARAMS = params
    
    main()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        view_best_trial(exp_name)
    else:
        # Default: view with default parameters
        main()