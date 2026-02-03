"""
viewer_single.py

Interactive viewer for magnetic wall-climbing robot hold mode.
Visualizes position-based optimization results.
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import mujoco
import mujoco.viewer
import numpy as np

from sim_sally_magnet_wall_single import rollout, run_xml_generator, generate_scene_with_robot


# =============================================================================
# GLOBAL STATE
# =============================================================================

INSPECT_PARAMS: Optional[Dict[str, Any]] = None
SIM_DURATION: float = 5.0
SETTLE_TIME: float = 1.0


# =============================================================================
# VIEWER FUNCTIONS
# =============================================================================

def main():
    """Launch interactive MuJoCo viewer."""
    
    # Generate XML
    print("Generating XML for viewer...")
    patched_robot_file = run_xml_generator(mode="sideways")
    print(f"Generated robot XML: {patched_robot_file}")
    
    # Verify robot file exists
    if not Path(patched_robot_file).exists():
        raise FileNotFoundError(f"Robot XML not found: {patched_robot_file}")
    
    # Run rollout to get trajectory FIRST
    print("Running simulation...")
    
    params = INSPECT_PARAMS.copy() if INSPECT_PARAMS else {}
    params["_xml_file_override"] = patched_robot_file
    
    result = rollout(
        params=params,
        sim_duration=SIM_DURATION,
        settle_time=SETTLE_TIME,
        log_stride=1,  # Log every step for smooth playback
    )
    
    print(f"Simulation complete: {result.termination}")
    print(f"Reward: {result.summary['reward']:.6f}")
    print(f"Position drift: {result.summary['position_drift_m']:.6f} m")
    
    # Verify robot file still exists after rollout
    if not Path(patched_robot_file).exists():
        raise FileNotFoundError(f"Robot XML was deleted during rollout: {patched_robot_file}")
    
    # Regenerate scene file for viewer (rollout deleted it)
    print(f"Regenerating scene file from: {patched_robot_file}")
    scene_file = generate_scene_with_robot(patched_robot_file)
    print(f"Generated scene file: {scene_file}")
    
    # Verify scene file exists
    if not Path(scene_file).exists():
        raise FileNotFoundError(f"Scene file was not created: {scene_file}")
    
    # Load model for viewer
    model = mujoco.MjModel.from_xml_path(scene_file)
    data = mujoco.MjData(model)
    
    # State for playback
    trajectory = result.trajectory
    current_frame = [0]  # Mutable to modify in callback
    paused = [False]
    playback_speed = [1.0]
    
    def key_callback(keycode):
        """Handle keyboard input."""
        if keycode == 32:  # Space bar
            paused[0] = not paused[0]
            print("Paused" if paused[0] else "Playing")
        elif keycode == 262:  # Right arrow
            current_frame[0] = min(current_frame[0] + 1, len(trajectory) - 1)
        elif keycode == 263:  # Left arrow
            current_frame[0] = max(current_frame[0] - 1, 0)
        elif keycode == 82 or keycode == 114:  # R or r
            current_frame[0] = 0
            print("Reset to start")
        elif keycode == 61:  # + key
            playback_speed[0] = min(playback_speed[0] * 1.5, 10.0)
            print(f"Speed: {playback_speed[0]:.1f}x")
        elif keycode == 45:  # - key
            playback_speed[0] = max(playback_speed[0] / 1.5, 0.1)
            print(f"Speed: {playback_speed[0]:.1f}x")
    
    # Launch viewer
    print("\nControls:")
    print("  SPACE: Pause/Play")
    print("  R: Reset to start")
    print("  +/-: Speed up/down")
    print("  Arrow keys: Step frame")
    print("  ESC: Exit\n")
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        last_time = time.time()
        
        while viewer.is_running():
            if not paused[0]:
                # Advance frame based on playback speed
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                frame_advance = dt * playback_speed[0] * 100  # 100 Hz nominal
                current_frame[0] += int(frame_advance)
                
                if current_frame[0] >= len(trajectory):
                    current_frame[0] = 0  # Loop
            
            # Update visualization
            idx = min(current_frame[0], len(trajectory) - 1)
            frame_data = trajectory[idx]
            
            # Set state
            data.qpos[:7] = frame_data["qpos"]
            data.qvel[:6] = frame_data["qvel"]
            
            # Forward kinematics
            mujoco.mj_forward(model, data)
            
            # Update viewer
            viewer.sync()
            time.sleep(0.01)
    
    # Cleanup
    try:
        Path(scene_file).unlink()
        Path(patched_robot_file).unlink()
    except Exception:
        pass


def view_best_trial(exp_name: str = "single_hold"):
    """View the best trial from an optimization run."""
    
    # Load best parameters
    best_path = Path("logs") / exp_name / "best.json"
    
    if not best_path.exists():
        print(f"Best parameters not found: {best_path}")
        return
    
    with best_path.open("r", encoding="utf-8") as f:
        best = json.load(f)
    
    print(f"Viewing best trial {best['trial_id']}")
    print(f"Reward: {best['reward']:.6f}")
    
    # Convert to rollout format
    from optimizer_single import build_rollout_params
    params = build_rollout_params(best["params"])
    
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