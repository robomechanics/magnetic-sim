"""
replay.py

Replay a simulation with saved parameters from optimization runs.

Usage:
    python replay.py                          # Interactive: select from available runs
    python replay.py logs/stage3_run1         # Replay specific experiment
    python replay.py logs/stage3_run1 --mode hold   # Replay specific mode only
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

# Import viewer for visualization
import viewer


def load_replay_params(replay_file: Path) -> Dict[str, Any]:
    """Load replay parameters from JSON file."""
    with replay_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_available_replays(logs_dir: Path = Path("logs")) -> list:
    """Find all available replay parameter files."""
    if not logs_dir.exists():
        return []
    
    replay_files = list(logs_dir.glob("*/replay_params.json"))
    return sorted(replay_files, key=lambda p: p.stat().st_mtime, reverse=True)

def format_reward(value) -> str:
    """Format reward value, handling None."""
    return f"{value:.3f}" if value is not None else "N/A"

def display_replay_info(replay_data: Dict[str, Any]):
    """Display information about the replay."""
    print("\n" + "="*60)
    print("REPLAY INFORMATION")
    print("="*60)
    print(f"Experiment: {replay_data.get('experiment_name', 'N/A')}")
    print(f"Trial ID: {replay_data.get('trial_id', 'N/A')}")
    print(f"Combined Reward: {format_reward(replay_data.get('reward'))}")
    print(f"  Hold Reward: {format_reward(replay_data.get('reward_hold'))}")
    print(f"  Drive Reward: {format_reward(replay_data.get('reward_drive'))}")
    print(f"Timestamp: {replay_data.get('timestamp', 'N/A')}")
    print("\nParameters:")
    
    param_descriptions = {
        "Br": "Magnetic remanence (Tesla)",
        "o_solref": "Contact solver reference [timeconst, dampratio]",
        "o_solimp": "Contact solver impedance [dmin, dmax, width, mid, power]",
        "wheel_friction": "Wheel friction [sliding, torsional, rolling]",
        "rocker_stiffness": "Rocker joint stiffness (N·m/rad)",
        "rocker_damping": "Rocker joint damping (N·m·s/rad)",
        "wheel_kp": "Wheel position controller P gain",
        "wheel_kv": "Wheel position controller D gain (velocity)",
    }
    
    params = replay_data.get("parameters", {})
    for k, v in params.items():
        if k not in ['mode', 'rollout_mode']:
            desc = param_descriptions.get(k, "")
            print(f"  {k}: {v}")
            if desc:
                print(f"      → {desc}")
    print("="*60 + "\n")


def replay_simulation(replay_data: Dict[str, Any], mode: Optional[str] = None):
    """
    Replay simulation with saved parameters.
    
    Args:
        replay_data: Loaded replay parameters
        mode: 'hold', 'drive', or None (both)
    """
    config = replay_data.get("config", {})
    params = replay_data["parameters"]
    
    modes_to_run = []
    if mode is None:
        modes_to_run = ["hold", "drive"]
    elif mode == "hold":
        modes_to_run = ["hold"]
    elif mode == "drive":
        modes_to_run = ["drive"]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'hold', 'drive', or None")
    
    for test_mode in modes_to_run:
        # Set up viewer parameters
        test_params = dict(params)
        
        if test_mode == "hold":
            test_params['mode'] = 'sideways'
            test_params['rollout_mode'] = 'hold'
            mode_name = "HOLD MODE (sideways)"
        else:  # drive
            test_params['mode'] = 'drive_up'
            test_params['rollout_mode'] = 'drive'
            mode_name = "DRIVE MODE (drive_up)"
        
        print("\n" + "="*60)
        print(f"REPLAYING: {mode_name}")
        print("="*60)
        
        # Configure viewer
        viewer.INSPECT_PARAMS = test_params
        viewer.SIM_DURATION = config.get("sim_duration", 5.0)
        viewer.FIXED_TORQUE = config.get("fixed_torque", 0.5)
        
        # Run viewer
        viewer.main()
        
        # Pause between modes
        if len(modes_to_run) > 1 and test_mode != modes_to_run[-1]:
            input("\nPress ENTER to continue to next mode...")


def main():
    parser = argparse.ArgumentParser(description="Replay optimized simulations")
    parser.add_argument("experiment", nargs="?", help="Path to experiment directory (e.g., logs/stage3_run1)")
    parser.add_argument("--mode", choices=["hold", "drive"], help="Replay only specific mode")
    parser.add_argument("--list", action="store_true", help="List available replays and exit")
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        replays = list_available_replays()
        if not replays:
            print("No replay files found in logs/ directory")
            return
        
        print("\nAvailable replays:")
        for i, replay_file in enumerate(replays, 1):
            exp_name = replay_file.parent.name
            try:
                data = load_replay_params(replay_file)
                reward = data.get("reward", "N/A")
                timestamp = data.get("timestamp", "N/A")
                print(f"{i}. {exp_name} (reward={reward:.3f}, {timestamp})")
            except Exception as e:
                print(f"{i}. {exp_name} (error loading: {e})")
        return
    
    # Interactive selection
    if args.experiment is None:
        replays = list_available_replays()
        if not replays:
            print("No replay files found in logs/ directory")
            print("Run optimizer.py first to generate replay files")
            return
        
        print("\nAvailable replays:")
        for i, replay_file in enumerate(replays, 1):
            exp_name = replay_file.parent.name
            try:
                data = load_replay_params(replay_file)
                reward = data.get("reward", "N/A")
                print(f"{i}. {exp_name} (reward={reward:.3f})")
            except Exception:
                print(f"{i}. {exp_name} (error loading)")
        
        try:
            choice = int(input("\nSelect replay number: ")) - 1
            if choice < 0 or choice >= len(replays):
                print("Invalid selection")
                return
            replay_file = replays[choice]
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled")
            return
    else:
        # Use specified experiment
        exp_path = Path(args.experiment)
        replay_file = exp_path / "replay_params.json"
        
        if not replay_file.exists():
            print(f"Error: Replay file not found at {replay_file}")
            print(f"Make sure the experiment directory contains replay_params.json")
            return
    
    # Load and display replay info
    try:
        replay_data = load_replay_params(replay_file)
    except Exception as e:
        print(f"Error loading replay file: {e}")
        return
    
    display_replay_info(replay_data)
    
    # Run replay
    try:
        replay_simulation(replay_data, mode=args.mode)
    except KeyboardInterrupt:
        print("\n\nReplay interrupted by user")
    except Exception as e:
        print(f"\nError during replay: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()