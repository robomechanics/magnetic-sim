"""
replay.py

Pure wrapper around viewer.py.

- Loads logs/*/replay_params.json
- Sets viewer.INSPECT_PARAMS
- Calls viewer.main()

Usage:
    python replay.py                          # Interactive selection
    python replay.py logs/stage3_run1         # Replay specific experiment
    python replay.py logs/stage3_run1 --mode hold
    python replay.py --list
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

import viewer


def load_replay_params(replay_file: Path) -> Dict[str, Any]:
    with replay_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_available_replays(logs_dir: Path = Path("logs")) -> list[Path]:
    if not logs_dir.exists():
        return []
    replay_files = list(logs_dir.glob("*/replay_params.json"))
    return sorted(replay_files, key=lambda p: p.stat().st_mtime, reverse=True)


def fmt(x) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def print_one_line_summary(exp_name: str, data: Dict[str, Any]) -> str:
    r = fmt(data.get("reward"))
    rh = fmt(data.get("reward_hold"))
    rd = fmt(data.get("reward_drive"))
    ts = data.get("timestamp", "N/A")
    return f"{exp_name} (reward={r}, hold={rh}, drive={rd}, {ts})"


def apply_mode_overrides(params: Dict[str, Any], mode: Optional[str]) -> Dict[str, Any]:
    """
    Minimal override: only set mode/rollout_mode when user asks.
    Otherwise don't force two-mode execution.
    """
    out = dict(params)
    if mode == "hold":
        out["mode"] = "sideways"
        out["rollout_mode"] = "hold"
    elif mode == "drive":
        out["mode"] = "drive_up"
        out["rollout_mode"] = "drive"
    return out


def main():
    parser = argparse.ArgumentParser(description="Replay optimized simulations (pure viewer wrapper)")
    parser.add_argument("experiment", nargs="?", help="Path to experiment directory (e.g., logs/stage3_run1)")
    parser.add_argument("--mode", choices=["hold", "drive"], help="Override mode/rollout_mode once, then run viewer")
    parser.add_argument("--list", action="store_true", help="List available replays and exit")
    args = parser.parse_args()

    # list
    if args.list:
        replays = list_available_replays()
        if not replays:
            print("No replay files found in logs/")
            return
        print("\nAvailable replays:")
        for i, rp in enumerate(replays, 1):
            exp_name = rp.parent.name
            try:
                data = load_replay_params(rp)
                print(f"{i}. {print_one_line_summary(exp_name, data)}")
            except Exception as e:
                print(f"{i}. {exp_name} (error loading: {e})")
        return

    # choose replay file
    if args.experiment is None:
        replays = list_available_replays()
        if not replays:
            print("No replay files found in logs/")
            return

        print("\nAvailable replays:")
        for i, rp in enumerate(replays, 1):
            exp_name = rp.parent.name
            try:
                data = load_replay_params(rp)
                print(f"{i}. {print_one_line_summary(exp_name, data)}")
            except Exception as e:
                print(f"{i}. {exp_name} (error loading: {e})")

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
        exp_path = Path(args.experiment)
        replay_file = exp_path / "replay_params.json"
        if not replay_file.exists():
            print(f"Replay file not found: {replay_file}")
            return

    # load
    replay_data = load_replay_params(replay_file)
    params = replay_data.get("parameters", {})
    if not isinstance(params, dict):
        raise ValueError("replay_params.json: 'parameters' must be a dict")

    # minimal override (optional)
    final_params = apply_mode_overrides(params, args.mode)

    # PURE: only set INSPECT_PARAMS, then call viewer
    viewer.INSPECT_PARAMS = final_params
    viewer.main()


if __name__ == "__main__":
    main()
