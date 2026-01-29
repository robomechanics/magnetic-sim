"""
inspect_rollout.py

STAGE 4 — ROLLOUT INSPECTION (VISUAL SANITY CHECK)

- Loads parameter sets from Stage-3 logs (JSONL)
- Re-runs rollout to confirm reward/metrics are reproducible
- Optionally launches viewer.py (manual sanity check) using the same XML on disk

Notes:
- Your rollout function is headless by design.
- Visualization is handled by a separate script (viewer.py), so we keep the sim fast.
"""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

from sim_sally_magnet_wall import rollout


def load_trials(jsonl_path: Path) -> List[Dict[str, Any]]:
    trials = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trials.append(json.loads(line))
    return trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to trials.jsonl")
    ap.add_argument("--trial-id", type=int, default=None, help="Trial id to replay")
    ap.add_argument("--best", action="store_true", help="Replay best (max reward) in the log")
    ap.add_argument("--launch-viewer", action="store_true", help="Launch viewer.py after replay")
    args = ap.parse_args()

    log_path = Path(args.log)
    trials = load_trials(log_path)
    if not trials:
        raise RuntimeError(f"No trials found in {log_path}")

    # Choose trial
    if args.best:
        trial = max(trials, key=lambda r: float(r.get("reward", -1e18)))
    elif args.trial_id is not None:
        matches = [t for t in trials if int(t.get("trial_id", -1)) == args.trial_id]
        if not matches:
            raise ValueError(f"Trial id {args.trial_id} not found in log")
        trial = matches[0]
    else:
        trial = trials[-1]  # default: last

    params = trial["params"]
    settings = trial.get("rollout_settings", {})

    print("\n--- REPLAY REQUEST ---")
    print(f"log: {log_path}")
    print(f"trial_id: {trial.get('trial_id')}")
    print(f"logged_reward: {trial.get('reward')}")
    print(f"logged_termination: {trial.get('termination')}")
    print("\nparams:")
    print(json.dumps(params, indent=2))

    # Re-run rollout
    print("\n--- REPLAY RUN ---")
    res = rollout(
        params=params,
        sim_duration=float(settings.get("sim_duration", 5.0)),
        settle_time=float(settings.get("settle_time", 0.0)),
        log_stride=int(settings.get("log_stride", 10)),
        fixed_torque=float(settings.get("fixed_torque", 0.5)),
    )

    print("\n--- REPLAY RESULT ---")
    print(f"termination: {res.termination}")
    print("summary:")
    print(json.dumps(res.summary, indent=2))

    # Optional: launch viewer.py for visual sanity
    if args.launch_viewer:
        viewer_path = Path(__file__).parent / "viewer.py"
        if not viewer_path.exists():
            print(f"[WARN] viewer.py not found at {viewer_path}. Skipping.")
            return
        print("\n--- LAUNCHING VIEWER ---")
        subprocess.run([sys.executable, str(viewer_path)], check=False)


if __name__ == "__main__":
    main()
