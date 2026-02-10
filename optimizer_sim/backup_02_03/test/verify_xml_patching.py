"""
test_baseline_viewer.py

Rehauled baseline visualization test that follows *exactly* the same
visualization pathway as optimizer.py / replay.py:

    viewer.INSPECT_PARAMS = {...}
    viewer.SIM_DURATION   = ...
    viewer.FIXED_TORQUE   = ...
    viewer.main()

Key point:
- This script is safe to run from ANY folder (including a separate test folder).
- It temporarily chdirs into the project root before calling viewer.main(),
  matching optimizer.py's assumptions ("Run from project root").
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

# -----------------------------------------------------------------------------
# Path / cwd handling
# -----------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent  # assumes: tests/ is one level under project root
# If your folder structure differs, change PROJECT_ROOT accordingly.

@contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))


# Make sure imports behave like optimizer/replay when run from project root
sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# Baseline params (paste your current baseline here)
# -----------------------------------------------------------------------------

BASELINE_PARAMS = {
    "Br": 1.609306,
    "o_solref": [0.0003, 38.6676],
    "o_solimp": [0.9900, 0.9900, 0.0010, 0.5, 2.0],
    "wheel_friction": [0.9338, 0.0736, 0.0924],
    "rocker_stiffness": 51.874403,
    "rocker_damping": 1.845923,
    "wheel_kp": 1.212559,
    "wheel_kv": 1.541354,
    "max_magnetic_distance": 0.041651,
}

# Viewer runtime knobs (match optimizer defaults / config usage)
SIM_DURATION = 10.0
FIXED_TORQUE = 0.5


def run_view(mode: str, rollout_mode: str, sim_duration: float, fixed_torque: float):
    """
    Run viewer the same way optimizer.py / replay.py do.
    """
    import viewer  # imported inside so it sees PROJECT_ROOT cwd when we pushd

    # Compose INSPECT_PARAMS exactly like replay.py does:
    # full param dict + mode metadata. viewer.py will read from this global.
    params = dict(BASELINE_PARAMS)
    params["mode"] = mode
    params["rollout_mode"] = rollout_mode

    viewer.INSPECT_PARAMS = params
    viewer.SIM_DURATION = float(sim_duration)
    viewer.FIXED_TORQUE = float(fixed_torque)

    viewer.main()


def main():
    print("=" * 70)
    print("BASELINE VISUALIZATION (viewer.py pipeline)")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"SIM_DURATION: {SIM_DURATION}")
    print(f"FIXED_TORQUE: {FIXED_TORQUE}")
    print()

    # IMPORTANT: viewer.py + sim_sally stack historically assume "run from project root".
    # So we temporarily run viewer from PROJECT_ROOT regardless of where this test lives.
    with pushd(PROJECT_ROOT):
        input("Press ENTER to view BASELINE HOLD (sideways), or Ctrl+C to exit...")
        print("\n" + "=" * 60)
        print("BASELINE - HOLD MODE (sideways)")
        print("=" * 60)
        run_view(mode="sideways", rollout_mode="hold", sim_duration=SIM_DURATION, fixed_torque=FIXED_TORQUE)

        input("\nPress ENTER to view BASELINE DRIVE (drive_up)...")
        print("\n" + "=" * 60)
        print("BASELINE - DRIVE MODE (drive_up)")
        print("=" * 60)
        run_view(mode="drive_up", rollout_mode="drive", sim_duration=SIM_DURATION, fixed_torque=FIXED_TORQUE)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
