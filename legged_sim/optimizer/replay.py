"""
replay.py — Replay floor-lift, wall-hold, and pull-off scenes without running the optimizer.

Fill in PARAMS below, then run:
    python replay.py
    python replay.py --mode floor
    python replay.py --mode wall
    python replay.py --mode pulloff
    python replay.py --mode both       # floor + wall
    python replay.py --mode all        # floor + wall + pulloff
"""

import argparse
import importlib.util
import os
import sys

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_LEGGED_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))

for _p in (_LEGGED_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load optimizer/viewer.py explicitly by path so the wrong viewer.py
# from legged_sim/ is never picked up regardless of sys.path ordering.
def _load_viewer():
    spec   = importlib.util.spec_from_file_location(
        "optimizer_viewer",
        os.path.join(_THIS_DIR, "viewer.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ── Paste your params here ────────────────────────────────────────────────────
PARAMS = {
    'ground_friction':       [0.0245472, 0.00408506, 4.24188e-05],
    'solref':                [0.0181143, 10.0],
    'solimp':                [0.308749, 0.9999, 0.000736676, 0.332268, 6.30009],
    'noslip_iterations':     50,
    'noslip_tolerance':      2.53973e-05,
    'margin':                0.0101808,
    'Br':                    1.99127,
    'max_magnetic_distance': 0.118207,
    'max_force_per_wheel':   867.437,
}

# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", default="all",
                        choices=["floor", "wall", "pulloff", "both", "all"],
                        help="Which simulation(s) to replay "
                             "(both=floor+wall, all=floor+wall+pulloff, default: all)")
    args = parser.parse_args()

    if not PARAMS:
        sys.exit("ERROR: PARAMS is empty — fill it in before running.")

    viewer = _load_viewer()

    if args.mode in ("floor", "both", "all"):
        viewer.run_floor(PARAMS)

    if args.mode in ("wall", "both", "all"):
        if args.mode in ("both", "all"):
            print("\n[replay] Floor viewer closed — launching wall scene.")
        viewer.run_wall(PARAMS)

    if args.mode in ("pulloff", "all"):
        if args.mode == "all":
            print("\n[replay] Wall viewer closed — launching pull-off scene.")
        viewer.run_pulloff(PARAMS)

    print("[replay] Done.")


if __name__ == "__main__":
    main()