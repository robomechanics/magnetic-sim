"""
sim_vert_optimizer.py — CMA-ES optimizer for the vertical wall-hold sim.

Tunes 14 physics parameters to minimize EE body XYZ drift while the robot
clings to the vertical wall with all magnets ON and all joints held.

Cost (lower = better):
  50% — mean absolute Z drift (gravity-direction slip; most critical)
  25% — mean absolute X drift (pull-off from wall)
  25% — mean absolute Y drift (lateral slip)

Usage:
    python sim_vert_optimizer.py
    python sim_vert_optimizer.py --n-calls 500 --suffix my_run
    python sim_vert_optimizer.py --resume-from results/20250101T120000_my_run
    python sim_vert_optimizer.py --warm-start-from results/20250101T120000_my_run
"""

import argparse
import csv
import multiprocessing
import os
import pathlib
import pickle
import shutil
import sys
import time
import uuid
from datetime import datetime
from typing import NamedTuple

import numpy as np

# Ensure this directory is on sys.path for both main process and workers
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sim_opt_config import (
    N_CALLS,
    BATCH_SIZE,
    CMAES_SIGMA0,
    OPTIMIZER_RANDOM_STATE,
    space,
    point_to_params,
    calculate_cost,
)


# ── Cost failure sentinel ─────────────────────────────────────────────────────

COST_FAILURE = 9999.0


class OptResult(NamedTuple):
    fun: float
    x:   list[float]


# ── CSV schema ────────────────────────────────────────────────────────────────

def _csv_fieldnames() -> list[str]:
    base = [
        "id", "cost", "elapsed_min",
        "x_drift_mm", "y_drift_mm", "z_drift_mm",
        "x_cost", "y_cost", "z_cost",
    ]
    return base + [dim.name for dim in space]


def _best_csv_fieldnames() -> list[str]:
    return ["timestamp", "elapsed_min", "n_eval"] + _csv_fieldnames()


# ── Multiprocessing worker ────────────────────────────────────────────────────

def _evaluate_one_candidate(args):
    """Run a single wall-hold sim trial for one candidate parameter point."""
    point_index, point = args

    import os, sys
    _optimizer_dir = os.path.dirname(os.path.abspath(__file__))
    _legged_dir    = os.path.abspath(os.path.join(_optimizer_dir, ".."))
    if _legged_dir not in sys.path:
        sys.path.insert(0, _legged_dir)
    if _optimizer_dir not in sys.path:
        sys.path.insert(0, _optimizer_dir)

    from sim_vertopt_sim import run_headless_lift
    from sim_opt_config import point_to_params, calculate_cost

    params = point_to_params(point)

    t0 = time.perf_counter()
    try:
        mean_abs_x, mean_abs_y, mean_abs_z = run_headless_lift(params)
    except Exception as e:
        print(f"  [WARN] Sim crashed (point {point_index}): {e}", flush=True)
        mean_abs_x = mean_abs_y = mean_abs_z = 0.0

    wall_time = time.perf_counter() - t0
    cost_data = calculate_cost(mean_abs_x, mean_abs_y, mean_abs_z)
    return point_index, cost_data, wall_time


# ── CMA-ES space mapping ──────────────────────────────────────────────────────

def _cmaes_space_info():
    x0, lower, upper, is_log = [], [], [], []
    for dim in space:
        lo, hi = dim.low, dim.high
        if dim.prior == "log-uniform":
            is_log.append(True)
            lower.append(np.log10(lo))
            upper.append(np.log10(hi))
            x0.append(0.5 * (np.log10(lo) + np.log10(hi)))
        else:
            is_log.append(False)
            lower.append(lo)
            upper.append(hi)
            x0.append(0.5 * (lo + hi))
    return x0, lower, upper, is_log


def _cmaes_to_real(x_internal, is_log):
    return [10.0 ** v if log else v for v, log in zip(x_internal, is_log)]


def _create_cmaes_optimizer(x0_override=None, es_override=None):
    import cma

    x0_raw, lower, upper, is_log = _cmaes_space_info()

    if es_override is not None:
        es = es_override
    else:
        if x0_override is not None:
            x0_raw = []
            for dim, log in zip(space, is_log):
                v = x0_override[dim.name]
                x0_raw.append(np.log10(v) if log else v)
        opts = {
            "bounds":  [lower, upper],
            "seed":    OPTIMIZER_RANDOM_STATE,
            "popsize": BATCH_SIZE,
            "verbose": -1,
            "tolfun":  1e-8,
            "tolx":    1e-10,
        }
        es = cma.CMAEvolutionStrategy(x0_raw, CMAES_SIGMA0, opts)

    def ask():
        internal_points = es.ask()
        ask._last_internal = internal_points
        return [_cmaes_to_real(p, is_log) for p in internal_points]

    ask._last_internal = []

    def tell(points, costs):
        es.tell(ask._last_internal, costs)

    return ask, tell, es


# ── Result aggregation ────────────────────────────────────────────────────────

def _build_result(point_index: int, point: list, cost_data: dict, wall_time: float) -> dict:
    return {
        "id":         str(uuid.uuid4().hex)[:8],
        "cost":       cost_data["total_cost"],
        "params":     {dim.name: val for dim, val in zip(space, point)},
        "x_drift_mm": cost_data["x_drift"] * 1000,
        "y_drift_mm": cost_data["y_drift"] * 1000,
        "z_drift_mm": cost_data["z_drift"] * 1000,
        "x_cost":     cost_data["x_cost"],
        "y_cost":     cost_data["y_cost"],
        "z_cost":     cost_data["z_cost"],
        "wall_time":  wall_time,
    }


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def _append_result_to_csv(csv_path: str, res: dict, elapsed_min: float) -> None:
    row = {
        "id":          res["id"],
        "cost":        res["cost"],
        "elapsed_min": f"{elapsed_min:.1f}",
        "x_drift_mm":  res["x_drift_mm"],
        "y_drift_mm":  res["y_drift_mm"],
        "z_drift_mm":  res["z_drift_mm"],
        "x_cost":      res["x_cost"],
        "y_cost":      res["y_cost"],
        "z_cost":      res["z_cost"],
    }
    row.update(res["params"])
    try:
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_csv_fieldnames()).writerow(row)
    except Exception as e:
        print(f"  [Warning] Could not append to CSV: {e}")


def _append_best_csv(best_csv_path: str, best: dict, n_done: int, elapsed_min: float) -> None:
    row = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "elapsed_min": f"{elapsed_min:.1f}",
        "n_eval":      n_done,
        "id":          best["id"],
        "cost":        best["cost"],
        "x_drift_mm":  best["x_drift_mm"],
        "y_drift_mm":  best["y_drift_mm"],
        "z_drift_mm":  best["z_drift_mm"],
        "x_cost":      best["x_cost"],
        "y_cost":      best["y_cost"],
        "z_cost":      best["z_cost"],
    }
    row.update(best["params"])
    try:
        with open(best_csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_best_csv_fieldnames()).writerow(row)
    except Exception as e:
        print(f"  [Warning] Could not append to best CSV: {e}")


# ── Printing ──────────────────────────────────────────────────────────────────

_best_cost_so_far: float = float("inf")


def _print_batch_results(results: list[dict]) -> None:
    for i, r in enumerate(results):
        print(
            f"    [{i+1}/{len(results)}] id={r['id']}  cost={r['cost']:.6f}  "
            f"drift: x={r['x_drift_mm']:+.2f}mm  y={r['y_drift_mm']:+.2f}mm  "
            f"z={r['z_drift_mm']:+.2f}mm  wall={r['wall_time']:.1f}s"
        )


def _print_best_so_far(
    all_results: list[dict], n_done: int, elapsed_min: float,
    best_csv_path: str,
) -> None:
    global _best_cost_so_far
    best   = min(all_results, key=lambda r: r["cost"])
    is_new = best["cost"] < _best_cost_so_far
    marker = " ★ NEW BEST" if is_new else ""
    print(
        f"  Best so far (n={n_done}): cost={best['cost']:.6f}  id={best['id']}{marker}\n"
        f"    drift: x={best['x_drift_mm']:.2f}mm  y={best['y_drift_mm']:.2f}mm  "
        f"z={best['z_drift_mm']:.2f}mm"
    )
    if is_new:
        _best_cost_so_far = best["cost"]
        _append_best_csv(best_csv_path, best, n_done, elapsed_min)


# ── Main optimization loop ────────────────────────────────────────────────────

def _run_optimization(
    all_results: list[dict],
    pool: multiprocessing.Pool,
    run_dir: pathlib.Path,
    csv_path: str,
    best_csv_path: str,
    n_calls: int,
    es_resume=None,
    x0_override=None,
) -> OptResult:
    ask, tell, es = _create_cmaes_optimizer(x0_override=x0_override, es_override=es_resume)

    if es_resume is not None:
        print(f"  Backend: CMA-ES RESUMED (sigma={es.sigma:.4g}, popsize={BATCH_SIZE})")
    else:
        warm = "warm-start" if x0_override is not None else "cold-start"
        print(f"  Backend: CMA-ES (sigma0={CMAES_SIGMA0}, popsize={BATCH_SIZE}, {warm})")

    n_done      = 0
    batch_num   = 0
    t_run_start = time.perf_counter()

    while n_done < n_calls:
        batch_num += 1
        t_batch   = time.perf_counter()
        points    = ask()
        n_this    = len(points)
        print(f"\n--- Batch {batch_num}: {n_this} points ({n_done+1}–{n_done+n_this} / {n_calls}) ---")

        tasks = [(i, point) for i, point in enumerate(points)]

        raw_results = list(pool.imap_unordered(_evaluate_one_candidate, tasks, chunksize=1))
        raw_results.sort(key=lambda x: x[0])

        results = []
        for point_index, cost_data, wall_time in raw_results:
            r = _build_result(point_index, points[point_index], cost_data, wall_time)
            results.append(r)

        costs = [r["cost"] for r in results]
        tell(points, costs)

        elapsed_min = (time.perf_counter() - t_run_start) / 60.0
        for r in results:
            all_results.append(r)
            _append_result_to_csv(csv_path, r, elapsed_min)

        n_done += n_this

        _print_batch_results(results)
        batch_wall = time.perf_counter() - t_batch
        print(
            f"  Batch wall: {batch_wall:.1f}s | Elapsed: {elapsed_min:.1f}min | "
            f"Costs: min={min(costs):.4f}, max={max(costs):.4f}"
        )
        _print_best_so_far(all_results, n_done, elapsed_min, best_csv_path)

        state_path = run_dir / "cmaes_state.pkl"
        with open(state_path, "wb") as f:
            pickle.dump({"es": es, "n_done": n_done}, f)

    best = min(all_results, key=lambda r: r["cost"])
    return OptResult(
        fun=best["cost"],
        x=[best["params"][dim.name] for dim in space],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMA-ES vertical wall-hold parameter optimization")
    parser.add_argument("--suffix",          "-s", type=str,  default="",   help="Suffix for results folder")
    parser.add_argument("--n-calls",         type=int,        default=None, help=f"Override N_CALLS (default: {N_CALLS})")
    parser.add_argument("--warm-start-from", type=str,        default=None, help="Results dir to warm-start from")
    parser.add_argument("--resume-from",     type=str,        default=None, help="Results dir with cmaes_state.pkl")
    args = parser.parse_args()

    if args.resume_from and args.warm_start_from:
        print("ERROR: --resume-from and --warm-start-from are mutually exclusive")
        sys.exit(1)

    n_calls     = args.n_calls if args.n_calls is not None else N_CALLS
    es_resume   = None
    x0_override = None

    if args.resume_from:
        resume_path = pathlib.Path(args.resume_from)
        if resume_path.is_dir():
            resume_path = resume_path / "cmaes_state.pkl"
        if not resume_path.exists():
            print(f"ERROR: resume state not found: {resume_path}")
            sys.exit(1)
        with open(resume_path, "rb") as f:
            state = pickle.load(f)
        es_resume = state["es"]
        print(f"Resuming from {resume_path} (sigma={es_resume.sigma:.4g}, prev evals={state['n_done']})")

    if args.warm_start_from:
        ws_path = pathlib.Path(args.warm_start_from)
        if ws_path.is_dir():
            ws_path = ws_path / "optimization_bests.csv"
        if not ws_path.exists():
            print(f"ERROR: warm-start file not found: {ws_path}")
            sys.exit(1)
        with open(ws_path) as f:
            ws_rows = list(csv.DictReader(f))
        if not ws_rows:
            print(f"ERROR: warm-start file is empty: {ws_path}")
            sys.exit(1)
        ws_last     = ws_rows[-1]
        x0_override = {dim.name: float(ws_last[dim.name]) for dim in space}
        print(f"Warm-starting from {ws_path} (cost={ws_last['cost']})")

    run_tag  = datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.suffix:
        run_tag += f"_{args.suffix}"
    run_dir  = pathlib.Path("results") / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pathlib.Path(__file__).parent / "sim_opt_config.py", run_dir / "sim_opt_config.py")

    csv_path      = str(run_dir / "optimization_results.csv")
    best_csv_path = str(run_dir / "optimization_bests.csv")

    print(f"\nSim vert optimizer: {n_calls} evals, batch={BATCH_SIZE}")
    print(f"Cost: 50% Z-drift (gravity slip) + 25% X-drift (pull-off) + 25% Y-drift (lateral)")
    print(f"Run directory: {run_dir}/")

    with open(csv_path,      "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_csv_fieldnames()).writeheader()
    with open(best_csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_best_csv_fieldnames()).writeheader()

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    all_results = []
    pool        = None

    try:
        pool_size = max(1, min(os.cpu_count() or 16, BATCH_SIZE))
        print(f"Worker pool size: {pool_size}")
        pool   = multiprocessing.Pool(processes=pool_size)
        result = _run_optimization(
            all_results, pool, run_dir,
            csv_path, best_csv_path,
            n_calls,
            es_resume=es_resume,
            x0_override=x0_override,
        )
    finally:
        if pool:
            pool.terminate()
            pool.join()

    print(f"\n--- Optimization Finished ---")
    print(f"Lowest cost: {result.fun:.6f}")

    best_params = point_to_params(result.x)
    p = best_params['solimp']
    print("\nPARAMS = {")
    print(f"    'ground_friction':       [{best_params['ground_friction'][0]:.6g}, {best_params['ground_friction'][1]:.6g}, {best_params['ground_friction'][2]:.6g}],")
    print(f"    'solref':                [{best_params['solref'][0]:.6g}, {best_params['solref'][1]}],")
    print(f"    'solimp':                [{p[0]:.6g}, {p[1]}, {p[2]:.6g}, {p[3]:.6g}, {p[4]:.6g}],")
    print(f"    'noslip_iterations':     {best_params['noslip_iterations']},")
    print(f"    'noslip_tolerance':      {best_params['noslip_tolerance']:.6g},")
    print(f"    'margin':                {best_params['margin']:.6g},")
    print(f"    'Br':                    {best_params['Br']:.6g},")
    print(f"    'max_magnetic_distance': {best_params['max_magnetic_distance']:.6g},")
    print(f"    'max_force_per_wheel':   {best_params['max_force_per_wheel']:.6g},")
    print("}")

    # Write best params to JSON then launch sim_vertopt_sim.py as a fresh
    # subprocess — avoids GL/display conflicts from the multiprocessing pool.
    import json, subprocess
    params_json = str(run_dir / "best_params.json")
    with open(params_json, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {params_json}")
    print("Launching viewer — close the window to exit.")
    subprocess.run(
        [sys.executable, os.path.join(_THIS_DIR, "sim_vertopt_sim.py"),
         "--view", params_json],
        check=False,
    )