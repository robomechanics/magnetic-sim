"""
sim_combined_optimizer.py — CMA-ES optimizer over both floor-lift and wall-hold sims.

Combined cost = 0.5 * floor_lift_cost + 0.5 * wall_hold_cost

Usage:
    python sim_combined_optimizer.py
    python sim_combined_optimizer.py --n-calls 500 --suffix my_run
    python sim_combined_optimizer.py --resume-from results/20250101T120000_my_run
    python sim_combined_optimizer.py --warm-start-from results/20250101T120000_my_run
"""

import argparse
import csv
import json
import multiprocessing
import os
import pathlib
import pickle
import shutil
import subprocess
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
    N_CALLS, BATCH_SIZE, CMAES_SIGMA0, OPTIMIZER_RANDOM_STATE,
    space, point_to_params,
    calculate_cost as floor_calculate_cost,
)
from sim_vertopt_config import (
    calculate_cost as wall_calculate_cost,
)

# ── Settings ──────────────────────────────────────────────────────────────────

COST_WEIGHT_FLOOR = 0.5
COST_WEIGHT_WALL  = 0.5
COST_FAILURE      = 9999.0


class OptResult(NamedTuple):
    fun: float
    x:   list[float]


# ── CSV schema ────────────────────────────────────────────────────────────────

def _csv_fieldnames() -> list[str]:
    base = [
        "id", "cost", "elapsed_min",
        "floor_x_mm", "floor_y_mm", "floor_z_mm", "floor_cost",
        "wall_x_mm",  "wall_y_mm",  "wall_z_mm",  "wall_cost",
    ]
    return base + [dim.name for dim in space]


def _best_csv_fieldnames() -> list[str]:
    return ["timestamp", "elapsed_min", "n_eval"] + _csv_fieldnames()


# ── Multiprocessing worker ────────────────────────────────────────────────────

def _evaluate_one_candidate(args):
    point_index, point = args

    import os, sys
    _dir = os.path.dirname(os.path.abspath(__file__))
    _legged = os.path.abspath(os.path.join(_dir, ".."))
    for p in (_legged, _dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    from sim_opt_sim      import run_headless_lift as floor_run
    from sim_vertopt_sim  import run_headless_lift as wall_run
    from sim_opt_config   import point_to_params, calculate_cost as floor_cost_fn
    from sim_vertopt_config import calculate_cost as wall_cost_fn

    params = point_to_params(point)
    t0     = time.perf_counter()

    try:
        fx, fy, fz = floor_run(params)
    except Exception as e:
        print(f"  [WARN] floor sim crashed (point {point_index}): {e}", flush=True)
        fx = fy = fz = 0.0

    try:
        wx, wy, wz = wall_run(params)
    except Exception as e:
        print(f"  [WARN] wall sim crashed (point {point_index}): {e}", flush=True)
        wx = wy = wz = 0.0

    fc = floor_cost_fn(fx, fy, fz)
    wc = wall_cost_fn(wx, wy, wz)

    total = COST_WEIGHT_FLOOR * fc["total_cost"] + COST_WEIGHT_WALL * wc["total_cost"]

    return point_index, {
        "total_cost": total,
        "floor_x":    fx, "floor_y": fy, "floor_z": fz,
        "floor_cost": fc["total_cost"],
        "wall_x":     wx, "wall_y":  wy, "wall_z":  wz,
        "wall_cost":  wc["total_cost"],
    }, time.perf_counter() - t0


# ── CMA-ES ────────────────────────────────────────────────────────────────────

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


# ── Result helpers ────────────────────────────────────────────────────────────

def _build_result(point_index, point, cost_data, wall_time):
    return {
        "id":         str(uuid.uuid4().hex)[:8],
        "cost":       cost_data["total_cost"],
        "params":     {dim.name: val for dim, val in zip(space, point)},
        "wall_time":  wall_time,
        "floor_x":    cost_data["floor_x"],
        "floor_y":    cost_data["floor_y"],
        "floor_z":    cost_data["floor_z"],
        "floor_cost": cost_data["floor_cost"],
        "wall_x":     cost_data["wall_x"],
        "wall_y":     cost_data["wall_y"],
        "wall_z":     cost_data["wall_z"],
        "wall_cost":  cost_data["wall_cost"],
    }


def _append_csv(path, fields, res, elapsed_min, extra=None):
    row = {
        "id":          res["id"],
        "cost":        res["cost"],
        "elapsed_min": f"{elapsed_min:.1f}",
        "floor_x_mm":  res["floor_x"] * 1000,
        "floor_y_mm":  res["floor_y"] * 1000,
        "floor_z_mm":  res["floor_z"] * 1000,
        "floor_cost":  res["floor_cost"],
        "wall_x_mm":   res["wall_x"] * 1000,
        "wall_y_mm":   res["wall_y"] * 1000,
        "wall_z_mm":   res["wall_z"] * 1000,
        "wall_cost":   res["wall_cost"],
        **(extra or {}),
    }
    row.update(res["params"])
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


# ── Printing ──────────────────────────────────────────────────────────────────

_best_cost_so_far = float("inf")


def _print_batch(results):
    for i, r in enumerate(results):
        print(
            f"    [{i+1}/{len(results)}] id={r['id']}  cost={r['cost']:.6f}  "
            f"floor(x={r['floor_x']*1000:+.2f} y={r['floor_y']*1000:+.2f} z={r['floor_z']*1000:+.2f})mm  "
            f"wall(x={r['wall_x']*1000:+.2f} y={r['wall_y']*1000:+.2f} z={r['wall_z']*1000:+.2f})mm  "
            f"t={r['wall_time']:.1f}s"
        )


def _print_best(all_results, n_done, elapsed_min, best_csv_path):
    global _best_cost_so_far
    best   = min(all_results, key=lambda r: r["cost"])
    is_new = best["cost"] < _best_cost_so_far
    marker = " ★ NEW BEST" if is_new else ""
    print(
        f"  Best (n={n_done}): cost={best['cost']:.6f}{marker}\n"
        f"    floor: x={best['floor_x']*1000:.2f}mm  y={best['floor_y']*1000:.2f}mm  z={best['floor_z']*1000:.2f}mm  cost={best['floor_cost']:.4f}\n"
        f"    wall:  x={best['wall_x']*1000:.2f}mm   y={best['wall_y']*1000:.2f}mm   z={best['wall_z']*1000:.2f}mm   cost={best['wall_cost']:.4f}"
    )
    if is_new:
        _best_cost_so_far = best["cost"]
        _append_csv(best_csv_path, _best_csv_fieldnames(), best, elapsed_min,
                    extra={"timestamp": datetime.now().isoformat(timespec="seconds"),
                           "elapsed_min": f"{elapsed_min:.1f}",
                           "n_eval": n_done})


# ── Main loop ─────────────────────────────────────────────────────────────────

def _run_optimization(all_results, pool, run_dir, csv_path, best_csv_path,
                      n_calls, es_resume=None, x0_override=None):
    ask, tell, es = _create_cmaes_optimizer(x0_override=x0_override, es_override=es_resume)

    warm = "RESUMED" if es_resume else ("warm-start" if x0_override else "cold-start")
    print(f"  Backend: CMA-ES ({warm}, sigma={es.sigma:.4g}, popsize={BATCH_SIZE})")

    n_done    = 0
    batch_num = 0
    t_start   = time.perf_counter()

    while n_done < n_calls:
        batch_num += 1
        t_batch    = time.perf_counter()
        points     = ask()
        n_this     = len(points)
        print(f"\n--- Batch {batch_num}: {n_this} points ({n_done+1}–{n_done+n_this} / {n_calls}) ---")

        tasks      = [(i, pt) for i, pt in enumerate(points)]
        raw        = sorted(pool.imap_unordered(_evaluate_one_candidate, tasks, chunksize=1),
                            key=lambda x: x[0])
        results    = [_build_result(idx, points[idx], cd, wt) for idx, cd, wt in raw]
        costs      = [r["cost"] for r in results]
        tell(points, costs)

        elapsed_min = (time.perf_counter() - t_start) / 60.0
        for r in results:
            all_results.append(r)
            _append_csv(csv_path, _csv_fieldnames(), r, elapsed_min)

        n_done += n_this

        _print_batch(results)
        print(f"  Batch wall: {time.perf_counter()-t_batch:.1f}s | "
              f"Elapsed: {elapsed_min:.1f}min | "
              f"Costs: min={min(costs):.4f}, max={max(costs):.4f}")
        _print_best(all_results, n_done, elapsed_min, best_csv_path)

        with open(run_dir / "cmaes_state.pkl", "wb") as f:
            pickle.dump({"es": es, "n_done": n_done}, f)

    best = min(all_results, key=lambda r: r["cost"])
    return OptResult(fun=best["cost"],
                     x=[best["params"][dim.name] for dim in space])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMA-ES combined floor-lift + wall-hold optimizer")
    parser.add_argument("--suffix",          "-s", type=str, default="")
    parser.add_argument("--n-calls",         type=int,       default=None)
    parser.add_argument("--warm-start-from", type=str,       default=None)
    parser.add_argument("--resume-from",     type=str,       default=None)
    args = parser.parse_args()

    if args.resume_from and args.warm_start_from:
        sys.exit("ERROR: --resume-from and --warm-start-from are mutually exclusive")

    n_calls     = args.n_calls or N_CALLS
    es_resume   = None
    x0_override = None

    if args.resume_from:
        p = pathlib.Path(args.resume_from)
        if p.is_dir(): p = p / "cmaes_state.pkl"
        state     = pickle.load(open(p, "rb"))
        es_resume = state["es"]
        print(f"Resuming (sigma={es_resume.sigma:.4g}, prev evals={state['n_done']})")

    if args.warm_start_from:
        p    = pathlib.Path(args.warm_start_from)
        if p.is_dir(): p = p / "optimization_bests.csv"
        rows = list(csv.DictReader(open(p)))
        x0_override = {dim.name: float(rows[-1][dim.name]) for dim in space}
        print(f"Warm-starting from {p} (cost={rows[-1]['cost']})")

    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S") + (f"_{args.suffix}" if args.suffix else "")
    run_dir = pathlib.Path("results") / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    for cfg in ("sim_opt_config.py", "sim_vertopt_config.py"):
        shutil.copy2(pathlib.Path(__file__).parent / cfg, run_dir / cfg)

    csv_path      = str(run_dir / "optimization_results.csv")
    best_csv_path = str(run_dir / "optimization_bests.csv")
    for path, fields in [(csv_path, _csv_fieldnames()),
                         (best_csv_path, _best_csv_fieldnames())]:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"\nCombined optimizer: {n_calls} evals, batch={BATCH_SIZE}")
    print(f"Floor lift weight: {COST_WEIGHT_FLOOR}  |  Wall hold weight: {COST_WEIGHT_WALL}")
    print(f"Run directory: {run_dir}/")

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
            n_calls, es_resume=es_resume, x0_override=x0_override,
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

    # Save best params to JSON for viewer subprocesses
    params_json = str(run_dir / "best_params.json")
    with open(params_json, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {params_json}")

    # Launch floor viewer, then wall viewer sequentially (close one → next opens)
    print("\nLaunching floor-lift viewer — close the window to continue to wall viewer.")
    subprocess.run(
        [sys.executable, os.path.join(_THIS_DIR, "sim_opt_sim.py"), "--view", params_json],
        check=False,
    )

    print("Launching wall-hold viewer — close the window to exit.")
    subprocess.run(
        [sys.executable, os.path.join(_THIS_DIR, "sim_vertopt_sim.py"), "--view", params_json],
        check=False,
    )