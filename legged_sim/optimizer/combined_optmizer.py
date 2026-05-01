"""
combined_optmizer.py — CMA-ES optimizer over floor-lift, wall-hold, and pull-off sims.

Combined cost = 0.4 * floor_lift_cost + 0.4 * wall_hold_cost + 0.2 * pulloff_cost

Floor cost (from combined_config.floor_calculate_cost):
  30% — mean XYZ drift norm of FR/BL/BR EE from settled baselines (m)
  30% — mean Z drop of FR/BL/BR below their settled baselines (m)
  40% — fraction of hold steps where any stance foot had zero mag force

Wall cost (from combined_config.wall_calculate_cost):
  30% — mean XYZ drift norm of FR/BL/BR EE from settled baselines (m)
  30% — FL electromagnet X penetration past its planted baseline (m)
  40% — fraction of hold steps where any stance foot had zero mag force

Pull-off cost (from combined_config.pulloff_calculate_cost):
  One-sided shortfall below params['max_force_per_wheel'] (N), normalised to [0, 1].
  Cost = 0 when pull-off force ≥ target; cost = 1 when force = 0.

Usage:
    python combined_optmizer.py
    python combined_optmizer.py --n-calls 500 --suffix my_run
    python combined_optmizer.py --resume-from results/20250101T120000_my_run
    python combined_optmizer.py --warm-start-from results/20250101T120000_my_run
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

from combined_config import (
    N_CALLS, BATCH_SIZE, CMAES_SIGMA0, OPTIMIZER_RANDOM_STATE,
    space, point_to_params,
    floor_calculate_cost,
    wall_calculate_cost,
    pulloff_calculate_cost,
)

# ── Settings ──────────────────────────────────────────────────────────────────

COST_WEIGHT_FLOOR   = 0.4
COST_WEIGHT_WALL    = 0.4
COST_WEIGHT_PULLOFF = 0.2


class OptResult(NamedTuple):
    fun: float
    x:   list[float]


# ── CSV schema ────────────────────────────────────────────────────────────────

def _csv_fieldnames() -> list[str]:
    base = [
        "id", "cost", "elapsed_min",
        # Floor fields
        "floor_norm_mm", "floor_neg_z_mm", "floor_zero_frac_pct", "floor_cost",
        # Wall fields
        "wall_norm_mm", "wall_into_x_mm", "wall_zero_frac_pct", "wall_cost",
        # Pull-off fields
        "pulloff_force_n", "pulloff_target_n", "pulloff_shortfall_n", "pulloff_cost",
    ]
    return base + [dim.name for dim in space]


def _best_csv_fieldnames() -> list[str]:
    return ["timestamp", "elapsed_min", "n_eval"] + _csv_fieldnames()


# ── Multiprocessing worker ────────────────────────────────────────────────────

def _evaluate_one_candidate(args):
    point_index, point = args

    import os, sys
    _dir = os.path.dirname(os.path.abspath(__file__))
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

    from sim_opt_sim      import run_headless_floor as floor_run
    from sim_wallopt_sim  import run_headless_wall   as wall_run
    from sim_pulloff_sim  import run_headless_lift   as pulloff_run
    from combined_config  import point_to_params, floor_calculate_cost, wall_calculate_cost, pulloff_calculate_cost

    params = point_to_params(point)
    t0     = time.perf_counter()

    import io, contextlib
    _suppress = contextlib.redirect_stdout(io.StringIO())

    try:
        with _suppress:
            f_norm, f_neg_z, f_zero_frac = floor_run(params)
    except Exception as e:
        print(f"  [WARN] floor sim crashed (point {point_index}): {e}", flush=True)
        f_norm = f_neg_z = 0.0
        f_zero_frac = 1.0

    try:
        with _suppress:
            w_norm, w_into_x, w_zero_frac = wall_run(params)
    except Exception as e:
        print(f"  [WARN] wall sim crashed (point {point_index}): {e}", flush=True)
        w_norm = w_into_x = 0.0
        w_zero_frac = 1.0

    try:
        with _suppress:
            pf = pulloff_run(params)
    except Exception as e:
        print(f"  [WARN] pulloff sim crashed (point {point_index}): {e}", flush=True)
        pf = 0.0

    fc = floor_calculate_cost(f_norm, f_neg_z, f_zero_frac)
    wc = wall_calculate_cost(w_norm, w_into_x, w_zero_frac)
    pc = pulloff_calculate_cost(pf, params['max_force_per_wheel'])

    total = (COST_WEIGHT_FLOOR   * fc["total_cost"]
           + COST_WEIGHT_WALL    * wc["total_cost"]
           + COST_WEIGHT_PULLOFF * pc["total_cost"])

    return point_index, {
        "total_cost":        total,
        # Floor
        "floor_norm":        f_norm,
        "floor_neg_z":       f_neg_z,
        "floor_zero_frac":   f_zero_frac,
        "floor_cost":        fc["total_cost"],
        # Wall
        "wall_norm":         w_norm,
        "wall_into_x":       w_into_x,
        "wall_zero_frac":    w_zero_frac,
        "wall_cost":         wc["total_cost"],
        # Pull-off
        "pulloff_force":     pf,
        "pulloff_target":    pc["pulloff_target"],
        "pulloff_shortfall": pc["shortfall_n"],
        "pulloff_cost":      pc["total_cost"],
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
        "id":              str(uuid.uuid4().hex)[:8],
        "cost":            cost_data["total_cost"],
        "params":          {dim.name: val for dim, val in zip(space, point)},
        "wall_time":       wall_time,
        # Floor
        "floor_norm":      cost_data["floor_norm"],
        "floor_neg_z":     cost_data["floor_neg_z"],
        "floor_zero_frac": cost_data["floor_zero_frac"],
        "floor_cost":      cost_data["floor_cost"],
        # Wall
        "wall_norm":       cost_data["wall_norm"],
        "wall_into_x":     cost_data["wall_into_x"],
        "wall_zero_frac":  cost_data["wall_zero_frac"],
        "wall_cost":       cost_data["wall_cost"],
        # Pull-off
        "pulloff_force":     cost_data["pulloff_force"],
        "pulloff_target":    cost_data["pulloff_target"],
        "pulloff_shortfall": cost_data["pulloff_shortfall"],
        "pulloff_cost":      cost_data["pulloff_cost"],
    }


def _append_csv(path, fields, res, elapsed_min, extra=None):
    row = {
        "id":            res["id"],
        "cost":          res["cost"],
        "elapsed_min":   f"{elapsed_min:.1f}",
        # Floor
        "floor_norm_mm":       res["floor_norm"] * 1000,
        "floor_neg_z_mm":      res["floor_neg_z"] * 1000,
        "floor_zero_frac_pct": res["floor_zero_frac"] * 100,
        "floor_cost":          res["floor_cost"],
        # Wall
        "wall_norm_mm":       res["wall_norm"] * 1000,
        "wall_into_x_mm":     res["wall_into_x"] * 1000,
        "wall_zero_frac_pct": res["wall_zero_frac"] * 100,
        "wall_cost":          res["wall_cost"],
        # Pull-off
        "pulloff_force_n":     res["pulloff_force"],
        "pulloff_target_n":    res["pulloff_target"],
        "pulloff_shortfall_n": res["pulloff_shortfall"],
        "pulloff_cost":        res["pulloff_cost"],
        **(extra or {}),
    }
    row.update(res["params"])
    try:
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
    except Exception as e:
        print(f"  [Warning] Could not append to CSV: {e}")


# ── Printing ──────────────────────────────────────────────────────────────────

_best_cost_so_far = float("inf")


def _print_batch(results):
    for i, r in enumerate(results):
        print(
            f"    [{i+1}/{len(results)}] id={r['id']}  cost={r['cost']:.6f}  "
            f"floor(norm={r['floor_norm']*1000:.2f}mm  neg_z={r['floor_neg_z']*1000:.2f}mm  zero={r['floor_zero_frac']*100:.1f}%)  "
            f"wall(norm={r['wall_norm']*1000:.2f}mm  +x={r['wall_into_x']*1000:.2f}mm  zero={r['wall_zero_frac']*100:.1f}%)  "
            f"pulloff={r['pulloff_force']:.1f}N  "
            f"t={r['wall_time']:.1f}s"
        )


def _print_best(all_results, n_done, elapsed_min, best_csv_path):
    global _best_cost_so_far
    best   = min(all_results, key=lambda r: r["cost"])
    is_new = best["cost"] < _best_cost_so_far
    marker = " ★ NEW BEST" if is_new else ""
    print(
        f"  Best (n={n_done}): cost={best['cost']:.6f}{marker}\n"
        f"    floor: norm={best['floor_norm']*1000:.2f}mm  neg_z={best['floor_neg_z']*1000:.2f}mm  zero={best['floor_zero_frac']*100:.1f}%  cost={best['floor_cost']:.4f}\n"
        f"    wall:  norm={best['wall_norm']*1000:.2f}mm  +x={best['wall_into_x']*1000:.2f}mm  zero={best['wall_zero_frac']*100:.1f}%  cost={best['wall_cost']:.4f}\n"
        f"    pulloff: force={best['pulloff_force']:.1f}N  target={best['pulloff_target']:.1f}N  "
        f"shortfall={best['pulloff_shortfall']:.1f}N  cost={best['pulloff_cost']:.4f}"
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

    if es_resume is not None:
        print(f"  Backend: CMA-ES RESUMED (sigma={es.sigma:.4g}, popsize={BATCH_SIZE})")
    else:
        warm = "warm-start" if x0_override is not None else "cold-start"
        print(f"  Backend: CMA-ES (sigma0={CMAES_SIGMA0}, popsize={BATCH_SIZE}, {warm})")

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
        if not p.exists():
            sys.exit(f"ERROR: resume state not found: {p}")
        with open(p, "rb") as _f:
            state = pickle.load(_f)
        es_resume = state["es"]
        print(f"Resuming from {p} (sigma={es_resume.sigma:.4g}, prev evals={state['n_done']})")

    if args.warm_start_from:
        p    = pathlib.Path(args.warm_start_from)
        if p.is_dir(): p = p / "optimization_bests.csv"
        if not p.exists():
            sys.exit(f"ERROR: warm-start file not found: {p}")
        with open(p) as _f:
            rows = list(csv.DictReader(_f))
        if not rows:
            sys.exit(f"ERROR: warm-start file is empty: {p}")
        x0_override = {dim.name: float(rows[-1][dim.name]) for dim in space}
        print(f"Warm-starting from {p} (cost={rows[-1]['cost']})")

    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S") + (f"_{args.suffix}" if args.suffix else "")
    run_dir = pathlib.Path("results") / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    for cfg in ("combined_config.py", "sim_opt_sim.py", "sim_wallopt_sim.py", "sim_pulloff_sim.py"):
        shutil.copy2(pathlib.Path(__file__).parent / cfg, run_dir / cfg)

    csv_path      = str(run_dir / "optimization_results.csv")
    best_csv_path = str(run_dir / "optimization_bests.csv")
    for path, fields in [(csv_path, _csv_fieldnames()),
                         (best_csv_path, _best_csv_fieldnames())]:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"\nCombined optimizer: {n_calls} evals, batch={BATCH_SIZE}")
    print(f"Floor lift weight: {COST_WEIGHT_FLOOR}  |  Wall hold weight: {COST_WEIGHT_WALL}  |  Pull-off weight: {COST_WEIGHT_PULLOFF}")
    print(f"Floor cost: 30% FR/BL/BR drift norm + 30% floor Z sag + 40% zero-contact frac (FLOOR_Z=0.0)")
    print(f"Wall cost:  30% FR/BL/BR drift norm + 30% FL wall penetration + 40% zero-contact frac (WALL_X=0.5)")
    print(f"Pull-off cost: one-sided shortfall below params['max_force_per_wheel'], normalised to [0, 1]")
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

    print("\nLaunching viewer — floor-lift first, then wall-hold.")
    print("Close each MuJoCo window to advance; close the force-plot window to continue.")
    subprocess.run(
        [sys.executable, os.path.join(_THIS_DIR, "viewer.py"), "--params", params_json, "--mode", "all"],
        check=False,
    )