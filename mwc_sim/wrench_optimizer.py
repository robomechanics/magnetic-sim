"""
wrench_optimizer.py — CMA-ES optimizer for the magnetic wrench/peel simulation.

Tunes 14 physics parameters to match a target detach force or wrench moment,
while penalizing XY drift of the magnet before detachment.

Cost (lower = better):
  70% — shortfall from GOAL_FORCE or GOAL_WRENCH (early detachment penalty)
  30% — normalized XY drift of magnet COM during ramp phase (sliding penalty)

Usage:
    python wrench_optimizer.py
    python wrench_optimizer.py --n-calls 500 --suffix my_run
    python wrench_optimizer.py --resume-from results/20250101T120000_my_run
    python wrench_optimizer.py --warm-start-from results/20250101T120000_my_run
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

from wrench_config import (
    N_CALLS,
    BATCH_SIZE,
    CMAES_SIGMA0,
    OPTIMIZER_RANDOM_STATE,
    PULL_RATE_OPT,
    GOAL_FORCE,
    GOAL_WRENCH,
    APPLY_FORCE,
    APPLY_MOMENT,
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
        "detach_force", "detach_moment", "xy_drift",
        "detach_cost", "drift_cost",
        "achieved", "goal", "shortfall",
    ]
    param_names = [dim.name for dim in space]
    return base + param_names


def _best_csv_fieldnames() -> list[str]:
    return ["timestamp", "elapsed_min", "n_eval"] + _csv_fieldnames()


# ── Multiprocessing worker ────────────────────────────────────────────────────

def _evaluate_one_candidate(args):
    """Run a single wrench sim trial for one candidate parameter point.

    Imported inside the worker to survive multiprocessing spawn.

    Args:
        args: tuple of (point_index, point, pull_rate)

    Returns:
        tuple of (point_index, cost_data dict, wall_time)
    """
    point_index, point, pull_rate = args

    # ── TODO: if you add multi-trial jitter (e.g. varying pull_rate or spawn position),
    #    loop here and aggregate across trials before returning. ──────────────────────

    from wrench_sim import run_headless
    from wrench_config import point_to_params, calculate_cost

    params = point_to_params(point)

    # DEBUG
    # print(f"  [DBG {point_index}] Br={params['Br']:.4f}  max_dist={params['max_magnetic_distance']:.5f}  friction={params['ground_friction'][0]:.4f}", flush=True)
    
    t0 = time.perf_counter()
    try:
        records, detach_force, detach_moment, xy_drift = run_headless(
            pull_rate=pull_rate,
            params=params,
        )
    except Exception as e:
        print(f"  [WARN] Sim crashed (point {point_index}): {e}", flush=True)
        detach_force  = 0.0
        detach_moment = 0.0
        xy_drift      = 0.0

    wall_time = time.perf_counter() - t0
    cost_data = calculate_cost(detach_force, detach_moment, xy_drift)
    cost_data["detach_force"]  = detach_force
    cost_data["detach_moment"] = detach_moment

    return point_index, cost_data, wall_time


# ── CMA-ES space mapping ──────────────────────────────────────────────────────

def _cmaes_space_info():
    """Return x0, lower, upper, is_log lists for internal CMA-ES space."""
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
    """Build CMA-ES ask/tell interface.

    Args:
        x0_override: dict of {dim.name: value} for warm-starting.
        es_override: existing CMAEvolutionStrategy object for resuming.

    Returns:
        ask() -> list of real-valued points (always full BATCH_SIZE population)
        tell(points, costs) -> None
        es: the CMAEvolutionStrategy object (for checkpointing)
    """
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

    # FIX: always ask for the full population — never slice.
    # CMA-ES requires tell() to receive the same count as ask() returned,
    # otherwise the covariance matrix update is silently corrupted.
    def ask():
        internal_points = es.ask()          # full BATCH_SIZE population
        ask._last_internal = internal_points
        return [_cmaes_to_real(p, is_log) for p in internal_points]

    ask._last_internal = []  # initialize before first call

    def tell(points, costs):
        # use the original internal points, not re-encoded real-space points
        es.tell(ask._last_internal, costs)

    return ask, tell, es


# ── Result aggregation ────────────────────────────────────────────────────────

def _build_result(point_index: int, point: list, cost_data: dict, wall_time: float) -> dict:
    """Package a single evaluated candidate into a result dict."""
    params = point_to_params(point)
    return {
        "id":            str(uuid.uuid4().hex)[:8],
        "cost":          cost_data["total_cost"],
        "params":        {dim.name: val for dim, val in zip(space, point)},
        "detach_force":  cost_data["detach_force"],
        "detach_moment": cost_data["detach_moment"],
        "xy_drift":      cost_data["xy_drift"],
        "detach_cost":   cost_data["detach_cost"],
        "drift_cost":    cost_data["drift_cost"],
        "achieved":      cost_data["achieved"],
        "goal":          cost_data["goal"],
        "shortfall":     cost_data["shortfall"],
        "wall_time":     wall_time,
    }


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def _append_result_to_csv(csv_path: str, res: dict, elapsed_min: float) -> None:
    row = {
        "id":            res["id"],
        "cost":          res["cost"],
        "elapsed_min":   f"{elapsed_min:.1f}",
        "detach_force":  res["detach_force"],
        "detach_moment": res["detach_moment"],
        "xy_drift":      res["xy_drift"],
        "detach_cost":   res["detach_cost"],
        "drift_cost":    res["drift_cost"],
        "achieved":      res["achieved"],
        "goal":          res["goal"],
        "shortfall":     res["shortfall"],
    }
    row.update(res["params"])
    try:
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_csv_fieldnames()).writerow(row)
    except Exception as e:
        print(f"  [Warning] Could not append to CSV: {e}")


def _append_best_csv(best_csv_path: str, best: dict, n_done: int, elapsed_min: float) -> None:
    # FIX: include all scalar fields that _best_csv_fieldnames() expects,
    # not just params. Previously missing cost/achieved/etc. left blank columns.
    row = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "elapsed_min":  f"{elapsed_min:.1f}",
        "n_eval":       n_done,
        "id":           best["id"],
        "cost":         best["cost"],
        "detach_force": best["detach_force"],
        "detach_moment":best["detach_moment"],
        "xy_drift":     best["xy_drift"],
        "detach_cost":  best["detach_cost"],
        "drift_cost":   best["drift_cost"],
        "achieved":     best["achieved"],
        "goal":         best["goal"],
        "shortfall":    best["shortfall"],
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
            f"    [{i+1}/{len(results)}] id={r['id']}  cost={r['cost']:.4f}  "
            f"achieved={r['achieved']:.3f}/{r['goal']:.3f}  "
            f"shortfall={r['shortfall']:.3f}  "
            f"xy_drift={r['xy_drift']*1000:.2f}mm  "
            f"time={r['wall_time']:.1f}s"
        )


def _print_best_so_far(
    all_results: list[dict], n_done: int, elapsed_min: float,
    best_csv_path: str
) -> None:
    global _best_cost_so_far
    best      = min(all_results, key=lambda r: r["cost"])
    is_new    = best["cost"] < _best_cost_so_far
    marker    = " ★ NEW BEST" if is_new else ""
    mode_str  = "moment" if (APPLY_MOMENT and not APPLY_FORCE) else "force"
    print(
        f"  Best so far (n={n_done}): cost={best['cost']:.6f}  id={best['id']}{marker}\n"
        f"    {mode_str}: {best['achieved']:.3f} / {best['goal']:.3f}  "
        f"shortfall={best['shortfall']:.3f}  "
        f"xy_drift={best['xy_drift']*1000:.2f}mm"
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
        # FIX: always ask for the full population (ask() no longer takes n_points).
        # t_batch and batch_num reinstated here after refactor removed them.
        batch_num += 1
        t_batch    = time.perf_counter()
        points     = ask()
        n_this     = len(points)
        print(f"\n--- Batch {batch_num}: {n_this} points ({n_done+1}–{n_done+n_this} / {n_calls}) ---")

        tasks = [
            (i, point, PULL_RATE_OPT)
            for i, point in enumerate(points)
        ]

        # ── TODO: if multi-trial support is added, expand tasks here and
        #    aggregate results before calling tell(). ──────────────────────────
        raw_results = list(pool.imap_unordered(_evaluate_one_candidate, tasks, chunksize=1))
        raw_results.sort(key=lambda x: x[0])  # sort by point_index

        results = []
        for point_index, cost_data, wall_time in raw_results:
            r = _build_result(point_index, points[point_index], cost_data, wall_time)
            results.append(r)

        costs = [r["cost"] for r in results]  # aligned with ask._last_internal
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

        # Checkpoint CMA-ES state every batch
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
    parser = argparse.ArgumentParser(description="CMA-ES wrench parameter optimization")
    parser.add_argument("--suffix",           "-s", type=str,   default="",   help="Suffix for results folder")
    parser.add_argument("--n-calls",          type=int,         default=None, help=f"Override N_CALLS (default: {N_CALLS})")
    parser.add_argument("--warm-start-from",  type=str,         default=None, help="Results dir to warm-start from (reads optimization_bests.csv)")
    parser.add_argument("--resume-from",      type=str,         default=None, help="Results dir with cmaes_state.pkl to resume from")
    args = parser.parse_args()

    if args.resume_from and args.warm_start_from:
        print("ERROR: --resume-from and --warm-start-from are mutually exclusive")
        sys.exit(1)

    n_calls     = args.n_calls if args.n_calls is not None else N_CALLS
    es_resume   = None
    x0_override = None

    # Resume from checkpoint
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

    # Warm-start from best CSV
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

    # Create run directory
    run_tag  = datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.suffix:
        run_tag += f"_{args.suffix}"
    run_dir  = pathlib.Path("results") / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pathlib.Path(__file__).parent / "wrench_config.py", run_dir / "wrench_config.py")

    csv_path      = str(run_dir / "optimization_results.csv")
    best_csv_path = str(run_dir / "optimization_bests.csv")

    mode_str = "moment" if (APPLY_MOMENT and not APPLY_FORCE) else "force"
    goal_val = GOAL_WRENCH if (APPLY_MOMENT and not APPLY_FORCE) else GOAL_FORCE
    print(f"\nWrench optimizer: {n_calls} evals, batch={BATCH_SIZE}")
    print(f"Mode: {mode_str} | Goal: {goal_val:.2f} | Pull rate: {PULL_RATE_OPT} N/s")
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
    print("Best parameters:")
    for dim, val in zip(space, result.x):
        print(f"  {dim.name}: {val:.6g}")

    # ── Replay best params in viewer ─────────────────────────────────────────
    print("\nReplaying best parameters in viewer...")
    from wrench_config import point_to_params  # already called above, but safe to re-import
    from wrench_sim import run_headless        # ← was dropped
    import wrench_config
    import wrench_viewer

    best_params = point_to_params(result.x)

    wrench_config.PARAMS = best_params

    records, df, dm, drift = run_headless(pull_rate=PULL_RATE_OPT, params=best_params)
    print(f"Replay: detach_force={df:.2f} N | moment={dm:.3f} Nm | drift={drift*1000:.2f} mm")

    wrench_viewer.run_viewer(pull_rate=PULL_RATE_OPT)