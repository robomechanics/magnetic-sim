"""
combined_optimizer.py — CMA-ES optimizer over both pull-off and wrench sims jointly.
Combined cost = COST_WEIGHT_PULLOFF * pulloff_cost + COST_WEIGHT_WRENCH * wrench_cost.

Usage:
    python combined_optimizer.py
    python combined_optimizer.py --n-calls 500 --suffix my_run
    python combined_optimizer.py --resume-from results/20250101T120000_my_run
    python combined_optimizer.py --warm-start-from results/20250101T120000_my_run
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

import numpy as np

from wrench_config import (
    space, point_to_params,
    N_CALLS, BATCH_SIZE, CMAES_SIGMA0, OPTIMIZER_RANDOM_STATE,
    GOAL_FORCE as WRENCH_GOAL_FORCE, GOAL_WRENCH, APPLY_FORCE, APPLY_MOMENT,
    calculate_cost as wrench_calculate_cost,
)
from pulloff_config import (
    GOAL_FORCE as PULLOFF_GOAL_FORCE,
    calculate_cost as pulloff_calculate_cost,
)
from wrench_optimizer import _cmaes_space_info, _cmaes_to_real, _create_cmaes_optimizer

# ── Settings ──────────────────────────────────────────────────────────────────
PULL_RATE_PULLOFF   = 40.0   # N/s
PULL_RATE_WRENCH    = 40.0   # N/s
COST_WEIGHT_PULLOFF = 0.5
COST_WEIGHT_WRENCH  = 0.5
COST_FAILURE        = 9999.0

# ── CSV schemas ───────────────────────────────────────────────────────────────
_CSV_FIELDS = [
    "id", "cost", "elapsed_min",
    "pulloff_force", "pulloff_shortfall", "pulloff_cost",
    "wrench_achieved", "wrench_shortfall", "wrench_detach_cost", "wrench_drift_cost", "xy_drift",
] + [dim.name for dim in space]

_BEST_CSV_FIELDS = ["timestamp", "n_eval"] + _CSV_FIELDS


# ── Multiprocessing worker ────────────────────────────────────────────────────

def _evaluate_one_candidate(args):
    point_index, point, pr_pulloff, pr_wrench = args

    from pulloff_sim    import run_headless as pulloff_run
    from wrench_sim     import run_headless as wrench_run
    from pulloff_config import calculate_cost as pulloff_cost_fn
    from wrench_config  import calculate_cost as wrench_cost_fn, point_to_params

    params = point_to_params(point)
    t0     = time.perf_counter()

    try:
        _, pulloff_force = pulloff_run(pull_rate=pr_pulloff, params=params)
    except Exception as e:
        print(f"  [WARN] pulloff crashed (point {point_index}): {e}", flush=True)
        pulloff_force = 0.0

    try:
        _, detach_force, detach_moment, xy_drift = wrench_run(pull_rate=pr_wrench, params=params)
    except Exception as e:
        print(f"  [WARN] wrench crashed (point {point_index}): {e}", flush=True)
        detach_force = detach_moment = xy_drift = 0.0

    pc = pulloff_cost_fn(pulloff_force)
    wc = wrench_cost_fn(detach_force, detach_moment, xy_drift)

    total = COST_FAILURE if (pc["total_cost"] >= COST_FAILURE or wc["total_cost"] >= COST_FAILURE) \
            else COST_WEIGHT_PULLOFF * pc["total_cost"] + COST_WEIGHT_WRENCH * wc["total_cost"]

    return point_index, {
        "total_cost":         total,
        "pulloff_force":      pulloff_force,
        "pulloff_shortfall":  pc["shortfall"],
        "pulloff_cost":       pc["total_cost"],
        "pulloff_goal":       pc["goal"],
        "wrench_achieved":    wc["achieved"],
        "wrench_shortfall":   wc["shortfall"],
        "wrench_detach_cost": wc["detach_cost"],
        "wrench_drift_cost":  wc["drift_cost"],
        "wrench_goal":        wc["goal"],
        "xy_drift":           xy_drift,
    }, time.perf_counter() - t0


def _build_result(point_index, point, cost_data, wall_time):
    return {
        "id":        str(uuid.uuid4().hex)[:8],
        "cost":      cost_data["total_cost"],
        "params":    {dim.name: val for dim, val in zip(space, point)},
        "wall_time": wall_time,
        **{k: v for k, v in cost_data.items() if k != "total_cost"},
    }


def _write_csv_row(path, fields, res, elapsed_min, extra=None):
    row = {
        "id":                 res["id"],
        "cost":               f"{res['cost']:.6f}",
        "elapsed_min":        f"{elapsed_min:.2f}",
        "pulloff_force":      f"{res['pulloff_force']:.4f}",
        "pulloff_shortfall":  f"{res['pulloff_shortfall']:.6f}",
        "pulloff_cost":       f"{res['pulloff_cost']:.6f}",
        "wrench_achieved":    f"{res['wrench_achieved']:.4f}",
        "wrench_shortfall":   f"{res['wrench_shortfall']:.6f}",
        "wrench_detach_cost": f"{res['wrench_detach_cost']:.6f}",
        "wrench_drift_cost":  f"{res['wrench_drift_cost']:.6f}",
        "xy_drift":           f"{res['xy_drift']*1000:.4f}",
        **({} if extra is None else extra),
        **{dim.name: f"{res['params'][dim.name]:.8g}" for dim in space},
    }
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMA-ES combined pull-off + wrench optimization")
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
        p         = pathlib.Path(args.resume_from)
        state     = pickle.load(open(p / "cmaes_state.pkl" if p.is_dir() else p, "rb"))
        es_resume = state["es"]
        print(f"Resuming (sigma={es_resume.sigma:.4g}, prev evals={state['n_done']})")

    if args.warm_start_from:
        p           = pathlib.Path(args.warm_start_from)
        rows        = list(csv.DictReader(open(p / "optimization_bests.csv" if p.is_dir() else p)))
        x0_override = {dim.name: float(rows[-1][dim.name]) for dim in space}
        print(f"Warm-starting from {p} (cost={rows[-1]['cost']})")

    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S") + (f"_{args.suffix}" if args.suffix else "")
    run_dir = pathlib.Path("results") / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    for cfg in ("wrench_config.py", "pulloff_config.py"):
        shutil.copy2(pathlib.Path(__file__).parent / cfg, run_dir / cfg)

    csv_path      = str(run_dir / "optimization_results.csv")
    best_csv_path = str(run_dir / "optimization_bests.csv")
    for path, fields in [(csv_path, _CSV_FIELDS), (best_csv_path, _BEST_CSV_FIELDS)]:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    wrench_mode = "moment" if (APPLY_MOMENT and not APPLY_FORCE) else "force"
    wrench_goal = GOAL_WRENCH if (APPLY_MOMENT and not APPLY_FORCE) else WRENCH_GOAL_FORCE
    print(f"\nCombined optimizer: {n_calls} evals, batch={BATCH_SIZE}")
    print(f"Pull-off goal:  {PULLOFF_GOAL_FORCE:.2f} N  (rate: {PULL_RATE_PULLOFF} N/s)  weight={COST_WEIGHT_PULLOFF}")
    print(f"Wrench goal:    {wrench_goal:.2f} ({wrench_mode})  (rate: {PULL_RATE_WRENCH} N/s)  weight={COST_WEIGHT_WRENCH}")
    print(f"Run dir: {run_dir}/")

    multiprocessing.set_start_method("spawn", force=True)
    ask, tell, es = _create_cmaes_optimizer(x0_override=x0_override, es_override=es_resume)
    warm = "RESUMED" if es_resume else ("warm-start" if x0_override else "cold-start")
    print(f"Backend: CMA-ES ({warm}, sigma={es.sigma:.4g}, popsize={BATCH_SIZE})")

    pool_size      = max(1, min(os.cpu_count() or 16, BATCH_SIZE))
    all_results    = []
    best_cost      = float("inf")
    n_done         = 0
    batch_num      = 0
    t_start        = time.perf_counter()

    with multiprocessing.Pool(processes=pool_size) as pool:
        while n_done < n_calls:
            batch_num += 1
            t_batch    = time.perf_counter()
            points     = ask()
            n_this     = len(points)
            print(f"\n--- Batch {batch_num}: {n_this} points ({n_done+1}–{n_done+n_this} / {n_calls}) ---")

            tasks   = [(i, pt, PULL_RATE_PULLOFF, PULL_RATE_WRENCH) for i, pt in enumerate(points)]
            raw     = sorted(pool.imap_unordered(_evaluate_one_candidate, tasks, chunksize=1), key=lambda x: x[0])
            results = [_build_result(idx, points[idx], cd, wt) for idx, cd, wt in raw]
            costs   = [r["cost"] for r in results]
            tell(points, costs)

            elapsed_min = (time.perf_counter() - t_start) / 60.0
            for r in results:
                all_results.append(r)
                _write_csv_row(csv_path, _CSV_FIELDS, r, elapsed_min)

            n_done += n_this

            for r in sorted(results, key=lambda x: x["cost"]):
                print(f"  id={r['id']}  cost={r['cost']:.4f}  "
                      f"pulloff={r['pulloff_force']:.1f}/{r['pulloff_goal']:.1f} N (sf={r['pulloff_shortfall']:.3f})  "
                      f"wrench={r['wrench_achieved']:.1f}/{r['wrench_goal']:.1f} (sf={r['wrench_shortfall']:.3f})  "
                      f"drift={r['xy_drift']*1000:.2f}mm  t={r['wall_time']:.1f}s")
            print(f"  Batch wall: {time.perf_counter()-t_batch:.1f}s | Elapsed: {elapsed_min:.1f}min | "
                  f"Costs: min={min(costs):.4f}, max={max(costs):.4f}")

            top = min(all_results, key=lambda r: r["cost"])
            is_new = top["cost"] < best_cost
            print(f"  Best (n={n_done}): cost={top['cost']:.6f}{' ★ NEW BEST' if is_new else ''}\n"
                  f"    pulloff={top['pulloff_force']:.2f}/{top['pulloff_goal']:.2f} N  sf={top['pulloff_shortfall']:.3f}  |  "
                  f"wrench={top['wrench_achieved']:.2f}/{top['wrench_goal']:.2f}  sf={top['wrench_shortfall']:.3f}  "
                  f"drift={top['xy_drift']*1000:.2f}mm")
            if is_new:
                best_cost = top["cost"]
                _write_csv_row(best_csv_path, _BEST_CSV_FIELDS, top, elapsed_min,
                               extra={"timestamp": datetime.now().isoformat(timespec="seconds"), "n_eval": n_done})

            with open(run_dir / "cmaes_state.pkl", "wb") as f:
                pickle.dump({"es": es, "n_done": n_done}, f)

    top         = min(all_results, key=lambda r: r["cost"])
    best_params = point_to_params([top["params"][dim.name] for dim in space])

    print(f"\n--- Optimization Finished ---  Lowest cost: {top['cost']:.6f}")
    for dim in space:
        print(f"  {dim.name}: {top['params'][dim.name]:.6g}")

    import pulloff_config, wrench_config, pulloff_viewer, wrench_viewer
    from pulloff_sim import run_headless as pulloff_run
    from wrench_sim  import run_headless as wrench_run

    pulloff_config.PARAMS = best_params
    wrench_config.PARAMS  = best_params

    _, pf = pulloff_run(pull_rate=PULL_RATE_PULLOFF, params=best_params)
    print(f"Pull-off replay: {pf:.2f} N")

    _, df, dm, drift = wrench_run(pull_rate=PULL_RATE_WRENCH, params=best_params)
    print(f"Wrench replay: detach_force={df:.2f} N | moment={dm:.3f} Nm | drift={drift*1000:.2f} mm")

    pulloff_viewer.run_viewer(pull_rate=PULL_RATE_PULLOFF)
    wrench_viewer.run_viewer(pull_rate=PULL_RATE_WRENCH)