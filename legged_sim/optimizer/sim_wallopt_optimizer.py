"""
sim_wallopt_optimizer.py — CMA-ES optimizer for FL wall-adhesion during FR swing.

Tunes 14 physics parameters to minimize FL EE drift on the wall while
FR executes its floor-to-wall sequence.

Cost (lower = better):
  40% — mean absolute Z drift of FL EE  (gravity-driven slip)
  30% — mean absolute X drift           (into/away from wall)
  30% — mean absolute Y drift           (sideways along wall)

Usage:
    python sim_wallopt_optimizer.py
    python sim_wallopt_optimizer.py --n-calls 500 --suffix wall_run
    python sim_wallopt_optimizer.py --resume-from results/20250101T120000_wall_run
    python sim_wallopt_optimizer.py --warm-start-from results/20250101T120000_wall_run

Warm-starting from a lift-optimizer run is also valid since both share the
same 14-parameter search space — pass the lift-opt best CSV as --warm-start-from.
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

from sim_wallopt_config import (
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
        "fl_x_drift_mm", "fl_y_drift_mm", "fl_z_drift_mm", "stance_drift_mm",
        "fl_x_cost", "fl_y_cost", "fl_z_cost", "stance_cost",
    ]
    return base + [dim.name for dim in space]


def _best_csv_fieldnames() -> list[str]:
    return ["timestamp", "elapsed_min", "n_eval"] + _csv_fieldnames()


# ── Multiprocessing worker ────────────────────────────────────────────────────

def _evaluate_one_candidate(args):
    """Run a single wall-adhesion trial for one candidate parameter point."""
    point_index, point = args

    import os, sys
    _optimizer_dir = os.path.dirname(os.path.abspath(__file__))
    _legged_dir    = os.path.abspath(os.path.join(_optimizer_dir, ".."))
    if _legged_dir    not in sys.path: sys.path.insert(0, _legged_dir)
    if _optimizer_dir not in sys.path: sys.path.insert(0, _optimizer_dir)

    from sim_wallopt_sim    import run_headless_wall
    from sim_wallopt_config import point_to_params, calculate_cost

    params = point_to_params(point)

    t0 = time.perf_counter()
    try:
        fl_x, fl_y, fl_z, stance = run_headless_wall(params)
    except Exception as e:
        print(f"  [WARN] Sim crashed (point {point_index}): {e}", flush=True)
        fl_x = fl_y = fl_z = stance = 1.0

    wall_time = time.perf_counter() - t0
    cost_data = calculate_cost(fl_x, fl_y, fl_z, stance)
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


# ── Pure-numpy CMA-ES ─────────────────────────────────────────────────────────
# Standard (mu/2, mu)-CMA-ES. No external dependencies.

class _CMAESState:
    """Minimal CMA-ES state — picklable for resume support."""

    def __init__(self, x0, sigma0, lower, upper, popsize, seed):
        rng       = np.random.default_rng(seed)
        n         = len(x0)
        mu        = popsize // 2

        # Recombination weights (log-sum normalised)
        w_raw     = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights   = w_raw / w_raw.sum()
        mueff     = 1.0 / (weights ** 2).sum()

        # Step-size control
        cs        = (mueff + 2) / (n + mueff + 5)
        ds        = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN      = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Covariance control
        cc        = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1        = 2 / ((n + 1.3) ** 2 + mueff)
        cmu       = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        self.n       = n
        self.popsize = popsize
        self.mu      = mu
        self.weights = weights
        self.mueff   = mueff
        self.lower   = np.array(lower)
        self.upper   = np.array(upper)

        # Strategy parameters
        self.cs, self.ds, self.chiN = cs, ds, chiN
        self.cc, self.c1, self.cmu  = cc, c1, cmu

        # State
        self.mean    = np.array(x0, dtype=float)
        self.sigma   = sigma0
        self.C       = np.eye(n)
        self.ps      = np.zeros(n)
        self.pc      = np.zeros(n)
        self.eigeneval  = 0
        self.counteval  = 0
        self.rng        = rng

        # Initialise eigen-decomposition of C = I
        self._D = np.ones(n)
        self._B = np.eye(n)

        # Last sample (needed for tell)
        self._last_z    = None   # shape (popsize, n) — standard normal samples
        self._last_y    = None   # shape (popsize, n) — scaled samples (before clamp)

    def ask(self) -> np.ndarray:
        """Sample popsize candidate points. Returns (popsize, n) array (internal space)."""
        n = self.n
        # Eigen-decompose C every n/10 steps for efficiency
        if self.counteval - self.eigeneval > self.popsize / (self.c1 + self.cmu) / n / 10:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T   # enforce symmetry
            eigvals, B = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-20)
            self._D  = np.sqrt(eigvals)
            self._B  = B
        D, B = self._D, self._B
        z = self.rng.standard_normal((self.popsize, n))
        y = (B @ (D[:, None] * z.T)).T          # shape (popsize, n)
        x = self.mean + self.sigma * y
        # Clip to bounds
        x = np.clip(x, self.lower, self.upper)
        self._last_z = z
        self._last_y = y
        return x

    def tell(self, xs: np.ndarray, costs: list[float]) -> None:
        """Update strategy from evaluated costs. xs: (popsize, n), costs: (popsize,)."""
        n = self.n
        order  = np.argsort(costs)
        best_z = self._last_z[order[:self.mu]]   # (mu, n)
        best_y = self._last_y[order[:self.mu]]   # (mu, n)

        # New mean
        y_mean = (self.weights[:, None] * best_y).sum(axis=0)
        self.mean = self.mean + self.sigma * y_mean
        self.mean = np.clip(self.mean, self.lower, self.upper)

        # Step-size control (CSA)
        z_mean   = (self.weights[:, None] * best_z).sum(axis=0)
        self.ps  = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z_mean
        hs       = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.counteval + 1)))
                    < (1.4 + 2 / (n + 1)) * self.chiN)
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma  = float(np.clip(self.sigma, 1e-10, 10.0))

        # Covariance update (CMA)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_mean
        C_mu    = sum(self.weights[i] * np.outer(best_y[i], best_y[i]) for i in range(self.mu))
        self.C  = ((1 - self.c1 - self.cmu) * self.C
                   + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                   + self.cmu * C_mu)
        self.counteval += self.popsize


def _create_cmaes_optimizer(x0_override=None, es_override=None):
    x0_raw, lower, upper, is_log = _cmaes_space_info()

    if es_override is not None:
        es = es_override
    else:
        if x0_override is not None:
            x0_raw = []
            for dim, log in zip(space, is_log):
                v = x0_override[dim.name]
                x0_raw.append(np.log10(v) if log else v)
        es = _CMAESState(
            x0=x0_raw, sigma0=CMAES_SIGMA0,
            lower=lower, upper=upper,
            popsize=BATCH_SIZE, seed=OPTIMIZER_RANDOM_STATE,
        )

    def ask():
        internal = es.ask()           # (popsize, n) ndarray
        ask._last_internal = internal
        return [_cmaes_to_real(internal[i], is_log) for i in range(len(internal))]

    ask._last_internal = None

    def tell(points, costs):
        es.tell(ask._last_internal, costs)

    return ask, tell, es


# ── Result aggregation ────────────────────────────────────────────────────────

def _build_result(point_index: int, point: list, cost_data: dict, wall_time: float) -> dict:
    return {
        "id":             str(uuid.uuid4().hex)[:8],
        "cost":           cost_data["total_cost"],
        "params":         {dim.name: val for dim, val in zip(space, point)},
        "fl_x_drift_mm":  cost_data["fl_x_drift"] * 1000,
        "fl_y_drift_mm":  cost_data["fl_y_drift"] * 1000,
        "fl_z_drift_mm":  cost_data["fl_z_drift"] * 1000,
        "stance_drift_mm": cost_data["stance_drift"] * 1000,
        "fl_x_cost":      cost_data["fl_x_cost"],
        "fl_y_cost":      cost_data["fl_y_cost"],
        "fl_z_cost":      cost_data["fl_z_cost"],
        "stance_cost":    cost_data["stance_cost"],
        "wall_time":      wall_time,
    }


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def _append_result_to_csv(csv_path: str, res: dict, elapsed_min: float) -> None:
    row = {
        "id":              res["id"],
        "cost":            res["cost"],
        "elapsed_min":     f"{elapsed_min:.1f}",
        "fl_x_drift_mm":   res["fl_x_drift_mm"],
        "fl_y_drift_mm":   res["fl_y_drift_mm"],
        "fl_z_drift_mm":   res["fl_z_drift_mm"],
        "stance_drift_mm": res["stance_drift_mm"],
        "fl_x_cost":       res["fl_x_cost"],
        "fl_y_cost":       res["fl_y_cost"],
        "fl_z_cost":       res["fl_z_cost"],
        "stance_cost":     res["stance_cost"],
    }
    row.update(res["params"])
    try:
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_csv_fieldnames()).writerow(row)
    except Exception as e:
        print(f"  [Warning] Could not append to CSV: {e}")


def _append_best_csv(best_csv_path: str, best: dict, n_done: int, elapsed_min: float) -> None:
    row = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "elapsed_min":     f"{elapsed_min:.1f}",
        "n_eval":          n_done,
        "id":              best["id"],
        "cost":            best["cost"],
        "fl_x_drift_mm":   best["fl_x_drift_mm"],
        "fl_y_drift_mm":   best["fl_y_drift_mm"],
        "fl_z_drift_mm":   best["fl_z_drift_mm"],
        "stance_drift_mm": best["stance_drift_mm"],
        "fl_x_cost":       best["fl_x_cost"],
        "fl_y_cost":       best["fl_y_cost"],
        "fl_z_cost":       best["fl_z_cost"],
        "stance_cost":     best["stance_cost"],
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
            f"FL: x={r['fl_x_drift_mm']:+.2f}mm y={r['fl_y_drift_mm']:+.2f}mm z={r['fl_z_drift_mm']:+.2f}mm  "
            f"stance={r['stance_drift_mm']:.2f}mm  wall={r['wall_time']:.1f}s"
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
        f"    FL drift: x={best['fl_x_drift_mm']:.2f}mm  y={best['fl_y_drift_mm']:.2f}mm  "
        f"z={best['fl_z_drift_mm']:.2f}mm  stance={best['stance_drift_mm']:.2f}mm"
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

        tasks       = [(i, point) for i, point in enumerate(points)]
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
    parser = argparse.ArgumentParser(
        description="CMA-ES optimizer — FL wall adhesion during FR swing")
    parser.add_argument("--suffix",          "-s", type=str,  default="",   help="Suffix for results folder")
    parser.add_argument("--n-calls",         type=int,        default=None, help=f"Override N_CALLS (default: {N_CALLS})")
    parser.add_argument("--warm-start-from", type=str,        default=None, help="Results dir or bests CSV to warm-start from")
    parser.add_argument("--resume-from",     type=str,        default=None, help="Results dir with cmaes_state.pkl to resume")
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

    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.suffix:
        run_tag += f"_{args.suffix}"
    run_dir = pathlib.Path("results") / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pathlib.Path(__file__).parent / "sim_wallopt_config.py",
                 run_dir / "sim_wallopt_config.py")

    csv_path      = str(run_dir / "optimization_results.csv")
    best_csv_path = str(run_dir / "optimization_bests.csv")

    print(f"\nFL wall-adhesion optimizer: {n_calls} evals, batch={BATCH_SIZE}")
    print(f"Cost: FL drift (65%: Z=39% X=13% Y=13%) + BL/BR stance drift (35%)")
    print(f"Sampling: FR_MEASURE_DELAY={0.5}s after FR starts → FR_REACH_DWELL={2.0}s in F2W_REACH")
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