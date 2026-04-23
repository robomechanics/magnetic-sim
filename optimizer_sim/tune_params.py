from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import pathlib
import pickle
import shutil
import time
import uuid
from datetime import datetime
from typing import NamedTuple

import numpy as np

import result as result_bundle
import sim_optimizer
from config import (
    BASELINE_PARAMS,
    BATCH_SIZE,
    CMAES_SIGMA0,
    COST_FAILURE,
    DEFAULT_MODE,
    DEFAULT_PARAMS,
    N_CALLS,
    OPTIMIZER_RANDOM_STATE,
    MODES,
    SEARCH_SPACE,
)


class OptResult(NamedTuple):
    fun: float
    x: list[float]


def _point_to_sim_params(point: list[float]) -> dict:
    p = {dim.name: val for dim, val in zip(SEARCH_SPACE, point)}
    return {
        "ground_friction": [p["sliding_friction"], p["torsional_friction"], p["rolling_friction"]],
        "solref": [p["solref_timeconst"], p["solref_dampratio"]],
        "solimp": [p["solimp_dmin"], p["solimp_dmax"], p["solimp_width"], 0.5, 1.0],
        "noslip_iterations": int(round(p["noslip_iterations"])),
        "Br": p["Br"],
        "max_magnetic_distance": p["max_magnetic_distance"],
        "rocker_stiffness": p["rocker_stiffness"],
        "rocker_damping": p["rocker_damping"],
        "wheel_kp": p["wheel_kp"],
        "wheel_kv": p["wheel_kv"],
        "max_force_per_wheel": p["max_force_per_wheel"],
    }


def _sim_params_to_x0(sim_params: dict) -> dict[str, float]:
    return {
        "Br": sim_params["Br"],
        "solref_timeconst": sim_params["solref"][0],
        "solref_dampratio": sim_params["solref"][1],
        "solimp_dmin": sim_params["solimp"][0],
        "solimp_dmax": sim_params["solimp"][1],
        "solimp_width": sim_params["solimp"][2],
        "sliding_friction": sim_params["ground_friction"][0],
        "torsional_friction": sim_params["ground_friction"][1],
        "rolling_friction": sim_params["ground_friction"][2],
        "rocker_stiffness": sim_params["rocker_stiffness"],
        "rocker_damping": sim_params["rocker_damping"],
        "wheel_kp": sim_params["wheel_kp"],
        "wheel_kv": sim_params["wheel_kv"],
        "max_magnetic_distance": sim_params["max_magnetic_distance"],
        "noslip_iterations": sim_params["noslip_iterations"],
        "max_force_per_wheel": sim_params["max_force_per_wheel"],
    }


def cost_minimize_slip(trajectory, mode_cfg):
    if not trajectory:
        return {"total_cost": COST_FAILURE, "total_movement": 0.0}
    settle_time = mode_cfg["settle_time"]
    start_idx = next((i for i, s in enumerate(trajectory) if s["time"] >= settle_time), 0)
    total_movement = 0.0
    x_wall_errs = []
    for i in range(start_idx + 1, len(trajectory)):
        dp = trajectory[i]["pos"] - trajectory[i - 1]["pos"]
        total_movement += float(np.linalg.norm(dp))
        x_wall_errs.append(abs(trajectory[i]["pos"][0] - 0.035))
    # Penalise drifting away from the wall (X should stay ~0.035 m)
    x_wall_penalty = float(np.mean(x_wall_errs)) * 5.0 if x_wall_errs else 0.0
    total_cost = total_movement + x_wall_penalty
    return {"total_cost": total_cost, "total_movement": total_movement}


def cost_drive(trajectory, mode_cfg):
    """
    Generic drive cost function — works for any drive mode.

    Reads tracking_axis and target_velocity_xyz from mode_cfg so this single
    function handles drive_sideways (axis=1, +Y) and drive_up (axis=2, +Z)
    without any hardcoded axis logic.

    Cost = velocity tracking error on primary axis
         + 0.2 * each off-axis endpoint displacement
    This matches the old per-mode cost functions that actually worked.
    The physics (max_force_per_wheel in [100,1000] N) is what prevents slip,
    not complex penalty terms.
    """
    axis       = mode_cfg["tracking_axis"]        # int: 1=Y (sideways), 2=Z (up)
    target_vec = mode_cfg["target_velocity_xyz"]  # np.array([0,1,0]) or ([0,0,1])
    target_vel = float(target_vec[axis])          # signed scalar, e.g. +1.0 m/s

    fail_ret = {"total_cost": COST_FAILURE, "avg_vel": 0.0, "avg_x_disp": 0.0, "avg_y_disp": 0.0, "avg_z_disp": 0.0}
    if not trajectory:
        return fail_ret

    settle_time = mode_cfg["settle_time"]
    settled     = [s for s in trajectory if s["time"] >= settle_time]
    start_state = settled[0] if settled else trajectory[0]
    end_state   = trajectory[-1]
    dt          = end_state["time"] - start_state["time"]
    if dt < 1e-6:
        return fail_ret

    p_start = start_state["pos"]
    p_end   = end_state["pos"]

    # Primary axis: average velocity vs target
    avg_vel      = (p_end[axis] - p_start[axis]) / dt
    vel_error    = abs(avg_vel - target_vel)

    # Off-axis: small penalty on every axis that is not the tracking axis
    off_axis_cost = sum(
        0.2 * abs(p_end[ax] - p_start[ax])
        for ax in range(3) if ax != axis
    )

    total_cost = vel_error + off_axis_cost

    print(
        f"  axis={axis} avg_vel={avg_vel:.4f} target={target_vel:.4f} | "
        f"vel_err={vel_error:.4f} off_axis={off_axis_cost:.4f} | cost={total_cost:.4f}"
    )
    return {
        "total_cost": total_cost,
        "avg_vel": avg_vel,
        "avg_x_disp": float(abs(p_end[0] - p_start[0])),
        "avg_y_disp": float(abs(p_end[1] - p_start[1])),
        "avg_z_disp": float(abs(p_end[2] - p_start[2])),
    }


COST_FUNCTIONS = {
    "minimize_slip": cost_minimize_slip,
    "drive": cost_drive,   # used by both drive_sideways and drive_up via tracking_axis + target_velocity_xyz
}


def _evaluate_one_candidate(args):
    point_idx, point, mode = args
    mode_cfg = MODES[mode]
    sim_params = _point_to_sim_params(point)
    t0 = time.perf_counter()
    try:
        trajectory = sim_optimizer.run_simulation(sim_params, mode=mode)
        if trajectory is None:
            cost_data = {"total_cost": COST_FAILURE}
        else:
            cost_fn = COST_FUNCTIONS[mode_cfg["cost_function"]]
            cost_data = cost_fn(trajectory, mode_cfg)
    except Exception as e:
        print(f"  [WARN] candidate {point_idx} crashed: {e}", flush=True)
        cost_data = {"total_cost": COST_FAILURE}
    wall_time = time.perf_counter() - t0
    return point_idx, cost_data, wall_time


def _cmaes_space_info():
    x0, lower, upper, is_log = [], [], [], []
    for dim in SEARCH_SPACE:
        lo, hi = dim.low, dim.high
        if getattr(dim, "prior", None) == "log-uniform":
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
    real = []
    for dim, v, log in zip(SEARCH_SPACE, x_internal, is_log):
        rv = 10.0 ** v if log else v
        if hasattr(dim, "low") and hasattr(dim, "high"):
            rv = min(max(rv, dim.low), dim.high)
        if dim.__class__.__name__ == "Integer":
            rv = int(round(rv))
            rv = min(max(rv, int(dim.low)), int(dim.high))
        real.append(rv)
    return real


def _create_cmaes_optimizer(x0_override=None, es_override=None):
    import cma
    x0_raw, lower, upper, is_log = _cmaes_space_info()
    if es_override is not None:
        es = es_override
    else:
        if x0_override is not None:
            x0_raw = []
            for dim, log in zip(SEARCH_SPACE, is_log):
                v = x0_override[dim.name]
                x0_raw.append(np.log10(v) if log else v)
        # CMA-ES inverse-bounds transform requires x0 strictly within [lower, upper].
        # Clamp so that stale defaults or out-of-range warm-starts never crash the optimizer.
        x0_raw = [float(np.clip(v, lo, hi)) for v, lo, hi in zip(x0_raw, lower, upper)]
        opts = {
            "bounds": [lower, upper],
            "seed": OPTIMIZER_RANDOM_STATE,
            "popsize": BATCH_SIZE,
            "verbose": -1,
            "tolfun": 1e-8,
            "tolx": 1e-10,
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


def _build_result(point, cost_data, wall_time):
    return {
        "id": str(uuid.uuid4().hex)[:8],
        "cost": float(cost_data["total_cost"]),
        "cost_data": cost_data,
        "params": {dim.name: val for dim, val in zip(SEARCH_SPACE, point)},
        "wall_time": wall_time,
    }


def _csv_fieldnames(extra_keys):
    return ["id", "cost", "wall_time"] + extra_keys + [dim.name for dim in SEARCH_SPACE]


def _append_results_csv(csv_path: pathlib.Path, results: list[dict], extra_keys: list[str]):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames(extra_keys))
        for res in results:
            row = {"id": res["id"], "cost": res["cost"], "wall_time": res["wall_time"]}
            row.update({k: res["cost_data"].get(k, "") for k in extra_keys})
            row.update(res["params"])
            writer.writerow(row)


def _print_batch_results(results: list[dict], batch_num: int, n_done: int, n_calls: int):
    print(f"\n--- Batch {batch_num}: {len(results)} points ({n_done + 1}–{n_done + len(results)} / {n_calls}) ---")
    for i, r in enumerate(results, start=1):
        print(f"    [{i}/{len(results)}] id={r['id']} cost={r['cost']:.6f} time={r['wall_time']:.1f}s")


def _run_optimization(mode: str, n_calls: int, run_dir: pathlib.Path, es_resume=None, x0_override=None) -> OptResult:
    ask, tell, es = _create_cmaes_optimizer(x0_override=x0_override, es_override=es_resume)
    all_results = []
    csv_path = run_dir / f"optimization_results_{mode}.csv"
    extra_keys = ["avg_vel", "avg_x_disp", "avg_y_disp", "avg_z_disp", "total_movement"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_csv_fieldnames(extra_keys)).writeheader()

    pool = mp.Pool(processes=max(1, min(mp.cpu_count() or 8, BATCH_SIZE)))
    try:
        n_done = 0
        batch_num = 0
        while n_done < n_calls:
            batch_num += 1
            points = ask()
            tasks = [(i, point, mode) for i, point in enumerate(points)]
            raw = list(pool.imap_unordered(_evaluate_one_candidate, tasks, chunksize=1))
            raw.sort(key=lambda x: x[0])
            results = [_build_result(points[idx], cost_data, wall_time) for idx, cost_data, wall_time in raw]
            costs = [r["cost"] for r in results]
            tell(points, costs)
            _append_results_csv(csv_path, results, extra_keys)
            all_results.extend(results)
            _print_batch_results(results, batch_num, n_done, n_calls)
            best = min(all_results, key=lambda r: r["cost"])
            print(f"  Best so far: cost={best['cost']:.6f} id={best['id']}")
            n_done += len(results)
            with open(run_dir / "cmaes_state.pkl", "wb") as f:
                pickle.dump({"es": es, "n_done": n_done}, f)
    finally:
        pool.terminate()
        pool.join()

    best = min(all_results, key=lambda r: r["cost"])
    with open(run_dir / "best_result.json", "w") as f:
        json.dump(best, f, indent=2)
    return OptResult(fun=best["cost"], x=[best["params"][dim.name] for dim in SEARCH_SPACE])


def main():
    parser = argparse.ArgumentParser(description="CMA-ES optimizer for Sally magnetic simulation")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=MODES.keys())
    parser.add_argument("--n-calls", type=int, default=N_CALLS)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--warm-start-from", type=str, default=None)
    args = parser.parse_args()

    if args.resume_from and args.warm_start_from:
        raise SystemExit("--resume-from and --warm-start-from are mutually exclusive")

    es_resume = None
    x0_override = _sim_params_to_x0(DEFAULT_PARAMS)

    if args.resume_from:
        resume_path = pathlib.Path(args.resume_from)
        if resume_path.is_dir():
            resume_path = resume_path / "cmaes_state.pkl"
        with open(resume_path, "rb") as f:
            state = pickle.load(f)
        es_resume = state["es"]
        print(f"Resuming CMA-ES from {resume_path}")

    if args.warm_start_from:
        ws_path = pathlib.Path(args.warm_start_from)
        if ws_path.is_dir():
            ws_path = ws_path / "best_result.json"
        with open(ws_path) as f:
            best = json.load(f)
        x0_override = {k: float(v) for k, v in best["params"].items()}
        print(f"Warm-starting CMA-ES from {ws_path}")

    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.suffix:
        run_tag += f"_{args.suffix}"
    run_dir = pathlib.Path("results") / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pathlib.Path(__file__).parent / "config.py", run_dir / "config.py")

    print(f"Running CMA-ES optimization (mode={args.mode}, evals={args.n_calls}, popsize={BATCH_SIZE})")
    result = _run_optimization(args.mode, args.n_calls, run_dir, es_resume=es_resume, x0_override=x0_override)

    print("\n--- Optimization Finished ---")
    print(f"Lowest Cost Found: {result.fun:.6f}")
    best_params = _point_to_sim_params(result.x)
    print("\nDEFAULT_PARAMS = {")
    print(f"    'ground_friction': [{best_params['ground_friction'][0]:.6f}, {best_params['ground_friction'][1]:.6f}, {best_params['ground_friction'][2]:.6f}],")
    print(f"    'solref': [{best_params['solref'][0]:.6f}, {best_params['solref'][1]:.6f}],")
    print(f"    'solimp': [{best_params['solimp'][0]:.6f}, {best_params['solimp'][1]:.6f}, {best_params['solimp'][2]:.6f}, 0.5, 1.0],")
    print(f"    'noslip_iterations': {best_params['noslip_iterations']},")
    print(f"    'rocker_stiffness': {best_params['rocker_stiffness']:.6f},")
    print(f"    'rocker_damping': {best_params['rocker_damping']:.6f},")
    print(f"    'wheel_kp': {best_params['wheel_kp']:.6f},")
    print(f"    'wheel_kv': {best_params['wheel_kv']:.6f},")
    print(f"    'Br': {best_params['Br']:.6f},")
    print(f"    'max_magnetic_distance': {best_params['max_magnetic_distance']:.6f},")
    print(f"    'max_force_per_wheel': {best_params['max_force_per_wheel']:.6f},")
    print("}")

    print("\n--- Generating preliminary results bundle ---")
    result_bundle.generate_results_bundle(
        mode=args.mode,
        baseline_params=BASELINE_PARAMS,
        optimized_params=best_params,
        run_dir=run_dir,
    )

    import viewer
    print("\n--- Replaying BASELINE simulation (close window to continue) ---")
    viewer.visualize_simulation(BASELINE_PARAMS, mode=args.mode)

    print("\n--- Replaying OPTIMIZED simulation (close window to finish) ---")
    viewer.visualize_simulation(best_params, mode=args.mode)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()