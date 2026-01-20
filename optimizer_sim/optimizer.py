"""
optimizer.py

STAGE 3 — OPTIMIZATION LOOP AND LOGGING

- Samples parameters x (random search by default; can be swapped later)
- Runs rollout f(x) from sim_sally_magnet_wall.py
- Reads reward R from result.summary["reward"]
- Logs every trial (params, reward, termination, key metrics)
- Supports parallel rollouts using multiprocessing

Log format:
- JSONL (one record per trial): replayable and diff-friendly
- Optional CSV summary for quick plotting
"""

from __future__ import annotations

import json
import time
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from multiprocessing import Pool, cpu_count

# Import your rollout function (unchanged)
from sim_sally_magnet_wall import rollout


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class OptimConfig:
    n_trials: int = 50
    seed: int = 0

    # Rollout settings
    sim_duration: float = 5.0
    settle_time: float = 0.0
    log_stride: int = 10
    fixed_torque: float = 0.5

    # Rollout mode for Stage-2 simplified objective
    # "drive" -> progress / stuck / detached
    # "hold"  -> mostly slip (you disabled slip now, but keep for future)
    rollout_mode: str = "drive"

    # Robot XML generator mode
    # must match what your generator expects (e.g., "sideways" / "drive_up")
    mode: str = "sideways"

    # Parallelism
    n_workers: int = 1  # set >1 to parallelize

    # Logging
    out_dir: str = "logs"
    exp_name: str = "stage3"


# -----------------------------
# Search space helpers
# -----------------------------
def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def sample_params_prior(rng: np.random.Generator) -> Dict[str, Any]:
    """
    Broad sampling of parameters you actually want to tune.
    Br is intentionally NOT included.
    """
    timestep = float(rng.choice([0.0005, 0.001, 0.002]))
    o_solref = [rng.uniform(1e-4, 8e-4), rng.uniform(10.0, 50.0)]
    o_solimp = [0.99, 0.99, 0.001, 0.5, 2.0]
    
    # NEW: Friction parameters (matching your milliquad ranges)
    wheel_friction = [
        rng.uniform(0.01, 0.95),   # sliding
        rng.uniform(0.001, 0.1),   # torsional
        rng.uniform(0.001, 0.1),   # rolling
    ]
    
    # NEW: Joint dynamics
    wheel_damping = rng.uniform(0.01, 1.0)
    rocker_stiffness = rng.uniform(10.0, 100.0)
    rocker_damping = rng.uniform(0.1, 5.0)

    return {
        "timestep": float(timestep),
        "o_solref": [float(o_solref[0]), float(o_solref[1])],
        "o_solimp": [float(x) for x in o_solimp],
        "wheel_friction": [float(x) for x in wheel_friction],
        "wheel_damping": float(wheel_damping),
        "rocker_stiffness": float(rocker_stiffness),
        "rocker_damping": float(rocker_damping),
    }



def sample_params_refined(
    rng: np.random.Generator,
    mean: Dict[str, float],
    std: Dict[str, float],
) -> Dict[str, Any]:
    """Adaptive sampling around elites (Gaussian) — Br is fixed (NOT optimized)."""
    timestep = float(rng.choice([0.0005, 0.001, 0.002]))

    solref0 = rng.normal(mean["o_solref0"], std["o_solref0"])
    solref0 = _clip(solref0, 1e-4, 8e-4)

    solref1 = rng.normal(mean["o_solref1"], std["o_solref1"])
    solref1 = _clip(solref1, 10.0, 50.0)

    o_solimp = [0.99, 0.99, 0.001, 0.5, 2.0]

    return {
        "timestep": float(timestep),
        "o_solref": [float(solref0), float(solref1)],
        "o_solimp": [float(x) for x in o_solimp],
    }




# -----------------------------
# Logging utilities
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow(row)


# -----------------------------
# Single trial execution
# -----------------------------
def run_one_trial(trial_id: int, params: Dict[str, Any], cfg: OptimConfig) -> Dict[str, Any]:
    """
    Run rollout in BOTH modes and compute combined reward.
    """
    t0 = time.time()
    
    # Run HOLD test (sideways, should not slip)
    params_hold = dict(params)
    params_hold["mode"] = "sideways"
    params_hold["rollout_mode"] = "hold"
    
    result_hold = rollout(
        params=params_hold,
        sim_duration=cfg.sim_duration,
        settle_time=cfg.settle_time,
        log_stride=cfg.log_stride,
        fixed_torque=cfg.fixed_torque,
    )
    
    # Run DRIVE test (drive_up, should climb)
    params_drive = dict(params)
    params_drive["mode"] = "drive_up"
    params_drive["rollout_mode"] = "drive"
    
    result_drive = rollout(
        params=params_drive,
        sim_duration=cfg.sim_duration,
        settle_time=cfg.settle_time,
        log_stride=cfg.log_stride,
        fixed_torque=cfg.fixed_torque,
    )
    
    wall = time.time() - t0

    # Extract summaries
    summary_hold = dict(result_hold.summary or {})
    summary_drive = dict(result_drive.summary or {})
    
    reward_hold = float(summary_hold.get("reward", -1e9))
    reward_drive = float(summary_drive.get("reward", -1e9))
    
    # Combined reward (both tests must pass)
    # If either fails, use the worse (more negative) reward
    combined_reward = min(reward_hold, reward_drive)
    
    termination_hold = str(result_hold.termination)
    termination_drive = str(result_drive.termination)

    return {
        "trial_id": int(trial_id),
        "time_unix": float(time.time()),
        "walltime_s": float(wall),
        "params": params,
        
        # Hold mode results
        "termination_hold": termination_hold,
        "reward_hold": reward_hold,
        "slip_m": summary_hold.get("slip_m", 0.0),
        "detached_hold": summary_hold.get("detached", False),
        
        # Drive mode results
        "termination_drive": termination_drive,
        "reward_drive": reward_drive,
        "progress_m": summary_drive.get("progress_m", 0.0),
        "progress_rate_mps": summary_drive.get("progress_rate_mps", 0.0),
        "detached_drive": summary_drive.get("detached", False),
        "stuck": summary_drive.get("stuck", False),
        
        # Combined
        "reward": combined_reward,
        
        "rollout_settings": {
            "sim_duration": cfg.sim_duration,
            "settle_time": cfg.settle_time,
            "log_stride": cfg.log_stride,
            "fixed_torque": cfg.fixed_torque,
        },
    }

def compute_elite_stats(elites: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    s0 = np.array([e["params"]["o_solref"][0] for e in elites], dtype=float)
    s1 = np.array([e["params"]["o_solref"][1] for e in elites], dtype=float)

    mean = {
        "o_solref0": float(np.mean(s0)),
        "o_solref1": float(np.mean(s1)),
    }
    std = {
        "o_solref0": float(np.std(s0) + 1e-9),
        "o_solref1": float(np.std(s1) + 1e-6),
    }
    return mean, std

def main(cfg: OptimConfig) -> None:
    out_dir = Path(cfg.out_dir) / cfg.exp_name
    ensure_dir(out_dir)

    jsonl_path = out_dir / "trials.jsonl"
    csv_path = out_dir / "trials.csv"
    csv_fields = [
        "trial_id",
        "reward",
        "reward_hold",
        "reward_drive",
        "termination_hold",
        "termination_drive",
        "slip_m",
        "progress_m",
        "progress_rate_mps",
        "detached_hold",
        "detached_drive",
        "stuck",
        "walltime_s",
        "timestep",
    ]

    rng = np.random.default_rng(cfg.seed)

    # Adaptive random search (CEM-lite)
    n_init = max(10, cfg.n_trials // 5)
    elite_frac = 0.2
    history: List[Dict[str, Any]] = []

    def make_params(p: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(p)
        p["mode"] = cfg.mode
        p["rollout_mode"] = cfg.rollout_mode
        return p

    best = {"reward": -1e18, "trial_id": None, "params": None}

    # Determine number of workers
    n_workers = cfg.n_workers if cfg.n_workers > 1 else 1
    use_parallel = n_workers > 1
    
    if use_parallel:
        print(f"[INFO] Using {n_workers} parallel workers")
        pool = Pool(processes=n_workers)
    
    # Process trials in batches for parallelization
    batch_size = n_workers if use_parallel else 1
    
    for batch_start in range(0, cfg.n_trials, batch_size):
        batch_end = min(batch_start + batch_size, cfg.n_trials)
        batch_trials = []
        
        # Generate parameters for this batch
        for i in range(batch_start, batch_end):
            if i < n_init or len(history) < max(5, int(elite_frac * len(history))):
                p = sample_params_prior(rng)
            else:
                sorted_hist = sorted(history, key=lambda r: r["reward"], reverse=True)
                n_elite = max(2, int(elite_frac * len(sorted_hist)))
                elites = sorted_hist[:n_elite]
                mean, std = compute_elite_stats(elites)
                std = {k: 1.2 * v for k, v in std.items()}
                p = sample_params_refined(rng, mean, std)
            
            p = make_params(p)
            batch_trials.append((i, p, cfg))
        
        # Run batch (parallel or serial)
        if use_parallel:
            results = pool.starmap(run_one_trial, batch_trials)  # ← starmap unpacks tuples
        else:
            results = [run_one_trial(tid, p, c) for tid, p, c in batch_trials]
        
        # Process results
        for rec in results:
            history.append(rec)
            write_jsonl(jsonl_path, rec)
            
            csv_row = {
                "trial_id": rec["trial_id"],
                "reward": rec["reward"],
                "reward_hold": rec.get("reward_hold", None),
                "reward_drive": rec.get("reward_drive", None),
                "termination_hold": rec.get("termination_hold", None),
                "termination_drive": rec.get("termination_drive", None),
                "slip_m": rec.get("slip_m", None),
                "progress_m": rec.get("progress_m", None),
                "progress_rate_mps": rec.get("progress_rate_mps", None),
                "detached_hold": rec.get("detached_hold", None),
                "detached_drive": rec.get("detached_drive", None),
                "stuck": rec.get("stuck", None),
                "walltime_s": rec.get("walltime_s", None),
                "timestep": rec["params"].get("timestep", None),
            }
            append_csv_row(csv_path, csv_fields, csv_row)
            
            if rec["reward"] > best["reward"]:
                best = {"reward": rec["reward"], "trial_id": rec["trial_id"], "params": rec["params"]}
            
            print(f"[trial {rec['trial_id']:04d}] "
                    f"reward={rec['reward']:.3f} "
                    f"hold={rec.get('termination_hold', 'N/A')} "
                    f"drive={rec.get('termination_drive', 'N/A')} "
                    f"best={best['reward']:.3f}")
    
    if use_parallel:
        pool.close()
        pool.join()

    # Save best
    best_path = out_dir / "best.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    
    print(f"\nBest trial: {best['trial_id']} reward={best['reward']:.3f}")
    print(f"Saved logs to: {out_dir}")



# At end of optimizer.py
if __name__ == "__main__":
    cfg = OptimConfig(
        n_trials=300,
        seed=0,
        sim_duration=5.0,
        fixed_torque=0.5,
        rollout_mode="drive",
        mode="sideways",
        n_workers=8,  # ← Change from 1 to 4 (or 8 if you have cores)
        out_dir="logs",
        exp_name="stage3_run1",
    )
    main(cfg)
    
    # Interactive viewer launch
    print("\n" + "="*60)
    print("Optimization complete!")
    print("="*60)
    
    # Load best parameters
    best_path = Path(cfg.out_dir) / cfg.exp_name / "best.json"
    with best_path.open("r", encoding="utf-8") as f:
        best = json.load(f)
    
    print(f"\nBest trial: #{best['trial_id']}")
    print(f"Best reward: {best['reward']:.3f}")
    print("\nBest parameters:")
    for k, v in best['params'].items():
        if k not in ['mode', 'rollout_mode']:
            print(f"  {k}: {v}")
    
    # --- COMPARISON VIEWING ---
    response = input("\nPress ENTER to view BASELINE (original params) run, or 'n' to skip: ").strip().lower()

    if response != 'n':
        print("\n" + "="*60)
        print("BASELINE - HOLD MODE (sideways)")
        print("="*60)
        
        import viewer
        
        # Baseline HOLD test
        viewer.INSPECT_PARAMS = {
            'mode': 'sideways',
            'rollout_mode': 'hold',
        }
        viewer.SIM_DURATION = cfg.sim_duration
        viewer.FIXED_TORQUE = cfg.fixed_torque
        viewer.main()
        
        input("\nPress ENTER to see baseline DRIVE mode...")
        
        print("\n" + "="*60)
        print("BASELINE - DRIVE MODE (drive_up)")
        print("="*60)
        
        # Baseline DRIVE test
        viewer.INSPECT_PARAMS = {
            'mode': 'drive_up',
            'rollout_mode': 'drive',
        }
        viewer.SIM_DURATION = cfg.sim_duration
        viewer.FIXED_TORQUE = cfg.fixed_torque
        viewer.main()
        
        # Now show best
        response2 = input("\nPress ENTER to view BEST OPTIMIZED runs, or 'n' to skip: ").strip().lower()
        
        if response2 != 'n':
            print("\n" + "="*60)
            print("BEST - HOLD MODE (sideways)")
            print("="*60)
            print(f"Hold reward: {best.get('reward_hold', 'N/A')}")
            
            # Best HOLD test
            best_params_hold = dict(best["params"])
            best_params_hold['mode'] = 'sideways'
            best_params_hold['rollout_mode'] = 'hold'
            
            viewer.INSPECT_PARAMS = best_params_hold
            viewer.SIM_DURATION = cfg.sim_duration
            viewer.FIXED_TORQUE = cfg.fixed_torque
            viewer.main()
            
            input("\nPress ENTER to see best DRIVE mode...")
            
            print("\n" + "="*60)
            print("BEST - DRIVE MODE (drive_up)")
            print("="*60)
            print(f"Drive reward: {best.get('reward_drive', 'N/A')}")
            
            # Best DRIVE test
            best_params_drive = dict(best["params"])
            best_params_drive['mode'] = 'drive_up'
            best_params_drive['rollout_mode'] = 'drive'
            
            viewer.INSPECT_PARAMS = best_params_drive
            viewer.SIM_DURATION = cfg.sim_duration
            viewer.FIXED_TORQUE = cfg.fixed_torque
            viewer.main()