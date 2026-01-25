"""
optimizer.py

Black-box parameter optimization for magnetic wall-climbing robot.

Optimization Strategy:
- Adaptive random search (CEM-lite: Cross-Entropy Method simplified)
- Runs rollouts in BOTH hold and drive modes
- Combined reward = min(hold_reward, drive_reward) - both must succeed
- Logs all trials to JSONL + CSV for analysis
- Supports parallel rollouts via multiprocessing

Output:
- logs/{exp_name}/trials.jsonl - Full trial records (one per line)
- logs/{exp_name}/trials.csv - Summary metrics for plotting
- logs/{exp_name}/best.json - Best parameters found
- logs/{exp_name}/replay_params.json - Best params in replay format
"""

from __future__ import annotations

import json
import time
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from multiprocessing import Pool

from sim_sally_magnet_wall import rollout


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OptimConfig:
    """Optimization run configuration."""
    # Optimization settings
    n_trials: int = 300
    seed: int = 0
    n_workers: int = 10  # Parallel workers (>1 for parallelism)
    
    # Rollout settings
    sim_duration: float = 5.0
    settle_time: float = 1.0
    log_stride: int = 10
    fixed_torque: float = 0.5  # Unused, kept for compatibility
    
    # Test modes
    rollout_mode: str = "drive"  # "drive" or "hold"
    mode: str = "sideways"  # "sideways" or "drive_up" for XML generation
    
    # Logging
    out_dir: str = "logs"
    exp_name: str = "stage3"


# =============================================================================
# PARAMETER SAMPLING
# =============================================================================

def _clip(x: float, lo: float, hi: float) -> float:
    """Clip value to range [lo, hi]."""
    return float(min(max(x, lo), hi))


def sample_params_prior(rng: np.random.Generator) -> Dict[str, Any]:
    """
    Sample parameters uniformly from prior ranges.
    Used for initial exploration.
    
    Args:
        rng: Random number generator
        
    Returns:
        Dictionary of sampled parameters
    """
    o_solref = [rng.uniform(1e-4, 8e-4), rng.uniform(10.0, 50.0)]
    o_solimp = [0.99, 0.99, 0.001, 0.5, 2.0]  # Fixed
    
    wheel_friction = [
        rng.uniform(0.90, 1.0),   # Sliding
        rng.uniform(0.05, 0.1),   # Torsional
        rng.uniform(0.05, 0.1),   # Rolling
    ]
    
    rocker_stiffness = rng.uniform(10.0, 100.0)
    rocker_damping = rng.uniform(0.1, 5.0)
    Br = rng.uniform(1.48 * 0.9, 1.48 * 1.1)
    wheel_kp = rng.uniform(1.0, 50.0)
    wheel_kv = rng.uniform(0.1, 10.0)
    max_magnetic_distance = rng.uniform(0.005, 0.1)
    
    return {
        "Br": float(Br),
        "o_solref": [float(o_solref[0]), float(o_solref[1])],
        "o_solimp": [float(x) for x in o_solimp],
        "wheel_friction": [float(x) for x in wheel_friction],
        "rocker_stiffness": float(rocker_stiffness),
        "rocker_damping": float(rocker_damping),
        "wheel_kp": float(wheel_kp),
        "wheel_kv": float(wheel_kv),
        "max_magnetic_distance": float(max_magnetic_distance),
    }


def sample_params_refined(
    rng: np.random.Generator,
    mean: Dict[str, float],
    std: Dict[str, float],
) -> Dict[str, Any]:
    """
    Sample parameters using Gaussian around elite mean.
    Used after initial exploration to refine promising regions.
    
    Currently only adapts o_solref; other params still sampled uniformly.
    
    Args:
        rng: Random number generator
        mean: Mean values from elite samples
        std: Standard deviations from elite samples
        
    Returns:
        Dictionary of sampled parameters
    """
    # Adaptive sampling for o_solref
    solref0 = rng.normal(mean["o_solref0"], std["o_solref0"])
    solref0 = _clip(solref0, 1e-4, 8e-4)
    
    solref1 = rng.normal(mean["o_solref1"], std["o_solref1"])
    solref1 = _clip(solref1, 10.0, 50.0)
    
    # Other parameters: uniform sampling (could be extended to adaptive)
    o_solimp = [0.99, 0.99, 0.001, 0.5, 2.0]
    Br = rng.uniform(1.48 * 0.9, 1.48 * 1.1)
    
    wheel_friction = [
        rng.uniform(0.90, 1.0),
        rng.uniform(0.05, 0.1),
        rng.uniform(0.05, 0.1),
    ]
    
    rocker_stiffness = rng.uniform(10.0, 100.0)
    rocker_damping = rng.uniform(0.1, 5.0)
    wheel_kp = rng.uniform(1.0, 50.0)
    wheel_kv = rng.uniform(0.1, 10.0)
    max_magnetic_distance = rng.uniform(0.005, 0.02)
    
    return {
        "Br": float(Br),
        "o_solref": [float(solref0), float(solref1)],
        "o_solimp": [float(x) for x in o_solimp],
        "wheel_friction": [float(x) for x in wheel_friction],
        "rocker_stiffness": float(rocker_stiffness),
        "rocker_damping": float(rocker_damping),
        "wheel_kp": float(wheel_kp),
        "wheel_kv": float(wheel_kv),
        "max_magnetic_distance": float(max_magnetic_distance),
    }


def compute_elite_stats(elites: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute mean and std of elite samples for adaptive sampling.
    
    Args:
        elites: List of elite trial records
        
    Returns:
        (mean, std) dictionaries for o_solref parameters
    """
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


# =============================================================================
# TRIAL EXECUTION
# =============================================================================

def run_one_trial(trial_id: int, params: Dict[str, Any], cfg: OptimConfig) -> Dict[str, Any]:
    """
    Run single optimization trial in BOTH hold and drive modes.
    
    Args:
        trial_id: Trial number
        params: Parameters to test
        cfg: Optimization configuration
        
    Returns:
        Trial record with results from both modes
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
    
    walltime = time.time() - t0
    
    # Extract summaries
    summary_hold = dict(result_hold.summary or {})
    summary_drive = dict(result_drive.summary or {})
    
    reward_hold = float(summary_hold.get("reward", -1e9))
    reward_drive = float(summary_drive.get("reward", -1e9))
    
    # Combined reward: both tests must succeed (use worse of the two)
    combined_reward = min(reward_hold, reward_drive)
    
    return {
        "trial_id": int(trial_id),
        "time_unix": float(time.time()),
        "walltime_s": float(walltime),
        "params": params,
        
        # Hold mode results
        "termination_hold": str(result_hold.termination),
        "reward_hold": reward_hold,
        "slip_m": summary_hold.get("slip_m", 0.0),
        "detached_hold": summary_hold.get("detached", False),
        "contact_percentage_hold": summary_hold.get("contact_percentage", None),
        "detachment_fraction_hold": summary_hold.get("detachment_fraction", None),
        
        # Drive mode results
        "termination_drive": str(result_drive.termination),
        "reward_drive": reward_drive,
        "progress_m": summary_drive.get("progress_m", 0.0),
        "progress_rate_mps": summary_drive.get("progress_rate_mps", 0.0),
        "detached_drive": summary_drive.get("detached", False),
        "stuck": summary_drive.get("stuck", False),
        "contact_percentage_drive": summary_drive.get("contact_percentage", None),
        "detachment_fraction_drive": summary_drive.get("detachment_fraction", None),
        
        # Combined
        "reward": combined_reward,
        
        "rollout_settings": {
            "sim_duration": cfg.sim_duration,
            "settle_time": cfg.settle_time,
            "log_stride": cfg.log_stride,
            "fixed_torque": cfg.fixed_torque,
        },
    }


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append record to JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    """Append row to CSV file, creating header if new."""
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow(row)


def print_parameter_info():
    """Print table of tunable parameters and their search ranges."""
    print("\n" + "="*80)
    print("TUNABLE PARAMETERS")
    print("="*80)
    
    params_info = [
        ("Br", "Magnetic remanence", "Linear", "1.332 - 1.628 T", "±10% of 1.48 T"),
        ("o_solref[0]", "Contact timeconst", "Linear", "0.0001 - 0.0008 s", "Solver stiffness"),
        ("o_solref[1]", "Contact dampratio", "Linear", "10.0 - 50.0", "Solver damping"),
        ("o_solimp", "Contact impedance", "Fixed", "[0.99, 0.99, 0.001, 0.5, 2]", "Force ramp-up"),
        ("wheel_friction[0]", "Sliding friction", "Linear", "0.90 - 1.00", "Tangential slip"),
        ("wheel_friction[1]", "Torsional friction", "Linear", "0.05 - 0.10", "Spinning resistance"),
        ("wheel_friction[2]", "Rolling friction", "Linear", "0.05 - 0.10", "Rolling resistance"),
        ("rocker_stiffness", "Rocker joint stiffness", "Linear", "10.0 - 100.0 N·m/rad", "Suspension compliance"),
        ("rocker_damping", "Rocker joint damping", "Linear", "0.1 - 5.0 N·m·s/rad", "Suspension damping"),
        ("wheel_kp", "Wheel position gain", "Linear", "1.0 - 50.0", "P controller gain"),
        ("wheel_kv", "Wheel velocity gain", "Linear", "0.1 - 10.0", "D controller gain"),
        ("max_magnetic_distance", "Magnetic cutoff", "Linear", "0.005 - 0.02 m", "0.5cm - 2cm range"),
    ]
    
    print(f"{'Parameter':<25} {'Sampling':<10} {'Range':<30} {'Description':<30}")
    print("-"*105)
    for name, desc, sampling, range_str, notes in params_info:
        print(f"{name:<25} {sampling:<10} {range_str:<30} {notes:<30}")
    
    print("="*105)
    print(f"Total tunable parameters: 11 (o_solimp is fixed)")
    print(f"Search space dimensionality: 13 (counting vector elements)")
    print("="*105 + "\n")


def print_results(best: Dict[str, Any]):
    """Print final optimization results with parameter table."""
    print("\n" + "="*110)
    print("OPTIMIZATION RESULTS")
    print("="*110)
    print(f"Best trial: #{best['trial_id']}")
    print(f"Best reward: {best['reward']:.3f}")
    print("="*110)
    print("\nOPTIMIZED PARAMETERS")
    print("="*110)
    
    # Parameter info with search bounds
    params_info = {
        "Br": ("Magnetic remanence (T)", "1.332 - 1.628"),
        "o_solref": ("Contact solver [timeconst, dampratio]", "[0.0001-0.0008, 10-50]"),
        "o_solimp": ("Contact impedance", "Fixed"),
        "wheel_friction": ("Wheel friction [slide, torsion, roll]", "[0.9-1.0, 0.05-0.1, 0.05-0.1]"),
        "rocker_stiffness": ("Rocker stiffness (N·m/rad)", "10.0 - 100.0"),
        "rocker_damping": ("Rocker damping (N·m·s/rad)", "0.1 - 5.0"),
        "wheel_kp": ("Wheel P gain", "1.0 - 50.0"),
        "wheel_kv": ("Wheel D gain", "0.1 - 10.0"),
        "max_magnetic_distance": ("Magnetic cutoff (m)", "0.005 - 0.02"),
    }
    
    print(f"{'Parameter':<25} {'Optimized Value':<30} {'Search Range':<25} {'Description':<30}")
    print("-"*110)
    
    for k, v in best['params'].items():
        if k not in ['mode', 'rollout_mode']:
            if k in params_info:
                desc, search_range = params_info[k]
                # Format value
                if isinstance(v, list):
                    val_str = "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
                elif isinstance(v, float):
                    val_str = f"{v:.6f}"
                else:
                    val_str = str(v)
                
                print(f"{k:<25} {val_str:<30} {search_range:<25} {desc:<30}")
            else:
                print(f"{k:<25} {str(v):<30}")
    
    print("="*110 + "\n")


# =============================================================================
# MAIN OPTIMIZATION LOOP
# =============================================================================

def main(cfg: OptimConfig) -> None:
    """
    Run optimization loop.
    
    Args:
        cfg: Optimization configuration
    """
    # Print parameter search space
    print_parameter_info()
    
    # Setup logging
    out_dir = Path(cfg.out_dir) / cfg.exp_name
    ensure_dir(out_dir)
    
    jsonl_path = out_dir / "trials.jsonl"
    csv_path = out_dir / "trials.csv"
    csv_fields = [
        "trial_id", "reward", "reward_hold", "reward_drive",
        "termination_hold", "termination_drive",
        "slip_m", "contact_percentage_hold", "detachment_fraction_hold", "detached_hold",
        "progress_m", "progress_rate_mps", "contact_percentage_drive", 
        "detachment_fraction_drive", "detached_drive", "stuck",
        "walltime_s",
    ]
    
    # Initialize RNG and optimization state
    rng = np.random.default_rng(cfg.seed)
    n_init = max(10, cfg.n_trials // 5)  # Initial random trials
    elite_frac = 0.2  # Top 20% are elites
    history: List[Dict[str, Any]] = []
    best = {"reward": -1e18, "trial_id": None, "params": None}
    
    # Setup parallelism
    n_workers = cfg.n_workers if cfg.n_workers > 1 else 1
    use_parallel = n_workers > 1
    
    if use_parallel:
        print(f"[INFO] Using {n_workers} parallel workers")
        pool = Pool(processes=n_workers)
    
    # Run optimization in batches
    batch_size = n_workers if use_parallel else 1
    
    for batch_start in range(0, cfg.n_trials, batch_size):
        batch_end = min(batch_start + batch_size, cfg.n_trials)
        batch_trials = []
        
        # Generate parameters for batch
        for i in range(batch_start, batch_end):
            # Initial exploration or adaptive refinement?
            if i < n_init or len(history) < max(5, int(elite_frac * len(history))):
                p = sample_params_prior(rng)
            else:
                # Adaptive sampling around elites
                sorted_hist = sorted(history, key=lambda r: r["reward"], reverse=True)
                n_elite = max(2, int(elite_frac * len(sorted_hist)))
                elites = sorted_hist[:n_elite]
                mean, std = compute_elite_stats(elites)
                std = {k: 1.2 * v for k, v in std.items()}  # Inflate std to maintain exploration
                p = sample_params_refined(rng, mean, std)
            
            # Add mode metadata
            p = dict(p)
            p["mode"] = cfg.mode
            p["rollout_mode"] = cfg.rollout_mode
            
            batch_trials.append((i, p, cfg))
        
        # Execute batch
        if use_parallel:
            results = pool.starmap(run_one_trial, batch_trials)
        else:
            results = [run_one_trial(tid, p, c) for tid, p, c in batch_trials]
        
        # Process results
        for rec in results:
            history.append(rec)
            write_jsonl(jsonl_path, rec)
            
            # CSV row
            csv_row = {
                "trial_id": rec.get("trial_id", None),
                "reward": rec.get("reward", None),
                "reward_hold": rec.get("reward_hold", None),
                "reward_drive": rec.get("reward_drive", None),
                "termination_hold": rec.get("termination_hold", None),
                "termination_drive": rec.get("termination_drive", None),
                "slip_m": rec.get("slip_m", None),
                "contact_percentage_hold": rec.get("contact_percentage_hold", None),
                "detachment_fraction_hold": rec.get("detachment_fraction_hold", None),
                "detached_hold": rec.get("detached_hold", None),
                "progress_m": rec.get("progress_m", None),
                "progress_rate_mps": rec.get("progress_rate_mps", None),
                "contact_percentage_drive": rec.get("contact_percentage_drive", None),
                "detachment_fraction_drive": rec.get("detachment_fraction_drive", None),
                "detached_drive": rec.get("detached_drive", None),
                "stuck": rec.get("stuck", None),
                "walltime_s": rec.get("walltime_s", None),
            }
            append_csv_row(csv_path, csv_fields, csv_row)
            
            # Update best
            if rec["reward"] > best["reward"]:
                best = {"reward": rec["reward"], "trial_id": rec["trial_id"], "params": rec["params"]}
            
            # Progress update
            print(f"[trial {rec['trial_id']:04d}] "
                  f"reward={rec['reward']:.3f} "
                  f"hold={rec.get('termination_hold', 'N/A')} "
                  f"drive={rec.get('termination_drive', 'N/A')} "
                  f"best={best['reward']:.3f}")
    
    if use_parallel:
        pool.close()
        pool.join()
    
    # Save best parameters
    best_path = out_dir / "best.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    
    # Save replay parameters
    replay_params_path = out_dir / "replay_params.json"
    replay_data = {
        "trial_id": best["trial_id"],
        "reward": best["reward"],
        "reward_hold": best.get("reward_hold"),
        "reward_drive": best.get("reward_drive"),
        "parameters": best["params"],
        "config": {
            "sim_duration": cfg.sim_duration,
            "settle_time": cfg.settle_time,
            "fixed_torque": cfg.fixed_torque,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": cfg.exp_name,
    }
    
    with replay_params_path.open("w", encoding="utf-8") as f:
        json.dump(replay_data, f, indent=2)
    
    print(f"\nBest trial: {best['trial_id']} reward={best['reward']:.3f}")
    print(f"Saved logs to: {out_dir}")
    print(f"Replay parameters saved to: {replay_params_path}")


# =============================================================================
# INTERACTIVE VIEWER
# =============================================================================

def launch_viewer(cfg: OptimConfig, best: Dict[str, Any]):
    """
    Launch interactive viewer to compare baseline vs optimized parameters.
    
    Args:
        cfg: Optimization configuration
        best: Best parameters found
    """
    import viewer
    
    # Baseline comparison
    response = input("\nPress ENTER to view BASELINE (default params), or 'n' to skip: ").strip().lower()
    
    if response != 'n':
        print("\n" + "="*60)
        print("BASELINE - HOLD MODE (sideways)")
        print("="*60)
        
        viewer.INSPECT_PARAMS = {'mode': 'sideways', 'rollout_mode': 'hold'}
        viewer.SIM_DURATION = cfg.sim_duration
        viewer.FIXED_TORQUE = cfg.fixed_torque
        viewer.main()
        
        input("\nPress ENTER to see baseline DRIVE mode...")
        
        print("\n" + "="*60)
        print("BASELINE - DRIVE MODE (drive_up)")
        print("="*60)
        
        viewer.INSPECT_PARAMS = {'mode': 'drive_up', 'rollout_mode': 'drive'}
        viewer.SIM_DURATION = cfg.sim_duration
        viewer.FIXED_TORQUE = cfg.fixed_torque
        viewer.main()
    
    # Optimized comparison
    response2 = input("\nPress ENTER to view BEST OPTIMIZED runs, or 'n' to skip: ").strip().lower()
    
    if response2 != 'n':
        print("\n" + "="*60)
        print("BEST - HOLD MODE (sideways)")
        print("="*60)
        print(f"Hold reward: {best.get('reward_hold', 'N/A')}")
        
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
        
        best_params_drive = dict(best["params"])
        best_params_drive['mode'] = 'drive_up'
        best_params_drive['rollout_mode'] = 'drive'
        
        viewer.INSPECT_PARAMS = best_params_drive
        viewer.SIM_DURATION = cfg.sim_duration
        viewer.FIXED_TORQUE = cfg.fixed_torque
        viewer.main()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    cfg = OptimConfig()
    
    # Run optimization
    main(cfg)
    
    # Load best results
    best_path = Path(cfg.out_dir) / cfg.exp_name / "best.json"
    with best_path.open("r", encoding="utf-8") as f:
        best = json.load(f)
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print_results(best)
    
    # Launch viewer
    launch_viewer(cfg, best)