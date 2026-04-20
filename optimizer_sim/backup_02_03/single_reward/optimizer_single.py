"""
optimizer_single.py

Simplified parameter optimization for magnetic wall-climbing robot.
Uses ONLY position-based reward for hold mode.

Optimization Strategy:
- Adaptive random search (CEM-lite)
- Single mode: HOLD only (vertical wall)
- Reward: Negative norm of position drift
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
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from multiprocessing import Pool
import xml.etree.ElementTree as ET
from sim_sally_magnet_wall_single import rollout

# Generate base XML file once at startup
print("Generating base XML file...")
env = os.environ.copy()
env["MODE"] = "sideways"
env["OUTPUT_FILE"] = "robot_sally_patched_sideways.xml"
env["NO_AUTOLAUNCH"] = "1"
subprocess.run([sys.executable, "generate_test_magnet_wall_env.py"], env=env, capture_output=True, timeout=30)
print("Base XML file generated.\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OptimConfig:
    """Optimization run configuration."""
    # Optimization settings
    n_trials: int = 100
    seed: int = 0
    n_workers: int = 10  # Parallel workers (>1 for parallelism)
    
    # Rollout settings
    sim_duration: float = 5.0
    settle_time: float = 1.0
    log_stride: int = 10
    
    # Logging
    out_dir: str = "logs"
    exp_name: str = "single_hold"


# =============================================================================
# PARAMETER SAMPLING - Explicit search space definition
# =============================================================================

# Search space definition with explicit distribution types
# Format: (low, high, distribution_type, name)
space = [
    # Magnetic parameters
    (1.48 * 0.9, 1.48 * 1.1, "uniform", "Br"),
    
    # Solver parameters
    (0.0001, 0.0008, "uniform", "solref_timeconst"),
    (10.0, 50.0, "uniform", "solref_dampratio"),
    (0.8, 0.99, "uniform", "solimp_dmin"),
    (0.9, 0.999, "uniform", "solimp_dmax"),
    (1e-4, 1e-2, "log-uniform", "solimp_width"),
    
    # Friction parameters (log-uniform for small values)
    (0.9, 1.0, "uniform", "sliding_friction"),
    (1e-5, 0.1, "log-uniform", "torsional_friction"),
    (1e-5, 0.1, "log-uniform", "rolling_friction"),
    
    # Joint dynamics (log-uniform spans order of magnitude)
    (10.0, 100.0, "log-uniform", "rocker_stiffness"),
    (0.1, 5.0, "log-uniform", "rocker_damping"),
    
    # Control gains (log-uniform)
    (1.0, 50.0, "log-uniform", "wheel_kp"),
    (0.1, 10.0, "log-uniform", "wheel_kv"),
    
    # Magnetic cutoff (log-uniform)
    (0.005, 0.1, "log-uniform", "max_magnetic_distance"),
    
    # Solver iterations (integer, uniform)
    (5, 30, "uniform", "noslip_iterations"),
]

# Fixed o_solimp values that won't be tuned
FIXED_SOLIMP_MIDPOINT = 0.5
FIXED_SOLIMP_POWER = 2.0

PARAM_RANGES = {name: (low, high) for (low, high, _, name) in space}

# Aliases used elsewhere in the script
PARAM_RANGES.update({
    "o_solref_0": PARAM_RANGES["solref_timeconst"],
    "o_solref_1": PARAM_RANGES["solref_dampratio"],
    "wheel_friction_slide": PARAM_RANGES["sliding_friction"],
    "wheel_friction_torsion": PARAM_RANGES["torsional_friction"],
    "wheel_friction_roll": PARAM_RANGES["rolling_friction"],
})

# FIXED o_solimp values (must be numeric)
SOLIMP_DMIN_FIXED = 0.895    # midpoint of 0.8–0.99
SOLIMP_DMAX_FIXED = 0.9495   # midpoint of 0.9–0.999
SOLIMP_WIDTH_FIXED = 1e-3    # geometric mean of 1e-4–1e-2

PARAM_RANGES["o_solimp"] = [
    SOLIMP_DMIN_FIXED,
    SOLIMP_DMAX_FIXED,
    SOLIMP_WIDTH_FIXED,
    FIXED_SOLIMP_MIDPOINT,
    FIXED_SOLIMP_POWER,
]


def _clip(x: float, lo: float, hi: float) -> float:
    """Clip value to range [lo, hi]."""
    return float(min(max(x, lo), hi))


def sample_from_space(rng: np.random.Generator, param_def: tuple) -> float:
    """
    Sample a parameter from its search space definition.
    
    Args:
        rng: Random number generator
        param_def: Tuple of (low, high, distribution, name)
        
    Returns:
        Sampled value
    """
    low, high, distribution, name = param_def
    
    if distribution == "uniform":
        return float(rng.uniform(low, high))
    elif distribution == "log-uniform":
        return float(10 ** rng.uniform(np.log10(low), np.log10(high)))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def get_param_by_name(name: str) -> tuple:
    """Get parameter definition from space by name."""
    for param_def in space:
        if param_def[3] == name:
            return param_def
    raise KeyError(f"Parameter {name} not found in search space")


# =============================================================================
# PARAMETER GENERATION
# =============================================================================

def sample_default_params() -> Dict[str, Any]:
    """Sample parameters from midpoint of search space."""
    params = {}
    
    for low, high, distribution, name in space:
        if distribution == "uniform":
            params[name] = (low + high) / 2.0
        elif distribution == "log-uniform":
            params[name] = 10 ** ((np.log10(low) + np.log10(high)) / 2.0)
    
    return params


def sample_random_params(rng: np.random.Generator) -> Dict[str, Any]:
    """Sample completely random parameters from search space."""
    params = {}
    
    for param_def in space:
        name = param_def[3]
        params[name] = sample_from_space(rng, param_def)
    
    return params


def sample_refined_params(
    rng: np.random.Generator, elite_params: List[Dict[str, Any]], noise_std: float = 0.15
) -> Dict[str, Any]:
    """
    Sample parameters from distribution around elite set.
    
    Args:
        rng: Random number generator
        elite_params: List of top-performing parameter sets
        noise_std: Standard deviation of noise (as fraction of range)
        
    Returns:
        New parameter set
    """
    if not elite_params:
        return sample_random_params(rng)
    
    # Compute mean of elite parameters
    mean_params = {}
    for param_def in space:
        name = param_def[3]
        values = [p[name] for p in elite_params if name in p]
        if values:
            mean_params[name] = np.mean(values)
        else:
            mean_params[name] = sample_from_space(rng, param_def)
    
    # Add noise
    params = {}
    for low, high, distribution, name in space:
        mean = mean_params[name]
        
        if distribution == "uniform":
            # Linear space noise
            range_size = high - low
            noise = rng.normal(0, noise_std * range_size)
            params[name] = _clip(mean + noise, low, high)
        elif distribution == "log-uniform":
            # Log space noise
            log_mean = np.log10(mean)
            log_range = np.log10(high) - np.log10(low)
            noise = rng.normal(0, noise_std * log_range)
            params[name] = _clip(10 ** (log_mean + noise), low, high)
    
    return params


def build_rollout_params(sampled: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert sampled parameters to rollout function format.
    
    Args:
        sampled: Sampled parameters from search space
        
    Returns:
        Parameters in rollout function format
    """
    params = {}
    
    # Magnetic parameters
    if "Br" in sampled:
        params["Br"] = sampled["Br"]
    
    # Solver parameters
    if "solref_timeconst" in sampled and "solref_dampratio" in sampled:
        params["o_solref"] = [sampled["solref_timeconst"], sampled["solref_dampratio"]]
    
    if "solimp_dmin" in sampled and "solimp_dmax" in sampled and "solimp_width" in sampled:
        params["o_solimp"] = [
            sampled["solimp_dmin"],
            sampled["solimp_dmax"],
            sampled["solimp_width"],
            FIXED_SOLIMP_MIDPOINT,
            FIXED_SOLIMP_POWER,
        ]
    
    # Friction parameters
    if "sliding_friction" in sampled and "torsional_friction" in sampled and "rolling_friction" in sampled:
        params["wheel_friction"] = [
            sampled["sliding_friction"],
            sampled["torsional_friction"],
            sampled["rolling_friction"],
        ]
    
    # Joint dynamics
    if "rocker_stiffness" in sampled:
        params["rocker_stiffness"] = sampled["rocker_stiffness"]
    if "rocker_damping" in sampled:
        params["rocker_damping"] = sampled["rocker_damping"]
    
    # Control gains
    if "wheel_kp" in sampled:
        params["wheel_kp"] = sampled["wheel_kp"]
    if "wheel_kv" in sampled:
        params["wheel_kv"] = sampled["wheel_kv"]
    
    # Magnetic cutoff
    if "max_magnetic_distance" in sampled:
        params["max_magnetic_distance"] = sampled["max_magnetic_distance"]
    
    # Solver iterations
    if "noslip_iterations" in sampled:
        params["noslip_iterations"] = int(sampled["noslip_iterations"])
    
    return params


# =============================================================================
# LOGGING
# =============================================================================

def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append single record to JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    """Append single row to CSV file."""
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# TRIAL EXECUTION
# =============================================================================

def run_one_trial(
    trial_id: int, params: Dict[str, Any], cfg: OptimConfig
) -> Dict[str, Any]:
    """
    Run single optimization trial.
    
    Args:
        trial_id: Trial identifier
        params: Parameters to test
        cfg: Optimization configuration
        
    Returns:
        Trial result record
    """
    start_time = time.time()
    
    # Build rollout parameters
    rollout_params = build_rollout_params(params)
    
    # Add XML file override
    rollout_params["_xml_file_override"] = f"robot_sally_trial_{trial_id}_sideways.xml"
    
    # Run rollout
    try:
        result = rollout(
            params=rollout_params,
            sim_duration=cfg.sim_duration,
            settle_time=cfg.settle_time,
            log_stride=cfg.log_stride,
        )
        
        reward = result.summary["reward"]
        termination = result.termination
        
    except Exception as e:
        print(f"[ERROR] Trial {trial_id} crashed: {e}")
        reward = -1000.0
        termination = "crashed"
        result = None
    
    walltime = time.time() - start_time
    
    # Build trial record
    record = {
        "trial_id": trial_id,
        "reward": reward,
        "termination": termination,
        "walltime_s": walltime,
        "params": params,
    }
    
    # Add summary metrics if available
    if result is not None:
        record.update({
            "position_drift_m": result.summary.get("position_drift_m", None),
            "max_position_drift_m": result.summary.get("max_position_drift_m", None),
            "avg_magnetic_force": result.summary.get("avg_total_magnetic_force", None),
        })
    
    return record


# =============================================================================
# MAIN OPTIMIZATION LOOP
# =============================================================================

def main(cfg: OptimConfig) -> None:
    """
    Run parameter optimization.
    
    Args:
        cfg: Optimization configuration
    """
    rng = np.random.default_rng(cfg.seed)
    
    # Setup output directory
    out_dir = Path(cfg.out_dir) / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = out_dir / "trials.jsonl"
    csv_path = out_dir / "trials.csv"
    
    # CSV fields
    csv_fields = [
        "trial_id",
        "reward",
        "termination",
        "position_drift_m",
        "max_position_drift_m",
        "avg_magnetic_force",
        "walltime_s",
    ]
    
    # Initialize tracking
    history = []
    best = {"reward": -np.inf, "trial_id": -1, "params": {}}
    
    # Parallel execution setup
    use_parallel = cfg.n_workers > 1
    if use_parallel:
        pool = Pool(processes=cfg.n_workers)
        batch_size = cfg.n_workers
    else:
        batch_size = 1
    
    print(f"Starting optimization: {cfg.n_trials} trials")
    print(f"Parallel workers: {cfg.n_workers}")
    print(f"Output directory: {out_dir}\n")
    
    # Run trials in batches
    for batch_start in range(0, cfg.n_trials, batch_size):
        batch_end = min(batch_start + batch_size, cfg.n_trials)
        batch_trials = []
        
        # Generate parameters and XML files for batch
        for i in range(batch_start, batch_end):
            # Sample parameters
            if i == 0:
                # First trial: use default/midpoint parameters
                p = sample_default_params()
            elif i < 20:
                # Early exploration: random sampling
                p = sample_random_params(rng)
            else:
                # Refinement: sample around elite set
                elite = sorted(history, key=lambda x: x["reward"], reverse=True)[:10]
                elite_params = [rec["params"] for rec in elite]
                p = sample_refined_params(rng, elite_params, noise_std=0.15)
            
            # Generate trial-specific XML with actuator gains
            base_xml = "robot_sally_patched_sideways.xml"
            
            if not os.path.exists(base_xml):
                raise FileNotFoundError(f"Base XML not found: {base_xml}")
            
            tree = ET.parse(base_xml)
            root = tree.getroot()
            
            # Update actuator gains
            for actuator in root.find("actuator").findall("position"):
                if actuator.get("name") in ["BR_wheel_motor", "FR_wheel_motor", "BL_wheel_motor", "FL_wheel_motor"]:
                    actuator.set("kp", str(p.get("wheel_kp", 10.0)))
                    actuator.set("kv", str(p.get("wheel_kv", 1.0)))
            
            # Write modified XML
            modified_xml = f"robot_sally_trial_{i}_sideways.xml"
            tree.write(modified_xml, encoding="utf-8", xml_declaration=True)
            
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
                "termination": rec.get("termination", None),
                "position_drift_m": rec.get("position_drift_m", None),
                "max_position_drift_m": rec.get("max_position_drift_m", None),
                "avg_magnetic_force": rec.get("avg_magnetic_force", None),
                "walltime_s": rec.get("walltime_s", None),
            }
            append_csv_row(csv_path, csv_fields, csv_row)
            
            # Update best
            if rec["reward"] > best["reward"]:
                best = {"reward": rec["reward"], "trial_id": rec["trial_id"], "params": rec["params"]}
            
            # Progress update
            print(f"[trial {rec['trial_id']:04d}] "
                  f"reward={rec['reward']:.6f} "
                  f"drift={rec.get('position_drift_m', 0):.6f}m "
                  f"term={rec.get('termination', 'N/A')} "
                  f"best={best['reward']:.6f}")
        
        # Cleanup: Delete temporary XML files for this batch
        for i in range(batch_start, batch_end):
            xml_file = f"robot_sally_trial_{i}_sideways.xml"
            scene_file = f"scene_robot_sally_trial_{i}_sideways.xml"
            try:
                if os.path.exists(xml_file):
                    os.remove(xml_file)
                if os.path.exists(scene_file):
                    os.remove(scene_file)
            except Exception:
                pass
    
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
        "parameters": best["params"],
        "config": {
            "sim_duration": cfg.sim_duration,
            "settle_time": cfg.settle_time,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": cfg.exp_name,
    }
    
    with replay_params_path.open("w", encoding="utf-8") as f:
        json.dump(replay_data, f, indent=2)
    
    print(f"\nBest trial: {best['trial_id']} reward={best['reward']:.6f}")
    print(f"Saved logs to: {out_dir}")
    print(f"Replay parameters saved to: {replay_params_path}")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_optimization_progress(cfg: OptimConfig) -> None:
    """Plot optimization progress over trials."""
    import matplotlib.pyplot as plt
    
    csv_path = Path(cfg.out_dir) / cfg.exp_name / "trials.csv"
    
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return
    
    # Load data
    data = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    trials = [int(row["trial_id"]) for row in data]
    rewards = [float(row["reward"]) for row in data]
    drifts = [float(row["position_drift_m"]) if row["position_drift_m"] else 0 for row in data]
    
    # Running best
    running_best = []
    best_so_far = -np.inf
    for r in rewards:
        best_so_far = max(best_so_far, r)
        running_best.append(best_so_far)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Reward plot
    axes[0].plot(trials, rewards, 'o', alpha=0.3, label='Trial reward')
    axes[0].plot(trials, running_best, 'r-', linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Optimization Progress: Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Position drift plot
    axes[1].plot(trials, drifts, 'o', alpha=0.3, label='Position drift')
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('Position Drift (m)')
    axes[1].set_title('Optimization Progress: Position Drift')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = Path(cfg.out_dir) / cfg.exp_name / "optimization_progress.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")
    plt.close()


# =============================================================================
# RESULTS PRINTING
# =============================================================================

def print_results(best: Dict[str, Any]) -> None:
    """Print optimization results."""
    print("\nBEST PARAMETERS:")
    print("=" * 60)
    
    params = best["params"]
    
    print(f"Trial ID: {best['trial_id']}")
    print(f"Reward: {best['reward']:.6f}")
    print()
    
    # Group parameters by category
    categories = {
        "Magnetic": ["Br"],
        "Solver": ["solref_timeconst", "solref_dampratio", "solimp_dmin", "solimp_dmax", "solimp_width", "noslip_iterations"],
        "Friction": ["sliding_friction", "torsional_friction", "rolling_friction"],
        "Joint Dynamics": ["rocker_stiffness", "rocker_damping"],
        "Control": ["wheel_kp", "wheel_kv"],
        "Magnetic Cutoff": ["max_magnetic_distance"],
    }
    
    for category, param_names in categories.items():
        print(f"{category}:")
        for name in param_names:
            if name in params:
                value = params[name]
                print(f"  {name:25s} = {value:.6e}")
        print()


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
    import viewer_single
    
    # Baseline comparison
    response = input("\nPress ENTER to view BASELINE (default params), or 'n' to skip: ").strip().lower()
    
    if response != 'n':
        print("\n" + "="*60)
        print("BASELINE - HOLD MODE (default parameters)")
        print("="*60)
        
        viewer_single.INSPECT_PARAMS = {"mode": "sideways"}  # ADD MODE!
        viewer_single.SIM_DURATION = cfg.sim_duration
        viewer_single.SETTLE_TIME = cfg.settle_time
        viewer_single.main()
            
    # Optimized comparison
    response2 = input("\nPress ENTER to view BEST OPTIMIZED run, or 'n' to skip: ").strip().lower()
    
    if response2 != 'n':
        print("\n" + "="*60)
        print("BEST - HOLD MODE (optimized parameters)")
        print("="*60)
        print(f"Reward: {best.get('reward', 'N/A'):.6f}")
        print(f"Position drift: {best['params'].get('position_drift_m', 'N/A')} m")
        
        best_params = build_rollout_params(best["params"])
        best_params["mode"] = "sideways"  # ADD MODE!

        viewer_single.INSPECT_PARAMS = best_params
        viewer_single.SIM_DURATION = cfg.sim_duration
        viewer_single.SETTLE_TIME = cfg.settle_time
        viewer_single.main()


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
    
    # Plot optimization progress
    print("\nGenerating plots...")
    plot_optimization_progress(cfg)
    
    # Launch viewer
    launch_viewer(cfg, best)