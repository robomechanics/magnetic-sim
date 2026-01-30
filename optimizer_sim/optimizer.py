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
    n_workers: int = 10 # Parallel workers (>1 for parallelism)
    
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

# FIXED o_solimp values (must be numeric, because refined sampling casts to float)
SOLIMP_DMIN_FIXED  = 0.895    # midpoint of 0.8–0.99
SOLIMP_DMAX_FIXED  = 0.9495   # midpoint of 0.9–0.999
SOLIMP_WIDTH_FIXED = 1e-3     # geometric mean of 1e-4–1e-2 (log-uniform mid)

PARAM_RANGES["o_solimp"] = [
    SOLIMP_DMIN_FIXED,
    SOLIMP_DMAX_FIXED,
    SOLIMP_WIDTH_FIXED,
    FIXED_SOLIMP_MIDPOINT,
    FIXED_SOLIMP_POWER,
]
def sampling_label_for_key(param_key: str) -> str:
    """
    Manual: use the distribution type from `space`.
    No heuristics, no auto-detection from (low, high).
    """
    # Map derived keys back to base keys in `space`
    alias = {
        "o_solref_0": "solref_timeconst",
        "o_solref_1": "solref_dampratio",
        "wheel_friction_slide": "sliding_friction",
        "wheel_friction_torsion": "torsional_friction",
        "wheel_friction_roll": "rolling_friction",
    }
    k = alias.get(param_key, param_key)

    for low, high, dist, name in space:
        if name == k:
            return "Log" if dist == "log-uniform" else "Uniform"

    # Composite/fixed that are not in `space`
    if param_key == "o_solimp":
        return "Fixed"
    return "Fixed/Derived"


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

def format_range(param_name: str) -> str:
    """Format parameter range as string from new space definition."""
    try:
        param_def = get_param_by_name(param_name)
        low, high, dist, _ = param_def
        dist_label = "LOG" if dist == "log-uniform" else "Linear"
        return f"{low} - {high} ({dist_label})"
    except KeyError:
        # Fallback for composite parameters
        if param_name == "o_solref":
            tc = get_param_by_name("solref_timeconst")
            dr = get_param_by_name("solref_dampratio")
            return f"[{tc[0]}-{tc[1]}, {dr[0]}-{dr[1]}]"
        elif param_name == "wheel_friction":
            sf = get_param_by_name("sliding_friction")
            tf = get_param_by_name("torsional_friction")
            rf = get_param_by_name("rolling_friction")
            return f"[{sf[0]}-{sf[1]}, {tf[0]}-{tf[1]}, {rf[0]}-{rf[1]}]"
        elif param_name == "o_solimp":
            dmin = get_param_by_name("solimp_dmin")
            dmax = get_param_by_name("solimp_dmax")
            width = get_param_by_name("solimp_width")
            return f"[{dmin[0]}-{dmin[1]}, {dmax[0]}-{dmax[1]}, {width[0]}-{width[1]}, {FIXED_SOLIMP_MIDPOINT}, {FIXED_SOLIMP_POWER}]"
        return "Unknown"

def sample_params_prior(rng: np.random.Generator) -> Dict[str, Any]:
    """Sample parameters from search space using specified distributions."""
    
    # Sample all parameters from space
    Br = sample_from_space(rng, get_param_by_name("Br"))
    solref_timeconst = sample_from_space(rng, get_param_by_name("solref_timeconst"))
    solref_dampratio = sample_from_space(rng, get_param_by_name("solref_dampratio"))
    
    solimp_dmin = sample_from_space(rng, get_param_by_name("solimp_dmin"))
    solimp_dmax = sample_from_space(rng, get_param_by_name("solimp_dmax"))
    solimp_width = sample_from_space(rng, get_param_by_name("solimp_width"))
    
    sliding_friction = sample_from_space(rng, get_param_by_name("sliding_friction"))
    torsional_friction = sample_from_space(rng, get_param_by_name("torsional_friction"))
    rolling_friction = sample_from_space(rng, get_param_by_name("rolling_friction"))
    
    rocker_stiffness = sample_from_space(rng, get_param_by_name("rocker_stiffness"))
    rocker_damping = sample_from_space(rng, get_param_by_name("rocker_damping"))
    
    wheel_kp = sample_from_space(rng, get_param_by_name("wheel_kp"))
    wheel_kv = sample_from_space(rng, get_param_by_name("wheel_kv"))
    
    max_magnetic_distance = sample_from_space(rng, get_param_by_name("max_magnetic_distance"))
    noslip_iterations = int(sample_from_space(rng, get_param_by_name("noslip_iterations")))
    
    return {
        "Br": float(Br),
        "o_solref": [float(solref_timeconst), float(solref_dampratio)],
        "o_solimp": [float(solimp_dmin), float(solimp_dmax), float(solimp_width), 
                     FIXED_SOLIMP_MIDPOINT, FIXED_SOLIMP_POWER],
        "wheel_friction": [float(sliding_friction), float(torsional_friction), float(rolling_friction)],
        "rocker_stiffness": float(rocker_stiffness),
        "rocker_damping": float(rocker_damping),
        "wheel_kp": float(wheel_kp),
        "wheel_kv": float(wheel_kv),
        "max_magnetic_distance": float(max_magnetic_distance),
        "noslip_iterations": int(noslip_iterations),
    }


def sample_params_refined(
    rng: np.random.Generator,
    mean: Dict[str, float],
    std: Dict[str, float],
) -> Dict[str, Any]:
    """
    Sample parameters using Gaussian around elite mean.
    Now ALL tunable parameters use adaptive sampling.
    
    Args:
        rng: Random number generator
        mean: Mean values from elite samples
        std: Standard deviations from elite samples
        
    Returns:
        Dictionary of sampled parameters
    """
    # Adaptive Gaussian sampling for ALL tunable parameters
    Br = rng.normal(mean["Br"], std["Br"])
    Br = _clip(Br, *PARAM_RANGES["Br"])
    
    solref0 = rng.normal(mean["o_solref0"], std["o_solref0"])
    solref0 = _clip(solref0, *PARAM_RANGES["o_solref_0"])
    
    solref1 = rng.normal(mean["o_solref1"], std["o_solref1"])
    solref1 = _clip(solref1, *PARAM_RANGES["o_solref_1"])
    
    # Friction components (each sampled independently)
    wheel_friction_slide = rng.normal(mean["wheel_friction_slide"], std["wheel_friction_slide"])
    wheel_friction_slide = _clip(wheel_friction_slide, *PARAM_RANGES["wheel_friction_slide"])
    
    wheel_friction_torsion = rng.normal(mean["wheel_friction_torsion"], std["wheel_friction_torsion"])
    wheel_friction_torsion = _clip(wheel_friction_torsion, *PARAM_RANGES["wheel_friction_torsion"])
    
    wheel_friction_roll = rng.normal(mean["wheel_friction_roll"], std["wheel_friction_roll"])
    wheel_friction_roll = _clip(wheel_friction_roll, *PARAM_RANGES["wheel_friction_roll"])
    
    wheel_friction = [wheel_friction_slide, wheel_friction_torsion, wheel_friction_roll]
    
    # Joint dynamics
    rocker_stiffness = rng.normal(mean["rocker_stiffness"], std["rocker_stiffness"])
    rocker_stiffness = _clip(rocker_stiffness, *PARAM_RANGES["rocker_stiffness"])
    
    rocker_damping = rng.normal(mean["rocker_damping"], std["rocker_damping"])
    rocker_damping = _clip(rocker_damping, *PARAM_RANGES["rocker_damping"])
    
    # Control gains
    wheel_kp = rng.normal(mean["wheel_kp"], std["wheel_kp"])
    wheel_kp = _clip(wheel_kp, *PARAM_RANGES["wheel_kp"])
    
    wheel_kv = rng.normal(mean["wheel_kv"], std["wheel_kv"])
    wheel_kv = _clip(wheel_kv, *PARAM_RANGES["wheel_kv"])
    
    # Magnetic parameters
    max_magnetic_distance = rng.normal(mean["max_magnetic_distance"], std["max_magnetic_distance"])
    max_magnetic_distance = _clip(max_magnetic_distance, *PARAM_RANGES["max_magnetic_distance"])
    
    # Fixed parameter (not tuned)
    o_solimp = PARAM_RANGES["o_solimp"]

    noslip_iterations = rng.normal(mean["noslip_iterations"], std["noslip_iterations"])
    noslip_iterations = int(_clip(noslip_iterations, *PARAM_RANGES["noslip_iterations"]))
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
        "noslip_iterations": int(noslip_iterations),
    }


def compute_elite_stats(elites: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute mean and std of elite samples for adaptive sampling.
    Now computes statistics for ALL tunable parameters.
    
    Args:
        elites: List of elite trial records
        
    Returns:
        (mean, std) dictionaries for all parameters
    """
    # Extract parameter arrays from elites
    Br_vals = np.array([e["params"]["Br"] for e in elites], dtype=float)
    s0_vals = np.array([e["params"]["o_solref"][0] for e in elites], dtype=float)
    s1_vals = np.array([e["params"]["o_solref"][1] for e in elites], dtype=float)
    
    # Extract friction components (3D array)
    friction_slide = np.array([e["params"]["wheel_friction"][0] for e in elites], dtype=float)
    friction_torsion = np.array([e["params"]["wheel_friction"][1] for e in elites], dtype=float)
    friction_roll = np.array([e["params"]["wheel_friction"][2] for e in elites], dtype=float)
    
    # Extract remaining scalar parameters
    rocker_stiffness = np.array([e["params"]["rocker_stiffness"] for e in elites], dtype=float)
    rocker_damping = np.array([e["params"]["rocker_damping"] for e in elites], dtype=float)
    wheel_kp = np.array([e["params"]["wheel_kp"] for e in elites], dtype=float)
    wheel_kv = np.array([e["params"]["wheel_kv"] for e in elites], dtype=float)
    max_magnetic_distance = np.array([e["params"]["max_magnetic_distance"] for e in elites], dtype=float)
    noslip_iterations = np.array([e["params"]["noslip_iterations"] for e in elites], dtype=float)
    # Compute means
    mean = {
        "Br": float(np.mean(Br_vals)),
        "o_solref0": float(np.mean(s0_vals)),
        "o_solref1": float(np.mean(s1_vals)),
        "wheel_friction_slide": float(np.mean(friction_slide)),
        "wheel_friction_torsion": float(np.mean(friction_torsion)),
        "wheel_friction_roll": float(np.mean(friction_roll)),
        "rocker_stiffness": float(np.mean(rocker_stiffness)),
        "rocker_damping": float(np.mean(rocker_damping)),
        "wheel_kp": float(np.mean(wheel_kp)),
        "wheel_kv": float(np.mean(wheel_kv)),
        "max_magnetic_distance": float(np.mean(max_magnetic_distance)),
        "noslip_iterations": float(np.mean(noslip_iterations)),
    }
    
    # Compute standard deviations with small epsilon to prevent collapse
    std = {
        "Br": float(np.std(Br_vals) + 1e-6),
        "o_solref0": float(np.std(s0_vals) + 1e-9),
        "o_solref1": float(np.std(s1_vals) + 1e-6),
        "wheel_friction_slide": float(np.std(friction_slide) + 1e-6),
        "wheel_friction_torsion": float(np.std(friction_torsion) + 1e-6),
        "wheel_friction_roll": float(np.std(friction_roll) + 1e-6),
        "rocker_stiffness": float(np.std(rocker_stiffness) + 1e-3),
        "rocker_damping": float(np.std(rocker_damping) + 1e-4),
        "wheel_kp": float(np.std(wheel_kp) + 1e-3),
        "wheel_kv": float(np.std(wheel_kv) + 1e-4),
        "max_magnetic_distance": float(np.std(max_magnetic_distance) + 1e-6),
        "noslip_iterations": float(np.std(noslip_iterations) + 0.5),
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
        ("Br", "Magnetic remanence", "Br", "±10% of 1.48 T"),
        ("o_solref[0]", "Contact timeconst", "o_solref_0", "Solver stiffness"),
        ("o_solref[1]", "Contact dampratio", "o_solref_1", "Solver damping"),
        ("o_solimp", "Contact impedance", "o_solimp", "Force ramp-up"),
        ("wheel_friction[0]", "Sliding friction", "wheel_friction_slide", "Tangential slip"),
        ("wheel_friction[1]", "Torsional friction", "wheel_friction_torsion", "Spinning resistance"),
        ("wheel_friction[2]", "Rolling friction", "wheel_friction_roll", "Rolling resistance"),
        ("rocker_stiffness", "Rocker joint stiffness", "rocker_stiffness", "Suspension compliance"),
        ("rocker_damping", "Rocker joint damping", "rocker_damping", "Suspension damping"),
        ("wheel_kp", "Wheel position gain", "wheel_kp", "P controller gain"),
        ("wheel_kv", "Wheel velocity gain", "wheel_kv", "D controller gain"),
        ("max_magnetic_distance", "Magnetic cutoff", "max_magnetic_distance", "0.5cm - 2cm range"),
        ("noslip_iterations", "PGS iterations", "noslip_iterations", "Friction solver accuracy"),
    ]
    
    print(f"{'Parameter':<25} {'Sampling':<10} {'Range':<35} {'Description':<30}")
    print("-"*110)
    
    for display_name, desc, param_key, notes in params_info:
        value = PARAM_RANGES.get(param_key)
        
        if value is None:
            # Parameter not in PARAM_RANGES, skip
            continue
        elif isinstance(value, list):
            # Fixed list (like o_solimp)
            sampling = "Fixed"
            range_str = str(value)
        elif isinstance(value, tuple) and len(value) == 2:
            # Tunable range
            low, high = value
            sampling = sampling_label_for_key(param_key)
            range_str = f"{low} - {high}"
        else:
            # Fixed single value
            sampling = "Fixed"
            range_str = str(value)
        
        print(f"{display_name:<25} {sampling:<10} {range_str:<35} {notes:<30}")
    
    print("="*110)
    print(f"Total tunable parameters: 10 (3 fixed)")
    print(f"Search space dimensionality: 12 (counting vector elements)")
    print("="*110 + "\n")


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
    
    # Parameter info with search bounds - now generated from PARAM_RANGES
    params_info = {
        "Br": ("Magnetic remanence (T)", format_range("Br")),
        "o_solref": ("Contact solver [timeconst, dampratio]", format_range("o_solref")),
        "o_solimp": ("Contact impedance", "Fixed"),
        "wheel_friction": ("Wheel friction [slide, torsion, roll]", format_range("wheel_friction")),
        "rocker_stiffness": ("Rocker stiffness (N·m/rad)", format_range("rocker_stiffness")),
        "rocker_damping": ("Rocker damping (N·m·s/rad)", format_range("rocker_damping")),
        "wheel_kp": ("Wheel P gain", format_range("wheel_kp")),
        "wheel_kv": ("Wheel D gain", format_range("wheel_kv")),
        "max_magnetic_distance": ("Magnetic cutoff (m)", format_range("max_magnetic_distance")),
        "noslip_iterations": ("PGS iterations", format_range("noslip_iterations")),
    }
    
    print(f"{'Parameter':<25} {'Optimized Value':<30} {'Search Range':<30} {'Description':<30}")
    print("-"*115)
    
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
                
                print(f"{k:<25} {val_str:<30} {search_range:<30} {desc:<30}")
            else:
                print(f"{k:<25} {str(v):<30}")
    
    print("="*115 + "\n")

def show_with_enter_to_close(fig, prompt: str = "Focus the plot window and press Enter to close...") -> None:
    """
    Show a matplotlib figure and allow closing it by pressing Enter/Return
    while the window is focused. This blocks until the window is closed.
    """
    import matplotlib.pyplot as plt

    def _on_key(event):
        if event.key in ("enter", "return"):
            plt.close(event.canvas.figure)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    print(f"\n[PLOT] {prompt}")
    plt.show() 

def plot_optimization_progress(cfg: OptimConfig) -> None:
    """
    Plot reward progression over trials for both hold and drive modes.
    Focuses on smooth curves and trends rather than individual points.
    
    Args:
        cfg: Optimization configuration
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path
    
    # Load CSV data
    csv_path = Path(cfg.out_dir) / cfg.exp_name / "trials.csv"
    
    if not csv_path.exists():
        print(f"[WARN] CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimization Progress', fontsize=16, fontweight='bold')
    
    # Calculate rolling averages
    window = max(3, len(df) // 10)  # Adaptive window size
    
    # --- Plot 1: Hold Reward Over Time ---
    ax1 = axes[0, 0]
    rolling_mean_hold = df['reward_hold'].rolling(window=window, center=True).mean()
    ax1.plot(df['trial_id'], rolling_mean_hold, '-', linewidth=2.5, 
             label=f'Hold Reward (smoothed)', color='#2E86AB')
    ax1.fill_between(df['trial_id'], 
                      df['reward_hold'].rolling(window=window, center=True).quantile(0.25),
                      df['reward_hold'].rolling(window=window, center=True).quantile(0.75),
                      alpha=0.2, color='#2E86AB', label='25-75% range')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Trial ID', fontsize=11)
    ax1.set_ylabel('Reward', fontsize=11)
    ax1.set_title('HOLD Mode Reward (Minimize Sideways Drift)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=10)
    
    # --- Plot 2: Drive Reward Over Time ---
    ax2 = axes[0, 1]
    rolling_mean_drive = df['reward_drive'].rolling(window=window, center=True).mean()
    ax2.plot(df['trial_id'], rolling_mean_drive, '-', linewidth=2.5, 
             label='Drive Reward (smoothed)', color='#06A77D')
    ax2.fill_between(df['trial_id'], 
                      df['reward_drive'].rolling(window=window, center=True).quantile(0.25),
                      df['reward_drive'].rolling(window=window, center=True).quantile(0.75),
                      alpha=0.2, color='#06A77D', label='25-75% range')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Trial ID', fontsize=11)
    ax2.set_ylabel('Reward', fontsize=11)
    ax2.set_title('DRIVE Mode Reward (Maximize Upward Climb)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(fontsize=10)
    
    # --- Plot 3: Combined Reward with Best So Far ---
    ax3 = axes[1, 0]
    rolling_mean_combined = df['reward'].rolling(window=window, center=True).mean()
    best_so_far = df['reward'].cummax()
    
    ax3.plot(df['trial_id'], rolling_mean_combined, '-', linewidth=2.5, 
             label='Combined Reward (smoothed)', color='#7209B7')
    ax3.plot(df['trial_id'], best_so_far, '--', linewidth=2.5, 
             label='Best So Far', color='#D62828', alpha=0.8)
    ax3.fill_between(df['trial_id'], 
                      df['reward'].rolling(window=window, center=True).quantile(0.25),
                      df['reward'].rolling(window=window, center=True).quantile(0.75),
                      alpha=0.2, color='#7209B7', label='25-75% range')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Trial ID', fontsize=11)
    ax3.set_ylabel('Reward', fontsize=11)
    ax3.set_title('Combined Reward (min of Hold & Drive)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.legend(fontsize=10)
    
    # --- Plot 4: Contact Percentage Trends ---
    ax4 = axes[1, 1]
    rolling_contact_hold = (df['contact_percentage_hold'] * 100).rolling(window=window, center=True).mean()
    rolling_contact_drive = (df['contact_percentage_drive'] * 100).rolling(window=window, center=True).mean()
    
    ax4.plot(df['trial_id'], rolling_contact_hold, '-', linewidth=2.5,
             label='Hold Contact %', color='#2E86AB')
    ax4.plot(df['trial_id'], rolling_contact_drive, '-', linewidth=2.5,
             label='Drive Contact %', color='#06A77D')
    ax4.axhline(y=20, color='#D62828', linestyle='--', linewidth=2, 
                label='Detachment Threshold', alpha=0.8)
    ax4.set_xlabel('Trial ID', fontsize=11)
    ax4.set_ylabel('Contact Percentage (%)', fontsize=11)
    ax4.set_title('Wall Contact Over Time', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 105])
    ax4.grid(True, alpha=0.3, linestyle=':')
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = Path(cfg.out_dir) / cfg.exp_name / "optimization_progress.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Optimization progress plot saved to: {plot_path}")
    
    # Show plot
    show_with_enter_to_close(fig, "Optimization progress: press Enter to close and continue.")



def plot_parameter_evolution(cfg: OptimConfig) -> None:
    """
    Plot how key parameters evolved during optimization.
    Shows smooth trends with standard deviation bands.
    
    Args:
        cfg: Optimization configuration
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import json
    from pathlib import Path
    
    # Load JSONL data (has full parameter info)
    jsonl_path = Path(cfg.out_dir) / cfg.exp_name / "trials.jsonl"
    
    if not jsonl_path.exists():
        print(f"[WARN] JSONL file not found: {jsonl_path}")
        return
    
    # Parse JSONL
    trials = []
    with jsonl_path.open("r") as f:
        for line in f:
            trials.append(json.loads(line))
    
    # Extract parameters
    df = pd.DataFrame([
        {
            'trial_id': t['trial_id'],
            'Br': t['params']['Br'],
            'max_magnetic_distance': t['params']['max_magnetic_distance'],
            'rocker_stiffness': t['params']['rocker_stiffness'],
            'wheel_kp': t['params']['wheel_kp'],
            'reward': t['reward']
        }
        for t in trials
    ])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Parameter Evolution During Optimization', fontsize=16, fontweight='bold')
    
    # Rolling window for smoothing
    window = max(3, len(df) // 10)
    
    # --- Plot 1: Magnetic Remanence (Br) ---
    ax1 = axes[0, 0]
    rolling_mean = df['Br'].rolling(window=window, center=True).mean()
    rolling_std = df['Br'].rolling(window=window, center=True).std()
    
    ax1.plot(df['trial_id'], rolling_mean, '-', linewidth=2.5, color='#E63946', label='Mean')
    ax1.fill_between(df['trial_id'], 
                      rolling_mean - rolling_std, 
                      rolling_mean + rolling_std,
                      alpha=0.3, color='#E63946', label='±1 std')
    ax1.set_xlabel('Trial ID', fontsize=11)
    ax1.set_ylabel('Br (Tesla)', fontsize=11)
    ax1.set_title('Magnetic Remanence Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=10)
    
    # Add range lines
    br_range = PARAM_RANGES["Br"]
    ax1.axhline(y=br_range[0], color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax1.axhline(y=br_range[1], color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # --- Plot 2: Max Magnetic Distance ---
    ax2 = axes[0, 1]
    rolling_mean = df['max_magnetic_distance'].rolling(window=window, center=True).mean()
    rolling_std = df['max_magnetic_distance'].rolling(window=window, center=True).std()
    
    ax2.plot(df['trial_id'], rolling_mean, '-', linewidth=2.5, color='#F77F00', label='Mean')
    ax2.fill_between(df['trial_id'], 
                      rolling_mean - rolling_std, 
                      rolling_mean + rolling_std,
                      alpha=0.3, color='#F77F00', label='±1 std')
    ax2.set_xlabel('Trial ID', fontsize=11)
    ax2.set_ylabel('Max Magnetic Distance (m)', fontsize=11)
    ax2.set_title('Magnetic Cutoff Distance Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(fontsize=10)
    
    # Add range lines
    dist_range = PARAM_RANGES["max_magnetic_distance"]
    ax2.axhline(y=dist_range[0], color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax2.axhline(y=dist_range[1], color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # --- Plot 3: Rocker Stiffness ---
    ax3 = axes[1, 0]
    rolling_mean = df['rocker_stiffness'].rolling(window=window, center=True).mean()
    rolling_std = df['rocker_stiffness'].rolling(window=window, center=True).std()
    
    ax3.plot(df['trial_id'], rolling_mean, '-', linewidth=2.5, color='#06A77D', label='Mean')
    ax3.fill_between(df['trial_id'], 
                      rolling_mean - rolling_std, 
                      rolling_mean + rolling_std,
                      alpha=0.3, color='#06A77D', label='±1 std')
    ax3.set_xlabel('Trial ID', fontsize=11)
    ax3.set_ylabel('Rocker Stiffness (N·m/rad)', fontsize=11)
    ax3.set_title('Rocker Stiffness Evolution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.legend(fontsize=10)
    
    # Add range lines
    stiff_range = PARAM_RANGES["rocker_stiffness"]
    ax3.axhline(y=stiff_range[0], color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax3.axhline(y=stiff_range[1], color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # --- Plot 4: Wheel Kp ---
    ax4 = axes[1, 1]
    rolling_mean = df['wheel_kp'].rolling(window=window, center=True).mean()
    rolling_std = df['wheel_kp'].rolling(window=window, center=True).std()
    
    ax4.plot(df['trial_id'], rolling_mean, '-', linewidth=2.5, color='#2E86AB', label='Mean')
    ax4.fill_between(df['trial_id'], 
                      rolling_mean - rolling_std, 
                      rolling_mean + rolling_std,
                      alpha=0.3, color='#2E86AB', label='±1 std')
    ax4.set_xlabel('Trial ID', fontsize=11)
    ax4.set_ylabel('Wheel Kp (P Gain)', fontsize=11)
    ax4.set_title('Wheel Controller Gain Evolution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle=':')
    ax4.legend(fontsize=10)
    
    # Add range lines
    kp_range = PARAM_RANGES["wheel_kp"]
    ax4.axhline(y=kp_range[0], color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax4.axhline(y=kp_range[1], color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = Path(cfg.out_dir) / cfg.exp_name / "parameter_evolution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Parameter evolution plot saved to: {plot_path}")
    
    show_with_enter_to_close(fig, "Parameter evolution: press Enter to close and continue.")
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
    
    # Plot optimization progress
    print("\nGenerating plots...")
    plot_optimization_progress(cfg)
    plot_parameter_evolution(cfg)
    
    # Launch viewer
    launch_viewer(cfg, best)