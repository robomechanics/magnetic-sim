"""
optimizer.py

Black-box parameter optimization for magnetic wall-climbing robot.

MODIFIED TO USE BAYESIAN OPTIMIZATION (matches tune_params.py structure)

Optimization Strategy:
- Bayesian optimization using scikit-optimize (gp_minimize)
- Runs rollouts in BOTH hold and drive modes
- Combined cost = max(hold_cost, drive_cost) - both must succeed
- Logs all trials to CSV for analysis
"""

from __future__ import annotations

import os
import csv
import uuid
from typing import Dict, Any, List

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Import your existing simulation module
from sim_sally_magnet_wall import rollout


# --- Optimization Configuration ---
N_CALLS = 30  # Number of optimization iterations

# This list will store detailed results from each trial
all_results = []


# 1. Define the search space for the parameters.
# Matches the structure from tune_params.py in skopt format
space = [
    # Magnetic parameters
    Real(1.48 * 0.9, 1.48 * 1.1, "uniform", name='Br'),
    
    # Solver parameters
    Real(0.0001, 0.0008, "uniform", name='solref_timeconst'),
    Real(10.0, 50.0, "uniform", name='solref_dampratio'),
    Real(0.8, 0.99, "uniform", name='solimp_dmin'),
    Real(0.9, 0.999, "uniform", name='solimp_dmax'),
    Real(1e-4, 1e-2, "log-uniform", name='solimp_width'),
    
    # Friction parameters (log-uniform for small values)
    Real(0.9, 1.0, "uniform", name='sliding_friction'),
    Real(1e-5, 0.1, "log-uniform", name='torsional_friction'),
    Real(1e-5, 0.1, "log-uniform", name='rolling_friction'),
    
    # Joint dynamics (log-uniform spans order of magnitude)
    Real(10.0, 100.0, "log-uniform", name='rocker_stiffness'),
    Real(0.1, 5.0, "log-uniform", name='rocker_damping'),
    
    # Control gains (log-uniform)
    Real(1.0, 50.0, "log-uniform", name='wheel_kp'),
    Real(0.1, 10.0, "log-uniform", name='wheel_kv'),
    
    # Magnetic cutoff (log-uniform)
    Real(0.005, 0.1, "log-uniform", name='max_magnetic_distance'),
    
    # Solver iterations (integer, uniform)
    Integer(5, 30, name='noslip_iterations'),
]

# Fixed o_solimp values that won't be tuned
FIXED_SOLIMP_MIDPOINT = 0.5
FIXED_SOLIMP_POWER = 2.0


def calculate_cost(result_hold, result_drive):
    """
    Calculates combined cost from hold and drive test results.
    Both tests must succeed - we take the worse (higher) cost.
    
    Cost is negative reward (we minimize cost, reward is maximized by metrics.py)
    
    Args:
        result_hold: RolloutResult from hold mode test
        result_drive: RolloutResult from drive mode test
        
    Returns:
        Dictionary with combined cost and individual metrics
    """
    # Extract rewards from summaries
    summary_hold = result_hold.summary or {}
    summary_drive = result_drive.summary or {}
    
    reward_hold = summary_hold.get("reward", -1e9)
    reward_drive = summary_drive.get("reward", -1e9)
    
    # Convert rewards to costs (cost = -reward)
    cost_hold = -reward_hold
    cost_drive = -reward_drive
    
    # Combined cost: both must succeed, so take the worse cost
    total_cost = max(cost_hold, cost_drive)
    
    # Extract additional metrics for logging
    slip_m = summary_hold.get("slip_m", 0.0)
    progress_m = summary_drive.get("progress_m", 0.0)
    progress_rate_mps = summary_drive.get("progress_rate_mps", 0.0)
    
    print(
        f"  Hold Cost: {cost_hold:.4f} | "
        f"Drive Cost: {cost_drive:.4f} | "
        f"Combined: {total_cost:.4f} | "
        f"Slip: {slip_m:.4f}m | "
        f"Progress: {progress_m:.4f}m"
    )
    
    return {
        'total_cost': total_cost,
        'cost_hold': cost_hold,
        'cost_drive': cost_drive,
        'reward_hold': reward_hold,
        'reward_drive': reward_drive,
        'slip_m': slip_m,
        'progress_m': progress_m,
        'progress_rate_mps': progress_rate_mps,
        'termination_hold': str(result_hold.termination),
        'termination_drive': str(result_drive.termination),
    }


# 2. Define the objective function to minimize.
# It takes the parameters, runs the simulation, and returns the cost.
@use_named_args(space)
def objective(**params):
    """
    Objective function for the Bayesian optimizer.
    """
    sim_params = {
        # Magnetic
        'Br': params['Br'],
        
        # Solver parameters
        'o_solref': [
            params['solref_timeconst'],
            params['solref_dampratio']
        ],
        'o_solimp': [
            params['solimp_dmin'],
            params['solimp_dmax'],
            params['solimp_width'],
            FIXED_SOLIMP_MIDPOINT,
            FIXED_SOLIMP_POWER
        ],
        
        # Friction (array format)
        'wheel_friction': [
            params['sliding_friction'],
            params['torsional_friction'],
            params['rolling_friction']
        ],
        
        # Joint dynamics
        'rocker_stiffness': params['rocker_stiffness'],
        'rocker_damping': params['rocker_damping'],
        
        # Control gains
        'wheel_kp': params['wheel_kp'],
        'wheel_kv': params['wheel_kv'],
        
        # Magnetic cutoff
        'max_magnetic_distance': params['max_magnetic_distance'],
        
        # Solver iterations
        'noslip_iterations': int(params['noslip_iterations']),
    }
    
    # Run HOLD test (sideways, should not slip)
    params_hold = dict(sim_params)
    params_hold["mode"] = "sideways"
    params_hold["rollout_mode"] = "hold"
    
    result_hold = rollout(
        params=params_hold,
        sim_duration=5.0,
        settle_time=1.0,
        log_stride=10,
    )
    
    # Run DRIVE test (drive_up, should climb)
    params_drive = dict(sim_params)
    params_drive["mode"] = "drive_up"
    params_drive["rollout_mode"] = "drive"
    
    result_drive = rollout(
        params=params_drive,
        sim_duration=5.0,
        settle_time=1.0,
        log_stride=10,
    )
    
    # Check for simulation failures
    if result_hold.termination != "ok" or result_drive.termination != "ok":
        print(f"  Simulation unstable. Hold: {result_hold.termination}, Drive: {result_drive.termination}")
        cost_data = {
            'total_cost': 1e6,
            'cost_hold': 1e6,
            'cost_drive': 1e6,
            'reward_hold': -1e6,
            'reward_drive': -1e6,
            'slip_m': 999.0,
            'progress_m': 0.0,
            'progress_rate_mps': 0.0,
            'termination_hold': str(result_hold.termination),
            'termination_drive': str(result_drive.termination),
        }
    else:
        cost_data = calculate_cost(result_hold, result_drive)
    
    # Store detailed results for this run
    run_id = str(uuid.uuid4().hex)[:8]
    all_results.append({
        'id': run_id,
        'cost': cost_data['total_cost'],
        'cost_hold': cost_data['cost_hold'],
        'cost_drive': cost_data['cost_drive'],
        'reward_hold': cost_data['reward_hold'],
        'reward_drive': cost_data['reward_drive'],
        'slip_m': cost_data['slip_m'],
        'progress_m': cost_data['progress_m'],
        'progress_rate_mps': cost_data['progress_rate_mps'],
        'termination_hold': cost_data['termination_hold'],
        'termination_drive': cost_data['termination_drive'],
        'params': params
    })
    
    return cost_data['total_cost']

# =============================================================================
# PLOTTING AND VISUALIZATION
# =============================================================================

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


def plot_optimization_progress():
    """
    Plot cost progression over trials for both hold and drive modes.
    Uses the CSV file to generate plots.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[WARN] matplotlib or pandas not installed. Skipping plots.")
        return
    
    # Load CSV data
    csv_path = 'optimization_results.csv'
    
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Add trial_id if not present
    if 'trial_id' not in df.columns:
        df['trial_id'] = range(len(df))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimization Progress', fontsize=16, fontweight='bold')
    
    # Calculate rolling averages
    window = max(3, len(df) // 10)  # Adaptive window size
    
    # --- Plot 1: Hold Cost Over Time ---
    ax1 = axes[0, 0]
    rolling_mean_hold = df['cost_hold'].rolling(window=window, center=True).mean()
    ax1.plot(df['trial_id'], rolling_mean_hold, '-', linewidth=2.5, 
             label=f'Hold Cost (smoothed)', color='#2E86AB')
    ax1.fill_between(df['trial_id'], 
                      df['cost_hold'].rolling(window=window, center=True).quantile(0.25),
                      df['cost_hold'].rolling(window=window, center=True).quantile(0.75),
                      alpha=0.2, color='#2E86AB', label='25-75% range')
    ax1.set_xlabel('Trial ID', fontsize=11)
    ax1.set_ylabel('Cost', fontsize=11)
    ax1.set_title('HOLD Mode Cost (Minimize Sideways Drift)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=10)
    
    # --- Plot 2: Drive Cost Over Time ---
    ax2 = axes[0, 1]
    rolling_mean_drive = df['cost_drive'].rolling(window=window, center=True).mean()
    ax2.plot(df['trial_id'], rolling_mean_drive, '-', linewidth=2.5, 
             label='Drive Cost (smoothed)', color='#06A77D')
    ax2.fill_between(df['trial_id'], 
                      df['cost_drive'].rolling(window=window, center=True).quantile(0.25),
                      df['cost_drive'].rolling(window=window, center=True).quantile(0.75),
                      alpha=0.2, color='#06A77D', label='25-75% range')
    ax2.set_xlabel('Trial ID', fontsize=11)
    ax2.set_ylabel('Cost', fontsize=11)
    ax2.set_title('DRIVE Mode Cost (Maximize Upward Climb)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(fontsize=10)
    
    # --- Plot 3: Combined Cost with Best So Far ---
    ax3 = axes[1, 0]
    rolling_mean_combined = df['cost'].rolling(window=window, center=True).mean()
    best_so_far = df['cost'].cummin()  # cummin because lower cost is better
    
    ax3.plot(df['trial_id'], rolling_mean_combined, '-', linewidth=2.5, 
             label='Combined Cost (smoothed)', color='#7209B7')
    ax3.plot(df['trial_id'], best_so_far, '--', linewidth=2.5, 
             label='Best So Far', color='#D62828', alpha=0.8)
    ax3.fill_between(df['trial_id'], 
                      df['cost'].rolling(window=window, center=True).quantile(0.25),
                      df['cost'].rolling(window=window, center=True).quantile(0.75),
                      alpha=0.2, color='#7209B7', label='25-75% range')
    ax3.set_xlabel('Trial ID', fontsize=11)
    ax3.set_ylabel('Cost', fontsize=11)
    ax3.set_title('Combined Cost (max of Hold & Drive)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.legend(fontsize=10)
    
    # --- Plot 4: Progress vs Slip ---
    ax4 = axes[1, 1]
    ax4.scatter(df['slip_m'], df['progress_m'], alpha=0.6, c=df['cost'], 
                cmap='viridis_r', s=50)
    ax4.set_xlabel('Slip (m)', fontsize=11)
    ax4.set_ylabel('Progress (m)', fontsize=11)
    ax4.set_title('Progress vs Slip (color = cost)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle=':')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Cost', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = 'optimization_progress.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Optimization progress plot saved to: {plot_path}")
    
    # Show plot
    show_with_enter_to_close(fig, "Optimization progress: press Enter to close and continue.")


# =============================================================================
# INTERACTIVE VIEWER
# =============================================================================

def launch_viewer(best: Dict[str, Any]):
    """
    Launch interactive viewer to compare baseline vs optimized parameters.
    
    Args:
        best: Best parameters found from optimization
    """
    try:
        import viewer
    except ImportError:
        print("\n[WARN] viewer module not found. Skipping interactive visualization.")
        response = input("Would you like to see the best parameters instead? (y/n): ").strip().lower()
        if response == 'y':
            print("\nBest Parameters Found:")
            print("="*80)
            for k, v in best['params'].items():
                if isinstance(v, list):
                    print(f"  {k}: {v}")
                elif isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
        return
    
    # Baseline comparison
    response = input("\nPress ENTER to view BASELINE (default params), or 'n' to skip: ").strip().lower()
    
    if response != 'n':
        print("\n" + "="*60)
        print("BASELINE - HOLD MODE (drive_up)")
        print("="*60)
        
        viewer.INSPECT_PARAMS = {'mode': 'drive_up', 'rollout_mode': 'hold'}
        viewer.SIM_DURATION = 5.0
        viewer.main()
        
        input("\nPress ENTER to see baseline DRIVE mode...")
        
        print("\n" + "="*60)
        print("BASELINE - DRIVE MODE (sideways)")
        print("="*60)
        
        viewer.INSPECT_PARAMS = {'mode': 'sideways', 'rollout_mode': 'drive'}
        viewer.SIM_DURATION = 5.0
        viewer.main()
    
    # Optimized comparison
    response2 = input("\nPress ENTER to view BEST OPTIMIZED runs, or 'n' to skip: ").strip().lower()
    
    if response2 != 'n':
        print("\n" + "="*60)
        print("BEST - HOLD MODE (drive_up)")
        print("="*60)
        print(f"Hold reward: {best.get('reward_hold', 'N/A')}")
        
        best_params_hold = dict(best["params"])
        best_params_hold['mode'] = 'drive_up'
        best_params_hold['rollout_mode'] = 'hold'
        
        viewer.INSPECT_PARAMS = best_params_hold
        viewer.SIM_DURATION = 5.0
        viewer.main()
        
        input("\nPress ENTER to see best DRIVE mode...")
        
        print("\n" + "="*60)
        print("BEST - DRIVE MODE (sideways)")
        print("="*60)
        print(f"Drive reward: {best.get('reward_drive', 'N/A')}")
        
        best_params_drive = dict(best["params"])
        best_params_drive['mode'] = 'sideways'
        best_params_drive['rollout_mode'] = 'drive'
        
        viewer.INSPECT_PARAMS = best_params_drive
        viewer.SIM_DURATION = 5.0
        viewer.main()


if __name__ == "__main__":
    # 3. Run the optimization.
    print(f"Running Bayesian optimization for {N_CALLS} iterations...")
    print(f"Optimizing for minimal detachment and drift (hold + drive)")
    
    result = gp_minimize(
        objective,
        space,
        n_calls=N_CALLS,
        random_state=42,
        # The optimizer builds a model of the cost function. When it encounters
        # bad parameters (high cost), it learns to avoid that region and samples
        # from more promising areas. Increasing n_initial_points gives it a
        # better starting model.
        n_initial_points=20
    )
    
    # 4. Save results to a CSV file
    if all_results:
        # Sort the results by cost before saving
        sorted_results = sorted(all_results, key=lambda r: r['cost'])
        
        param_names = [dim.name for dim in space]
        fieldnames = [
            'id', 'cost', 'cost_hold', 'cost_drive',
            'reward_hold', 'reward_drive',
            'slip_m', 'progress_m', 'progress_rate_mps',
            'termination_hold', 'termination_drive'
        ] + param_names
        try:
            with open('optimization_results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for res in sorted_results:
                    row = {
                        'id': res['id'],
                        'cost': res['cost'],
                        'cost_hold': res['cost_hold'],
                        'cost_drive': res['cost_drive'],
                        'reward_hold': res['reward_hold'],
                        'reward_drive': res['reward_drive'],
                        'slip_m': res['slip_m'],
                        'progress_m': res['progress_m'],
                        'progress_rate_mps': res['progress_rate_mps'],
                        'termination_hold': res['termination_hold'],
                        'termination_drive': res['termination_drive'],
                    }
                    row.update(res['params'])
                    writer.writerow(row)
            print("\n--- Optimization results saved to optimization_results.csv (sorted by cost) ---")
        except Exception as e:
            print(f"\n--- Could not save optimization results to CSV: {e} ---")


    print("\n--- Optimization Finished ---")
    best_cost = result.fun
    print(f"Lowest Cost Found: {best_cost:.6f}")
    print("Best Parameters:")
    for dim, value in zip(space, result.x):
        print(f"  {dim.name}: {value:.6f}")
    
    # Print best trial details
    if sorted_results:
        best_result = sorted_results[0]
        print("\nBest Trial Details:")
        print(f"  ID: {best_result['id']}")
        print(f"  Hold Reward: {best_result['reward_hold']:.6f}")
        print(f"  Drive Reward: {best_result['reward_drive']:.6f}")
        print(f"  Slip: {best_result['slip_m']:.4f} m")
        print(f"  Progress: {best_result['progress_m']:.4f} m")
        print(f"  Termination (hold): {best_result['termination_hold']}")
        print(f"  Termination (drive): {best_result['termination_drive']}")
        
        # Plot optimization progress
        print("\nGenerating plots...")
        plot_optimization_progress()
        
        # Launch interactive viewer
        launch_viewer(best_result)
    
    print("\n✅ All done! Check optimization_results.csv for full results.")