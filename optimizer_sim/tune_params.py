from __future__ import annotations

import csv
import uuid
import argparse
import sim_optimizer
from config import MODES, DEFAULT_MODE
from typing import Dict, Any, List

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


all_results = []
N_CALLS = 100

# Parse mode from CLI
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=MODES.keys())
args = parser.parse_args()
MODE = args.mode
mode_cfg = MODES[MODE]

# 1. Define the search space for the parameters.
space = [
    # Magnetic parameters
    Real(1.48 * 0.9, 1.48 * 1.1, "uniform", name='Br'),
    
    # Solver parameters
    Real(0.0001, 0.0008, "uniform", name='solref_timeconst'),
    Real(10.0, 50.0, "uniform", name='solref_dampratio'),
    Real(0.8, 0.99, "uniform", name='solimp_dmin'),
    Real(0.9, 1.0, "uniform", name='solimp_dmax'),
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


def cost_minimize_slip(trajectory, mode_cfg):
    """Hold mode: minimize total movement (want zero slip)."""
    if not trajectory:
        return {'total_cost': 1e6, 'total_movement': 0}

    settle_time = mode_cfg["settle_time"]
    start_idx = 0
    for idx, state in enumerate(trajectory):
        if state['time'] >= settle_time:
            start_idx = idx
            break

    total_movement = 0
    for i in range(start_idx + 1, len(trajectory)):
        dx = trajectory[i]['pos'][0] - trajectory[i-1]['pos'][0]
        dy = trajectory[i]['pos'][1] - trajectory[i-1]['pos'][1]
        dz = trajectory[i]['pos'][2] - trajectory[i-1]['pos'][2]
        total_movement += np.sqrt(dx**2 + dy**2 + dz**2)

    total_cost = abs(total_movement - 0.0)  # target = zero slip
    print(f"  Total Movement: {total_movement:.4f} | Cost: {total_cost:.4f}")
    return {'total_cost': total_cost, 'total_movement': total_movement}

def cost_drive_side(trajectory, mode_cfg):
    """Drive sideways mode: match average Y velocity to target."""
    if not trajectory:
        return {'total_cost': 1e6, 'avg_vel': 0}

    settle_time = mode_cfg["settle_time"]
    start_state = next((s for s in trajectory if s['time'] >= settle_time), trajectory[0])
    end_state = trajectory[-1]

    dt = end_state['time'] - start_state['time']
    if dt < 1e-6:
        return {'total_cost': 1e6, 'avg_vel': 0}

    y_disp = end_state['pos'][1] - start_state['pos'][1]
    avg_vel = y_disp / dt
    target_vel = mode_cfg["actuator_target_ms"]

    total_cost = abs(avg_vel - target_vel)
    print(f"  Avg Y vel: {avg_vel:.4f} m/s | Target: {target_vel:.4f} | Cost: {total_cost:.4f}")
    return {'total_cost': total_cost, 'avg_vel': avg_vel}

COST_FUNCTIONS = {
    "minimize_slip": cost_minimize_slip,
    "drive_side": cost_drive_side,
}
# 2. Define the objective function to minimize.
# It takes the parameters, runs the simulation, and returns the error.
@use_named_args(space)
def objective(**params):
    """
    Objective function for the Bayesian optimizer.
    """
    sim_params = {
        # Friction parameters
        'ground_friction': [
            params['sliding_friction'],
            params['torsional_friction'],
            params['rolling_friction']
        ],
        
        # Solver parameters
        'solref': [params['solref_timeconst'], params['solref_dampratio']],
        'solimp': [
            params['solimp_dmin'],
            params['solimp_dmax'],
            params['solimp_width'],
            0.5,  # default midpoint
            1.0   # default power
        ],
        'noslip_iterations': params['noslip_iterations'],
        
        # Magnetic parameters
        'Br': params['Br'],
        'max_magnetic_distance': params['max_magnetic_distance'],
        
        # Joint dynamics
        'rocker_stiffness': params['rocker_stiffness'],
        'rocker_damping': params['rocker_damping'],
        
        # Control gains
        'wheel_kp': params['wheel_kp'],
        'wheel_kv': params['wheel_kv'],
    }

    trajectory = sim_optimizer.run_simulation(
        sim_params, mode=MODE)
    
    if trajectory is None:
        print("  Simulation unstable. Assigning large penalty.")
        cost_data = {'total_cost': 1e6}
    else:
        cost_fn = COST_FUNCTIONS[mode_cfg["cost_function"]]
        cost_data = cost_fn(trajectory, mode_cfg)

    # Store detailed results for this run
    run_id = str(uuid.uuid4().hex)[:8]
    all_results.append({
        'id': run_id,
        'cost': cost_data['total_cost'],
        'cost_data': cost_data,
        'params': params
    })

    return cost_data['total_cost']

if __name__ == "__main__":
    # 3. Run the optimization.
    print(f"Running Bayesian optimization for {N_CALLS} iterations (mode={MODE})...")
    
    # Optionally, start from the best known parameters for the "hold" mode to speed up convergence.
    best_hold_x0 = [1.628, 0.0008, 10.0, 0.9094, 0.9946, 0.000667,
                     0.9876, 0.000592, 0.00001, 100.0, 2.3945,
                     1.1759, 5.7573, 0.03275, 28]
    
    result = gp_minimize(
        objective,
        space,
        n_calls=N_CALLS,
        random_state=42,
        n_initial_points=N_CALLS // 5,
        x0=best_hold_x0, # Start from the best known hold parameters, remove if not working)
    )

    # 4. Save results to a CSV file
    if all_results:
        # Sort the results by cost before saving
        sorted_results = sorted(all_results, key=lambda r: r['cost'])

        param_names = [dim.name for dim in space]
        extra_keys = [k for k in sorted_results[0]['cost_data'].keys() if k != 'total_cost']
        fieldnames = ['id', 'cost'] + extra_keys + param_names
        try:
            with open(f'optimization_results_{MODE}.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for res in sorted_results:
                    row = {'id': res['id'], 'cost': res['cost']}
                    row.update({k: res['cost_data'][k] for k in extra_keys})
                    row.update(res['params'])
                    writer.writerow(row)
            print(f"\n--- Results saved to optimization_results_{MODE}.csv ---")
        except Exception as e:
            print(f"\n--- Could not save optimization results to CSV: {e} ---")

    print("\n--- Optimization Finished ---")
    best_cost = result.fun
    print(f"Lowest Cost Found: {best_cost:.6f}")
    print("Best Parameters:")
    for dim, value in zip(space, result.x):
        print(f"  {dim.name}: {value:.6f}")
    
    # --- Launch best result with visualization ---
    print("\n--- Launching best result with visualization ---")
    best_result = sorted_results[0]
    
    best_sim_params = {
        'ground_friction': [
            best_result['params']['sliding_friction'],
            best_result['params']['torsional_friction'],
            best_result['params']['rolling_friction']
        ],
        'solref': [
            best_result['params']['solref_timeconst'],
            best_result['params']['solref_dampratio']
        ],
        'solimp': [
            best_result['params']['solimp_dmin'],
            best_result['params']['solimp_dmax'],
            best_result['params']['solimp_width'], 
            0.5, 
            1.0
        ],
        'noslip_iterations': best_result['params']['noslip_iterations'],
        'Br': best_result['params']['Br'],
        'max_magnetic_distance': best_result['params']['max_magnetic_distance'],
        'rocker_stiffness': best_result['params']['rocker_stiffness'],
        'rocker_damping': best_result['params']['rocker_damping'],
        'wheel_kp': best_result['params']['wheel_kp'],
        'wheel_kv': best_result['params']['wheel_kv'],
    }
    
    print(f"Best cost: {best_result['cost']:.6f}")
    print("Launching viewer...")
    
    sim_optimizer.run_simulation(
        best_sim_params,
        visualize=True,
        mode=MODE
    )