from __future__ import annotations

import csv
import uuid
import argparse
import optimizer_sim.backup_04_22.sim_optimizer as sim_optimizer
from optimizer_sim.backup_04_22.config import MODES, DEFAULT_MODE, N_CALLS, SEARCH_SPACE, DEFAULT_PARAMS
from typing import Dict, Any, List

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


all_results = []

# Parse mode from CLI
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=MODES.keys())
args = parser.parse_args()
MODE = args.mode
mode_cfg = MODES[MODE]

# 1. Define the search space for the parameters.
space = SEARCH_SPACE


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
        return {'total_cost': 1e6, 'avg_vel': 0, 'avg_z_disp': 0, 'avg_x_disp': 0}

    settle_time = mode_cfg["settle_time"]
    start_state = next((s for s in trajectory if s['time'] >= settle_time), trajectory[0])
    end_state = trajectory[-1]

    dt = end_state['time'] - start_state['time']
    if dt < 1e-6:
        return {'total_cost': 1e6, 'avg_vel': 0, 'avg_z_disp': 0, 'avg_x_disp': 0}

    # Main objective: Y velocity (sideways)
    y_disp = end_state['pos'][1] - start_state['pos'][1]
    avg_y_vel = y_disp / dt
    target_vel = mode_cfg["actuator_target_ms"]
    y_vel_error = abs(avg_y_vel - target_vel)
    
    # Penalties: minimize unwanted X and Z displacement
    x_disp = abs(end_state['pos'][0] - start_state['pos'][0])  # Penalize any X displacement
    z_disp = abs(end_state['pos'][2] - start_state['pos'][2])  # Penalize any Z displacement (falling/climbing)
    # Calculate how far the robot actually traveled in Y direction
    y_distance_traveled = abs(y_disp)
    target_distance = target_vel * dt  # How far it SHOULD have traveled

    # Penalty for not reaching the expected distance
    y_distance_error = abs(y_distance_traveled - target_distance) * 0.6

    # Total cost: primary objective + smaller penalties for off-axis displacement
    total_cost = y_vel_error + y_distance_error + 0.2 * x_disp + 0.2 * z_disp
    # total_cost = y_vel_error + 0.2 * x_disp + 0.2 * z_disp

    print(f"  Avg Y vel: {avg_y_vel:.4f} m/s | Target: {target_vel:.4f} | X disp: {x_disp:.4f} m | Z disp: {z_disp:.4f} m | Cost: {total_cost:.4f}")
    return {
        'total_cost': total_cost, 
        'avg_vel': avg_y_vel,
        'avg_x_disp': x_disp,
        'avg_z_disp': z_disp
    }

def cost_drive_up(trajectory, mode_cfg):
    """Drive up mode: match average Z velocity to target."""
    if not trajectory:
        return {'total_cost': 1e6, 'avg_vel': 0, 'avg_x_disp': 0, 'avg_y_disp': 0}

    settle_time = mode_cfg["settle_time"]
    start_state = next((s for s in trajectory if s['time'] >= settle_time), trajectory[0])
    end_state = trajectory[-1]

    dt = end_state['time'] - start_state['time']
    if dt < 1e-6:
        return {'total_cost': 1e6, 'avg_vel': 0, 'avg_x_disp': 0, 'avg_y_disp': 0}

    # Main objective: Z velocity (upward)
    z_disp = end_state['pos'][2] - start_state['pos'][2]
    avg_z_vel = z_disp / dt
    target_vel = mode_cfg["actuator_target_ms"]
    z_vel_error = abs(avg_z_vel - target_vel)
    
    # Penalties: minimize unwanted X and Y displacement
    x_disp = abs(end_state['pos'][0] - start_state['pos'][0])  # Penalize any X displacement
    y_disp = abs(end_state['pos'][1] - start_state['pos'][1])  # Penalize any Y displacement (sideways drift)
    
    # Total cost: primary objective + smaller penalties for off-axis displacement
    total_cost = z_vel_error + 0.2 * x_disp + 0.2 * y_disp

    print(f"  Avg Z vel: {avg_z_vel:.4f} m/s | Target: {target_vel:.4f} | X disp: {x_disp:.4f} m | Y disp: {y_disp:.4f} m | Cost: {total_cost:.4f}")
    return {
        'total_cost': total_cost, 
        'avg_vel': avg_z_vel,
        'avg_x_disp': x_disp,
        'avg_y_disp': y_disp
    }

COST_FUNCTIONS = {
    "minimize_slip": cost_minimize_slip,
    "drive_side": cost_drive_side,
    "drive_up": cost_drive_up,
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
                      
        'max_force_per_wheel': params['max_force_per_wheel'],
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
    
    # Optionally, start from the best known parameters
    # Convert DEFAULT_PARAMS dict to x0 list matching the space order
    best_hold_x0 = [
        DEFAULT_PARAMS['Br'],                           # Br
        DEFAULT_PARAMS['solref'][0],                    # solref_timeconst
        DEFAULT_PARAMS['solref'][1],                    # solref_dampratio
        DEFAULT_PARAMS['solimp'][0],                    # solimp_dmin
        DEFAULT_PARAMS['solimp'][1],                    # solimp_dmax
        DEFAULT_PARAMS['solimp'][2],                    # solimp_width
        DEFAULT_PARAMS['ground_friction'][0],           # sliding_friction
        DEFAULT_PARAMS['ground_friction'][1],           # torsional_friction
        DEFAULT_PARAMS['ground_friction'][2],           # rolling_friction
        DEFAULT_PARAMS['rocker_stiffness'],             # rocker_stiffness
        DEFAULT_PARAMS['rocker_damping'],               # rocker_damping
        DEFAULT_PARAMS['wheel_kp'],                     # wheel_kp
        DEFAULT_PARAMS['wheel_kv'],                     # wheel_kv
        DEFAULT_PARAMS['max_magnetic_distance'],        # max_magnetic_distance
        DEFAULT_PARAMS['noslip_iterations'],            # noslip_iterations
        DEFAULT_PARAMS['max_force_per_wheel']           # max_force_per_wheel
    ]
    
    result = gp_minimize(
        objective,
        space,
        n_calls=N_CALLS,
        random_state=42,
        n_initial_points=N_CALLS // 5,
        # x0=best_hold_x0,  # Start from the best known hold parameters
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
    
    # Print in DEFAULT_PARAMS format for easy copy-paste
    best_params_dict = dict(zip([dim.name for dim in space], result.x))
    print("\nDEFAULT_PARAMS = {")
    print(f"    'ground_friction': [{best_params_dict['sliding_friction']:.6f}, {best_params_dict['torsional_friction']:.6f}, {best_params_dict['rolling_friction']:.6f}],")
    print(f"    'solref': [{best_params_dict['solref_timeconst']:.6f}, {best_params_dict['solref_dampratio']:.6f}],")
    print(f"    'solimp': [{best_params_dict['solimp_dmin']:.6f}, {best_params_dict['solimp_dmax']:.6f}, {best_params_dict['solimp_width']:.6f}, 0.5, 1.0],")
    print(f"    'noslip_iterations': {int(best_params_dict['noslip_iterations'])},")
    print(f"    'rocker_stiffness': {best_params_dict['rocker_stiffness']:.6f},")
    print(f"    'rocker_damping': {best_params_dict['rocker_damping']:.6f},")
    print(f"    'wheel_kp': {best_params_dict['wheel_kp']:.6f},")
    print(f"    'wheel_kv': {best_params_dict['wheel_kv']:.6f},")
    print(f"    'Br': {best_params_dict['Br']:.6f},")
    print(f"    'max_magnetic_distance': {best_params_dict['max_magnetic_distance']:.6f},")
    print(f"    'max_force_per_wheel': {best_params_dict['max_force_per_wheel']:.6f},")
    print("}")
    
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
        'max_force_per_wheel': best_result['params']['max_force_per_wheel'],
    }
    
    print(f"Best cost: {best_result['cost']:.6f}")
    print("Launching viewer...")
    
    sim_optimizer.run_simulation(
        best_sim_params,
        visualize=True,
        mode=MODE
    )