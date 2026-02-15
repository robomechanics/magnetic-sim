"""
replay.py - Quick replay of optimization results
Usage: python replay.py [--rank RANK] [--duration DURATION] [--mode MODE]
"""

import csv
import argparse
import viewer
from config import DEFAULT_MODE


def main():
    parser = argparse.ArgumentParser(description="Replay optimization result")
    parser.add_argument("--rank", type=int, default=1, help="Rank to replay (1=best)")
    parser.add_argument("--duration", type=float, default=None, help="Duration (s)")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, help="Mode: hold, drive_sideways, drive_up")
    args = parser.parse_args()
    
    # Ask user: original or optimized params
    print("\n1. Use optimized parameters from CSV")
    print("2. Use default/original parameters")
    choice = input("Select (1 or 2): ").strip()
    
    if choice == "2":
        # Default parameters
        print("\nUsing default parameters...")
        params = {
            'ground_friction': [0.95, 0.01, 0.01],
            'solref': [0.0004, 25.0],
            'solimp': [0.9, 0.95, 0.001, 0.5, 1.0],
            'noslip_iterations': 15,
            'rocker_stiffness': 30.0,
            'rocker_damping': 1.0,
            'wheel_kp': 10.0,
            'wheel_kv': 1.0,
            'Br': 1.48,
            'max_magnetic_distance': 0.01
        }
        viewer.visualize_simulation(params, args.duration, mode=args.mode)
        
    elif choice == "1":
        # Load from CSV
        try:
            with open(f'optimization_results_{args.mode}.csv', 'r') as f:
                results = list(csv.DictReader(f))
        except FileNotFoundError:
            print(f"Error: optimization_results_{args.mode}.csv not found.")
            return
        
        if args.rank > len(results):
            print(f"Error: Rank {args.rank} out of bounds (max: {len(results)})")
            return
        
        s = results[args.rank - 1]
        print(f"\nReplaying Rank #{args.rank}")
        print(f"Cost: {float(s['cost']):.6f}\n")
        
        params = {
            'ground_friction': [float(s['sliding_friction']), float(s['torsional_friction']), float(s['rolling_friction'])],
            'solref': [float(s['solref_timeconst']), float(s['solref_dampratio'])],
            'solimp': [float(s['solimp_dmin']), float(s['solimp_dmax']), float(s['solimp_width']), 0.5, 1.0],
            'noslip_iterations': int(s['noslip_iterations']),
            'Br': float(s['Br']),
            'max_magnetic_distance': float(s['max_magnetic_distance']),
            'rocker_stiffness': float(s['rocker_stiffness']),
            'rocker_damping': float(s['rocker_damping']),
            'wheel_kp': float(s['wheel_kp']),
            'wheel_kv': float(s['wheel_kv']),
            'max_force_per_wheel': float(s['max_force_per_wheel']),
        }
        
        viewer.visualize_simulation(params, args.duration, mode=args.mode)
        
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()