"""
config.py - Mode configuration for Sally magnetic wall-climbing robot.

Modes:
    hold  - Wheels stationary, optimize for minimum slip (wall adhesion)
    drive - Wheels spinning, optimize for sideways driving performance
"""

import numpy as np
from skopt.space import Real, Integer

# DEFAULT_MODE = "hold"
# DEFAULT_MODE = "drive_sideways"
DEFAULT_MODE = "drive_up"

N_CALLS = 20
WHEEL_RADIUS = 0.025

MODES = {
    "hold": {
        # this means we will set a constant position target (0) for the wheel joints to hold them in place
        # not actuator type 
        "actuator_mode": "position", 
        "actuator_target": 0.0,
        "cost_function": "minimize_slip",
        "sim_duration": 5.0,
        "settle_time": 0.2,
        "pivot_angle": 0.0,            # 0 - wheels sideways
    },
    "drive_sideways": {
        # velocity is a position ramp
        # produces approximately constant velocity behavior, because the PD controller is always “chasing” a moving angle target.
        "actuator_mode": "velocity",
        "actuator_target_ms": 2.0,
        "cost_function": "drive_side",
        "sim_duration": 5.0,
        "settle_time": 0.2,
        "pivot_angle": 0.0,            # np.pi/2 - wheels sideways
    },
    "drive_up": {
        "actuator_mode": "velocity",
        "actuator_target_ms": 2.0,
        "cost_function": "drive_up",
        "sim_duration": 10.0,
        "settle_time": 0.2,
        "pivot_angle": -1.5708,              # wheels forward (drive up)
    },
}

# Derived: convert m/s -> rad/s for the drive target
for mode_name in ["drive_sideways", "drive_up"]:
    MODES[mode_name]["actuator_target_rads"] = (
        MODES[mode_name]["actuator_target_ms"] / WHEEL_RADIUS
    )

# This set runs well with hold and sideways mode 
DEFAULT_PARAMS = {
    'ground_friction': [0.9876, 0.000592, 0.00001],
    'solref': [0.0008, 10.0],
    'solimp': [0.9094, 0.9946, 0.000667, 0.5, 1.0],
    'noslip_iterations': 28,
    'rocker_stiffness': 100.0,
    'rocker_damping': 2.3945,
    'wheel_kp': 1.1759,
    'wheel_kv': 5.7573,
    'Br': 1.628,
    'max_magnetic_distance': 0.03275,
    'max_force_per_wheel': 100.0,
}

# Bayesian optimization search space
SEARCH_SPACE = [
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
    Real(100.0, 1000.0, "uniform", name='rocker_stiffness'),
    Real(0.1, 5.0, "log-uniform", name='rocker_damping'),
    
    # Control gains (log-uniform)
    Real(1.0, 50.0, "log-uniform", name='wheel_kp'),
    Real(0.1, 10.0, "log-uniform", name='wheel_kv'),
    
    # Magnetic cutoff (log-uniform)
    Real(0.005, 0.1, "log-uniform", name='max_magnetic_distance'),
    
    # Solver iterations (integer, uniform)
    Integer(5, 30, name='noslip_iterations'),
    Real(20.0, 300.0, "log-uniform", name='max_force_per_wheel'),
]

