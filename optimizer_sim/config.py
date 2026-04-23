"""
config.py - Mode configuration and optimization settings for Sally magnetic wall-climbing simulation.
"""

from __future__ import annotations

import numpy as np
from skopt.space import Integer, Real

# DEFAULT_MODE = "hold"
# DEFAULT_MODE = "drive_up"
DEFAULT_MODE = "drive_sideways"
DEFAULT_XML_PATH = "XML/scene.xml"
RESULTS_DIR = "results"

# Optimization settings
N_CALLS = 160
BATCH_SIZE = 8
CMAES_SIGMA0 = 0.35
OPTIMIZER_RANDOM_STATE = 42
COST_FAILURE = 1e6
WHEEL_RADIUS = 0.025

MODES = {
    "hold": {
        "actuator_mode": "position",
        "actuator_target": 0.0,
        "cost_function": "minimize_slip",
        "sim_duration": 5.0,
        "settle_time": 0.2,
        "pivot_angle": 0.0,
        "target_velocity_xyz": np.array([0.0, 0.0, 0.0]),  # no motion target — hold still
        "tracking_axis": None,                              # no primary axis for hold mode
    },
    "drive_sideways": {
        "actuator_mode": "velocity",
        "actuator_target_ms": 1.0,
        "cost_function": "drive",
        "sim_duration": 10.0,
        "settle_time": 0.2,
        "pivot_angle": 0.0,
        "target_velocity_xyz": np.array([0.0, 1.0, 0.0]),  # +Y at 1 m/s
        "tracking_axis": 1,                                 # Y is the primary axis
    },
    "drive_up": {
        "actuator_mode": "velocity",
        "actuator_target_ms": 1.0,
        "cost_function": "drive",
        "sim_duration": 10.0,
        "settle_time": 0.2,
        "pivot_angle": -1.5708,
        "target_velocity_xyz": np.array([0.0, 0.0, 1.0]),  # +Z at 1 m/s
        "tracking_axis": 2,                                 # Z is the primary axis
    },
}

for mode_name in ["drive_sideways", "drive_up"]:
    MODES[mode_name]["actuator_target_rads"] = (
        MODES[mode_name]["actuator_target_ms"] / WHEEL_RADIUS
    )

BASELINE_PARAMS = {
    "ground_friction": [0.95, 0.01, 0.01],
    "solref": [0.0004, 25.0],
    "solimp": [0.9, 0.95, 0.001, 0.5, 1.0],
    "noslip_iterations": 15,
    "rocker_stiffness": 500.0,  # FIX: was 30.0, outside search space [100, 1000]
    "rocker_damping": 1.0,
    "wheel_kp": 10.0,
    "wheel_kv": 1.0,
    "Br": 1.48,
    "max_magnetic_distance": 0.010,
    "max_force_per_wheel": 50.0,
}

DEFAULT_PARAMS = BASELINE_PARAMS.copy()

SEARCH_SPACE = [
    Real(1.48 * 0.9, 1.48 * 1.1, prior="uniform", name="Br"),
    Real(0.0001, 0.0008, prior="uniform", name="solref_timeconst"),
    Real(10.0, 50.0, prior="uniform", name="solref_dampratio"),
    Real(0.8, 0.99, prior="uniform", name="solimp_dmin"),
    Real(0.9, 1.0, prior="uniform", name="solimp_dmax"),
    Real(1e-4, 1e-2, prior="log-uniform", name="solimp_width"),
    Real(0.9, 1.0, prior="uniform", name="sliding_friction"),
    Real(1e-5, 0.1, prior="log-uniform", name="torsional_friction"),
    Real(1e-5, 0.1, prior="log-uniform", name="rolling_friction"),
    Real(100.0, 1000.0, prior="uniform", name="rocker_stiffness"),
    Real(0.1, 5.0, prior="log-uniform", name="rocker_damping"),
    Real(1.0, 50.0, prior="log-uniform", name="wheel_kp"),
    Real(0.1, 10.0, prior="log-uniform", name="wheel_kv"),
    Real(0.005, 0.1, prior="log-uniform", name="max_magnetic_distance"),
    Integer(5, 30, name="noslip_iterations"),
    Real(100.0, 1000.0, prior="uniform", name="max_force_per_wheel"),  # RESTORED: old code [100,1000] — strong adhesion prevents slip
]