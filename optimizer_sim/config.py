"""
config.py - Mode configuration for Sally magnetic wall-climbing robot.

Modes:
    hold  - Wheels stationary, optimize for minimum slip (wall adhesion)
    drive - Wheels spinning, optimize for sideways driving performance
"""

import numpy as np

WHEEL_RADIUS = 0.025  # meters, from XML cylinder geom

MODES = {
    "hold": {
        # this means we will set a constant position target (0) for the wheel joints to hold them in place
        # not actuator type 
        "actuator_mode": "position", 
        "actuator_target": 0.0,
        "cost_function": "minimize_slip",
        "sim_duration": 5.0,
        "settle_time": 0.2,
        "pivot_angle": 1.5708,            # np.pi/2 - wheels sideways
    },
    "drive_sideways": {
        # velocity is a position ramp
        # produces approximately constant velocity behavior, because the PD controller is always “chasing” a moving angle target.
        "actuator_mode": "velocity",
        "actuator_target_ms": 2.0,
        "cost_function": "drive_side",
        "sim_duration": 10.0,
        "settle_time": 0.2,
        "pivot_angle": 1.5708,            # np.pi/2 - wheels sideways
    },
    "drive_up": {
        "actuator_mode": "velocity",
        "actuator_target_ms": 2.0,
        "cost_function": "drive_up",
        "sim_duration": 3.0,
        "settle_time": 0.2,
        "pivot_angle": -1.5708,              # wheels forward (drive up)
    },
}

# Derived: convert m/s -> rad/s for the drive target
for mode_name in ["drive_sideways", "drive_up"]:
    MODES[mode_name]["actuator_target_rads"] = (
        MODES[mode_name]["actuator_target_ms"] / WHEEL_RADIUS
    )

DEFAULT_MODE = "drive_up"