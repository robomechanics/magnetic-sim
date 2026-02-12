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
        "actuator_mode": "position",
        "actuator_target": 0.0,
        "cost_function": "minimize_slip",
        "sim_duration": 5.0,
        "settle_time": 0.2,
        "target_slip": 0.0,
    },
    "drive_sideways": {
        "actuator_mode": "velocity",
        "actuator_target_ms": 2.0,        # m/s linear velocity
        "cost_function": "drive_side",
        "sim_duration": 10.0,
        "settle_time": 0.2,
    },
}

# Derived: convert m/s -> rad/s for the drive target
MODES["drive_sideways"]["actuator_target_rads"] = (
    MODES["drive_sideways"]["actuator_target_ms"] / WHEEL_RADIUS
)

DEFAULT_MODE = "drive_sideways"