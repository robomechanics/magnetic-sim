"""
pulloff_config.py — Single source of truth for all pull-off sim/viewer/optimizer parameters.
Import this in pulloff_sim.py, pulloff_viewer.py, and pulloff_optimizer.py.
"""

import numpy as np
from skopt.space import Real

# ── Physics constants ────────────────────────────────────────────────────────
MU_0          = 4 * np.pi * 1e-7
MAGNET_VOLUME = np.pi * (0.0315 ** 2) * 0.025   # r=31.5mm, h=25mm

# ── Scene / XML ───────────────────────────────────────────────────────────────
SCENE_XML        = "mwc_mjcf/scene.xml"
MAGNET_BODY_NAME = "1103___pp___aws_pem_215lbs__eml63mm_24"
PLATE_GEOM_NAME  = "plate_geom"

# ── Simulation timing ─────────────────────────────────────────────────────────
TIMESTEP      = 0.0005   # s  — 2000 Hz
PULL_RATE     = 20.0     # N/s — default ramp rate (overridable via CLI)
SETTLE_TIME   = 2.0      # s  — phase 1: gravity only (0→1.0s), phase 2: mag (1.0→2.0s)
SIM_DURATION  = 20.0    # s  — hard stop
DETACH_DIST   = 10.0     # mm — min COM-Z displacement to count as detached
DETACH_HOLD   = 1.0      # s  — must stay above DETACH_DIST for this long

# ── Viewer ────────────────────────────────────────────────────────────────────
REAL_TIME_FACTOR   = 2.0
ARROW_RADIUS       = 0.004
MAG_ARROW_SCALE    = 0.001   # m per N — blue magnetic arrow length scaling
FORCE_ARROW_SCALE  = 0.005   # m per 10 N — red pull arrow length scaling
TELEMETRY_INTERVAL = 0.1     # s — terminal print interval

# ── Optimization target ───────────────────────────────────────────────────────
# Target pull-off force in Newtons.
# EML63mm-12 — Round Permanent Electromagnet 63 mm Dia. 12 V DC — Holding 215 lbs
GOAL_FORCE = 956.37   # N

# ── CMA-ES optimizer settings ─────────────────────────────────────────────────
N_CALLS               = 200
BATCH_SIZE            = 20
CMAES_SIGMA0          = 0.3
OPTIMIZER_RANDOM_STATE = 42
PULL_RATE_OPT         = 40.0   # N/s — fixed pull rate used during optimization

# ── Parameter search space (13 dims) ─────────────────────────────────────────
# solimp_dmax is fixed at 0.9999.
# solref_dampratio is fixed at SOLREF_DAMPRATIO_FIXED (not tuned).
# dof_damping excluded (freejoint body).
# APPLY_FORCE / APPLY_MOMENT / xy_drift not applicable — pure Z pull-off.
space: list = [
    Real(0.01,  2.0,   "log-uniform", name="sliding_friction"),
    Real(1e-6,  10.0,  "log-uniform", name="torsional_friction"),
    Real(1e-6,  1e-3,  "log-uniform", name="rolling_friction"),
    Real(1e-5,  1.0,   "log-uniform", name="solref_timeconst"),
    Real(0.001, 0.999, "uniform",     name="solimp_dmin"),
    Real(1e-7,  1.0,   "log-uniform", name="solimp_width"),
    Real(0.01,  0.99,  "uniform",     name="solimp_midpoint"),
    Real(2.0,   7.0,   "uniform",     name="solimp_power"),
    Real(0,     60,    "uniform",     name="noslip_iterations"),
    Real(1e-6,  1e-3,  "log-uniform", name="noslip_tolerance"),
    Real(0.0,   0.005, "uniform",     name="margin"),
    Real(0.5,   2.0,   "log-uniform", name="Br"),
    Real(0.012, 0.1,   "log-uniform", name="max_magnetic_distance"),
    Real(300.0, 1200,  "log-uniform", name="max_force_per_wheel"),
]

# Fixed solref damping ratio (not tuned)
SOLREF_DAMPRATIO_FIXED = 10.0


def point_to_params(point: list | dict) -> dict:
    """Convert a raw CMA-ES point (list or named dict) to a pull-off PARAMS dict.

    solimp layout: [dmin, dmax=0.9999, width, midpoint, power]
    solref layout: [timeconst, dampratio=SOLREF_DAMPRATIO_FIXED]
    ground_friction layout: [sliding, torsional, rolling]
    """
    if isinstance(point, list):
        p = {dim.name: val for dim, val in zip(space, point)}
    else:
        p = point

    return {
        "ground_friction":       [p["sliding_friction"], p["torsional_friction"], p["rolling_friction"]],
        "solref":                [p["solref_timeconst"], SOLREF_DAMPRATIO_FIXED],
        "solimp":                [p["solimp_dmin"], 0.9999, p["solimp_width"], p["solimp_midpoint"], p["solimp_power"]],
        "noslip_iterations":     int(round(p["noslip_iterations"])),
        "noslip_tolerance":      p["noslip_tolerance"],
        "margin":                p["margin"],
        "Br":                    p["Br"],
        "max_magnetic_distance": p["max_magnetic_distance"],
        "max_force_per_wheel":   p["max_force_per_wheel"],
    }


def calculate_cost(pulloff_force: float) -> dict:
    """Cost solely based on shortfall from GOAL_FORCE (lower = better).

    True no-attachment (pulloff_force == 0): hard penalty of 9999.
    Achieved >= goal: zero cost (held through full ramp).
    """
    goal = GOAL_FORCE

    if pulloff_force == 0.0:
        return {
            "total_cost":  9999.0,
            "detach_cost": 9999.0,
            "achieved":    0.0,
            "goal":        goal,
            "shortfall":   1.0,
        }

    shortfall  = max(0.0, (goal - pulloff_force) / goal)
    total_cost = shortfall

    return {
        "total_cost":  total_cost,
        "detach_cost": total_cost,
        "achieved":    pulloff_force,
        "goal":        goal,
        "shortfall":   shortfall,
    }


# ── Bayesian-optimised parameter presets ─────────────────────────────────────
# Set ACTIVE_PRESET to one of: 'hold', 'drive_sideways', 'drive_up'

ACTIVE_PRESET = 'pull_off'   # set to one of: 'hold', 'drive_sideways', 'drive_up'

PARAM_PRESETS = {
    'hold': {
        'ground_friction':       [0.9, 0.034585, 0.001734],
        'solref':                [0.000572, 10.000000],
        'solimp':                [0.860956, 0.9999, 0.000100, 0.5, 1.0],
        'noslip_iterations':     20,
        'noslip_tolerance':      1e-6,
        'margin':                0.0,
        'Br':                    1.332000,
        'max_magnetic_distance': 0.011413,
        'max_force_per_wheel':   139.324024,
    },
    'drive_sideways': {
        'ground_friction':       [0.975785, 0.000342, 0.000372],
        'solref':                [0.000129, 33.807584],
        'solimp':                [0.875461, 0.9999, 0.002193, 0.5, 1.0],
        'noslip_iterations':     20,
        'noslip_tolerance':      1e-6,
        'margin':                0.0,
        'Br':                    1.476441,
        'max_magnetic_distance': 0.029736,
        'max_force_per_wheel':   264.173934,
    },
    'drive_up': {
        'ground_friction':       [0.973464, 0.000202, 0.000010],
        'solref':                [0.000100, 10.000000],
        'solimp':                [0.926007, 0.9999, 0.001311, 0.5, 1.0],
        'noslip_iterations':     23,
        'noslip_tolerance':      1e-6,
        'margin':                0.0,
        'Br':                    1.599175,
        'max_magnetic_distance': 0.018389,
        'max_force_per_wheel':   205.270447,
    },
    'pull_off':{
        'ground_friction':       [0.130563, 0.00256843, 1.13878e-05],
        'solref':                [0.00192329, 10.000000],
        'solimp':                [0.361806, 0.9999, 0.00065636, 0.603088, 3.97107],
        'noslip_iterations':     31,
        'noslip_tolerance':      2.42366e-05,
        'margin':                0.00149908,
        'Br':                    1.32665,
        'max_magnetic_distance': 0.0706161,
        'max_force_per_wheel':   917.216,
            }
}

PARAMS = PARAM_PRESETS[ACTIVE_PRESET]