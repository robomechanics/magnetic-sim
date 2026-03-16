"""
wrench_config.py — Single source of truth for all wrench sim/viewer/optimizer parameters.
Import this in wrench_sim.py, wrench_viewer.py, and wrench_optimizer.py.
"""

import numpy as np
from skopt.space import Real

# ── Physics constants ────────────────────────────────────────────────────────
MU_0          = 4 * np.pi * 1e-7
MAGNET_VOLUME = np.pi * (0.0315 ** 2) * 0.025   # r=31.5mm, h=25mm

# ── Geometry ─────────────────────────────────────────────────────────────────
STICK_TIP_LOCAL = np.array([0.0, 0.0, 0.0825])   # tip in magnet body frame
LEVER_ARM       = np.linalg.norm(STICK_TIP_LOCAL) # 0.0825 m

# ── Scene / XML ───────────────────────────────────────────────────────────────
SCENE_XML        = "mwc_mjcf/scene.xml"
MAGNET_BODY_NAME = "1103___pp___aws_pem_215lbs__eml63mm_24"
PLATE_GEOM_NAME  = "plate_geom"
TIP_SITE_NAME    = "stick_tip"

# ── Simulation timing ─────────────────────────────────────────────────────────
TIMESTEP      = 0.0005   # s  — 2000 Hz
PULL_RATE     = 40.0     # N/s — default ramp rate (overridable via CLI)
SETTLE_TIME   = 1.0      # s  — phase 1: gravity only (0 → 0.5s), phase 2: mag (0.5 → 1.0s)
SIM_DURATION  = 40.0     # s  — hard stop
DETACH_HOLD   = 0.5      # s  — f_mag must stay < DETACH_THRESHOLD for this long
DETACH_THRESHOLD = 0.01  # N  — magnetic force below this counts as detached

# ── Wrench application toggles ───────────────────────────────────────────────
APPLY_FORCE  = False  # apply horizontal force at stick tip
APPLY_MOMENT = True   # apply resulting torque at magnet COM

# ── Viewer ────────────────────────────────────────────────────────────────────
REAL_TIME_FACTOR    = 0.8    # <1 = slow-mo, 1 = real time
ARROW_RADIUS        = 0.004  # m — shaft radius for all force/torque arrows
TORQUE_ARROW_SCALE  = 0.05   # m per N·m — green torque arrow length scaling
FORCE_ARROW_SCALE   = 0.005  # m per 10 N — red force arrow length scaling
MAG_ARROW_SCALE     = 0.001  # m per N — blue magnetic arrow length scaling
TELEMETRY_INTERVAL  = 0.1    # s — terminal print interval

# ── Optimization targets ──────────────────────────────────────────────────────
# Set one of these to match your physical measurement target.
# GOAL_FORCE  — target detach force in Newtons (used when APPLY_FORCE=True, APPLY_MOMENT=False)
# GOAL_WRENCH — target detach moment in N·m    (used when APPLY_MOMENT=True)
# Only the one matching your current APPLY_FORCE / APPLY_MOMENT toggle is used by calculate_cost.
GOAL_FORCE  = 956.37   # 956.37 N   — EML63mm-12 – Round Permanent Electromagnet 63 mm Dia. 12 volts DC – Holding 215 lbs
PEEL_R = 0.057 # m
GOAL_WRENCH = GOAL_FORCE * PEEL_R     # 56.22 N·m



# Cost weights (must sum to 1.0)
COST_WEIGHT_DETACH  = 0.30   # penalty for detaching before GOAL
COST_WEIGHT_XY_DRIFT = 0.70  # penalty for XY position change before detachment

# ── CMA-ES optimizer settings ─────────────────────────────────────────────────
N_CALLS              = 200   # total candidate evaluations
BATCH_SIZE           = 20     # CMA-ES population size
CMAES_SIGMA0         = 0.3    # initial step size
OPTIMIZER_RANDOM_STATE = 42
PULL_RATE_OPT        = 40.0   # N/s — fixed pull rate used during optimization

# ── Parameter search space (14 dims) ─────────────────────────────────────────
# solimp_dmax is fixed at 0.9999 (no delta_d).
# solref_dampratio is fixed at the preset value (not tuned).
# magnetic_moment_fudge and magnetic_field_fudge are excluded.
# dof_damping is excluded (single freejoint body, not applicable).
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
    Real(0.012,  0.1,   "log-uniform", name="max_magnetic_distance"),
    Real(300.0,  1200, "log-uniform", name="max_force_per_wheel"),
]

# Fixed solref damping ratio (not tuned)
SOLREF_DAMPRATIO_FIXED = 10.0


def point_to_params(point: list | dict) -> dict:
    """Convert a raw CMA-ES point (list or named dict) to a wrench PARAMS dict.

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


def calculate_cost(detach_force: float, detach_moment: float, xy_drift: float) -> dict:
    # Select the right metric and goal based on which mode is active
    if APPLY_MOMENT and not APPLY_FORCE:
        achieved = detach_moment
        goal     = GOAL_WRENCH
    elif APPLY_FORCE and not APPLY_MOMENT:
        achieved = detach_force
        goal     = GOAL_FORCE
    else:
        achieved = detach_force
        goal     = GOAL_FORCE

    # True no-attachment: magnet never engaged at all (both zero means
    # apply_mag returned 0 every step — Br too low, max_distance too small).
    # Distinguish from "held through full ramp" (achieved > 0, no detach).
    if detach_force == 0.0 and detach_moment == 0.0:
        return {
            "total_cost":  9999.0,
            "detach_cost": 9999.0,
            "drift_cost":  0.0,
            "achieved":    0.0,
            "goal":        goal,
            "shortfall":   1.0,
            "xy_drift":    xy_drift,
        }

    # No detachment and achieved >= goal: magnet held through the full ramp.
    # This is the ideal outcome — zero shortfall penalty.
    # Still penalize XY drift so the optimizer prefers stable adhesion.
    shortfall = max(0.0, (goal - achieved) / goal)  # 0 if achieved >= goal

    detach_cost = COST_WEIGHT_DETACH * shortfall

    DRIFT_REFERENCE_M = 0.01
    drift_normalized  = min(xy_drift / DRIFT_REFERENCE_M, 1.0) if DRIFT_REFERENCE_M > 0 else 0.0
    drift_cost        = COST_WEIGHT_XY_DRIFT * drift_normalized

    total_cost = detach_cost + drift_cost

    return {
        "total_cost":   total_cost,
        "detach_cost":  detach_cost,
        "drift_cost":   drift_cost,
        "achieved":     achieved,
        "goal":         goal,
        "shortfall":    shortfall,
        "xy_drift":     xy_drift,
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
            },
    'combined':{
        'ground_friction':       [0.0895322, 0.00569811, 1.82912e-05],
        'solref':                [0.00292132, 10.000000],
        'solimp':                [0.651498, 0.9999, 0.000575084, 0.139903, 4.39965],
        'noslip_iterations':     30,
        'noslip_tolerance':      2.0137e-05,
        'margin':                0.00457664,
        'Br':                    1.2058,
        'max_magnetic_distance': 0.0145149,
        'max_force_per_wheel':   912.252,
            },
}

PARAMS = PARAM_PRESETS[ACTIVE_PRESET]