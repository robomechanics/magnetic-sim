"""
sim_opt_config.py — configuration for the sim lift optimizer.

Cost (lower = better):
  50% — mean absolute Z drift of body during lift hold
  25% — mean absolute X drift
  25% — mean absolute Y drift
"""

from skopt.space import Real

# ── Optimization settings ─────────────────────────────────────────────────────

N_CALLS              = 300
BATCH_SIZE           = 16
CMAES_SIGMA0         = 0.3
OPTIMIZER_RANDOM_STATE = 42

# ── Search space ──────────────────────────────────────────────────────────────

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
    Real(800.0, 1500,  "log-uniform", name="max_force_per_wheel"),
]

# ── Default / warm-start params ───────────────────────────────────────────────

PARAMS = {
    'ground_friction':       [0.130563, 0.00256843, 1.13878e-05],
    'solref':                [0.00192329, 10.0],
    'solimp':                [0.361806, 0.9999, 0.00065636, 0.603088, 3.97107],
    'noslip_iterations':     31,
    'noslip_tolerance':      2.42366e-05,
    'margin':                0.00149908,
    'Br':                    1.32665,
    'max_magnetic_distance': 0.0706161,
    'max_force_per_wheel':   917.216,
}


# ── Space → PARAMS ────────────────────────────────────────────────────────────

def point_to_params(point: list) -> dict:
    """Map a flat CMA-ES point (14 values) to a PARAMS dict."""
    (
        sliding_friction, torsional_friction, rolling_friction,
        solref_timeconst,
        solimp_dmin, solimp_width, solimp_midpoint, solimp_power,
        noslip_iterations, noslip_tolerance,
        margin,
        Br, max_magnetic_distance, max_force_per_wheel,
    ) = point

    return {
        'ground_friction':       [sliding_friction, torsional_friction, rolling_friction],
        'solref':                [solref_timeconst, 10.0],
        'solimp':                [solimp_dmin, 0.9999, solimp_width, solimp_midpoint, solimp_power],
        'noslip_iterations':     int(round(noslip_iterations)),
        'noslip_tolerance':      noslip_tolerance,
        'margin':                margin,
        'Br':                    Br,
        'max_magnetic_distance': max_magnetic_distance,
        'max_force_per_wheel':   max_force_per_wheel,
    }


# ── Cost function ─────────────────────────────────────────────────────────────

def calculate_cost(mean_abs_x: float, mean_abs_y: float, mean_abs_z: float) -> dict:
    """
    Weighted body drift cost during lift hold.
      50% Z, 25% X, 25% Y  (all in metres)
    """
    x_cost = 0.25 * mean_abs_x
    y_cost = 0.25 * mean_abs_y
    z_cost = 0.50 * mean_abs_z
    total  = x_cost + y_cost + z_cost
    return {
        "total_cost": total,
        "x_drift":    mean_abs_x,
        "y_drift":    mean_abs_y,
        "z_drift":    mean_abs_z,
        "x_cost":     x_cost,
        "y_cost":     y_cost,
        "z_cost":     z_cost,
    }