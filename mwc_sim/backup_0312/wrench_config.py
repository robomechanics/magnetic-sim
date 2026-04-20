"""
wrench_config.py — Single source of truth for all wrench sim/viewer parameters.
Import this in wrench_sim.py and wrench_viewer.py instead of defining values locally.
"""

import numpy as np

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
PULL_RATE     = 20.0     # N/s — default ramp rate (overridable via CLI)
SETTLE_TIME   = 1.0      # s  — phase 1: gravity only (0 → 0.5s), phase 2: mag (0.5 → 1.0s)
SIM_DURATION  = 50.0     # s  — hard stop
DETACH_HOLD   = 0.5      # s  — f_mag must stay < DETACH_THRESHOLD for this long
DETACH_THRESHOLD = 0.01  # N  — magnetic force below this counts as detached

# ── Wrench application toggles ───────────────────────────────────────────────
APPLY_FORCE  = False # apply horizontal force at stick tip
APPLY_MOMENT = True   # apply resulting torque at magnet COM

# ── Viewer ────────────────────────────────────────────────────────────────────
REAL_TIME_FACTOR    = 0.8    # <1 = slow-mo, 1 = real time
ARROW_RADIUS        = 0.004  # m — shaft radius for all force/torque arrows
TORQUE_ARROW_SCALE  = 0.05   # m per N·m — green torque arrow length scaling
FORCE_ARROW_SCALE   = 0.005  # m per 10 N — red force arrow length scaling
MAG_ARROW_SCALE     = 0.001  # m per N — blue magnetic arrow length scaling
TELEMETRY_INTERVAL  = 0.1    # s — terminal print interval

# ── Bayesian-optimised parameter presets ─────────────────────────────────────
# Set ACTIVE_PRESET to one of: 'hold', 'drive_sideways', 'drive_up'

# ACTIVE_PRESET = 'hold'
# ACTIVE_PRESET = 'drive_sideways'
ACTIVE_PRESET = 'drive_up'

PARAM_PRESETS = {
    'hold': {
        'ground_friction':       [0.9, 0.034585, 0.001734],
        'solref':                [0.000572, 10.000000],
        'solimp':                [0.860956, 0.987761, 0.000100, 0.5, 1.0],
        'noslip_iterations':     20,
        'Br':                    1.332000,
        'max_magnetic_distance': 0.011413,
        'max_force_per_wheel':   139.324024,
    },
    'drive_sideways': {
        'ground_friction':       [0.975785, 0.000342, 0.000372],
        'solref':                [0.000129, 33.807584],
        'solimp':                [0.875461, 0.934908, 0.002193, 0.5, 1.0],
        'noslip_iterations':     20,
        'Br':                    1.476441,
        'max_magnetic_distance': 0.029736,
        'max_force_per_wheel':   264.173934,
    },
    'drive_up': {
        'ground_friction':       [0.973464, 0.000202, 0.000010],
        'solref':                [0.000100, 10.000000],
        'solimp':                [0.926007, 0.943979, 0.001311, 0.5, 1.0],
        'noslip_iterations':     23,
        'Br':                    1.599175,
        'max_magnetic_distance': 0.018389,
        'max_force_per_wheel':   205.270447,
    },
}

PARAMS = PARAM_PRESETS[ACTIVE_PRESET]