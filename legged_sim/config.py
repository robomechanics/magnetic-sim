"""
config.py — Single source of truth for adhesion sim/viewer parameters.
"""

import numpy as np

MAG_ENABLED = False

# ── Physics constants ─────────────────────────────────────────────────────────
MU_0          = 4 * np.pi * 1e-7
MAGNET_VOLUME = np.pi * (0.0315 ** 2) * 0.025   # r=31.5mm, h=25mm

# ── Scene / XML ───────────────────────────────────────────────────────────────
SCENE_XML         = "mwc_mjcf/scene.xml"
MAGNET_BODY_NAMES = ["electromagnet_BL", "electromagnet_FL", "electromagnet_BR", "electromagnet_FR"]
PLATE_GEOM_NAME   = "wall"

# ── Simulation timing ─────────────────────────────────────────────────────────
TIMESTEP    = 0.0005   # s — 2000 Hz
SETTLE_TIME = 2.0      # s — phase 1: gravity only (0→1.0s), phase 2: mag (1.0→2.0s)
SIM_DURATION = 10.0    # s — hard stop

# ── Viewer ────────────────────────────────────────────────────────────────────
REAL_TIME_FACTOR   = 2.0
ARROW_RADIUS       = 0.004
MAG_ARROW_SCALE    = 0.001   # m per N
TELEMETRY_INTERVAL = 0.1     # s

# ── Parameters ────────────────────────────────────────────────────────────────
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