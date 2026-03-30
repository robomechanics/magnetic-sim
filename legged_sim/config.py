"""
config.py — Single source of truth for adhesion sim/viewer parameters.
"""

import re
import numpy as np

MAG_ENABLED = True

# ── Physics constants ─────────────────────────────────────────────────────────
MU_0          = 4 * np.pi * 1e-7
MAGNET_VOLUME = np.pi * (0.0315 ** 2) * 0.025   # r=31.5mm, h=25mm

# ── Scene / XML ───────────────────────────────────────────────────────────────
SCENE_XML         = "mwc_mjcf/scene.xml"
ROBOT_XML         = "mwc_mjcf/robot.xml"
ROBOT_ORIGINAL_XML = "mwc_mjcf/robot_original.xml"
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

# ── Joint geometry baking ─────────────────────────────────────────────────────
#
# Zero-pose vectors for each leg's knee geom endpoint (= wrist_link pos in knee frame).
# These are the original XML values before any angle baking.
# Format: (x, y, z) in metres.
#
KNEE_GEOM_ZERO = {
    'FL': np.array([ 0.04911,  0.04911,  0.08797]),
    'FR': np.array([ 0.04911, -0.04911,  0.08797]),
    'BL': np.array([-0.04911,  0.04911,  0.08797]),
    'BR': np.array([-0.04911, -0.04911,  0.08797]),
}

# Knee joint axes per leg (unit vectors, hinge axis in parent frame)
KNEE_AXIS = {
    'FL': np.array([ 0.7071068, -0.7071068,  0.0]),
    'FR': np.array([-0.7071068, -0.7071068,  0.0]),
    'BL': np.array([ 0.7071068,  0.7071068,  0.0]),
    'BR': np.array([-0.7071068,  0.7071068,  0.0]),
}

# Angle (degrees) to bake into each leg's knee geom endpoint.
# Positive = flex toward body, negative = extend outward.
KNEE_BAKE_DEG = {
    'FL': -0.0,
    'FR': -0.0,
    'BL': -0.0,
    'BR': -0.0,
}


def _rodrigues(v, k, theta_deg):
    """Rotate vector v by theta_deg degrees about unit axis k (Rodrigues formula)."""
    theta = np.radians(theta_deg)
    k = k / np.linalg.norm(k)
    return (v * np.cos(theta) +
            np.cross(k, v) * np.sin(theta) +
            k * np.dot(k, v) * (1.0 - np.cos(theta)))


def _fmt(v):
    """Format a 3-vector as 5-decimal XML string."""
    return f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}"


def bake_joint_angles(xml_path=None):
    """
    Recompute knee geom endpoints from KNEE_GEOM_ZERO + KNEE_BAKE_DEG using
    Rodrigues rotation, then overwrite the fromto and child body pos attributes
    directly in the robot XML file.

    Called once at sim startup — no keyframe needed.
    """
    if xml_path is None:
        xml_path = ROBOT_XML

    original_path = xml_path.replace("robot.xml", "robot_original.xml")

    with open(original_path, 'r') as f:   # always read from clean copy
        xml = f.read()

    for leg in ('FL', 'FR', 'BL', 'BR'):
        v0     = KNEE_GEOM_ZERO[leg]
        axis   = KNEE_AXIS[leg]
        angle  = KNEE_BAKE_DEG[leg]

        new_end = _rodrigues(v0, axis, angle)
        new_str = _fmt(new_end)

        old_str = _fmt(v0)

        # Replace knee_geom fromto end:  "0 0 0  <old_end>"
        xml = re.sub(
            rf'(name="knee_geom_{leg}"[^/]*?fromto=")0 0 0\s+{re.escape(old_str)}',
            rf'\g<1>0 0 0  {new_str}',
            xml, flags=re.DOTALL
        )

        # Replace wrist_link_XX pos (same vector, child body placement)
        xml = re.sub(
            rf'(name="wrist_link_{leg}"\s+pos="){re.escape(old_str)}"',
            rf'\g<1>{new_str}"',
            xml
        )

        print(f"[bake] {leg} knee: {old_str}  →  {new_str}")
    
    with open(xml_path, 'w') as f:        # write to working robot.xml
        f.write(xml)

    print(f"[bake] Written to {xml_path}")