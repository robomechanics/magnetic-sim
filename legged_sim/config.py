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
# Zero-pose endpoint vectors read directly from robot_original.xml (knee geom fromto,
# second point = wrist_link body pos in knee_link frame).
# Format: (x, y, z) in metres.
#
KNEE_GEOM_ZERO = {
    'FL': np.array([ 0.07280,  0.07280, -0.04432]),
    'FR': np.array([ 0.07280, -0.07280, -0.04432]),
    'BL': np.array([-0.07280,  0.07280, -0.04432]),
    'BR': np.array([-0.07280, -0.07280, -0.04432]),
}

# Knee joint axes per leg (unit vectors, hinge axis in parent frame)
KNEE_AXIS = {
    'FL': np.array([ 0.7071068, -0.7071068,  0.0]),
    'FR': np.array([-0.7071068, -0.7071068,  0.0]),
    'BL': np.array([ 0.7071068,  0.7071068,  0.0]),
    'BR': np.array([-0.7071068,  0.7071068,  0.0]),
}

# Angle (degrees) to bake into each leg's knee geom endpoint.
KNEE_BAKE_DEG = {
    'FL': -0.0,
    'FR': -0.0,
    'BL': -0.0,
    'BR': -0.0,
}

# ── Wrist geometry baking ─────────────────────────────────────────────────────
#
# Zero-pose endpoint vectors read directly from robot_original.xml (wrist geom fromto,
# second point = EE body pos in wrist_link frame).
#
WRIST_GEOM_ZERO = {
    'FL': np.array([ 0.07425,  0.07425, -0.18187]),
    'FR': np.array([ 0.07425, -0.07425, -0.18187]),
    'BL': np.array([-0.07425,  0.07425, -0.18187]),
    'BR': np.array([-0.07425, -0.07425, -0.18187]),
}

# Wrist joint axes per leg (same diagonal axes as knee)
WRIST_AXIS = {
    'FL': np.array([ 0.7071068, -0.7071068,  0.0]),
    'FR': np.array([-0.7071068, -0.7071068,  0.0]),
    'BL': np.array([ 0.7071068,  0.7071068,  0.0]),
    'BR': np.array([-0.7071068,  0.7071068,  0.0]),
}

# Angle (degrees) to bake into each leg's wrist geom endpoint.
WRIST_BAKE_DEG = {
    'FL': -0.0,
    'FR': -0.0,
    'BL': -0.0,
    'BR': -0.0,
}

# ── EE geometry baking ────────────────────────────────────────────────────────
#
# Zero-pose endpoint vector for the EE link geom (ee_link_geom_XX fromto second point
# = electromagnet body pos in EE_XX frame).
# Original XML: +X direction = (0.066, 0, 0).
# Electromagnet offset pos = (0.066075, 0, 0).
#
# The EE joint axis is expressed in EE_XX's local frame (verified numerically).
# EM_QUAT_ZERO varies per leg — FL/BR use (0.5, 0.5, 0.5, -0.5),
#                                BL/FR use (-0.5, 0.5, 0.5, 0.5).
#
EE_GEOM_ZERO = {
    'FL': np.array([0.066,    0.0, 0.0]),
    'FR': np.array([0.066,    0.0, 0.0]),
    'BL': np.array([0.066,    0.0, 0.0]),
    'BR': np.array([0.066,    0.0, 0.0]),
}

EM_POS_ZERO = {
    'FL': np.array([0.066075, 0.0, 0.0]),
    'FR': np.array([0.066075, 0.0, 0.0]),
    'BL': np.array([0.066075, 0.0, 0.0]),
    'BR': np.array([0.066075, 0.0, 0.0]),
}

# EE joint axes per leg (hinge axis in EE_XX frame)
EE_AXIS = {
    'FL': np.array([0.0, -0.7071068,  0.7071068]),
    'FR': np.array([0.0, -0.7071068, -0.7071068]),
    'BL': np.array([0.0,  0.7071068,  0.7071068]),
    'BR': np.array([0.0,  0.7071068, -0.7071068]),
}

# Original electromagnet quats (w x y z) per leg, from robot_original.xml:
#   FL: ( 0.5,  0.5,  0.5, -0.5)   FR: (-0.5,  0.5,  0.5,  0.5)
#   BL: (-0.5,  0.5,  0.5,  0.5)   BR: ( 0.5,  0.5,  0.5, -0.5)
EM_QUAT_ZERO = {
    'FL': np.array([ 0.5,  0.5,  0.5, -0.5]),
    'FR': np.array([-0.5,  0.5,  0.5,  0.5]),
    'BL': np.array([-0.5,  0.5,  0.5,  0.5]),
    'BR': np.array([ 0.5,  0.5,  0.5, -0.5]),
}

# Angle (degrees) to bake into each leg's EE link geom endpoint.
EE_BAKE_DEG = {
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


def _axis_angle_to_rot(k, theta_deg):
    """Rotation matrix from axis k and angle in degrees."""
    theta = np.radians(theta_deg)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _quat_to_rot(q):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def _rot_to_quat(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def _fmt(v):
    """Format a 3-vector as 5-decimal XML string."""
    return f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}"


def _fmt_quat(q):
    """Format a quaternion (w x y z) as 7-decimal XML string."""
    return f"{q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}"


def bake_joint_angles(xml_path=None):
    """
    Recompute knee, wrist, and EE geom endpoints from zero-pose vectors +
    bake angles using Rodrigues rotation, then overwrite the relevant
    fromto / pos / quat attributes directly in the robot XML file.

    Called once at sim startup — no keyframe needed.
    """
    if xml_path is None:
        xml_path = ROBOT_XML

    original_path = xml_path.replace("robot.xml", "robot_original.xml")

    with open(original_path, 'r') as f:   # always read from clean copy
        xml = f.read()

    for leg in ('FL', 'FR', 'BL', 'BR'):

        # ── Knee ──────────────────────────────────────────────────────────────
        v0    = KNEE_GEOM_ZERO[leg]
        axis  = KNEE_AXIS[leg]
        angle = KNEE_BAKE_DEG[leg]

        new_end = _rodrigues(v0, axis, angle)
        old_str = _fmt(v0)
        new_str = _fmt(new_end)

        xml = re.sub(
            rf'(name="knee_geom_{leg}"[^/]*?fromto=")0 0 0\s+{re.escape(old_str)}',
            rf'\g<1>0 0 0  {new_str}',
            xml, flags=re.DOTALL
        )
        xml = re.sub(
            rf'(name="wrist_link_{leg}"\s+pos="){re.escape(old_str)}"',
            rf'\g<1>{new_str}"',
            xml
        )
        print(f"[bake] {leg} knee:  {old_str}  →  {new_str}")

        # ── Wrist ─────────────────────────────────────────────────────────────
        v0    = WRIST_GEOM_ZERO[leg]
        axis  = WRIST_AXIS[leg]
        angle = WRIST_BAKE_DEG[leg]

        new_end = _rodrigues(v0, axis, angle)
        old_str = _fmt(v0)
        new_str = _fmt(new_end)

        xml = re.sub(
            rf'(name="wrist_geom_{leg}"[^/]*?fromto=")0 0 0\s+{re.escape(old_str)}',
            rf'\g<1>0 0 0  {new_str}',
            xml, flags=re.DOTALL
        )
        xml = re.sub(
            rf'(name="EE_{leg}"\s+pos="){re.escape(old_str)}"',
            rf'\g<1>{new_str}"',
            xml
        )
        print(f"[bake] {leg} wrist: {old_str}  →  {new_str}")

        # ── EE ────────────────────────────────────────────────────────────────
        axis  = EE_AXIS[leg]
        angle = EE_BAKE_DEG[leg]

        # EE link geom endpoint
        v0      = EE_GEOM_ZERO[leg]
        new_end = _rodrigues(v0, axis, angle)
        old_str = _fmt(v0)
        new_str = _fmt(new_end)

        xml = re.sub(
            rf'(name="ee_link_geom_{leg}"[^/]*?fromto=")0 0 0\s+{re.escape(old_str)}',
            rf'\g<1>0 0 0  {new_str}',
            xml, flags=re.DOTALL
        )
        print(f"[bake] {leg} ee:    {old_str}  →  {new_str}")

        # Electromagnet pos
        v0      = EM_POS_ZERO[leg]
        new_end = _rodrigues(v0, axis, angle)
        old_pos = _fmt(v0)
        new_pos = _fmt(new_end)

        xml = re.sub(
            rf'(name="electromagnet_{leg}"\s+pos="){re.escape(old_pos)}"',
            rf'\g<1>{new_pos}"',
            xml
        )

        # Electromagnet quat
        R_rot   = _axis_angle_to_rot(axis, angle)
        R_new   = R_rot @ _quat_to_rot(EM_QUAT_ZERO[leg])
        q_new   = _rot_to_quat(R_new)
        old_quat = _fmt_quat(EM_QUAT_ZERO[leg])
        new_quat = _fmt_quat(q_new)

        xml = re.sub(
            rf'(name="electromagnet_{leg}"[^>]*?quat="){re.escape(old_quat)}"',
            rf'\g<1>{new_quat}"',
            xml, flags=re.DOTALL
        )
        print(f"[bake] {leg} em:    pos {old_pos} → {new_pos}  |  quat → {new_quat}")

    with open(xml_path, 'w') as f:        # write to working robot.xml
        f.write(xml)

    print(f"[bake] Written to {xml_path}")