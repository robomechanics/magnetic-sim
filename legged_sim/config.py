"""config.py — Single source of truth for adhesion sim/viewer parameters."""

import re
import numpy as np

MAG_ENABLED = True

MU_0          = 4 * np.pi * 1e-7
MAGNET_VOLUME = np.pi * (0.0315 ** 2) * 0.025

SCENE_XML          = "mwc_mjcf/scene.xml"
ROBOT_XML          = "mwc_mjcf/robot.xml"
ROBOT_ORIGINAL_XML = "mwc_mjcf/robot_original.xml"
MAGNET_BODY_NAMES  = ["electromagnet_BL", "electromagnet_FL", "electromagnet_BR", "electromagnet_FR"]

TIMESTEP           = 0.0005
SETTLE_TIME        = 2.0
SIM_DURATION       = 30.0

REAL_TIME_FACTOR   = 2.0
ARROW_RADIUS       = 0.004
MAG_ARROW_SCALE    = 0.001
TELEMETRY_INTERVAL = 0.5

# PARAMS = {
#     'ground_friction':       [0.130563, 0.00256843, 1.13878e-05],
#     'solref':                [0.00192329, 10.0],
#     'solimp':                [0.361806, 0.9999, 0.00065636, 0.603088, 3.97107],
#     'noslip_iterations':     31,
#     'noslip_tolerance':      2.42366e-05,
#     'margin':                0.00149908,
#     'Br':                    1.32665,
#     'max_magnetic_distance': 0.0706161,
#     'max_force_per_wheel':   917.216,
# }

PARAMS = {
    'ground_friction':       [0.054176, 0.181922, 7.4833e-06],
    'solref':                [0.00854906, 10.0],
    'solimp':                [0.153179, 0.9999, 0.000654778, 0.417417, 3.69318],
    'noslip_iterations':     30,
    'noslip_tolerance':      7.13222e-05,
    'margin':                0.00388617,
    'Br':                    1.78509,
    'max_magnetic_distance': 0.0816543,
    'max_force_per_wheel':   803.941,
}

JOINT_DAMPING  = 5.0
JOINT_ARMATURE = 0.05
SERVO_KP       = 200.0
SERVO_KV       = 10.0

STANCE_KP      = 600.0
STANCE_KV      = 20.0
STANCE_DAMPING = 15.0

SWING_DURATION    = 3
SWING_LIFT_HEIGHT = 0.1
STEP_LENGTH       = 0.3
DEMAGNETIZE_HOLD  = 0.10
MAGNETIZE_HOLD    = 0.15

# ── sequence selector ───────────────────────────────────────────────────
# Options: "basic"   lift → swing → reach → drop
#          "orient"  lift → swing → reach (EE face +X) → drop
SEQUENCE = "orient"

KNEE_GEOM_ZERO = {
    'FL': np.array([ 0.07280,  0.07280, -0.04432]),
    'FR': np.array([ 0.07280, -0.07280, -0.04432]),
    'BL': np.array([-0.07280,  0.07280, -0.04432]),
    'BR': np.array([-0.07280, -0.07280, -0.04432]),
}
KNEE_AXIS = {
    'FL': np.array([ 0.7071068, -0.7071068, 0.0]),
    'FR': np.array([-0.7071068, -0.7071068, 0.0]),
    'BL': np.array([ 0.7071068,  0.7071068, 0.0]),
    'BR': np.array([-0.7071068,  0.7071068, 0.0]),
}
KNEE_BAKE_DEG = {'FL': 0.0, 'FR': 0.0, 'BL': 0.0, 'BR': 0.0}

WRIST_GEOM_ZERO = {
    'FL': np.array([ 0.07425,  0.07425, -0.18187]),
    'FR': np.array([ 0.07425, -0.07425, -0.18187]),
    'BL': np.array([-0.07425,  0.07425, -0.18187]),
    'BR': np.array([-0.07425, -0.07425, -0.18187]),
}
WRIST_AXIS = {
    'FL': np.array([ 0.7071068, -0.7071068, 0.0]),
    'FR': np.array([-0.7071068, -0.7071068, 0.0]),
    'BL': np.array([ 0.7071068,  0.7071068, 0.0]),
    'BR': np.array([-0.7071068,  0.7071068, 0.0]),
}
WRIST_BAKE_DEG = {'FL': 0.0, 'FR': 0.0, 'BL': 0.0, 'BR': 0.0}

EE_GEOM_ZERO = {leg: np.array([0.066, 0.0, 0.0]) for leg in ('FL', 'FR', 'BL', 'BR')}
EM_POS_ZERO  = {leg: np.array([0.066075, 0.0, 0.0]) for leg in ('FL', 'FR', 'BL', 'BR')}
EE_AXIS = {
    'FL': np.array([0.0, -0.7071068,  0.7071068]),
    'FR': np.array([0.0, -0.7071068, -0.7071068]),
    'BL': np.array([0.0,  0.7071068,  0.7071068]),
    'BR': np.array([0.0,  0.7071068, -0.7071068]),
}
EM_QUAT_ZERO = {
    'FL': np.array([ 0.5,  0.5,  0.5, -0.5]),
    'FR': np.array([-0.5,  0.5,  0.5,  0.5]),
    'BL': np.array([-0.5,  0.5,  0.5,  0.5]),
    'BR': np.array([ 0.5,  0.5,  0.5, -0.5]),
}
EE_BAKE_DEG = {'FL': 0.0, 'FR': 0.0, 'BL': 0.0, 'BR': 0.0}


def _rodrigues(v, k, theta_deg):
    theta = np.radians(theta_deg)
    k = k / np.linalg.norm(k)
    return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))

def _axis_angle_to_rot(k, theta_deg):
    theta = np.radians(theta_deg)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def _quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])

def _rot_to_quat(R):
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

def _fmt(v):      return f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}"
def _fmt_quat(q): return f"{q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}"


def bake_joint_angles(xml_path=None):
    if xml_path is None:
        xml_path = ROBOT_XML

    with open(xml_path.replace("robot.xml", "robot_original.xml")) as f:
        xml = f.read()

    for leg in ('FL', 'FR', 'BL', 'BR'):
        for geom_name, zero_dict, axis_dict, bake_dict, body_name in [
            ('knee_geom',    KNEE_GEOM_ZERO,  KNEE_AXIS,  KNEE_BAKE_DEG,  f'wrist_link_{leg}'),
            ('wrist_geom',   WRIST_GEOM_ZERO, WRIST_AXIS, WRIST_BAKE_DEG, f'EE_{leg}'),
        ]:
            v0      = zero_dict[leg]
            new_end = _rodrigues(v0, axis_dict[leg], bake_dict[leg])
            old_str = _fmt(v0)
            new_str = _fmt(new_end)
            xml = re.sub(
                rf'(name="{geom_name}_{leg}"[^/]*?fromto=")0 0 0\s+{re.escape(old_str)}',
                rf'\g<1>0 0 0  {new_str}', xml, flags=re.DOTALL)
            xml = re.sub(
                rf'(name="{body_name}"\s+pos="){re.escape(old_str)}"',
                rf'\g<1>{new_str}"', xml)
            print(f"[bake] {leg} {geom_name}: {old_str} → {new_str}")

        # EE geom
        v0      = EE_GEOM_ZERO[leg]
        axis    = EE_AXIS[leg]
        angle   = EE_BAKE_DEG[leg]
        new_end = _rodrigues(v0, axis, angle)
        old_str = _fmt(v0)
        new_str = _fmt(new_end)
        xml = re.sub(
            rf'(name="ee_link_geom_{leg}"[^/]*?fromto=")0 0 0\s+{re.escape(old_str)}',
            rf'\g<1>0 0 0  {new_str}', xml, flags=re.DOTALL)

        # EM pos
        v0      = EM_POS_ZERO[leg]
        new_end = _rodrigues(v0, axis, angle)
        old_pos = _fmt(v0)
        new_pos = _fmt(new_end)
        xml = re.sub(
            rf'(name="electromagnet_{leg}"\s+pos="){re.escape(old_pos)}"',
            rf'\g<1>{new_pos}"', xml)

        # EM quat
        q_new    = _rot_to_quat(_axis_angle_to_rot(axis, angle) @ _quat_to_rot(EM_QUAT_ZERO[leg]))
        old_quat = _fmt_quat(EM_QUAT_ZERO[leg])
        new_quat = _fmt_quat(q_new)
        xml = re.sub(
            rf'(name="electromagnet_{leg}"[^>]*?quat="){re.escape(old_quat)}"',
            rf'\g<1>{new_quat}"', xml, flags=re.DOTALL)
        print(f"[bake] {leg} em: pos {old_pos} → {new_pos} | quat → {new_quat}")

    with open(xml_path, 'w') as f:
        f.write(xml)
    print(f"[bake] Written to {xml_path}")