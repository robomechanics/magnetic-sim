import xml.etree.ElementTree as ET
import math
import subprocess
import sys
import os

# ===========================
# Parameters
# ===========================
RADIUS = 0.025         # outer radius (m)
HALF_HEIGHT = 0.0125   # half height (m)
EMBED_R = 0.003        # radial embed
EMBED_Z = 0.002        # axial embed
N_PER_RING = 8
WHEEL_OFFSET_X = 0.125   # (not used in patch, kept for compatibility)
WHEEL_OFFSET_TILT = 0.01 # small tilt angle [rad]
WALL_POS_X = 0.180       # (not used in patch)

TORQUE_MAX = 3.0  # motor drive torque

WHEEL_DAMPING_NEW = 0.1
ROCKER_STIFFNESS_NEW = 30.0
ROCKER_DAMPING_NEW = 1.0

# +90 deg about Y for sideways turning
SIDEWAYS_ROT = [0.7071068, 0.0, 0.7071068, 0.0]

# 180 deg yaw (π rotation about Z)
YAW_180 = [0.0, 0.0, 0.0, 1.0]   # (w=0, z=1)

# Options:
#   "drive_up"  – wheels roll up/down the wall
#   "sideways"  – wheels roll sideways along the wall
MODE = "drive_up"

# Derived
RING_RADIUS = RADIUS - EMBED_R
RING_Z = [HALF_HEIGHT - EMBED_Z, 0.0, -HALF_HEIGHT + EMBED_Z]

# Debug / visual control for sampling points
DEBUG_VISIBLE = True
POINT_COLOR = "0 0 1 1" if DEBUG_VISIBLE else "0 0 0 0"
POINT_SIZE = "0.002" if DEBUG_VISIBLE else "0.001"

# ===========================
# File paths
# ===========================
INPUT_FILE = "robot_sally.xml"
OUTPUT_FILE = "robot_sally_patched.xml"


# ===========================
# Helpers
# ===========================
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def quat_norm(q):
    n = math.sqrt(sum(v*v for v in q))
    return [v/n for v in q]


def remove_self_includes(root):
    """Remove any <include file='robot_sally_patched.xml'> from the root."""
    for incl in root.findall("include"):
        fname = incl.get("file", "")
        if "robot_sally_patched.xml" in fname:
            root.remove(incl)


def find_bodies_by_name(root, names):
    """
    Return dict name -> <body> element for each name in `names`.
    Searches recursively.
    """
    result = {n: None for n in names}

    def recurse(elem):
        if elem.tag == "body":
            name = elem.get("name")
            if name in result:
                result[name] = elem
        for child in elem:
            recurse(child)

    recurse(root)
    return result


# ---------- wheel geom: sampling spheres + cylinder size ----------
def patch_wheel_geom_body(body_elem):
    """
    For a single *wheel_geom body* (e.g., BR_wheel_geom), do:
      - Update cylinder size = (RADIUS, HALF_HEIGHT)
      - Delete all existing sphere geoms
      - Regenerate the 24 sampling spheres.
    """
    # 1) Update cylinder size if present
    for geom in body_elem.findall("geom"):
        if geom.get("type") == "cylinder":
            geom.set("size", f"{RADIUS:.3f} {HALF_HEIGHT:.3f}")
            # friction left as-is

    # 2) Remove all existing sampling spheres
    to_remove = []
    for geom in body_elem.findall("geom"):
        if geom.get("type") == "sphere":
            to_remove.append(geom)
    for g in to_remove:
        body_elem.remove(g)

    # 3) Regenerate the full 24-point ring array
    for z in RING_Z:
        for i in range(N_PER_RING):
            theta = 2 * math.pi * i / N_PER_RING
            x = RING_RADIUS * math.cos(theta)
            y = RING_RADIUS * math.sin(theta)
            ET.SubElement(
                body_elem,
                "geom",
                {
                    "type": "sphere",
                    "size": POINT_SIZE,
                    "pos": f"{x:.6f} {y:.6f} {z:.6f}",
                    "rgba": POINT_COLOR,
                    "contype": "0",
                    "conaffinity": "0",
                    "friction": "0.1 0.1 0.1",
                    "mass": "1e-6",
                },
            )


# ---------- orientation / sideways mode ----------
def patch_wheel_orientation(root):
    if MODE != "sideways":
        print("[INFO] drive_up mode, no orientation patch.")
        return

    # +90° rotation about Z (yaw)
    ROT_90Z = [0.70710678, 0.0, 0.0, 0.70710678]

    # Rotate BR + BL only
    ROT_MAP = {
        "rocker_linkage":      ROT_90Z,   # BR
        "rocker_linkage_2":    ROT_90Z,   # FR
        "rocker_linkage_3":    ROT_90Z,   # BL
        "rocker_linkage_4":    ROT_90Z,   # FL
    }

    for body in root.iter("body"):
        name = body.get("name", "")
        if name not in ROT_MAP:
            continue

        rot = ROT_MAP[name]
        q_old = [float(v) for v in body.get("quat", "1 0 0 0").split()]

        if rot is None:
            print(f"[KEEP] {name}: {q_old}")
            continue

        q_new = quat_mul(q_old, rot)
        body.set("quat", " ".join(f"{v:.6f}" for v in q_new))
        print(f"[ROTATE] {name}: {q_old} → {q_new}")



# ---------- actuators ----------
def patch_wheel_actuators(root):
    """
    In the <actuator> block, set forcerange on wheel motors:
      BR_wheel_motor, FR_wheel_motor, BL_wheel_motor, FL_wheel_motor
    """
    act_root = root.find("actuator")
    if act_root is None:
        print("[WARN] No <actuator> element found; skipping actuator patch.")
        return

    motor_names = [
        "BR_wheel_motor",
        "FR_wheel_motor",
        "BL_wheel_motor",
        "FL_wheel_motor",
    ]

    for motor in act_root.findall("motor"):
        name = motor.get("name", "")
        if name in motor_names:
            motor.set("forcerange", f"{-TORQUE_MAX} {TORQUE_MAX}")


def patch_joint_dynamics(root):
    """
    Patch joint dynamics:
      - Increase damping on wheel hinge joints.
      - Adjust stiffness/damping on rocker hinge joints.
    """
    wheel_joint_names = ["BR_wheel", "FR_wheel", "BL_wheel", "FL_wheel"]

    rocker_joint_names = [
        "left_hinge",
        "right_hinge",
        "BR_pivot",
        "FR_pivot",
        "BL_pivot",
        "FL_pivot",
    ]

    for joint in root.iter("joint"):
        name = joint.get("name", "")

        # Wheels: increase damping
        if name in wheel_joint_names:
            old_damping = joint.get("damping", "0")
            joint.set("damping", str(WHEEL_DAMPING_NEW))
            print(f"[INFO] Wheel joint '{name}': damping {old_damping} → {WHEEL_DAMPING_NEW}")

        # Rockers: set stiffness & damping
        if name in rocker_joint_names:
            old_k = joint.get("stiffness", "0")
            old_c = joint.get("damping", "0")

            joint.set("stiffness", str(ROCKER_STIFFNESS_NEW))
            joint.set("damping", str(ROCKER_DAMPING_NEW))

            print(
                f"[INFO] Rocker joint '{name}': "
                f"stiffness {old_k} → {ROCKER_STIFFNESS_NEW}, "
                f"damping {old_c} → {ROCKER_DAMPING_NEW}"
            )


def insert_wheel_geom_body(linkage_body, wheel_name):
    """
    Create <body name='XX_wheel_geom'> inside the wheel_linkage body.
    Adds the cylinder geom; sampling spheres are added later by patch_wheel_geom_body.
    """
    # z-offset: three wheels use -0.0165, FL uses +0.0165 as before
    z_offset = "-0.0165" if wheel_name in ["BR", "FR", "BL"] else "0.0165"

    geom_body = ET.SubElement(linkage_body, "body", {
        "name": f"{wheel_name}_wheel_geom",
        "pos": f"0 0 {z_offset}",
    })

    ET.SubElement(geom_body, "geom", {
        "name": f"{wheel_name}_cyl",
        "type": "cylinder",
        "size": f"{RADIUS:.3f} {HALF_HEIGHT:.3f}",
        "rgba": "0.8 0.2 0.2 1",
        "contype": "1",
        "conaffinity": "1",
        "friction": "0.95 0.01 0.01",
    })

    return geom_body


def launch_simulation():
    """
    Launch sim_sally_magnet_wall.py after generating the patched XML.
    """
    sim_path = os.path.join(os.path.dirname(__file__), "sim_sally_magnet_wall.py")

    if not os.path.exists(sim_path):
        print(f"[ERROR] Could not find simulation file at {sim_path}")
        return

    print("\n🚀 Launching simulation: sim_sally_magnet_wall.py\n")
    subprocess.run([sys.executable, sim_path])


# ===========================
# Main patch routine
# ===========================
print(f"[INFO] Loading {INPUT_FILE} ...")
tree = ET.parse(INPUT_FILE)
root = tree.getroot()
remove_self_includes(root)

# 1) MODE-dependent orientation / rocker rotation
patch_wheel_orientation(root)

# 2) Ensure wheel geom bodies exist, then patch their sampling spheres + cylinder sizes
print("[INFO] Inserting wheel cylinder bodies if missing...")

# Map wheel → linkage body name in robot_sally.xml
WHEEL_LINKAGE_MAP = {
    "BR": "wheel_linkage",
    "FR": "wheel_linkage_2",
    "BL": "wheel_linkage_3",
    "FL": "wheel_linkage_4",
}

geom_body_map = {}

for wheel, linkage_name in WHEEL_LINKAGE_MAP.items():
    # 2.1 Find the linkage body
    linkage_body = None
    for b in root.iter("body"):
        if b.get("name") == linkage_name:
            linkage_body = b
            break

    if linkage_body is None:
        print(f"[WARN] Cannot find linkage '{linkage_name}' – skipping wheel '{wheel}'.")
        continue

    geom_name = f"{wheel}_wheel_geom"

    # 2.2 Check if wheel_geom already exists as a child body
    existing = None
    for b in linkage_body.findall("body"):
        if b.get("name") == geom_name:
            existing = b
            break

    if existing is None:
        print(f"[ADD] Creating missing wheel geom body '{geom_name}' under '{linkage_name}'")
        new_body = insert_wheel_geom_body(linkage_body, wheel)
        geom_body_map[geom_name] = new_body
    else:
        print(f"[FOUND] Existing wheel geom '{geom_name}' under '{linkage_name}'")
        geom_body_map[geom_name] = existing

# 2.3 Patch sampling spheres & cylinder size on each wheel_geom
for name, body in geom_body_map.items():
    print(f"[INFO] Patching sampling spheres and cylinder on '{name}'")
    patch_wheel_geom_body(body)

# 3) Patch wheel actuators' torque limits
patch_wheel_actuators(root)

# 3.5) Patch joint dynamics (damping/stiffness)
patch_joint_dynamics(root)

# 4) Write patched file
ET.indent(tree, space="  ")
tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)

print(f"✅ Patched XML written to {OUTPUT_FILE}")
print(f"   MODE={MODE}")
print(f"   RADIUS={RADIUS}, HALF_HEIGHT={HALF_HEIGHT}")
print(f"   RING_RADIUS={RING_RADIUS}, EMBED_Z={EMBED_Z}, N_PER_RING={N_PER_RING}")
print(f"   TORQUE_MAX={TORQUE_MAX}")
print(f"   DEBUG_VISIBLE={DEBUG_VISIBLE}, POINT_SIZE={POINT_SIZE}")
for body in root.iter("body"):
    if body.get("name") in ["rocker_linkage", "rocker_linkage_2", "rocker_linkage_3", "rocker_linkage_4"]:
        print(body.get("name"), body.get("quat"))

# === Auto-launch simulation ===
launch_simulation()
