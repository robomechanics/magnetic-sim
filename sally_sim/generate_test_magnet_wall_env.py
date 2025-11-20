import xml.etree.ElementTree as ET
import math

# ===========================
# Parameters (unchanged)
# ===========================
RADIUS = 0.025         # outer radius (m)
HALF_HEIGHT = 0.0125   # half height (m)
EMBED_R = 0.003        # radial embed
EMBED_Z = 0.002        # axial embed
N_PER_RING = 8
WHEEL_OFFSET_X = 0.125  # (not used in patch, kept for compatibility)
WHEEL_OFFSET_TILT = 0.01  # small tilt angle [rad]
WALL_POS_X = 0.180       # (not used in patch)

TORQUE_MAX = 3.0  # motor drive torque

# Options:
#   "drive_up"  – wheels roll up/down the wall
#   "sideways"  – wheels roll sideways along the wall
MODE = "drive_up"
# MODE = "sideways"

# Derived (same as old generator)
RING_RADIUS = RADIUS - EMBED_R
RING_Z = [HALF_HEIGHT - EMBED_Z, 0.0, -HALF_HEIGHT + EMBED_Z]

# Debug / visual control
DEBUG_VISIBLE = True
POINT_COLOR = "0 0 1 1" if DEBUG_VISIBLE else "0 0 0 0"
POINT_SIZE = "0.002" if DEBUG_VISIBLE else "0.001"

# ===========================
# File paths
# ===========================
INPUT_FILE = "robot_sally.xml"           # existing model
OUTPUT_FILE = "robot_sally_patched.xml"  # patched version


# ===========================
# Helpers
# ===========================
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
      - Regenerate the 24 sampling spheres with the old generator's logic
    """
    # 1) Update cylinder size/friction if present
    for geom in body_elem.findall("geom"):
        if geom.get("type") == "cylinder":
            geom.set("size", f"{RADIUS:.3f} {HALF_HEIGHT:.3f}")
            # (friction left as-is)

    # 2) Remove all existing sampling spheres
    to_remove = []
    for geom in body_elem.findall("geom"):
        if geom.get("type") == "sphere":
            to_remove.append(geom)
    for g in to_remove:
        body_elem.remove(g)

    # 3) Regenerate the full 24-point ring array
    for z, layer in zip(RING_Z, ["t", "m", "b"]):
        for i in range(N_PER_RING):
            theta = 2 * math.pi * i / N_PER_RING
            x = RING_RADIUS * math.cos(theta)
            y = RING_RADIUS * math.sin(theta)
            # No 'name' to avoid duplicates
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


# ---------- wheel_linkage: orientation (mode switch) ----------
def patch_wheel_orientation(root):
    wheel_bodies = [
        "BR_wheel_geom",
        "FR_wheel_geom",
        "BL_wheel_geom",
        "FL_wheel_geom",
    ]

    wheel_map = find_bodies_by_name(root, wheel_bodies)

    for name, body in wheel_map.items():
        if body is None:
            print(f"[WARN] Could not find '{name}' to rotate.")
            continue

        # Remove old orientation attributes
        for attr in ["quat", "euler", "xyaxes", "zaxis", "axisangle"]:
            if attr in body.attrib:
                del body.attrib[attr]

        # Apply mode-based rotation
        if MODE == "drive_up":
            # Vertical wheel (rolls like a tank wheel)
            body.set("euler", f"0 0 {WHEEL_OFFSET_TILT}")

        elif MODE == "sideways":
            # Rotate wheel axis sideways
            body.set("euler", f"1.5708 0 {WHEEL_OFFSET_TILT}")

        print(f"[INFO] Set orientation for {name} → MODE={MODE}")


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


# ===========================
# Main patch routine
# ===========================
print(f"[INFO] Loading {INPUT_FILE} ...")
tree = ET.parse(INPUT_FILE)
root = tree.getroot()
remove_self_includes(root)

# 1) MODE-dependent orientation of wheel_linkage bodies
patch_wheel_orientation(root)

# 2) Patch the four wheel *geom* bodies (sampling spheres + cylinder size)
wheel_geom_names = [
    "BR_wheel_geom",
    "FR_wheel_geom",
    "BL_wheel_geom",
    "FL_wheel_geom",
]
geom_body_map = find_bodies_by_name(root, wheel_geom_names)

for bname, body in geom_body_map.items():
    if body is None:
        print(f"[WARN] Could not find body '{bname}' in XML – skipping geom patch.")
    else:
        print(f"[INFO] Patching sampling spheres and cylinder on '{bname}'")
        patch_wheel_geom_body(body)


# 3) Patch wheel actuators' torque limits
patch_wheel_actuators(root)

# 4) Write patched file
ET.indent(tree, space="  ")  # Python 3.9+
tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)

print(f"✅ Patched XML written to {OUTPUT_FILE}")
print(f"   MODE={MODE}")
print(f"   RADIUS={RADIUS}, HALF_HEIGHT={HALF_HEIGHT}")
print(f"   RING_RADIUS={RING_RADIUS}, EMBED_Z={EMBED_Z}, N_PER_RING={N_PER_RING}")
print(f"   TORQUE_MAX={TORQUE_MAX}")
print(f"   DEBUG_VISIBLE={DEBUG_VISIBLE}, POINT_SIZE={POINT_SIZE}")
