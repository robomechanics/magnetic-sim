import xml.etree.ElementTree as ET
import math

# ===========================
# Parameters
# ===========================
RADIUS = 0.025         # outer radius (m)
HALF_HEIGHT = 0.0125   # half height (m)
EMBED_R = 0.002        # radial embed
EMBED_Z = 0.002        # axial embed
N_PER_RING = 8
WHEEL_OFFSET_X = 0.125  # distance from origin to wall
WHEEL_OFFSET_TILT = 0.01  # radians
WALL_POS_X = 0.180     # wall x position

# Derived
RING_RADIUS = RADIUS - EMBED_R
RING_Z = [HALF_HEIGHT - EMBED_Z, 0.0, -HALF_HEIGHT + EMBED_Z]

# Debug / visual control
DEBUG_VISIBLE = True
POINT_COLOR = "0 0 1 1" if DEBUG_VISIBLE else "0 0 0 0"
POINT_SIZE = "0.002" if DEBUG_VISIBLE else "0.001"

# ===========================
# Root structure
# ===========================
mujoco = ET.Element("mujoco", model="magnet_wall_env")
ET.SubElement(mujoco, "compiler", angle="radian")
ET.SubElement(mujoco, "option",
              integrator="implicit",
              cone="elliptic",
              impratio="10",
              noslip_iterations="15")

default = ET.SubElement(mujoco, "default", **{"class": "main"})
ET.SubElement(default, "geom", condim="6")

visual = ET.SubElement(mujoco, "visual")
ET.SubElement(visual, "global", offwidth="1920", offheight="1080")

worldbody = ET.SubElement(mujoco, "worldbody")

# ===========================
# Wall body
# ===========================
wall = ET.SubElement(worldbody, "body", name="wall", pos=f"{WALL_POS_X} 0 0.205")
ET.SubElement(wall, "geom",
              name="magnetic_wall",
              type="box",
              size="0.02 0.25 0.25",
              rgba="0.7 0.7 0.7 1")

# ===========================
# Magnet wheel body
# ===========================
magnet_body = ET.SubElement(worldbody, "body",
                            name="magnet_wheel",
                            pos=f"{WHEEL_OFFSET_X} 0 0.15",
                            euler=f"0 0 {WHEEL_OFFSET_TILT}")

ET.SubElement(magnet_body, "inertial",
              mass="0.33",
              pos="0 0 0",
              diaginertia="0.000069 0.000103 0.000069")
ET.SubElement(magnet_body, "freejoint")

ET.SubElement(magnet_body, "geom",
              name="magnet_geom",
              type="cylinder",
              size=f"{RADIUS:.3f} {HALF_HEIGHT:.3f}",
              rgba="0.8 0.2 0.2 0.8",
              friction="1 0.01 0.01")


# ===========================
# Magnetic contact points
# ===========================
for z, layer in zip(RING_Z, ["t", "m", "b"]):
    for i in range(N_PER_RING):
        theta = 2 * math.pi * i / N_PER_RING
        x = RING_RADIUS * math.cos(theta)
        y = RING_RADIUS * math.sin(theta)
        name = f"mag_pt{i}_{layer}"
        ET.SubElement(magnet_body, "geom",
                      name=name,
                      type="sphere",
                      size=POINT_SIZE,
                      pos=f"{x:.6f} {y:.6f} {z:.6f}",
                      rgba=POINT_COLOR,
                      contype="0",
                      conaffinity="0",
                      friction="0.1 0.1 0.1",   # <── forces MuJoCo to render it
                      mass="1e-6")               # <── ensures it’s renderable but inert)


# ===========================
# Output XML
# ===========================
tree = ET.ElementTree(mujoco)
ET.indent(tree, space="  ")  # requires Python ≥3.9
output_file = "test_magnet_wall_env.xml"
tree.write(output_file, encoding="utf-8", xml_declaration=True)

print(f"✅ Generated {output_file}")
print(f"   radius={RADIUS}  embed_r={EMBED_R}  embed_z={EMBED_Z}")
print(f"   wheel_offset_x={WHEEL_OFFSET_X}  wall_pos_x={WALL_POS_X}")
print(f"   Visible dots: {DEBUG_VISIBLE},  Size={POINT_SIZE}")
