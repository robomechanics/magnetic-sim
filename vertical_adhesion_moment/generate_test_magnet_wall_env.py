import xml.etree.ElementTree as ET
import math

# ===========================
# Parameters
# ===========================
RADIUS = 0.025         # cylinder outer radius (m)
HALF_HEIGHT = 0.0125   # cylinder half-height (m)
EMBED_R = 0.002        # radial embed (m) (into cylinder wall)
EMBED_Z = 0.002        # axial embed (m) (from top/bottom faces)
N_PER_RING = 8
WHEEL_OFFSET_X = 0.125  # distance from world origin toward wall (m)
WHEEL_OFFSET_TILT = 0.01  # rad
WALL_POS_X = 0.180     # wall x-position (tune air gap)

# Derived
RING_RADIUS = RADIUS - EMBED_R
RING_Z = [HALF_HEIGHT - EMBED_Z, 0.0, -HALF_HEIGHT + EMBED_Z]  # top/mid/bottom

# ===========================
# Root structure
# ===========================
mujoco = ET.Element("mujoco", model="magnet_wall_env")
ET.SubElement(mujoco, "compiler", angle="radian")
ET.SubElement(mujoco, "option",
              integrator="implicit", cone="elliptic",
              impratio="10", noslip_iterations="15")
ET.SubElement(mujoco, "default", **{"class": "main"}).append(ET.Element("geom", condim="6"))
ET.SubElement(mujoco, "visual").append(
    ET.Element("global", offwidth="1920", offheight="1080")
)

worldbody = ET.SubElement(mujoco, "worldbody")

# ===========================
# Wall body
# ===========================
wall = ET.SubElement(worldbody, "body", name="wall", pos=f"{WALL_POS_X} 0 0.205")
ET.SubElement(wall, "geom", name="magnetic_wall", type="box", size="0.02 0.25 0.25")

# ===========================
# Magnet wheel body
# ===========================
magnet_body = ET.SubElement(
    worldbody, "body",
    name="magnet_wheel",
    pos=f"{WHEEL_OFFSET_X} 0 0.15",  # <── variable offset
    euler=f"0 0 {WHEEL_OFFSET_TILT}"  # <── variable tilt
)

ET.SubElement(magnet_body, "inertial",
              mass="0.33", pos="0 0 0",
              diaginertia="0.000069 0.000103 0.000069")

ET.SubElement(magnet_body, "freejoint")

# Magnet cylinder
ET.SubElement(magnet_body, "geom",
              name="magnet_geom", type="cylinder",
              size=f"{RADIUS:.3f} {HALF_HEIGHT:.3f}",
              rgba="0.8 0.2 0.2 0.8", friction="1 0.01 0.01")

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
                      name=name, type="sphere",
                      size="0.001",
                      pos=f"{x:.6f} {y:.6f} {z:.6f}")

# ===========================
# Output XML
# ===========================
tree = ET.ElementTree(mujoco)
ET.indent(tree, space="  ")  # Python 3.9+
tree.write("test_magnet_wall_env.xml", encoding="utf-8", xml_declaration=True)

print(f"✅ Generated test_magnet_wall_env.xml with {len(RING_Z)*N_PER_RING} magnet points")
print(f"   radius={RADIUS}  embed_r={EMBED_R}  embed_z={EMBED_Z}")
print(f"   wheel_offset_x={WHEEL_OFFSET_X}  wall_pos_x={WALL_POS_X}")
