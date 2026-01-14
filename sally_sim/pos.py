import mujoco

XML = "scene.xml"  # <-- set this to the EXACT file you load in sim/viewer

model = mujoco.MjModel.from_xml_path(XML)

print("Loaded:", XML)
print("ngeom:", model.ngeom)
print("\n--- All geom names ---")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if name:
        print(i, name)