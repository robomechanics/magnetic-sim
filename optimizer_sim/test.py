import mujoco

# Load from optimizer_sim/ directory
model = mujoco.MjModel.from_xml_path("XML/scene.xml")

# Print all DOF information
print(f"Total DOFs: {model.nv}")
print(f"\nDOF names and damping values:")
for i in range(model.nv):
    dof_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.dof_jntid[i])
    print(f"  DOF {i}: {dof_name} -> damping = {model.dof_damping[i]}")

print(f"\nLast 4 DOFs:")
print(f"  {model.dof_damping[-4:]}")