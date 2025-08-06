import os
import mujoco
import mujoco.viewer

file_path = "robot_test/scene.xml"

# Load your model
model = mujoco.MjModel.from_xml_path(file_path)
# Create a simulation data structure
data = mujoco.MjData(model)
# mujoco.mj_saveLastXML("old/old_robot_files/mugatu_nice_feet_fixed_urdf/robot.xml", model)

# Launch the viewer (GUI)
mujoco.viewer.launch(model, data)
