import mujoco
import mujoco.viewer
import time
import numpy as np

file_path = "franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(file_path)
data = mujoco.MjData(model)

t_max = 10000


with mujoco.viewer.launch_passive(model, data) as viewer:
    
    for i in range(t_max):

        mujoco.mj_step(model, data)

        data.ctrl[3] = 1 * np.sin(data.time) - 1

        viewer.sync()  # Update the viewer to reflect the latest simulation state
        time.sleep(0.001)  # Optional: slow down the simulation for better visualization