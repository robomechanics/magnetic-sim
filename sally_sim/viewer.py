import mujoco
import mujoco.viewer
import time

# =============================
# File paths
# =============================
SCENE_XML = "scene.xml"        # your scene file
ROBOT_XML = "robot_sally_body.xml"  # referenced inside scene.xml (for clarity)

# =============================
# Load the model and create data
# =============================
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# =============================
# Launch the viewer
# =============================
print(f"Loaded scene: {SCENE_XML}")
print(f"Including robot: {ROBOT_XML}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation started. Press [Esc] to quit.")

    # Run indefinitely (until viewer is closed)
    start_time = time.time()
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        # simple real-time pacing
        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

    print("Viewer closed. Exiting.")
