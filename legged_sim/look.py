import mujoco
import mujoco.viewer
import os

SCENE_XML = os.path.join(os.path.dirname(__file__), "mwc_mjcf", "scene.xml")

def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("spider_rest").id)

    print(f"Loaded: {SCENE_XML}")
    print(f"  Bodies   : {model.nbody}")
    print(f"  Joints   : {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print(f"  Geoms    : {model.ngeom}")

    def key_callback(keycode):
        if keycode == 32:
            viewer.paused = not viewer.paused

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.3]
        viewer.cam.distance  = 1.8
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -20
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.paused = True

        while viewer.is_running():
            data.ctrl[:] = data.qpos[7:]
            if not viewer.paused:
                mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()