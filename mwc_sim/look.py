#!/usr/bin/env python3
"""
look.py

Viewer for magnet + flat plate setup.

Folder structure:
  look.py
  XML/
      scene.xml
      flat_plate_pull.xml
"""

import time
import mujoco
import mujoco.viewer


XML_PATH = "XML/scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print(f"[look.py] Loaded: {XML_PATH}")
    print(f"nbody={model.nbody}, ngeom={model.ngeom}, njnt={model.njnt}")

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Camera tuned for small (63mm) magnet
        viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
        viewer.cam.distance = 0.35
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 45

        timestep = model.opt.timestep
        last = time.time()

        print("[look.py] Running viewer...")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

            # real-time pacing
            now = time.time()
            dt = timestep - (now - last)
            if dt > 0:
                time.sleep(dt)
            last = now


if __name__ == "__main__":
    main()