"""
look.py - Spawn the magnet assembly for visual inspection only.
Tkinter slider window lets you tune bracket euler/pos in real time.
Final values are printed on close for copy-paste into XML.

Sliders use DEGREES for intuitive tuning.
XML output uses RADIANS to match <compiler angle="radian" />.

Usage:
    python look.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import tkinter as tk
import time

XMLPATH = "XML/flat_plate_wrench.xml"


# ── slider ranges ─────────────────────────────────────────────────────────────
RANGES = {
    'euler_x': (-180, 180),
    'euler_y': (-180, 180),
    'euler_z': (-180, 180),
    'pos_x':   (-0.2, 0.2),
    'pos_y':   (-0.2, 0.2),
    'pos_z':   (-0.2, 0.2),
}

# ── initial values read from XML so sliders start at current pose ─────────────
state = {
    'euler_x': 180.0,
    'euler_y':  90.0,
    'euler_z':   0.0,
    'pos_x':     0.0,
    'pos_y':     0.0,
    'pos_z':     0.0,
}


def build_slider_window():
    root = tk.Tk()
    root.title("Bracket Pose Tuner")
    root.resizable(False, False)

    def make_slider(parent, key, label, row):
        lo, hi = RANGES[key]
        tk.Label(parent, text=label, width=12, anchor='w').grid(row=row, column=0, padx=6, pady=3)
        var = tk.DoubleVar(value=state[key])

        def on_change(val, k=key, v=var):
            state[k] = v.get()

        s = tk.Scale(parent, variable=var, from_=lo, to=hi,
                     resolution=(hi - lo) / 1000.0,
                     orient=tk.HORIZONTAL, length=320,
                     command=on_change)
        s.grid(row=row, column=1, padx=6)

        val_label = tk.Label(parent, textvariable=var, width=8)
        val_label.grid(row=row, column=2, padx=4)

    frame = tk.LabelFrame(root, text="Euler (deg)", padx=8, pady=4)
    frame.grid(row=0, column=0, padx=10, pady=6, sticky='ew')
    make_slider(frame, 'euler_x', 'Euler X', 0)
    make_slider(frame, 'euler_y', 'Euler Y', 1)
    make_slider(frame, 'euler_z', 'Euler Z', 2)

    frame2 = tk.LabelFrame(root, text="Position (m)", padx=8, pady=4)
    frame2.grid(row=1, column=0, padx=10, pady=6, sticky='ew')
    make_slider(frame2, 'pos_x', 'Pos X', 0)
    make_slider(frame2, 'pos_y', 'Pos Y', 1)
    make_slider(frame2, 'pos_z', 'Pos Z', 2)

    def print_values():
        ex, ey, ez = state['euler_x'], state['euler_y'], state['euler_z']
        px, py, pz = state['pos_x'],   state['pos_y'],   state['pos_z']

        # Convert degrees -> radians for XML (compiler angle="radian")
        ex_r, ey_r, ez_r = np.radians(ex), np.radians(ey), np.radians(ez)

        print(f'\nCurrent pose (degrees):')
        print(f'  euler="{ex:.1f} {ey:.1f} {ez:.1f}" deg')
        print(f'  pos="{px:.4f} {py:.4f} {pz:.4f}"')
        print(f'\nXML line (radians, for <compiler angle="radian" />):')
        print(f'  euler="{ex_r:.6f} {ey_r:.6f} {ez_r:.6f}" pos="{px:.4f} {py:.4f} {pz:.4f}"')

    tk.Button(root, text="Print current values", command=print_values,
              bg="#2980b9", fg="white", padx=8).grid(row=2, column=0, pady=8)

    root.protocol("WM_DELETE_WINDOW", lambda: (print_values(), root.destroy()))
    root.mainloop()


def main():
    model = mujoco.MjModel.from_xml_path(XMLPATH)
    data  = mujoco.MjData(model)

    model.vis.map.znear = 0.001
    model.vis.map.zfar  = 10.0
    model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.diffuse[:] = [1.0, 1.0, 1.0]

    # Find bracket geom id
    bracket_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bracket")
    if bracket_id == -1:
        print("WARNING: 'bracket' geom not found in XML")

    # Launch slider window in a separate thread
    t = threading.Thread(target=build_slider_window, daemon=True)
    t.start()

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.distance  = 0.35
        v.cam.azimuth   = 45
        v.cam.elevation = -20
        v.cam.lookat[:] = [0.0, 0.0, 0.02]

        while v.is_running():
            if bracket_id != -1:
                # Update geom position
                model.geom_pos[bracket_id] = [
                    state['pos_x'], state['pos_y'], state['pos_z']
                ]

                # Convert euler degrees -> quaternion -> geom_quat
                ex = np.radians(state['euler_x'])
                ey = np.radians(state['euler_y'])
                ez = np.radians(state['euler_z'])

                cx, sx = np.cos(ex/2), np.sin(ex/2)
                cy, sy = np.cos(ey/2), np.sin(ey/2)
                cz, sz = np.cos(ez/2), np.sin(ez/2)

                # Quaternion from intrinsic XYZ euler (w, x, y, z)
                w = cx*cy*cz + sx*sy*sz
                x = sx*cy*cz - cx*sy*sz
                y = cx*sy*cz + sx*cy*sz
                z = cx*cy*sz - sx*sy*cz
                model.geom_quat[bracket_id] = [w, x, y, z]

            mujoco.mj_forward(model, data)
            v.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()