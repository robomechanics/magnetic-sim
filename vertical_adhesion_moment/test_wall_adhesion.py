import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

# source .venv/bin/activate
# === CONFIGURATION ==========================================================
CONFIG = {
    "xml_file": "torque_wall_scene.xml",  # include the wall env
    "timestep": 0.001,
    "t_max": 100.0,
    "real_time_sync": True,
    "pause_key": " ",

    # Magnetic parameters
    "Br": 1.48/1.5, # Tesla, 1 Tesla = 10000 Gauss
    "magnet_volume": np.pi * ((0.025/2)**2 - (0.016/2)**2) * 0.025,  # m^3
    "max_total_force": 200.0 * 8,
    "distance_min": 0.01,
    "distance_max": 0.05,
    "MU_0": 4 * np.pi * 1e-7,

    # Visual arrows
    "arrow_radius": 0.005,
    "arrow_color": (0, 0, 1, 1),

    # Geometry / naming (updated)
    "magnet_geom_name": "magnet_geom",
    "metal_geom_name": "magnetic_wall",
    "magnet_body_name": "magnet_wheel",
    "num_mag_pts": 8,
    "magnet_pt_prefix": "mag_pt",
}


# ============================================================================

def calculate_magnetic_force(distance: float, Br: float, V: float, MU_0: float) -> float:
    """Dipole approximation for magnet-plate attractive force."""
    if distance <= 0:
        return 0
    m = (Br * V) / MU_0
    force = (3 * MU_0 * m ** 2) / (2 * np.pi * (2 * distance) ** 4)
    return force

def add_visual_arrow(scene, from_point, to_point, radius=0.005, rgba=(0, 0, 1, 1)):
    """Add a visual debug arrow to the viewer scene."""
    if scene.ngeom >= scene.maxgeom:
        print("Warning: Maximum number of geoms reached. Cannot add arrow.")
        return

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([radius, radius, np.linalg.norm(to_point - from_point)]),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32)
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        radius,
        from_point,
        to_point
    )
    scene.ngeom += 1

# Keyboard control
key_state = {"paused": True, "mode": "side-y"}
def key_callback(keycode):
    c = chr(keycode)
    if c == CONFIG["pause_key"]:
        key_state["paused"] = not key_state["paused"]
    elif c.lower() in ["f", "x", "y"]:
        key_state["mode"] = {"f": "face", "x": "side-x", "y": "side-y"}[c.lower()]
        print(f"Switched to {key_state['mode'].upper()} magnetization")

# === MAIN SIMULATION ========================================================

file = CONFIG["xml_file"]
model = mujoco.MjModel.from_xml_path(file)
data = mujoco.MjData(model)

# Physics configuration
model.opt.timestep = CONFIG["timestep"]
model.opt.enableflags |= 1 << 0
model.opt.o_solref[:] = [4e-4, 25]
model.opt.o_solimp[:] = [0.99, 0.99, 0.001, 0.5, 2]
model.opt.o_friction[:] = [1, 1, 0.001, 0.0005, 0.0005]

# IDs
# 3 layers: front (f), middle (m), back (b)
layers = ["f", "m", "b"]
mag_ids = []
for layer in layers:
    for i in range(8):
        name = f"{CONFIG['magnet_pt_prefix']}{i}_{layer}"
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id == -1:
            print(f"[WARN] geom {name} not found!")
        else:
            mag_ids.append(geom_id)

box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CONFIG["metal_geom_name"])
mag_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CONFIG["magnet_geom_name"])
mag_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONFIG["magnet_body_name"])

timestep = model.opt.timestep
start_time = time.time()
print(f"[INFO] Starting magnetic adhesion simulation in {CONFIG['xml_file']}")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:

    while viewer.is_running():

        viewer.user_scn.ngeom = 0
        obj_pos = data.body(CONFIG["magnet_body_name"]).xpos
        tot_wrench = np.zeros(6, dtype=np.float64)
        time.sleep(0.01)
        for mag_id in mag_ids:
            
            raw_fromto = np.zeros(6, dtype=np.float64)
            raw_distance = mujoco.mj_geomDistance(model, data, mag_id, box_id, 50, raw_fromto)
            raw_vec = raw_fromto[3:6] - raw_fromto[0:3]          # from magnet point → wall
            distance = np.linalg.norm(raw_vec)

            if distance < 1e-6:
                continue  # skip degenerate cases

            n_hat = raw_vec / distance   

            mag_fromto = np.zeros(6, dtype=np.float64)
            mujoco.mj_geomDistance(model, data, mag_id, mag_geom_id, 1, mag_fromto)
            mag_vec = mag_fromto[3:6] - mag_fromto[0:3]
            mag_vec /= np.linalg.norm(mag_vec)

            #proj_vec = np.dot(raw_vec, mag_vec) * mag_vec
            arrow_length = min(distance, 0.05)  # short arrow for clarity
            proj_vec = n_hat * arrow_length
            fromto = np.concatenate([raw_fromto[0:3], raw_fromto[0:3] + proj_vec])

            if CONFIG["distance_min"] < distance < CONFIG["distance_max"]:
                add_visual_arrow(viewer.user_scn, fromto[0:3], fromto[3:6],
                                 radius=CONFIG["arrow_radius"], rgba=CONFIG["arrow_color"])

            # --- Magnetic force magnitude ---
            fmag = calculate_magnetic_force(
                distance, CONFIG["Br"], CONFIG["magnet_volume"], CONFIG["MU_0"]
            )
            fmag = min(fmag, CONFIG["max_total_force"] / len(mag_ids))

            # --- Magnetization orientation scaling ---
            mag_xmat = data.body(CONFIG["magnet_body_name"]).xmat.reshape(3, 3)
            if key_state["mode"] == "face":
                m_hat = n_hat
            elif key_state["mode"] == "side-x":
                m_hat = mag_xmat[0] / np.linalg.norm(mag_xmat[0])
            elif key_state["mode"] == "side-y":
                m_hat = mag_xmat[1] / np.linalg.norm(mag_xmat[1])
            else:
                m_hat = n_hat

            orientation_scale = (np.dot(m_hat, n_hat)) ** 2
            fmag *= orientation_scale


            # --- Force direction (always defined now) ---
            if key_state["mode"] == "face":
                # normal pull toward wall
                direction = n_hat
            elif key_state["mode"] == "side-x":
                direction = mag_xmat[0] / np.linalg.norm(mag_xmat[0])
            elif key_state["mode"] == "side-y":
                direction = mag_xmat[1] / np.linalg.norm(mag_xmat[1])
            else:
                direction = n_hat

            force_vector = fmag * direction  # <-- now always defined

            # --- Moment calculation ---
            mag_pt_pos = data.geom(mag_id).xpos
            moment_arm = mag_pt_pos - obj_pos
            moment = np.cross(moment_arm, force_vector)
            tot_wrench += np.concatenate((force_vector, moment))




        data.xfrc_applied[mag_body_id] += tot_wrench

        print(f"time={data.time:.3f}, mode={key_state['mode']}, "
              f"distance={distance:.5f} m, total_force={np.linalg.norm(tot_wrench[:3]):.3f} N")

        if not key_state["paused"]:
            mujoco.mj_step(model, data)
            viewer.sync()

        # Real-time synchronization
        if CONFIG["real_time_sync"]:
            step_start_time = time.time()
            time_until_next_step = timestep - (time.time() - step_start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        if time.time() - start_time >= CONFIG["t_max"]:
            print("Simulation finished.")
            break
