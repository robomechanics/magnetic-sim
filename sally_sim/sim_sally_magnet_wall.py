import mujoco
import mujoco.viewer
import numpy as np
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "xml_file": "scene.xml",     # <-- now uses the new Sally wall scene
    "timestep": 0.001,
    "t_max": 100.0,
    "real_time_sync": True,
    "pause_key": " ",

    # Magnetic parameters
    "Br": 1.48 / 1.5,  # Tesla
    "magnet_volume": np.pi * ((0.025 / 2) ** 2 - (0.016 / 2) ** 2) * 0.025,
    "max_total_force": 200.0 * 4,   # 4 wheels
    "distance_min": 0.001,
    "distance_max": 0.05,
    "MU_0": 4 * np.pi * 1e-7,

    # Visual arrows
    "arrow_radius": 0.005,
    "arrow_color": (0, 0, 1, 1),

    # Geom / naming
    "wall_geom_name": "wall_geom",
    "wheel_names": ["FL_cyl", "FR_cyl", "BL_cyl", "BR_cyl"],

}


# =============================================================================
# FORCE MODEL
# =============================================================================
def calculate_magnetic_force(distance, Br, V, MU_0):
    """Dipole approximation for magnet-plate attractive force."""
    if distance <= 0:
        return 0.0
    m = (Br * V) / MU_0
    return (3 * MU_0 * m ** 2) / (2 * np.pi * (2 * distance) ** 4)


def add_visual_arrow(scene, from_point, to_point, radius=0.005, rgba=(0, 0, 1, 1)):
    """Add a debug arrow to MuJoCo viewer scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([radius, radius, np.linalg.norm(to_point - from_point)]),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        radius,
        from_point,
        to_point,
    )
    scene.ngeom += 1


# =============================================================================
# KEYBOARD HANDLER
# =============================================================================
key_state = {"paused": True}
def key_callback(keycode):
    c = chr(keycode)
    if c == CONFIG["pause_key"]:
        key_state["paused"] = not key_state["paused"]


# =============================================================================
# MAIN
# =============================================================================
print(f"[INFO] Loading {CONFIG['xml_file']} ...")
model = mujoco.MjModel.from_xml_path(CONFIG["xml_file"])
data = mujoco.MjData(model)

# Configure physics solver
model.opt.timestep = CONFIG["timestep"]
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
model.opt.o_solref[:] = [4e-4, 25]
model.opt.o_solimp[:] = [0.99, 0.99, 0.001, 0.5, 2]

# Get IDs
wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CONFIG["wall_geom_name"])
wheel_geom_ids = []
wheel_body_ids = []
for w in CONFIG["wheel_names"]:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, w)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, w.replace("_cyl", "_wheel_geom"))
    if gid == -1 or bid == -1:
        print(f"[WARN] Could not find geom/body for {w}, skipping.")
        continue
    wheel_geom_ids.append(gid)
    wheel_body_ids.append(bid)

print("wall_id ->", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, wall_id))
for gid, bid in zip(wheel_geom_ids, wheel_body_ids):
    print("wheel geom:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
          " | body:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid))


timestep = model.opt.timestep
start_time = time.time()
sim_time_accum = 0.0  # <--- simulated time that pauses correctly

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    print("[INFO] Starting magnetic adhesion simulation... Press SPACE to pause/resume.")

    last_wall_time = time.time()

    while viewer.is_running():
        viewer.user_scn.ngeom = 0
        tot_force = np.zeros(3)
        tot_moment = np.zeros(3)

        for wheel_geom_id, wheel_body_id in zip(wheel_geom_ids, wheel_body_ids):
            fromto = np.zeros(6)
            dist = mujoco.mj_geomDistance(model, data, wheel_geom_id, wall_id, 50, fromto)
            if dist < 0:
                continue
            n_hat = (fromto[3:6] - fromto[0:3])
            norm = np.linalg.norm(n_hat)
            if norm < 1e-6:
                continue
            n_hat /= norm

            # Force magnitude
            fmag = calculate_magnetic_force(dist, CONFIG["Br"], CONFIG["magnet_volume"], CONFIG["MU_0"])
            fmag = np.clip(fmag, 0, CONFIG["max_total_force"] / len(wheel_geom_ids))

            # Apply along wall normal (always attractive)
            fvec = fmag * (n_hat)
            data.xfrc_applied[wheel_body_id, :3] = fvec

            add_visual_arrow(viewer.user_scn, fromto[0:3], fromto[0:3] + 0.05 * (-n_hat),
                             radius=CONFIG["arrow_radius"], rgba=CONFIG["arrow_color"])
            tot_force += fvec

        # === Simulation control ===
        if not key_state["paused"]:
            mujoco.mj_step(model, data)
            viewer.sync()
            sim_time_accum += timestep   # advance only while unpaused
        else:
            viewer.sync()

        # === Timing control ===
        if CONFIG["real_time_sync"]:
            time.sleep(timestep)

        # === Output logging ===
        print(f"time={sim_time_accum:.3f}s, total |F|={np.linalg.norm(tot_force):.3f} N", flush=True)

        # Stop after simulated time exceeds t_max
        if sim_time_accum >= CONFIG["t_max"]:
            print("\n[INFO] Simulation complete.")
            break

