import mujoco
import mujoco.viewer
import numpy as np
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "xml_file": "scene.xml",         # your Sally wall scene
    "timestep": 0.001,
    "t_max": 100.0,
    "real_time_sync": True,
    "pause_key": " ",

    # Magnetic parameters
    "Br": 1.48 / 1.5,  
    "magnet_volume": np.pi * ((0.025 / 2) ** 2 - (0.016 / 2) ** 2) * 0.025,
    "max_total_force": 200.0 * 4,     # 4 wheels
    "distance_min": 0.0001,
    "distance_max": 0.01,
    "MU_0": 4 * np.pi * 1e-7,

    # Visual arrows
    "arrow_radius": 0.005,
    "arrow_color": (0, 0, 1, 1),

    # Geom naming (Sally XML)
    "wall_geom_name": "wall_geom",
    "wheel_names": ["FL_cyl", "FR_cyl", "BL_cyl", "BR_cyl"],
}

# Motor torque magnitude
TORQUE = 10.0 # N·m


# =============================================================================
# FORCE MODEL
# =============================================================================
def calculate_magnetic_force(distance, Br, V, MU_0):
    if distance <= 0:
        return 0.0
    m = (Br * V) / MU_0
    return (3 * MU_0 * m ** 2) / (2 * np.pi * (2 * distance) ** 4)


def add_visual_arrow(scene, from_point, to_point, radius=0.005, rgba=(0, 0, 1, 1)):
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
# KEYBOARD
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

model.opt.timestep = CONFIG["timestep"]
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
model.opt.o_solref[:] = [4e-4, 25]
model.opt.o_solimp[:] = [0.99, 0.99, 0.001, 0.5, 2]


# --- Get wall ID ---
wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CONFIG["wall_geom_name"])

# --- Get wheel geom + body IDs ---
wheel_geom_ids = []
wheel_body_ids = []

for w in CONFIG["wheel_names"]:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, w)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, w.replace("_cyl", "_wheel_geom"))

    if gid == -1 or bid == -1:
        print(f"[WARN] Missing wheel {w}")
        continue

    wheel_geom_ids.append(gid)
    wheel_body_ids.append(bid)


# --- Get motor actuator IDs ---
ACTUATOR_NAMES = [
    "FL_wheel_motor",
    "FR_wheel_motor",
    "BL_wheel_motor",
    "BR_wheel_motor"
]

wheel_actuators = {}
for name in ACTUATOR_NAMES:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid >= 0:
        wheel_actuators[name] = aid
    else:
        print("[WARN] Missing actuator:", name)


# =============================================================================
# SIMULATION LOOP
# =============================================================================
sim_time = 0.0
timestep = model.opt.timestep

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    print("[INFO] Running adhesion + wheel torque simulation...")
    print("[INFO] Press SPACE to pause/resume.")

    while viewer.is_running():

        viewer.user_scn.ngeom = 0
        total_force = np.zeros(3)
        wheel_forces = {}   # <-- NEW: store individual wheel adhesion values

        # -----------------------------
        # Compute magnetic forces
        # -----------------------------
        for w, geom_id, body_id in zip(CONFIG["wheel_names"], wheel_geom_ids, wheel_body_ids):

            fromto = np.zeros(6)
            dist = mujoco.mj_geomDistance(model, data, geom_id, wall_id, 50, fromto)

            if dist < 0:
                wheel_forces[w] = 0.0
                continue

            n = fromto[3:6] - fromto[0:3]
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-6:
                wheel_forces[w] = 0.0
                continue

            n_hat = n / n_norm

            fmag = calculate_magnetic_force(
                dist, CONFIG["Br"], CONFIG["magnet_volume"], CONFIG["MU_0"]
            )
            fmag = np.clip(fmag, 0, CONFIG["max_total_force"] / 4)

            wheel_forces[w] = fmag   # <-- NEW

            fvec = fmag * n_hat
            data.xfrc_applied[body_id, :3] = fvec
            total_force += fvec

            add_visual_arrow(
                viewer.user_scn,
                fromto[0:3],
                fromto[0:3] + (-0.05) * n_hat,
                radius=CONFIG["arrow_radius"],
                rgba=CONFIG["arrow_color"]
            )



        # -----------------------------
        # Apply wheel torques
        # -----------------------------
        for aid in wheel_actuators.values():
            data.ctrl[aid] = TORQUE

        # -----------------------------
        # Step simulation
        # -----------------------------
        # Step simulation ONLY when running
        if not key_state["paused"]:
            mujoco.mj_step(model, data)
            sim_time += timestep

            # Log only when running (no scrolling when paused)
            print(
                f"t={sim_time:.3f}s | Total |F|={np.linalg.norm(total_force):.2f}N | "
                f"FL={wheel_forces.get('FL_cyl',0):.1f}  "
                f"FR={wheel_forces.get('FR_cyl',0):.1f}  "
                f"BL={wheel_forces.get('BL_cyl',0):.1f}  "
                f"BR={wheel_forces.get('BR_cyl',0):.1f}"
            )

        viewer.sync()

        # Optional real-time pacing only when running
        if CONFIG["real_time_sync"] and not key_state["paused"]:
            time.sleep(timestep)



        # End simulation
        if sim_time >= CONFIG["t_max"]:
            print("[INFO] Simulation finished.")
            break
