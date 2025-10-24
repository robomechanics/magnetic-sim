import mujoco
import mujoco.viewer
import time 
import numpy as np

file = "scene.xml"

model = mujoco.MjModel.from_xml_path(file)
data = mujoco.MjData(model)

t_max = 100.0  # Maximum simulation time in seconds
start_time = time.time()

MU_0 = np.pi * 1e-7
max_force = 20

# Default magnetization mode
MAGNET_MODE = "side-y"   # options: "face", "side-x", "side-y"

def calculate_magnetic_force(
    distance: float,
    magnet_remanence: float,
    magnet_volume: float
) -> float:
    """Dipole approximation for magnet-plate attractive force."""
    if distance <= 0:
        return 0

    magnetic_moment = (magnet_remanence * magnet_volume) / MU_0
    z = 2 * distance ** 2
    force = (3 * MU_0 * magnetic_moment**2) / (2 * np.pi * z**4)
    return force

def add_visual_arrow(scene, from_point, to_point, radius=0.005, rgba=(0, 0, 1, 1)):
    """Visual debug arrow."""
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
key_dict = {"paused": False}
def key_callback(keycode):
    global key_dict, MAGNET_MODE
    c = chr(keycode)
    if c == ' ':
        key_dict["paused"] = not key_dict["paused"]
    elif c.lower() == 'f':
        MAGNET_MODE = "face"
        print("Switched to FACE magnetization")
    elif c.lower() == 'x':
        MAGNET_MODE = "side-x"
        print("Switched to SIDE-X magnetization")
    elif c.lower() == 'y':
        MAGNET_MODE = "side-y"
        print("Switched to SIDE-Y magnetization")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:


    model.opt.timestep = 0.0005 
    model.opt.enableflags |= 1 << 0  
    #model.opt.o_solref[:] = [4e-4, 25]
    #model.opt.o_solimp[:] = [0.99, 0.99, 1e-3]
    model.opt.o_solref[:] = [4e-4, 25]                 # (len=2, okay)
    model.opt.o_solimp[:] = [0.99, 0.99, 0.001, 0.5, 2] # (len=5 required)

    model.opt.o_friction[:] = [1, 1, 0.001, 0.0005, 0.0005]

    timestep = model.opt.timestep

    # Collect geom and body IDs
    mag_ids = []
    for i in range(8):   # match however many mag_pts you defined
        mag_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"mag_pt{i}"))
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "metal_box")
    mag_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mag_box")
    mag_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "magnet_body")

    while viewer.is_running():
        viewer.user_scn.ngeom = 0
        obj_pos = data.body("magnet_box").xpos
        tot_wrench = np.zeros(6, dtype=np.float64)

        for mag_id in mag_ids:
            raw_fromto = np.zeros(6, dtype=np.float64)
            raw_distance = mujoco.mj_geomDistance(model, data, mag_id, box_id, 50, raw_fromto)
            raw_vec = raw_fromto[3:6] - raw_fromto[0:3]

            mag_fromto = np.zeros(6, dtype=np.float64)
            mujoco.mj_geomDistance(model, data, mag_id, mag_geom_id, 1, mag_fromto)
            mag_vec = mag_fromto[3:6] - mag_fromto[0:3]
            mag_vec /= np.linalg.norm(mag_vec)

            proj_vec = np.dot(raw_vec, mag_vec) * mag_vec
            fromto = np.concatenate([mag_fromto[0:3], mag_fromto[0:3] + proj_vec])
            vec = fromto[3:6] - fromto[0:3]

            distance = raw_distance if np.dot(raw_vec, mag_vec) > 0 else 0      
            wrench = np.zeros(6, dtype=np.float64)

            if distance > 1e-3 and distance < 0.05:
                add_visual_arrow(viewer.user_scn, fromto[0:3], fromto[3:6], rgba=(0, 0, 1, 1))
                magnet_remanence = 1.3  
                magnet_volume = (0.02**3) 

                force_magnitude = calculate_magnetic_force(distance, magnet_remanence, magnet_volume)
                if force_magnitude > max_force / len(mag_ids):
                    force_magnitude = max_force / len(mag_ids)

                # Magnetization axis logic
                mag_xmat = data.body("magnet_box").xmat.reshape(3, 3)
                if   MAGNET_MODE == "face":
                    direction_vector = to_point - from_point
                    normalized_direction = direction_vector / np.linalg.norm(direction_vector)
                    force_vector = force_magnitude * normalized_direction
                elif MAGNET_MODE == "side-x":
                    mag_axis = mag_xmat[0] / np.linalg.norm(mag_xmat[0])
                    force_vector = force_magnitude * mag_axis
                elif MAGNET_MODE == "side-y":
                    mag_axis = mag_xmat[1] / np.linalg.norm(mag_xmat[1])
                    force_vector = force_magnitude * mag_axis

                mag_pt_pos = data.geom(mag_id).xpos
                moment_arm = mag_pt_pos - obj_pos
                moment = np.cross(moment_arm, force_vector)
                wrench = np.round(np.concatenate((force_vector, moment)),5)

                tot_wrench += wrench

        print(f"time: {data.time:.3f}, mode={MAGNET_MODE}, wrench={wrench}, distance={distance:.5f} m")
        data.xfrc_applied[mag_body_id] += tot_wrench

        step_start_time = time.time()
        if not key_dict['paused']:
            mujoco.mj_step(model, data)
            viewer.sync()

        time_until_next_step = timestep - (time.time() - step_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        if time.time() - start_time >= t_max:
            print("Simulation finished")
            break
