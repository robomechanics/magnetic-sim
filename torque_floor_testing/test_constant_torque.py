import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

file = "torque_scene.xml"
model = mujoco.MjModel.from_xml_path(file)
data = mujoco.MjData(model)

model.opt.timestep = 0.001
model.opt.gravity[:] = [0, 0, -9.81]
model.dof_damping[:] = 0.2

t_max = 5
applied_torque = np.array([-0.5, 0.0, 0.0])
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "magnet_box")

print(f"\n[INFO] Applying torque {applied_torque} Nm to 'magnet_box'\n")

time_log, omega_log = [], []

# Real-time sync reference
real_start = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < t_max:
        floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "magnetic_floor")
        mag_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mag_box")

        # compute distance between magnet and floor
        fromto = np.zeros(6)
        dist = mujoco.mj_geomDistance(model, data, mag_geom_id, floor_geom_id, 1, fromto)

        if 0 < dist < 0.05:
            MU_0 = np.pi * 1e-7
            Br = 1.3
            V = (0.02**3)
            m = (Br * V) / MU_0
            F = (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist**2)**2)
            Fz = -min(F, 50.0)   # cap the force
            data.xfrc_applied[body_id, 2] += Fz
        sim_start = time.time()

        # Apply torque for first 5 s
        if data.time < 5.0:
            data.xfrc_applied[body_id, 3:6] = applied_torque
        else:
            data.xfrc_applied[body_id, 3:6] = 0

        mujoco.mj_step(model, data)
        viewer.sync()

        # Logging
        time_log.append(data.time)
        omega_log.append(np.linalg.norm(data.qvel[3:6]))

        # --- Throttle to real-time ---
        elapsed = time.time() - sim_start
        wait = model.opt.timestep - elapsed
        if wait > 0:
            time.sleep(wait)

print("\nSimulation complete.\n")

# --- Plot angular velocity ---
plt.figure(figsize=(8, 4))
plt.plot(time_log, omega_log, lw=2)
plt.axvline(5.0, color='r', ls='--', label='Torque off')
plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Rolling Magnet: Angular Velocity vs Time (Real-Time Sim)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
