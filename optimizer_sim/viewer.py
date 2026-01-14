"""
viewer.py

Visualization / sanity-check entry point.

Purpose:
- Visually inspect a rollout using the same simulation logic as optimization
- Verify that tuned parameters actually change robot behavior
- No optimization, no logging, no physics duplication
"""

import mujoco
import mujoco.viewer
import time

from sim_sally_magnet_wall import build_model, reset_data, apply_params, calculate_magnetic_force, DEFAULT_CONFIG


# =============================
# Optional: parameters to inspect
# (later this can be loaded from CSV / JSON)
# =============================
INSPECT_PARAMS = {
    # Example overrides (can be empty)
    # "Br": 1.2,
    # "o_solref": [3e-4, 20],
}


# =============================
# Simulation settings
# =============================
SIM_DURATION = 20.0
FIXED_TORQUE = 0.5


def main():
    # Build model using shared sim code
    model, ids = build_model(DEFAULT_CONFIG)
    apply_params(model, INSPECT_PARAMS)
    data = reset_data(model)

    dt = model.opt.timestep
    n_steps = int(SIM_DURATION / dt)

    # Magnetic parameters (same logic as rollout)
    Br = INSPECT_PARAMS.get("Br", DEFAULT_CONFIG["Br"])
    V = DEFAULT_CONFIG["magnet_volume"]
    MU_0 = DEFAULT_CONFIG["MU_0"]
    max_force = DEFAULT_CONFIG["max_total_force"] / len(ids["wheel_body_ids"])

    fromto = [0.0] * 6

    print("Launching viewer for rollout inspection...")
    print("Press [Esc] to quit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(n_steps):
            if not viewer.is_running():
                break

            # Clear applied forces
            for bid in ids["wheel_body_ids"]:
                data.xfrc_applied[bid, :3] = 0.0

            # Magnetic force computation (same as rollout)
            for gid, bid in zip(ids["wheel_geom_ids"], ids["wheel_body_ids"]):
                dist = mujoco.mj_geomDistance(
                    model, data, gid, ids["wall_id"], 50, fromto
                )
                if dist < 0:
                    continue

                n = [
                    fromto[3] - fromto[0],
                    fromto[4] - fromto[1],
                    fromto[5] - fromto[2],
                ]
                norm = (n[0]**2 + n[1]**2 + n[2]**2) ** 0.5
                if norm < 1e-9:
                    continue

                fmag = calculate_magnetic_force(dist, Br, V, MU_0)
                fmag = min(fmag, max_force)

                data.xfrc_applied[bid, :3] = [
                    fmag * n[0] / norm,
                    fmag * n[1] / norm,
                    fmag * n[2] / norm,
                ]

            # Fixed actuation
            for aid in ids["wheel_actuator_ids"]:
                data.ctrl[aid] = FIXED_TORQUE

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(dt)

    print("Viewer closed.")


if __name__ == "__main__":
    main()
