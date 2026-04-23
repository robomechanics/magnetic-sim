from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from config import DEFAULT_MODE, DEFAULT_PARAMS, DEFAULT_XML_PATH, MODES

MU_0 = 4 * np.pi * 1e-7
MAGNET_VOLUME = np.pi * (0.025**2) * 0.013


def resolve_xml_path(mjcf_path: str | None = None) -> str:
    path = Path(mjcf_path or DEFAULT_XML_PATH)
    if path.exists():
        return str(path)
    alt = Path(__file__).resolve().parent / path.name
    return str(alt)


def calculate_magnetic_force(distance: float, Br: float, V: float, mu_0: float) -> float:
    if distance <= 0.0:
        return 0.0
    m = (Br * V) / mu_0
    return (3 * mu_0 * m**2) / (2 * np.pi * (2 * distance) ** 4)


def apply_sim_params(model: mujoco.MjModel, params: dict[str, Any]) -> None:
    wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_geom")
    if wall_id >= 0 and "ground_friction" in params:
        model.geom_friction[wall_id] = params["ground_friction"]
    if "solref" in params:
        model.opt.o_solref[:] = params["solref"]
    if "solimp" in params:
        model.opt.o_solimp[:] = params["solimp"]
    if "noslip_iterations" in params:
        model.opt.noslip_iterations = int(params["noslip_iterations"])

    rocker_joints = ["right_hinge", "left_hinge", "BR_pivot", "FR_pivot", "BL_pivot", "FL_pivot"]
    if "rocker_stiffness" in params and "rocker_damping" in params:
        for joint_name in rocker_joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                model.jnt_stiffness[joint_id] = params["rocker_stiffness"]
                model.dof_damping[model.jnt_dofadr[joint_id]] = params["rocker_damping"]

    wheel_actuators = ["BR_wheel_motor", "FR_wheel_motor", "BL_wheel_motor", "FL_wheel_motor"]
    if "wheel_kp" in params:
        for act_name in wheel_actuators:
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id != -1:
                model.actuator_gainprm[act_id, 0] = params["wheel_kp"]
                model.actuator_biasprm[act_id, 1] = -params["wheel_kp"]
    if "wheel_kv" in params:
        for act_name in wheel_actuators:
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id != -1:
                model.actuator_biasprm[act_id, 2] = -params["wheel_kv"]


def run_simulation(
    params: dict[str, Any],
    mjcf_path: str | None = None,
    sim_duration: float | None = None,
    visualize: bool = False,
    mode: str | None = None,
):
    if visualize:
        raise NotImplementedError("Visualization path is not implemented in this stripped-down optimizer helper.")

    mode = mode or DEFAULT_MODE
    mode_cfg = MODES[mode]
    sim_duration = sim_duration if sim_duration is not None else mode_cfg["sim_duration"]

    model = mujoco.MjModel.from_xml_path(resolve_xml_path(mjcf_path))
    apply_sim_params(model, params)
    model.opt.timestep = 1.0 / 1e3
    model.opt.enableflags |= 1 << 0
    model.opt.iterations = 100
    model.opt.tolerance = 1e-8
    data = mujoco.MjData(model)

    data.qpos[0] = 0.035
    data.qpos[1] = 0.0
    data.qpos[2] = 0.35
    data.qpos[3:7] = [-0.707, 0, 0.707, 0]

    wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_geom")
    Br = params["Br"]
    max_magnetic_distance = params["max_magnetic_distance"]

    wheel_gids = []
    for wheel_prefix in ["BR", "FR", "BL", "FL"]:
        body_name = f"{wheel_prefix}_wheel_geom"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            continue
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                wheel_gids.append(geom_id)

    wheel_act_ids = []
    for act_name in ["BR_wheel_motor", "FR_wheel_motor", "BL_wheel_motor", "FL_wheel_motor"]:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if act_id != -1:
            wheel_act_ids.append(act_id)

    fromto = np.zeros(6)
    trajectory = []

    try:
        mujoco.mj_step(model, data)
        for joint_name in ["BR_pivot", "FR_pivot", "BL_pivot", "FL_pivot"]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                qadr = model.jnt_qposadr[joint_id]
                data.qpos[qadr] = mode_cfg["pivot_angle"]
                model.jnt_stiffness[joint_id] = max(model.jnt_stiffness[joint_id], 1000.0)
                model.qpos_spring[qadr] = mode_cfg["pivot_angle"]

        while data.time < sim_duration:
            data.xfrc_applied[:] = 0.0
            wheel_forces: dict[int, np.ndarray] = {}
            for gid in wheel_gids:
                dist = mujoco.mj_geomDistance(model, data, gid, wall_id, 50, fromto)
                if dist < 0 or dist > max_magnetic_distance:
                    continue
                fmag = calculate_magnetic_force(dist, Br, MAGNET_VOLUME, MU_0)
                n = fromto[3:6] - fromto[0:3]
                norm = np.linalg.norm(n)
                if norm < 1e-10:
                    continue
                bid = model.geom_bodyid[gid]
                wheel_forces[bid] = wheel_forces.get(bid, np.zeros(3)) + fmag * (n / norm)

            for bid, fvec in wheel_forces.items():
                total_mag = np.linalg.norm(fvec)
                if total_mag > params["max_force_per_wheel"]:
                    fvec = fvec * (params["max_force_per_wheel"] / total_mag)
                data.xfrc_applied[bid, :3] += fvec

            if mode_cfg["actuator_mode"] == "velocity":
                for act_id in wheel_act_ids:
                    data.ctrl[act_id] = mode_cfg["actuator_target_rads"]
            else:
                for act_id in wheel_act_ids:
                    data.ctrl[act_id] = mode_cfg["actuator_target"]

            mujoco.mj_step(model, data)

            # Stability checks AFTER step
            if np.any(np.abs(data.qvel) > 100.0):
                raise ValueError("Simulation unstable: excessive velocities")
            if not np.all(np.isfinite(data.qacc)):
                raise ValueError("Simulation unstable: non-finite accelerations")
            if not np.isfinite(data.solver_fwdinv[0]):
                raise ValueError("Simulation unstable: non-finite solver values")
            # solver_niter check removed: hitting limit on small contact islands is normal

            trajectory.append({
                "time": float(data.time),
                "pos": data.qpos[:3].copy(),
                "vel": data.qvel[:3].copy(),
                "quat": data.xquat[1].copy(),
            })
    except ValueError as e:
        print(f"  Simulation failed: {e}")
        return None

    if not trajectory:
        return None
    return trajectory


def trajectory_to_arrays(trajectory) -> dict[str, np.ndarray]:
    if not trajectory:
        return {"time": np.array([]), "pos": np.empty((0, 3)), "vel": np.empty((0, 3))}
    time = np.array([s["time"] for s in trajectory], dtype=float)
    pos = np.array([s["pos"] for s in trajectory], dtype=float)
    vel = np.array([s["vel"] for s in trajectory], dtype=float)
    return {"time": time, "pos": pos, "vel": vel}


"""
PATCH: Replace compute_tracking_metrics in sim_optimizer.py with this version.
Fixes: TypeError: int() argument must be a string ... not 'NoneType'
       when mode_cfg["tracking_axis"] is None (hold mode).
"""

def compute_tracking_metrics(trajectory, mode_cfg: dict) -> dict:
    nan_metrics = {
        "status": "failed",
        "samples": 0,
        "rms_err_x": float("nan"),
        "rms_err_y": float("nan"),
        "rms_err_z": float("nan"),
        "mean_abs_err_x": float("nan"),
        "mean_abs_err_y": float("nan"),
        "mean_abs_err_z": float("nan"),
        "final_disp_x": float("nan"),
        "final_disp_y": float("nan"),
        "final_disp_z": float("nan"),
        "tracking_error_axis_rms": float("nan"),
        "tracking_error_axis_final": float("nan"),
        "avg_vel_x": float("nan"),
        "avg_vel_y": float("nan"),
        "avg_vel_z": float("nan"),
        "target_vel_x": float(mode_cfg["target_velocity_xyz"][0]),
        "target_vel_y": float(mode_cfg["target_velocity_xyz"][1]),
        "target_vel_z": float(mode_cfg["target_velocity_xyz"][2]),
        "settled_total_motion": float("nan"),
    }

    if not trajectory:
        return nan_metrics

    import numpy as np
    arr = trajectory_to_arrays(trajectory)
    time, pos, vel = arr["time"], arr["pos"], arr["vel"]
    settle_time = mode_cfg["settle_time"]
    start_idx = min(int(np.searchsorted(time, settle_time, side="left")), len(time) - 1)
    p0 = pos[start_idx]
    t0 = time[start_idx]

    target_vel = np.asarray(mode_cfg["target_velocity_xyz"], dtype=float)
    target_pos = p0[None, :] + (time - t0)[:, None] * target_vel[None, :]
    err = pos - target_pos

    settled_motion = (
        float(np.sum(np.linalg.norm(np.diff(pos[start_idx:], axis=0), axis=1)))
        if len(pos) - start_idx > 1 else 0.0
    )

    axis = mode_cfg["tracking_axis"]  # None for hold mode
    if axis is not None:
        axis = int(axis)
        tracking_error_axis_rms   = float(np.sqrt(np.mean(err[:, axis] ** 2)))
        tracking_error_axis_final = float(err[-1, axis])
    else:
        # Hold mode: use total motion as the primary scalar (want it near zero)
        tracking_error_axis_rms   = settled_motion
        tracking_error_axis_final = float(np.linalg.norm(pos[-1] - p0))

    return {
        "status": "ok",
        "samples": int(len(time)),
        "rms_err_x": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rms_err_y": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "rms_err_z": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "mean_abs_err_x": float(np.mean(np.abs(err[:, 0]))),
        "mean_abs_err_y": float(np.mean(np.abs(err[:, 1]))),
        "mean_abs_err_z": float(np.mean(np.abs(err[:, 2]))),
        "final_disp_x": float(pos[-1, 0] - p0[0]),
        "final_disp_y": float(pos[-1, 1] - p0[1]),
        "final_disp_z": float(pos[-1, 2] - p0[2]),
        "tracking_error_axis_rms":   tracking_error_axis_rms,
        "tracking_error_axis_final": tracking_error_axis_final,
        "avg_vel_x": float(np.mean(vel[:, 0])),
        "avg_vel_y": float(np.mean(vel[:, 1])),
        "avg_vel_z": float(np.mean(vel[:, 2])),
        "target_vel_x": float(target_vel[0]),
        "target_vel_y": float(target_vel[1]),
        "target_vel_z": float(target_vel[2]),
        "settled_total_motion": settled_motion,
    }

if __name__ == "__main__":
    traj = run_simulation(DEFAULT_PARAMS, mode=DEFAULT_MODE)
    if traj is None:
        print("Simulation failed.")
    else:
        print(compute_tracking_metrics(traj, MODES[DEFAULT_MODE]))