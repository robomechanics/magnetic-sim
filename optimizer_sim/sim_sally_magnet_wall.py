"""
sim_sally_magnet_wall.py

Stage 1: Headless, modular simulation rollout for parameter optimization.

Purpose
-------
This module provides a fast, deterministic MuJoCo rollout function that:
- Applies physics and magnetic parameters
- Runs a fixed-input simulation (no control optimization)
- Returns trajectory data and summary metrics
- Can be called repeatedly by an optimizer or inspection tool

Visualization, keyboard input, and real-time synchronization are intentionally
excluded from this file. Rendering is handled separately.
"""

from __future__ import annotations

import os
import sys
import subprocess
import mujoco
import numpy as np

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from scipy.spatial.transform import Rotation as R

# =============================================================================
# DEFAULT SIMULATION CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = {
    # MJCF scene file
    "xml_file": "scene.xml",

    # Simulation timestep
    "timestep": 0.001,

    # Solver configuration (can become tunable later)
    "integrator": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "o_solref": [4e-4, 25],
    "o_solimp": [0.99, 0.99, 0.001, 0.5, 2],

    # Magnetic model parameters
    "Br": 1.48,
    "magnet_volume": np.pi * ((0.025 / 2) ** 2 - (0.016 / 2) ** 2) * 0.025,
    "max_total_force": 200.0 * 4,
    "target_wheel_speed_rad_s": 5.0,  # Desired wheel rotation speed (rad/s) ≈ 48 RPM

    "MU_0": 4 * np.pi * 1e-7,

    # Wall orientation (euler angles in radians: [roll, pitch, yaw])
    "wall_euler": [0, 0.785398163, 0],  # 45° pitch = 45° incline
    "robot_body_name": "frame",
    "robot_initial_quat": None, # If None, compute from wall_euler automatically
    "robot_position_offset": [0.0, 0.0, 0.0],  # Offset in wall's local frame [along, across, normal]

    # Geometry and actuator naming (must match MJCF)
    "wall_geom_name": "wall_geom",
    "wall_body_name": "magnetic_wall",
    "wheel_names": ["FL_cyl", "FR_cyl", "BL_cyl", "BR_cyl"],
    "actuator_names": [
        "FL_wheel_motor",
        "FR_wheel_motor",
        "BL_wheel_motor",
        "BR_wheel_motor",
    ],
}

def run_xml_generator(mode: str = "sideways") -> str:
    """
    Runs the XML generator script to produce robot_sally_patched_<timestamp>.xml.
    Returns the path to the generated file (in current directory).
    """
    script_path = Path(__file__).parent / "generate_test_magnet_wall_env.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find generator script: {script_path}")

    # Generate unique filename in CURRENT directory (not subfolder)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pid = os.getpid()
    output_file = f"robot_sally_patched_{timestamp}_pid{pid}.xml"
    
    env = os.environ.copy()
    env["MODE"] = mode
    env["OUTPUT_FILE"] = output_file
    env["NO_AUTOLAUNCH"] = "1"
    
    print(f"[INFO] Generating patched robot XML (MODE={mode}) -> {output_file}")
    subprocess.run([sys.executable, str(script_path)], check=True, env=env)
    
    return output_file


def generate_scene_with_robot(patched_robot_file: str) -> str:
    """
    Create a temporary scene.xml that includes the specified patched robot file.
    Returns path to the temporary scene file (in current directory).
    """
    import xml.etree.ElementTree as ET
    
    # Parse original scene.xml
    scene_tree = ET.parse("scene.xml")
    scene_root = scene_tree.getroot()
    
    # Get just the filename for the include
    patched_robot_filename = Path(patched_robot_file).name
    
    # Find and update the include element
    for elem in scene_root.iter():
        if elem.tag == "include":
            old_file = elem.get("file", "")
            if "robot_sally_patched" in old_file:
                elem.set("file", patched_robot_filename)
                break
    
    # Write to temporary file in CURRENT directory (same as assets/)
    temp_scene_file = f"scene_{Path(patched_robot_file).stem}.xml"
    ET.indent(scene_tree, space="  ")
    scene_tree.write(temp_scene_file, encoding="utf-8", xml_declaration=True)
    
    return temp_scene_file

# =============================================================================
# ROLLOUT RESULT STRUCTURE
# =============================================================================
@dataclass
class RolloutResult:
    """
    Container for rollout output.

    termination:
        Reason the rollout ended ("ok" or "unstable")
    steps:
        Number of simulation steps executed
    sim_time:
        Final simulation time in seconds
    trajectory:
        Downsampled state history
    summary:
        Aggregate metrics for evaluation
    params_applied:
        Readback of parameters actually applied to the model
    """
    termination: str
    steps: int
    sim_time: float
    trajectory: List[Dict[str, Any]]
    summary: Dict[str, Any]
    params_applied: Dict[str, Any]


# =============================================================================
# MAGNETIC FORCE MODEL
# =============================================================================
def calculate_magnetic_force(distance: float, Br: float, V: float, MU_0: float) -> float:
    """
    Computes magnetic attraction force magnitude as a function of distance.

    Args:
        distance: Separation between magnet and wall
        Br: Residual flux density
        V: Magnet volume
        MU_0: Magnetic permeability of free space

    Returns:
        Scalar magnetic force magnitude
    """
    if distance <= 0.0:
        return 0.0
    m = (Br * V) / MU_0
    return (3 * MU_0 * m ** 2) / (2 * np.pi * (2 * distance) ** 4)

# =============================================================================
# MODEL CONSTRUCTION AND RESET
# =============================================================================
def build_model(config: Dict[str, Any]) -> Tuple[mujoco.MjModel, Dict[str, Any]]:
    """
    Builds the MuJoCo model and resolves all geometry/body/actuator IDs once.
    """
    model = mujoco.MjModel.from_xml_path(config["xml_file"])

    # Deterministic solver configuration
    model.opt.timestep = float(config["timestep"])
    model.opt.integrator = config["integrator"]
    model.opt.o_solref[:] = np.array(config["o_solref"], dtype=np.float64)
    model.opt.o_solimp[:] = np.array(config["o_solimp"], dtype=np.float64)

    # Apply wall rotation
    if "wall_euler" in config:
        apply_wall_rotation(model, config["wall_euler"], config["wall_body_name"])

    # NOTE: Robot initial pose is applied in reset_data() since it modifies qpos
    
    # Resolve wall geometry
    wall_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, config["wall_geom_name"]
    )
    if wall_id < 0:
        raise ValueError(f"Wall geometry '{config['wall_geom_name']}' not found")

    # Resolve wheel geometries and corresponding bodies
    wheel_geom_ids = []
    wheel_body_ids = []
    wheel_names = []

    for name in config["wheel_names"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        body_name = name.replace("_cyl", "_wheel_geom")
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        if gid >= 0 and bid >= 0:
            wheel_names.append(name)
            wheel_geom_ids.append(gid)
            wheel_body_ids.append(bid)

    if not wheel_geom_ids:
        raise ValueError("No valid wheel geometries found")

    # Resolve wheel actuators
    wheel_actuator_ids = []
    actuator_names = []
    for name in config["actuator_names"]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            actuator_names.append(name)
            wheel_actuator_ids.append(aid)

    ids = {
        "wall_id": wall_id,
        "wheel_names": wheel_names,
        "wheel_geom_ids": wheel_geom_ids,
        "wheel_body_ids": wheel_body_ids,
        "actuator_names": actuator_names,
        "wheel_actuator_ids": wheel_actuator_ids,
    }
    return model, ids


def reset_data(model: mujoco.MjModel, config: Dict[str, Any]) -> mujoco.MjData:
    """
    Creates a fresh simulation state with cleared forces and controls.
    Also sets robot's initial orientation and position.
    """
    data = mujoco.MjData(model)
    data.xfrc_applied[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0
    
    # Apply robot initial pose
    wall_euler = config.get("wall_euler", [0, 0.785398163, 0])
    robot_quat = config.get("robot_initial_quat", None)
    position_offset = config.get("robot_position_offset", [0.0, 0.0, 0.0])
    
    apply_robot_initial_pose(
        model, 
        data, 
        wall_euler, 
        config["robot_body_name"],
        robot_quat,
        position_offset
    )
    
    return data


# In sim_sally_magnet_wall.py, update apply_params():

def apply_params(
    model: mujoco.MjModel, params: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Applies tunable simulation parameters to the model.
    """
    # Existing timestep/solver params
    if "timestep" in params:
        model.opt.timestep = float(params["timestep"])
    if "o_solref" in params:
        model.opt.o_solref[:] = np.array(params["o_solref"], dtype=np.float64)
    if "o_solimp" in params:
        model.opt.o_solimp[:] = np.array(params["o_solimp"], dtype=np.float64)
    
    # Wall rotation (optional tunable parameter)
    if "wall_euler" in params:
        apply_wall_rotation(model, params["wall_euler"], config["wall_body_name"])
    
    # Friction parameters (applied to wheel cylinder geoms)
    # if "wheel_friction" in params:
    #     friction = params["wheel_friction"]  # [sliding, torsional, rolling]
    #     wheel_cyl_names = ["BR_cyl", "FR_cyl", "BL_cyl", "FL_cyl"]
        
    #     for name in wheel_cyl_names:
    #         gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    #         if gid >= 0:
    #             model.geom_friction[gid, :] = np.array(friction, dtype=np.float64)
    
    # Joint damping parameters
    if "wheel_damping" in params:
        wheel_joint_names = ["BR_wheel", "FR_wheel", "BL_wheel", "FL_wheel"]
        for name in wheel_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                model.dof_damping[jid] = float(params["wheel_damping"])
    
    if "rocker_stiffness" in params:
        rocker_joint_names = ["left_hinge", "right_hinge", "BR_pivot", "FR_pivot", "BL_pivot", "FL_pivot"]
        for name in rocker_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                model.jnt_stiffness[jid] = float(params["rocker_stiffness"])
    
    if "rocker_damping" in params:
        rocker_joint_names = ["left_hinge", "right_hinge", "BR_pivot", "FR_pivot", "BL_pivot", "FL_pivot"]
        for name in rocker_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                model.dof_damping[jid] = float(params["rocker_damping"])

    if "wheel_kp" in params or "wheel_kv" in params:
        wheel_actuator_names = ["BR_wheel_motor", "FR_wheel_motor", "BL_wheel_motor", "FL_wheel_motor"]
        for name in wheel_actuator_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                if "wheel_kp" in params:
                    model.actuator_gainprm[aid, 0] = float(params["wheel_kp"])
                if "wheel_kv" in params:
                    model.actuator_biasprm[aid, 2] = float(params["wheel_kv"])
    readback = {
        "timestep": float(model.opt.timestep),
        "o_solref": model.opt.o_solref.copy().tolist(),
        "o_solimp": model.opt.o_solimp.copy().tolist(),
        # "wheel_friction": params.get("wheel_friction", [0.95, 0.01, 0.01]),
        "wheel_damping": params.get("wheel_damping", 0.1),
        "rocker_stiffness": params.get("rocker_stiffness", 30.0),
        "rocker_damping": params.get("rocker_damping", 1.0),
        "wheel_kp": params.get("wheel_kp", 10.0),
        "wheel_kv": params.get("wheel_kv", 1.0),
    }
    
    # Include wall_euler in readback if it was applied
    if "wall_euler" in params:
        readback["wall_euler"] = params["wall_euler"]
    
    return readback

def apply_wall_rotation(model: mujoco.MjModel, euler: List[float], wall_body_name: str) -> None:
    """
    Apply wall rotation from euler angles [roll, pitch, yaw] in radians.
    
    Args:
        model: MuJoCo model
        euler: [roll, pitch, yaw] rotation in radians
        wall_body_name: Name of the wall body in the model
    """
    wall_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wall_body_name)
    if wall_body_id < 0:
        print(f"[WARN] Wall body '{wall_body_name}' not found, skipping rotation")
        return
    
    # Convert euler (XYZ intrinsic) to quaternion
    rot = R.from_euler('xyz', euler, degrees=False)
    quat = rot.as_quat()  # [x, y, z, w] (scipy format)
    
    # MuJoCo uses [w, x, y, z] format
    model.body_quat[wall_body_id] = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)
    
    print(f"[INFO] Applied wall rotation: euler={euler} -> quat=[{quat[3]:.4f}, {quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}]")

def apply_robot_initial_pose(
    model: mujoco.MjModel, 
    data: mujoco.MjData,
    wall_euler: List[float], 
    robot_body_name: str,
    robot_quat: Optional[List[float]] = None,
    position_offset: Optional[List[float]] = None
) -> None:
    """
    Apply robot's initial orientation and position.
    position_offset is in wall's local frame: [along_wall, across_wall, away_from_wall]
    """
    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body_name)
    if robot_body_id < 0:
        print(f"[WARN] Robot body '{robot_body_name}' not found, skipping pose")
        return
    
    # Compute orientation
    if robot_quat is not None:
        quat_mujoco = np.array(robot_quat, dtype=np.float64)
    else:
        wall_rot = R.from_euler('xyz', wall_euler, degrees=False)
        perp_rot = R.from_euler('y', -np.pi/2, degrees=False)
        robot_rot = wall_rot * perp_rot
        
        quat_scipy = robot_rot.as_quat()
        quat_mujoco = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]], dtype=np.float64)
    
    # Compute position offset in world frame
    if position_offset is not None and any(position_offset):
        wall_rot = R.from_euler('xyz', wall_euler, degrees=False)
        # Transform offset from wall's local frame to world frame
        offset_world = wall_rot.apply(position_offset)
    else:
        offset_world = np.zeros(3)
    
    # Apply to freejoint
    for i in range(model.njnt):
        if model.jnt_bodyid[i] == robot_body_id and model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            jnt_qposadr = model.jnt_qposadr[i]
            
            # Position (qpos[0:3])
            current_pos = data.qpos[jnt_qposadr:jnt_qposadr + 3].copy()
            data.qpos[jnt_qposadr:jnt_qposadr + 3] = current_pos + offset_world
            
            # Orientation (qpos[3:7])
            data.qpos[jnt_qposadr + 3:jnt_qposadr + 7] = quat_mujoco
            
            print(f"[INFO] Robot position offset: {offset_world}, orientation: {quat_mujoco}")
            return

# =============================================================================
# HEADLESS ROLLOUT FUNCTION
# =============================================================================
def rollout(
    params: Optional[Dict[str, Any]] = None,
    *,
    sim_duration: float = 5.0,
    config: Optional[Dict[str, Any]] = None,
    fixed_torque: float = 0.5,
    settle_time: float = 0.0,
    log_stride: int = 10,
) -> RolloutResult:
    from metrics import compute_metrics, MetricConfig

    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    params = params or {}

    rollout_mode = params.get("rollout_mode", "drive")
    xml_mode = params.get("mode", "sideways")

    # Generate patched robot XML
    patched_robot_file = run_xml_generator(mode=xml_mode)
    
    # Generate temporary scene.xml that includes the patched robot
    temp_scene_file = generate_scene_with_robot(patched_robot_file)
    
    # Use the temporary scene file
    cfg["xml_file"] = temp_scene_file

    try:
        model, ids = build_model(cfg)
        params_applied = apply_params(model, params, cfg)
        data = reset_data(model, cfg)
        base_pos0 = data.qpos[:3].copy()
        step_records: List[Dict[str, Any]] = []
        
        # Create metric config with proper mode
        metric_cfg = MetricConfig(mode=rollout_mode)

        dt = model.opt.timestep
        n_steps = int(np.ceil(sim_duration / dt))

        Br = params.get("Br", cfg["Br"])
        magnet_volume = params.get("magnet_volume", cfg["magnet_volume"])
        MU_0 = params.get("MU_0", cfg["MU_0"])
        n_wheels = max(1, len(ids["wheel_body_ids"]))
        max_force_per_wheel = cfg["max_total_force"] / n_wheels

        fromto = np.zeros(6)
        trajectory = []
        termination = "ok"
        total_force_accum = 0.0
        force_samples = 0

        # Optional settling phase
        k = -1
        settle_steps = int(np.ceil(settle_time / dt))
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)

        try:
            for k in range(n_steps):
                # Clear applied forces to avoid accumulation
                for bid in ids["wheel_body_ids"]:
                    data.xfrc_applied[bid, :3] = 0.0

                total_force = np.zeros(3)
                wheel_forces = {}
                wheel_dists = []

                # Magnetic force computation
                for name, gid, bid in zip(
                    ids["wheel_names"], ids["wheel_geom_ids"], ids["wheel_body_ids"]
                ):
                    dist = mujoco.mj_geomDistance(
                        model, data, gid, ids["wall_id"], 50, fromto
                    )

                    if dist < 0:
                        wheel_forces[name] = 0.0
                        wheel_dists.append(float("inf"))
                        continue
                    else:
                        wheel_dists.append(float(dist))

                    n = fromto[3:6] - fromto[0:3]
                    norm = np.linalg.norm(n)
                    if norm < 1e-9:
                        wheel_forces[name] = 0.0
                        continue

                    fmag = calculate_magnetic_force(dist, Br, magnet_volume, MU_0)
                    fmag = np.clip(fmag, 0.0, max_force_per_wheel)
                    wheel_forces[name] = float(fmag)

                    fvec = fmag * (n / norm)
                    data.xfrc_applied[bid, :3] = fvec
                    total_force += fvec

                # Fixed actuation
                # for aid in ids["wheel_actuator_ids"]:
                #     data.ctrl[aid] = fixed_torque
                # Position control: set target wheel angle (increments continuously to spin)
                target_speed = cfg.get("target_wheel_speed_rad_s", 5.0)
                for aid in ids["wheel_actuator_ids"]:
                    data.ctrl[aid] = data.time * target_speed

                mujoco.mj_step(model, data)
                if k % 500 == 0:  # Every 100 steps
                    for name, aid in zip(ids["wheel_names"], ids["wheel_actuator_ids"]):
                        joint_id = model.actuator_trnid[aid][0]
                        qpos_addr = model.jnt_qposadr[joint_id]
                        dof_adr = model.jnt_dofadr[joint_id]
                        angle = data.qpos[qpos_addr]
                        #print(f"  {name}: angle={angle:.2f} rad, vel={data.qvel[dof_adr]:.2f} rad/s")               # Instability detection
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    termination = "unstable"
                    break

                # Record per-step data for metrics
                step_records.append({
                    "time": float(data.time),
                    "base_pos": data.qpos[:3].copy(),
                    "wheel_dists": list(wheel_dists),
                })

                # Downsampled logging
                if k % log_stride == 0 or k == n_steps - 1:
                    # Get wheel velocities
                    wheel_vels = {}
                    for name, aid in zip(ids["wheel_names"], ids["wheel_actuator_ids"]):
                        joint_id = model.actuator_trnid[aid][0]
                        dof_adr = model.jnt_dofadr[joint_id]
                        wheel_vels[name] = float(data.qvel[dof_adr])
                    
                    trajectory.append({
                        "time": float(data.time),
                        "qpos": data.qpos[:7].copy(),
                        "qvel": data.qvel[:6].copy(),
                        "total_force": total_force.copy(),
                        "wheel_forces": wheel_forces,
                        "wheel_velocities": wheel_vels,
                    })

                total_force_accum += np.linalg.norm(total_force)
                force_samples += 1

        except Exception as e:
            print(f"[ERROR] Simulation crashed: {e}")
            import traceback
            traceback.print_exc()
            termination = "unstable"

        # After simulation loop, compute metrics:
        metrics_result = compute_metrics(step_records, metric_cfg)

        summary = {
            "final_base_position": data.qpos[:3].copy().tolist(),
            "avg_total_magnetic_force": (
                total_force_accum / max(1, force_samples)
            ),

            # Primary scalar objective
            "reward": metrics_result["reward"],

            # Existing metrics (kept for compatibility/logging)
            "progress_m": metrics_result["progress_m"],
            "progress_rate_mps": metrics_result["progress_rate_mps"],
            "detached": metrics_result["detached"],
            "stuck": metrics_result["stuck"],
            "slip_m": metrics_result["slip_m"],

            # NEW: metrics supporting the new cost function
            "contact_percentage": metrics_result.get("contact_percentage", None),
            "detachment_fraction": metrics_result.get("detachment_fraction", None),
        }

        # Override termination reason if metrics detected a failure
        if metrics_result["termination_reason"] != "ok":
            termination = metrics_result["termination_reason"]

        return RolloutResult(
            termination=termination,
            steps=max(0, k + 1),
            sim_time=float(data.time),
            trajectory=trajectory,
            summary=summary,
            params_applied={
                **params_applied,
                "Br": Br,
                "magnet_volume": magnet_volume,
                "MU_0": MU_0,
                "target_wheel_speed_rad_s": target_speed,
            },
        )
    
    finally:
        # Clean up temporary XML files
        try:
            Path(temp_scene_file).unlink()
            Path(patched_robot_file).unlink()
            print(f"[CLEANUP] Removed {patched_robot_file}")
        except Exception as e:
            pass
    

# =============================================================================
# SIMPLE HEADLESS TEST
# =============================================================================
if __name__ == "__main__":
    result = rollout(sim_duration=2.0)
    for rec in result.trajectory:
        print(f"t={rec['time']:.2f}s: wheel vels = {rec['wheel_velocities']}")
    print("Termination:", result.termination)
    print("Summary:", result.summary)
