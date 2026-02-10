"""
sim_sally_magnet_wall_single.py

Simplified headless simulation for magnetic wall-climbing robot.
Uses ONLY position-based reward for hold mode optimization.

Reward: Negative norm of position change from initial position
Goal: Minimize movement (stay attached to wall)
"""

from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Scene file
    "xml_file": "scene.xml",
    
    # Physics solver
    "timestep": 0.001,
    "integrator": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "o_solref": [4e-4, 25],  # [timeconst, dampratio]
    "o_solimp": [0.99, 0.99, 0.001, 0.5, 2],  # [dmin, dmax, width, mid, power]
    
    # Magnetic parameters
    "Br": 1.48,  # Tesla
    "magnet_volume": np.pi * ((0.025 / 2) ** 2 - (0.016 / 2) ** 2) * 0.025,  # m^3
    "max_total_force": 200.0 * 4,  # N (4 wheels)
    "max_magnetic_distance": 0.01,  # m (1cm cutoff)
    "MU_0": 4 * np.pi * 1e-7,  # Permeability of free space
    
    # Control (no wheel motion in hold mode)
    "target_wheel_speed_rad_s": 0.0,
    
    # Robot pose (SIMPLIFIED - matching multi-mode approach)
    "wall_euler": [0, 0.785398163, 0],  # [roll, pitch, yaw] in radians (45° pitch like multi-mode)
    "robot_body_name": "frame",
    "robot_initial_quat": None,  # If None, compute from wall_euler + perpendicular rotation
    "robot_position_offset": [-0.2, 0.0, 0.0],  # [along, across, normal] in wall frame - XML has position
    
    # Model naming (must match MJCF)
    "wall_geom_name": "wall_geom",
    "wall_body_name": "magnetic_wall",
    "wheel_names": ["FL_cyl", "FR_cyl", "BL_cyl", "BR_cyl"],
    "actuator_names": ["FL_wheel_motor", "FR_wheel_motor", "BL_wheel_motor", "BR_wheel_motor"],
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RolloutResult:
    """Container for simulation rollout output."""
    termination: str  # "ok" or "unstable"
    steps: int
    sim_time: float
    trajectory: List[Dict[str, Any]]
    summary: Dict[str, Any]
    params_applied: Dict[str, Any]


# =============================================================================
# XML GENERATION
# =============================================================================

def run_xml_generator(mode: str = "sideways") -> str:
    """
    Generate patched robot XML file via external generator script.
    
    Args:
        mode: "sideways" for vertical wall hold
        
    Returns:
        Path to generated XML file
    """
    script_path = Path(__file__).parent / "generate_test_magnet_wall_env.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find generator script: {script_path}")
    
    # Generate unique filename per call
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pid = os.getpid()
    output_file = f"robot_sally_patched_{mode}_{timestamp}_pid{pid}.xml"
    
    # Run generator with environment variables
    env = os.environ.copy()
    env["MODE"] = mode
    env["OUTPUT_FILE"] = output_file
    env["NO_AUTOLAUNCH"] = "1"
    
    subprocess.run([sys.executable, str(script_path)], check=True, env=env)
    
    return output_file


def generate_scene_with_robot(patched_robot_file: str) -> str:
    """
    Create temporary scene.xml that includes the specified patched robot file.
    """
    import xml.etree.ElementTree as ET
    
    # Parse original scene
    scene_tree = ET.parse("scene.xml")
    scene_root = scene_tree.getroot()
    
    # Update include element with absolute path
    patched_robot_abs = str(Path(patched_robot_file).absolute())
    for elem in scene_root.iter():
        if elem.tag == "include":
            old_file = elem.get("file", "")
            if "robot_sally" in old_file:
                elem.set("file", patched_robot_abs)
                break
    
    # Write unique scene file
    temp_scene_file = f"scene_{Path(patched_robot_file).stem}.xml"
    ET.indent(scene_tree, space="  ")
    scene_tree.write(temp_scene_file, encoding="utf-8", xml_declaration=True)
    
    return temp_scene_file


# =============================================================================
# MAGNETIC FORCE MODEL
# =============================================================================

def calculate_magnetic_force(distance: float, Br: float, V: float, MU_0: float) -> float:
    """
    Compute magnetic attraction force using dipole-dipole approximation.
    
    Args:
        distance: Separation between magnet and wall (m)
        Br: Residual flux density (T)
        V: Magnet volume (m^3)
        MU_0: Magnetic permeability of free space
        
    Returns:
        Force magnitude (N)
    """
    if distance <= 0.0:
        return 0.0
    
    m = (Br * V) / MU_0
    return (3 * MU_0 * m ** 2) / (2 * np.pi * (2 * distance) ** 4)


# =============================================================================
# MODEL SETUP
# =============================================================================

def build_model(config: Dict[str, Any]) -> Tuple[mujoco.MjModel, Dict[str, Any]]:
    """
    Build MuJoCo model and resolve all geometry/body/actuator IDs.
    
    Args:
        config: Simulation configuration
        
    Returns:
        (model, ids) where ids contains resolved entity IDs
    """
    model = mujoco.MjModel.from_xml_path(config["xml_file"])
    
    # Configure solver
    model.opt.timestep = float(config["timestep"])
    model.opt.integrator = config["integrator"]
    model.opt.o_solref[:] = np.array(config["o_solref"], dtype=np.float64)
    model.opt.o_solimp[:] = np.array(config["o_solimp"], dtype=np.float64)
    
    # Apply wall rotation
    if "wall_euler" in config:
        apply_wall_rotation(model, config["wall_euler"], config["wall_body_name"])
    
    # Resolve wall geometry
    wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, config["wall_geom_name"])
    if wall_id < 0:
        raise ValueError(f"Wall geometry '{config['wall_geom_name']}' not found")
    
    # Resolve wheel geometries and bodies
    wheel_geom_ids = []
    wheel_body_ids = []
    wheel_names = []
    
    for name in config["wheel_names"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        body_name = name.replace("_cyl", "_wheel_geom")
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        
        if gid < 0 or bid < 0:
            print(f"[WARN] Could not resolve wheel: {name} (geom={gid}, body={bid})")
        else:
            wheel_geom_ids.append(gid)
            wheel_body_ids.append(bid)
            wheel_names.append(name)
    
    # Resolve actuators
    actuator_names = []
    wheel_actuator_ids = []
    
    for name in config["actuator_names"]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            print(f"[WARN] Actuator '{name}' not found")
        else:
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


def apply_wall_rotation(model: mujoco.MjModel, euler: List[float], wall_body_name: str) -> None:
    """
    Apply wall rotation from euler angles.
    
    Args:
        model: MuJoCo model
        euler: [roll, pitch, yaw] in radians
        wall_body_name: Name of wall body in model
    """
    wall_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wall_body_name)
    if wall_body_id < 0:
        print(f"[WARN] Wall body '{wall_body_name}' not found")
        return
    
    # Convert euler to quaternion (scipy uses [x,y,z,w], MuJoCo uses [w,x,y,z])
    rot = R.from_euler('xyz', euler, degrees=False)
    quat = rot.as_quat()
    model.body_quat[wall_body_id] = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


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
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        wall_euler: Wall orientation [roll, pitch, yaw]
        robot_body_name: Name of robot body
        robot_quat: Optional explicit quaternion [w,x,y,z]
        position_offset: Optional offset [along, across, normal] in wall frame
    """
    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body_name)
    if robot_body_id < 0:
        print(f"[WARN] Robot body '{robot_body_name}' not found")
        return
    
    # Compute orientation
    if robot_quat is not None:
        quat_mujoco = np.array(robot_quat, dtype=np.float64)
    else:
        # Perpendicular to wall (SAME AS MULTI-MODE)
        wall_rot = R.from_euler('xyz', wall_euler, degrees=False)
        perp_rot = R.from_euler('y', -np.pi/2, degrees=False)
        robot_rot = wall_rot * perp_rot
        quat_scipy = robot_rot.as_quat()
        quat_mujoco = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]], dtype=np.float64)
    
    # Compute position offset in world frame
    if position_offset is not None and any(position_offset):
        wall_rot = R.from_euler('xyz', wall_euler, degrees=False)
        offset_world = wall_rot.apply(position_offset)
    else:
        offset_world = np.zeros(3)
    
    # Apply to freejoint (SAME AS MULTI-MODE - ADDS to current position!)
    for i in range(model.njnt):
        if model.jnt_bodyid[i] == robot_body_id and model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            jnt_qposadr = model.jnt_qposadr[i]
            
            # Position: ADD offset to current position (preserves XML position!)
            current_pos = data.qpos[jnt_qposadr:jnt_qposadr + 3].copy()
            data.qpos[jnt_qposadr:jnt_qposadr + 3] = current_pos + offset_world
            
            # Orientation
            data.qpos[jnt_qposadr + 3:jnt_qposadr + 7] = quat_mujoco
            break
    
    # Forward kinematics to update dependent bodies
    mujoco.mj_forward(model, data)


def reset_data(model: mujoco.MjModel, config: Dict[str, Any]) -> mujoco.MjData:
    """
    Create fresh simulation state with robot in initial pose.
    
    Args:
        model: MuJoCo model
        config: Configuration containing initial pose settings
        
    Returns:
        Initialized MjData
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
        model, data, wall_euler, config["robot_body_name"], robot_quat, position_offset
    )
    
    # Apply wheel rotations based on mode
    mode = config.get("mode", "sideways")
    apply_wheel_rotations(model, data, mode)
    
    return data

def apply_wheel_rotations(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mode: str
) -> None:
    """
    Apply wheel orientation based on mode by setting joint positions.
    
    Sideways mode: wheels oriented for lateral movement
    Drive_up mode: wheels oriented for vertical climbing
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        mode: "sideways" or "drive_up"
    """
    if mode not in ["sideways", "drive_up"]:
        return
    
    # Wheel body names and their target quaternions
    # Format: [w, x, y, z] in MuJoCo convention
    if mode == "sideways":
        wheel_quats = {
            "rocker_linkage": [0.5, -0.5, 0.5, 0.5],
            "rocker_linkage_2": [0.5, -0.5, 0.5, 0.5],
            "rocker_linkage_3": [0.707107, 0, -0.707107, 0],
            "rocker_linkage_4": [0.707107, 0, -0.707107, 0],
        }
    else:  # drive_up
        wheel_quats = {
            "rocker_linkage": [0.707107, 0, 0, 0.707107],
            "rocker_linkage_2": [0.707107, 0, 0, 0.707107],
            "rocker_linkage_3": [0.5, -0.5, -0.5, 0.5],
            "rocker_linkage_4": [0.5, -0.5, -0.5, 0.5],
        }
    
    # Apply rotations to each wheel linkage body
    for body_name, quat in wheel_quats.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        
        # Find the joint associated with this body
        for i in range(model.njnt):
            if model.jnt_bodyid[i] == body_id:
                # Check if it's a ball joint or free joint with rotation
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_BALL:
                    # Ball joint: 4 qpos values (quaternion)
                    qposadr = model.jnt_qposadr[i]
                    data.qpos[qposadr:qposadr+4] = np.array(quat, dtype=np.float64)
                elif model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    # Free joint: 3 pos + 4 quat
                    qposadr = model.jnt_qposadr[i]
                    data.qpos[qposadr+3:qposadr+7] = np.array(quat, dtype=np.float64)
                break

def apply_params(
    model: mujoco.MjModel, params: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply tunable simulation parameters to model.
    
    Args:
        model: MuJoCo model
        params: Parameter dictionary
        config: Base configuration
        
    Returns:
        Readback of applied parameters
    """
    # Solver parameters
    if "timestep" in params:
        model.opt.timestep = float(params["timestep"])
    if "o_solref" in params:
        model.opt.o_solref[:] = np.array(params["o_solref"], dtype=np.float64)
    if "o_solimp" in params:
        model.opt.o_solimp[:] = np.array(params["o_solimp"], dtype=np.float64)
    
    # Wall rotation
    if "wall_euler" in params:
        apply_wall_rotation(model, params["wall_euler"], config["wall_body_name"])
    
    # Apply wheel friction
    if "wheel_friction" in params:
        friction = params["wheel_friction"]
        
        # Apply to wheels
        wheel_cyl_names = ["BR_cyl", "FR_cyl", "BL_cyl", "FL_cyl"]
        for name in wheel_cyl_names:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                model.geom_friction[gid, :] = np.array(friction, dtype=np.float64)
        
        # ALSO apply to wall
        wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, config["wall_geom_name"])
        if wall_id >= 0:
            model.geom_friction[wall_id, :] = np.array(friction, dtype=np.float64)
    
    # Rocker joint stiffness
    if "rocker_stiffness" in params:
        rocker_joints = ["left_hinge", "right_hinge", "BR_pivot", "FR_pivot", "BL_pivot", "FL_pivot"]
        for name in rocker_joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                model.jnt_stiffness[jid] = float(params["rocker_stiffness"])
    
    # Rocker joint damping
    if "rocker_damping" in params:
        rocker_joints = ["left_hinge", "right_hinge", "BR_pivot", "FR_pivot", "BL_pivot", "FL_pivot"]
        for name in rocker_joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                model.dof_damping[jid] = float(params["rocker_damping"])
    
    # Wheel PD gains (applied from XML, but can be overridden here)
    if "wheel_kp" in params or "wheel_kv" in params:
        wheel_actuators = ["BR_wheel_motor", "FR_wheel_motor", "BL_wheel_motor", "FL_wheel_motor"]
        for name in wheel_actuators:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                if "wheel_kp" in params:
                    model.actuator_gainprm[aid, 0] = float(params["wheel_kp"])
                if "wheel_kv" in params:
                    model.actuator_biasprm[aid, 2] = float(params["wheel_kv"])
    
    if "noslip_iterations" in params:
        model.opt.noslip_iterations = int(params["noslip_iterations"])
    
    # Readback
    readback = {
        "timestep": float(model.opt.timestep),
        "o_solref": model.opt.o_solref.copy().tolist(),
        "o_solimp": model.opt.o_solimp.copy().tolist(),
        "wheel_friction": params.get("wheel_friction", [0.95, 0.0005, 0.0001]),
        "rocker_stiffness": params.get("rocker_stiffness", 30.0),
        "rocker_damping": params.get("rocker_damping", 1.0),
        "wheel_kp": params.get("wheel_kp", 10.0),
        "wheel_kv": params.get("wheel_kv", 1.0),
        "noslip_iterations": int(model.opt.noslip_iterations),
    }
    
    if "wall_euler" in params:
        readback["wall_euler"] = params["wall_euler"]
    
    return readback


# =============================================================================
# POSITION-BASED REWARD COMPUTATION
# =============================================================================

def compute_position_reward(trajectory: List[Dict[str, Any]], initial_pos: np.ndarray) -> Dict[str, Any]:
    """
    Compute reward based on position stability.
    
    Reward = -norm(final_position - initial_position)
    Goal: Minimize movement (stay attached)
    
    Args:
        trajectory: List of step records with 'base_pos' entries
        initial_pos: Initial position [x, y, z]
        
    Returns:
        Dictionary with reward and metrics
    """
    if len(trajectory) == 0:
        return {
            "reward": -1000.0,  # Penalty for no data
            "position_drift_m": 1000.0,
            "max_position_drift_m": 1000.0,
            "final_position": initial_pos.tolist(),
        }
    
    # Get final position
    final_pos = trajectory[-1]["base_pos"]
    
    # Compute position drift
    drift = np.linalg.norm(final_pos - initial_pos)
    
    # Compute maximum drift during trajectory
    max_drift = 0.0
    for rec in trajectory:
        pos = rec["base_pos"]
        step_drift = np.linalg.norm(pos - initial_pos)
        max_drift = max(max_drift, step_drift)
    
    # Reward is negative drift (closer to 0 is better)
    reward = -drift
    
    return {
        "reward": float(reward),
        "position_drift_m": float(drift),
        "max_position_drift_m": float(max_drift),
        "final_position": final_pos.tolist(),
        "initial_position": initial_pos.tolist(),
    }


# =============================================================================
# ROLLOUT FUNCTION
# =============================================================================

def rollout(
    params: Optional[Dict[str, Any]] = None,
    sim_duration: float = 5.0,
    settle_time: float = 1.0,
    log_stride: int = 10,
) -> RolloutResult:
    """
    Run headless simulation rollout with position-based reward.
    
    Args:
        params: Parameter overrides
        sim_duration: Simulation duration (s)
        settle_time: Time to settle before main simulation (s)
        log_stride: Trajectory logging frequency
        
    Returns:
        RolloutResult with trajectory and summary
    """
    params = params or {}
    
    # Build configuration
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(params)
    
    # Extract mode from params (defaults to "sideways")
    xml_mode = params.get("mode", "sideways")
    
    # Use XML file from params if provided, otherwise generate one
    if "_xml_file_override" in params:
        patched_robot_file = params["_xml_file_override"]
    else:
        patched_robot_file = "robot_sally_patched.xml"
    
    temp_scene_file = generate_scene_with_robot(patched_robot_file)
    cfg["xml_file"] = temp_scene_file
    cfg["mode"] = xml_mode  # Pass mode to reset_data for wheel rotations
    
    try:
        # Build model and initialize data
        model, ids = build_model(cfg)
        params_applied = apply_params(model, params, cfg)
        data = reset_data(model, cfg)
        
        # Store initial position
        initial_pos = data.qpos[:3].copy()
        
        # Extract parameters
        dt = model.opt.timestep
        Br = params.get("Br", cfg["Br"])
        magnet_volume = params.get("magnet_volume", cfg["magnet_volume"])
        MU_0 = params.get("MU_0", cfg["MU_0"])
        max_magnetic_distance = params.get("max_magnetic_distance", cfg["max_magnetic_distance"])
        n_wheels = max(1, len(ids["wheel_body_ids"]))
        max_force_per_wheel = cfg["max_total_force"] / n_wheels
        
        # =====================================================================
        # SETTLING PHASE: Let robot fall and attach before main simulation
        # =====================================================================
        settle_steps = int(settle_time / dt)
        fromto = np.zeros(6)
        
        for _ in range(settle_steps):
            # Apply magnetic forces (no wheel control)
            for name, gid, bid in zip(ids["wheel_names"], ids["wheel_geom_ids"], ids["wheel_body_ids"]):
                data.xfrc_applied[bid, :3] = 0.0
                dist = mujoco.mj_geomDistance(model, data, gid, ids["wall_id"], 50, fromto)
                
                # Check distance threshold
                if dist >= 0 and dist <= max_magnetic_distance:
                    n = fromto[3:6] - fromto[0:3]
                    norm = np.linalg.norm(n)
                    if norm > 1e-9:
                        fmag = calculate_magnetic_force(dist, Br, magnet_volume, MU_0)
                        fmag = np.clip(fmag, 0.0, max_force_per_wheel)
                        fvec = fmag * (n / norm)
                        data.xfrc_applied[bid, :3] = fvec
            
            mujoco.mj_step(model, data)
        
        # Record initial position AFTER settling
        initial_pos = data.qpos[:3].copy()
        
        # =====================================================================
        # MAIN SIMULATION (HOLD MODE - NO WHEEL MOTION)
        # =====================================================================
        n_steps = int(np.ceil(sim_duration / dt))
        step_records: List[Dict[str, Any]] = []
        trajectory = []
        termination = "ok"
        total_force_accum = 0.0
        force_samples = 0
        k = -1
        
        try:
            for k in range(n_steps):
                # Clear forces
                for bid in ids["wheel_body_ids"]:
                    data.xfrc_applied[bid, :3] = 0.0
                
                total_force = np.zeros(3)
                wheel_forces = {}
                
                # Magnetic force computation
                for name, gid, bid in zip(ids["wheel_names"], ids["wheel_geom_ids"], ids["wheel_body_ids"]):
                    dist = mujoco.mj_geomDistance(model, data, gid, ids["wall_id"], 50, fromto)
                    
                    # Apply magnetic force if within max_magnetic_distance
                    if dist >= 0 and dist <= max_magnetic_distance:
                        n = fromto[3:6] - fromto[0:3]
                        norm = np.linalg.norm(n)
                        if norm > 1e-9:
                            fmag = calculate_magnetic_force(dist, Br, magnet_volume, MU_0)
                            fmag = np.clip(fmag, 0.0, max_force_per_wheel)
                            wheel_forces[name] = float(fmag)
                            
                            fvec = fmag * (n / norm)
                            data.xfrc_applied[bid, :3] = fvec
                            total_force += fvec
                        else:
                            wheel_forces[name] = 0.0
                    else:
                        wheel_forces[name] = 0.0
                
                # NO wheel control (hold mode)
                for aid in ids["wheel_actuator_ids"]:
                    data.ctrl[aid] = 0.0
                
                # Step simulation
                mujoco.mj_step(model, data)
                
                # Check stability
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    termination = "unstable"
                    break
                
                # Record step data
                step_records.append({
                    "time": float(data.time),
                    "base_pos": data.qpos[:3].copy(),
                })
                
                # Downsampled trajectory logging
                if k % log_stride == 0 or k == n_steps - 1:
                    trajectory.append({
                        "time": float(data.time),
                        "qpos": data.qpos[:7].copy(),
                        "qvel": data.qvel[:6].copy(),
                        "total_force": total_force.copy(),
                        "wheel_forces": wheel_forces,
                    })
                
                total_force_accum += np.linalg.norm(total_force)
                force_samples += 1
        
        except Exception as e:
            print(f"[ERROR] Simulation crashed: {e}")
            import traceback
            traceback.print_exc()
            termination = "unstable"
        
        # =====================================================================
        # COMPUTE POSITION-BASED REWARD
        # =====================================================================
        reward_result = compute_position_reward(step_records, initial_pos)
        
        summary = {
            "final_base_position": data.qpos[:3].copy().tolist(),
            "avg_total_magnetic_force": total_force_accum / max(1, force_samples),
            **reward_result,
        }
        
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
                "max_magnetic_distance": max_magnetic_distance,
                "target_wheel_speed_rad_s": 0.0,
            },
        )
    
    finally:
        # Clean up temporary scene file only (keep base robot XML)
        try:
            Path(temp_scene_file).unlink()
        except Exception:
            pass


# =============================================================================
# SIMPLE HEADLESS TEST
# =============================================================================

if __name__ == "__main__":
    result = rollout(sim_duration=2.0)
    print("Termination:", result.termination)
    print("Summary:", result.summary)
    print(f"Position drift: {result.summary['position_drift_m']:.6f} m")
    print(f"Reward: {result.summary['reward']:.6f}")