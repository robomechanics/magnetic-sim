import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import imageio
import imageio.plugins.ffmpeg
import pathlib

# TODO: Add max_magnetic_distance to reward we updated reward function with distance %

# Magnetic constants
MU_0 = 4 * np.pi * 1e-7  # Magnetic permeability of free space (H/m)
MAGNET_VOLUME = np.pi * (0.025**2) * 0.013  # Cylinder: r=0.025m, h=0.013m
MAX_FORCE_PER_WHEEL = 50.0  # N

def calculate_magnetic_force(distance, Br, V, MU_0):
    """Compute magnetic attraction force using dipole-dipole approximation."""
    if distance <= 0.0:
        return 0.0
    m = (Br * V) / MU_0
    return (3 * MU_0 * m ** 2) / (2 * np.pi * (2 * distance) ** 4)

def run_simulation(params, mjcf_path="XML/scene.xml", sim_duration=10.0, visualize=False):
    """
    Runs a MuJoCo simulation with given parameters and returns the trajectory.
    
    Returns:
        list or None: Trajectory data or None if unstable.
    """
    # If visualization requested, delegate to viewer module
    if visualize:
        import viewer
        viewer.visualize_simulation(params, sim_duration)
        return None
    
    # Otherwise run headless simulation
    model = mujoco.MjModel.from_xml_path(mjcf_path)

    # Apply parameters
    wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_geom")
    if 'ground_friction' in params:
        model.geom_friction[wall_id] = params['ground_friction']
        
    if 'solref' in params:
        model.opt.o_solref = params['solref']
        
    if 'solimp' in params:
        model.opt.o_solimp = params['solimp']

    if 'noslip_iterations' in params:
        model.opt.noslip_iterations = params['noslip_iterations']

    if 'rocker_stiffness' in params and 'rocker_damping' in params:
        rocker_joints = ['right_hinge', 'left_hinge', 'BR_pivot', 'FR_pivot', 'BL_pivot', 'FL_pivot']
        for joint_name in rocker_joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                model.jnt_stiffness[joint_id] = params['rocker_stiffness']
                model.dof_damping[model.jnt_dofadr[joint_id]] = params['rocker_damping']

    if 'wheel_kp' in params and 'wheel_kv' in params:
        wheel_actuators = ['BR_wheel_motor', 'FR_wheel_motor', 'BL_wheel_motor', 'FL_wheel_motor']
        for act_name in wheel_actuators:
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id != -1:
                model.actuator_gainprm[act_id, 0] = params['wheel_kp']
                model.actuator_biasprm[act_id, 1] = -params['wheel_kv']

    model.opt.timestep = 1./1e3
    model.opt.enableflags |= 1 << 0

    data = mujoco.MjData(model)
    data.qpos[0] = 0.035              # X = 35mm from wall
    data.qpos[1] = 0.0                # Y = centered
    data.qpos[2] = 0.35               # Z = height along wall
    data.qpos[3:7] = [-0.707, 0, 0.707, 0]  # Rotated to face wall

    settle_time = 0.2
    trajectory = []
    
    # Get parameters for magnetic force
    Br = params.get('Br', 1.48)
    max_magnetic_distance = params.get('max_magnetic_distance', 0.01)

    # Get all sampling sphere geoms (24 per wheel = 96 total)
    # TODO: Alternative simpler method - use cylinder center only (4 geoms instead of 96)
    wheel_gids = []
    for wheel_prefix in ['BR', 'FR', 'BL', 'FL']:
        # Find the wheel_geom body
        body_name = f"{wheel_prefix}_wheel_geom"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            continue
        
        # Get all geoms in this body
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id:
                if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    wheel_gids.append(geom_id)
    
    fromto = np.zeros(6)

    try:
        mujoco.mj_step(model, data)

        # Set wheels sideways
        pivot_joints = ['BR_pivot', 'FR_pivot', 'BL_pivot', 'FL_pivot']
        for joint_name in pivot_joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                data.qpos[model.jnt_qposadr[joint_id]] = np.pi/2

        def simulation_step():
            data.xfrc_applied[:] = 0.0 
            
            for gid in wheel_gids:
                if gid == -1 or gid >= model.ngeom:  # Add bounds check
                    continue
                    
                dist = mujoco.mj_geomDistance(model, data, gid, wall_id, 50, fromto)
                
                if dist < 0 or dist > max_magnetic_distance:
                    continue
                
                fmag = calculate_magnetic_force(dist, Br, MAGNET_VOLUME, MU_0)
                fmag = np.clip(fmag, 0.0, MAX_FORCE_PER_WHEEL)

                n = fromto[3:6] - fromto[0:3]
                norm = np.linalg.norm(n)
                if norm > 1e-10:
                    data.xfrc_applied[model.geom_bodyid[gid], :3] = fmag * (n / norm)
            mujoco.mj_step(model, data)

            if not np.all(np.isfinite(data.qacc)):
                raise ValueError("Simulation unstable: Non-finite accelerations.")
            
            if (data.solver_niter >= model.opt.iterations).any():
                raise ValueError("Simulation unstable: Solver iteration limit.")

            if not np.isfinite(data.solver_fwdinv[0]):
                raise ValueError("Simulation unstable: Non-finite solver values.")

            trajectory.append({
                'time': data.time,
                'pos': data.qpos[:3].copy(),
                'vel': data.qvel[:3].copy(),
                'quat': data.xquat[1].copy()
            })

        while data.time < sim_duration:
            simulation_step()

    except ValueError as e:
        if "Simulation unstable" in str(e):
            print(f"  Simulation failed: {e}")
            return None
        raise e

    if not trajectory or not np.all(np.isfinite([d['pos'][0] for d in trajectory])):
        print("Warning: Invalid trajectory.")
        return None
        
    return trajectory

if __name__ == "__main__":
    default_params = {
        'ground_friction': [0.95, 0.01, 0.01],
        'solref': [0.0004, 25.0],
        'solimp': [0.9, 0.95, 0.001, 0.5, 1.0],
        'noslip_iterations': 15,
        'rocker_stiffness': 30.0,
        'rocker_damping': 1.0,
        'wheel_kp': 10.0,
        'wheel_kv': 1.0,
        'Br': 1.48
    }
    
    print("Running simulation...")
    trajectory = run_simulation(default_params, sim_duration=5.0, visualize=True)

    if trajectory:
        settle_time = 0.2
        start_idx = next((i for i, s in enumerate(trajectory) if s['time'] >= settle_time), 0)
        
        total_movement = sum(
            np.linalg.norm(trajectory[i]['pos'] - trajectory[i-1]['pos'])
            for i in range(start_idx + 1, len(trajectory))
        )

        print(f"Total movement: {total_movement:.6f} m")
        print(f"Cost: {total_movement:.6f}")
    else:
        print("Simulation failed.")