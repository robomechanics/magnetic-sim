"""
viewer.py - Visualize optimization results

Usage:
    python viewer.py --rank 1 --duration 10.0
"""

import mujoco
import mujoco.viewer
import numpy as np
import csv
import argparse
import sim_optimizer

from config import MODES, DEFAULT_MODE

ARROW_RADIUS = 0.003
key_state = {"paused": True}


def add_arrow(scene, start, end, color):
    if scene.ngeom >= scene.maxgeom:
        return
    length = np.linalg.norm(end - start)
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([ARROW_RADIUS, ARROW_RADIUS, length]),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(color, dtype=np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        ARROW_RADIUS,
        start,
        end,
    )
    scene.ngeom += 1


def key_callback(keycode):
    if keycode == 32:  # space
        key_state["paused"] = not key_state["paused"]
    elif keycode == 257:  # enter
        key_state["paused"] = False

def visualize_simulation(params, sim_duration=None, mode=None):
    """Run simulation with visualization (arrows, viewer)."""
    from config import MODES, DEFAULT_MODE
    if mode is None:
        mode = DEFAULT_MODE
    mode_cfg = MODES[mode]
    if sim_duration is None:
        sim_duration = mode_cfg["sim_duration"]

    model = mujoco.MjModel.from_xml_path("XML/scene.xml")
    
    # Apply parameters (same as sim_optimizer.run_simulation)
    wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_geom")
    model.geom_friction[wall_id] = params['ground_friction']
    model.opt.o_solref = params['solref']
    model.opt.o_solimp = params['solimp']
    model.opt.noslip_iterations = params['noslip_iterations']
    
    rocker_joints = ['right_hinge', 'left_hinge', 'BR_pivot', 'FR_pivot', 'BL_pivot', 'FL_pivot']
    for joint_name in rocker_joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            model.jnt_stiffness[joint_id] = params['rocker_stiffness']
            model.dof_damping[model.jnt_dofadr[joint_id]] = params['rocker_damping']
    
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
    
    Br = params['Br']
    max_mag_dist = params['max_magnetic_distance']

    # Get all sampling sphere geoms (24 per wheel = 96 total)
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

    # Collect wheel actuator IDs for control
    wheel_act_ids = []
    for act_name in ['BR_wheel_motor', 'FR_wheel_motor', 'BL_wheel_motor', 'FL_wheel_motor']:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if act_id != -1:
            wheel_act_ids.append(act_id)

    fromto = np.zeros(6)
    
    mujoco.mj_step(model, data)
        
    # Set pivot angle from mode config and lock it
    pivot_joints = ['BR_pivot', 'FR_pivot', 'BL_pivot', 'FL_pivot']
    for joint_name in pivot_joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            data.qpos[model.jnt_qposadr[joint_id]] = mode_cfg["pivot_angle"]
            model.jnt_stiffness[joint_id] = 1000.0
            model.qpos_spring[model.jnt_qposadr[joint_id]] = mode_cfg["pivot_angle"]

    print(f"\nViewer - Press ENTER to start, SPACE to play/pause")
    print(f"Br: {Br:.3f} T | Max dist: {max_mag_dist*1000:.1f} mm | Duration: {sim_duration}s\n")
    
    model.vis.map.znear = 0.01
    model.vis.map.zfar = 100.0

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -45
        
        while viewer.is_running() and data.time < sim_duration:
            if not key_state["paused"]:
                viewer.user_scn.ngeom = 0
                data.xfrc_applied[:] = 0.0
                
                # Apply magnetic forces with blue arrows
                for gid in wheel_gids:
                    if gid == -1:
                        continue
                    
                    dist = mujoco.mj_geomDistance(model, data, gid, wall_id, 50, fromto)
                    
                    if 0 <= dist <= max_mag_dist:
                        fmag = sim_optimizer.calculate_magnetic_force(dist, Br, sim_optimizer.MAGNET_VOLUME, sim_optimizer.MU_0)
                        max_force = params['max_force_per_wheel']
                        fmag = np.clip(fmag, 0.0, max_force)
                        
                        n = fromto[3:6] - fromto[0:3]
                        norm = np.linalg.norm(n)
                        if norm > 1e-10:
                            n_hat = n / norm
                            data.xfrc_applied[model.geom_bodyid[gid], :3] = fmag * n_hat
                            
                            # Blue arrow scaled by force (2mm per 10N)
                            arrow_len = 0.002 * fmag / 10.0
                            add_arrow(viewer.user_scn, fromto[0:3], fromto[0:3] + arrow_len * n_hat, (0, 0, 1, 1))
                
                # Actuator control
                if mode_cfg["actuator_mode"] == "velocity":
                    for act_id in wheel_act_ids:
                        data.ctrl[act_id] = mode_cfg["actuator_target_rads"] * data.time
                else:
                    for act_id in wheel_act_ids:
                        data.ctrl[act_id] = mode_cfg["actuator_target"]

                mujoco.mj_step(model, data)

                # Print COM velocity every 0.5s
                if int(data.time * 1000) % 500 == 0:
                    frame_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "frame")
                    vel = data.cvel[frame_id, 3:]  # linear velocity (cvel is [angular(3), linear(3)])
                    print(f"  t={data.time:.2f}s | vel: X={vel[0]:.4f} Y={vel[1]:.4f} Z={vel[2]:.4f} m/s | speed={np.linalg.norm(vel):.4f} m/s")
            
            viewer.sync()


def main():
    parser = argparse.ArgumentParser(description="Visualize optimization results")
    parser.add_argument("--rank", type=int, default=1, help="Rank to visualize (1=best)")
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration (s)")
    parser.add_argument("--mode", type=str, default="hold", help="Mode: hold, drive_sideways")
    args = parser.parse_args()
    
    # Load results from CSV
    try:
        with open(f'optimization_results_{args.mode}.csv', 'r') as f:
            results = list(csv.DictReader(f))
    except FileNotFoundError:
        print("Error: optimization_results.csv not found. Run tune_params.py first.")
        return
    
    if args.rank > len(results):
        print(f"Error: Rank {args.rank} out of bounds (max: {len(results)})")
        return
    
    s = results[args.rank - 1]
    print(f"\nRank #{args.rank}")
    print(f"Cost: {float(s['cost']):.6f}")
    
    # Reconstruct parameters
    params = {
        'ground_friction': [float(s['sliding_friction']), float(s['torsional_friction']), float(s['rolling_friction'])],
        'solref': [float(s['solref_timeconst']), float(s['solref_dampratio'])],
        'solimp': [float(s['solimp_dmin']), float(s['solimp_dmax']), float(s['solimp_width']), 0.5, 1.0],
        'noslip_iterations': int(s['noslip_iterations']),
        'Br': float(s['Br']),
        'max_magnetic_distance': float(s['max_magnetic_distance']),
        'rocker_stiffness': float(s['rocker_stiffness']),
        'rocker_damping': float(s['rocker_damping']),
        'wheel_kp': float(s['wheel_kp']),
        'wheel_kv': float(s['wheel_kv']),
        'max_force_per_wheel': float(s['max_force_per_wheel']),
    }
    
    visualize_simulation(params, args.duration, mode=args.mode)


if __name__ == "__main__":
    main()