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

def visualize_simulation(params, sim_duration=10.0):
    """Run simulation with visualization (arrows, viewer)."""
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
    
    Br = params.get('Br', 1.48)
    max_mag_dist = params.get('max_magnetic_distance', 0.01)

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

    fromto = np.zeros(6)  # Add this - was missing!
    
    mujoco.mj_step(model, data)
    
    # Set wheels sideways
    for joint_name in ['BR_pivot', 'FR_pivot', 'BL_pivot', 'FL_pivot']:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid != -1:
            data.qpos[model.jnt_qposadr[jid]] = np.pi/2
    
    print(f"\nViewer - Press ENTER to start, SPACE to play/pause")
    print(f"Br: {Br:.3f} T | Max dist: {max_mag_dist*1000:.1f} mm | Duration: {sim_duration}s\n")
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 0.15
        
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
                        fmag = np.clip(fmag, 0.0, sim_optimizer.MAX_FORCE_PER_WHEEL)
                        
                        n = fromto[3:6] - fromto[0:3]
                        norm = np.linalg.norm(n)
                        if norm > 1e-10:
                            n_hat = n / norm
                            data.xfrc_applied[model.geom_bodyid[gid], :3] = fmag * n_hat
                            
                            # Blue arrow scaled by force (2mm per 10N)
                            arrow_len = 0.002 * fmag / 10.0
                            add_arrow(viewer.user_scn, fromto[0:3], fromto[0:3] + arrow_len * n_hat, (0, 0, 1, 1))
                
                # # Red arrows for wheel angular velocity
                # for gid in wheel_gids:
                #     if gid == -1:
                #         continue
                #     bid = model.geom_bodyid[gid]
                #     vel = data.qvel[model.body_dofadr[bid]:model.body_dofadr[bid]+6]
                #     ang_vel = vel[3:6]
                #     ang_speed = np.linalg.norm(ang_vel)
                #     if ang_speed > 0.1:
                #         arrow_len = 0.01 * ang_speed
                #         add_arrow(viewer.user_scn, data.xpos[bid], data.xpos[bid] + arrow_len * ang_vel/ang_speed, (1, 0, 0, 1))
                
                mujoco.mj_step(model, data)
            
            viewer.sync()


def main():
    parser = argparse.ArgumentParser(description="Visualize optimization results")
    parser.add_argument("--rank", type=int, default=1, help="Rank to visualize (1=best)")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (s)")
    args = parser.parse_args()
    
    # Load results from CSV
    try:
        with open('optimization_results.csv', 'r') as f:
            results = list(csv.DictReader(f))
    except FileNotFoundError:
        print("Error: optimization_results.csv not found. Run tune_params.py first.")
        return
    
    if args.rank > len(results):
        print(f"Error: Rank {args.rank} out of bounds (max: {len(results)})")
        return
    
    s = results[args.rank - 1]
    print(f"\nRank #{args.rank}")
    print(f"Cost: {float(s['cost']):.6f} | Movement: {float(s['total_movement']):.4f} m")
    
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
    }
    
    visualize_simulation(params, args.duration)


if __name__ == "__main__":
    main()