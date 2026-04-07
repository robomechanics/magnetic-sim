"""
sim.py — FL lift → 45° CCW arc → arc back → lower.

Phase 1 (0 → SETTLE/2):    gravity only.
Phase 2 (SETTLE/2 → SETTLE): magnets engage.
Phase 3 (SETTLE → done):    FL lift → ARC_OUT → ARC_BACK → LOWER → DONE.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from config import (
    MU_0, MAGNET_VOLUME, SCENE_XML, MAGNET_BODY_NAMES,
    TIMESTEP, SETTLE_TIME, SIM_DURATION, TELEMETRY_INTERVAL,
    PARAMS, MAG_ENABLED, REAL_TIME_FACTOR,
    JOINT_DAMPING, JOINT_ARMATURE, SERVO_KP, SERVO_KV,
    STANCE_KP, STANCE_KV, STANCE_DAMPING,
    KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG,
    bake_joint_angles,
)
from trajectory import SurfacePatch, GAIT_ORDER, FootTarget, HIP_OFFSET_BODY, _quintic
from ik import MinkRobot, CTRL_JOINT_ORDER, EE_FRAME_NAME

TEST_FOOT  = 'FL'
LIFT_HEIGHT = 0.2
ARC_DEG     = 45.0
LIFT_DUR    = 5.0
ARC_DUR     = 10.0

WEIGHT_POS_SWING_TEST  = 15.0
WEIGHT_ORI_SWING_TEST  =  0.0
WEIGHT_POS_STANCE_TEST = 30.0
WEIGHT_ORI_STANCE_TEST = 15.0

FLOOR_QUAT = np.array([0.7071068, -0.7071068, 0.0, 0.0])

FLOOR_SURFACE = SurfacePatch(normal=np.array([ 0.0, 0.0, 1.0]), origin=np.array([-1.0, 0.0, 0.0]))
WALL_SURFACE  = SurfacePatch(normal=np.array([-1.0, 0.0, 0.0]), origin=np.array([ 0.5, 0.0, 0.5]))
SURFACES      = [FLOOR_SURFACE, WALL_SURFACE]

LEG_JOINTS = {
    'FL': ['hip_pitch_FL', 'knee_FL', 'wrist_FL'],
    'FR': ['hip_pitch_FR', 'knee_FR', 'wrist_FR'],
    'BL': ['hip_pitch_BL', 'knee_BL', 'wrist_BL'],
    'BR': ['hip_pitch_BR', 'knee_BR', 'wrist_BR'],
}


def set_leg_stiffness(model, foot, is_stance):
    kp, kv, damping = (STANCE_KP, STANCE_KV, STANCE_DAMPING) if is_stance else (SERVO_KP, SERVO_KV, JOINT_DAMPING)
    for joint_name in LEG_JOINTS[foot]:
        jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        model.dof_damping[model.jnt_dofadr[jid]] = damping
        aid = CTRL_JOINT_ORDER.index(joint_name)
        model.actuator_gainprm[aid, 0] =  kp
        model.actuator_biasprm[aid, 1] = -kv
        model.actuator_biasprm[aid, 2] = -kp


def lock_stance_joints(data, swing_foot, stance_ctrl_targets, joint_indices):
    DRIFT_THRESHOLD = np.radians(2.0)
    for foot in GAIT_ORDER:
        if foot == swing_foot:
            continue
        for joint_name in LEG_JOINTS[foot]:
            qidx, aid = joint_indices[(foot, joint_name)]
            if abs(data.qpos[qidx] - stance_ctrl_targets[(foot, joint_name)]) > DRIFT_THRESHOLD:
                data.ctrl[aid] = stance_ctrl_targets[(foot, joint_name)]


def mag_force(dist, Br):
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)


def setup_model(params):
    bake_joint_angles()
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP
    mujoco.mj_resetData(model, data)

    for leg in ('FL', 'FR', 'BL', 'BR'):
        for jname, bake_dict in [
            (f'knee_{leg}',  KNEE_BAKE_DEG),
            (f'wrist_{leg}', WRIST_BAKE_DEG),
            (f'ee_{leg}',    EE_BAKE_DEG),
        ]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                data.qpos[model.jnt_qposadr[jid]] = np.radians(bake_dict[leg])

    mujoco.mj_forward(model, data)

    plate_ids: set[int] = set()
    for geom_name in ("floor", "wall"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid == -1:
            raise ValueError(f"[sim] Geom '{geom_name}' not found.")
        plate_ids.add(gid)
        model.geom_friction[gid] = params['ground_friction']

    model.opt.o_solref          = params['solref']
    model.opt.o_solimp          = params['solimp']
    model.opt.noslip_iterations = params['noslip_iterations']
    model.opt.noslip_tolerance  = params['noslip_tolerance']
    model.opt.o_margin          = params['margin']

    model.dof_damping[:]         = JOINT_DAMPING
    model.dof_armature[:]        = JOINT_ARMATURE
    model.actuator_gainprm[:, 0] = SERVO_KP
    model.actuator_biasprm[:, 1] = -SERVO_KV
    model.actuator_biasprm[:, 2] = -SERVO_KP

    magnet_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in MAGNET_BODY_NAMES
    ]
    sphere_gids = {
        mid: [
            gid for gid in range(model.ngeom)
            if model.geom_bodyid[gid] == mid and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
        ]
        for mid in magnet_ids
    }
    return model, data, plate_ids, magnet_ids, sphere_gids


def apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params,
              magnet_states=None, foot_for_mid=None):
    if foot_for_mid is None:
        foot_for_mid = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{f}"): f
            for f in GAIT_ORDER
        }
    total_fz = 0.0
    _fromto  = np.zeros(6)
    for mid in magnet_ids:
        foot = foot_for_mid.get(mid)
        if magnet_states is not None and foot is not None and not magnet_states.get(foot, True):
            continue
        fvec = np.zeros(3)
        for gid in sphere_gids[mid]:
            best_dist, best_fromto = np.inf, None
            for pid in plate_ids:
                d = mujoco.mj_geomDistance(model, data, gid, pid, 50.0, _fromto)
                if d < best_dist:
                    best_dist, best_fromto = d, _fromto.copy()
            if best_dist <= 0 or best_dist > params['max_magnetic_distance']:
                continue
            n = best_fromto[3:6] - best_fromto[0:3]
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            fvec += mag_force(best_dist, params['Br']) * (n / norm)
        total_mag = np.linalg.norm(fvec)
        if total_mag > params['max_force_per_wheel']:
            fvec *= params['max_force_per_wheel'] / total_mag
        data.xfrc_applied[mid, :3] += fvec
        total_fz += fvec[2]
    return total_fz


def _rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def _arc_pos(hip_pos, p_lifted, angle_rad):
    center = np.array([hip_pos[0], hip_pos[1], p_lifted[2]])
    return center + _rot_z(angle_rad) @ (p_lifted - center)

def _build_targets(foot_pos, swing_pos):
    targets = []
    for foot in GAIT_ORDER:
        if foot == TEST_FOOT:
            targets.append(FootTarget(foot=foot, pos_world=swing_pos.copy(), quat_world=FLOOR_QUAT,
                                      weight_pos=WEIGHT_POS_SWING_TEST, weight_ori=WEIGHT_ORI_SWING_TEST,
                                      magnet_on=False))
        else:
            targets.append(FootTarget(foot=foot, pos_world=foot_pos[foot].copy(), quat_world=FLOOR_QUAT,
                                      weight_pos=WEIGHT_POS_STANCE_TEST, weight_ori=WEIGHT_ORI_STANCE_TEST,
                                      magnet_on=True))
    return targets

def _print_telemetry(t, phase, phase_t, robot, targets):
    target_map = {ft.foot: ft for ft in targets}
    print(f"\nt={t:.2f}s  [{phase}  phase_t={phase_t:.2f}s]")
    for foot in GAIT_ORDER:
        actual  = robot.ee_pos_world(foot)
        planned = target_map[foot].pos_world
        err_vec_mm = (actual - planned) * 1000
        err_mm     = np.linalg.norm(err_vec_mm)
        role = "SWING " if foot == TEST_FOOT else "stance"
        flag = "  *** BAD TRACK ***" if err_mm > 20.0 else ""
        print(f"  {foot} [{role}]  "
              f"plan=({planned[0]:+.3f},{planned[1]:+.3f},{planned[2]:+.3f})  "
              f"actual=({actual[0]:+.3f},{actual[1]:+.3f},{actual[2]:+.3f})  "
              f"err=({err_vec_mm[0]:+.1f},{err_vec_mm[1]:+.1f},{err_vec_mm[2]:+.1f})mm  "
              f"|e|={err_mm:.1f}mm{flag}")


_key_state = {"paused": True, "step_once": False}

def _key_callback(keycode):
    if keycode in (32, 257):
        _key_state["paused"] = not _key_state["paused"]
    elif keycode == 262:
        _key_state["step_once"] = True


def run(params=None):
    if params is None:
        params = PARAMS

    model, data, plate_ids, magnet_ids, sphere_gids = setup_model(params)
    dt_wall = float(model.opt.timestep) / REAL_TIME_FACTOR

    foot_for_mid = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{f}"): f
        for f in GAIT_ORDER
    }

    phase               = 'SETTLE'
    phase_t             = 0.0
    robot               = None
    foot_pos            = {}
    lifted_pos          = None
    hip_pos             = None
    magnet_states       = {f: True for f in GAIT_ORDER}
    targets             = []
    swing_pos           = np.zeros(3)
    last_print          = -1.0
    stance_ctrl_targets = {}
    joint_indices       = {}
    swing_hip_ctrl      = None
    swing_hip_aid       = None

    print("Press SPACE or ENTER to start.")

    with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as v:
        v.cam.distance  = 2.0
        v.cam.azimuth   = 135
        v.cam.elevation = -20
        v.cam.lookat[:] = [-2.0, 0.0, 0.35]

        while v.is_running() and data.time < SIM_DURATION:

            if _key_state["paused"] and not _key_state["step_once"]:
                v.sync(); time.sleep(0.02); continue
            _key_state["step_once"] = False

            t0 = time.perf_counter()
            data.xfrc_applied[:] = 0.0

            if phase == 'SETTLE':
                if data.time >= SETTLE_TIME / 2 and MAG_ENABLED:
                    apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                              params, magnet_states=None, foot_for_mid=foot_for_mid)

                if data.time >= SETTLE_TIME:
                    mujoco.mj_forward(model, data)
                    robot    = MinkRobot(model, data)
                    foot_pos = {f: robot.ee_pos_world(f) for f in GAIT_ORDER}
                    hip_pos  = robot.body_pos_world() + robot.body_R_world() @ HIP_OFFSET_BODY[TEST_FOOT]
                    lifted_pos = foot_pos[TEST_FOOT] + np.array([0.0, 0.0, LIFT_HEIGHT])
                    swing_pos  = foot_pos[TEST_FOOT].copy()
                    magnet_states[TEST_FOOT] = False

                    for foot in GAIT_ORDER:
                        set_leg_stiffness(model, foot, is_stance=(foot != TEST_FOOT))

                    for foot in GAIT_ORDER:
                        if foot == TEST_FOOT:
                            continue
                        for joint_name in LEG_JOINTS[foot]:
                            jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                            qidx = model.jnt_qposadr[jid]
                            aid  = CTRL_JOINT_ORDER.index(joint_name)
                            joint_indices[(foot, joint_name)]       = (qidx, aid)
                            stance_ctrl_targets[(foot, joint_name)] = data.qpos[qidx]

                    hip_name       = f"hip_pitch_{TEST_FOOT}"
                    hip_jid        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, hip_name)
                    swing_hip_aid  = CTRL_JOINT_ORDER.index(hip_name)
                    swing_hip_ctrl = data.qpos[model.jnt_qposadr[hip_jid]]

                    phase, phase_t = 'LIFT', 0.0
                    print(f"\n[sim] SETTLE done at t={data.time:.2f}s")
                    print(f"  {TEST_FOOT} contact : {foot_pos[TEST_FOOT]}")
                    print(f"  {TEST_FOOT} lifted  : {lifted_pos}")
                    print(f"  hip pivot           : {hip_pos}")
                    print(f"  arc end ({ARC_DEG:.0f}° CCW) : {_arc_pos(hip_pos, lifted_pos, np.radians(ARC_DEG))}")

            else:
                phase_t += TIMESTEP

                if phase == 'LIFT':
                    swing_pos = (1 - _quintic(min(phase_t / LIFT_DUR, 1.0))) * foot_pos[TEST_FOOT] + \
                                _quintic(min(phase_t / LIFT_DUR, 1.0)) * lifted_pos
                    if phase_t >= LIFT_DUR:
                        phase, phase_t = 'ARC_OUT', 0.0; print("[sim] LIFT done → ARC_OUT")

                elif phase == 'ARC_OUT':
                    swing_pos = _arc_pos(hip_pos, lifted_pos, np.radians(ARC_DEG) * _quintic(phase_t / ARC_DUR))
                    if phase_t >= ARC_DUR:
                        phase, phase_t = 'ARC_BACK', 0.0; print("[sim] ARC_OUT done → ARC_BACK")

                elif phase == 'ARC_BACK':
                    swing_pos = _arc_pos(hip_pos, lifted_pos, np.radians(ARC_DEG) * (1.0 - _quintic(phase_t / ARC_DUR)))
                    if phase_t >= ARC_DUR:
                        phase, phase_t = 'LOWER', 0.0; print("[sim] ARC_BACK done → LOWER")

                elif phase == 'LOWER':
                    s = _quintic(phase_t / LIFT_DUR)
                    swing_pos = (1 - s) * lifted_pos + s * foot_pos[TEST_FOOT]
                    if phase_t >= LIFT_DUR:
                        for foot in GAIT_ORDER:
                            set_leg_stiffness(model, foot, is_stance=True)
                        magnet_states[TEST_FOOT] = True
                        phase, phase_t = 'DONE', 0.0; print("[sim] LOWER done → DONE")

                elif phase == 'DONE':
                    swing_pos = foot_pos[TEST_FOOT].copy()

                curr = robot.ee_pos_world(TEST_FOOT).copy()
                if np.linalg.norm(curr - swing_pos) < 0.003:
                    swing_pos = curr

                targets = _build_targets(foot_pos, swing_pos)
                robot.solve_ik(targets, TIMESTEP)

                if phase == 'LIFT':
                    data.ctrl[swing_hip_aid] = swing_hip_ctrl

                lock_stance_joints(data, TEST_FOOT, stance_ctrl_targets, joint_indices)

                if MAG_ENABLED:
                    apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                              params, magnet_states=magnet_states, foot_for_mid=foot_for_mid)

                if data.time - last_print >= TELEMETRY_INTERVAL:
                    last_print = data.time
                    _print_telemetry(data.time, phase, phase_t, robot, targets)

            mujoco.mj_step(model, data)
            v.sync()

            elapsed = time.perf_counter() - t0
            if dt_wall - elapsed > 0:
                time.sleep(dt_wall - elapsed)


if __name__ == "__main__":
    run()