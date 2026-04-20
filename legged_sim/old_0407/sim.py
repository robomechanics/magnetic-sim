"""
sim.py — Sally IK sim

Motion sequence (quintic profile):
    LIFT       : FL foot up 2 cm          (3 s)
    SWING_OUT  : arc 45° around hip       (3 s)
    SWING_BACK : arc back to start        (3 s)
    LOWER      : FL foot down 2 cm        (3 s)
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from legged_sim.old_0407.config import (
    MU_0, MAGNET_VOLUME, SCENE_XML, MAGNET_BODY_NAMES,
    TIMESTEP, SETTLE_TIME, SIM_DURATION, TELEMETRY_INTERVAL,
    PARAMS, MAG_ENABLED, REAL_TIME_FACTOR,
    JOINT_DAMPING, JOINT_ARMATURE, SERVO_KP, SERVO_KV,
    STANCE_KP, STANCE_KV, STANCE_DAMPING,
    KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG,
    bake_joint_angles,
)
from legged_sim.old_0407.trajectory import GAIT_ORDER, FootTarget, HIP_OFFSET_BODY, _quintic
from legged_sim.old_0407.ik import MinkRobot, CTRL_JOINT_ORDER

# ── Motion parameters ─────────────────────────────────────────────────────────
TEST_FOOT   = 'FL'
LIFT_HEIGHT = 0.02
ARC_DEG     = 45.0
LIFT_DUR    = 3.0
ARC_DUR     = 3.0

WEIGHT_POS_SWING  = 15.0
WEIGHT_ORI_SWING  =  0.0
WEIGHT_POS_STANCE = 30.0
WEIGHT_ORI_STANCE = 15.0

FLOOR_QUAT = np.array([0.7071068, -0.7071068, 0.0, 0.0])

LEG_JOINTS = {
    'FL': ['hip_pitch_FL', 'knee_FL', 'wrist_FL'],
    'FR': ['hip_pitch_FR', 'knee_FR', 'wrist_FR'],
    'BL': ['hip_pitch_BL', 'knee_BL', 'wrist_BL'],
    'BR': ['hip_pitch_BR', 'knee_BR', 'wrist_BR'],
}

PHASE_FROZEN_JOINTS = {
    'LIFT':  lambda foot: LEG_JOINTS[foot][:1],
    'LOWER': lambda foot: LEG_JOINTS[foot][:1],
}

_JOINT_LIMITS = {'hip_pitch_FL': 45, 'knee_FL': 45, 'wrist_FL': 67.5, 'ee_FL': 45}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def _arc_pos(hip_pos, p_lifted, angle_rad):
    center = np.array([hip_pos[0], hip_pos[1], p_lifted[2]])
    return center + _rot_z(angle_rad) @ (p_lifted - center)

def _build_targets(foot_pos, swing_pos):
    targets = []
    for foot in GAIT_ORDER:
        if foot == TEST_FOOT:
            targets.append(FootTarget(
                foot=foot, pos_world=swing_pos.copy(), quat_world=FLOOR_QUAT,
                weight_pos=WEIGHT_POS_SWING, weight_ori=WEIGHT_ORI_SWING, magnet_on=False,
            ))
        else:
            targets.append(FootTarget(
                foot=foot, pos_world=foot_pos[foot].copy(), quat_world=FLOOR_QUAT,
                weight_pos=WEIGHT_POS_STANCE, weight_ori=WEIGHT_ORI_STANCE, magnet_on=True,
            ))
    return targets


# ── Magnetics ─────────────────────────────────────────────────────────────────
def _mag_force(dist, Br):
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)

def apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params,
              magnet_states=None, foot_for_mid=None):
    _fromto = np.zeros(6)
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
            n    = best_fromto[3:6] - best_fromto[0:3]
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            fvec += _mag_force(best_dist, params['Br']) * (n / norm)
        total = np.linalg.norm(fvec)
        if total > params['max_force_per_wheel']:
            fvec *= params['max_force_per_wheel'] / total
        data.xfrc_applied[mid, :3] += fvec


# ── Model setup ───────────────────────────────────────────────────────────────
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

    magnet_ids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in MAGNET_BODY_NAMES]
    sphere_gids = {
        mid: [g for g in range(model.ngeom)
              if model.geom_bodyid[g] == mid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        for mid in magnet_ids
    }
    return model, data, plate_ids, magnet_ids, sphere_gids


# ── Telemetry ─────────────────────────────────────────────────────────────────
def _print_telemetry(t, phase_name, phase_t, phase_dur, robot, targets, model, data):
    target_map  = {ft.foot: ft for ft in targets}
    pct         = int(100 * min(phase_t / phase_dur, 1.0))
    sw          = robot.ee_pos_world(TEST_FOOT)
    sp          = target_map[TEST_FOOT].pos_world
    se          = (sw - sp) * 1000
    smag        = np.linalg.norm(se)
    flag        = "!!!" if smag > 20 else "ok "
    stance_errs = " ".join(
        f"{f}={np.linalg.norm((robot.ee_pos_world(f) - target_map[f].pos_world)*1000):.0f}"
        for f in GAIT_ORDER if f != TEST_FOOT
    )
    print(f"t={t:6.2f}s [{phase_name:<10} {pct:3d}%] "
          f"SWING tgt=({sp[0]:+.3f},{sp[1]:+.3f},{sp[2]:+.3f}) "
          f"e=({se[0]:+.0f},{se[1]:+.0f},{se[2]:+.0f})mm {smag:4.0f}mm {flag} "
          f"| stance {stance_errs}mm")

    parts = []
    for jname in LEG_JOINTS[TEST_FOOT] + [f"ee_{TEST_FOOT}"]:
        jid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        aid   = CTRL_JOINT_ORDER.index(jname)
        ctrl  = np.degrees(data.ctrl[aid])
        qpos  = np.degrees(data.qpos[model.jnt_qposadr[jid]])
        lim   = _JOINT_LIMITS.get(jname, 45)
        sat   = "*" if abs(qpos) >= lim - 1 else " "
        short = jname.replace(f"_{TEST_FOOT}", "").replace("hip_pitch", "hip")
        parts.append(f"{short}={ctrl:+5.1f}/{qpos:+5.1f}{sat}")
    print(f"  joints  {' | '.join(parts)}")


# ── Pause / play ──────────────────────────────────────────────────────────────
_key_state = {"paused": True, "step_once": False}

def _key_callback(keycode):
    if keycode in (32, 257):
        _key_state["paused"] = not _key_state["paused"]
        print(f"[sim] {'PAUSED' if _key_state['paused'] else 'RUNNING'}")
    elif keycode == 262:
        _key_state["step_once"] = True


# ── Main ──────────────────────────────────────────────────────────────────────
def run(params=None):
    if params is None:
        params = PARAMS

    model, data, plate_ids, magnet_ids, sphere_gids = setup_model(params)
    dt_wall = float(model.opt.timestep) / REAL_TIME_FACTOR

    foot_for_mid = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{f}"): f
        for f in GAIT_ORDER
    }

    settled       = False
    last_print    = -TELEMETRY_INTERVAL
    robot         = None
    foot_pos      = {}
    swing_pos     = np.zeros(3)
    magnet_states = {f: True for f in GAIT_ORDER}
    targets       = []

    # Stance state — locked at settle time
    stance_ctrl_targets = {}
    joint_indices       = {}

    # Phase sequencer — each entry: (name, duration_s, target_fn(t_norm) -> pos)
    phase_seq  = []
    phase_idx  = 0
    phase_t    = 0.0

    print("Press SPACE or ENTER to start. RIGHT ARROW to single-step.")

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

            # ── Settle ────────────────────────────────────────────────────────
            if not settled:
                if data.time >= SETTLE_TIME / 2 and MAG_ENABLED:
                    apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                              params, foot_for_mid=foot_for_mid)

                if data.time >= SETTLE_TIME:
                    mujoco.mj_forward(model, data)
                    settled = True

                    robot       = MinkRobot(model, data)
                    foot_pos    = {f: robot.ee_pos_world(f) for f in GAIT_ORDER}
                    contact_pos = foot_pos[TEST_FOOT].copy()
                    lifted_pos  = contact_pos + np.array([0.0, 0.0, LIFT_HEIGHT])
                    hip_pos     = robot.body_pos_world() + robot.body_R_world() @ HIP_OFFSET_BODY[TEST_FOOT]
                    swing_pos   = contact_pos.copy()
                    magnet_states[TEST_FOOT] = False

                    # Set per-foot actuator stiffness
                    for foot in GAIT_ORDER:
                        is_stance = (foot != TEST_FOOT)
                        kp  = STANCE_KP      if is_stance else SERVO_KP
                        kv  = STANCE_KV      if is_stance else SERVO_KV
                        dmp = STANCE_DAMPING if is_stance else JOINT_DAMPING
                        for jname in LEG_JOINTS[foot]:
                            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                            model.dof_damping[model.jnt_dofadr[jid]] = dmp
                            aid = CTRL_JOINT_ORDER.index(jname)
                            model.actuator_gainprm[aid, 0] =  kp
                            model.actuator_biasprm[aid, 1] = -kv
                            model.actuator_biasprm[aid, 2] = -kp

                    # Snapshot stance joint positions
                    for foot in GAIT_ORDER:
                        if foot == TEST_FOOT:
                            continue
                        for jname in LEG_JOINTS[foot]:
                            jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                            qidx = model.jnt_qposadr[jid]
                            aid  = CTRL_JOINT_ORDER.index(jname)
                            joint_indices[(foot, jname)]       = (qidx, aid)
                            stance_ctrl_targets[(foot, jname)] = data.qpos[qidx]

                    # Build phase table
                    cp, lp, hp = contact_pos.copy(), lifted_pos.copy(), hip_pos.copy()
                    deg = np.radians(ARC_DEG)
                    phase_seq = [
                        ('LIFT',       LIFT_DUR, lambda t, cp=cp, lp=lp: (1-t)*cp + t*lp),
                        ('SWING_OUT',  ARC_DUR,  lambda t, hp=hp, lp=lp, deg=deg: _arc_pos(hp, lp, deg*t)),
                        ('SWING_BACK', ARC_DUR,  lambda t, hp=hp, lp=lp, deg=deg: _arc_pos(hp, lp, deg*(1-t))),
                        ('LOWER',      LIFT_DUR, lambda t, cp=cp, lp=lp: (1-t)*lp + t*cp),
                    ]
                    phase_idx, phase_t = 0, 0.0
                    print(f"\n[sim] SETTLE done — starting IK sequence")
                    print(f"  contact={contact_pos}  lifted={lifted_pos}")
                    print(f"  arc end={_arc_pos(hip_pos, lifted_pos, np.radians(ARC_DEG))}")

            # ── IK loop ───────────────────────────────────────────────────────
            else:
                # Step phase sequencer
                if phase_idx < len(phase_seq):
                    name, dur, target_fn = phase_seq[phase_idx]
                    swing_pos = target_fn(_quintic(phase_t / dur))

                    phase_t += TIMESTEP
                    if phase_t >= dur:
                        phase_t = 0.0
                        phase_idx += 1
                        next_name = phase_seq[phase_idx][0] if phase_idx < len(phase_seq) else 'DONE'
                        print(f"[sim] {name} done → {next_name}")
                        if next_name == 'DONE':
                            magnet_states[TEST_FOOT] = True
                else:
                    name      = 'DONE'
                    dur       = 1.0
                    swing_pos = foot_pos[TEST_FOOT].copy()

                # Set targets and solve IK
                targets   = _build_targets(foot_pos, swing_pos)
                freeze_fn = PHASE_FROZEN_JOINTS.get(name if phase_idx < len(phase_seq) else 'DONE')
                frozen    = freeze_fn(TEST_FOOT) if freeze_fn else None
                robot.solve_ik(targets, TIMESTEP, frozen_joints=frozen)

                # Hold stance joints at settled positions
                for foot in GAIT_ORDER:
                    if foot == TEST_FOOT:
                        continue
                    for jname in LEG_JOINTS[foot]:
                        _, aid = joint_indices[(foot, jname)]
                        data.ctrl[aid] = stance_ctrl_targets[(foot, jname)]

                if MAG_ENABLED:
                    apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                              params, magnet_states=magnet_states, foot_for_mid=foot_for_mid)

                if data.time - last_print >= TELEMETRY_INTERVAL:
                    last_print = data.time
                    _print_telemetry(data.time, name, phase_t, dur, robot, targets, model, data)

            mujoco.mj_step(model, data)
            v.sync()

            elapsed = time.perf_counter() - t0
            if dt_wall - elapsed > 0:
                time.sleep(dt_wall - elapsed)


if __name__ == "__main__":
    run()