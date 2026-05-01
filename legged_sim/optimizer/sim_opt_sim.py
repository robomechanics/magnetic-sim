"""
sim_opt_sim.py — Headless FL-lift runner for the floor-lift optimizer.

Phases executed:
  1. Settle  (SETTLE_TIME s)   — all magnets ON, robot drops onto floor
  2. FL LIFT (sequence "f2w")  — FL runner advances until LIFT phase completes,
                                  then ctrl_targets are frozen (foot held up)
  3. Hold    (LIFT_HOLD s)     — FL held at lifted position; FR/BL/BR drift sampled here

Reward signal (minimise):
  - Mean XYZ drift norm of FR/BL/BR EE from their settled baselines (30% of cost)
  - Mean max(0, settled_z - ee_z) of FR/BL/BR stance feet (30% of cost)
  - Fraction of hold steps where any stance foot had zero mag force (40% of cost)
"""

import fcntl
import os
import sys
import numpy as np
import mujoco
import mink

# ── Paths ─────────────────────────────────────────────────────────────────────
# OPTIMIZER_DIR: directory containing this file (and combined_config.py).
# LEGGED_DIR:    parent directory containing config.py and sequences.py.
# Both are added to sys.path so this file works standalone AND when imported
# as a module by the combined optimizer's worker subprocesses.

OPTIMIZER_DIR = os.path.abspath(os.path.dirname(__file__))
LEGGED_DIR    = os.path.abspath(os.path.join(OPTIMIZER_DIR, ".."))
SCENE_XML     = os.path.join(OPTIMIZER_DIR, "mwc_mjcf", "scene.xml")

for _p in (OPTIMIZER_DIR, LEGGED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Physics constants: prefer combined_config (combined optimizer context);
# fall back to config (standalone / legged_sim context).
try:
    from combined_config import MU_0, MAGNET_VOLUME, MAGNET_BODY_NAMES, TIMESTEP
except ImportError:
    from config import MU_0, MAGNET_VOLUME, MAGNET_BODY_NAMES, TIMESTEP

# Legged-sim-specific: always from the parent legged_sim config.
from config import KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG, bake_joint_angles
from sequences import SEQUENCES, SequenceRunner

# ── Constants ─────────────────────────────────────────────────────────────────

FEET            = ('FL', 'FR', 'BL', 'BR')
STANCE_FEET     = ('FR', 'BL', 'BR')   # feet whose stability is measured during FL lift
SWING_FOOT      = "FL"    # leg lifted during the floor-lift scenario

SETTLE_TIME     = 2.0     # s — all magnets ON
LIFT_HOLD       = 3.0     # s — hold FL at lifted position
LIFT_DZ         = 0.05    # m — foot lift height; exported for viewer use (match your sequence)

# NOTE: "LIFT" must match the phase name in SEQUENCES["f2w"]. Update if different.
LIFT_PHASE_NAME = "LIFT"

IK_DAMPING  = 1e-3
IK_EVERY_N  = 10    # IK solve every N physics steps

PID_KP      = 500.0
PID_KI      = 200.0
PID_KD      = 30.0
PID_I_CLAMP = 100.0


# ── Magnetic force ────────────────────────────────────────────────────────────

def _mag_force(dist: float, Br: float) -> float:
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m ** 2) / (2 * np.pi * (2 * dist) ** 4)


def _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params, off_mids=None) -> dict:
    """Apply magnetic forces and return {body_id: force_magnitude_N} for each magnet body."""
    if off_mids is None:
        off_mids = set()
    _fromto = np.zeros(6)
    magnitudes = {}
    for mid in magnet_ids:
        if mid in off_mids:
            magnitudes[mid] = 0.0
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
        tot = np.linalg.norm(fvec)
        if tot > params['max_force_per_wheel']:
            fvec *= params['max_force_per_wheel'] / tot
            tot   = params['max_force_per_wheel']
        data.xfrc_applied[mid, :3] += fvec
        magnitudes[mid] = tot
    return magnitudes


# ── Model setup ───────────────────────────────────────────────────────────────

def _setup_model(params: dict):
    robot_xml = os.path.join(OPTIMIZER_DIR, "mwc_mjcf", "robot.xml")
    lock_path = robot_xml + ".lock"
    with open(lock_path, "w") as _lock:
        fcntl.flock(_lock, fcntl.LOCK_EX)
        bake_joint_angles(robot_xml)
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        fcntl.flock(_lock, fcntl.LOCK_UN)
    data  = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP
    mujoco.mj_resetData(model, data)

    for leg in FEET:
        for jname, bake_dict in [
            (f'knee_{leg}',  KNEE_BAKE_DEG),
            (f'wrist_{leg}', WRIST_BAKE_DEG),
            (f'ee_{leg}',    EE_BAKE_DEG),
        ]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                data.qpos[model.jnt_qposadr[jid]] = np.radians(bake_dict[leg])
    mujoco.mj_forward(model, data)

    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root_jid != -1:
        data.qpos[model.jnt_qposadr[root_jid]] += 2.05
        mujoco.mj_forward(model, data)

    plate_ids = set()
    for name in ("floor", "wall"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        plate_ids.add(gid)
        model.geom_friction[gid] = params['ground_friction']

    model.opt.o_solref          = params['solref']
    model.opt.o_solimp          = params['solimp']
    model.opt.noslip_iterations = params['noslip_iterations']
    model.opt.noslip_tolerance  = params['noslip_tolerance']
    model.opt.o_margin          = params['margin']
    model.dof_damping[:]        = 2.0
    model.dof_armature[:]       = 0.01

    magnet_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in MAGNET_BODY_NAMES
    ]
    sphere_gids = {
        mid: [gid for gid in range(model.ngeom)
              if model.geom_bodyid[gid] == mid
              and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE]
        for mid in magnet_ids
    }

    # em_coll_gids removed — floor penetration now uses settled Z baseline, not mj_geomDistance
    return model, data, plate_ids, magnet_ids, sphere_gids


# ── PID controller ────────────────────────────────────────────────────────────

class _PID:
    def __init__(self, model):
        self.nu        = model.nu
        self.ctrl_jids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.integral  = np.zeros(model.nu)
        self.prev_err  = np.zeros(model.nu)

    def compute(self, model, data, targets, dt):
        torques = np.zeros(self.nu)
        for i, jid in enumerate(self.ctrl_jids):
            err = targets[i] - data.qpos[model.jnt_qposadr[jid]]
            self.integral[i] = np.clip(self.integral[i] + err * dt, -PID_I_CLAMP, PID_I_CLAMP)
            derr = (err - self.prev_err[i]) / dt if dt > 0 else 0.0
            self.prev_err[i] = err
            torques[i] = PID_KP * err + PID_KI * self.integral[i] + PID_KD * derr
        return torques


# ── IK solver ────────────────────────────────────────────────────────────────

def _se3_from_pos(pos):
    T = np.eye(4); T[:3, 3] = pos
    return mink.SE3.from_matrix(T)


class _IK:
    """Whole-body IK matching sim.py IKSolver, including wall-foot orientation lock."""

    def __init__(self, model):
        self.model  = model
        self.config = mink.Configuration(model)

        self.ee_bids    = {}
        self.foot_tasks = {}
        for foot in FEET:
            frame = f"electromagnet_{foot}"
            self.ee_bids[foot] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame)
            self.foot_tasks[foot] = mink.FrameTask(
                frame_name=frame, frame_type="body",
                position_cost=10.0, orientation_cost=0.0, lm_damping=IK_DAMPING)

        self.body_task = mink.FrameTask(
            frame_name="main_frame", frame_type="body",
            position_cost=50.0, orientation_cost=50.0, lm_damping=IK_DAMPING)
        self.posture_task = mink.PostureTask(model=model, cost=0.01, lm_damping=IK_DAMPING)
        self.config_limit = mink.ConfigurationLimit(model)
        self.ctrl_jids    = [model.actuator_trnid[i, 0] for i in range(model.nu)]

        self.stance_targets  = {}
        self.stance_se3      = {}
        self._ori_target_rot = None

    def ee_pos(self, data, foot) -> np.ndarray:
        return data.xpos[self.ee_bids[foot]].copy()

    def record_stance(self, data):
        for foot in FEET:
            self.stance_targets[foot] = self.ee_pos(data, foot)
        self.posture_task.set_target(data.qpos.copy())
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")
        T = np.eye(4)
        T[:3, :3] = data.xmat[bid].reshape(3, 3)
        T[:3,  3] = data.xpos[bid]
        self.body_task.set_target(mink.SE3.from_matrix(T))

    def lock_stance_orientation(self, data, foot):
        bid = self.ee_bids[foot]
        T = np.eye(4)
        T[:3, :3] = data.xmat[bid].reshape(3, 3)
        T[:3,  3] = data.xpos[bid]
        self.stance_se3[foot] = T

    def solve(self, swing_target, phys_data, dt, swing_foot: str,
              face_axis=None, position_cost=None, orientation_cost=None,
              n_iter=10) -> np.ndarray:
        ik_qpos  = phys_data.qpos.copy()
        pos_cost = position_cost    if position_cost    is not None else 10.0
        ori_cost = orientation_cost if orientation_cost is not None else 0.0

        if face_axis is not None:
            if self._ori_target_rot is None:
                goal = np.asarray(face_axis, float); goal /= np.linalg.norm(goal)
                cur  = phys_data.xmat[self.ee_bids[swing_foot]].reshape(3, 3).copy()
                cur_neg_y = -cur[:, 1]
                axis  = np.cross(cur_neg_y, goal)
                sin_a = np.linalg.norm(axis)
                cos_a = float(np.dot(cur_neg_y, goal))
                if sin_a < 1e-6:
                    self._ori_target_rot = cur.copy() if cos_a >= 0 else (
                        (-np.eye(3) + 2 * np.outer(cur_neg_y, cur_neg_y)) @ cur)
                else:
                    axis /= sin_a
                    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
                    self._ori_target_rot = (np.eye(3) + sin_a*K + (1.-cos_a)*(K@K)) @ cur
            T = np.eye(4)
            T[:3, :3] = self._ori_target_rot
            T[:3,  3] = swing_target
            target_se3 = mink.SE3.from_matrix(T)
        else:
            self._ori_target_rot = None
            target_se3 = _se3_from_pos(swing_target)

        if self.foot_tasks[swing_foot].orientation_cost != ori_cost:
            self.foot_tasks[swing_foot] = mink.FrameTask(
                frame_name=f"electromagnet_{swing_foot}", frame_type="body",
                position_cost=pos_cost, orientation_cost=ori_cost, lm_damping=IK_DAMPING)
        self.foot_tasks[swing_foot].set_target(target_se3)
        self.foot_tasks[swing_foot].position_cost = pos_cost
        self.body_task.orientation_cost = 0.0 if face_axis is not None else 50.0

        for foot in FEET:
            if foot != swing_foot:
                if foot in self.stance_se3:
                    se3_tgt = mink.SE3.from_matrix(self.stance_se3[foot])
                    if self.foot_tasks[foot].orientation_cost != 30.0:
                        self.foot_tasks[foot] = mink.FrameTask(
                            frame_name=f"electromagnet_{foot}", frame_type="body",
                            position_cost=50.0, orientation_cost=30.0, lm_damping=IK_DAMPING)
                    self.foot_tasks[foot].set_target(se3_tgt)
                else:
                    self.foot_tasks[foot].set_target(_se3_from_pos(self.stance_targets[foot]))
                    self.foot_tasks[foot].position_cost = 50.0

        tasks = [self.body_task] + [self.foot_tasks[f] for f in FEET] + [self.posture_task]

        for _ in range(n_iter):
            self.config.update(ik_qpos)
            vel     = mink.solve_ik(self.config, tasks, dt,
                                    solver="quadprog", damping=IK_DAMPING,
                                    limits=[self.config_limit])
            ik_qpos = self.config.integrate(vel, dt)

        ctrl = np.zeros(self.model.nu)
        for i, jid in enumerate(self.ctrl_jids):
            ctrl[i] = ik_qpos[self.model.jnt_qposadr[jid]]
        return ctrl


# ── Main headless runner ──────────────────────────────────────────────────────

def run_headless_floor(params: dict) -> tuple[float, float, float]:
    """
    Run settle → FL LIFT → hold at lifted position.

    Sampling window:
      Start: when FL exits the LIFT phase (foot is up, ctrl_targets frozen)
      End:   LIFT_HOLD seconds later

    Returns:
        (stance_norm, stance_floor_pen, zero_contact_frac) — all in metres / fraction.
        stance_norm:       mean ‖drift‖ of FR/BL/BR EE from their settled baselines.
        stance_floor_pen:  mean max(0, settled_z - ee_z) of FR/BL/BR stance feet;
                           positive when a foot has sagged below its settled height.
        zero_contact_frac: fraction of sampling steps where ANY stance foot had
                           zero magnetic force. 0.0 = full contact, 1.0 = always lost.
        Returns (1.0, 1.0, 1.0) on failure (LIFT phase never completed).
    """
    model, data, plate_ids, magnet_ids, sphere_gids = _setup_model(params)
    ik  = _IK(model)
    pid = _PID(model)

    # foot-name → magnet body ID for zero-contact tracking
    magnet_bid = {
        name.split('_', 1)[1]: bid
        for name, bid in zip(MAGNET_BODY_NAMES, magnet_ids)
    }

    fl_mag_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "electromagnet_FL")
    fl_hip_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_pitch_FL")

    # _wall_dist_fn is passed to the SequenceRunner for completeness (the "f2w"
    # sequence accepts it). In this floor-lift sim the runner exits after LIFT,
    # so the wall-distance phases (F2W_ORIENT, F2W_MEASURE, F2W_REACH) are never
    # reached and this function is never called.
    _wall_gid      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    _ray_geomgroup = np.zeros(6, np.uint8)
    _geomid_out    = np.array([-1], dtype=np.int32)

    def _wall_dist_fn(ray_dir_override=None):
        pos     = data.xpos[fl_mag_bid].copy()
        ray_dir = (np.array(ray_dir_override, float) if ray_dir_override is not None
                   else -data.xmat[fl_mag_bid].reshape(3, 3)[:, 1])
        ray_dir /= np.linalg.norm(ray_dir)
        _geomid_out[0] = -1
        dist = mujoco.mj_ray(model, data, pos, ray_dir, _ray_geomgroup, 1, fl_mag_bid, _geomid_out)
        if dist < 0 or int(_geomid_out[0]) != _wall_gid:
            return np.inf
        return float(dist)

    ctrl_targets = np.array([
        data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
        for i in range(model.nu)
    ])

    fl_off = {fl_mag_bid}

    runner = SequenceRunner(SEQUENCES["f2w"])

    # Sample accumulators — FR/BL/BR stance feet
    stance_norm_samples:      list[float] = []   # mean ‖drift‖ across FR/BL/BR each step
    stance_floor_pen_samples: list[float] = []   # mean Z drop below settled baseline each step
    # Zero-contact tracking — a step is lost if ANY stance foot has zero mag force
    zero_contact_steps: int = 0
    total_sample_steps: int = 0

    settled            = False
    lift_complete      = False
    hold_start_t       = None
    was_in_lift        = False
    ik_counter         = [0]
    target_pos         = np.zeros(3)
    face_axis          = None
    stance_z_baselines: dict[str, float] = {}   # settled Z of each stance EE (floor-drop reference)

    # 10.0 s budget covers the maximum expected f2w sequence duration up to LIFT exit
    max_time = SETTLE_TIME + 10.0 + LIFT_HOLD + 1.0

    while data.time < max_time:
        t = data.time

        # ── Settle ────────────────────────────────────────────────────────────
        if t < SETTLE_TIME:
            data.xfrc_applied[:] = 0
            _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params)
            data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
            mujoco.mj_step(model, data)
            continue

        # ── Post-settle init (once) ────────────────────────────────────────────
        if not settled:
            settled = True
            ik.record_stance(data)
            target_pos = ik.ee_pos(data, "FL").copy()
            runner.start(t, {
                'foot':              "FL",
                'ee_home':           target_pos.copy(),
                'ee_pos_fn':         lambda: ik.ee_pos(data, "FL"),
                'hip_pivot_fn':      lambda: data.xanchor[fl_hip_jid].copy(),
                'wall_dist_fn':      _wall_dist_fn,
                'magnet_disable_fn': lambda: fl_off.add(fl_mag_bid),
                'magnet_enable_fn':  lambda: fl_off.discard(fl_mag_bid),
            })
            # Capture settled Z for each stance foot — reference for floor-drop penalty
            for foot in STANCE_FEET:
                stance_z_baselines[foot] = ik.ee_pos(data, foot)[2]

        # ── Magnets ───────────────────────────────────────────────────────────
        data.xfrc_applied[:] = 0
        mag_forces = _apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params,
                                off_mids=fl_off)

        # ── IK solve (only while still in LIFT phase) ─────────────────────────
        if not lift_complete:
            ik_counter[0] += 1
            if ik_counter[0] >= IK_EVERY_N:
                ik_counter[0] = 0
                target_pos, face_axis, pos_cost, ori_cost = runner.step(t, {
                    'ee_pos_fn':    lambda: ik.ee_pos(data, "FL"),
                    'hip_pivot_fn': lambda: data.xanchor[fl_hip_jid].copy(),
                })
                ctrl_targets = ik.solve(
                    target_pos, data, IK_EVERY_N * TIMESTEP,
                    swing_foot="FL", face_axis=face_axis,
                    position_cost=pos_cost, orientation_cost=ori_cost,
                )

            # ── Detect LIFT phase exit → freeze and start hold ─────────────
            ph = runner.current_phase
            in_lift = (ph is not None and ph.name == LIFT_PHASE_NAME)
            if was_in_lift and not in_lift:
                lift_complete = True
                hold_start_t  = t
            was_in_lift = in_lift

        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

        # ── Sampling during hold ───────────────────────────────────────────────
        # 30%: mean ‖drift‖ of FR/BL/BR EE from their settled baselines.
        # 30%: mean Z drop of FR/BL/BR below their settled Z baselines.
        # 40%: fraction of steps where any stance foot loses magnetic contact.
        if lift_complete:
            norms, pens = [], []
            for foot in STANCE_FEET:
                ee = ik.ee_pos(data, foot)
                norms.append(np.linalg.norm(ee - ik.stance_targets[foot]))
                pens.append(max(0.0, stance_z_baselines[foot] - ee[2]))
            stance_norm_samples.append(float(np.mean(norms)))
            stance_floor_pen_samples.append(float(np.mean(pens)))
            # Zero-contact: step is lost if ANY stance foot has zero mag force
            if any(mag_forces.get(magnet_bid[foot], 0.0) == 0.0 for foot in STANCE_FEET):
                zero_contact_steps += 1
            total_sample_steps += 1

        # ── Stop condition ─────────────────────────────────────────────────────
        if lift_complete and (t - hold_start_t) >= LIFT_HOLD:
            break

    if not stance_norm_samples:
        return 1.0, 1.0, 1.0

    zero_contact_frac = zero_contact_steps / total_sample_steps
    return (float(np.mean(stance_norm_samples)),
            float(np.mean(stance_floor_pen_samples)),
            zero_contact_frac)


if __name__ == "__main__":
    try:
        from combined_config import PARAMS
    except ImportError:
        from config import PARAMS
    print("Running single FL-lift trial with default PARAMS...")
    stance_norm, stance_floor_pen, zero_frac = run_headless_floor(PARAMS)
    print(f"Stance hold — norm: {stance_norm*1000:.2f}mm  "
          f"floor-pen: {stance_floor_pen*1000:.4f}mm  "
          f"zero-contact: {zero_frac*100:.1f}%")