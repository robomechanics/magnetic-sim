"""
MWC sim: robot on ground, IK-controlled FL foot swing sequence.
Trajectory logic lives in sequences.py; this file owns sim, IK, PID, and viz.

Usage:
    python sim.py                    # GUI, orient sequence (default)
    python sim.py --sequence f2w     # GUI, floor-to-wall sequence
    python sim.py --headless --sequence orient
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import mink

# ── paths ─────────────────────────────────────────────────────────────────────
LEGGED_DIR = os.path.join(os.path.dirname(__file__), "..", "legged_sim")
SCENE_XML  = os.path.join(LEGGED_DIR, "mwc_mjcf", "scene.xml")
sys.path.insert(0, LEGGED_DIR)

from config import (
    MU_0, MAGNET_VOLUME, MAGNET_BODY_NAMES, PARAMS,
    TIMESTEP, JOINT_DAMPING, JOINT_ARMATURE,
    KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG,
    bake_joint_angles, SEQUENCE as DEFAULT_SEQUENCE,
)
from sequences import SEQUENCES, SequenceRunner, IKTarget, PhaseContext, ControlMode
from viewer import draw_markers, _build_joint_vis


# ── constants ─────────────────────────────────────────────────────────────────

FEET        = ('FL', 'FR', 'BL', 'BR')
SWING_FOOT  = "FL"
SETTLE_TIME = 2.0
IK_DAMPING  = 1e-3
IK_EVERY_N  = 10      # physics steps between IK solves

# f2w_both: seconds to dwell in F2W_REACH before considering a foot "planted".
DUAL_REACH_DWELL = 3.0

PID_KP, PID_KI, PID_KD, PID_I_CLAMP = 500.0, 200.0, 30.0, 100.0

_BAR_WIDTH = 12


# ── helpers ───────────────────────────────────────────────────────────────────

def _se3_pos(pos):
    T = np.eye(4); T[:3, 3] = pos
    return mink.SE3.from_matrix(T)

def _mag_force(dist, Br):
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)


# ── physics ───────────────────────────────────────────────────────────────────

def apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, off_mids=frozenset()):
    """Apply dipole-dipole magnetic forces for all active magnets."""
    _ft = np.zeros(6)
    for mid in magnet_ids:
        if mid in off_mids:
            continue
        fvec = np.zeros(3)
        for gid in sphere_gids[mid]:
            best_d, best_ft = np.inf, None
            for pid in plate_ids:
                d = mujoco.mj_geomDistance(model, data, gid, pid, 50.0, _ft)
                if d < best_d:
                    best_d, best_ft = d, _ft.copy()
            if best_d <= 0 or best_d > PARAMS['max_magnetic_distance']:
                continue
            n = best_ft[3:6] - best_ft[0:3]
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            fvec += _mag_force(best_d, PARAMS['Br']) * (n / norm)
        tot = np.linalg.norm(fvec)
        if tot > PARAMS['max_force_per_wheel']:
            fvec *= PARAMS['max_force_per_wheel'] / tot
        data.xfrc_applied[mid, :3] += fvec


def setup_model():
    """Load, configure, and return (model, data, plate_ids, magnet_ids, sphere_gids)."""
    bake_joint_angles(os.path.join(LEGGED_DIR, "mwc_mjcf", "robot.xml"))
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP
    mujoco.mj_resetData(model, data)

    for leg in FEET:
        for jname, bake in [(f'knee_{leg}',  KNEE_BAKE_DEG),
                            (f'wrist_{leg}', WRIST_BAKE_DEG),
                            (f'ee_{leg}',    EE_BAKE_DEG)]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                data.qpos[model.jnt_qposadr[jid]] = np.radians(bake[leg])
    mujoco.mj_forward(model, data)

    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root_jid != -1:
        data.qpos[model.jnt_qposadr[root_jid]] += 2.1 # Spawn offset
        mujoco.mj_forward(model, data)

    plate_ids = set()
    for name in ("floor", "wall"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        plate_ids.add(gid)
        model.geom_friction[gid] = PARAMS['ground_friction']
    for i in range(model.ngeom):
        if i not in plate_ids:
            model.geom_rgba[i, 3] = 0.5

    model.opt.o_solref          = PARAMS['solref']
    model.opt.o_solimp          = PARAMS['solimp']
    model.opt.noslip_iterations = PARAMS['noslip_iterations']
    model.opt.noslip_tolerance  = PARAMS['noslip_tolerance']
    model.opt.o_margin          = PARAMS['margin']
    model.dof_damping[:]  = 2.0
    model.dof_armature[:] = 0.01

    magnet_ids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                   for n in MAGNET_BODY_NAMES]
    sphere_gids = {
        mid: [gid for gid in range(model.ngeom)
              if model.geom_bodyid[gid] == mid
              and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE]
        for mid in magnet_ids
    }
    return model, data, plate_ids, magnet_ids, sphere_gids


# ── IK ────────────────────────────────────────────────────────────────────────

class IKSolver:
    """Whole-body IK: stance feet locked, swing foot tracks IKTarget."""

    def __init__(self, model):
        self.model  = model
        self.config = mink.Configuration(model)

        self.ee_bids    = {}
        self.foot_tasks = {}
        for foot in FEET:
            frame = f"electromagnet_{foot}"
            self.ee_bids[foot]    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame)
            self.foot_tasks[foot] = mink.FrameTask(
                frame_name=frame, frame_type="body",
                position_cost=10.0, orientation_cost=0.0, lm_damping=IK_DAMPING)

        self.body_task    = mink.FrameTask(frame_name="main_frame", frame_type="body",
                                           position_cost=50.0, orientation_cost=50.0,
                                           lm_damping=IK_DAMPING)
        self.posture_task = mink.PostureTask(model=model, cost=0.01, lm_damping=IK_DAMPING)
        self.config_limit = mink.ConfigurationLimit(model)
        self.ctrl_jids    = [model.actuator_trnid[i, 0] for i in range(model.nu)]

        self.stance_targets  = {}
        self.stance_se3      = {}   # foot → 4×4 matrix for wall-planted feet (pos + rot)
        self._ori_target_rot = None # cached rotation for the current orientation phase

        # ── joint-keypoint task (universal across joints / legs) ────────────
        self.joint_task     = None
        self._joint_task_key: tuple = ((), 0.0)

        # ── T1: per-phase joint-limit override (Control 1) ──────────────────
        # Default ranges saved once; restored after each C1 solve.
        # _override_config_limit is rebuilt only when jnt_range_override changes
        # (cache-keyed by frozenset, same pattern as _joint_task_key).
        self._default_jnt_range:    np.ndarray                       = model.jnt_range.copy()
        self._range_override_key:   frozenset                        = frozenset()
        self._override_config_limit: Optional[mink.ConfigurationLimit] = None

        # ── T2: per-phase body target override ───────────────────────────────
        # Set by record_stance(); restored each tick when body_target_se3 is None.
        self._settled_body_T: Optional[np.ndarray] = None

    def ee_pos(self, data, foot) -> np.ndarray:
        return data.xpos[self.ee_bids[foot]].copy()

    def joint_qpos(self, data, name: str) -> float:
        """Read current qpos for a named joint. Used by phases that capture
        on-entry joint values for joint-space smooth-stepping."""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint_qpos: unknown joint '{name}'")
        return float(data.qpos[self.model.jnt_qposadr[jid]])

    def _ensure_joint_task(self, dof_adrs: list, joint_cost: float):
        """Build/rebuild self.joint_task if the active DOF set or scalar cost
        changed. mink's PostureTask bakes the per-DOF cost vector at __init__,
        so any change to the cost pattern requires reconstruction. Targets are
        updated separately via set_target each tick (cheap, no rebuild)."""
        key = (tuple(sorted(int(a) for a in dof_adrs)), float(joint_cost))
        if key == self._joint_task_key:
            return
        cost_vec = np.zeros(self.model.nv)
        for adr in dof_adrs:
            cost_vec[int(adr)] = joint_cost
        self.joint_task = mink.PostureTask(
            model=self.model, cost=cost_vec, lm_damping=IK_DAMPING)
        self._joint_task_key = key
        print(f"[ik] joint_task rebuilt: {len(dof_adrs)} active DOFs, "
              f"cost={joint_cost}  dofs={list(self._joint_task_key[0])}")

    def lock_stance_orientation(self, data, foot):
        """Snapshot full SE3 for a wall-planted foot so IK holds both pos and rot."""
        bid = self.ee_bids[foot]
        T = np.eye(4)
        T[:3, :3] = data.xmat[bid].reshape(3, 3)
        T[:3,  3] = data.xpos[bid]
        self.stance_se3[foot] = T
        print(f"[ik] orientation locked for stance foot {foot}  "
              f"-Y={(-T[:3,1]).round(3)}")

    def record_stance(self, data):
        """Snapshot EE positions and body pose as stance references."""
        for foot in FEET:
            self.stance_targets[foot] = self.ee_pos(data, foot)
        self.posture_task.set_target(data.qpos.copy())
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")
        T = np.eye(4)
        T[:3, :3] = data.xmat[bid].reshape(3, 3)
        T[:3,  3] = data.xpos[bid]
        self._settled_body_T = T.copy()   # T2: save for restore when body_target_se3 is None
        self.body_task.set_target(mink.SE3.from_matrix(T))
        print("[ik] stance targets recorded:")
        for f, p in self.stance_targets.items():
            print(f"  {f}: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")

    def solve(self, target: IKTarget, phys_data, dt: float, n_iter: int = 10,
              swing_foot: str = None) -> np.ndarray:
        """Run IK given an IKTarget and return joint position targets (ctrl array).

        The IKTarget carries position, optional face_axis (for EE orientation),
        and per-task costs — replacing the old (pos, face_axis, pos_cost, ori_cost) signature.
        """
        _sf     = swing_foot if swing_foot is not None else SWING_FOOT
        ik_qpos = phys_data.qpos.copy()

        # ── build swing-foot SE3 target ───────────────────────────────────────
        if target.face_axis is not None:
            # Compute and cache the orientation rotation on first appearance.
            if self._ori_target_rot is None:
                self._ori_target_rot = self._compute_ori_target(
                    phys_data.xmat[self.ee_bids[_sf]].reshape(3, 3).copy(),
                    target.face_axis)
                print(f"[ik] orientation locked — EE -Y -> "
                      f"{(-self._ori_target_rot[:, 1]).round(3)}")
            T = np.eye(4)
            T[:3, :3] = self._ori_target_rot
            T[:3,  3] = target.position
            target_se3 = mink.SE3.from_matrix(T)
        else:
            # No orientation constraint — reset cache for next phase.
            self._ori_target_rot = None
            target_se3 = _se3_pos(target.position)

        # ── rebuild swing-foot task if cost changed (mink bakes at construction) ──
        if self.foot_tasks[_sf].orientation_cost != target.orientation_cost:
            self.foot_tasks[_sf] = mink.FrameTask(
                frame_name=f"electromagnet_{_sf}", frame_type="body",
                position_cost=target.position_cost,
                orientation_cost=target.orientation_cost,
                lm_damping=IK_DAMPING)
        self.foot_tasks[_sf].set_target(target_se3)
        self.foot_tasks[_sf].position_cost = target.position_cost

        # Body orientation competes with EE orientation task — zero it during
        # orient phases, UNLESS body_target_se3 is explicitly commanding a pose.
        if target.face_axis is not None and target.body_target_se3 is None:
            self.body_task.orientation_cost = 0.0
        else:
            self.body_task.orientation_cost = 50.0

        # ── T2: body target override ──────────────────────────────────────────
        # Every tick: set body task target from IKTarget or fall back to settled.
        if target.body_target_se3 is not None:
            self.body_task.set_target(mink.SE3.from_matrix(target.body_target_se3))
        elif self._settled_body_T is not None:
            self.body_task.set_target(mink.SE3.from_matrix(self._settled_body_T))

        # ── stance feet ────────────────────────────────────────────────────────
        for foot in FEET:
            if foot == _sf:
                continue
            if foot in self.stance_se3:
                # Wall foot: lock both position and rotation.
                se3_target = mink.SE3.from_matrix(self.stance_se3[foot])
                if self.foot_tasks[foot].orientation_cost != 30.0:
                    self.foot_tasks[foot] = mink.FrameTask(
                        frame_name=f"electromagnet_{foot}", frame_type="body",
                        position_cost=50.0, orientation_cost=30.0, lm_damping=IK_DAMPING)
                self.foot_tasks[foot].set_target(se3_target)
            else:
                # Floor foot: position-only lock.
                self.foot_tasks[foot].set_target(_se3_pos(self.stance_targets[foot]))
                self.foot_tasks[foot].position_cost = 50.0

        # ── DONE: joint-space directive (universal joint-keypoint task) ───────
        # When IKTarget carries joint_targets, build a per-DOF-cost PostureTask
        # whose target qpos copies live qpos for every DOF except the listed
        # joints (which are overwritten with the requested values). Inactive
        # DOFs get cost 0, so their target value is irrelevant.
        joint_active = bool(target.joint_targets)
        if joint_active:
            target_q   = phys_data.qpos.copy()
            dof_adrs   = []
            for jname, jval in target.joint_targets.items():
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    print(f"[ik] WARN joint '{jname}' not found in model; skipping")
                    continue
                target_q[self.model.jnt_qposadr[jid]] = float(jval)
                dof_adrs.append(self.model.jnt_dofadr[jid])
            self._ensure_joint_task(dof_adrs, target.joint_cost)
            self.joint_task.set_target(target_q)
            joint_active = bool(dof_adrs)   # downgrade if every name was bogus

        tasks = [self.body_task] + [self.foot_tasks[f] for f in FEET] + [self.posture_task]
        if joint_active:
            tasks.append(self.joint_task)

        # ── T1: select configuration limit (C1 uses widened ranges) ──────────
        if target.jnt_range_override:
            key = frozenset(
                (jname, lo, hi)
                for jname, (lo, hi) in target.jnt_range_override.items()
            )
            if key != self._range_override_key:
                # Temporarily widen model.jnt_range to bake a new ConfigurationLimit,
                # then immediately restore defaults. ConfigurationLimit stores ranges
                # at construction time, so the model is safe to restore right away.
                for jname, (lo, hi) in target.jnt_range_override.items():
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    if jid >= 0:
                        self.model.jnt_range[jid] = [lo, hi]
                self._override_config_limit = mink.ConfigurationLimit(self.model)
                self.model.jnt_range[:] = self._default_jnt_range
                self._range_override_key = key
                print(f"[ik] C1 limit rebuilt: "
                      f"{ {k: (round(lo,3), round(hi,3)) for k, (lo, hi) in target.jnt_range_override.items()} }")
            active_limit = self._override_config_limit
        else:
            active_limit = self.config_limit

        for _ in range(n_iter):
            self.config.update(ik_qpos)
            vel     = mink.solve_ik(self.config, tasks, dt,
                                    solver="quadprog", damping=IK_DAMPING,
                                    limits=[active_limit])
            ik_qpos = self.config.integrate(vel, dt)

        ctrl = np.zeros(self.model.nu)
        for i, jid in enumerate(self.ctrl_jids):
            ctrl[i] = ik_qpos[self.model.jnt_qposadr[jid]]
        return ctrl

    @staticmethod
    def _compute_ori_target(current_rot: np.ndarray, face_axis) -> np.ndarray:
        """Minimal rotation aligning EE local -Y to face_axis (Rodrigues)."""
        goal = np.asarray(face_axis, float); goal /= np.linalg.norm(goal)
        cur_neg_y = -current_rot[:, 1]
        axis  = np.cross(cur_neg_y, goal)
        sin_a = np.linalg.norm(axis)
        cos_a = float(np.dot(cur_neg_y, goal))
        if sin_a < 1e-6:
            return current_rot.copy() if cos_a >= 0 else (
                (-np.eye(3) + 2 * np.outer(cur_neg_y, cur_neg_y)) @ current_rot)
        axis /= sin_a
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return (np.eye(3) + sin_a * K + (1. - cos_a) * (K @ K)) @ current_rot


# ── control ───────────────────────────────────────────────────────────────────

class PIDController:
    """Joint-space PID: position targets → torque commands."""

    def __init__(self, model):
        self.nu        = model.nu
        self.ctrl_jids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.integral  = np.zeros(model.nu)
        self.prev_err  = np.zeros(model.nu)

    def compute(self, model, data, targets, dt) -> np.ndarray:
        torques = np.zeros(self.nu)
        for i, jid in enumerate(self.ctrl_jids):
            err = targets[i] - data.qpos[model.jnt_qposadr[jid]]
            self.integral[i] = np.clip(self.integral[i] + err * dt, -PID_I_CLAMP, PID_I_CLAMP)
            derr = (err - self.prev_err[i]) / dt if dt > 0 else 0.
            self.prev_err[i] = err
            torques[i] = PID_KP * err + PID_KI * self.integral[i] + PID_KD * derr
        return torques


# ── surface contact / penetration ─────────────────────────────────────────────

def _foot_surfaces(swing_foot: str, wall_feet: set) -> dict:
    """Return expected surface for each foot: 'floor', 'wall', or 'swing'."""
    out = {}
    for foot in FEET:
        if foot == swing_foot:
            out[foot] = 'swing'
        elif foot in wall_feet:
            out[foot] = 'wall'
        else:
            out[foot] = 'floor'
    return out


_CONTACT_LIFT_THRESH = 0.008   # m — EE more than 8 mm above settle baseline = lost contact


def _read_surface_penetration(data, ik, foot_surfaces: dict,
                               wall_baselines: dict) -> dict:
    """Measure EE intrusion into the expected surface using relative displacement.

    Floor feet: drop below ik.stance_targets[foot][2] = floor penetration.
    Wall feet:  advance past wall_baselines[foot][0]  = wall penetration.
    """
    result = {}
    for foot in FEET:
        surface = foot_surfaces.get(foot, 'floor')
        ee = data.xpos[ik.ee_bids[foot]]

        if surface == 'swing':
            result[foot] = ('swing', False, 0.0)

        elif surface == 'floor':
            if foot not in ik.stance_targets:
                result[foot] = ('floor', False, 0.0)
                continue
            baseline_z = ik.stance_targets[foot][2]
            drop       = baseline_z - ee[2]
            contact    = (ee[2] - baseline_z) < _CONTACT_LIFT_THRESH
            result[foot] = ('floor', contact, max(0.0, drop))

        elif surface == 'wall':
            if foot not in wall_baselines:
                result[foot] = ('wall', False, 0.0)
                continue
            baseline_x = wall_baselines[foot][0]
            advance    = ee[0] - baseline_x
            result[foot] = ('wall', True, max(0.0, advance))

    return result


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", choices=list(SEQUENCES.keys()), default=DEFAULT_SEQUENCE)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--magnets",  action="store_true")
    parser.add_argument("--no-ik",    action="store_true")
    args = parser.parse_args()

    # ── setup ──────────────────────────────────────────────────────────────────
    model, data, plate_ids, magnet_ids, sphere_gids = setup_model()
    ik  = IKSolver(model)
    pid = PIDController(model)

    # ── mutable swing-foot state ───────────────────────────────────────────────
    swing_foot_ref    = [SWING_FOOT]
    hip_jid_ref       = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                            f"hip_pitch_{SWING_FOOT}")]
    swing_mag_bid_ref = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                            f"electromagnet_{SWING_FOOT}")]
    body_id           = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")
    joint_vis         = _build_joint_vis(model)

    def _swing_foot():    return swing_foot_ref[0]
    def _hip_jid():       return hip_jid_ref[0]
    def _swing_mag_bid(): return swing_mag_bid_ref[0]

    # ── f2w wall-distance helper ───────────────────────────────────────────────
    _wall_gid      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    _ray_geomgroup = np.zeros(6, np.uint8)
    _geomid_out    = np.array([-1], dtype=np.int32)

    def _compute_max_reach(foot: str) -> float:
        from itertools import product as iproduct
        scratch   = mujoco.MjData(model)
        ee_bid    = ik.ee_bids[foot]
        foot_jids = [(model.actuator_trnid[i, 0], model.jnt_qposadr[model.actuator_trnid[i, 0]])
                     for i in range(model.nu)
                     if foot in (mujoco.mj_id2name(
                         model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0]) or "")]
        fhip_jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"hip_pitch_{foot}")
        limit_vals = [[float(lo), float(hi)] if model.jnt_limited[jid] else [-np.pi, np.pi]
                      for jid, _ in foot_jids for lo, hi in [model.jnt_range[jid]]]
        best = 0.0
        for combo in iproduct(*limit_vals):
            scratch.qpos[:] = data.qpos[:]
            for k, (_, qadr) in enumerate(foot_jids):
                scratch.qpos[qadr] = combo[k]
            mujoco.mj_kinematics(model, scratch)
            d = np.linalg.norm(scratch.xpos[ee_bid] - scratch.xanchor[fhip_jid])
            if d > best: best = d
        return best

    _max_reach_cache: dict = {}

    def _get_max_reach(foot: str) -> float:
        if foot not in _max_reach_cache:
            r = _compute_max_reach(foot)
            _max_reach_cache[foot] = r
            print(f"[f2w] {foot} max reach from hip: {r * 1000:.1f} mm")
        return _max_reach_cache[foot]

    _get_max_reach(SWING_FOOT)

    def _wall_dist_fn(ray_dir_override=None) -> float:
        sf      = _swing_foot()
        ee_bid  = ik.ee_bids[sf]
        pos     = data.xpos[ee_bid].copy()
        ray_dir = (np.array(ray_dir_override, float) if ray_dir_override is not None
                   else -data.xmat[ee_bid].reshape(3, 3)[:, 1])
        ray_dir /= np.linalg.norm(ray_dir)

        _geomid_out[0] = -1
        dist    = mujoco.mj_ray(model, data, pos, ray_dir, _ray_geomgroup, 1, ee_bid, _geomid_out)
        hit_gid = int(_geomid_out[0])

        if dist < 0 or hit_gid != _wall_gid:
            print(f"[f2w] ⚠  Wall ray missed for {sf}")
            return np.inf

        fhip_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"hip_pitch_{sf}")
        if np.linalg.norm(pos + dist * ray_dir - data.xanchor[fhip_jid]) > _get_max_reach(sf):
            print(f"[f2w] ✗  Wall UNREACHABLE for {sf} — move robot closer and restart.")
            sys.exit(1)

        return float(dist)

    # ── sim state ──────────────────────────────────────────────────────────────
    ctrl_targets = np.array([data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
                              for i in range(model.nu)])

    _is_dual    = (args.sequence == "f2w")
    _first_seq  = "f2w" if _is_dual else args.sequence
    _second_seq = "f2w_fr"
    runner      = SequenceRunner(SEQUENCES[_first_seq])
    fr_started  = [False]

    swing_off  = {_swing_mag_bid()}
    settled    = False
    ik_target  = IKTarget.position_only(np.zeros(3))   # updated each IK tick
    last_print = 0.0
    ik_counter = [0]

    wall_feet      = set()
    wall_baselines = {}
    _fr_reach_t0   = [None]

    print(f"sequence={args.sequence}  phases={len(SEQUENCES[_first_seq])}  "
          f"magnets={'on' if args.magnets else 'off'}"
          + ("  [FL → FR]" if _is_dual else ""))

    # ── runner-start helper ────────────────────────────────────────────────────
    def _start_runner_for(foot: str, seq_name: str, t: float):
        """Wire up a fresh SequenceRunner for `foot` using a typed PhaseContext."""
        nonlocal runner
        runner   = SequenceRunner(SEQUENCES[seq_name])
        ee_home  = ik.ee_pos(data, foot).copy()
        ik._ori_target_rot = None   # reset cached orientation for new foot
        ctx = PhaseContext(
            foot=foot,
            ee_home=ee_home,
            ee_pos_fn=lambda: ik.ee_pos(data, swing_foot_ref[0]),
            hip_pivot_fn=lambda: data.xanchor[hip_jid_ref[0]].copy(),
            wall_dist_fn=_wall_dist_fn,
            magnet_disable_fn=lambda: swing_off.add(swing_mag_bid_ref[0]),
            magnet_enable_fn=lambda: swing_off.discard(swing_mag_bid_ref[0]),
            # DONE: joint-space readback for phases driving joint waypoints
            joint_qpos_fn=lambda jname: ik.joint_qpos(data, jname),
        )
        runner.start(t, ctx)
        print(f"\n── starting {seq_name} for {foot} ──  phases={len(SEQUENCES[seq_name])}")

    # ── sim step ───────────────────────────────────────────────────────────────
    def sim_step():
        nonlocal ctrl_targets, settled, ik_target, last_print
        t = data.time

        data.xfrc_applied[:] = 0
        if args.magnets and settled:
            apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, off_mids=swing_off)

        # ── settle phase ───────────────────────────────────────────────────────
        if t < SETTLE_TIME:
            data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
            mujoco.mj_step(model, data)
            if t - last_print >= 0.5:
                last_print = t
                pct = t / SETTLE_TIME
                bar = "█" * int(pct * _BAR_WIDTH) + "░" * (_BAR_WIDTH - int(pct * _BAR_WIDTH))
                print(f"t={t:5.1f}  [{bar}] SETTLE {pct*100:5.1f}%")
            return

        if not settled:
            settled = True
            if not args.no_ik:
                ik.record_stance(data)
                print(f"[diag] EE local +X at settle: "
                      f"{data.xmat[ik.ee_bids[_swing_foot()]].reshape(3,3)[:,0].round(3)}")
                _start_runner_for(_swing_foot(), _first_seq, t)
                ik_target = IKTarget.position_only(ik.ee_pos(data, _swing_foot()).copy())

        # ── dual-leg transition: FL planted → launch FR ────────────────────────
        if _is_dual and not fr_started[0]:
            ph = runner.current_phase
            fl_planted = (
                runner.done
                or (ph is not None
                    and ph.name == "F2W_REACH"
                    and (data.time - runner.phase_t0) >= DUAL_REACH_DWELL)
            )
            if fl_planted:
                fr_started[0] = True
                runner.force_complete()
                ik.record_stance(data)
                ik.lock_stance_orientation(data, "FL")
                swing_foot_ref[0]    = "FR"
                hip_jid_ref[0]       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                          "hip_pitch_FR")
                new_fr_mag           = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                                          "electromagnet_FR")
                swing_mag_bid_ref[0] = new_fr_mag
                swing_off.clear()
                swing_off.add(new_fr_mag)
                _start_runner_for("FR", _second_seq, t)

        # ── wall_feet tracking ─────────────────────────────────────────────────
        if _is_dual and fr_started[0] and "FL" not in wall_feet:
            wall_feet.add("FL")
            wall_baselines["FL"] = ik.ee_pos(data, "FL").copy()

        if _is_dual and fr_started[0] and "FR" not in wall_feet:
            ph = runner.current_phase
            if ph is not None and ph.name == "F2W_REACH":
                if _fr_reach_t0[0] is None:
                    _fr_reach_t0[0] = t
                if (t - _fr_reach_t0[0]) >= DUAL_REACH_DWELL:
                    wall_feet.add("FR")
                    wall_baselines["FR"] = ik.ee_pos(data, "FR").copy()
            else:
                _fr_reach_t0[0] = None

        # ── IK solve ───────────────────────────────────────────────────────────
        if not args.no_ik:
            ik_counter[0] += 1
            if ik_counter[0] >= IK_EVERY_N:
                ik_counter[0] = 0
                sf = _swing_foot()
                ik_target = runner.step(
                    t,
                    ee_pos_fn=lambda: ik.ee_pos(data, swing_foot_ref[0]),
                    hip_pivot_fn=lambda: data.xanchor[hip_jid_ref[0]].copy(),
                )

                if ik_target.control_mode == ControlMode.C3:
                    # ── Control 3: direct PID bypass (no IK solve) ──────────
                    # ctrl_override joints written straight to PID position refs.
                    # Unlisted joints hold their last ctrl_targets value.
                    for jname, jval in ik_target.ctrl_override.items():
                        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                        if jid < 0:
                            print(f"[bypass] WARN joint '{jname}' not in model")
                            continue
                        qadr = model.jnt_qposadr[jid]
                        for i, cjid in enumerate(ik.ctrl_jids):
                            if model.jnt_qposadr[cjid] == qadr:
                                ctrl_targets[i] = jval
                                break
                else:
                    # ── Control 1 / Control 2: IK solve ─────────────────────
                    # C1: jnt_range_override widens limits inside ik.solve().
                    # C2: default limits, standard mink QP.
                    ctrl_targets = ik.solve(ik_target, data, IK_EVERY_N * TIMESTEP, swing_foot=sf)

        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

        if t - last_print < 1.0:
            return
        last_print = t

        if args.no_ik:
            print(f"t={t:5.1f}  body_z={data.xpos[body_id, 2]:.4f}")
            return

        sf = _swing_foot()
        phase_name, pct = runner.progress(t)
        bar = "█" * int(pct * _BAR_WIDTH) + "░" * (_BAR_WIDTH - int(pct * _BAR_WIDTH))

        actual  = ik.ee_pos(data, sf)
        diff    = (actual - ik_target.position) * 1000
        pos_str = (f"  pos_err={np.linalg.norm(diff):5.1f}mm"
                   f"  (Δx={diff[0]:+5.1f} Δy={diff[1]:+5.1f} Δz={diff[2]:+5.1f})"
                   if ik_target is not None else "  pos=unconstrained")

        ori_str = ""
        if ik_target.face_axis is not None:
            ee_neg_y = -data.xmat[ik.ee_bids[sf]].reshape(3, 3)[:, 1]
            fa_norm  = ik_target.face_axis / np.linalg.norm(ik_target.face_axis)
            ang      = np.degrees(np.arccos(np.clip(np.dot(ee_neg_y, fa_norm), -1., 1.)))
            ori_str  = f"  ang_to_goal={ang:5.1f}°"

        worst_d, worst_f = 0., ""
        for foot in FEET:
            if foot == sf: continue
            d = np.linalg.norm(ik.ee_pos(data, foot) - ik.stance_targets[foot]) * 1000
            if d > worst_d: worst_d, worst_f = d, foot

        cur_mag_bid = _swing_mag_bid()
        print(f"t={t:5.1f}  [{bar}] {sf}/{phase_name:<6} {pct*100:5.1f}%"
              f"{pos_str}{ori_str}"
              f"  stance={worst_f}/{worst_d:.1f}mm"
              f"  mag={'OFF' if cur_mag_bid in swing_off else 'ON '}")

        foot_surfaces = _foot_surfaces(_swing_foot(), wall_feet)
        pen_data      = _read_surface_penetration(data, ik, foot_surfaces, wall_baselines)
        for foot in FEET:
            surf, contact, pen_m = pen_data[foot]
            role    = "SWING" if surf == 'swing' else surf.upper()
            c_sym   = "●" if contact else "○"
            pen_str = f"pen={pen_m*1000:5.2f}mm" if surf != 'swing' else "airborne    "
            flag    = " ◄" if foot == sf else ""
            print(f"         {foot} [{role:<5}]  contact={c_sym}  {pen_str}{flag}")

    # ── run loop ───────────────────────────────────────────────────────────────
    if args.headless:
        while data.time < args.duration:
            sim_step()
        return

    paused = [True]

    def key_callback(keycode):
        if keycode == 32:
            paused[0] = not paused[0]
            print(f"{'PAUSED' if paused[0] else 'RUNNING'}  t={data.time:.3f}")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.3]
        viewer.cam.distance  = 1.8
        viewer.cam.azimuth   = 90
        viewer.cam.elevation = -20
        print("PAUSED — press Space to start")

        frame_dt        = 1.0 / 60.0
        steps_per_frame = max(1, int(frame_dt / model.opt.timestep))

        while viewer.is_running():
            frame_start = time.perf_counter()
            if not paused[0]:
                for _ in range(steps_per_frame):
                    sim_step()
            draw_markers(viewer, model, data, ik, joint_vis,
                         args, settled, ik_target.position,
                         ik_target.face_axis, swing_off,
                         _swing_mag_bid(), _swing_foot())
            viewer.sync()
            elapsed = time.perf_counter() - frame_start
            if (remaining := frame_dt - elapsed) > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()