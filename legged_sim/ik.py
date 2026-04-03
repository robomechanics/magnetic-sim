"""
ik.py — Mink (MuJoCo-native) differential-IK wrapper for Sally.

No URDF or Pinocchio required — works directly on MjModel/MjData.

Dependencies:
    mink        (kevinzakka/mink, already installed)
    quadprog    (pip install quadprog)

Public surface:
    robot = MinkRobot(model, data)
    robot.solve_ik(foot_targets, dt)   # call once per sim step

Joint ordering in data.ctrl matches the <actuator> block in robot_original.xml:
    hip_pitch_BL, knee_BL, wrist_BL,
    hip_pitch_FL, knee_FL, wrist_FL,
    hip_pitch_BR, knee_BR, wrist_BR,
    hip_pitch_FR, knee_FR, wrist_FR,
    ee_BL, ee_FL, ee_BR, ee_FR
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import mujoco
import mink

from trajectory import FootTarget, GAIT_ORDER


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Actuator → joint name mapping, in data.ctrl index order.
# Order matches the <actuator> block in robot_original.xml exactly:
#   hip/knee/wrist grouped by leg (BL, FL, BR, FR), then all EE joints.
CTRL_JOINT_ORDER: Tuple[str, ...] = (
    "hip_pitch_BL", "knee_BL", "wrist_BL",
    "hip_pitch_FL", "knee_FL", "wrist_FL",
    "hip_pitch_BR", "knee_BR", "wrist_BR",
    "hip_pitch_FR", "knee_FR", "wrist_FR",
    "ee_BL", "ee_FL", "ee_BR", "ee_FR",
)

# Mink frame names for each foot's end-effector.
# frame_type="body" — must match body names in robot_original.xml.
EE_FRAME_NAME: Dict[str, str] = {
    'FL': "electromagnet_FL",
    'FR': "electromagnet_FR",
    'BL': "electromagnet_BL",
    'BR': "electromagnet_BR",
}

# Solver settings.
IK_DAMPING          = 1e-4
POSTURE_WEIGHT      = 0.01
STANCE_DAMPING_WEIGHT = 5.0   # penalises joint velocity on stance legs in QP

# Nominal joint angles for posture regulariser (radians). All zeros = rest pose.
POSTURE_NOMINAL: Dict[str, float] = {joint: 0.0 for joint in CTRL_JOINT_ORDER}


# ─────────────────────────────────────────────────────────────────────────────
# SE3 helper — mink uses its own SE3 type
# ─────────────────────────────────────────────────────────────────────────────

def _se3_from_pos_quat(pos: np.ndarray, q_wxyz: np.ndarray) -> mink.SE3:
    """
    Build a mink.SE3 from world-frame position and (w, x, y, z) quaternion.
    mink.SE3 stores a 4×4 homogeneous transform.
    """
    w, x, y, z = q_wxyz
    # Build rotation matrix from quaternion (w, x, y, z).
    R = np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y)  ],
        [2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x)  ],
        [2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = pos
    return mink.SE3.from_matrix(T)


# ─────────────────────────────────────────────────────────────────────────────
# MinkRobot
# ─────────────────────────────────────────────────────────────────────────────

class MinkRobot:
    """
    Thin wrapper around mink for Sally's whole-body IK.

    Parameters
    ----------
    mj_model : mujoco.MjModel
    mj_data  : mujoco.MjData
        solve_ik reads qpos from here and writes results to data.ctrl.
    """

    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        self._mj_model = mj_model
        self._mj_data  = mj_data

        # ── Mink configuration ────────────────────────────────────────────────
        # Initialised at neutral (zeros). _sync_from_mujoco() updates it each step.
        self._config = mink.Configuration(mj_model)

        # ── MuJoCo joint id → ctrl index mapping ─────────────────────────────
        self._ctrl_to_mj_jid: List[int] = []
        for joint_name in CTRL_JOINT_ORDER:
            jid = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if jid == -1:
                raise ValueError(
                    f"[ik] Joint '{joint_name}' not found in MuJoCo model. "
                    "Check CTRL_JOINT_ORDER against robot.xml actuator block."
                )
            self._ctrl_to_mj_jid.append(jid)

        # ── Actuator ctrl ranges — tighter than jnt_range, used for clamping ──
        # model.actuator_ctrlrange shape: (nu, 2), order matches CTRL_JOINT_ORDER.
        self._ctrl_ranges: List[Tuple[float, float]] = [
            (float(mj_model.actuator_ctrlrange[i, 0]),
             float(mj_model.actuator_ctrlrange[i, 1]))
            for i in range(len(CTRL_JOINT_ORDER))
        ]

        # ── Per-foot FrameTasks ───────────────────────────────────────────────
        # Created once; targets and weights updated each step.
        self._foot_tasks: Dict[str, mink.FrameTask] = {}
        for foot in GAIT_ORDER:
            self._foot_tasks[foot] = mink.FrameTask(
                frame_name       = EE_FRAME_NAME[foot],
                frame_type       = "body",
                position_cost    = 1.0,
                orientation_cost = 1.0,
                lm_damping       = IK_DAMPING,
            )

        # ── Per-leg damping tasks (zero joint-velocity for stance legs) ──────
        # DampingTask penalises joint velocities in the QP directly.
        # One task per leg; cost set to STANCE_DAMPING_WEIGHT when in stance,
        # 0 when that leg is swinging.
        self._nv = mj_model.nv
        self._stance_damping_tasks: Dict[str, mink.DampingTask] = {}
        for foot in GAIT_ORDER:
            self._stance_damping_tasks[foot] = mink.DampingTask(
                model = mj_model,
                cost  = np.zeros(mj_model.nv),   # inactive until first solve_ik call
            )

        # ── Posture task (null-space regulariser) ─────────────────────────────
        self._posture_task = mink.PostureTask(
            model      = mj_model,
            cost       = POSTURE_WEIGHT,
            lm_damping = IK_DAMPING,
        )
        # Target is the neutral (zero) configuration — matches baked rest pose.
        # PostureTask.set_target takes a full qpos vector.
        q_neutral = np.zeros(mj_model.nq)
        self._posture_task.set_target(q_neutral)

        # ── Configuration limit (joint range enforcement in QP) ──────────
        # Reads joint ranges directly from MjModel — no manual specification.
        self._config_limit = mink.ConfigurationLimit(mj_model)

        print(f"[ik] MinkRobot initialised — "
              f"{len(CTRL_JOINT_ORDER)} actuated DOFs, "
              f"{len(GAIT_ORDER)} foot tasks.")

    # ── Sync ──────────────────────────────────────────────────────────────────

    def _sync_from_mujoco(self) -> None:
        """Push current MjData.qpos into the mink Configuration."""
        self._config.update(self._mj_data.qpos.copy())

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve_ik(self, foot_targets: List[FootTarget], dt: float) -> None:
        """
        Run one differential-IK step and write results to data.ctrl.

        Parameters
        ----------
        foot_targets : List[FootTarget]
            One per foot from TrajectoryPlanner.step().
        dt : float
            Sim timestep in seconds.
        """
        self._sync_from_mujoco()

        target_map = {ft.foot: ft for ft in foot_targets}
        swing_foot = next(
            (ft.foot for ft in foot_targets if ft.weight_pos < 5.0), None
        )
        tasks: List = [self._posture_task]

        for foot in GAIT_ORDER:
            ft   = target_map[foot]
            task = self._foot_tasks[foot]

            task.set_target(_se3_from_pos_quat(ft.pos_world, ft.quat_world))
            task.position_cost    = ft.weight_pos
            task.orientation_cost = ft.weight_ori
            tasks.append(task)

            # Zero-velocity damping: active on all stance legs, off on swing.
            damp = self._stance_damping_tasks[foot]
            damp.cost = (np.zeros(self._nv) if foot == swing_foot
                         else np.full(self._nv, STANCE_DAMPING_WEIGHT))
            tasks.append(damp)

        # solve_ik returns joint velocities in the nv-dimensional tangent
        # space. nq (23) != nv (22) because the freejoint uses 7 qpos coords
        # but only 6 velocity coords. We only care about the 16 actuated
        # joints — read their velocities via jnt_dofadr and integrate directly
        # into qpos via jnt_qposadr, leaving the freejoint untouched.
        velocity = mink.solve_ik(
            self._config,
            tasks,
            dt,
            solver  = "quadprog",
            damping = IK_DAMPING,
            limits  = [self._config_limit],
        )

        for i, mj_jid in enumerate(self._ctrl_to_mj_jid):
            qidx  = self._mj_model.jnt_qposadr[mj_jid]   # position index in qpos
            vidx  = self._mj_model.jnt_dofadr[mj_jid]    # velocity index in nv
            q_new = self._mj_data.qpos[qidx] + velocity[vidx] * dt
            # Hard clamp to actuator ctrlrange — tighter than jnt_range.
            lo, hi = self._ctrl_ranges[i]
            q_new  = float(np.clip(q_new, lo, hi))
            self._mj_data.ctrl[i] = q_new

    # ── Accessors ─────────────────────────────────────────────────────────────

    def body_R_world(self) -> np.ndarray:
        """3×3 rotation world→body. Identity while base is fixed."""
        return np.eye(3)

    def body_pos_world(self) -> np.ndarray:
        """Body origin in world frame. Read from MuJoCo FK."""
        bid = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "main_frame"
        )
        return self._mj_data.xpos[bid].copy()

    def ee_pos_world(self, foot: str) -> np.ndarray:
        """Current EE position from MuJoCo FK. Seeds the planner at startup."""
        body_name = EE_FRAME_NAME[foot]
        bid = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if bid == -1:
            raise ValueError(
                f"[ik] Body '{body_name}' not found. "
                "Check EE_FRAME_NAME against robot.xml."
            )
        return self._mj_data.xpos[bid].copy()