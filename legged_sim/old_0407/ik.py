"""ik.py — Mink differential-IK wrapper for Sally."""

import numpy as np
import mujoco
import mink

from legged_sim.old_0407.trajectory import FootTarget, GAIT_ORDER


CTRL_JOINT_ORDER: tuple[str, ...] = (
    "hip_pitch_BL", "knee_BL", "wrist_BL",
    "hip_pitch_FL", "knee_FL", "wrist_FL",
    "hip_pitch_BR", "knee_BR", "wrist_BR",
    "hip_pitch_FR", "knee_FR", "wrist_FR",
    "ee_BL", "ee_FL", "ee_BR", "ee_FR",
)

EE_FRAME_NAME: dict[str, str] = {
    'FL': "electromagnet_FL",
    'FR': "electromagnet_FR",
    'BL': "electromagnet_BL",
    'BR': "electromagnet_BR",
}

IK_DAMPING            = 1e-4
POSTURE_WEIGHT        = 0.01
STANCE_DAMPING_WEIGHT = 5.0


def _se3_from_pos_quat(pos, q_wxyz):
    w, x, y, z = q_wxyz
    R = np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y)  ],
        [2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x)  ],
        [2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = pos
    return mink.SE3.from_matrix(T)


class MinkRobot:

    def __init__(self, mj_model, mj_data):
        self._mj_model = mj_model
        self._mj_data  = mj_data
        self._config   = mink.Configuration(mj_model)

        self._ctrl_to_mj_jid = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in CTRL_JOINT_ORDER
        ]
        self._ctrl_ranges = [
            (float(mj_model.actuator_ctrlrange[i, 0]), float(mj_model.actuator_ctrlrange[i, 1]))
            for i in range(len(CTRL_JOINT_ORDER))
        ]

        self._foot_tasks = {
            foot: mink.FrameTask(
                frame_name=EE_FRAME_NAME[foot], frame_type="body",
                position_cost=1.0, orientation_cost=1.0, lm_damping=IK_DAMPING,
            )
            for foot in GAIT_ORDER
        }

        self._nv = mj_model.nv
        self._stance_damping_tasks = {
            foot: mink.DampingTask(model=mj_model, cost=np.zeros(mj_model.nv))
            for foot in GAIT_ORDER
        }

        self._posture_task = mink.PostureTask(model=mj_model, cost=POSTURE_WEIGHT, lm_damping=IK_DAMPING)
        self._posture_task.set_target(np.zeros(mj_model.nq))
        self._config_limit = mink.ConfigurationLimit(mj_model)

        self._main_frame_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")
        self._ee_body_ids = {
            foot: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, EE_FRAME_NAME[foot])
            for foot in GAIT_ORDER
        }
        print(f"[ik] MinkRobot initialised — {len(CTRL_JOINT_ORDER)} DOFs, {len(GAIT_ORDER)} feet.")

    def _sync_from_mujoco(self):
        self._config.update(self._mj_data.qpos.copy())

    def solve_ik(self, foot_targets: list[FootTarget], dt: float,
                 frozen_joints: list[str] | None = None):
        self._sync_from_mujoco()
        target_map = {ft.foot: ft for ft in foot_targets}
        swing_foot = next((ft.foot for ft in foot_targets if not ft.magnet_on), None)
        tasks = [self._posture_task]

        for foot in GAIT_ORDER:
            ft   = target_map[foot]
            task = self._foot_tasks[foot]
            task.set_target(_se3_from_pos_quat(ft.pos_world, ft.quat_world))
            task.position_cost    = ft.weight_pos
            task.orientation_cost = ft.weight_ori
            tasks.append(task)

            cost = np.zeros(self._nv)
            if foot != swing_foot:
                for joint_name in CTRL_JOINT_ORDER:
                    if joint_name.endswith(f"_{foot}") or joint_name == f"ee_{foot}":
                        jid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                        cost[self._mj_model.jnt_dofadr[jid]] = STANCE_DAMPING_WEIGHT
            damp      = self._stance_damping_tasks[foot]
            damp.cost = cost
            tasks.append(damp)

        velocity = mink.solve_ik(
            self._config, tasks, dt,
            solver="quadprog", damping=IK_DAMPING, limits=[self._config_limit],
        )

        # Zero frozen joints before integrating — prevents the IK from allocating
        # velocity to joints that will be held externally, which would leave the
        # remaining joints under-compensated and cause error accumulation.
        if frozen_joints:
            for jname in frozen_joints:
                jid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                velocity[self._mj_model.jnt_dofadr[jid]] = 0.0

        for i, mj_jid in enumerate(self._ctrl_to_mj_jid):
            joint_name = CTRL_JOINT_ORDER[i]
            # Skip stance joints — they are written directly by lock_stance_joints().
            # Only integrate velocity for the swing foot's joints.
            if swing_foot and not (joint_name.endswith(f"_{swing_foot}") or
                                   joint_name == f"ee_{swing_foot}"):
                continue
            qidx  = self._mj_model.jnt_qposadr[mj_jid]
            vidx  = self._mj_model.jnt_dofadr[mj_jid]
            q_new = float(np.clip(
                self._mj_data.qpos[qidx] + velocity[vidx] * dt,
                *self._ctrl_ranges[i]
            ))
            self._mj_data.ctrl[i] = q_new

    def body_R_world(self):
        return self._mj_data.xmat[self._main_frame_bid].reshape(3, 3).copy()

    def body_pos_world(self):
        return self._mj_data.xpos[self._main_frame_bid].copy()

    def ee_pos_world(self, foot):
        return self._mj_data.xpos[self._ee_body_ids[foot]].copy()