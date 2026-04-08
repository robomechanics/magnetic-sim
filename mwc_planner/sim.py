"""
MWC sim: robot on ground, Mink IK moves FL foot up/down.
3 stance feet held via IK, body free under gravity.

Usage:
    python sim.py              # GUI viewer
    python sim.py --headless   # headless with telemetry
"""
import argparse
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import mink

# ── paths ──────────────────────────────────────────────────────────────
LEGGED_DIR = os.path.join(os.path.dirname(__file__), "..", "legged_sim")
SCENE_XML = os.path.join(LEGGED_DIR, "mwc_mjcf", "scene.xml")

sys.path.insert(0, LEGGED_DIR)
from config import (
    MU_0, MAGNET_VOLUME, MAGNET_BODY_NAMES, PARAMS,
    TIMESTEP, JOINT_DAMPING, JOINT_ARMATURE,
    KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG,
    bake_joint_angles,
)

# ── config ─────────────────────────────────────────────────────────────
FEET = ('FL', 'FR', 'BL', 'BR')
SWING_FOOT = "FL"
BOUNCE_AMPLITUDE = 0.05  # 5 cm (smaller since on ground)
BOUNCE_PERIOD = 3.0      # seconds

IK_DAMPING = 1e-3
SETTLE_TIME = 2.0  # let robot settle before swinging

# PID gains
PID_KP = 500.0
PID_KI = 200.0
PID_KD = 30.0
PID_I_CLAMP = 100.0  # anti-windup


# ── magnetic force ────────────────────────────────────────────────────
def mag_force(dist, Br):
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist) ** 4)


def apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, off_mids=None):
    """Apply magnetic forces. off_mids = set of body IDs with magnets disabled."""
    if off_mids is None:
        off_mids = set()
    _fromto = np.zeros(6)
    for mid in magnet_ids:
        if mid in off_mids:
            continue
        fvec = np.zeros(3)
        for gid in sphere_gids[mid]:
            best_dist, best_fromto = np.inf, None
            for pid in plate_ids:
                d = mujoco.mj_geomDistance(model, data, gid, pid, 50.0, _fromto)
                if d < best_dist:
                    best_dist, best_fromto = d, _fromto.copy()
            if best_dist <= 0 or best_dist > PARAMS['max_magnetic_distance']:
                continue
            n = best_fromto[3:6] - best_fromto[0:3]
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            fvec += mag_force(best_dist, PARAMS['Br']) * (n / norm)
        total_mag = np.linalg.norm(fvec)
        if total_mag > PARAMS['max_force_per_wheel']:
            fvec *= PARAMS['max_force_per_wheel'] / total_mag
        data.xfrc_applied[mid, :3] += fvec


# ── model setup ────────────────────────────────────────────────────────
def setup_model():
    bake_joint_angles(os.path.join(LEGGED_DIR, "mwc_mjcf", "robot.xml"))

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP
    mujoco.mj_resetData(model, data)

    for leg in FEET:
        for jname, bake_dict in [
            (f'knee_{leg}', KNEE_BAKE_DEG),
            (f'wrist_{leg}', WRIST_BAKE_DEG),
            (f'ee_{leg}', EE_BAKE_DEG),
        ]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                data.qpos[model.jnt_qposadr[jid]] = np.radians(bake_dict[leg])

    mujoco.mj_forward(model, data)

    # Make all robot geoms semi-transparent
    for i in range(model.ngeom):
        model.geom_rgba[i, 3] = 0.5

    # Magnetic surfaces: floor + wall
    plate_ids = set()
    for name in ("floor", "wall"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        plate_ids.add(gid)
        model.geom_friction[gid] = PARAMS['ground_friction']

    model.opt.o_solref = PARAMS['solref']
    model.opt.o_solimp = PARAMS['solimp']
    model.opt.noslip_iterations = PARAMS['noslip_iterations']
    model.opt.noslip_tolerance = PARAMS['noslip_tolerance']
    model.opt.o_margin = PARAMS['margin']

    model.dof_damping[:] = 2.0
    model.dof_armature[:] = 0.01

    # Magnet IDs
    magnet_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in MAGNET_BODY_NAMES
    ]
    sphere_gids = {
        mid: [
            gid for gid in range(model.ngeom)
            if model.geom_bodyid[gid] == mid
            and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
        ]
        for mid in magnet_ids
    }

    return model, data, plate_ids, magnet_ids, sphere_gids


# ── IK solver ─────────────────────────────────────────────────────────
def _se3_from_pos(pos):
    T = np.eye(4)
    T[:3, 3] = pos
    return mink.SE3.from_matrix(T)


class PIDController:
    """Per-joint PID controller. ctrl = torque."""
    def __init__(self, model):
        self.nu = model.nu
        self.ctrl_jids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.integral = np.zeros(model.nu)
        self.prev_err = np.zeros(model.nu)

    def compute(self, model, data, targets, dt):
        """Compute torques for motor actuators given target joint positions (radians)."""
        torques = np.zeros(self.nu)
        for i, jid in enumerate(self.ctrl_jids):
            qidx = model.jnt_qposadr[jid]
            vidx = model.jnt_dofadr[jid]
            err = targets[i] - data.qpos[qidx]
            self.integral[i] = np.clip(
                self.integral[i] + err * dt, -PID_I_CLAMP, PID_I_CLAMP)
            derr = (err - self.prev_err[i]) / dt if dt > 0 else 0.0
            self.prev_err[i] = err
            torques[i] = PID_KP * err + PID_KI * self.integral[i] + PID_KD * derr
        return torques


class IKSolver:
    def __init__(self, model):
        self.model = model
        self.config = mink.Configuration(model)

        # Task for each foot
        self.foot_tasks = {}
        self.ee_bids = {}
        for foot in FEET:
            frame = f"electromagnet_{foot}"
            self.ee_bids[foot] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, frame)
            self.foot_tasks[foot] = mink.FrameTask(
                frame_name=frame, frame_type="body",
                position_cost=10.0, orientation_cost=0.0,
                lm_damping=IK_DAMPING,
            )

        # Task for body — hold position + orientation
        self.body_task = mink.FrameTask(
            frame_name="main_frame", frame_type="body",
            position_cost=50.0, orientation_cost=50.0,
            lm_damping=IK_DAMPING,
        )

        self.posture_task = mink.PostureTask(
            model=model, cost=0.01, lm_damping=IK_DAMPING,
        )
        self.config_limit = mink.ConfigurationLimit(model)

        # Freeze passive joints only (ee2, em_z) — body held by body_task
        frozen_dofs = []
        self._passive_jnames = []
        for i in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jname and (jname.startswith('ee2_') or jname.startswith('em_z_')):
                frozen_dofs.append(model.jnt_dofadr[i])
                self._passive_jnames.append(jname)
        self.freeze_passive = mink.DofFreezingTask(
            model=model, dof_indices=frozen_dofs,
        ) if frozen_dofs else None
        print(f"[ik] Frozen passive DOFs: {self._passive_jnames}")

        self.ctrl_jids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.stance_targets = {}
        self.body_target = None

    def ee_pos(self, data, foot):
        return data.xpos[self.ee_bids[foot]].copy()

    def record_stance(self, data):
        """Snapshot current foot positions, body pose, and IK state."""
        self._ik_qpos = data.qpos.copy()
        for foot in FEET:
            self.stance_targets[foot] = self.ee_pos(data, foot)
        self.posture_task.set_target(data.qpos.copy())

        # Record body pose
        body_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")
        body_pos = data.xpos[body_bid].copy()
        body_mat = data.xmat[body_bid].reshape(3, 3).copy()
        T = np.eye(4)
        T[:3, :3] = body_mat
        T[:3, 3] = body_pos
        self.body_target = mink.SE3.from_matrix(T)
        self.body_task.set_target(self.body_target)

        print("[ik] Stance targets recorded:")
        for f, p in self.stance_targets.items():
            tag = " (swing)" if f == SWING_FOOT else ""
            print(f"  {f}: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]{tag}")
        print(f"  body: [{body_pos[0]:.4f}, {body_pos[1]:.4f}, {body_pos[2]:.4f}]")

    def resync(self, phys_data):
        """Resync IK state from physics — call at start of each swing cycle."""
        self._ik_qpos = phys_data.qpos.copy()

    def solve(self, swing_target, phys_data, dt, n_iter=10):
        """Solve IK from physics state."""
        ik_qpos = phys_data.qpos.copy()

        # Swing foot: track target
        self.foot_tasks[SWING_FOOT].set_target(_se3_from_pos(swing_target))
        self.foot_tasks[SWING_FOOT].position_cost = 10.0

        # Stance feet: hold position, high cost
        for foot in FEET:
            if foot != SWING_FOOT:
                self.foot_tasks[foot].set_target(
                    _se3_from_pos(self.stance_targets[foot]))
                self.foot_tasks[foot].position_cost = 50.0

        tasks = [self.body_task]
        tasks += [self.foot_tasks[f] for f in FEET]
        tasks += [self.posture_task]
        if self.freeze_passive:
            tasks.append(self.freeze_passive)

        for _ in range(n_iter):
            self.config.update(ik_qpos)
            velocity = mink.solve_ik(
                self.config, tasks, dt,
                solver="quadprog", damping=IK_DAMPING,
                limits=[self.config_limit],
            )
            ik_qpos = self.config.integrate(velocity, dt)

        # Persist kinematic state for next call
        self._ik_qpos = ik_qpos

        # Extract joint targets only (skip freejoint)
        ctrl_targets = np.zeros(self.model.nu)
        for i, jid in enumerate(self.ctrl_jids):
            ctrl_targets[i] = ik_qpos[self.model.jnt_qposadr[jid]]
        return ctrl_targets


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--magnets", action="store_true", help="enable magnets")
    parser.add_argument("--no-ik", action="store_true", help="hold default pose, no IK swing")
    args = parser.parse_args()

    model, data, plate_ids, magnet_ids, sphere_gids = setup_model()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")
    swing_mag_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"electromagnet_{SWING_FOOT}")

    ik = IKSolver(model)
    pid = PIDController(model)

    last_print = 0.0
    last_cycle = -1
    target_pos = np.zeros(3)
    # Initial ctrl targets: only actuated joints
    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]
    settled = False
    ee_home = np.zeros(3)
    ik_step_counter = [0]
    IK_EVERY_N = 10  # IK at 200Hz, physics at 2000Hz

    print(f"mode={'headless' if args.headless else 'gui'}  "
          f"swing={SWING_FOOT}  bounce={BOUNCE_AMPLITUDE*100:.0f}cm  "
          f"period={BOUNCE_PERIOD}s  magnets={'on' if args.magnets else 'off'}  "
          f"settle={SETTLE_TIME}s")

    def sim_step():
        nonlocal last_print, last_cycle, target_pos, ctrl_targets, settled, ee_home
        t = data.time

        data.xfrc_applied[:] = 0
        if args.magnets:
            # During settle: all magnets on. During swing: swing foot magnet off.
            off = {swing_mag_bid} if settled else set()
            apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, off)

        # Phase 1: settle — hold default pose with PID
        if t < SETTLE_TIME:
            data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
            mujoco.mj_step(model, data)
            return

        # Snapshot stance targets once settled
        if not settled:
            settled = True
            if not args.no_ik:
                ik.record_stance(data)
                ee_home = ik.ee_pos(data, SWING_FOOT).copy()
                target_pos = ee_home.copy()

        if args.no_ik:
            pass  # ctrl_targets stays at default
        else:
            elapsed = t - SETTLE_TIME
            cycle = int(elapsed // BOUNCE_PERIOD)
            phase = (elapsed % BOUNCE_PERIOD) / BOUNCE_PERIOD
            dz = BOUNCE_AMPLITUDE * (1 - np.cos(2 * np.pi * phase)) / 2
            target_pos = ee_home + np.array([0.0, 0.0, dz])

            # IK at 200Hz, PID at full physics rate
            ik_step_counter[0] += 1
            if ik_step_counter[0] >= IK_EVERY_N:
                ik_step_counter[0] = 0
                ctrl_targets = ik.solve(target_pos, data, IK_EVERY_N * TIMESTEP)

        # PID torque control
        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

        # Telemetry
        if t - last_print >= 0.5:
            last_print = t
            body_z = data.xpos[body_id, 2]

            if args.no_ik:
                print(f"t={t:5.1f}  body_z={body_z:.4f}")
            else:
                actual_sw = ik.ee_pos(data, SWING_FOOT)
                diff_sw = (actual_sw - target_pos) * 1000
                err_sw = np.linalg.norm(diff_sw)

                max_stance_err = 0.0
                worst_foot = ""
                for foot in FEET:
                    if foot == SWING_FOOT:
                        continue
                    actual = ik.ee_pos(data, foot)
                    drift = np.linalg.norm(actual - ik.stance_targets[foot]) * 1000
                    if drift > max_stance_err:
                        max_stance_err = drift
                        worst_foot = foot

                print(f"t={t:5.1f}  body_z={body_z:.4f}  "
                      f"swing: dx={diff_sw[0]:+5.1f} dy={diff_sw[1]:+5.1f} "
                      f"dz={diff_sw[2]:+5.1f}mm |{err_sw:.1f}|  "
                      f"stance: {worst_foot}={max_stance_err:.1f}mm")

    # Pre-compute joint info for axis visualization
    joint_vis = []  # (joint_id, body_id, local_axis, color)
    JOINT_COLORS = {
        'hip_pitch': [1.0, 1.0, 0.0, 0.9],   # yellow
        'knee':      [0.0, 1.0, 1.0, 0.9],   # cyan
        'wrist':     [1.0, 0.0, 1.0, 0.9],   # magenta
        'ee2':       [0.2, 1.0, 0.5, 0.9],   # green (passive)
        'em_z':      [0.5, 0.5, 1.0, 0.9],   # light blue (passive Z)
        'ee':        [1.0, 0.5, 0.0, 0.9],   # orange
    }
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not jname or jname == "root":
            continue
        color = [0.5, 0.5, 0.5, 0.8]
        for prefix, c in JOINT_COLORS.items():
            if jname.startswith(prefix):
                color = c
                break
        joint_vis.append((i, model.jnt_bodyid[i], model.jnt_axis[i].copy(), color))

    def draw_target_marker(viewer):
        scn = viewer._user_scn
        scn.ngeom = 0

        # Joint axis cylinders (always drawn)
        AXIS_VIS_LEN = 0.09
        AXIS_VIS_RAD = 0.009
        for jid, bid, local_axis, color in joint_vis:
            if scn.ngeom >= scn.maxgeom:
                break
            # Joint position in world
            jpos = data.xanchor[jid]
            # World-frame axis
            body_rot = data.xmat[bid].reshape(3, 3)
            waxis = body_rot @ local_axis
            waxis = waxis / np.linalg.norm(waxis)

            # Capsule from jpos - axis*half to jpos + axis*half
            center = jpos
            # Build rotation matrix with Z along waxis
            z = waxis
            x = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
            x = x - np.dot(x, z) * z
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            rot = np.column_stack([x, y, z])

            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                [AXIS_VIS_RAD, AXIS_VIS_LEN / 2, 0],
                center, rot.flatten(), color)
            scn.ngeom += 1

        if args.no_ik or not settled:
            return

        p = target_pos
        AXIS_LEN = 0.04
        AXIS_RAD = 0.002

        # Swing target (green sphere)
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0],
                p, np.eye(3).flatten(), [0.2, 1.0, 0.2, 0.5])
            scn.ngeom += 1

        axes = [
            (np.array([AXIS_LEN/2, 0, 0]),
             np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=float),
             [1, 0.2, 0.2, 0.8]),
            (np.array([0, AXIS_LEN/2, 0]),
             np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=float),
             [0.2, 1, 0.2, 0.8]),
            (np.array([0, 0, AXIS_LEN/2]),
             np.eye(3),
             [0.2, 0.2, 1, 0.8]),
        ]
        for offset, rot, rgba in axes:
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_CAPSULE, [AXIS_RAD, AXIS_LEN/2, 0],
                    p + offset, rot.flatten(), rgba)
                scn.ngeom += 1

        # Stance targets (yellow spheres)
        for foot in FEET:
            if foot == SWING_FOOT:
                continue
            if foot in ik.stance_targets and scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.005, 0, 0],
                    ik.stance_targets[foot], np.eye(3).flatten(),
                    [1.0, 1.0, 0.2, 0.5])
                scn.ngeom += 1

    if args.headless:
        while data.time < args.duration:
            sim_step()
    else:
        paused = [True]  # start paused

        def key_callback(keycode):
            if keycode == 32:  # space
                paused[0] = not paused[0]
                print(f"{'PAUSED' if paused[0] else 'RUNNING'}  t={data.time:.3f}")

        with mujoco.viewer.launch_passive(
                model, data, key_callback=key_callback) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.3]
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20

            print("PAUSED — press Space to start")

            TARGET_FPS = 60
            frame_dt = 1.0 / TARGET_FPS
            steps_per_frame = max(1, int(frame_dt / model.opt.timestep))

            while viewer.is_running():
                frame_start = time.perf_counter()

                if not paused[0]:
                    for _ in range(steps_per_frame):
                        sim_step()

                draw_target_marker(viewer)
                viewer.sync()

                # Spin-wait for precise frame timing
                target = frame_start + frame_dt
                while time.perf_counter() < target:
                    pass


if __name__ == "__main__":
    main()
