"""
viz_joints.py — interactive joint limit visualizer for Sally legged sim.

For every hinge joint in the model, draws:
  • A grey capsule along the joint rotation axis
  • A cyan arc fan swept from lower limit to upper limit
  • A bright yellow spoke at the current joint angle (moves live)
  • A green spoke at zero / rest position

Press O to toggle IK orient mode: the FL EE will try to point its
contact face (-Y local axis) toward global +X.  While active:
  • Magenta arrow  = goal direction (+X world)
  • Orange  arrow  = current EE local -Y
  • Console prints ang_to_goal every 0.5 s

Controls:
  Space        — pause / unpause physics
  O            — toggle IK orient-to-+X mode
  Mouse drag   — rotate camera
  Scroll       — zoom
  Q / Esc      — quit

Usage:
    python viz_joints.py --leg FL
    python viz_joints.py --leg FL --arc-steps 24
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
SCENE_XML  = os.path.join(LEGGED_DIR, "mwc_mjcf", "scene.xml")
sys.path.insert(0, LEGGED_DIR)

from config import (
    TIMESTEP, KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG,
    bake_joint_angles,
)

FEET       = ('FL', 'FR', 'BL', 'BR')
SWING_FOOT = "FL"
IK_DAMPING = 1e-3
FACE_GOAL  = np.array([1., 0., 0.])   # global +X


# ── SE3 helper (same as sim.py) ─────────────────────────────────────────
def _se3_pos_rot(pos, face_axis):
    """Build SE3 target that aligns EE local -Y with face_axis."""
    neg_y = np.asarray(face_axis, float); neg_y /= np.linalg.norm(neg_y)
    y  = -neg_y
    up = np.array([0., 0., 1.]) if abs(y[2]) < 0.9 else np.array([0., 1., 0.])
    x  = np.cross(up, y); x /= np.linalg.norm(x)
    z  = np.cross(x, y)
    T  = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, pos
    return mink.SE3.from_matrix(T)


# ── geometry helpers ────────────────────────────────────────────────────

def _capsule_frame(axis):
    cz  = np.asarray(axis, float); cz /= np.linalg.norm(cz)
    ref = np.array([1., 0., 0.]) if abs(cz[0]) < 0.9 else np.array([0., 1., 0.])
    cx  = ref - np.dot(ref, cz) * cz; cx /= np.linalg.norm(cx)
    cy  = np.cross(cz, cx)
    return cx, cy, cz

def _add_capsule(scn, pos, axis, half_len, radius, rgba):
    if scn.ngeom >= scn.maxgeom: return False
    cx, cy, cz = _capsule_frame(axis)
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        [radius, half_len, 0], pos,
                        np.column_stack([cx, cy, cz]).flatten(), rgba)
    scn.ngeom += 1
    return True

def _add_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom: return False
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        [radius, 0, 0], pos, np.eye(3).flatten(), rgba)
    scn.ngeom += 1
    return True

def _add_arrow(scn, origin, direction, length, radius, rgba):
    """Capsule shaft + sphere tip."""
    d   = np.asarray(direction, float); d /= np.linalg.norm(d)
    mid = origin + d * length / 2
    _add_capsule(scn, mid, d, length / 2, radius, rgba)
    _add_sphere(scn, origin + d * length, radius * 2.5, rgba)


# ── arc drawing ─────────────────────────────────────────────────────────

def _draw_arc(scn, anchor, world_axis, ref_vec, angle_lo, angle_hi,
              spoke_len, spoke_rad, rgba, n_steps):
    for i in range(n_steps + 1):
        a    = angle_lo + (angle_hi - angle_lo) * i / n_steps
        c, s = np.cos(a), np.sin(a)
        k    = world_axis / np.linalg.norm(world_axis)
        v    = ref_vec*c + np.cross(k, ref_vec)*s + k*np.dot(k, ref_vec)*(1-c)
        v   /= np.linalg.norm(v)
        _add_capsule(scn, anchor + v*spoke_len/2, v, spoke_len/2, spoke_rad, rgba)
    for i in range(n_steps + 1):
        a    = angle_lo + (angle_hi - angle_lo) * i / n_steps
        c, s = np.cos(a), np.sin(a)
        k    = world_axis / np.linalg.norm(world_axis)
        v    = ref_vec*c + np.cross(k, ref_vec)*s + k*np.dot(k, ref_vec)*(1-c)
        _add_sphere(scn, anchor + v/np.linalg.norm(v)*spoke_len, spoke_rad*1.5, rgba)

def _draw_spoke(scn, anchor, world_axis, ref_vec, angle,
                spoke_len, spoke_rad, rgba):
    k    = world_axis / np.linalg.norm(world_axis)
    c, s = np.cos(angle), np.sin(angle)
    v    = ref_vec*c + np.cross(k, ref_vec)*s + k*np.dot(k, ref_vec)*(1-c)
    v   /= np.linalg.norm(v)
    _add_capsule(scn, anchor + v*spoke_len/2, v, spoke_len/2, spoke_rad, rgba)
    _add_sphere(scn, anchor + v*spoke_len, spoke_rad*2.5, rgba)


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg",       default=None, choices=list(FEET))
    parser.add_argument("--joint",     default=None)
    parser.add_argument("--arc-steps", type=int,   default=16)
    parser.add_argument("--spoke-len", type=float, default=0.07)
    args = parser.parse_args()

    # ── load model ──────────────────────────────────────────────────────
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

    # ── IK setup ────────────────────────────────────────────────────────
    ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                f"electromagnet_{SWING_FOOT}")

    config = mink.Configuration(model)

    ee_task = mink.FrameTask(
        frame_name=f"electromagnet_{SWING_FOOT}", frame_type="body",
        position_cost=0.0, orientation_cost=50.0,
        lm_damping=IK_DAMPING)

    posture_task = mink.PostureTask(model=model, cost=0.01,
                                    lm_damping=IK_DAMPING)
    posture_task.set_target(data.qpos.copy())

    config_limit = mink.ConfigurationLimit(model)

    ik_qpos = [None]   # warm-start across frames

    # ── collect joints for visualisation ────────────────────────────────
    joints = []
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if name == "root":
            continue
        if args.leg   and args.leg   not in name: continue
        if args.joint and args.joint not in name: continue
        lo = model.jnt_range[i, 0]
        hi = model.jnt_range[i, 1]
        joints.append((i, name, lo, hi))

    if not joints:
        print("[viz] No joints matched — check --leg / --joint args.")
        return

    print(f"[viz] Visualising {len(joints)} joint(s):")
    for jid, name, lo, hi in joints:
        print(f"  {name:<28}  range=[{np.degrees(lo):+.1f}°, {np.degrees(hi):+.1f}°]")

    # colours
    ARC_RGBA      = [0.0, 0.9, 0.9, 0.35]
    AXIS_RGBA     = [0.6, 0.6, 0.6, 0.8]
    ZERO_RGBA     = [0.2, 1.0, 0.2, 0.9]
    CURR_RGBA     = [1.0, 1.0, 0.0, 1.0]
    LIMIT_RGBA    = [1.0, 0.2, 0.2, 0.8]
    GOAL_RGBA     = [1.0, 0.0, 1.0, 1.0]    # magenta — goal +X
    CURR_ORI_RGBA = [1.0, 0.5, 0.0, 1.0]    # orange  — current EE -Y

    AXIS_HALF  = 0.05
    AXIS_RAD   = 0.004
    SPOKE_RAD  = 0.002
    SPOKE_LEN  = args.spoke_len
    ARROW_LEN  = 0.12
    ARROW_RAD  = 0.004

    # ── state ───────────────────────────────────────────────────────────
    paused      = [True]
    orient_mode = [False]
    last_print  = [0.0]

    def key_callback(keycode):
        if keycode == 32:                               # Space
            paused[0] = not paused[0]
            print(f"{'PAUSED' if paused[0] else 'RUNNING'}")
        elif keycode in (ord('O'), ord('o')):           # O
            orient_mode[0] = not orient_mode[0]
            ik_qpos[0] = None                           # reset warm-start
            if orient_mode[0]:
                print("[orient] ON  — IK driving EE local -Y → global +X")
            else:
                print("[orient] OFF — physics only")

    # ── draw ────────────────────────────────────────────────────────────
    def draw_markers(viewer):
        scn = viewer._user_scn
        scn.ngeom = 0
        mujoco.mj_forward(model, data)

        for jid, name, lo, hi in joints:
            bid      = model.jnt_bodyid[jid]
            anchor   = data.xanchor[jid].copy()
            body_rot = data.xmat[bid].reshape(3, 3)
            waxis    = body_rot @ model.jnt_axis[jid]
            waxis   /= np.linalg.norm(waxis)

            _add_capsule(scn, anchor, waxis, AXIS_HALF, AXIS_RAD, AXIS_RGBA)

            ref = np.array([1., 0., 0.]) if abs(waxis[0]) < 0.9 \
                  else np.array([0., 1., 0.])
            ref = ref - np.dot(ref, waxis) * waxis
            ref /= np.linalg.norm(ref)

            _draw_arc(scn, anchor, waxis, ref, lo, hi,
                      SPOKE_LEN, SPOKE_RAD, ARC_RGBA, args.arc_steps)
            _draw_spoke(scn, anchor, waxis, ref, lo,
                        SPOKE_LEN*1.05, SPOKE_RAD*2, LIMIT_RGBA)
            _draw_spoke(scn, anchor, waxis, ref, hi,
                        SPOKE_LEN*1.05, SPOKE_RAD*2, LIMIT_RGBA)
            _draw_spoke(scn, anchor, waxis, ref, 0.0,
                        SPOKE_LEN*0.8,  SPOKE_RAD*2, ZERO_RGBA)
            q_cur = data.qpos[model.jnt_qposadr[jid]]
            _draw_spoke(scn, anchor, waxis, ref, q_cur,
                        SPOKE_LEN, SPOKE_RAD*3, CURR_RGBA)

        # orient-mode overlays on EE
        if orient_mode[0]:
            ee_pos   = data.xpos[ee_bid].copy()
            ee_rot   = data.xmat[ee_bid].reshape(3, 3)
            ee_neg_y = -ee_rot[:, 1]

            _add_arrow(scn, ee_pos, FACE_GOAL, ARROW_LEN, ARROW_RAD, GOAL_RGBA)
            _add_arrow(scn, ee_pos, ee_neg_y,  ARROW_LEN, ARROW_RAD, CURR_ORI_RGBA)

            t_now = data.time
            if t_now - last_print[0] >= 0.5:
                last_print[0] = t_now
                ang = np.degrees(np.arccos(
                    np.clip(np.dot(ee_neg_y, FACE_GOAL), -1., 1.)))
                print(f"  t={t_now:6.2f}  ang_to_goal={ang:5.1f}°  "
                      f"ee_neg_y=[{ee_neg_y[0]:+.3f},"
                      f"{ee_neg_y[1]:+.3f},{ee_neg_y[2]:+.3f}]")

        viewer.sync()

    # ── IK step ─────────────────────────────────────────────────────────
    def run_ik():
        # Use current EE world position so position cost stays zero
        # and only orientation is driven
        ee_pos = data.xpos[ee_bid].copy()
        ee_task.set_target(_se3_pos_rot(ee_pos, FACE_GOAL))

        q = ik_qpos[0] if ik_qpos[0] is not None else data.qpos.copy()
        for _ in range(20):
            config.update(q)
            vel = mink.solve_ik(config, [ee_task, posture_task],
                                TIMESTEP, solver="quadprog",
                                damping=IK_DAMPING,
                                limits=[config_limit])
            q = config.integrate(vel, TIMESTEP)
        ik_qpos[0] = q

        # Apply directly to qpos — pure kinematics, no physics integration
        data.qpos[:] = q
        mujoco.mj_forward(model, data)

    # ── run loop ────────────────────────────────────────────────────────
    frame_dt        = 1.0 / 60
    steps_per_frame = max(1, int(frame_dt / model.opt.timestep))

    with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback) as viewer:

        viewer.cam.lookat[:] = [-2.0, 0.0, 0.3]
        viewer.cam.distance  = 1.8
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -20

        print("\n[viz] Viewer open. PAUSED — press Space to simulate.")
        print("  Press O  to toggle IK orient-to-+X mode")
        print("  Magenta arrow = goal (+X world)")
        print("  Orange  arrow = current EE local -Y")
        print("  Yellow spoke  = current joint angle")
        print("  Green  spoke  = zero angle")
        print("  Red    spokes = joint limits\n")

        while viewer.is_running():
            t0 = time.perf_counter()

            if orient_mode[0]:
                # kinematic IK: moves joints directly, skips physics
                run_ik()
            elif not paused[0]:
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)

            draw_markers(viewer)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0., frame_dt - elapsed))


if __name__ == "__main__":
    main()