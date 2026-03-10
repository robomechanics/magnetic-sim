"""
demo.py - Sally wall-climbing robot demo

Spawns Sally at the bottom-left corner of the magnetic wall plate with a
cinematic camera view. Magnetic forces, pivot locking, and velocity arrows
are all active. Press ENTER to start, SPACE to pause/resume.

Usage:
    python demo.py [--duration 15.0] [--mode hold|drive_sideways|drive_up]
"""

import mujoco
import mujoco.viewer
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import copy
import io

# ── Wall geometry (from robot_sally_patched.xml) ────────────────────────────
# magnetic_wall body: pos="0.25 -5.0 0.35"
# wall_geom size="0.02 10.0 2.0"  (half-extents in X, Y, Z)
#   Wall surface (inner face) at X = 0.25 - 0.02 = 0.23
#   Wall spans Y: -5.0 - 10.0 = -15.0  →  -5.0 + 10.0 = +5.0
#   Wall spans Z:  0.35 - 2.0 = -1.65  →   0.35 + 2.0 = +2.35
WALL_Y_MIN = -15.0
WALL_Y_MAX =   5.0
WALL_Z_MIN =  -1.65
WALL_Z_MAX =   2.35

# ── Robot spawn: bottom-left corner of the plate ────────────────────────────
# viewer.py sets data.qpos[0:3] = [0.035, 0.0, 0.35] (world-frame freejoint pos)
# "bottom" = low Z, "left" = near +Y edge (when facing the wall from +X side)
# Inset 30 cm from each edge so the robot is fully on the plate.
ROBOT_X =  0.035                    # same X offset as viewer.py (35 mm from wall)
ROBOT_Y =  WALL_Y_MAX - 0.30        # near +Y edge:  5.0 - 0.30 = +4.70
ROBOT_Z =  WALL_Z_MIN + 0.30        # near bottom:  -1.65 + 0.30 = -1.35

# Quaternion (w x y z) — identical to viewer.py data.qpos[3:7]
ROBOT_QUAT = [-0.707, 0.0, 0.707, 0.0]

# ── Physics / magnetic constants (reasonable demo defaults) ──────────────────
Br               = 1.2          # remanence [T]
MAX_MAG_DIST     = 0.025        # 25 mm attraction cut-off
MAX_FORCE_PW     = 80.0         # max force per sampling sphere [N]
MU_0             = 4 * np.pi * 1e-7
MAGNET_VOLUME    = (0.02 ** 3)  # 20 mm cube

ARROW_RADIUS = 0.003

key_state = {"paused": True}


# ── Helpers ──────────────────────────────────────────────────────────────────

def calculate_magnetic_force(dist, Br, volume, mu_0):
    """Dipole–dipole attractive force [N]. Returns 0 for dist ≤ 0."""
    if dist <= 0:
        return 0.0
    M  = Br / mu_0
    m  = M * volume
    F  = (3 * mu_0 * m ** 2) / (2 * np.pi * dist ** 4)
    return float(F)


def add_arrow(scene, start, end, color):
    if scene.ngeom >= scene.maxgeom:
        return
    length = np.linalg.norm(end - start)
    if length < 1e-9:
        return
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
    if keycode == 32:    # SPACE  → toggle pause
        key_state["paused"] = not key_state["paused"]
    elif keycode == 257:  # ENTER  → unpause
        key_state["paused"] = False


# ── XML patching ─────────────────────────────────────────────────────────────

def _patch_worldbody(root: ET.Element) -> bool:
    """
    Find <worldbody> inside *root* and patch frame / magnetic_wall positions.
    Returns True if a worldbody was found and patched.
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        return False

    for body in worldbody.iter("body"):
        name = body.get("name", "")
        if name == "frame":
            body.set("pos", f"{ROBOT_X:.6f} {ROBOT_Y:.6f} {ROBOT_Z:.6f}")
            body.set("quat", "0 -0.7071 0 0.7071")
        # magnetic_wall stays exactly where it is — robot moves to it

    return True


def patch_xml_for_demo(xml_path: str):
    """
    Patch the robot spawn and wall position for the demo.

    Copies the entire XML directory into a temp folder, patches either
    scene.xml directly or its <include> file, then returns
      (patched_scene_path, tmp_dir_path)
    so that MuJoCo can load with from_xml_path() and resolve all relative
    asset / include paths correctly.

    The caller should delete tmp_dir_path when done (or just let it leak for
    a short-lived demo process).
    """
    import os, shutil, tempfile

    xml_path    = os.path.abspath(xml_path)
    xml_dir     = os.path.dirname(xml_path)
    scene_name  = os.path.basename(xml_path)

    # Copy the whole XML folder into a temp dir so assets resolve correctly
    tmp_dir     = tempfile.mkdtemp(prefix="sally_demo_")
    tmp_xml_dir = os.path.join(tmp_dir, "XML")
    shutil.copytree(xml_dir, tmp_xml_dir)

    tmp_scene   = os.path.join(tmp_xml_dir, scene_name)

    scene_tree  = ET.parse(tmp_scene)
    scene_root  = scene_tree.getroot()

    # Case 1: worldbody lives directly in scene.xml
    if _patch_worldbody(scene_root):
        scene_tree.write(tmp_scene, encoding="unicode", xml_declaration=False)
        print(f"  → Patched scene.xml directly")
        return tmp_scene, tmp_dir

    # Case 2: worldbody is in an <include> file
    include_el = scene_root.find("include")
    if include_el is None:
        raise RuntimeError(
            "No <worldbody> in scene XML and no <include> tag found."
        )
    include_file = include_el.get("file", "")
    if not include_file:
        raise RuntimeError("<include> tag has no 'file' attribute.")

    inc_path = os.path.join(tmp_xml_dir, include_file)
    if not os.path.exists(inc_path):
        raise RuntimeError(
            f"Included file not found in temp copy: {inc_path}\n"
            f"Ensure '{include_file}' exists alongside scene.xml."
        )

    print(f"  → Patching included file: {include_file}")
    inc_tree = ET.parse(inc_path)
    inc_root = inc_tree.getroot()

    if not _patch_worldbody(inc_root):
        raise RuntimeError(
            f"No <worldbody> found in included file '{include_file}'."
        )

    inc_tree.write(inc_path, encoding="unicode", xml_declaration=False)
    return tmp_scene, tmp_dir


# ── Main simulation ───────────────────────────────────────────────────────────

def run_demo(sim_duration: float = 15.0, mode: str = "hold"):
    """
    Modes
    -----
    hold           – all wheel velocities = 0, just cling to the wall
    drive_sideways – drive in +Y direction
    drive_up       – drive in +Z direction
    """

    # ── Mode config ───────────────────────────────────────────────────────
    MODE_CFG = {
        "hold": {
            "pivot_angle":        np.deg2rad(0),
            "actuator_mode":      "hold",
            "actuator_target_ms": 0.0,
            "settle_time":        1.5,
        },
        "drive_sideways": {
            "pivot_angle":        np.deg2rad(90),
            "actuator_mode":      "velocity",
            "actuator_target_ms": 0.10,   # m/s
            "settle_time":        2.0,
        },
        "drive_up": {
            "pivot_angle":        np.deg2rad(0),
            "actuator_mode":      "velocity",
            "actuator_target_ms": 0.08,
            "settle_time":        2.0,
        },
    }
    if mode not in MODE_CFG:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODE_CFG)}")

    cfg = MODE_CFG[mode]
    WHEEL_RADIUS = 0.025   # metres (matches cylinder size in XML)

    # rad/s from m/s
    target_rads = cfg["actuator_target_ms"] / WHEEL_RADIUS if WHEEL_RADIUS > 0 else 0.0

    # ── Build model from patched XML ──────────────────────────────────────
    import shutil
    print("Patching XML …")
    tmp_scene_path, tmp_dir = patch_xml_for_demo("XML/scene.xml")
    model = mujoco.MjModel.from_xml_path(tmp_scene_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)  # clean up temp files

    # Raise resolution
    model.opt.timestep         = 1.0 / 1000.0
    model.opt.enableflags     |= 1 << 0     # energy
    model.opt.iterations       = 100
    model.opt.tolerance        = 1e-8
    model.vis.map.znear        = 0.005
    model.vis.map.zfar         = 100.0

    # ── Data & initial state ──────────────────────────────────────────────
    data = mujoco.MjData(model)

    # Place robot at patched position
    data.qpos[0] = ROBOT_X
    data.qpos[1] = ROBOT_Y
    data.qpos[2] = ROBOT_Z
    data.qpos[3] = ROBOT_QUAT[0]
    data.qpos[4] = ROBOT_QUAT[1]
    data.qpos[5] = ROBOT_QUAT[2]
    data.qpos[6] = ROBOT_QUAT[3]

    # ── Geometry IDs ──────────────────────────────────────────────────────
    wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_geom")

    # Sampling sphere geom IDs (24 per wheel)
    wheel_gids = []
    for prefix in ["BR", "FR", "BL", "FL"]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_wheel_geom")
        if body_id == -1:
            continue
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == body_id:
                if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    wheel_gids.append(gid)

    # Wheel actuator IDs
    wheel_act_ids = []
    for act_name in ["BR_wheel_motor", "FR_wheel_motor", "BL_wheel_motor", "FL_wheel_motor"]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if aid != -1:
            wheel_act_ids.append(aid)

    # Pivot joint IDs
    pivot_names = ["BR_pivot", "FR_pivot", "BL_pivot", "FL_pivot"]
    pivot_ids = []
    for jn in pivot_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid != -1:
            pivot_ids.append(jid)

    # ── Lock pivots to desired angle & set spring reference ───────────────
    mujoco.mj_step(model, data)   # settle geometry first

    for jid in pivot_ids:
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr]         = cfg["pivot_angle"]
        model.jnt_stiffness[jid] = 1000.0
        model.qpos_spring[qadr] = cfg["pivot_angle"]

    fromto = np.zeros(6)

    # ── Camera: positioned to show the bottom-left corner clearly ─────────
    # We'll use a tracking camera offset to the robot's lower-right,
    # looking up and inward at ~45 degrees.
    print(f"\n{'='*60}")
    print(f"  Sally Demo  |  mode={mode}  |  duration={sim_duration}s")
    print(f"  Robot spawn : X={ROBOT_X:.3f}  Y={ROBOT_Y:.3f}  Z={ROBOT_Z:.3f}")
    print(f"  Wall corner : Y≈{WALL_Y_MAX:.1f}  Z≈{WALL_Z_MIN:.2f}")
    print(f"{'='*60}")
    print(f"\n  Press  ENTER  to start")
    print(f"  Press  SPACE  to pause / resume\n")

    settle_time    = cfg["settle_time"]
    trajectory_log = []

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # ── Camera setup ──────────────────────────────────────────────────
        # Track the robot frame, pull back so the whole wall corner is in view
        viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "frame")
        viewer.cam.distance    = 2.5    # metres from robot
        # Azimuth: 135° puts camera behind-and-to-the-right so both wall
        # edges (Y edge and Z bottom edge) are visible simultaneously.
        viewer.cam.azimuth     = 135.0
        viewer.cam.elevation   = -25.0  # looking slightly downward

        while viewer.is_running() and data.time < sim_duration:
            if not key_state["paused"]:
                viewer.user_scn.ngeom = 0
                data.xfrc_applied[:]  = 0.0

                # ── Magnetic forces ───────────────────────────────────────
                for gid in wheel_gids:
                    dist = mujoco.mj_geomDistance(
                        model, data, gid, wall_id, 50.0, fromto
                    )
                    if 0 <= dist <= MAX_MAG_DIST:
                        fmag = calculate_magnetic_force(
                            dist, Br, MAGNET_VOLUME, MU_0
                        )
                        fmag = np.clip(fmag, 0.0, MAX_FORCE_PW)

                        n    = fromto[3:6] - fromto[0:3]
                        norm = np.linalg.norm(n)
                        if norm > 1e-10:
                            n_hat = n / norm
                            bid   = model.geom_bodyid[gid]
                            data.xfrc_applied[bid, :3] += fmag * n_hat

                            # Blue arrow scaled to force magnitude
                            arrow_len = 0.002 * fmag / 10.0
                            add_arrow(
                                viewer.user_scn,
                                fromto[0:3],
                                fromto[0:3] + arrow_len * n_hat,
                                (0.2, 0.5, 1.0, 0.9),
                            )

                # ── Wheel control ─────────────────────────────────────────
                if cfg["actuator_mode"] == "velocity":
                    for aid in wheel_act_ids:
                        data.ctrl[aid] = target_rads
                else:
                    for aid in wheel_act_ids:
                        data.ctrl[aid] = 0.0

                # ── Forward-direction arrow (red) ─────────────────────────
                frame_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "frame")
                robot_pos = data.xpos[frame_id].copy()
                robot_quat = data.xquat[frame_id].copy()

                from scipy.spatial.transform import Rotation as R
                rot         = R.from_quat([robot_quat[1], robot_quat[2],
                                           robot_quat[3], robot_quat[0]])
                forward_vec = rot.apply([0, 1, 0])
                add_arrow(
                    viewer.user_scn,
                    robot_pos,
                    robot_pos + 0.25 * forward_vec,
                    (1.0, 0.1, 0.1, 1.0),
                )

                # ── Step ──────────────────────────────────────────────────
                mujoco.mj_step(model, data)

                # Trajectory logging (post-settle)
                if data.time >= settle_time:
                    trajectory_log.append((data.time, data.qpos[:3].copy()))

                # Console velocity readout every 0.5 s
                if int(data.time * 1000) % 500 == 0:
                    vel = data.qvel[0:3]
                    speed = np.linalg.norm(vel)
                    print(
                        f"  t={data.time:5.2f}s | "
                        f"X={vel[0]:+.4f}  Y={vel[1]:+.4f}  Z={vel[2]:+.4f} m/s | "
                        f"speed={speed:.4f} m/s"
                    )

            viewer.sync()

    # ── Post-run summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")

    if len(trajectory_log) >= 2:
        t0, p0 = trajectory_log[0]
        t1, p1 = trajectory_log[-1]
        dt = t1 - t0
        if dt > 1e-6:
            dp          = p1 - p0
            avg_vel     = dp / dt
            avg_speed   = np.linalg.norm(avg_vel)

            print(f"  Mode          : {mode}")
            print(f"  Settle time   : {settle_time:.2f}s  |  Active: {dt:.2f}s")
            print(f"  Avg velocity  : X={avg_vel[0]:+.4f}  Y={avg_vel[1]:+.4f}  Z={avg_vel[2]:+.4f} m/s")
            print(f"  Avg speed     : {avg_speed:.4f} m/s")

            if mode == "hold":
                slip = np.linalg.norm(dp)
                print(f"  Total slip    : {slip*1000:.2f} mm")
                print(f"  Result        : {'✓ Held position' if slip < 0.005 else '✗ Slipped'}")
            elif mode == "drive_sideways":
                target = cfg["actuator_target_ms"]
                gap    = (abs(avg_vel[1]) - target) / target * 100 if target else float("inf")
                print(f"  Target Y vel  : {target:.4f} m/s")
                print(f"  Actual Y vel  : {abs(avg_vel[1]):.4f} m/s")
                print(f"  Gap           : {gap:+.1f}%")
            elif mode == "drive_up":
                target = cfg["actuator_target_ms"]
                gap    = (abs(avg_vel[2]) - target) / target * 100 if target else float("inf")
                print(f"  Target Z vel  : {target:.4f} m/s")
                print(f"  Actual Z vel  : {abs(avg_vel[2]):.4f} m/s")
                print(f"  Gap           : {gap:+.1f}%")
    else:
        print("  Not enough data collected (sim ended before settle time?).")

    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sally wall-climbing robot demo")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Simulation duration in seconds (default: 15)")
    parser.add_argument("--mode", type=str, default="hold",
                        choices=["hold", "drive_sideways", "drive_up"],
                        help="Robot mode (default: hold)")
    args = parser.parse_args()
    run_demo(sim_duration=args.duration, mode=args.mode)


if __name__ == "__main__":
    main()