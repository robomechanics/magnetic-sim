"""
wrench_sim.py - Magnetic wrench/peel test via lever arm.
Applies a horizontal force at the tip of a stick rigidly attached to the magnet,
generating a contact wrench (shear + moment) at the magnet-plate interface.
Runs headless, plots results, then launches viewer.

Usage:
    python wrench_sim.py
    python wrench_sim.py --pull-rate 20
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import mujoco

MU_0          = 4 * np.pi * 1e-7
MAGNET_VOLUME = np.pi * (0.0315 ** 2) * 0.025   # r=31.5mm, h=25mm
XMLPATH       = "XML/flat_plate_wrench.xml"
PULL_RATE     = 20.0   # N/s
SETTLE_TIME   = 0.3    # s
SIM_DURATION  = 100.0   # s
DETACH_ANGLE  = 45.0   # deg — tilt from vertical to count as detached
DETACH_DIST   = 0.5    # mm  — min COM displacement (AND condition with angle)
DETACH_HOLD   = 0.5    # s   — must satisfy both conditions for this long

# Stick tip position in magnet body frame:
# magnet half-h=0.0125, stick half-l=0.035 → stick top = 0.0125 + 0.070 = 0.0825
STICK_TIP_LOCAL = np.array([0.0, 0.0, 0.0825])  # in magnet body frame

# PARAMS = {
#     'ground_friction':       [0.923048, 0.000546, 0.000241],
#     'solref':                [0.000349, 30.867833],
#     'solimp':                [0.870259, 0.980192, 0.000135, 0.5, 1.0],
#     'noslip_iterations':     12,
#     'Br':                    1.490629,
#     'max_magnetic_distance': 0.009070,
#     'max_force_per_wheel':   53.821678,
# }

# Sideways
# PARAMS = {
#     'ground_friction': [0.975785, 0.000342, 0.000372],
#     'solref': [0.000129, 33.807584],
#     'solimp': [0.875461, 0.934908, 0.002193, 0.5, 1.0],
#     'noslip_iterations': 20,
#     'Br': 1.476441,
#     'max_magnetic_distance': 0.029736,
#     'max_force_per_wheel': 264.173934,
# }

# Up
# PARAMS = {
#     'ground_friction': [0.990289, 0.027662, 0.000010],
#     'solref': [0.000122, 43.825782],
#     'solimp': [0.884461, 0.959793, 0.006224, 0.5, 1.0],
#     'noslip_iterations': 14,
#     'Br': 1.628000,
#     'max_magnetic_distance': 0.013189,
#     'max_force_per_wheel': 300.000000,
# }

# Hold
PARAMS = {
    'ground_friction': [0.900000, 0.034585, 0.001734],
    'solref': [0.000572, 10.000000],
    'solimp': [0.860956, 0.987761, 0.000100, 0.5, 1.0],
    'noslip_iterations': 20,
    'Br': 1.332000,
    'max_magnetic_distance': 0.011413,
    'max_force_per_wheel': 139.324024,
}

def mag_force(dist, Br):
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)


def setup_model(xmlpath=XMLPATH):
    model    = mujoco.MjModel.from_xml_path(xmlpath)
    data     = mujoco.MjData(model)
    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_geom")

    model.geom_friction[plate_id] = PARAMS['ground_friction']
    model.opt.o_solref             = PARAMS['solref']
    model.opt.o_solimp             = PARAMS['solimp']
    model.opt.noslip_iterations    = PARAMS['noslip_iterations']

    magnet_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "magnet")
    joint_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "magnet_free")
    sphere_gids = [
        gid for gid in range(model.ngeom)
        if model.geom_bodyid[gid] == magnet_id
        and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    return model, data, plate_id, magnet_id, joint_id, sphere_gids


def apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto):
    """Apply dipole-dipole forces. Returns total Fz."""
    total_fz = 0.0
    for gid in sphere_gids:
        dist = mujoco.mj_geomDistance(model, data, gid, plate_id, 50.0, fromto)
        if dist <= 0 or dist > PARAMS['max_magnetic_distance']:
            continue
        f    = np.clip(mag_force(dist, PARAMS['Br']), 0.0, PARAMS['max_force_per_wheel'])
        n    = fromto[3:6] - fromto[0:3]
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            continue
        fvec = f * (n / norm)
        data.xfrc_applied[magnet_id, :3] += fvec
        total_fz += fvec[2]
    return total_fz


def get_tilt_and_disp(model, data, magnet_id, pos0):
    """
    Returns (tilt_deg, disp_mm):
      tilt_deg  — angle between magnet Z-axis and world Z (0 = upright)
      disp_mm   — horizontal COM displacement from initial position
    """
    # Magnet body orientation: extract local Z axis in world frame
    xmat      = data.xmat[magnet_id].reshape(3, 3)
    local_z   = xmat[:, 2]                        # third column = body Z in world
    tilt_deg  = np.degrees(np.arccos(np.clip(local_z[2], -1.0, 1.0)))

    # Horizontal COM displacement (XY only)
    pos_now  = data.xpos[magnet_id].copy()
    disp_mm  = np.linalg.norm((pos_now[:2] - pos0[:2])) * 1000
    return tilt_deg, disp_mm


def apply_wrench_force(data, magnet_id, model, f_pull):
    """
    Apply horizontal force (+X direction) at the stick tip in world frame.
    The stick tip world position = magnet_pos + R @ STICK_TIP_LOCAL.
    We apply it as a body force + moment (equivalent wrench) via xfrc_applied.
    """
    xmat     = data.xmat[magnet_id].reshape(3, 3)
    tip_world = data.xpos[magnet_id] + xmat @ STICK_TIP_LOCAL

    # Force in +X (horizontal, perpendicular to stick axis which is along Z)
    force = np.array([f_pull, 0.0, 0.0])

    # Moment about COM = r × F, where r = tip_world - COM
    r      = tip_world - data.xpos[magnet_id]
    moment = np.cross(r, force)

    data.xfrc_applied[magnet_id, :3] += force
    data.xfrc_applied[magnet_id, 3:] += moment


def run_headless(pull_rate=PULL_RATE):
    model, data, plate_id, magnet_id, joint_id, sphere_gids = setup_model()
    fromto = np.zeros(6)

    # Settle
    while data.time < SETTLE_TIME:
        data.xfrc_applied[:] = 0.0
        apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto)
        mujoco.mj_step(model, data)

    pos0      = data.xpos[magnet_id].copy()
    ramp_t0   = data.time
    records   = []
    detach_force = 0.0
    separated    = False
    detach_start = None

    while data.time < SIM_DURATION:
        data.xfrc_applied[:] = 0.0
        f_mag_z = apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto)

        t_ramp = data.time - ramp_t0
        f_pull = pull_rate * t_ramp
        apply_wrench_force(data, magnet_id, model, f_pull)

        tilt_deg, disp_mm = get_tilt_and_disp(model, data, magnet_id, pos0)
        moment_Nm = f_pull * np.linalg.norm(STICK_TIP_LOCAL)  # F × lever arm

        records.append({
            't':        t_ramp,
            'f_pull':   f_pull,
            'moment':   moment_Nm,
            'f_mag':    -f_mag_z,
            'tilt_deg': tilt_deg,
            'disp_mm':  disp_mm,
        })

        if not separated:
            detach_force = max(detach_force, f_pull)
            if tilt_deg > DETACH_ANGLE and disp_mm > DETACH_DIST:
                if detach_start is None:
                    detach_start = data.time
                elif data.time - detach_start >= DETACH_HOLD:
                    separated = True
                    print(f"Detached | force: {detach_force:.2f} N | moment: {moment_Nm:.3f} Nm | tilt: {tilt_deg:.1f}°")
            else:
                detach_start = None

        mujoco.mj_step(model, data)

    if not separated:
        print(f"No detachment. Max force: {detach_force:.2f} N")

    return records, detach_force


def smooth(data, window=200):
    arr = np.array(data, dtype=float)
    return np.convolve(arr, np.ones(window) / window, mode="same")


def plot(records, detach_force, pull_rate):
    t        = np.array([r['t']        for r in records])
    f_pull   = np.array([r['f_pull']   for r in records])
    moment   = np.array([r['moment']   for r in records])
    f_mag    = np.array([r['f_mag']    for r in records])
    tilt_deg = np.array([r['tilt_deg'] for r in records])
    disp_mm  = np.array([r['disp_mm']  for r in records])

    f_mag_sm   = smooth(f_mag)
    tilt_sm    = smooth(tilt_deg)
    disp_sm    = smooth(disp_mm)
    step       = max(1, len(t) // 1000)

    # Detect detach index: tilt crosses 45° after peak
    detach_idx = None
    cross      = np.where(tilt_sm > DETACH_ANGLE)[0]
    if len(cross):
        detach_idx = cross[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Wrench/Peel Test  |  Br={PARAMS['Br']:.3f} T  |  "
        f"Ramp={pull_rate} N/s  |  Detach force={detach_force:.1f} N  |  "
        f"Lever={np.linalg.norm(STICK_TIP_LOCAL)*100:.1f} cm",
        fontweight="bold", fontsize=11
    )

    def vline(ax):
        if detach_idx is not None:
            ax.axvline(t[detach_idx], color="#e67e22", ls=":", lw=1.5, label="Detachment")

    # Left: Force & Moment vs Time
    ax = axes[0]
    ax.plot(t[::step], f_pull[::step],   color="#e74c3c", lw=2, label="Applied force (N)")
    ax.plot(t[::step], moment[::step],   color="#8e44ad", lw=2, label="Moment (N·m × 10)")
    ax.plot(t[::step], f_mag_sm[::step], color="#2980b9", lw=2, label="Mag. attraction Fz (smoothed)")
    ax.axhline(detach_force, color="#333", ls="--", lw=1.2, label=f"Detach: {detach_force:.1f} N")
    vline(ax)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (N) / Moment (N·m)")
    ax.set_title("Force & Moment vs Time")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Middle: Tilt vs Time
    ax = axes[1]
    ax.plot(t[::step], tilt_sm[::step], color="#27ae60", lw=2, label="Tilt angle (smoothed)")
    ax.axhline(DETACH_ANGLE, color="#e67e22", ls="--", lw=1.2, label=f"{DETACH_ANGLE}° threshold")
    vline(ax)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Tilt (deg)")
    ax.set_title("Magnet Tilt vs Time")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Right: Force vs Tilt
    ax = axes[2]
    ax.plot(tilt_sm[::step], f_pull[::step], color="#e74c3c", lw=2, label="Applied force")
    ax.plot(tilt_sm[::step], f_mag_sm[::step], color="#2980b9", lw=2, label="Mag. attraction")
    ax.axvline(DETACH_ANGLE, color="#e67e22", ls="--", lw=1.2, label=f"{DETACH_ANGLE}° detach")
    ax.set_xlabel("Tilt (deg)"); ax.set_ylabel("Force (N)")
    ax.set_title("Force vs Tilt Angle")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("wrench_results.png", dpi=150, bbox_inches="tight")
    print("Plot saved: wrench_results.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull-rate', type=float, default=PULL_RATE)
    args = parser.parse_args()

    print(f"Running wrench simulation (pull_rate={args.pull_rate} N/s)...")
    records, detach_force = run_headless(args.pull_rate)

    print("Launching viewer...")
    import wrench_viewer
    wrench_viewer.run_viewer(args.pull_rate)

    plot(records, detach_force, args.pull_rate)