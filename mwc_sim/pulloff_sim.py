"""
pulloff_sim.py - Magnetic pull-off simulation.
Runs headless, plots results, then launches viewer.

Usage:
    python pulloff_sim.py
    python pulloff_sim.py --pull-rate 50
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import mujoco

MU_0          = 4 * np.pi * 1e-7
MAGNET_VOLUME = np.pi * (0.0315 ** 2) * 0.025   # r=31.5mm, h=25mm
XMLPATH       = "XML/flat_plate_pull.xml"
PULL_RATE     = 20.0   # N/s
SETTLE_TIME   = 0.3    # s
SIM_DURATION  = 100.0   # s
DETACH_DIST   = 10.0    # mm — min displacement to count as detached
DETACH_HOLD   = 1.0    # s  — must stay above DETACH_DIST for this long

PARAMS = {
    'ground_friction':       [0.923048, 0.000546, 0.000241],
    'solref':                [0.000349, 30.867833],
    'solimp':                [0.870259, 0.980192, 0.000135, 0.5, 1.0],
    'noslip_iterations':     12,
    'Br':                    1.490629,
    'max_magnetic_distance': 0.009070,
    'max_force_per_wheel':   53.821678,
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
    joint_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pull_joint")
    sphere_gids = [
        gid for gid in range(model.ngeom)
        if model.geom_bodyid[gid] == magnet_id
        and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    return model, data, plate_id, magnet_id, joint_id, sphere_gids


def apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto):
    """Apply dipole-dipole forces to magnet body. Returns total Fz."""
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


def run_headless(pull_rate=PULL_RATE):
    model, data, plate_id, magnet_id, joint_id, sphere_gids = setup_model()
    fromto = np.zeros(6)

    # Settle phase
    while data.time < SETTLE_TIME:
        data.xfrc_applied[:] = 0.0
        apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto)
        mujoco.mj_step(model, data)

    z0            = data.qpos[model.jnt_qposadr[joint_id]]
    ramp_t0       = data.time
    records       = []
    pulloff_force = 0.0
    separated     = False
    lift_start    = None   # time when z_disp first exceeded 5mm

    while data.time < SIM_DURATION:
        data.xfrc_applied[:] = 0.0
        f_mag_z = apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto)

        t_ramp = data.time - ramp_t0
        f_pull = pull_rate * t_ramp
        data.xfrc_applied[magnet_id, 2] += f_pull

        z_disp = (data.qpos[model.jnt_qposadr[joint_id]] - z0) * 1000  # mm

        records.append({'t': t_ramp, 'f_pull': f_pull, 'f_mag': -f_mag_z, 'z_disp': z_disp})

        if not separated:
            pulloff_force = max(pulloff_force, f_pull)
            if z_disp > DETACH_DIST:
                if lift_start is None:
                    lift_start = data.time
                elif data.time - lift_start >= DETACH_HOLD:
                    separated = True
                    print(f"Detached | pull-off force: {pulloff_force:.2f} N | disp: {z_disp:.3f} mm")
            else:
                lift_start = None  # reset if it dips back below 5mm

        mujoco.mj_step(model, data)

    if not separated:
        print(f"No detachment. Max pull reached: {pulloff_force:.2f} N")

    return records, pulloff_force


def smooth(data, window=200):
    arr = np.array(data, dtype=float)
    return np.convolve(arr, np.ones(window)/window, mode="same")


def plot(records, pulloff_force, pull_rate):
    t      = np.array([r["t"]      for r in records])
    f_pull = np.array([r["f_pull"] for r in records])
    f_mag  = np.array([r["f_mag"]  for r in records])
    z_disp = np.array([r["z_disp"] for r in records])

    f_mag_sm = smooth(f_mag)
    z_disp_sm = smooth(z_disp)
    step     = max(1, len(t) // 1000)

    # Detect detach: where smoothed mag force drops to <10% of peak after peak
    detach_idx = None
    peak_idx   = int(np.argmax(f_mag_sm))
    drop       = np.where(f_mag_sm[peak_idx:] < f_mag_sm[peak_idx] * 0.1)[0]
    if len(drop):
        detach_idx = peak_idx + drop[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Magnetic Pull-Off Test  |  Br={PARAMS['Br']:.3f} T  |  "
        f"Ramp={pull_rate} N/s  |  Pull-off={pulloff_force:.1f} N",
        fontweight="bold", fontsize=12
    )

    # Left: Force vs Time
    ax = axes[0]
    ax.plot(t[::step], f_pull[::step], color="#e74c3c", lw=2,   label="Applied pull (ramp)")
    ax.plot(t[::step], f_mag_sm[::step], color="#2980b9", lw=2, label="Magnetic attraction (smoothed)")
    ax.axhline(pulloff_force, color="#333", ls="--", lw=1.2, label=f"Pull-off: {pulloff_force:.1f} N")
    if detach_idx is not None:
        ax.axvline(t[detach_idx], color="#e67e22", ls=":", lw=1.5, label="Detachment")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (N)"); ax.set_title("Force vs Time")
    ax.set_ylim(bottom=0); ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # Right: Force vs Displacement — zoomed to detach region
    ax = axes[1]
    if detach_idx is not None:
        end = min(detach_idx + 2000, len(z_disp))
        mask = np.arange(len(z_disp)) <= end
        ax.plot(z_disp_sm[mask][::step], f_pull[mask][::step], color="#e74c3c", lw=2, label="Applied pull")
        ax.plot(z_disp_sm[mask][::step], f_mag_sm[mask][::step], color="#2980b9", lw=2, label="Magnetic attraction")
        ax.axvline(z_disp_sm[detach_idx], color="#e67e22", ls=":", lw=1.5,
                   label=f"Detach @ {z_disp_sm[detach_idx]:.2f} mm")
    else:
        ax.plot(z_disp_sm[::step], f_pull[::step], color="#e74c3c", lw=2, label="Applied pull")
        ax.plot(z_disp_sm[::step], f_mag_sm[::step], color="#2980b9", lw=2, label="Magnetic attraction")
    ax.set_xlabel("Displacement (mm)"); ax.set_ylabel("Force (N)"); ax.set_title("Force vs Displacement")
    ax.set_ylim(bottom=0); ax.legend(fontsize=9); ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("pulloff_results.png", dpi=150, bbox_inches="tight")
    print("Plot saved: pulloff_results.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull-rate', type=float, default=PULL_RATE)
    args = parser.parse_args()

    print(f"Running simulation (pull_rate={args.pull_rate} N/s)...")
    records, pulloff_force = run_headless(args.pull_rate)

    print("Launching viewer...")
    import pulloff_viewer
    pulloff_viewer.run_viewer(args.pull_rate)

    plot(records, pulloff_force, args.pull_rate)