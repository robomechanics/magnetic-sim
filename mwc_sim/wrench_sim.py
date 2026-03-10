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
SCENE_XML     = "mwc_mjcf/scene.xml"
PULL_RATE     = 20.0   # N/s
SETTLE_TIME   = 1.0    # s
SIM_DURATION  = 50.0  # s
DETACH_HOLD   = 0.5    # s — mag force must be ~0 for this long to count as detached

# Stick tip at y=+0.0825 in magnet body frame (0.0475 body offset + 0.035 half-length)
# STICK_TIP_LOCAL = np.array([0.0, 0.0825, 0.0])  # stick extends in +Y
STICK_TIP_LOCAL = np.array([0.0, 0.0, 0.0825])  # stick extends in local +Z
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
PARAMS = {
    'ground_friction': [0.975785, 0.000342, 0.000372],
    'solref': [0.000129, 33.807584],
    'solimp': [0.875461, 0.934908, 0.002193, 0.5, 1.0],
    'noslip_iterations': 20,
    'Br': 1.476441,
    'max_magnetic_distance': 0.029736,
    'max_force_per_wheel': 264.173934,
}

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
# PARAMS = {
#     'ground_friction': [0.9, 0.034585, 0.001734],
#     'solref': [0.000572, 10.000000],
#     'solimp': [0.860956, 0.987761, 0.000100, 0.5, 1.0],
#     'noslip_iterations': 20,
#     'Br': 1.332000,
#     'max_magnetic_distance': 0.011413,
#     'max_force_per_wheel': 139.324024,
# }

def mag_force(dist, Br):
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)


def setup_model():
    print(f"[setup_model] Loading: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)

    plate_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_geom")
    magnet_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "1103___pp___aws_pem_215lbs__eml63mm_24")
    tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "stick_tip")

    if plate_id    == -1: raise ValueError("'plate_geom' not found")
    if magnet_id   == -1: raise ValueError("'1103___pp___aws_pem_215lbs__eml63mm_24' body not found")
    if tip_site_id == -1: raise ValueError("'stick_tip' site not found")

    model.geom_friction[plate_id] = PARAMS['ground_friction']
    model.opt.o_solref             = PARAMS['solref']
    model.opt.o_solimp             = PARAMS['solimp']
    model.opt.noslip_iterations    = PARAMS['noslip_iterations']

    sphere_gids = [
        gid for gid in range(model.ngeom)
        if model.geom_bodyid[gid] == magnet_id
        and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    return model, data, plate_id, magnet_id, sphere_gids, tip_site_id


def apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto):
    """Apply dipole-dipole forces. Returns total magnetic force magnitude."""
    total = 0.0
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
        total += f
    return total


def apply_wrench_force(data, magnet_id, f_pull, tip_site_id):
    """
    Apply horizontal force (+X direction) at the stick tip in world frame.
    Uses the site's actual world position for accurate moment arm.
    """
    tip_world = data.site_xpos[tip_site_id].copy()

    # Force in +X — perpendicular to stick (which runs along Y)
    force  = np.array([f_pull, 0.0, 0.0])
    r      = tip_world - data.xpos[magnet_id]
    moment = np.cross(r, force)

    data.xfrc_applied[magnet_id, :3] += force
    data.xfrc_applied[magnet_id, 3:] += moment


def run_headless(pull_rate=PULL_RATE):
    model, data, plate_id, magnet_id, sphere_gids, tip_site_id = setup_model()
    fromto = np.zeros(6)

    # Phase 1: gravity only
    while data.time < SETTLE_TIME / 2:
        data.xfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)

    # Phase 2: mag force engages, settle against plate
    while data.time < SETTLE_TIME:
        data.xfrc_applied[:] = 0.0
        apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto)
        mujoco.mj_step(model, data)

    ramp_t0      = data.time
    records      = []
    detach_force = 0.0
    separated    = False
    detach_start = None

    while data.time < SIM_DURATION:
        data.xfrc_applied[:] = 0.0
        f_mag = apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto)

        t_ramp    = data.time - ramp_t0
        f_pull    = pull_rate * t_ramp
        apply_wrench_force(data, magnet_id, f_pull, tip_site_id)

        moment_Nm = f_pull * np.linalg.norm(STICK_TIP_LOCAL)  # F × lever arm

        records.append({
            't':      t_ramp,
            'f_pull': f_pull,
            'moment': moment_Nm,
            'f_mag':  f_mag,
        })

        if not separated:
            detach_force = max(detach_force, f_pull)
            # Detachment: magnetic force drops to ~0 (spheres out of range)
            if f_mag < 0.01:
                if detach_start is None:
                    detach_start = data.time
                elif data.time - detach_start >= DETACH_HOLD:
                    separated = True
                    print(f"Detached | force: {detach_force:.2f} N | moment: {moment_Nm:.3f} Nm")
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
    t       = np.array([r['t']       for r in records])
    f_pull  = np.array([r['f_pull']  for r in records])
    moment  = np.array([r['moment']  for r in records])
    f_mag   = np.array([r['f_mag']   for r in records])

    f_mag_sm = smooth(f_mag)
    step     = max(1, len(t) // 1000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Wrench/Peel Test  |  Br={PARAMS['Br']:.3f} T  |  "
        f"Ramp={pull_rate} N/s  |  Detach force={detach_force:.1f} N  |  "
        f"Lever={np.linalg.norm(STICK_TIP_LOCAL)*100:.1f} cm",
        fontweight="bold", fontsize=11
    )

    # Left: Force & Moment vs Time
    ax = axes[0]
    ax.plot(t[::step], f_pull[::step],   color="#e74c3c", lw=2, label="Applied force (N)")
    ax.plot(t[::step], moment[::step],   color="#8e44ad", lw=2, label="Moment (N·m)")
    ax.plot(t[::step], f_mag_sm[::step], color="#2980b9", lw=2, label="Mag. attraction (smoothed)")
    ax.axhline(detach_force, color="#333", ls="--", lw=1.2, label=f"Detach: {detach_force:.1f} N")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (N) / Moment (N·m)")
    ax.set_title("Force & Moment vs Time")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Right: Magnetic Force vs Applied Force over time
    ax = axes[1]
    ax.plot(t[::step], f_mag_sm[::step], color="#2980b9", lw=2, label="Mag. attraction (smoothed)")
    ax.plot(t[::step], f_pull[::step],   color="#e74c3c", lw=2, label="Applied force (N)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (N)")
    ax.set_title("Magnetic Force vs Applied Force")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("results/wrench_results.png", dpi=150, bbox_inches="tight")
    print("Plot saved: results/wrench_results.png")
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