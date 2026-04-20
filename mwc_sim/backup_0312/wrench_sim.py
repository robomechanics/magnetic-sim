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

from wrench_config import (
    MU_0, MAGNET_VOLUME,
    STICK_TIP_LOCAL, LEVER_ARM,
    SCENE_XML, MAGNET_BODY_NAME, PLATE_GEOM_NAME, TIP_SITE_NAME,
    TIMESTEP, PULL_RATE, SETTLE_TIME, SIM_DURATION,
    DETACH_HOLD, DETACH_THRESHOLD,
    APPLY_FORCE, APPLY_MOMENT,
    PARAMS, ACTIVE_PRESET,
)


def mag_force(dist, Br):
    """Dipole-dipole attractive force (N) for one sampling sphere at closest-point
    distance `dist` from the plate. Formula: F = (3μ₀m²) / (2π(2d)⁴)."""
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)


def setup_model():
    print(f"[setup_model] Loading: {SCENE_XML}  (preset: {ACTIVE_PRESET})")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP

    plate_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PLATE_GEOM_NAME)
    magnet_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MAGNET_BODY_NAME)
    tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE_NAME)

    if plate_id    == -1: raise ValueError(f"'{PLATE_GEOM_NAME}' geom not found")
    if magnet_id   == -1: raise ValueError(f"'{MAGNET_BODY_NAME}' body not found")
    if tip_site_id == -1: raise ValueError(f"'{TIP_SITE_NAME}' site not found")

    model.geom_friction[plate_id] = PARAMS['ground_friction']
    model.opt.o_solref             = PARAMS['solref']
    model.opt.o_solimp             = PARAMS['solimp']
    model.opt.noslip_iterations    = PARAMS['noslip_iterations']

    # Collect all sphere geoms on the magnet body — these are the force sampling points.
    sphere_gids = [
        gid for gid in range(model.ngeom)
        if model.geom_bodyid[gid] == magnet_id
        and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    return model, data, plate_id, magnet_id, sphere_gids, tip_site_id


def apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto):
    """Apply dipole-dipole forces to magnet body. Returns total force magnitude."""
    fvec_total = np.zeros(3)
    total      = 0.0
    for gid in sphere_gids:
        dist = mujoco.mj_geomDistance(model, data, gid, plate_id, 50.0, fromto)
        if dist <= 0 or dist > PARAMS['max_magnetic_distance']:
            continue
        f    = mag_force(dist, PARAMS['Br'])
        n    = fromto[3:6] - fromto[0:3]
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            continue
        fvec_total += f * (n / norm)
        total      += f

    # Clip total assembly force, not per-sphere
    total_mag = np.linalg.norm(fvec_total)
    if total_mag > PARAMS['max_force_per_wheel']:
        fvec_total *= PARAMS['max_force_per_wheel'] / total_mag
        total       = PARAMS['max_force_per_wheel']

    data.xfrc_applied[magnet_id, :3] += fvec_total
    return total


def apply_wrench_force(model, data, magnet_id, f_pull, tip_site_id):
    """Apply horizontal pull force (+X) at the stick tip with independent toggles.

    APPLY_FORCE  — shear component: horizontal force at magnet COM
    APPLY_MOMENT — peel component:  torque cross(r, F) where r = tip_world - COM

    Both on   -> full lever-arm wrench (realistic pull at tip of stick)
    Force only -> pure shear, no rotation
    Moment only -> pure peel/rotation, no net shear
    """
    tip_world = data.site_xpos[tip_site_id].copy()
    force_vec = np.array([f_pull, 0.0, 0.0])
    r         = tip_world - data.xpos[magnet_id]
    moment    = np.cross(r, force_vec)

    if APPLY_FORCE:
        data.xfrc_applied[magnet_id, :3] += force_vec
    if APPLY_MOMENT:
        data.xfrc_applied[magnet_id, 3:] += moment


def run_headless(pull_rate=PULL_RATE):
    model, data, plate_id, magnet_id, sphere_gids, tip_site_id = setup_model()
    fromto = np.zeros(6)

    # Phase 1: gravity only — magnet falls onto plate.
    while data.time < SETTLE_TIME / 2:
        data.xfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)

    # Phase 2: magnetic force engages — magnet snaps and settles against plate.
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
        f_mag  = apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto)

        t_ramp = data.time - ramp_t0
        f_pull = pull_rate * t_ramp
        apply_wrench_force(model, data, magnet_id, f_pull, tip_site_id)

        records.append({
            't':      t_ramp,
            'f_pull': f_pull,
            'moment': f_pull * LEVER_ARM,
            'f_mag':  f_mag,
        })

        if not separated:
            detach_force = max(detach_force, f_pull)
            if f_mag < DETACH_THRESHOLD:
                if detach_start is None:
                    detach_start = data.time
                elif data.time - detach_start >= DETACH_HOLD:
                    separated = True
                    print(f"Detached | force: {detach_force:.2f} N | moment: {f_pull * LEVER_ARM:.3f} Nm")
            else:
                detach_start = None

        mujoco.mj_step(model, data)

    if not separated:
        print(f"No detachment. Max force: {detach_force:.2f} N")

    return records, detach_force


def smooth(data, window=800):
    """Box-filter (moving average) smoother. Window of 800 steps = 0.4 s at dt=0.0005."""
    arr = np.array(data, dtype=float)
    return np.convolve(arr, np.ones(window) / window, mode="same")


def plot(records, detach_force, pull_rate):
    t      = np.array([r['t']      for r in records])
    f_pull = np.array([r['f_pull'] for r in records])
    moment = np.array([r['moment'] for r in records])
    f_mag  = np.array([r['f_mag']  for r in records])

    f_mag_sm = smooth(f_mag)
    step     = max(1, len(t) // 1000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Wrench/Peel Test  |  Preset: {ACTIVE_PRESET}  |  Br={PARAMS['Br']:.3f} T  |  "
        f"Ramp={pull_rate} N/s  |  Detach={detach_force:.1f} N  |  "
        f"Lever={LEVER_ARM * 100:.1f} cm",
        fontweight="bold", fontsize=11
    )

    ax = axes[0]
    ax.plot(t[::step], f_pull[::step],   color="#e74c3c", lw=2, label="Applied force (N)")
    ax.plot(t[::step], moment[::step],   color="#8e44ad", lw=2, label="Moment (N·m)")
    ax.plot(t[::step], f_mag_sm[::step], color="#2980b9", lw=2, label="Mag. attraction (smoothed)")
    ax.axhline(detach_force, color="#333", ls="--", lw=1.2, label=f"Detach: {detach_force:.1f} N")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (N) / Moment (N·m)")
    ax.set_title("Force & Moment vs Time")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=0.25)

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