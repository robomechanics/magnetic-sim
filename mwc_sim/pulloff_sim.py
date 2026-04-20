"""
pulloff_sim.py - Magnetic pull-off simulation.
Runs headless, plots results, then launches viewer.

Pull force is applied at the magnet body COM (+Z). Displacement is tracked
as the change in magnet COM Z position from ramp start.

Usage:
    python pulloff_sim.py
    python pulloff_sim.py --pull-rate 50
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import mujoco

from pulloff_config import (
    MU_0, MAGNET_VOLUME,
    SCENE_XML, MAGNET_BODY_NAME, PLATE_GEOM_NAME,
    TIMESTEP, PULL_RATE, SETTLE_TIME, SIM_DURATION,
    DETACH_DIST, DETACH_HOLD,
    PARAMS, ACTIVE_PRESET,
)


def mag_force(dist, Br):
    """Dipole-dipole attractive force (N) for one sampling sphere at closest-point
    distance `dist` from the plate. Formula: F = (3μ₀m²) / (2π(2d)⁴)."""
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)


def setup_model(params):
    """Load the MuJoCo model and apply parameters.

    Args:
        params: PARAMS dict to apply. Must be provided — no fallback.

    Returns:
        model, data, plate_id, magnet_id, sphere_gids
    """
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP

    plate_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PLATE_GEOM_NAME)
    magnet_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MAGNET_BODY_NAME)

    if plate_id  == -1: raise ValueError(f"'{PLATE_GEOM_NAME}' geom not found")
    if magnet_id == -1: raise ValueError(f"'{MAGNET_BODY_NAME}' body not found")

    model.geom_friction[plate_id]   = params['ground_friction']
    model.opt.o_solref               = params['solref']
    model.opt.o_solimp               = params['solimp']
    model.opt.noslip_iterations      = params['noslip_iterations']
    model.opt.noslip_tolerance       = params['noslip_tolerance']
    model.opt.o_margin               = params['margin']

    # Collect all sphere geoms on the magnet body — these are the force sampling points.
    sphere_gids = [
        gid for gid in range(model.ngeom)
        if model.geom_bodyid[gid] == magnet_id
        and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    return model, data, plate_id, magnet_id, sphere_gids


def apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto, params):
    """Apply dipole-dipole forces to magnet body. Returns Fz (negative = toward plate).

    Args:
        params: PARAMS dict containing 'Br', 'max_magnetic_distance', 'max_force_per_wheel'.
                Must be provided — no fallback.
    """
    fvec_total = np.zeros(3)
    for gid in sphere_gids:
        dist = mujoco.mj_geomDistance(model, data, gid, plate_id, 50.0, fromto)
        if dist <= 0 or dist > params['max_magnetic_distance']:
            continue
        f    = mag_force(dist, params['Br'])
        n    = fromto[3:6] - fromto[0:3]
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            continue
        fvec_total += f * (n / norm)

    # Clip total assembly force, not per-sphere
    total_mag = np.linalg.norm(fvec_total)
    if total_mag > params['max_force_per_wheel']:
        fvec_total *= params['max_force_per_wheel'] / total_mag

    data.xfrc_applied[magnet_id, :3] += fvec_total
    return fvec_total[2]   # Fz (negative = attraction toward plate)


def run_headless(pull_rate=PULL_RATE, params=None):
    """Run headless pull-off simulation and return records and peak pull-off force.

    Args:
        pull_rate: Force ramp rate in N/s.
        params:    PARAMS dict. Falls back to module-level PARAMS if not provided.

    Returns:
        records:       list of per-step dicts with 't', 'f_pull', 'f_mag', 'z_disp'
        pulloff_force: peak pull force at detachment (N)
    """
    if params is None:
        params = PARAMS

    model, data, plate_id, magnet_id, sphere_gids = setup_model(params)
    fromto = np.zeros(6)

    # Phase 1: gravity only — magnet falls onto plate.
    while data.time < SETTLE_TIME / 2:
        data.xfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)

    # Phase 2: magnetic force engages — magnet snaps and settles against plate.
    while data.time < SETTLE_TIME:
        data.xfrc_applied[:] = 0.0
        apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto, params)
        mujoco.mj_step(model, data)

    z0            = data.xpos[magnet_id][2]   # COM Z at ramp start
    ramp_t0       = data.time
    records       = []
    pulloff_force = 0.0
    separated     = False
    lift_start    = None

    while data.time < SIM_DURATION:
        data.xfrc_applied[:] = 0.0
        f_mag_z = apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto, params)

        t_ramp = data.time - ramp_t0
        f_pull = pull_rate * t_ramp
        data.xfrc_applied[magnet_id, 2] += f_pull   # upward force at COM

        z_disp = (data.xpos[magnet_id][2] - z0) * 1000   # mm

        records.append({'t': t_ramp, 'f_pull': f_pull, 'f_mag': -f_mag_z, 'z_disp': z_disp})

        if not separated:
            pulloff_force = max(pulloff_force, f_pull)
            if z_disp > DETACH_DIST:
                if lift_start is None:
                    lift_start = data.time
                elif data.time - lift_start >= DETACH_HOLD:
                    separated = True
                    print(f"Detached | pull-off force: {pulloff_force:.2f} N | disp: {z_disp:.3f} mm")
                    break
            else:
                lift_start = None

        mujoco.mj_step(model, data)

    if not separated:
        print(f"No detachment. Max pull reached: {pulloff_force:.2f} N")

    return records, pulloff_force


def smooth(data, window=200):
    arr = np.array(data, dtype=float)
    return np.convolve(arr, np.ones(window) / window, mode="same")


def plot(records, pulloff_force, pull_rate, params):
    t      = np.array([r["t"]      for r in records])
    f_pull = np.array([r["f_pull"] for r in records])
    f_mag  = np.array([r["f_mag"]  for r in records])
    z_disp = np.array([r["z_disp"] for r in records])

    f_mag_sm  = smooth(f_mag)
    z_disp_sm = smooth(z_disp)
    step      = max(1, len(t) // 1000)

    # Detect detach: where smoothed mag force drops to <10% of peak after peak
    detach_idx = None
    peak_idx   = int(np.argmax(f_mag_sm))
    drop       = np.where(f_mag_sm[peak_idx:] < f_mag_sm[peak_idx] * 0.1)[0]
    if len(drop):
        detach_idx = peak_idx + drop[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Magnetic Pull-Off Test  |  Preset: {ACTIVE_PRESET}  |  Br={params['Br']:.3f} T  |  "
        f"Ramp={pull_rate} N/s  |  Pull-off={pulloff_force:.1f} N",
        fontweight="bold", fontsize=12
    )

    # Left: Force vs Time
    ax = axes[0]
    ax.plot(t[::step], f_pull[::step],   color="#e74c3c", lw=2, label="Applied pull (ramp)")
    ax.plot(t[::step], f_mag_sm[::step], color="#2980b9", lw=2, label="Magnetic attraction (smoothed)")
    ax.axhline(pulloff_force, color="#333", ls="--", lw=1.2, label=f"Pull-off: {pulloff_force:.1f} N")
    if detach_idx is not None:
        ax.axvline(t[detach_idx], color="#e67e22", ls=":", lw=1.5, label="Detachment")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (N)"); ax.set_title("Force vs Time")
    ax.set_ylim(bottom=0); ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # Right: Force vs Displacement — zoomed to detach region
    ax = axes[1]
    if detach_idx is not None:
        end  = min(detach_idx + 2000, len(z_disp))
        mask = np.arange(len(z_disp)) <= end
        ax.plot(z_disp_sm[mask][::step], f_pull[mask][::step],   color="#e74c3c", lw=2, label="Applied pull")
        ax.plot(z_disp_sm[mask][::step], f_mag_sm[mask][::step], color="#2980b9", lw=2, label="Magnetic attraction")
        ax.axvline(z_disp_sm[detach_idx], color="#e67e22", ls=":", lw=1.5,
                   label=f"Detach @ {z_disp_sm[detach_idx]:.2f} mm")
    else:
        ax.plot(z_disp_sm[::step], f_pull[::step],   color="#e74c3c", lw=2, label="Applied pull")
        ax.plot(z_disp_sm[::step], f_mag_sm[::step], color="#2980b9", lw=2, label="Magnetic attraction")
    ax.set_xlabel("Displacement (mm)"); ax.set_ylabel("Force (N)"); ax.set_title("Force vs Displacement")
    ax.set_ylim(bottom=0); ax.legend(fontsize=9); ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("results/pulloff_results.png", dpi=150, bbox_inches="tight")
    print("Plot saved: results/pulloff_results.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull-rate', type=float, default=PULL_RATE)
    args = parser.parse_args()

    print(f"Running simulation (pull_rate={args.pull_rate} N/s, preset={ACTIVE_PRESET})...")
    records, pulloff_force = run_headless(args.pull_rate, PARAMS)

    print("Launching viewer...")
    import pulloff_viewer
    pulloff_viewer.run_viewer(args.pull_rate)

    plot(records, pulloff_force, args.pull_rate, PARAMS)