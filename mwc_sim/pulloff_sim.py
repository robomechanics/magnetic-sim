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

MU_0          = 4 * np.pi * 1e-7
MAGNET_VOLUME = np.pi * (0.0315 ** 2) * 0.025   # r=31.5mm, h=25mm
SCENE_XML     = "mwc_mjcf/scene.xml"
PULL_RATE     = 20.0   # N/s
SETTLE_TIME   = 2.0    # s — two-phase: gravity only (0→0.5s), then mag engages (0.5→1.0s)
SIM_DURATION  = 100.0  # s — hard stop
DETACH_DIST   = 10.0   # mm — min COM-Z displacement to count as detached
DETACH_HOLD   = 1.0    # s  — must stay above DETACH_DIST for this long

# ---------------------------------------------------------------------------
# PARAMS PRESETS — uncomment one block, keep the others commented out.
# All values are Bayesian-optimized against physical hardware experiments.
# ---------------------------------------------------------------------------

# --- Hold (active) ---
# PARAMS = {
#     'ground_friction':       [0.9, 0.034585, 0.001734],
#     'solref':                [0.000572, 10.000000],
#     'solimp':                [0.860956, 0.987761, 0.000100, 0.5, 1.0],
#     'noslip_iterations':     20,
#     'Br':                    1.332000,
#     'max_magnetic_distance': 0.011413,
#     'max_force_per_wheel':   139.324024,
# }

# --- Drive sideways ---
# PARAMS = {
#     'ground_friction':       [0.975785, 0.000342, 0.000372],
#     'solref':                [0.000129, 33.807584],
#     'solimp':                [0.875461, 0.934908, 0.002193, 0.5, 1.0],
#     'noslip_iterations':     20,
#     'Br':                    1.476441,
#     'max_magnetic_distance': 0.029736,
#     'max_force_per_wheel':   264.173934,
# }

# --- Drive up ---
PARAMS = {
    'ground_friction': [0.973464, 0.000202, 0.000010],
    'solref': [0.000100, 10.000000],
    'solimp': [0.926007, 0.943979, 0.001311, 0.5, 1.0],
    'noslip_iterations': 23,
    'Br': 1.599175,
    'max_magnetic_distance': 0.018389,
    'max_force_per_wheel': 205.270447,
}



def mag_force(dist, Br):
    """Dipole-dipole attractive force (N) for one sampling sphere at closest-point
    distance `dist` from the plate. Formula: F = (3μ₀m²) / (2π(2d)⁴)."""
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)


def setup_model():
    print(f"[setup_model] Loading: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = 0.0005  # 2000 Hz

    plate_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_geom")
    magnet_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "1103___pp___aws_pem_215lbs__eml63mm_24")

    if plate_id  == -1: raise ValueError("'plate_geom' not found")
    if magnet_id == -1: raise ValueError("'1103___pp___aws_pem_215lbs__eml63mm_24' body not found")

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
    return model, data, plate_id, magnet_id, sphere_gids


def apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto):
    """Apply dipole-dipole forces to magnet body.

    Clips the total assembly force vector (not per-sphere) to max_force_per_wheel.
    Returns the Z component of the applied force (negative = toward plate).
    """
    fvec_total = np.zeros(3)
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

    # Clip total assembly force, not per-sphere
    total_mag = np.linalg.norm(fvec_total)
    if total_mag > PARAMS['max_force_per_wheel']:
        fvec_total *= PARAMS['max_force_per_wheel'] / total_mag

    data.xfrc_applied[magnet_id, :3] += fvec_total
    return fvec_total[2]   # Fz (negative = attraction toward plate)


def run_headless(pull_rate=PULL_RATE):
    model, data, plate_id, magnet_id, sphere_gids = setup_model()
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

    z0            = data.xpos[magnet_id][2]   # COM Z at ramp start
    ramp_t0       = data.time
    records       = []
    pulloff_force = 0.0
    separated     = False
    lift_start    = None

    while data.time < SIM_DURATION:
        data.xfrc_applied[:] = 0.0
        f_mag_z = apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto)

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
            else:
                lift_start = None

        mujoco.mj_step(model, data)

    if not separated:
        print(f"No detachment. Max pull reached: {pulloff_force:.2f} N")

    return records, pulloff_force


def smooth(data, window=200):
    arr = np.array(data, dtype=float)
    return np.convolve(arr, np.ones(window) / window, mode="same")


def plot(records, pulloff_force, pull_rate):
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
        f"Magnetic Pull-Off Test  |  Br={PARAMS['Br']:.3f} T  |  "
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

    print(f"Running simulation (pull_rate={args.pull_rate} N/s)...")
    records, pulloff_force = run_headless(args.pull_rate)

    print("Launching viewer...")
    import pulloff_viewer
    pulloff_viewer.run_viewer(args.pull_rate)

    plot(records, pulloff_force, args.pull_rate)