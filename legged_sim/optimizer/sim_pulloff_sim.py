"""
sim_pulloff_sim.py - Magnetic pull-off simulation.
Runs headless, plots results, then launches viewer.

Pull force is applied at the FL electromagnet body COM (+Z). Displacement
is tracked as the change in magnet COM Z position from ramp start.

Usage:
    python sim_pulloff_sim.py
    python sim_pulloff_sim.py --pull-rate 50
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mujoco

# ── Paths ─────────────────────────────────────────────────────────────────────
# OPTIMIZER_DIR: directory containing this file (and combined_config.py).
# LEGGED_DIR:    parent directory containing config.py.
# Both are added to sys.path so this file works standalone AND when imported
# as a module by the combined optimizer's worker subprocesses.

OPTIMIZER_DIR = os.path.abspath(os.path.dirname(__file__))
LEGGED_DIR    = os.path.abspath(os.path.join(OPTIMIZER_DIR, ".."))
SCENE_XML     = os.path.join(OPTIMIZER_DIR, "mwc_mjcf", "scene.xml")

for _p in (OPTIMIZER_DIR, LEGGED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Physics constants: prefer combined_config (combined optimizer context);
# fall back to config (standalone / legged_sim context).
try:
    from combined_config import MU_0, MAGNET_VOLUME, MAGNET_BODY_NAMES, TIMESTEP
except ImportError:
    from config import MU_0, MAGNET_VOLUME, MAGNET_BODY_NAMES, TIMESTEP

# Legged-sim-specific: always from the parent legged_sim config.
from config import KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG

# Timing / preset constants: prefer combined_config, fall back to sim_pulloff_config.
try:
    from combined_config import (
        PULL_RATE, PULL_RATE_OPT, SETTLE_TIME, SIM_DURATION, DETACH_HOLD,
        PULLOFF_PARAMS as PARAMS,
    )
    ACTIVE_PRESET = 'pull_off'
except ImportError:
    from sim_pulloff_config import (
        PULL_RATE, PULL_RATE_OPT, SETTLE_TIME, SIM_DURATION, DETACH_HOLD,
        PARAMS, ACTIVE_PRESET,
    )

# The FL electromagnet is the test subject for the pull-off sim.
PULL_FOOT      = "FL"
PULL_BODY_NAME = MAGNET_BODY_NAMES[0]   # electromagnet_FL
PLATE_GEOM     = "floor"
FEET           = ('FL', 'FR', 'BL', 'BR')


def mag_force(dist, Br):
    """Dipole-dipole attractive force (N) for one sampling sphere at closest-point
    distance `dist` from the plate. Formula: F = (3μ₀m²) / (2π(2d)⁴)."""
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)


# Fraction of params['max_force_per_wheel'] below which a sustained force
# drop is classified as detachment.  50% means the magnet has clearly
# broken away rather than just momentarily losing some contact area.
DETACH_FORCE_FRAC = 0.5


def setup_model(params):
    """Load scene.xml and apply parameters — mirrors sim_opt_sim._setup_model.

    Returns:
        model, data, plate_id, magnet_id, sphere_gids
        plate_id   — floor geom ID
        magnet_id  — electromagnet_FL body ID (the pull-off test subject)
        sphere_gids — list of sphere geom IDs on electromagnet_FL
    """
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP
    mujoco.mj_resetData(model, data)

    # Bake joint angles exactly as the floor/wall sims do.
    for leg in FEET:
        for jname, bake_dict in [
            (f'knee_{leg}',  KNEE_BAKE_DEG),
            (f'wrist_{leg}', WRIST_BAKE_DEG),
            (f'ee_{leg}',    EE_BAKE_DEG),
        ]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                data.qpos[model.jnt_qposadr[jid]] = np.radians(bake_dict[leg])

    mujoco.mj_forward(model, data)

    # Plate = floor geom (same name as in sim_opt_sim).
    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PLATE_GEOM)
    if plate_id == -1:
        raise ValueError(f"'{PLATE_GEOM}' geom not found in scene.xml")
    model.geom_friction[plate_id] = params['ground_friction']

    model.opt.o_solref          = params['solref']
    model.opt.o_solimp          = params['solimp']
    model.opt.noslip_iterations = params['noslip_iterations']
    model.opt.noslip_tolerance  = params['noslip_tolerance']
    model.opt.o_margin          = params['margin']
    model.dof_damping[:]        = 2.0
    model.dof_armature[:]       = 0.01

    # Single test magnet: FL electromagnet.
    magnet_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PULL_BODY_NAME)
    if magnet_id == -1:
        raise ValueError(f"'{PULL_BODY_NAME}' body not found in scene.xml")

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


# ── PID controller (mirrors sim_opt_sim._PID) ─────────────────────────────────

PID_KP      = 500.0
PID_KI      = 200.0
PID_KD      = 30.0
PID_I_CLAMP = 100.0

class _PID:
    def __init__(self, model):
        self.nu        = model.nu
        self.ctrl_jids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.integral  = np.zeros(model.nu)
        self.prev_err  = np.zeros(model.nu)

    def compute(self, model, data, targets, dt):
        torques = np.zeros(self.nu)
        for i, jid in enumerate(self.ctrl_jids):
            qidx = model.jnt_qposadr[jid]
            err  = targets[i] - data.qpos[qidx]
            self.integral[i] = np.clip(
                self.integral[i] + err * dt, -PID_I_CLAMP, PID_I_CLAMP)
            derr = (err - self.prev_err[i]) / dt if dt > 0 else 0.0
            self.prev_err[i] = err
            torques[i] = PID_KP * err + PID_KI * self.integral[i] + PID_KD * derr
        return torques


def run_headless(pull_rate=PULL_RATE, params=None):
    """Run headless pull-off simulation and return records, pull-off force, and detach time.

    Args:
        pull_rate: Force ramp rate in N/s.
        params:    PARAMS dict. Falls back to module-level PARAMS if not provided.

    Returns:
        records:       list of per-step dicts with 't', 'f_pull', 'f_mag', 'z_disp'
        pulloff_force: applied force (N) at the moment the sustained force drop begins
        detach_time:   ramp time (s) at confirmed detachment, or None if no detachment

    Detachment detection:
        Engages once f_mag first exceeds DETACH_FORCE_FRAC * max_force_per_wheel.
        Detachment is declared when f_mag then drops below that same threshold and
        stays there for DETACH_HOLD seconds continuously.  Transient dips that
        recover reset the clock so brief contact-area fluctuations are ignored.
        pulloff_force is recorded at the START of the sustained drop (i.e., the
        applied load at the moment adhesion begins to fail), not the running max.
    """
    if params is None:
        params = PARAMS

    model, data, plate_id, magnet_id, sphere_gids = setup_model(params)
    fromto = np.zeros(6)

    # PID holds the robot in its baked pose throughout — without this the body
    # sags under gravity during settle and the FL wheel lifts off the floor,
    # preventing magnetic engagement.
    pid = _PID(model)
    ctrl_targets = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_targets[i] = data.qpos[model.jnt_qposadr[jid]]

    # Phase 1: gravity + PID hold (no magnetics yet).
    while data.time < SETTLE_TIME / 2:
        data.xfrc_applied[:] = 0.0
        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

    # Phase 2: magnetics engage, PID holds robot body.
    while data.time < SETTLE_TIME:
        data.xfrc_applied[:] = 0.0
        apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto, params)
        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)
        mujoco.mj_step(model, data)

    z0      = data.xpos[magnet_id][2]   # COM Z at ramp start
    ramp_t0 = data.time
    records = []

    # Detachment state
    detach_threshold = DETACH_FORCE_FRAC * params['max_force_per_wheel']
    engaged      = False   # True once f_mag has exceeded threshold (fully adhered)
    detach_start = None    # sim time when sustained drop began
    pulloff_force = 0.0    # f_pull recorded at start of the sustained drop
    detach_time   = None   # ramp time (s) when detachment is confirmed
    separated     = False

    while data.time < SIM_DURATION:
        data.xfrc_applied[:] = 0.0
        f_mag_z = apply_mag(model, data, sphere_gids, plate_id, magnet_id, fromto, params)

        t_ramp  = data.time - ramp_t0
        f_pull  = pull_rate * t_ramp
        data.xfrc_applied[magnet_id, 2] += f_pull   # upward force at FL COM only

        data.ctrl[:] = pid.compute(model, data, ctrl_targets, TIMESTEP)

        z_disp  = (data.xpos[magnet_id][2] - z0) * 1000   # mm
        f_mag   = -f_mag_z   # positive = attractive toward plate

        records.append({'t': t_ramp, 'f_pull': f_pull, 'f_mag': f_mag, 'z_disp': z_disp})

        # Step 1: wait for magnet to fully engage before watching for drops.
        if not engaged:
            if f_mag > detach_threshold:
                engaged = True
        else:
            if f_mag < detach_threshold:
                # Force has dropped below threshold — start/continue timing the drop.
                if detach_start is None:
                    detach_start  = data.time
                    pulloff_force = f_pull   # applied load at the moment of failure onset
                elif data.time - detach_start >= DETACH_HOLD:
                    # Sustained drop confirmed — real detachment.
                    separated   = True
                    detach_time = t_ramp
                    print(
                        f"Detached | pull-off force: {pulloff_force:.2f} N "
                        f"(target={params['max_force_per_wheel']:.1f} N) "
                        f"| f_mag at detach: {f_mag:.2f} N "
                        f"| t={detach_time:.3f}s"
                    )
                    break
            else:
                # Force recovered — transient dip, not a real detachment, reset clock.
                detach_start = None

        mujoco.mj_step(model, data)

    if not separated:
        print(f"No detachment detected. Final f_mag: {records[-1]['f_mag']:.2f} N, f_pull: {records[-1]['f_pull']:.2f} N")

    return records, pulloff_force, detach_time


def smooth(data, window=200):
    """Box-car moving average with window size `window` steps."""
    arr = np.array(data, dtype=float)
    return np.convolve(arr, np.ones(window) / window, mode="same")


def plot(records, pulloff_force, detach_time, pull_rate, params):
    t      = np.array([r["t"]      for r in records])
    f_pull = np.array([r["f_pull"] for r in records])
    f_mag  = np.array([r["f_mag"]  for r in records])
    z_disp = np.array([r["z_disp"] for r in records])

    f_mag_sm  = smooth(f_mag)
    z_disp_sm = smooth(z_disp)
    step      = max(1, len(t) // 1000)

    max_adhesion    = params['max_force_per_wheel']
    detach_thresh   = DETACH_FORCE_FRAC * max_adhesion

    # Find detachment index by ramp time (force-based, matches run_headless logic)
    detach_idx = None
    if detach_time is not None:
        diffs = np.abs(t - detach_time)
        detach_idx = int(np.argmin(diffs))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Magnetic Pull-Off Test  |  Preset: {ACTIVE_PRESET}  |  Br={params['Br']:.3f} T  |  "
        f"Ramp={pull_rate} N/s  |  Pull-off={pulloff_force:.1f} N  |  "
        f"Max adhesion={max_adhesion:.1f} N",
        fontweight="bold", fontsize=11
    )

    # Left: Force vs Time
    ax = axes[0]
    ax.plot(t[::step], f_pull[::step],   color="#e74c3c", lw=2, label="Applied pull (ramp)")
    ax.plot(t[::step], f_mag_sm[::step], color="#2980b9", lw=2, label="Magnetic attraction (smoothed)")
    ax.axhline(max_adhesion,  color="#27ae60", ls="-",  lw=1.4, label=f"max_force_per_wheel: {max_adhesion:.1f} N")
    ax.axhline(detach_thresh, color="#8e44ad", ls="--", lw=1.2, label=f"Detach threshold ({DETACH_FORCE_FRAC*100:.0f}%): {detach_thresh:.1f} N")
    ax.axhline(pulloff_force, color="#333",    ls=":",  lw=1.2, label=f"Pull-off: {pulloff_force:.1f} N")
    if detach_idx is not None:
        ax.axvline(t[detach_idx], color="#e67e22", ls=":", lw=1.5,
                   label=f"Detach @ t={detach_time:.3f}s")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (N)"); ax.set_title("Force vs Time")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Right: Force vs Displacement — zoomed to detach region
    ax = axes[1]
    if detach_idx is not None:
        end  = min(detach_idx + 2000, len(z_disp))
        mask = np.arange(len(z_disp)) <= end
        ax.plot(z_disp_sm[mask][::step], f_pull[mask][::step],   color="#e74c3c", lw=2, label="Applied pull")
        ax.plot(z_disp_sm[mask][::step], f_mag_sm[mask][::step], color="#2980b9", lw=2, label="Magnetic attraction")
        ax.axhline(detach_thresh, color="#8e44ad", ls="--", lw=1.2,
                   label=f"Detach threshold: {detach_thresh:.1f} N")
        ax.axvline(z_disp_sm[detach_idx], color="#e67e22", ls=":", lw=1.5,
                   label=f"Detach @ {z_disp_sm[detach_idx]:.2f} mm")
    else:
        ax.plot(z_disp_sm[::step], f_pull[::step],   color="#e74c3c", lw=2, label="Applied pull")
        ax.plot(z_disp_sm[::step], f_mag_sm[::step], color="#2980b9", lw=2, label="Magnetic attraction")
        ax.axhline(detach_thresh, color="#8e44ad", ls="--", lw=1.2,
                   label=f"Detach threshold: {detach_thresh:.1f} N")
    ax.set_xlabel("Displacement (mm)"); ax.set_ylabel("Force (N)"); ax.set_title("Force vs Displacement")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("results/pulloff_results.png", dpi=150, bbox_inches="tight")
    print("Plot saved: results/pulloff_results.png")
    plt.show()


# ── Optimizer interface ───────────────────────────────────────────────────────
# run_headless_lift is the entry point for combined_optimizer.py so all three
# sims share a uniform single-float return interface.

def run_headless_lift(params: dict) -> float:
    """
    Thin wrapper for combined_optimizer.py.

    Uses PULL_RATE_OPT (not PULL_RATE) so the ramp can reach the full
    max_force_per_wheel search space ceiling.
      PULL_RATE_OPT × (SIM_DURATION - SETTLE_TIME) = max ramp force
      e.g. 40 N/s × 38 s = 1520 N  ≥  1500 N (combined optimizer space max)  ✓
           40 N/s × 38 s = 1520 N  ≥  1200 N (standalone PULLOFF_SPACE max)   ✓
    """
    _, pulloff_force, _ = run_headless(pull_rate=PULL_RATE_OPT, params=params)
    return pulloff_force


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull-rate', type=float, default=PULL_RATE)
    args = parser.parse_args()

    print(f"Running simulation (pull_rate={args.pull_rate} N/s, preset={ACTIVE_PRESET})...")
    records, pulloff_force, detach_time = run_headless(args.pull_rate, PARAMS)

    # Plot results first; launch viewer after so the plot is not blocked.
    plot(records, pulloff_force, detach_time, args.pull_rate, PARAMS)

    print("Launching viewer...")
    import pulloff_viewer
    pulloff_viewer.run_viewer(args.pull_rate)