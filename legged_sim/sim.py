"""
sim.py — Magnetic adhesion simulation with crawl gait.

Phase 1 (0 → SETTLE_TIME/2):   gravity only, robot falls onto surface.
Phase 2 (SETTLE_TIME/2 → SETTLE_TIME): magnetic force engages, robot settles.
Phase 3 (SETTLE_TIME → SIM_DURATION): crawl gait — TrajectoryPlanner +
                                        MinkRobot IK + per-leg magnet control.

Usage:
    python sim.py
"""

import numpy as np
import mujoco

from config import (
    MU_0, MAGNET_VOLUME,
    SCENE_XML, MAGNET_BODY_NAMES,
    TIMESTEP, SETTLE_TIME, SIM_DURATION, TELEMETRY_INTERVAL,
    PARAMS, MAG_ENABLED,
    JOINT_DAMPING, JOINT_ARMATURE, SERVO_KP, SERVO_KV,
    SWING_DURATION, SWING_LIFT_HEIGHT,
    DEMAGNETIZE_HOLD, MAGNETIZE_HOLD,
    bake_joint_angles,
)

# ── Gait / IK imports ─────────────────────────────────────────────────────────
from trajectory import TrajectoryPlanner, SurfacePatch, GAIT_ORDER
from ik import MinkRobot, EE_FRAME_NAME  # EE_FRAME_NAME used by viewer

# Scene geometry read from scene.xml:
#
#   floor: box pos="0 0 -0.005" size="0.5 0.5 0.005"
#          top face at Z = -0.005 + 0.005 = 0.0  →  normal +Z, origin (0,0,0)
#
#   wall:  box pos="0.505 0 0.5" size="0.005 0.5 0.5"
#          inner face at X = 0.505 - 0.005 = 0.500
#          normal faces toward robot (-X direction), origin (0.5, 0, 0.5)
#
FLOOR_SURFACE = SurfacePatch(
    normal = np.array([ 0.0, 0.0, 1.0]),
    origin = np.array([-1.0, 0.0, 0.0]),   # center of extended 3 m floor
)
WALL_SURFACE = SurfacePatch(
    normal = np.array([-1.0, 0.0, 0.0]),
    origin = np.array([ 0.5, 0.0, 0.5]),
)
SURFACES = [FLOOR_SURFACE, WALL_SURFACE]

# Geom names on each EM body that can make contact with the magnetic surface.
# Used by _read_contacts to look up active contact pairs.
# If the EM body has a single capsule/box contact geom, list it here.
# Set to None to fall back to body-level contact scanning.
EM_CONTACT_GEOM: dict[str, str | None] = {
    'FL': None,
    'FR': None,
    'BL': None,
    'BR': None,
}


# ─────────────────────────────────────────────────────────────────────────────
# Unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

def mag_force(dist, Br):
    """Dipole-dipole attractive force (N) for one sampling sphere."""
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)


def setup_model(params):
    bake_joint_angles()   # recompute + overwrite knee/wrist/EE geometry in robot.xml
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP

    mujoco.mj_resetData(model, data)

    # ── Set initial joint angles from config ──────────────────────────────────
    from config import KNEE_BAKE_DEG, WRIST_BAKE_DEG, EE_BAKE_DEG
    for leg in ('FL', 'FR', 'BL', 'BR'):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"knee_{leg}")
        if jid != -1:
            data.qpos[model.jnt_qposadr[jid]] = np.radians(KNEE_BAKE_DEG[leg])

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"wrist_{leg}")
        if jid != -1:
            data.qpos[model.jnt_qposadr[jid]] = np.radians(WRIST_BAKE_DEG[leg])

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"ee_{leg}")
        if jid != -1:
            data.qpos[model.jnt_qposadr[jid]] = np.radians(EE_BAKE_DEG[leg])

    mujoco.mj_forward(model, data)
    # ─────────────────────────────────────────────────────────────────────────

    # Both magnetic surfaces need friction and solver params applied.
    # plate_ids is a set of geom ids — used by apply_mag and _read_contacts.
    plate_ids: set[int] = set()
    for geom_name in ("floor", "wall"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid == -1:
            raise ValueError(f"[sim] Geom '{geom_name}' not found in scene.")
        plate_ids.add(gid)
        model.geom_friction[gid] = params['ground_friction']

    model.opt.o_solref          = params['solref']
    model.opt.o_solimp          = params['solimp']
    model.opt.noslip_iterations = params['noslip_iterations']
    model.opt.noslip_tolerance  = params['noslip_tolerance']
    model.opt.o_margin          = params['margin']

    # ── Servo and joint dynamics from config ──────────────────────────────
    model.dof_damping[:]          = JOINT_DAMPING
    model.dof_armature[:]         = JOINT_ARMATURE
    # Position servo: gainprm[:,0]=kp, biasprm[:,1]=-kv, biasprm[:,2]=-kp
    model.actuator_gainprm[:, 0]  = SERVO_KP
    model.actuator_biasprm[:, 1]  = -SERVO_KV
    model.actuator_biasprm[:, 2]  = -SERVO_KP

    magnet_ids = []
    for name in MAGNET_BODY_NAMES:
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if mid == -1: raise ValueError(f"'{name}' body not found")
        magnet_ids.append(mid)

    # sphere_gids: dict mapping magnet_id → list of sphere geom ids on that body
    sphere_gids = {
        mid: [
            gid for gid in range(model.ngeom)
            if model.geom_bodyid[gid] == mid
            and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
        ]
        for mid in magnet_ids
    }
    return model, data, plate_ids, magnet_ids, sphere_gids


# ─────────────────────────────────────────────────────────────────────────────
# apply_mag — one new parameter: magnet_states
# Only change: skip bodies whose magnet is commanded off.
# Force math, capping, and xfrc_applied writes are identical to original.
# ─────────────────────────────────────────────────────────────────────────────

def apply_mag(model, data, sphere_gids, plate_ids, magnet_ids, params,
              magnet_states: dict[str, bool] | None = None):
    """
    Apply dipole-dipole forces to all magnet bodies. Returns total Fz.

    Parameters
    ----------
    plate_ids : set[int]
        Geom ids of all magnetic surfaces (floor + wall). Each sampling sphere
        is tested against every surface; the closest one governs.
    magnet_states : dict {foot: bool} or None
        Per-leg magnet enable flag from the planner. If None (phases 1 & 2),
        all magnets are treated as on — preserving the original settle behaviour.
    """
    total_fz = 0.0
    _fromto  = np.zeros(6)   # scratch buffer per sphere-surface pair

    for idx, mid in enumerate(magnet_ids):
        # Resolve which foot this body belongs to.
        body_name = MAGNET_BODY_NAMES[idx]   # e.g. "electromagnet_BL"
        foot      = body_name.split("_")[-1] # e.g. "BL"

        # Per-leg gate: skip force application if magnet is off.
        if magnet_states is not None and not magnet_states.get(foot, True):
            continue

        fvec = np.zeros(3)
        for gid in sphere_gids[mid]:
            # Find the nearest surface geom and the distance to it.
            best_dist   = np.inf
            best_fromto = None
            for pid in plate_ids:
                d = mujoco.mj_geomDistance(model, data, gid, pid, 50.0, _fromto)
                if d < best_dist:
                    best_dist   = d
                    best_fromto = _fromto.copy()

            if best_dist <= 0 or best_dist > params['max_magnetic_distance']:
                continue
            f    = mag_force(best_dist, params['Br'])
            n    = best_fromto[3:6] - best_fromto[0:3]
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            fvec += f * (n / norm)

        total_mag = np.linalg.norm(fvec)
        if total_mag > params['max_force_per_wheel']:
            fvec *= params['max_force_per_wheel'] / total_mag

        data.xfrc_applied[mid, :3] += fvec
        total_fz += fvec[2]
    return total_fz


# ─────────────────────────────────────────────────────────────────────────────
# Contact detection
# ─────────────────────────────────────────────────────────────────────────────

def _build_em_body_ids(model: mujoco.MjModel) -> dict[str, int]:
    """
    Build a {foot: body_id} map for the four EM bodies.
    Called once after setup_model.
    """
    ids = {}
    for foot in GAIT_ORDER:
        name = f"electromagnet_{foot}"
        bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            raise ValueError(f"[sim] Body '{name}' not found — check MAGNET_BODY_NAMES.")
        ids[foot] = bid
    return ids


def _read_contacts(
    model:       mujoco.MjModel,
    data:        mujoco.MjData,
    plate_ids:   set[int],
    em_body_ids: dict[str, int],
) -> dict[str, bool]:
    """
    Return {foot: bool} — True if the foot's EM body has an active MuJoCo
    contact against any magnetic surface geom (floor or wall) this timestep.

    Strategy: scan data.contact[:data.ncon] for any pair where one geom
    belongs to the EM body and the other is in plate_ids.  A contact is
    active if dist ≤ params margin (MuJoCo reports contacts within margin
    before physical touch; threshold here is conservatively 1 mm).
    """
    # Collect geom ids that belong to each EM body.
    em_geom_ids: dict[str, set[int]] = {
        foot: {
            gid for gid in range(model.ngeom)
            if model.geom_bodyid[gid] == bid
        }
        for foot, bid in em_body_ids.items()
    }

    contact_states: dict[str, bool] = {foot: False for foot in GAIT_ORDER}

    for i in range(data.ncon):
        c    = data.contact[i]
        g1   = int(c.geom1)
        g2   = int(c.geom2)
        dist = float(c.dist)

        if dist > 1e-3:
            continue

        for foot, gids in em_geom_ids.items():
            if (g1 in gids and g2 in plate_ids) or \
               (g2 in gids and g1 in plate_ids):
                contact_states[foot] = True
                break

    return contact_states


# ─────────────────────────────────────────────────────────────────────────────
# run_headless — gait loop
# ─────────────────────────────────────────────────────────────────────────────

def run_headless(params=None):
    """
    Run the crawl-gait simulation headlessly.
    Returns records list with 't', 'f_mag', and 'gait_phase'.
    """
    if params is None:
        params = PARAMS

    model, data, plate_ids, magnet_ids, sphere_gids = setup_model(params)
    records = []

    # ── Phase 1: gravity only ─────────────────────────────────────────────────
    while data.time < SETTLE_TIME / 2:
        data.xfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)

    # ── Phase 2: mag engages, all magnets on ──────────────────────────────────
    while data.time < SETTLE_TIME:
        data.xfrc_applied[:] = 0.0
        if MAG_ENABLED:
            apply_mag(model, data, sphere_gids, plate_ids, magnet_ids,
                      params, magnet_states=None)
        mujoco.mj_step(model, data)

    print(f"Settled. Starting crawl gait at t={data.time:.3f}s ...")

    # ── Build IK robot — reads current FK for initial foot positions ──────────
    mujoco.mj_forward(model, data)   # ensure xpos/xmat are current after settle
    robot = MinkRobot(model, data)

    initial_foot_pos = {foot: robot.ee_pos_world(foot) for foot in GAIT_ORDER}
    print("[sim] Initial foot positions (world frame):")
    for foot, pos in initial_foot_pos.items():
        print(f"  {foot}: {pos}")

    # ── Build trajectory planner ──────────────────────────────────────────────
    planner = TrajectoryPlanner(
        surfaces         = SURFACES,
        initial_foot_pos = initial_foot_pos,
        body_R_world     = robot.body_R_world,
        body_pos_world   = robot.body_pos_world,
        walk_dir         = np.array([1.0, 0.0, 0.0]),   # +X = forward
        lift_height      = SWING_LIFT_HEIGHT,
        swing_duration   = SWING_DURATION,
        demagnetize_hold = DEMAGNETIZE_HOLD,
        magnetize_hold   = MAGNETIZE_HOLD,
    )

    # ── Pre-build EM body id map for contact detection ────────────────────────
    em_body_ids = _build_em_body_ids(model)

    # ── Phase 3: crawl gait ───────────────────────────────────────────────────
    _last_print = SETTLE_TIME
    while data.time < SIM_DURATION:
        data.xfrc_applied[:] = 0.0

        # 1. Contact sensing.
        contact_states = _read_contacts(model, data, plate_ids, em_body_ids)

        # 2. Planner step → foot targets + desired magnet states.
        foot_targets   = planner.step(TIMESTEP, contact_states)
        magnet_states  = planner.magnet_states()

        # 3. IK → writes data.ctrl (position servos).
        robot.solve_ik(foot_targets, TIMESTEP)

        # 4. Magnetic force application — per-leg gated by planner magnet state.
        f_mag_z = 0.0
        if MAG_ENABLED:
            f_mag_z = apply_mag(
                model, data, sphere_gids, plate_ids, magnet_ids,
                params, magnet_states=magnet_states,
            )

        records.append({
            't':          data.time,
            'f_mag':      -f_mag_z,
            'gait_phase': planner.phase.value,
            'swing_foot': planner.swing_foot,
        })

        if data.time - _last_print >= TELEMETRY_INTERVAL:
            _last_print = data.time
            swing = planner.swing_foot or '—'
            conts = [f for f, v in contact_states.items() if v]
            mags  = [f for f, v in magnet_states.items() if v]
            print(
                f"\nt={data.time:.2f}s  [CRAWL]  "
                f"gait={planner.phase.value}  swing={swing}  "
                f"phase_t={planner._phase_t:.2f}s  F_mag={-f_mag_z:.1f}N"
            )
            if foot_targets:
                target_map = {ft.foot: ft for ft in foot_targets}
                for foot in GAIT_ORDER:
                    ft = target_map[foot]
                    role    = 'SWING ' if foot == planner.swing_foot else 'stance'
                    contact = '●' if contact_states.get(foot) else '○'
                    mag     = 'M' if magnet_states.get(foot) else 'm'
                    p = ft.pos_world
                    print(f"  {foot} [{role}]  contact={contact}  mag={mag}  "
                          f"target=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})  "
                          f"w_pos={ft.weight_pos:.1f}")

        mujoco.mj_step(model, data)

    print(f"Done. Mean magnetic force: {np.mean([r['f_mag'] for r in records]):.2f} N")
    return records


if __name__ == "__main__":
    run_headless()

    import viewer
    viewer.run_viewer()