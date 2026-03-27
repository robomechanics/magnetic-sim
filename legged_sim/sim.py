"""
sim.py — Magnetic adhesion simulation (hold only, no external forces).

Phase 1 (0 → SETTLE_TIME/2): gravity only, robot falls onto wall.
Phase 2 (SETTLE_TIME/2 → SETTLE_TIME): magnetic force engages, robot settles.
Phase 3 (SETTLE_TIME → SIM_DURATION): hold — magnetic force only, sim runs to timeout.

Usage:
    python sim.py
"""

import numpy as np
import mujoco

from config import (
    MU_0, MAGNET_VOLUME,
    SCENE_XML, MAGNET_BODY_NAMES, PLATE_GEOM_NAME,
    TIMESTEP, SETTLE_TIME, SIM_DURATION,
    PARAMS, MAG_ENABLED,
)


def mag_force(dist, Br):
    """Dipole-dipole attractive force (N) for one sampling sphere."""
    m = (Br * MAGNET_VOLUME) / MU_0
    return (3 * MU_0 * m**2) / (2 * np.pi * (2 * dist)**4)


def setup_model(params):
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    model.opt.timestep = TIMESTEP

    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("spider_rest").id)
    
    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PLATE_GEOM_NAME)
    if plate_id == -1: raise ValueError(f"'{PLATE_GEOM_NAME}' geom not found")

    magnet_ids = []
    for name in MAGNET_BODY_NAMES:
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if mid == -1: raise ValueError(f"'{name}' body not found")
        magnet_ids.append(mid)

    model.geom_friction[plate_id]  = params['ground_friction']
    model.opt.o_solref             = params['solref']
    model.opt.o_solimp             = params['solimp']
    model.opt.noslip_iterations    = params['noslip_iterations']
    model.opt.noslip_tolerance     = params['noslip_tolerance']
    model.opt.o_margin             = params['margin']

    # sphere_gids: dict mapping magnet_id → list of sphere geom ids on that body
    sphere_gids = {
        mid: [
            gid for gid in range(model.ngeom)
            if model.geom_bodyid[gid] == mid
            and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
        ]
        for mid in magnet_ids
    }
    return model, data, plate_id, magnet_ids, sphere_gids


def apply_mag(model, data, sphere_gids, plate_id, magnet_ids, fromto, params):
    """Apply dipole-dipole forces to all magnet bodies. Returns total Fz."""
    total_fz = 0.0
    for mid in magnet_ids:
        fvec = np.zeros(3)
        for gid in sphere_gids[mid]:
            dist = mujoco.mj_geomDistance(model, data, gid, plate_id, 50.0, fromto)
            if dist <= 0 or dist > params['max_magnetic_distance']:
                continue
            f    = mag_force(dist, params['Br'])
            n    = fromto[3:6] - fromto[0:3]
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


def run_headless(params=None):
    """Run adhesion sim. Returns records list with 't' and 'f_mag'."""
    if params is None:
        params = PARAMS

    model, data, plate_id, magnet_ids, sphere_gids = setup_model(params)
    fromto  = np.zeros(6)
    records = []

    # Phase 1: gravity only
    while data.time < SETTLE_TIME / 2:
        data.xfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)

    # Phase 2: mag engages
    while data.time < SETTLE_TIME:
        data.xfrc_applied[:] = 0.0
        if MAG_ENABLED:
            apply_mag(model, data, sphere_gids, plate_id, magnet_ids, fromto, params)
        mujoco.mj_step(model, data)

    print(f"Settled. Holding until t={SIM_DURATION}s ...")

    # Phase 3: hold
    while data.time < SIM_DURATION:
        data.xfrc_applied[:] = 0.0
        f_mag_z = apply_mag(model, data, sphere_gids, plate_id, magnet_ids, fromto, params) if MAG_ENABLED else 0.0
        records.append({'t': data.time, 'f_mag': -f_mag_z})
        mujoco.mj_step(model, data)

    print(f"Done. Mean magnetic force: {np.mean([r['f_mag'] for r in records]):.2f} N")
    return records




if __name__ == "__main__":
    run_headless()

    import viewer
    viewer.run_viewer()