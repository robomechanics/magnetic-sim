"""
viewer.py — Visualization primitives for Sally's f2w sim.

Exported:
    _build_joint_vis(model)  → joint_vis list (call once after setup_model)
    draw_markers(...)        → renders joint axes, magnet status, IK targets,
                               stance spheres, and EE local frame each frame

Imported by sim.py; does NOT import from sim.py to avoid circular dependencies.
"""

import numpy as np
import mujoco

# ── local constants (mirrors sim.py — keep in sync) ──────────────────────────
FEET       = ('FL', 'FR', 'BL', 'BR')
SWING_FOOT = "FL"

# ── joint axis color palette ──────────────────────────────────────────────────
_JOINT_COLORS = {
    'hip_pitch': [1.0, 1.0, 0.0, 0.9],
    'knee':      [0.0, 1.0, 1.0, 0.9],
    'wrist':     [1.0, 0.0, 1.0, 0.9],
    'ee2':       [0.2, 1.0, 0.5, 0.9],
    'em_z':      [0.5, 0.5, 1.0, 0.9],
    'ee':        [1.0, 0.5, 0.0, 0.9],
}


# ── primitive helpers ─────────────────────────────────────────────────────────

def _add_capsule(scn, size, pos, rot_flat, rgba):
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, size, pos, rot_flat, rgba)
        scn.ngeom += 1


def _add_sphere(scn, radius, pos, rgba):
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            [radius, 0, 0], pos, np.eye(3).flatten(), rgba)
        scn.ngeom += 1


# ── setup ─────────────────────────────────────────────────────────────────────

def _build_joint_vis(model):
    """Pre-compute (jid, bid, local_axis, rgba) tuples for draw_markers.
    Call once after setup_model(); pass the result to draw_markers each frame.
    """
    out = []
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not jname or jname == "root":
            continue
        color = next((c for p, c in _JOINT_COLORS.items() if jname.startswith(p)),
                     [0.5, 0.5, 0.5, 0.8])
        out.append((i, model.jnt_bodyid[i], model.jnt_axis[i].copy(), color))
    return out


# ── main draw call ─────────────────────────────────────────────────────────────

def draw_markers(viewer, model, data, ik, joint_vis,
                 args, settled, target_pos, face_axis, swing_off,
                 swing_mag_bid, swing_foot=SWING_FOOT):
    """Render joint axes, magnet indicators, IK targets, stance spheres, and EE frame.

    Call once per display frame from the viewer run loop in sim.py.
    """
    scn = viewer._user_scn
    scn.ngeom = 0

    # ── joint rotation axes ──────────────────────────────────────────────────
    for jid, bid, local_axis, color in joint_vis:
        z = data.xmat[bid].reshape(3, 3) @ local_axis
        z /= np.linalg.norm(z)
        x = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
        x = x - np.dot(x, z) * z;  x /= np.linalg.norm(x)
        _add_capsule(scn, [0.009, 0.045, 0], data.xanchor[jid],
                     np.column_stack([x, np.cross(z, x), z]).flatten(), color)

    # ── magnet status: red = swing/off, green = stance/on ────────────────────
    for foot in FEET:
        pos = data.xpos[ik.ee_bids[foot]].copy(); pos[2] += 0.035
        color = ([1.0, 0.1, 0.1, 0.9] if (settled and foot == swing_foot)
                 else [0.1, 1.0, 0.1, 0.9])
        _add_capsule(scn, [0.005, 0.025, 0], pos, np.eye(3).flatten(), color)

    if args.no_ik or not settled:
        return

    # ── IK swing target: sphere + XYZ triad ──────────────────────────────────
    if target_pos is not None:
        _add_sphere(scn, 0.008, target_pos, [0.2, 1.0, 0.2, 0.5])
        for offset, rot, rgba in [
            (np.array([0.02, 0, 0]),  np.array([[0,0,1],[0,1,0],[-1,0,0]], float), [1,.2,.2,.8]),
            (np.array([0, 0.02, 0]),  np.array([[1,0,0],[0,0,1],[0,-1,0]], float), [.2,1,.2,.8]),
            (np.array([0, 0, 0.02]),  np.eye(3),                                   [.2,.2,1,.8]),
        ]:
            _add_capsule(scn, [0.002, 0.02, 0],
                         target_pos + offset, rot.flatten(), rgba)

    # ── stance target spheres (yellow) ───────────────────────────────────────
    for foot in FEET:
        if foot != swing_foot and foot in ik.stance_targets:
            _add_sphere(scn, 0.005, ik.stance_targets[foot], [1.0, 1.0, 0.2, 0.5])

    # ── EE local frame triad for active swing foot (X=red, Y=green, Z=blue) ──
    ee_pos = data.xpos[ik.ee_bids[swing_foot]].copy()
    ee_rot = data.xmat[ik.ee_bids[swing_foot]].reshape(3, 3)
    for col, rgba in [(0, [1.,.2,.2,.9]), (1, [.2,1.,.2,.9]), (2, [.2,.2,1.,.9])]:
        cz = ee_rot[:, col]
        cx = np.array([1,0,0]) if abs(cz[0]) < 0.9 else np.array([0,1,0])
        cx = cx - np.dot(cx, cz) * cz;  cx /= np.linalg.norm(cx)
        _add_capsule(scn, [0.002, 0.02, 0], ee_pos + cz * 0.02,
                     np.column_stack([cx, np.cross(cz, cx), cz]).flatten(), rgba)