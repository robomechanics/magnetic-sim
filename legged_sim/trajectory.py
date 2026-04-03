"""
trajectory.py — Trajectory planner for Sally's crawl gait.

Produces per-foot world-frame position + orientation targets each timestep,
consumed by the Mink IK layer (ik.py).

Call site in the sim loop:
    foot_targets = planner.step(dt, contact_states)

Gait order: FL → BR → FR → BL (diagonal pairs, crawl).
State machine per swing foot:
    STANCE_ALL → DEMAGNETIZE → SWING → MAGNETIZE → STANCE_ALL

Swing arc (quintic ease-in/ease-out + surface-normal lift):
    s(t)   = 10t³ - 15t⁴ + 6t⁵
    pos(t) = lerp(p_start, p_end, s(t)) + n_hat * h * sin(π * s(t))

EE orientation: slerp from departure-surface-aligned to landing-surface-aligned
over the same quintic profile.

Foothold planning (FK-workspace-based):
    The reachable workspace for each leg is sampled via forward kinematics over
    the full joint limit grid (hip_pitch × knee × wrist × ee). The candidate
    landing point is chosen as the workspace point that maximises progress along
    the desired walk direction, subject to lying on the correct surface plane and
    staying away from the workspace boundary (interior_margin).

    AABB clamping has been removed. The workspace geometry is the constraint.

Surface classification: FLOOR (normal ≈ +Z) or WALL (normal ≈ +X or +Y).
Corner-crossing arc blending: NOT YET IMPLEMENTED (raises NotImplementedError).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GAIT_ORDER: Tuple[str, ...] = ('FL', 'BR', 'FR', 'BL')

# Hip mount positions in body frame (from robot_original.xml).
HIP_OFFSET_BODY: Dict[str, np.ndarray] = {
    'FL': np.array([ 0.0965,  0.092, 0.023]),
    'FR': np.array([ 0.0965, -0.092, 0.023]),
    'BL': np.array([-0.0965,  0.092, 0.023]),
    'BR': np.array([-0.0965, -0.092, 0.023]),
}

# Joint limits (degrees) matching actuator ctrlrange in robot_original.xml.
# hip_pitch: ±45  knee: ±45  wrist: ±67.5  ee: ±45
HIP_PITCH_RANGE  = (-45.0,  45.0)
KNEE_RANGE       = (-45.0,  45.0)
WRIST_RANGE      = (-67.5,  67.5)
EE_RANGE         = (-45.0,  45.0)

# Workspace sampling resolution (points per joint).
# 12×12×10×8 ≈ 11 520 points per leg — fast enough to build at startup.
WS_SAMPLES_HIP   = 12
WS_SAMPLES_KNEE  = 12
WS_SAMPLES_WRIST = 10
WS_SAMPLES_EE    =  8

# Fraction of the workspace boundary to exclude when selecting footholds.
# 0.10 = keep away from the outer 10 % of the radial extent.
WS_INTERIOR_MARGIN = 0.10

# Swing trajectory parameters.
SWING_LIFT_HEIGHT = 0.04   # m
SWING_DURATION    = 0.6    # s
STEP_LENGTH       = 0.06   # m  (kept for reference; workspace planner uses walk_dir)

# Magnet engagement window.
DEMAGNETIZE_HOLD = 0.10   # s
MAGNETIZE_HOLD   = 0.15   # s

# Mink task weights.
WEIGHT_POS_STANCE    = 10.0
WEIGHT_ORI_STANCE    =  5.0
WEIGHT_POS_SWING     =  2.0
WEIGHT_ORI_SWING     =  1.0
WEIGHT_POS_TOUCHDOWN = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SurfacePatch:
    """An infinite plane defined by an outward normal and a point on the surface."""
    normal: np.ndarray   # unit outward normal, world frame
    origin: np.ndarray   # any point on the surface, world frame

    def project(self, p: np.ndarray) -> np.ndarray:
        """Project world-frame point p onto this surface plane."""
        return p - np.dot(p - self.origin, self.normal) * self.normal

    def signed_distance(self, p: np.ndarray) -> float:
        return float(np.dot(p - self.origin, self.normal))

    def foot_quat(self) -> np.ndarray:
        """
        Quaternion (w, x, y, z) that aligns the foot's contact face with the surface.
        Convention: foot Z-axis points along –normal (into surface).
        """
        z_foot = -self.normal
        x_ref  = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(z_foot, x_ref)) > 0.99:
            x_ref = np.array([0.0, 1.0, 0.0])
        x_foot = np.cross(np.cross(z_foot, x_ref), z_foot)
        x_foot /= np.linalg.norm(x_foot)
        y_foot  = np.cross(z_foot, x_foot)
        R = np.column_stack([x_foot, y_foot, z_foot])
        return _rot_to_quat(R)


@dataclass
class FootTarget:
    """Per-foot IK target produced each timestep by TrajectoryPlanner.step()."""
    foot:       str
    pos_world:  np.ndarray
    quat_world: np.ndarray
    weight_pos: float = WEIGHT_POS_STANCE
    weight_ori: float = WEIGHT_ORI_STANCE
    magnet_on:  bool  = True


class GaitPhase(enum.Enum):
    STANCE_ALL  = "STANCE_ALL"
    DEMAGNETIZE = "DEMAGNETIZE"
    SWING       = "SWING"
    MAGNETIZE   = "MAGNETIZE"


@dataclass
class _LegState:
    """Internal per-leg bookkeeping."""
    contact_pos:  np.ndarray
    contact_surf: Optional[SurfacePatch]
    magnet_on:    bool = True


# ─────────────────────────────────────────────────────────────────────────────
# FK-based workspace
# ─────────────────────────────────────────────────────────────────────────────

class LegWorkspace:
    """
    Reachable EE positions for one leg, sampled via forward kinematics over the
    full joint-limit grid.  Expressed in the hip pivot frame so the cloud can be
    rigidly transformed to world frame as the body moves.

    FK chain (all angles in radians, axes from robot_original.xml):

        hip pivot  →  hip_pitch (Z axis)
                   →  hip link  (fixed offset to knee pivot)
                   →  knee      (diagonal axis in XY)
                   →  wrist     (same diagonal axis)
                   →  ee        (diagonal axis in YZ)
                   →  EE tip

    Link vectors (body frame of each joint, zero-angle):
        hip→knee  : hip_geom fromto endpoint  = (±0.03350, ±0.03350, 0.060)
        knee→wrist: knee_geom fromto endpoint = (±0.07280, ±0.07280, -0.04432)
        wrist→EE  : wrist_geom fromto endpoint= (±0.07425, ±0.07425, -0.18187)
        EE→tip    : ee_link_geom endpoint     = (0.066, 0, 0)  in EE local frame

    Signs per leg:
        FL: (+, +)   FR: (+, -)   BL: (-, +)   BR: (-, -)
    """

    # Fixed link vectors at zero angle, in each joint's local frame.
    # Axes and vectors read directly from robot_original.xml.
    _HIP_TO_KNEE: Dict[str, np.ndarray] = {
        'FL': np.array([ 0.03350,  0.03350, 0.060]),
        'FR': np.array([ 0.03350, -0.03350, 0.060]),
        'BL': np.array([-0.03350,  0.03350, 0.060]),
        'BR': np.array([-0.03350, -0.03350, 0.060]),
    }
    _KNEE_TO_WRIST: Dict[str, np.ndarray] = {
        'FL': np.array([ 0.07280,  0.07280, -0.04432]),
        'FR': np.array([ 0.07280, -0.07280, -0.04432]),
        'BL': np.array([-0.07280,  0.07280, -0.04432]),
        'BR': np.array([-0.07280, -0.07280, -0.04432]),
    }
    _WRIST_TO_EE: Dict[str, np.ndarray] = {
        'FL': np.array([ 0.07425,  0.07425, -0.18187]),
        'FR': np.array([ 0.07425, -0.07425, -0.18187]),
        'BL': np.array([-0.07425,  0.07425, -0.18187]),
        'BR': np.array([-0.07425, -0.07425, -0.18187]),
    }
    # EE tip in the EE joint local frame (same for all legs).
    _EE_TIP = np.array([0.066, 0.0, 0.0])

    # Joint axes in parent frame (unit vectors from robot_original.xml).
    _HIP_AXIS = np.array([0.0, 0.0, 1.0])   # Z — same for all legs
    _KNEE_AXIS: Dict[str, np.ndarray] = {
        'FL': np.array([ 0.7071068, -0.7071068, 0.0]),
        'FR': np.array([-0.7071068, -0.7071068, 0.0]),
        'BL': np.array([ 0.7071068,  0.7071068, 0.0]),
        'BR': np.array([-0.7071068,  0.7071068, 0.0]),
    }
    _WRIST_AXIS = _KNEE_AXIS   # same axis as knee per leg
    _EE_AXIS: Dict[str, np.ndarray] = {
        'FL': np.array([0.0, -0.7071068,  0.7071068]),
        'FR': np.array([0.0, -0.7071068, -0.7071068]),
        'BL': np.array([0.0,  0.7071068,  0.7071068]),
        'BR': np.array([0.0,  0.7071068, -0.7071068]),
    }

    def __init__(self, foot: str):
        self.foot = foot
        # points shape: (N, 3) in hip pivot frame
        self.points: np.ndarray = self._sample()
        # Precompute radial distances from hip for interior margin.
        self._radii: np.ndarray = np.linalg.norm(self.points, axis=1)
        self._r_max: float = float(self._radii.max())
        self._r_min: float = float(self._radii.min())
        print(f"[workspace] {foot}: {len(self.points)} FK points  "
              f"r=[{self._r_min:.3f}, {self._r_max:.3f}] m")

    # ── FK helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _rot(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        """Rodrigues rotation matrix."""
        k = axis / np.linalg.norm(axis)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)

    def _fk_tip(self, foot: str, q_hip: float, q_knee: float,
                q_wrist: float, q_ee: float) -> np.ndarray:
        """
        Forward kinematics from hip pivot → EE tip.
        Returns position in hip pivot frame.
        All angles in radians.
        """
        R_hip   = self._rot(self._HIP_AXIS, q_hip)
        p_knee  = R_hip @ self._HIP_TO_KNEE[foot]

        ax_k    = R_hip @ self._KNEE_AXIS[foot]
        R_knee  = R_hip @ self._rot(self._KNEE_AXIS[foot], q_knee)
        p_wrist = p_knee + R_knee @ self._KNEE_TO_WRIST[foot]

        R_wrist = R_knee @ self._rot(self._KNEE_AXIS[foot], q_wrist)
        p_ee    = p_wrist + R_wrist @ self._WRIST_TO_EE[foot]

        R_ee    = R_wrist @ self._rot(self._EE_AXIS[foot], q_ee)
        p_tip   = p_ee + R_ee @ self._EE_TIP

        return p_tip

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _sample(self) -> np.ndarray:
        """
        Sample EE positions over the full joint-limit grid.
        Returns (N, 3) array in hip pivot frame.
        """
        foot = self.foot
        hips   = np.linspace(*HIP_PITCH_RANGE,  WS_SAMPLES_HIP,   endpoint=True)
        knees  = np.linspace(*KNEE_RANGE,        WS_SAMPLES_KNEE,  endpoint=True)
        wrists = np.linspace(*WRIST_RANGE,       WS_SAMPLES_WRIST, endpoint=True)
        ees    = np.linspace(*EE_RANGE,           WS_SAMPLES_EE,    endpoint=True)

        pts = []
        for h in np.radians(hips):
            for k in np.radians(knees):
                for w in np.radians(wrists):
                    for e in np.radians(ees):
                        pts.append(self._fk_tip(foot, h, k, w, e))
        return np.array(pts)

    # ── Query ─────────────────────────────────────────────────────────────────

    def best_foothold(
        self,
        p_hip_world:  np.ndarray,
        R_body_world: np.ndarray,   # 3×3, rotates body→world  (R_bw.T)
        surf:         SurfacePatch,
        walk_dir:     np.ndarray,   # unit vector in world frame
    ) -> np.ndarray:
        """
        Return the workspace point (world frame, projected onto surf) that
        maximises progress along walk_dir while staying in the interior of the
        reachable workspace.

        Steps:
          1. Transform the whole point cloud to world frame.
          2. Project each point onto the surface plane.
          3. Discard points whose surface distance is too large (>5 cm off-plane).
          4. Discard points in the outer WS_INTERIOR_MARGIN radial shell.
          5. Among remaining points, return the one with maximum dot(walk_dir).
        """
        # 1. Cloud → world frame.
        pts_world = (R_body_world @ self.points.T).T + p_hip_world   # (N, 3)

        # 2. Signed distance to surface for each point.
        diff = pts_world - surf.origin   # (N, 3)
        sd   = diff @ surf.normal         # (N,)  signed distance

        # 3. Keep only near-surface points (within 5 cm of the plane).
        on_surf = np.abs(sd) < 0.05
        if not np.any(on_surf):
            # Fallback: take the closest point to the surface.
            on_surf = np.abs(sd) < np.abs(sd).min() + 1e-3

        pts_f   = pts_world[on_surf]                         # (M, 3)
        radii_f = self._radii[on_surf]                       # (M,)

        # 4. Interior margin: exclude outer radial shell.
        r_thresh = self._r_min + (1.0 - WS_INTERIOR_MARGIN) * (self._r_max - self._r_min)
        interior = radii_f <= r_thresh
        if np.any(interior):
            pts_f = pts_f[interior]

        # 5. Project onto surface and score by walk direction.
        pts_proj  = pts_f - (((pts_f - surf.origin) @ surf.normal)[:, None] * surf.normal)
        scores    = pts_proj @ walk_dir
        best_idx  = int(np.argmax(scores))
        return pts_proj[best_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion (w, x, y, z). Shepperd method."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s,                 (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1  = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta_0 = np.arccos(dot)
    theta   = theta_0 * t
    sin_t0  = np.sin(theta_0)
    return (np.sin(theta_0 - theta) / sin_t0) * q0 + (np.sin(theta) / sin_t0) * q1


def _quintic(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 10*t**3 - 15*t**4 + 6*t**5


def _classify_surface(surfaces: List[SurfacePatch], p: np.ndarray) -> SurfacePatch:
    best   = surfaces[0]
    best_d = abs(best.signed_distance(p))
    for surf in surfaces[1:]:
        d = abs(surf.signed_distance(p))
        if d < best_d:
            best   = surf
            best_d = d
    return best


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryPlanner
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryPlanner:
    """
    Generates per-foot world-frame IK targets for Sally's crawl gait.

    Parameters
    ----------
    surfaces : list of SurfacePatch
    initial_foot_pos : dict {foot: np.ndarray}
    body_R_world : callable () → np.ndarray   (3×3, body→world rotation)
    body_pos_world : callable () → np.ndarray  (3,)
    walk_dir : np.ndarray or None
        Desired walk direction in world frame (unit vector, projected onto
        surface).  Defaults to +X (forward).
    lift_height : float
    swing_duration : float
    demagnetize_hold : float
    magnetize_hold : float
    """

    def __init__(
        self,
        surfaces:          List[SurfacePatch],
        initial_foot_pos:  Dict[str, np.ndarray],
        body_R_world:      callable,
        body_pos_world:    callable,
        walk_dir:          Optional[np.ndarray] = None,
        # step_length removed — workspace planner doesn't use a fixed step scalar
        lift_height:       float = SWING_LIFT_HEIGHT,
        swing_duration:    float = SWING_DURATION,
        demagnetize_hold:  float = DEMAGNETIZE_HOLD,
        magnetize_hold:    float = MAGNETIZE_HOLD,
    ):
        self._surfaces         = surfaces
        self._body_R_world     = body_R_world
        self._body_pos_world   = body_pos_world
        self._lift_height      = lift_height
        self._swing_duration   = swing_duration
        self._demagnetize_hold = demagnetize_hold
        self._magnetize_hold   = magnetize_hold

        # Desired walk direction (unit, world frame).  Default = +X.
        if walk_dir is None:
            walk_dir = np.array([1.0, 0.0, 0.0])
        self._walk_dir: np.ndarray = walk_dir / np.linalg.norm(walk_dir)

        # Build FK workspaces once at startup (cheap — pure numpy).
        self._workspaces: Dict[str, LegWorkspace] = {
            foot: LegWorkspace(foot) for foot in GAIT_ORDER
        }

        # Per-leg state.
        self._legs: Dict[str, _LegState] = {}
        for foot, pos in initial_foot_pos.items():
            surf = _classify_surface(surfaces, pos)
            self._legs[foot] = _LegState(
                contact_pos  = pos.copy(),
                contact_surf = surf,
                magnet_on    = True,
            )

        # Gait sequencer state.
        self._gait_idx:  int       = 0
        self._phase:     GaitPhase = GaitPhase.STANCE_ALL
        self._phase_t:   float     = 0.0

        # Swing trajectory snapshot.
        self._swing_foot:    Optional[str]        = None
        self._swing_p_start: Optional[np.ndarray] = None
        self._swing_p_end:   Optional[np.ndarray] = None
        self._swing_q_start: Optional[np.ndarray] = None
        self._swing_q_end:   Optional[np.ndarray] = None
        self._swing_n_hat:   Optional[np.ndarray] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def step(
        self,
        dt:             float,
        contact_states: Dict[str, bool],
    ) -> List[FootTarget]:
        self._phase_t += dt
        self._advance_gait(contact_states)
        return self._build_targets()

    @property
    def phase(self) -> GaitPhase:
        return self._phase

    @property
    def swing_foot(self) -> Optional[str]:
        return self._swing_foot

    def magnet_states(self) -> Dict[str, bool]:
        return {foot: ls.magnet_on for foot, ls in self._legs.items()}

    def set_walk_dir(self, walk_dir: np.ndarray) -> None:
        """Update desired walk direction at runtime (unit vector, world frame)."""
        self._walk_dir = walk_dir / np.linalg.norm(walk_dir)

    # ── Gait state machine ────────────────────────────────────────────────────

    def _advance_gait(self, contact_states: Dict[str, bool]) -> None:
        swing_foot = GAIT_ORDER[self._gait_idx]

        if self._phase is GaitPhase.STANCE_ALL:
            self._legs[swing_foot].magnet_on = False
            self._swing_foot                 = swing_foot
            self._phase                      = GaitPhase.DEMAGNETIZE
            self._phase_t                    = 0.0

        elif self._phase is GaitPhase.DEMAGNETIZE:
            foot_lifted = not contact_states.get(swing_foot, True)
            if self._phase_t >= self._demagnetize_hold and foot_lifted:
                self._begin_swing(swing_foot)
                self._phase   = GaitPhase.SWING
                self._phase_t = 0.0

        elif self._phase is GaitPhase.SWING:
            if self._phase_t >= self._swing_duration:
                self._legs[swing_foot].contact_pos  = self._swing_p_end.copy()
                self._legs[swing_foot].contact_surf = _classify_surface(
                    self._surfaces, self._swing_p_end
                )
                self._legs[swing_foot].magnet_on    = True
                self._phase                         = GaitPhase.MAGNETIZE
                self._phase_t                       = 0.0

        elif self._phase is GaitPhase.MAGNETIZE:
            foot_landed = contact_states.get(swing_foot, False)
            if self._phase_t >= self._magnetize_hold and foot_landed:
                self._gait_idx   = (self._gait_idx + 1) % len(GAIT_ORDER)
                self._phase      = GaitPhase.STANCE_ALL
                self._phase_t    = 0.0
                self._swing_foot = None

    def _begin_swing(self, foot: str) -> None:
        leg   = self._legs[foot]
        p_dep = leg.contact_pos.copy()
        q_dep = leg.contact_surf.foot_quat() if leg.contact_surf else np.array([1., 0., 0., 0.])

        p_land    = self._plan_foothold(foot)
        surf_land = _classify_surface(self._surfaces, p_land)
        q_land    = surf_land.foot_quat()

        if leg.contact_surf is not None and surf_land is not leg.contact_surf:
            raise NotImplementedError(
                f"Corner-crossing swing for {foot}: departure surface normal "
                f"{leg.contact_surf.normal} ≠ landing surface normal {surf_land.normal}. "
                "Floor→wall arc blending is not yet implemented."
            )

        self._swing_n_hat   = leg.contact_surf.normal.copy() if leg.contact_surf else np.array([0., 0., 1.])
        self._swing_p_start = p_dep
        self._swing_p_end   = p_land
        self._swing_q_start = q_dep
        self._swing_q_end   = q_land

    # ── Foothold planning (FK workspace) ─────────────────────────────────────

    def _plan_foothold(self, foot: str) -> np.ndarray:
        """
        Select the landing foothold from the FK-sampled reachable workspace.

        1. Compute the hip pivot position in world frame.
        2. Get the rotation matrix that maps body frame → world frame.
        3. Ask LegWorkspace.best_foothold() to find the workspace point that
           maximises progress along self._walk_dir, projected onto the surface.
        """
        R_bw   = self._body_R_world()          # body→world (identity for fixed base)
        p_body = self._body_pos_world()         # world frame
        p_hip  = p_body + R_bw.T @ HIP_OFFSET_BODY[foot]

        leg  = self._legs[foot]
        surf = leg.contact_surf or _classify_surface(self._surfaces, leg.contact_pos)

        # Project walk_dir onto the surface plane so the step stays on-surface.
        wd      = self._walk_dir
        wd_surf = wd - np.dot(wd, surf.normal) * surf.normal
        norm    = np.linalg.norm(wd_surf)
        if norm > 1e-6:
            wd_surf /= norm
        else:
            wd_surf = wd   # surface normal parallel to walk direction (edge case)

        return self._workspaces[foot].best_foothold(
            p_hip_world  = p_hip,
            R_body_world = R_bw.T,   # rotates body frame → world frame
            surf         = surf,
            walk_dir     = wd_surf,
        )

    # ── Swing arc ─────────────────────────────────────────────────────────────

    def _swing_pos(self, t_norm: float) -> np.ndarray:
        s   = _quintic(t_norm)
        pos = (1.0 - s) * self._swing_p_start + s * self._swing_p_end
        pos = pos + self._swing_n_hat * self._lift_height * np.sin(np.pi * s)
        return pos

    def _swing_quat(self, t_norm: float) -> np.ndarray:
        s = _quintic(t_norm)
        return _quat_slerp(self._swing_q_start, self._swing_q_end, s)

    # ── Target construction ───────────────────────────────────────────────────

    def _build_targets(self) -> List[FootTarget]:
        targets: List[FootTarget] = []

        for foot in GAIT_ORDER:
            leg = self._legs[foot]

            if foot != self._swing_foot:
                targets.append(FootTarget(
                    foot       = foot,
                    pos_world  = leg.contact_pos.copy(),
                    quat_world = leg.contact_surf.foot_quat() if leg.contact_surf
                                 else np.array([1., 0., 0., 0.]),
                    weight_pos = WEIGHT_POS_STANCE,
                    weight_ori = WEIGHT_ORI_STANCE,
                    magnet_on  = leg.magnet_on,
                ))
            else:
                if self._phase is GaitPhase.SWING:
                    t_norm     = min(self._phase_t / self._swing_duration, 1.0)
                    pos        = self._swing_pos(t_norm)
                    quat       = self._swing_quat(t_norm)
                    weight_pos = WEIGHT_POS_SWING
                    weight_ori = WEIGHT_ORI_SWING
                elif self._phase is GaitPhase.MAGNETIZE:
                    pos        = self._swing_p_end.copy()
                    quat       = self._swing_q_end.copy()
                    weight_pos = WEIGHT_POS_TOUCHDOWN
                    weight_ori = WEIGHT_ORI_STANCE
                else:
                    # DEMAGNETIZE — hold at departure point.
                    pos  = self._swing_p_start.copy() if self._swing_p_start is not None \
                           else leg.contact_pos.copy()
                    quat = self._swing_q_start.copy() if self._swing_q_start is not None \
                           else (leg.contact_surf.foot_quat() if leg.contact_surf
                                 else np.array([1., 0., 0., 0.]))
                    weight_pos = WEIGHT_POS_STANCE
                    weight_ori = WEIGHT_ORI_STANCE

                targets.append(FootTarget(
                    foot       = foot,
                    pos_world  = pos,
                    quat_world = quat,
                    weight_pos = weight_pos,
                    weight_ori = weight_ori,
                    magnet_on  = leg.magnet_on,
                ))

        return targets