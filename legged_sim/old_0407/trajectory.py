"""
trajectory.py — Crawl-gait trajectory planner for Sally.

Gait order: FL → FR → BR → BL (clockwise).
State machine per swing foot: STANCE_ALL → DEMAGNETIZE → SWING → MAGNETIZE.
Swing arc: quintic s(t), pos(t) = lerp + n_hat * h * sin(π*s)
"""

import enum
from dataclasses import dataclass
import numpy as np


GAIT_ORDER: tuple[str, ...] = ('FL', 'FR', 'BR', 'BL')

HIP_OFFSET_BODY: dict[str, np.ndarray] = {
    'FL': np.array([ 0.0965,  0.092, 0.023]),
    'FR': np.array([ 0.0965, -0.092, 0.023]),
    'BL': np.array([-0.0965,  0.092, 0.023]),
    'BR': np.array([-0.0965, -0.092, 0.023]),
}

HIP_PITCH_RANGE  = (-45.0,  45.0)
KNEE_RANGE       = (-45.0,  45.0)
WRIST_RANGE      = (-67.5,  67.5)
EE_RANGE         = (-45.0,  45.0)

WS_SAMPLES_HIP     = 12
WS_SAMPLES_KNEE    = 12
WS_SAMPLES_WRIST   = 10
WS_SAMPLES_EE      =  8
WS_INTERIOR_MARGIN = 0.10

WEIGHT_POS_STANCE    = 10.0
WEIGHT_ORI_STANCE    =  5.0
WEIGHT_POS_SWING     =  2.0
WEIGHT_ORI_SWING     =  1.0
WEIGHT_POS_TOUCHDOWN = 10.0


@dataclass
class SurfacePatch:
    normal: np.ndarray
    origin: np.ndarray

    def project(self, p):
        return p - np.dot(p - self.origin, self.normal) * self.normal

    def signed_distance(self, p):
        return float(np.dot(p - self.origin, self.normal))

    def foot_quat(self):
        z_foot = -self.normal
        x_ref  = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(z_foot, x_ref)) > 0.99:
            x_ref = np.array([0.0, 1.0, 0.0])
        x_foot = np.cross(np.cross(z_foot, x_ref), z_foot)
        x_foot /= np.linalg.norm(x_foot)
        y_foot  = np.cross(z_foot, x_foot)
        return _rot_to_quat(np.column_stack([x_foot, y_foot, z_foot]))


@dataclass
class FootTarget:
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
    contact_pos:  np.ndarray
    contact_surf: SurfacePatch | None
    magnet_on:    bool = True


class LegWorkspace:
    """Reachable EE positions sampled via FK, expressed in hip pivot frame."""

    _HIP_TO_KNEE = {
        'FL': np.array([ 0.03350,  0.03350, 0.060]),
        'FR': np.array([ 0.03350, -0.03350, 0.060]),
        'BL': np.array([-0.03350,  0.03350, 0.060]),
        'BR': np.array([-0.03350, -0.03350, 0.060]),
    }
    _KNEE_TO_WRIST = {
        'FL': np.array([ 0.07280,  0.07280, -0.04432]),
        'FR': np.array([ 0.07280, -0.07280, -0.04432]),
        'BL': np.array([-0.07280,  0.07280, -0.04432]),
        'BR': np.array([-0.07280, -0.07280, -0.04432]),
    }
    _WRIST_TO_EE = {
        'FL': np.array([ 0.07425,  0.07425, -0.18187]),
        'FR': np.array([ 0.07425, -0.07425, -0.18187]),
        'BL': np.array([-0.07425,  0.07425, -0.18187]),
        'BR': np.array([-0.07425, -0.07425, -0.18187]),
    }
    _EE_TIP   = np.array([0.066, 0.0, 0.0])
    _HIP_AXIS = np.array([0.0, 0.0, 1.0])
    _KNEE_AXIS = {
        'FL': np.array([ 0.7071068, -0.7071068, 0.0]),
        'FR': np.array([-0.7071068, -0.7071068, 0.0]),
        'BL': np.array([ 0.7071068,  0.7071068, 0.0]),
        'BR': np.array([-0.7071068,  0.7071068, 0.0]),
    }
    _WRIST_AXIS = _KNEE_AXIS
    _EE_AXIS = {
        'FL': np.array([0.0, -0.7071068,  0.7071068]),
        'FR': np.array([0.0, -0.7071068, -0.7071068]),
        'BL': np.array([0.0,  0.7071068,  0.7071068]),
        'BR': np.array([0.0,  0.7071068, -0.7071068]),
    }

    def __init__(self, foot: str):
        self.foot   = foot
        self.points = self._sample()
        self._radii = np.linalg.norm(self.points, axis=1)
        self._r_max = float(self._radii.max())
        self._r_min = float(self._radii.min())
        print(f"[workspace] {foot}: {len(self.points)} FK points  r=[{self._r_min:.3f}, {self._r_max:.3f}] m")

    @staticmethod
    def _rot(axis, angle_rad):
        k = axis / np.linalg.norm(axis)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)

    def _fk_tip(self, foot, q_hip, q_knee, q_wrist, q_ee):
        R_hip   = self._rot(self._HIP_AXIS, q_hip)
        p_knee  = R_hip @ self._HIP_TO_KNEE[foot]
        R_knee  = R_hip @ self._rot(self._KNEE_AXIS[foot], q_knee)
        p_wrist = p_knee + R_knee @ self._KNEE_TO_WRIST[foot]
        R_wrist = R_knee @ self._rot(self._KNEE_AXIS[foot], q_wrist)
        p_ee    = p_wrist + R_wrist @ self._WRIST_TO_EE[foot]
        R_ee    = R_wrist @ self._rot(self._EE_AXIS[foot], q_ee)
        return p_ee + R_ee @ self._EE_TIP

    def _sample(self):
        foot = self.foot
        pts  = [
            self._fk_tip(foot, h, k, w, e)
            for h in np.radians(np.linspace(*HIP_PITCH_RANGE,  WS_SAMPLES_HIP))
            for k in np.radians(np.linspace(*KNEE_RANGE,        WS_SAMPLES_KNEE))
            for w in np.radians(np.linspace(*WRIST_RANGE,       WS_SAMPLES_WRIST))
            for e in np.radians(np.linspace(*EE_RANGE,           WS_SAMPLES_EE))
        ]
        return np.array(pts)

    def best_foothold(self, p_hip_world, R_body_world, surf, walk_dir):
        pts_world = (R_body_world @ self.points.T).T + p_hip_world
        sd        = (pts_world - surf.origin) @ surf.normal
        on_surf   = np.abs(sd) < 0.05
        if not np.any(on_surf):
            on_surf = np.abs(sd) < np.abs(sd).min() + 1e-3
        pts_f   = pts_world[on_surf]
        radii_f = self._radii[on_surf]
        r_thresh = self._r_min + (1.0 - WS_INTERIOR_MARGIN) * (self._r_max - self._r_min)
        interior = radii_f <= r_thresh
        if np.any(interior):
            pts_f = pts_f[interior]
        pts_proj = pts_f - (((pts_f - surf.origin) @ surf.normal)[:, None] * surf.normal)
        return pts_proj[int(np.argmax(pts_proj @ walk_dir))]


def _quintic(t):
    t = float(np.clip(t, 0.0, 1.0))
    return 10*t**3 - 15*t**4 + 6*t**5

def _classify_surface(surfaces, p):
    return min(surfaces, key=lambda s: abs(s.signed_distance(p)))

def _rot_to_quat(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])

def _quat_slerp(q0, q1, t):
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    theta_0 = np.arccos(dot)
    theta   = theta_0 * t
    sin_t0  = np.sin(theta_0)
    return (np.sin(theta_0 - theta) / sin_t0) * q0 + (np.sin(theta) / sin_t0) * q1


class TrajectoryPlanner:

    def __init__(self, surfaces, initial_foot_pos, body_R_world, body_pos_world,
                 lift_height, swing_duration, demagnetize_hold, magnetize_hold,
                 walk_dir=None):
        self._surfaces         = surfaces
        self._body_R_world     = body_R_world
        self._body_pos_world   = body_pos_world
        self._lift_height      = lift_height
        self._swing_duration   = swing_duration
        self._demagnetize_hold = demagnetize_hold
        self._magnetize_hold   = magnetize_hold
        self._walk_dir         = (np.array([1.0, 0.0, 0.0]) if walk_dir is None
                                  else walk_dir / np.linalg.norm(walk_dir))

        self._workspaces = {foot: LegWorkspace(foot) for foot in GAIT_ORDER}
        self._legs = {
            foot: _LegState(contact_pos=pos.copy(),
                            contact_surf=_classify_surface(surfaces, pos))
            for foot, pos in initial_foot_pos.items()
        }

        self._gait_idx     = 0
        self._phase        = GaitPhase.STANCE_ALL
        self._phase_t      = 0.0
        self._swing_foot   = None
        self._swing_p_start = None
        self._swing_p_end   = None
        self._swing_q_start = None
        self._swing_q_end   = None
        self._swing_n_hat   = None

    def step(self, dt, contact_states):
        self._phase_t += dt
        self._advance_gait(contact_states)
        return self._build_targets()

    @property
    def phase(self): return self._phase

    @property
    def swing_foot(self): return self._swing_foot

    def magnet_states(self):
        return {foot: ls.magnet_on for foot, ls in self._legs.items()}

    def _advance_gait(self, contact_states):
        swing_foot = GAIT_ORDER[self._gait_idx]

        if self._phase is GaitPhase.STANCE_ALL:
            self._legs[swing_foot].magnet_on = False
            self._swing_foot = swing_foot
            self._phase, self._phase_t = GaitPhase.DEMAGNETIZE, 0.0

        elif self._phase is GaitPhase.DEMAGNETIZE:
            if self._phase_t >= self._demagnetize_hold and not contact_states.get(swing_foot, True):
                self._begin_swing(swing_foot)
                self._phase, self._phase_t = GaitPhase.SWING, 0.0

        elif self._phase is GaitPhase.SWING:
            if self._phase_t >= self._swing_duration:
                self._legs[swing_foot].contact_pos  = self._swing_p_end.copy()
                self._legs[swing_foot].contact_surf = _classify_surface(self._surfaces, self._swing_p_end)
                self._legs[swing_foot].magnet_on    = True
                self._phase, self._phase_t = GaitPhase.MAGNETIZE, 0.0

        elif self._phase is GaitPhase.MAGNETIZE:
            if self._phase_t >= self._magnetize_hold and contact_states.get(swing_foot, False):
                self._gait_idx   = (self._gait_idx + 1) % len(GAIT_ORDER)
                self._swing_foot = None
                self._phase, self._phase_t = GaitPhase.STANCE_ALL, 0.0

    def _begin_swing(self, foot):
        leg    = self._legs[foot]
        q_dep  = leg.contact_surf.foot_quat() if leg.contact_surf else np.array([1., 0., 0., 0.])
        p_land = self._plan_foothold(foot)
        surf_land = _classify_surface(self._surfaces, p_land)

        if leg.contact_surf is not None and surf_land is not leg.contact_surf:
            raise NotImplementedError(f"Corner-crossing swing for {foot}: floor->wall arc not yet implemented.")

        self._swing_n_hat   = leg.contact_surf.normal.copy() if leg.contact_surf else np.array([0., 0., 1.])
        self._swing_p_start = leg.contact_pos.copy()
        self._swing_p_end   = p_land
        self._swing_q_start = q_dep
        self._swing_q_end   = surf_land.foot_quat()

    def _plan_foothold(self, foot):
        R_bw  = self._body_R_world()
        p_hip = self._body_pos_world() + R_bw @ HIP_OFFSET_BODY[foot]
        leg   = self._legs[foot]
        surf  = leg.contact_surf or _classify_surface(self._surfaces, leg.contact_pos)
        wd_surf = self._walk_dir - np.dot(self._walk_dir, surf.normal) * surf.normal
        norm    = np.linalg.norm(wd_surf)
        wd_surf = wd_surf / norm if norm > 1e-6 else self._walk_dir
        return self._workspaces[foot].best_foothold(p_hip, R_bw, surf, wd_surf)

    def _swing_pos(self, t_norm):
        s = _quintic(t_norm)
        return (1.0 - s) * self._swing_p_start + s * self._swing_p_end + self._swing_n_hat * self._lift_height * np.sin(np.pi * s)

    def _swing_quat(self, t_norm):
        return _quat_slerp(self._swing_q_start, self._swing_q_end, _quintic(t_norm))

    def _build_targets(self):
        targets = []
        for foot in GAIT_ORDER:
            leg = self._legs[foot]
            if foot != self._swing_foot:
                targets.append(FootTarget(
                    foot=foot, pos_world=leg.contact_pos.copy(),
                    quat_world=leg.contact_surf.foot_quat() if leg.contact_surf else np.array([1., 0., 0., 0.]),
                    weight_pos=WEIGHT_POS_STANCE, weight_ori=WEIGHT_ORI_STANCE,
                    magnet_on=leg.magnet_on,
                ))
            else:
                if self._phase is GaitPhase.SWING:
                    t_norm = min(self._phase_t / self._swing_duration, 1.0)
                    pos, quat = self._swing_pos(t_norm), self._swing_quat(t_norm)
                    wp, wo = WEIGHT_POS_SWING, WEIGHT_ORI_SWING
                elif self._phase is GaitPhase.MAGNETIZE:
                    pos, quat = self._swing_p_end.copy(), self._swing_q_end.copy()
                    wp, wo = WEIGHT_POS_TOUCHDOWN, WEIGHT_ORI_STANCE
                else:  # DEMAGNETIZE — hold at departure
                    pos  = self._swing_p_start.copy() if self._swing_p_start is not None else leg.contact_pos.copy()
                    quat = (self._swing_q_start.copy() if self._swing_q_start is not None
                            else (leg.contact_surf.foot_quat() if leg.contact_surf else np.array([1., 0., 0., 0.])))
                    wp, wo = WEIGHT_POS_STANCE, WEIGHT_ORI_STANCE
                targets.append(FootTarget(
                    foot=foot, pos_world=pos, quat_world=quat,
                    weight_pos=wp, weight_ori=wo, magnet_on=leg.magnet_on,
                ))
        return targets