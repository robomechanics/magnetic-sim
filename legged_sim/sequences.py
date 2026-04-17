"""
Trajectory sequences for Sally FL foot.

Defines Phase primitives and named SEQUENCES.
SequenceRunner is the runtime object sim.py calls each step.

Usage (from sim.py):
    from sequences import SEQUENCES, SequenceRunner
    runner = SequenceRunner(SEQUENCES["orient"])
    ...
    target_pos, face_axis = runner.step(t, ctx)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


# ── quintic smooth-step ─────────────────────────────────────────────────
def smooth(t_rel, duration, q0, q1):
    s = np.clip(t_rel / duration, 0.0, 1.0)
    s = s**3 * (6*s**2 - 15*s + 10)
    return q0 + s * (q1 - q0)


# ── Phase primitive ─────────────────────────────────────────────────────
@dataclass
class Phase:
    name:      str
    duration:  float
    # target_pos(t_rel, ctx) -> (3,) world-frame EE position
    target_pos: Callable
    # face_axis(t_rel, ctx) -> (3,) unit vec or None
    face_axis:  Optional[Callable] = None
    # called once when this phase is entered; may write into ctx
    on_enter:   Optional[Callable] = None
    # if True, phase holds forever (no duration-based advance)
    unbounded:  bool = False
    # override IK costs; None → solver defaults
    position_cost:    Optional[float] = None
    orientation_cost: Optional[float] = None


# ── phase factories ─────────────────────────────────────────────────────
def lift_phase(height=0.10, duration=3.0):
    def target(t, ctx):
        return ctx['ee_home'] + np.array([0., 0., smooth(t, duration, 0., height)])
    return Phase("LIFT", duration, target)


def swing_phase(angle_deg=-45.0, duration=1.5, unbounded=False, face_axis=None):
    angle_rad = np.radians(angle_deg)

    def on_enter(ctx):
        ctx['phase_start'] = ctx['ee_pos_fn']().copy()
        ctx['hip_pivot']   = ctx['hip_pivot_fn']().copy()

    def target(t, ctx):
        r   = ctx['phase_start'] - ctx['hip_pivot']
        ang = smooth(t, duration, 0., angle_rad)
        c, s = np.cos(ang), np.sin(ang)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return ctx['hip_pivot'] + Rz @ r

    fa = (lambda t, ctx: np.array(face_axis)) if face_axis is not None else None
    return Phase("SWING", duration, target, face_axis=fa, on_enter=on_enter, unbounded=unbounded)


def reach_phase(dx=0.05, duration=1.5, face_axis=None, unbounded=False):
    """Extend EE by dx along global X from wherever the swing landed."""
    def on_enter(ctx):
        ctx['phase_start'] = ctx['ee_pos_fn']().copy()

    def target(t, ctx):
        return ctx['phase_start'] + np.array([smooth(t, duration, 0., dx), 0., 0.])

    fa = (lambda t, ctx: np.array(face_axis)) if face_axis is not None else None
    return Phase("REACH", duration, target, face_axis=fa, on_enter=on_enter, unbounded=unbounded)


def drop_phase(duration=1.5):
    def on_enter(ctx):
        ctx['phase_start'] = ctx['ee_pos_fn']().copy()

    def target(t, ctx):
        start = ctx['phase_start']
        z = smooth(t, duration, start[2], ctx['ee_home'][2])
        return np.array([start[0], start[1], z])

    return Phase("DROP", duration, target, on_enter=on_enter)


def hold_phase(face_axis=None, position_cost=8.0, orientation_cost=50.0):
    """Hold the swing landing position while driving EE orientation.

    position_cost anchors IK spatially so joints don't flee to limits
    searching for orientation solutions. orientation_cost drives EE
    local -Y toward face_axis.
    """
    def on_enter(ctx):
        ctx['phase_start'] = ctx['ee_pos_fn']().copy()

    def target(t, ctx):
        return ctx['phase_start']

    fa = (lambda t, ctx: np.array(face_axis)) if face_axis is not None else None
    pc = position_cost    if face_axis is not None else None
    oc = orientation_cost if face_axis is not None else None
    return Phase("HOLD", duration=0.0, target_pos=target, face_axis=fa,
                 on_enter=on_enter, unbounded=True,
                 position_cost=pc, orientation_cost=oc)


# ── f2w phase factories (floor-to-wall transition) ──────────────────────
#
# Three sequential phases:
#   f2w_orient_phase  — hold raise_z above swing landing, rotate EE to face wall
#   f2w_measure_phase — ray fires (EE now facing wall), stores reach target
#   (Phase 5: reach + hold — see TODO below)
#
# wall_normal convention: points FROM the EE TOWARD the wall.
#   face_axis = wall_normal  →  EE local -Y → wall_normal  →  EE faces wall ✓
#   ray_dir   = wall_normal  →  fires from EE toward wall                    ✓

def f2w_orient_phase(raise_z=0.20, wall_normal=None,
                     duration=3.0, position_cost=8.0, orientation_cost=50.0):
    """Phase 1 of f2w: raise EE above swing landing and rotate to face the wall."""
    def on_enter(ctx):
        swing_land = ctx['ee_pos_fn']().copy()
        ctx['phase_start']       = swing_land
        ctx['f2w_orient_target'] = swing_land + np.array([0., 0., raise_z])
        print(f"  [f2w:orient] swing landing  : {swing_land.round(4)}")
        print(f"  [f2w:orient] raised target  : {ctx['f2w_orient_target'].round(4)}"
              f"  (+{raise_z:.3f} m Z)")
        print(f"  [f2w:orient] rotating EE -Y → {wall_normal}  ({duration:.1f}s)")

    def target(t, ctx):
        return ctx['f2w_orient_target']

    fa = (lambda t, ctx: np.array(wall_normal)) if wall_normal is not None else None
    return Phase(
        "F2W_ORIENT", duration=duration, target_pos=target, face_axis=fa,
        on_enter=on_enter, unbounded=False,
        position_cost=position_cost, orientation_cost=orientation_cost,
    )


def f2w_measure_phase(wall_normal=None, wall_face_pos=None, foot_standoff=0.008,
                      position_cost=8.0, orientation_cost=50.0):
    """Phase 2 of f2w: compute distance to wall and store reach target in ctx.

    duration=0 so the runner advances immediately after on_enter. Writes:
        ctx['f2w_wall_dist']    : float (metres), EE to wall inner face
        ctx['f2w_reach_target'] : np.array(3), world position foot_standoff short of wall

    Distance methods (in priority order):
      1. Analytic (preferred) — if wall_face_pos is given:
             dist = dot(wall_face_pos - ee_pos, ray_dir)
      2. mj_ray fallback — uses ctx['wall_dist_fn'] if wall_face_pos is None.
    """
    def on_enter(ctx):
        current_pos = ctx['ee_pos_fn']().copy()
        ctx['phase_start'] = current_pos

        n_wall  = np.array(wall_normal, float)
        n_wall /= np.linalg.norm(n_wall)
        ray_dir = n_wall    # wall_normal points FROM ee TOWARD wall

        if wall_face_pos is not None:
            wfp  = np.array(wall_face_pos, float)
            dist = float(np.dot(wfp - current_pos, ray_dir))
            print(f"  [f2w:measure] analytic dist : {dist * 1000:.1f} mm")
            print(f"  [f2w:measure] wall_face_pos : {wfp}  EE: {current_pos.round(4)}")
        elif 'wall_dist_fn' in ctx:
            dist = ctx['wall_dist_fn'](ray_dir_override=ray_dir.tolist())
            print(f"  [f2w:measure] mj_ray dist   : {dist * 1000:.1f} mm  "
                  f"ray_dir={ray_dir.tolist()}")
        else:
            print("  [f2w:measure] ⚠ no wall_face_pos and no wall_dist_fn — cannot measure")
            ctx['f2w_wall_dist']    = np.inf
            ctx['f2w_reach_target'] = current_pos.copy()
            return

        ctx['f2w_wall_dist'] = dist

        if 0 < dist < np.inf:
            effective_dist = max(dist - foot_standoff, 0.001)
            if effective_dist <= foot_standoff:
                print(f"  [f2w:measure] ⚠ dist ({dist*1000:.1f} mm) ≤ standoff "
                      f"({foot_standoff*1000:.1f} mm) — clamping to 1 mm")
            ctx['f2w_reach_target'] = current_pos + effective_dist * ray_dir
            print(f"  [f2w:measure] standoff      : {foot_standoff*1000:.1f} mm  "
                  f"→ effective {effective_dist*1000:.1f} mm")
            print(f"  [f2w:measure] reach target  : {ctx['f2w_reach_target'].round(4)}")
        else:
            print(f"  [f2w:measure] ⚠ distance invalid (dist={dist:.4f}) — check:")
            print(f"                  EE at X={current_pos[0]:.4f}; "
                  f"wall inner face at X={wall_face_pos[0] if wall_face_pos else '?'}")
            print(f"                  EE must be at X < 0.500 and pointing toward wall")
            ctx['f2w_reach_target'] = current_pos.copy()

    def target(t, ctx):
        return ctx['phase_start']

    fa = (lambda t, ctx: np.array(wall_normal)) if wall_normal is not None else None
    return Phase(
        "F2W_MEASURE", duration=0.0, target_pos=target, face_axis=fa,
        on_enter=on_enter, unbounded=False,
        position_cost=position_cost, orientation_cost=orientation_cost,
    )

def f2w_reach_phase(wall_normal=[+1., 0., 0.], duration=2.0,
                    position_cost=8.0, orientation_cost=50.0):
    def on_enter(ctx):
        ctx['phase_start'] = ctx['ee_pos_fn']().copy()

    def target(t, ctx):
        return smooth(t, duration, ctx['phase_start'], ctx['f2w_reach_target'])

    fa = lambda t, ctx: np.array(wall_normal)
    return Phase("F2W_REACH", duration=duration, target_pos=target, face_axis=fa,
                 on_enter=on_enter, unbounded=True,
                 position_cost=position_cost, orientation_cost=orientation_cost)
# ── named sequences ─────────────────────────────────────────────────────
SEQUENCES = {
    # lift → swing 45° → reach 5 cm → drop
    "basic": [
        lift_phase(height=0.10, duration=3.0),
        swing_phase(angle_deg=-45.0, duration=1.5),
        reach_phase(dx=0.05, duration=1.5),
        drop_phase(duration=1.5),
    ],

    # lift → swing 45° → hold with EE local -Y toward global -X
    "orient": [
        lift_phase(height=0.10, duration=3.0),
        swing_phase(angle_deg=-45.0, duration=1.5),
        hold_phase(face_axis=[-1., 0., 0.]),
    ],

    # lift → swing 45° → orient EE to face wall → measure dist → (reach — TODO)
    #
    # wall_normal = [-1, 0, 0]: wall at +X, robot approaches from -X.
    # wall_face_pos = [0.500, 0, 0.5]: geom pos=0.505, half-size=0.005 → inner face X=0.500.
    "f2w": [
        lift_phase(height=0.10, duration=3.0),
        swing_phase(angle_deg=-45.0, duration=1.5),
        f2w_orient_phase(
            raise_z=0.20,
            wall_normal=[-1., 0., 0.],
            duration=3.0,
            position_cost=8.0,
            orientation_cost=50.0,
        ),
        f2w_measure_phase(
            wall_normal=[-1., 0., 0.],
            wall_face_pos=[0.500, 0., 0.5],
            foot_standoff=0.008,
            position_cost=8.0,
            orientation_cost=50.0,
        ),
        # Phase 5: reach to wall + hold
        f2w_reach_phase(
            wall_normal=[-1., 0., 0.],
            duration=2.0,
            position_cost=8.0,
            orientation_cost=50.0,
),
    ],
}


# ── runtime runner ───────────────────────────────────────────────────────
class SequenceRunner:
    """
    Tracks phase progression and returns (target_pos, face_axis) each step.

    ctx must be populated by sim.py before calling start():
        ctx['ee_home']      : np.array(3) — EE position at settle
        ctx['ee_pos_fn']    : callable() -> np.array(3), current EE world pos
        ctx['hip_pivot_fn'] : callable() -> np.array(3), current hip anchor world pos
        ctx['wall_dist_fn'] : callable(ray_dir_override=None) -> float (metres).
                              mj_ray closure in sim.py. Required by "f2w",
                              optional for all other sequences.
    """

    def __init__(self, sequence):
        self.sequence  = sequence
        self.phase_idx = -1
        self.phase_t0  = 0.0
        self.done      = False
        self._ctx      = {}

    @property
    def current_phase(self):
        if 0 <= self.phase_idx < len(self.sequence):
            return self.sequence[self.phase_idx]
        return None

    def start(self, t, ctx):
        """Call once after settle, before first step()."""
        self._ctx = ctx.copy()
        self._enter(0, t)

    def _enter(self, idx, t):
        self.phase_idx = idx
        self.phase_t0  = t
        ph = self.sequence[idx]
        if ph.on_enter:
            ph.on_enter(self._ctx)
        print(f"\n── phase {idx+1}/{len(self.sequence)}: {ph.name} ({ph.duration:.1f}s) ──")

    def step(self, t, ctx):
        """
        Call every IK step. Live ctx values (ee_pos_fn, hip_pivot_fn) must be current.
        Returns (target_pos, face_axis, position_cost, orientation_cost).
        """
        self._ctx['ee_pos_fn']    = ctx['ee_pos_fn']
        self._ctx['hip_pivot_fn'] = ctx['hip_pivot_fn']
        if 'wall_dist_fn' in ctx:
            self._ctx['wall_dist_fn'] = ctx['wall_dist_fn']

        if self.done or self.phase_idx < 0:
            return self._ctx.get('last_target', self._ctx['ee_home']), None, None, None

        ph    = self.sequence[self.phase_idx]
        t_rel = t - self.phase_t0

        target_pos = ph.target_pos(t_rel, self._ctx)
        face_axis  = ph.face_axis(t_rel, self._ctx) if ph.face_axis else None
        self._ctx['last_target'] = target_pos

        if t_rel >= ph.duration and not ph.unbounded:
            if self.phase_idx + 1 < len(self.sequence):
                self._enter(self.phase_idx + 1, t)
            else:
                self.done = True
                print("\n── sequence complete ──")

        return target_pos, face_axis, ph.position_cost, ph.orientation_cost

    def progress(self, t):
        """Returns (phase_name, pct 0-1) for telemetry."""
        if self.done:          return "DONE", 1.0
        if self.phase_idx < 0: return "IDLE", 0.0
        ph  = self.sequence[self.phase_idx]
        pct = np.clip((t - self.phase_t0) / ph.duration, 0., 1.) if ph.duration > 0 else 1.0
        return ph.name, pct