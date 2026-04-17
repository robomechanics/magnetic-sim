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
    """Extend EE by dx along global X from wherever the swing landed.
    face_axis: optional world-frame unit vector the EE local Z should point toward.
    unbounded: if True, holds at final position indefinitely.
    """
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
    """Hold the position where swing landed while driving EE orientation.
    
    position_cost > 0 anchors IK spatially so it doesn't throw joints to
    limits searching for orientation solutions in free space.
    orientation_cost drives EE local -Y toward face_axis.
    """
    def on_enter(ctx):
        ctx['phase_start'] = ctx['ee_pos_fn']().copy()

    def target(t, ctx):
        # Always return the swing landing position — never None.
        # position_cost controls how hard IK holds it; orientation wins
        # when ori_cost >> pos_cost but the arm stays near where it landed.
        return ctx['phase_start']

    fa = (lambda t, ctx: np.array(face_axis)) if face_axis is not None else None
    pc = position_cost    if face_axis is not None else None
    oc = orientation_cost if face_axis is not None else None
    return Phase("HOLD", duration=0.0, target_pos=target, face_axis=fa,
                 on_enter=on_enter, unbounded=True,
                 position_cost=pc, orientation_cost=oc)


# ── f2w phase factories (floor-to-wall transition) ──────────────────────────
#
# Split into three sequential phases so the ray fires only after IK has
# already rotated the EE to face the wall:
#
#   f2w_orient_phase  — hold raise_z above swing landing, rotate EE to face wall
#   f2w_measure_phase — ray fires NOW (EE facing wall), stores dist + reach target
#   f2w_reach_phase   — smooth translation along wall normal to contact point, hold
#
# All three share the same wall_normal and IK cost params for consistency.
#
# IMPORTANT — wall_normal convention:
#   wall_normal must point FROM the EE TOWARD the wall (i.e. in the direction
#   the ray should travel to hit the wall). It is also used as the IK face_axis
#   target (EE local -Y is driven to point along wall_normal).
#
#   Example: wall surface faces +X toward the robot (wall outward normal is +X),
#            the robot foot is approaching from the +X side:
#               wall_normal = [+1., 0., 0.]   ← ray travels in +X, hits wall ✓
#
#   Common mistake: using the wall's surface normal as seen from inside the wall
#   (i.e. [-1, 0, 0] when wall is at +X) → ray fires away from wall, always misses.

def f2w_orient_phase(raise_z=0.20, wall_normal=None,
                     duration=3.0, position_cost=8.0, orientation_cost=50.0):
    """
    Phase 1 of f2w: raise EE above the swing landing and rotate to face the wall.

    Holds the raised position (swing_landing + [0, 0, raise_z]) for `duration`
    seconds while IK drives EE local -Y toward wall_normal. Once this phase
    expires the EE should be pointing at the wall, ready for raycasting.

    Args:
        raise_z          : metres to raise above swing landing (default 0.20 m).
        wall_normal      : wall outward normal pointing TOWARD the robot (used as
                           face_axis). e.g. [-1., 0., 0.] for a wall at +X whose
                           face points toward the robot at -X.
        duration         : seconds to spend rotating; increase if IK is slow to converge.
        position_cost    : IK weight anchoring the raised hold position.
        orientation_cost : IK weight driving -Y toward wall_normal.
    """
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
    """
    Phase 2 of f2w: compute distance to wall and store reach target.

    duration=0 so the runner advances immediately after on_enter. Writes:
        ctx['f2w_wall_dist']    : float (metres), EE to wall inner face
        ctx['f2w_reach_target'] : np.array(3), world position foot_standoff
                                  short of the wall inner face

    DISTANCE METHOD (in priority order):
      1. Analytic (preferred) — if wall_face_pos is given, uses a dot-product
         projection onto the ray direction. Zero chance of a miss, no dependency
         on sim.py raycasting internals.
             dist = dot(wall_face_pos - ee_pos, ray_dir)
         For scene.xml: wall_face_pos=[0.500, 0, 0.5] (centre of inner face at X=0.500)
      2. mj_ray fallback — uses ctx['wall_dist_fn'] if wall_face_pos is None.
         Requires sim.py to register wall_dist_fn correctly.

    SIGN CONVENTION — wall_normal is the wall's outward normal TOWARD the robot:
        wall at +X, robot at -X  →  wall_normal = [-1., 0., 0.]
            face_axis = [-1,0,0]  →  EE local -Y → -X  →  EE body faces +X ✓
            ray_dir   = [+1,0,0]  →  -wall_normal, fires TOWARD wall        ✓
        face_axis and ray_dir intentionally differ in sign.

    Args:
        wall_normal   : wall outward normal pointing TOWARD the robot.
        wall_face_pos : any point on the wall's inner face in world coords.
                        e.g. [0.500, 0.0, 0.5] from scene.xml
                        (wall geom pos=[0.505,0,0.5], half-size x=0.005
                         → inner face at X = 0.505 - 0.005 = 0.500).
                        If None, falls back to ctx['wall_dist_fn'] (mj_ray).
        foot_standoff : metres short of wall face for IK target (default 8 mm).
    """
    def on_enter(ctx):
        current_pos = ctx['ee_pos_fn']().copy()
        ctx['phase_start'] = current_pos

        n_wall  = np.array(wall_normal, float)
        n_wall /= np.linalg.norm(n_wall)
        ray_dir = -n_wall   # from robot toward wall (opposite of wall outward normal)

        # ── Method 1: analytic distance (preferred) ─────────────────────────
        if wall_face_pos is not None:
            wfp  = np.array(wall_face_pos, float)
            dist = float(np.dot(wfp - current_pos, ray_dir))
            print(f"  [f2w:measure] analytic dist : {dist * 1000:.1f} mm")
            print(f"  [f2w:measure] wall_face_pos : {wfp}  EE: {current_pos.round(4)}")
        # ── Method 2: mj_ray fallback ────────────────────────────────────────
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
            effective_dist = dist - foot_standoff
            if effective_dist <= 0:
                print(f"  [f2w:measure] ⚠ dist ({dist*1000:.1f} mm) ≤ standoff "
                      f"({foot_standoff*1000:.1f} mm) — clamping to 1 mm")
                effective_dist = 0.001

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


def f2w_reach_phase(wall_normal=None, duration=2.0,
                    position_cost=8.0, orientation_cost=50.0):
    """
    Phase 3 of f2w: smoothly translate EE to the wall contact point and hold.

    Reads ctx['f2w_reach_target'] written by f2w_measure_phase and quintic-
    interpolates from the current EE position to that target over `duration`
    seconds, maintaining face_axis orientation throughout. Holds indefinitely
    once the reach completes.

    Args:
        wall_normal      : same unit vec used throughout f2w; maintained as face_axis.
        duration         : seconds for the reach translation (default 2.0 s).
        position_cost    : IK weight for position tracking during reach.
        orientation_cost : IK weight maintaining wall-facing during reach.
    """
    def on_enter(ctx):
        ctx['phase_start']     = ctx['ee_pos_fn']().copy()
        ctx['f2w_reach_start'] = ctx['phase_start'].copy()
        reach_target = ctx.get('f2w_reach_target', ctx['phase_start'])
        dist_to_wall = np.linalg.norm(reach_target - ctx['phase_start']) * 1000
        print(f"  [f2w:reach] start  : {ctx['f2w_reach_start'].round(4)}")
        print(f"  [f2w:reach] target : {reach_target.round(4)}"
              f"  ({dist_to_wall:.1f} mm, {duration:.1f}s)")

    def target(t, ctx):
        start = ctx['f2w_reach_start']
        end   = ctx.get('f2w_reach_target', start)
        s     = smooth(t, duration, 0., 1.)
        return start + s * (end - start)

    fa = (lambda t, ctx: np.array(wall_normal)) if wall_normal is not None else None
    return Phase(
        "F2W_REACH", duration=duration, target_pos=target, face_axis=fa,
        on_enter=on_enter, unbounded=True,
        position_cost=position_cost, orientation_cost=orientation_cost,
    )


# ── named sequences ─────────────────────────────────────────────────────
SEQUENCES = {
    # lift → swing 45° → reach 5 cm → drop
    "basic": [
        lift_phase(height=0.10, duration=3.0),
        swing_phase(angle_deg=-45.0, duration=1.5),
        reach_phase(dx=0.05, duration=1.5),
        drop_phase(duration=1.5),
    ],

    # lift → swing 45° → hold with EE local Z toward global +X
    "orient": [
        lift_phase(height=0.10, duration=3.0),
        swing_phase(angle_deg=-45.0, duration=1.5),
        hold_phase(face_axis=[-1., 0., 0.]),  # EE local -Y → global -X
    ],

    # lift → swing 45° → orient EE to face wall → measure dist → reach to wall
    #
    # wall_normal = wall's outward normal pointing TOWARD the robot.
    # scene.xml: wall geom at pos="0.505 0 0.5", inner face at X=0.500,
    #            robot approaches from -X side → wall faces -X → wall_normal=[-1,0,0]
    #
    # face_axis (orient/reach)  : wall_normal = [-1,0,0]
    #   → IK drives EE local -Y toward -X → EE body faces +X (toward wall) ✓
    # ray_dir (measure)         : -wall_normal = [+1,0,0]
    #   → fires from EE at X≈0.46 toward wall at X=0.500 ✓
    #
    # These two vectors have OPPOSITE signs by design — do not unify them.
    "f2w": [
        lift_phase(height=0.10, duration=3.0),
        swing_phase(angle_deg=-45.0, duration=1.5),
        f2w_orient_phase(
            raise_z=0.20,
            wall_normal=[-1., 0., 0.],  # wall outward normal → EE faces +X (toward wall)
            duration=3.0,
            position_cost=8.0,
            orientation_cost=50.0,
        ),
        f2w_measure_phase(
            wall_normal=[-1., 0., 0.],    # ray fires along -wall_normal = [+1,0,0] internally
            wall_face_pos=[0.500, 0., 0.5], # inner face: pos=0.505 - half_size=0.005 = 0.500
            foot_standoff=0.008,
            position_cost=8.0,
            orientation_cost=50.0,
        ),
        f2w_reach_phase(
            wall_normal=[-1., 0., 0.],  # face_axis maintained during reach + hold
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
        Returns (target_pos, face_axis). face_axis is None when not controlled.
        """
        self._ctx['ee_pos_fn']    = ctx['ee_pos_fn']
        self._ctx['hip_pivot_fn'] = ctx['hip_pivot_fn']

        # ── BUG 4 FIX ──────────────────────────────────────────────────────
        # wall_dist_fn was never refreshed after start(), so if sim.py created
        # the closure after calling runner.start(), f2w_measure would silently
        # skip the ray with "wall_dist_fn missing". Refresh it each step if
        # it exists in the live ctx.
        # ───────────────────────────────────────────────────────────────────
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
        if self.done:   return "DONE", 1.0
        if self.phase_idx < 0: return "IDLE", 0.0
        ph  = self.sequence[self.phase_idx]
        pct = np.clip((t - self.phase_t0) / ph.duration, 0., 1.) if ph.duration > 0 else 1.0
        return ph.name, pct