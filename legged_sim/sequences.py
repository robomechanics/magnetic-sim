"""
Trajectory sequences for Sally's feet (FL and FR floor-to-wall).

Usage (from sim.py):
    runner = SequenceRunner(SEQUENCES["f2w"])
    target_pos, face_axis, pos_cost, ori_cost = runner.step(t, ctx)

ctx keys supplied by sim.py:
    foot                                — label of the active swing foot, e.g. "FL" or "FR"
    ee_home, ee_pos_fn, hip_pivot_fn   — always required
    wall_dist_fn                        — required by f2w sequences
    magnet_disable_fn, magnet_enable_fn — optional
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


# ── data ───────────────────────────────────────────────────────────────────

@dataclass
class Phase:
    """One motion primitive in a sequence.

    target_pos(t_rel, ctx) -> (3,) world-frame EE position.
    face_axis(t_rel, ctx)  -> (3,) unit vec; IK aligns EE local -Y to it.
    unbounded: hold forever (ignore duration).
    position_cost / orientation_cost: override IK defaults (10.0 / 0.0).
    """
    name:             str
    duration:         float
    target_pos:       Callable
    face_axis:        Optional[Callable] = None
    on_enter:         Optional[Callable] = None
    unbounded:        bool               = False
    position_cost:    Optional[float]    = None
    orientation_cost: Optional[float]    = None


# ── helpers ─────────────────────────────────────────────────────────────────

def _smooth(t_rel, duration, q0, q1):
    """Quintic smooth-step from q0 to q1 over [0, duration]. Scalar or array."""
    s = np.clip(t_rel / duration, 0.0, 1.0)
    s = s**3 * (6*s**2 - 15*s + 10)
    return q0 + s * (q1 - q0)

def _fa_const(vec) -> Callable:
    arr = np.array(vec)
    return lambda t, ctx: arr


# ── phase factories ──────────────────────────────────────────────────────────

def lift_phase(height=0.10, duration=3.0) -> Phase:
    """Raise EE by `height` from ee_home. Disables the swing foot's magnet on entry."""
    def on_enter(ctx):
        if fn := ctx.get('magnet_disable_fn'):
            foot = ctx.get('foot', '?')
            fn(); print(f"  [magnet] {foot} magnet DISABLED")

    def target(t, ctx):
        return ctx['ee_home'] + np.array([0., 0., _smooth(t, duration, 0., height)])

    return Phase("LIFT", duration, target, on_enter=on_enter)


def swing_phase(angle_deg=-45.0, duration=1.5, unbounded=False, face_axis=None) -> Phase:
    """Arc EE around the hip pivot by `angle_deg` in the XY plane."""
    angle_rad = np.radians(angle_deg)

    def on_enter(ctx):
        ctx['phase_start'] = ctx['ee_pos_fn']().copy()
        ctx['hip_pivot']   = ctx['hip_pivot_fn']().copy()

    def target(t, ctx):
        r  = ctx['phase_start'] - ctx['hip_pivot']
        a  = _smooth(t, duration, 0., angle_rad)
        c, s = np.cos(a), np.sin(a)
        return ctx['hip_pivot'] + np.array([[c,-s,0],[s,c,0],[0,0,1]]) @ r

    fa = _fa_const(face_axis) if face_axis is not None else None
    return Phase("SWING", duration, target, face_axis=fa,
                 on_enter=on_enter, unbounded=unbounded)


def reach_phase(dx=0.05, duration=1.5, face_axis=None, unbounded=False) -> Phase:
    """Extend EE by `dx` along global X from swing landing. Enables the swing foot's magnet."""
    def on_enter(ctx):
        ctx['phase_start'] = ctx['ee_pos_fn']().copy()
        if fn := ctx.get('magnet_enable_fn'):
            foot = ctx.get('foot', '?')
            fn(); print(f"  [magnet] {foot} magnet ENABLED")

    def target(t, ctx):
        return ctx['phase_start'] + np.array([_smooth(t, duration, 0., dx), 0., 0.])

    fa = _fa_const(face_axis) if face_axis is not None else None
    return Phase("REACH", duration, target, face_axis=fa,
                 on_enter=on_enter, unbounded=unbounded)


def hold_phase(face_axis=None, position_cost=8.0, orientation_cost=50.0,
               mag_enable_delay=None) -> Phase:
    """Hold landing position; drive EE local -Y toward face_axis.

    position_cost anchors IK so joints don't flee limits seeking orientation.
    mag_enable_delay: seconds before enabling the magnet (None = skip).
    """
    def on_enter(ctx):
        ctx['phase_start']        = ctx['ee_pos_fn']().copy()
        ctx['hold_mag_triggered'] = False

    def target(t, ctx):
        if mag_enable_delay is not None and not ctx['hold_mag_triggered'] and t >= mag_enable_delay:
            if fn := ctx.get('magnet_enable_fn'):
                foot = ctx.get('foot', '?')
                fn(); print(f"  [magnet] {foot} magnet ENABLED  (after {mag_enable_delay:.1f}s hold)")
            ctx['hold_mag_triggered'] = True
        return ctx['phase_start']

    fa = _fa_const(face_axis) if face_axis is not None else None
    pc = position_cost    if face_axis is not None else None
    oc = orientation_cost if face_axis is not None else None
    return Phase("HOLD", duration=0.0, target_pos=target, face_axis=fa,
                 on_enter=on_enter, unbounded=True, position_cost=pc, orientation_cost=oc)


# ── f2w phase factories ───────────────────────────────────────────────────────
#
# wall_normal: FROM the EE TOWARD the wall.
#   ray_dir   =  wall_normal  →  EE → wall distance cast
#   face_axis = -wall_normal  →  IK drives EE local -Y → face_axis,
#               so EE local +Y (contact face) points toward wall

def f2w_orient_phase(raise_z=0.20, wall_normal=None, duration=3.0,
                     position_cost=20.0, orientation_cost=30.0) -> Phase:
    """Raise EE above swing landing and rotate to face the wall."""
    def on_enter(ctx):
        swing_land = ctx['ee_pos_fn']().copy()
        ctx['phase_start'] = swing_land
        # Use pre-lift Z (ee_home) as the vertical reference for the orient target.
        # Swing landing Z is unreliable — if the previous wall foot slipped during swing,
        # the body drops and drags the landing Z down with it. ee_home is captured
        # before lift when the foot is still grounded and the body is stable.
        ref_z = ctx['ee_home'][2]
        ctx['f2w_orient_target'] = np.array([swing_land[0], swing_land[1], ref_z + raise_z])
        print(f"  [f2w:orient] landing={swing_land.round(4)}  "
              f"target={ctx['f2w_orient_target'].round(4)}  "
              f"(ref_z={ref_z:.4f} +{raise_z:.3f}m)")

    def target(t, ctx):
        return ctx['f2w_orient_target']

    fa = (lambda t, ctx: -np.array(wall_normal)) if wall_normal is not None else None
    return Phase("F2W_ORIENT", duration=duration, target_pos=target, face_axis=fa,
                 on_enter=on_enter, position_cost=position_cost, orientation_cost=orientation_cost)


def f2w_measure_phase(wall_normal=None, wall_face_pos=None, foot_standoff=0.008,
                      wall_offset=0.02,
                      position_cost=8.0, orientation_cost=50.0) -> Phase:
    """Instant phase (duration=0): measure EE→wall dist, write ctx['f2w_reach_target'].

    Uses analytic dot-product if wall_face_pos given, else mj_ray via ctx['wall_dist_fn'].
    wall_offset: extra clearance (m) subtracted from reach on top of foot_standoff,
                 pulling the target away from the wall (default 0.02 = 2 cm).
    """
    def on_enter(ctx):
        current_pos = ctx['ee_pos_fn']().copy()
        ctx['phase_start'] = current_pos
        ray_dir = np.array(wall_normal, float)
        ray_dir /= np.linalg.norm(ray_dir)

        if wall_face_pos is not None:
            wfp  = np.array(wall_face_pos, float)
            dist = float(np.dot(wfp - current_pos, ray_dir))
            print(f"  [f2w:measure] analytic {dist*1000:.1f}mm  wfp={wfp}  ee={current_pos.round(4)}")
        elif 'wall_dist_fn' in ctx:
            dist = ctx['wall_dist_fn'](ray_dir_override=ray_dir.tolist())
            print(f"  [f2w:measure] mj_ray {dist*1000:.1f}mm  dir={ray_dir.tolist()}")
        else:
            print("  [f2w:measure] ⚠ no wall_face_pos and no wall_dist_fn")
            ctx['f2w_wall_dist'] = np.inf
            ctx['f2w_reach_target'] = current_pos.copy()
            return

        ctx['f2w_wall_dist'] = dist
        if 0 < dist < np.inf:
            eff = dist - foot_standoff - wall_offset
            if eff <= 0:
                # Already inside the wall_offset safety zone — drop wall_offset and reach
                # to foot_standoff distance from the wall instead of clamping to 1mm.
                eff = dist - foot_standoff
                print(f"  [f2w:measure] inside wall_offset zone — reaching to standoff only")
            if eff <= 0.001:
                print(f"  [f2w:measure] ⚠ dist {dist*1000:.1f}mm <= standoff — clamping to 1mm")
                eff = 0.001
            ctx['f2w_reach_target'] = current_pos + eff * ray_dir
            print(f"  [f2w:measure] standoff={foot_standoff*1000:.1f}mm  "
                  f"wall_offset={wall_offset*1000:.1f}mm  "
                  f"reach={ctx['f2w_reach_target'].round(4)}")
        else:
            print(f"  [f2w:measure] ⚠ invalid dist={dist:.4f}  EE.x={current_pos[0]:.4f}")
            ctx['f2w_reach_target'] = current_pos.copy()

    fa = (lambda t, ctx: -np.array(wall_normal)) if wall_normal is not None else None
    return Phase("F2W_MEASURE", duration=0.0, target_pos=lambda t, ctx: ctx['phase_start'],
                 face_axis=fa, on_enter=on_enter,
                 position_cost=position_cost, orientation_cost=orientation_cost)


def f2w_reach_phase(wall_normal=None, duration=2.0,
                    position_cost=8.0, orientation_cost=50.0) -> Phase:
    """Interpolate EE to wall contact point. Enables the swing foot's magnet when motion finishes."""
    def on_enter(ctx):
        ctx['phase_start']             = ctx['ee_pos_fn']().copy()
        ctx['f2w_reach_mag_triggered'] = False

    def target(t, ctx):
        if t >= duration and not ctx['f2w_reach_mag_triggered']:
            if fn := ctx.get('magnet_enable_fn'):
                foot = ctx.get('foot', '?')
                fn(); print(f"  [magnet] {foot} magnet ENABLED  (reach complete)")
            ctx['f2w_reach_mag_triggered'] = True
        return _smooth(t, duration, ctx['phase_start'], ctx['f2w_reach_target'])

    fa = (lambda t, ctx: -np.array(wall_normal)) if wall_normal is not None else None
    return Phase("F2W_REACH", duration=duration, target_pos=target, face_axis=fa,
                 on_enter=on_enter, unbounded=True,
                 position_cost=position_cost, orientation_cost=orientation_cost)


# ── sequences ────────────────────────────────────────────────────────────────

SEQUENCES = {
    # lift → swing 45° → hold EE local -Y toward global -X
    "orient": [
        lift_phase(height=0.10, duration=3.0),
        swing_phase(angle_deg=-45.0, duration=1.5),
        hold_phase(face_axis=[-1., 0., 0.], mag_enable_delay=5.0),
    ],

    # FL: lift → swing -45° → orient → measure → reach to wall (+X face)
    # wall_normal=[+1,0,0]: EE +Y contact face points toward wall
    # wall_face_pos=[0.500,0,0.5]: geom pos=0.505, half-size=0.005 → inner face X=0.500
    "f2w": [
        lift_phase(height=0.10, duration=3.0),
        swing_phase(angle_deg=-45.0, duration=1.5),
        f2w_orient_phase(raise_z=0.25,          wall_normal=[+1.,0.,0.], duration=3.0,
                         position_cost=20.0,     orientation_cost=30.0),
        f2w_measure_phase(wall_normal=[+1.,0.,0.], wall_face_pos=[0.500,0.,0.5],
                          foot_standoff=0.008,   wall_offset=0.02,
                          position_cost=8.0,     orientation_cost=50.0),
        f2w_reach_phase(wall_normal=[+1.,0.,0.], duration=2.0,
                        position_cost=8.0,       orientation_cost=50.0),
    ],

    # FR: mirror of f2w — swing is +45° (CW from above) to bring right foot toward +X wall.
    # Everything else (wall_normal, face_axis, wall_face_pos) is identical to f2w.
    "f2w_fr": [
        lift_phase(height=0.10, duration=3.0),
        swing_phase(angle_deg=+45.0, duration=1.5),   # ← mirrored from -45°
        f2w_orient_phase(raise_z=0.25,          wall_normal=[+1.,0.,0.], duration=3.0,
                         position_cost=20.0,     orientation_cost=30.0),
        f2w_measure_phase(wall_normal=[+1.,0.,0.], wall_face_pos=[0.500,0.,0.5],
                          foot_standoff=0.008,   wall_offset=0.02,
                          position_cost=8.0,     orientation_cost=50.0),
        f2w_reach_phase(wall_normal=[+1.,0.,0.], duration=2.0,
                        position_cost=8.0,       orientation_cost=50.0),
    ],

    # f2w_fr is the internal FR sub-sequence used by sim.py when running "f2w".
    # It is not meant to be run standalone.
}


# ── runner ───────────────────────────────────────────────────────────────────

class SequenceRunner:
    """Drives a list of Phase objects and returns IK targets each step.

    ctx contract (set before start()):
        foot                                — label of the active swing foot, e.g. "FL" or "FR"
        ee_home, ee_pos_fn, hip_pivot_fn   — always required
        wall_dist_fn                        — required by f2w sequences
        magnet_disable_fn, magnet_enable_fn — optional
    """

    def __init__(self, sequence):
        self.sequence  = sequence
        self.phase_idx = -1
        self.phase_t0  = 0.0
        self.done      = False
        self._ctx      = {}

    @property
    def current_phase(self) -> Optional[Phase]:
        if 0 <= self.phase_idx < len(self.sequence):
            return self.sequence[self.phase_idx]
        return None

    def start(self, t: float, ctx: dict):
        """Call once after settle, before first step()."""
        self._ctx = ctx.copy()
        self._enter(0, t)

    def step(self, t: float, ctx: dict):
        """Return (target_pos, face_axis, position_cost, orientation_cost)."""
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

    def progress(self, t: float):
        """Return (phase_name, pct 0–1) for telemetry."""
        if self.done:          return "DONE", 1.0
        if self.phase_idx < 0: return "IDLE", 0.0
        ph  = self.sequence[self.phase_idx]
        pct = np.clip((t - self.phase_t0) / ph.duration, 0., 1.) if ph.duration > 0 else 1.0
        return ph.name, pct

    def force_complete(self):
        """Mark the runner done without waiting for the last phase to expire.

        The last evaluated target is preserved in _ctx['last_target'] so
        subsequent step() calls return that frozen position — the foot stays
        locked at its wall contact point and IK treats it as a stance foot.
        Called by sim.py at the FL→FR handoff once the reach phase has dwelt long enough.
        """
        self.done = True
        print("\n── sequence force-completed (handing off to next foot) ──")

    def _enter(self, idx: int, t: float):
        self.phase_idx = idx
        self.phase_t0  = t
        ph = self.sequence[idx]
        if ph.on_enter:
            ph.on_enter(self._ctx)
        print(f"\n── phase {idx+1}/{len(self.sequence)}: {ph.name} ({ph.duration:.1f}s) ──")