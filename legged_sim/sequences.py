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


def hold_phase(face_axis=None):
    """Hold current EE position indefinitely with optional face_axis orientation.
    If face_axis is set, position is unconstrained and only orientation is solved.
    """
    def on_enter(ctx):
        ctx['phase_start'] = ctx['ee_pos_fn']().copy()

    def target(t, ctx):
        return None if face_axis is not None else ctx['phase_start']

    fa = (lambda t, ctx: np.array(face_axis)) if face_axis is not None else None
    return Phase("HOLD", duration=0.0, target_pos=target, face_axis=fa,
                 on_enter=on_enter, unbounded=True,
                 position_cost=0.0 if face_axis is not None else None,
                 orientation_cost=10.0 if face_axis is not None else None)


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
        hold_phase(face_axis=[1., 0., 0.]),  # EE local -Y → global +X
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