"""
Trajectory sequences for Sally's feet (FL and FR floor-to-wall).

Usage (from sim.py):
    runner = SequenceRunner(SEQUENCES["f2w"])
    runner.start(t, ctx)           # ctx: PhaseContext
    target: IKTarget = runner.step(t, ee_pos_fn, hip_pivot_fn)

Key types
---------
IKTarget     — complete IK specification (position + optional orientation + costs).
               Replaces the old 4-tuple (target_pos, face_axis, pos_cost, ori_cost).
PhaseContext  — typed context injected into phase callbacks.
               Replaces the untyped ctx dict; cross-phase state lives in ctx.shared.
Phase         — motion primitive: step(t_rel, ctx) -> IKTarget, optional on_enter(ctx).
               Replaces the old Phase with separate target_pos / face_axis callables.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

# DONE: pull leg-agnostic contracted-pose builder from config so sequences
# only know joint *names*, never specific numeric values. This keeps the
# joint-keypoint framework reusable across legs and across robots.
from config import contracted_pose


# ── control mode ───────────────────────────────────────────────────────────────

class ControlMode(Enum):
    """Which control path an IKTarget activates in sim.py.

    C1 — IK with per-phase relaxed joint limits (jnt_range_override).
         Solver routes through configurations normally blocked by XML range=.
         Use for Step 2 / Step 3 of f2w where FL needs extended reach.

    C2 — Pure IK, default joint limits enforced.
         Standard mink QP solve. All existing phases use this mode.

    C3 — Direct joint PID, IK solve skipped entirely.
         ctrl_override dict written straight to PID position references.
         Use for Step 1 of f2w (contracted pose) and joint tuning.
    """
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"


# ── IK target ─────────────────────────────────────────────────────────────────

@dataclass
class IKTarget:
    """Complete IK specification for one step of the swing foot.

    face_axis: world-frame direction that EE local -Y should point toward.
               None means no orientation constraint.
    joint_targets: optional joint-space waypoint applied on top of the EE
               task. Maps joint name -> target qpos (radians for hinges).
               Joints not listed are unconstrained by this task. Empty/None
               disables the joint-space task entirely. The framework is
               leg-agnostic: any subset of model joints may be specified.
    joint_cost: scalar cost applied uniformly to every joint listed in
               joint_targets. Tune relative to position_cost / orientation_cost
               to balance Cartesian vs joint-space tracking.
    """
    position:         np.ndarray
    position_cost:    float                = 10.0
    face_axis:        Optional[np.ndarray] = None   # world dir for EE local -Y
    orientation_cost: float                = 0.0
    # ── DONE: joint-space directive (universal across joints / legs) ──────────
    joint_targets:    Optional[Dict[str, float]] = None
    joint_cost:       float                = 5.0

    # ── DONE: direct PID bypass — skips IK solve entirely ────────────────────
    # Maps joint name -> target qpos (rad). When set, sim.py skips the QP
    # solve and writes these values directly as PID position references.
    # All other joints hold their last ctrl_targets value. IK framework stays
    # intact — this is purely a bypass path for tuning without IK interference.
    ctrl_override:    Optional[Dict[str, float]] = None

    # ── control mode + per-phase joint limit override (T1) ────────────────────
    # control_mode identifies which sim.py dispatch path this target uses.
    # Set explicitly via the control1/2/3 named constructors — never inferred.
    control_mode:      ControlMode                              = ControlMode.C2
    # jnt_range_override: used by Control 1 only.
    # Maps joint name -> (lo_rad, hi_rad). IKSolver temporarily widens these
    # joints' ConfigurationLimit ranges for the duration of this IK solve,
    # then restores defaults. Cached by frozenset — only rebuilt on change.
    jnt_range_override: Optional[Dict[str, Tuple[float, float]]] = None

    # ── T2: per-phase body task target override ───────────────────────────────
    # 4×4 SE3 matrix (float64) commanding the main_frame body pose for this
    # tick. When set, IKSolver.solve() calls body_task.set_target(...) with
    # this matrix every tick, overriding the settled snapshot. When None,
    # the settled body SE3 (from record_stance) is restored each tick.
    # Store as np.ndarray to keep mink out of sequences.py; conversion to
    # mink.SE3 happens inside IKSolver.solve().
    body_target_se3:   Optional[np.ndarray]                    = None

    # ── convenience constructors ──────────────────────────────────────────────

    @staticmethod
    def control1(pos: np.ndarray,
                 jnt_range_override: Dict[str, Tuple[float, float]],
                 face_axis: Optional[np.ndarray] = None,
                 pos_cost: float = 8.0,
                 ori_cost: float = 0.0,
                 joint_targets: Optional[Dict[str, float]] = None,
                 joint_cost: float = 5.0,
                 body_target_se3: Optional[np.ndarray] = None) -> "IKTarget":
        """Control 1 — IK with per-phase relaxed joint limits."""
        return IKTarget(
            position=np.asarray(pos, float),
            position_cost=pos_cost,
            face_axis=np.asarray(face_axis, float) if face_axis is not None else None,
            orientation_cost=ori_cost,
            joint_targets=joint_targets,
            joint_cost=joint_cost,
            jnt_range_override=dict(jnt_range_override),
            control_mode=ControlMode.C1,
            body_target_se3=body_target_se3,
        )

    @staticmethod
    def control2(pos: np.ndarray,
                 face_axis: Optional[np.ndarray] = None,
                 pos_cost: float = 10.0,
                 ori_cost: float = 0.0,
                 joint_targets: Optional[Dict[str, float]] = None,
                 joint_cost: float = 5.0,
                 body_target_se3: Optional[np.ndarray] = None) -> "IKTarget":
        """Control 2 — pure IK, default joint limits."""
        return IKTarget(
            position=np.asarray(pos, float),
            position_cost=pos_cost,
            face_axis=np.asarray(face_axis, float) if face_axis is not None else None,
            orientation_cost=ori_cost,
            joint_targets=joint_targets,
            joint_cost=joint_cost,
            control_mode=ControlMode.C2,
            body_target_se3=body_target_se3,
        )

    @staticmethod
    def control3(pos: np.ndarray,
                 joint_targets: Dict[str, float]) -> "IKTarget":
        """Control 3 — direct joint PID, IK solve skipped entirely."""
        return IKTarget(
            position=np.asarray(pos, float),
            ctrl_override=dict(joint_targets),
            control_mode=ControlMode.C3,
        )

    @staticmethod
    def position_only(pos: np.ndarray, cost: float = 10.0) -> "IKTarget":
        return IKTarget(position=np.asarray(pos, float), position_cost=cost,
                        control_mode=ControlMode.C2)

    @staticmethod
    def with_orientation(pos: np.ndarray, face_axis: np.ndarray,
                         pos_cost: float = 8.0,
                         ori_cost: float = 50.0) -> "IKTarget":
        return IKTarget(
            position=np.asarray(pos, float),
            position_cost=pos_cost,
            face_axis=np.asarray(face_axis, float),
            orientation_cost=ori_cost,
            control_mode=ControlMode.C2,
        )

    @staticmethod
    def with_joint_targets(pos: np.ndarray,
                           joint_targets: Dict[str, float],
                           joint_cost: float = 5.0,
                           pos_cost: float = 1.0,
                           face_axis: Optional[np.ndarray] = None,
                           ori_cost: float = 0.0) -> "IKTarget":
        """C2: EE position + joint-space waypoint blended via IK."""
        return IKTarget(
            position=np.asarray(pos, float),
            position_cost=pos_cost,
            face_axis=np.asarray(face_axis, float) if face_axis is not None else None,
            orientation_cost=ori_cost,
            joint_targets=dict(joint_targets),
            joint_cost=joint_cost,
            control_mode=ControlMode.C2,
        )


# ── phase context ──────────────────────────────────────────────────────────────

@dataclass
class PhaseContext:
    """Typed context injected into every phase callback.

    ee_pos_fn and hip_pivot_fn are refreshed each step by SequenceRunner.step().
    joint_qpos_fn returns the current qpos value (rad/m) for a named joint;
        used by phases that capture on-entry joint values to interpolate from.
    shared is a free dict for cross-phase communication (e.g. f2w_measure → f2w_reach).
    """
    foot:              str
    ee_home:           np.ndarray
    ee_pos_fn:         Callable[[], np.ndarray]
    hip_pivot_fn:      Callable[[], np.ndarray]
    wall_dist_fn:      Optional[Callable]              = None
    magnet_disable_fn: Optional[Callable]              = None
    magnet_enable_fn:  Optional[Callable]              = None
    # ── DONE: joint-space readback (set by sim.py:_start_runner_for) ──────────
    joint_qpos_fn:     Optional[Callable[[str], float]] = None
    shared:            dict                            = field(default_factory=dict)

    # ── convenience methods ───────────────────────────────────────────────────

    def ee_pos(self) -> np.ndarray:
        return self.ee_pos_fn()

    def hip_pivot(self) -> np.ndarray:
        return self.hip_pivot_fn()

    def joint_qpos(self, name: str) -> float:
        if self.joint_qpos_fn is None:
            raise RuntimeError("joint_qpos_fn not wired into PhaseContext")
        return self.joint_qpos_fn(name)

    def magnet_disable(self):
        if self.magnet_disable_fn:
            self.magnet_disable_fn()
            print(f"  [magnet] {self.foot} DISABLED")

    def magnet_enable(self):
        if self.magnet_enable_fn:
            self.magnet_enable_fn()
            print(f"  [magnet] {self.foot} ENABLED")


# ── phase ──────────────────────────────────────────────────────────────────────

@dataclass
class Phase:
    """One motion primitive in a sequence.

    step(t_rel, ctx) -> IKTarget   called every IK tick.
    on_enter(ctx)                  called once when this phase becomes active.
    unbounded: if True, the phase holds after duration expires (never auto-advances).
    """
    name:      str
    duration:  float
    step:      Callable[[float, PhaseContext], IKTarget]
    on_enter:  Optional[Callable[[PhaseContext], None]] = None
    unbounded: bool = False


# ── smooth interpolation ───────────────────────────────────────────────────────

def _smooth(t_rel: float, duration: float, q0, q1):
    """Quintic smooth-step from q0 to q1 over [0, duration]. Scalar or array."""
    s = np.clip(t_rel / duration, 0.0, 1.0)
    s = s**3 * (6*s**2 - 15*s + 10)
    return q0 + s * (q1 - q0)


# ── floor-motion phase factories ───────────────────────────────────────────────

def lift_phase(height: float = 0.10, duration: float = 3.0,
               joint_targets: Optional[Dict[str, float]] = None,
               joint_cost: float = 5.0,
               position_cost: float = 10.0) -> Phase:
    """Raise EE straight up by `height` from its settle position. Disables
    swing magnet on entry.

    DONE: optionally drives a set of named joints to target qpos values over
    the same duration via a quintic smooth-step. Each joint's start value is
    captured from live qpos on phase entry, so the motion always begins from
    wherever the chain currently sits.

    Tips
    ----
    - To let joints dominate (the contracted-pose case), pass a low
      `position_cost` (e.g. 1.0). The EE target then acts as a soft hint and
      the chain folds primarily under the joint task.
    - To keep the EE strict and add a mild joint bias, pass position_cost=10.0
      with a small joint_cost (e.g. 1–2) — the joint targets become a tie-breaker.
    - The framework is universal: any joint name(s) accepted by the model may
      be specified. Not just knee/wrist/EE.
    """
    start  = np.zeros(3)
    q0     : Dict[str, float] = {}      # captured on entry, per active joint

    def on_enter(ctx: PhaseContext):
        nonlocal start, q0
        start = ctx.ee_pos().copy()
        ctx.magnet_disable()
        q0 = {}
        if joint_targets:
            for jname in joint_targets:
                try:
                    q0[jname] = ctx.joint_qpos(jname)
                except Exception as e:
                    print(f"  [lift] WARN could not read joint '{jname}': {e}")
            print(f"  [lift] joint path  start={ {k: round(v,3) for k,v in q0.items()} }"
                  f"  goal={ {k: round(v,3) for k,v in joint_targets.items()} }")

    def step(t: float, ctx: PhaseContext) -> IKTarget:
        target = start.copy()
        target[2] += _smooth(t, duration, 0.0, height)

        if joint_targets and q0:
            joint_now = {jn: float(_smooth(t, duration, q0[jn], joint_targets[jn]))
                         for jn in q0}
            return IKTarget(
                position=target,
                position_cost=position_cost,
                joint_targets=joint_now,
                joint_cost=joint_cost,
            )
        return IKTarget.position_only(target, position_cost)

    return Phase("LIFT", duration, step, on_enter=on_enter)


def swing_phase(angle_deg: float = -45.0, duration: float = 1.5,
                unbounded: bool = False,
                face_axis: Optional[np.ndarray] = None) -> Phase:
    """Arc EE around the hip pivot by `angle_deg` in the XY plane."""
    angle_rad = np.radians(angle_deg)
    fa = np.asarray(face_axis, float) if face_axis is not None else None
    start = np.zeros(3)
    pivot = np.zeros(3)

    def on_enter(ctx: PhaseContext):
        nonlocal start, pivot
        start = ctx.ee_pos().copy()
        pivot = ctx.hip_pivot().copy()

    def step(t: float, ctx: PhaseContext) -> IKTarget:
        r    = start - pivot
        a    = _smooth(t, duration, 0.0, angle_rad)
        c, s = np.cos(a), np.sin(a)
        pos  = pivot + np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ r
        if fa is not None:
            return IKTarget.with_orientation(pos, fa)
        return IKTarget.position_only(pos)

    return Phase("SWING", duration, step, on_enter=on_enter, unbounded=unbounded)


def reach_phase(dx: float = 0.05, duration: float = 1.5,
                face_axis: Optional[np.ndarray] = None,
                unbounded: bool = False) -> Phase:
    """Extend EE by `dx` along global X. Enables swing magnet on entry."""
    fa = np.asarray(face_axis, float) if face_axis is not None else None
    start = np.zeros(3)

    def on_enter(ctx: PhaseContext):
        nonlocal start
        start = ctx.ee_pos().copy()
        ctx.magnet_enable()

    def step(t: float, ctx: PhaseContext) -> IKTarget:
        pos = start + np.array([_smooth(t, duration, 0.0, dx), 0.0, 0.0])
        if fa is not None:
            return IKTarget.with_orientation(pos, fa)
        return IKTarget.position_only(pos)

    return Phase("REACH", duration, step, on_enter=on_enter, unbounded=unbounded)


def hold_phase(face_axis: Optional[np.ndarray] = None,
               position_cost: float = 8.0,
               orientation_cost: float = 50.0,
               mag_enable_delay: Optional[float] = None) -> Phase:
    """Hold landing position. Optionally aligns EE local -Y toward face_axis.

    mag_enable_delay: seconds after entry before enabling the swing magnet. None = skip.
    position_cost anchors IK so joints do not drift while the orientation task runs.
    """
    fa = np.asarray(face_axis, float) if face_axis is not None else None
    start         = np.zeros(3)
    mag_triggered = [False]

    def on_enter(ctx: PhaseContext):
        nonlocal start
        start              = ctx.ee_pos().copy()
        mag_triggered[0]   = False

    def step(t: float, ctx: PhaseContext) -> IKTarget:
        if mag_enable_delay is not None and not mag_triggered[0] and t >= mag_enable_delay:
            ctx.magnet_enable()
            mag_triggered[0] = True
        if fa is not None:
            return IKTarget.with_orientation(start, fa, position_cost, orientation_cost)
        return IKTarget.position_only(start, position_cost)

    return Phase("HOLD", duration=0.0, step=step, on_enter=on_enter, unbounded=True)


# ── floor-to-wall phase factories ──────────────────────────────────────────────
#
# Convention: wall_normal points FROM the EE TOWARD the wall.
#   face_axis = -wall_normal  →  IK drives EE local -Y → face_axis,
#               so EE local +Y (contact face) points toward the wall.

def f2w_orient_phase(raise_z: float = 0.20,
                     wall_normal: Optional[np.ndarray] = None,
                     duration: float = 3.0,
                     position_cost: float = 20.0,
                     orientation_cost: float = 30.0) -> Phase:
    """Raise EE above swing landing and rotate to face the wall."""
    fa           = -np.asarray(wall_normal, float) if wall_normal is not None else None
    orient_target = np.zeros(3)

    def on_enter(ctx: PhaseContext):
        nonlocal orient_target
        land          = ctx.ee_pos().copy()
        # Use ee_home Z as the vertical reference — swing landing Z is unreliable
        # if the body dropped during swing (e.g. the wall foot slipped).
        ref_z         = ctx.ee_home[2]
        orient_target = np.array([land[0], land[1], ref_z + raise_z])
        print(f"  [f2w:orient] landing={land.round(4)}  "
              f"target={orient_target.round(4)}  (ref_z={ref_z:.4f} +{raise_z:.3f}m)")

    def step(t: float, ctx: PhaseContext) -> IKTarget:
        if fa is not None:
            return IKTarget.with_orientation(orient_target, fa, position_cost, orientation_cost)
        return IKTarget.position_only(orient_target, position_cost)

    return Phase("F2W_ORIENT", duration, step, on_enter=on_enter)


def f2w_measure_phase(wall_normal: Optional[np.ndarray] = None,
                      wall_face_pos: Optional[np.ndarray] = None,
                      foot_standoff: float = 0.008,
                      wall_offset: float = 0.02,
                      position_cost: float = 8.0,
                      orientation_cost: float = 50.0) -> Phase:
    """Instant phase (duration=0): measure EE→wall distance, write ctx.shared['f2w_reach_target'].

    Uses analytic dot-product projection if wall_face_pos is given,
    otherwise falls back to mj_ray via ctx.wall_dist_fn.
    wall_offset: extra clearance subtracted from the reach distance on top of foot_standoff.
    """
    fa       = -np.asarray(wall_normal, float) if wall_normal is not None else None
    hold_pos = np.zeros(3)

    def on_enter(ctx: PhaseContext):
        nonlocal hold_pos
        current   = ctx.ee_pos().copy()
        hold_pos  = current
        ray_dir   = np.asarray(wall_normal, float)
        ray_dir  /= np.linalg.norm(ray_dir)

        if wall_face_pos is not None:
            wfp  = np.asarray(wall_face_pos, float)
            dist = float(np.dot(wfp - current, ray_dir))
            print(f"  [f2w:measure] analytic {dist*1000:.1f} mm  "
                  f"wfp={wfp}  ee={current.round(4)}")
        elif ctx.wall_dist_fn is not None:
            dist = ctx.wall_dist_fn(ray_dir_override=ray_dir.tolist())
            print(f"  [f2w:measure] mj_ray {dist*1000:.1f} mm  dir={ray_dir.tolist()}")
        else:
            print("  [f2w:measure] ⚠ no wall_face_pos and no wall_dist_fn")
            ctx.shared['f2w_reach_target'] = current.copy()
            return

        if 0 < dist < np.inf:
            eff = dist - foot_standoff - wall_offset
            if eff <= 0:
                # Inside the safety zone — reach to standoff only, ignore wall_offset.
                eff = dist - foot_standoff
                print("  [f2w:measure] inside wall_offset zone — reaching to standoff only")
            if eff <= 0.001:
                print(f"  [f2w:measure] ⚠ dist {dist*1000:.1f} mm ≤ standoff — clamping to 1 mm")
                eff = 0.001
            ctx.shared['f2w_reach_target'] = current + eff * ray_dir
            print(f"  [f2w:measure] standoff={foot_standoff*1000:.1f} mm  "
                  f"wall_offset={wall_offset*1000:.1f} mm  "
                  f"reach={ctx.shared['f2w_reach_target'].round(4)}")
        else:
            print(f"  [f2w:measure] ⚠ invalid dist={dist:.4f}  EE.x={current[0]:.4f}")
            ctx.shared['f2w_reach_target'] = current.copy()

    def step(t: float, ctx: PhaseContext) -> IKTarget:
        # Hold position during the zero-duration measure phase.
        if fa is not None:
            return IKTarget.with_orientation(hold_pos, fa, position_cost, orientation_cost)
        return IKTarget.position_only(hold_pos, position_cost)

    return Phase("F2W_MEASURE", duration=0.0, step=step, on_enter=on_enter)


def f2w_reach_phase(wall_normal: Optional[np.ndarray] = None,
                    duration: float = 2.0,
                    position_cost: float = 8.0,
                    orientation_cost: float = 50.0) -> Phase:
    """Smooth-step EE from orient position to wall contact point.

    Reads ctx.shared['f2w_reach_target'] written by the preceding f2w_measure_phase.
    Enables the swing magnet once the motion completes.
    """
    fa            = -np.asarray(wall_normal, float) if wall_normal is not None else None
    start         = np.zeros(3)
    mag_triggered = [False]

    def on_enter(ctx: PhaseContext):
        nonlocal start
        start            = ctx.ee_pos().copy()
        mag_triggered[0] = False

    def step(t: float, ctx: PhaseContext) -> IKTarget:
        if t >= duration and not mag_triggered[0]:
            ctx.magnet_enable()
            mag_triggered[0] = True
        reach_target = ctx.shared.get('f2w_reach_target', start)
        pos = _smooth(t, duration, start, reach_target)
        if fa is not None:
            return IKTarget.with_orientation(pos, fa, position_cost, orientation_cost)
        return IKTarget.position_only(pos, position_cost)

    return Phase("F2W_REACH", duration, step, on_enter=on_enter, unbounded=True)


def joint_phase(joint_targets: Dict[str, float],
                duration: float = 3.0,
                height: float = 0.0,
                disable_magnet: bool = False,
                enable_magnet: bool = False,
                unbounded: bool = False) -> Phase:
    """Drive named joints directly to target qpos via PID, bypassing IK entirely.

    Universal drop-in for any position in any sequence. Can replace or precede
    IK-based phases — the ctrl_override flag in IKTarget is checked every tick
    in sim.py regardless of which sequence is running.

    Parameters
    ----------
    joint_targets   joint name -> target qpos (rad). Start values captured from
                    live qpos on phase entry and quintic-smooth-stepped to targets.
    duration        phase duration in seconds.
    height          optional vertical EE lift (m) smooth-stepped alongside joints.
                    The EE position in IKTarget is a dummy when ctrl_override is
                    set — but height here drives the hip/knee combination via the
                    same PID bypass, not via IK. Set to 0 (default) to fold in place.
    disable_magnet  if True, disables swing foot magnet on entry.
    enable_magnet   if True, enables swing foot magnet on entry.
    unbounded       if True, holds at final pose indefinitely after duration.
    """
    q0: Dict[str, float] = {}
    ee_start = np.zeros(3)

    def on_enter(ctx: PhaseContext):
        nonlocal q0, ee_start
        ee_start = ctx.ee_pos().copy()
        if disable_magnet:
            ctx.magnet_disable()
        if enable_magnet:
            ctx.magnet_enable()
        q0 = {}
        for jname in joint_targets:
            try:
                q0[jname] = ctx.joint_qpos(jname)
            except Exception as e:
                print(f"  [joint_phase] WARN could not read joint '{jname}': {e}")
        print(f"  [joint_phase] direct PID — "
              f"height={height:.3f}m  "
              f"start={ {k: round(v, 3) for k, v in q0.items()} } "
              f"goal={ {k: round(v, 3) for k, v in joint_targets.items()} }")

    def step(t: float, ctx: PhaseContext) -> IKTarget:
        override = {jn: float(_smooth(t, duration, q0[jn], joint_targets[jn]))
                    for jn in q0}
        # EE position is carried for telemetry / pos_err display only.
        # sim.py does NOT feed it to the QP when control_mode == C3.
        ee_target = ee_start.copy()
        ee_target[2] += _smooth(t, duration, 0.0, height)
        return IKTarget.control3(pos=ee_target, joint_targets=override)

    return Phase("JOINT", duration, step, on_enter=on_enter, unbounded=unbounded)


# ── sequences ─────────────────────────────────────────────────────────────────

SEQUENCES = {
    # lift → swing 45° → hold EE local -Y toward global -X
    "orient": [
        lift_phase(height=0.10, duration=3.0),
        swing_phase(angle_deg=-45.0, duration=1.5),
        hold_phase(face_axis=[-1.0, 0.0, 0.0], mag_enable_delay=5.0),
    ],

    # FL: lift → swing -45° → orient → measure → reach (+X wall)
    # wall_normal=[+1,0,0]: EE +Y contact face points toward wall
    # wall_face_pos=[0.500,0,0.5]: geom pos=0.505, half-size=0.005 → inner face X=0.500
    "f2w": [
        lift_phase(height=0.10, duration=3.0),
        swing_phase(angle_deg=-45.0, duration=1.5),
        f2w_orient_phase(raise_z=0.40,       wall_normal=[+1.0, 0.0, 0.0], duration=3.0,
                         position_cost=20.0, orientation_cost=30.0),
        f2w_measure_phase(wall_normal=[+1.0, 0.0, 0.0], wall_face_pos=[0.500, 0.0, 0.5],
                          foot_standoff=0.008, wall_offset=0.02,
                          position_cost=8.0, orientation_cost=50.0),
        f2w_reach_phase(wall_normal=[+1.0, 0.0, 0.0], duration=2.0,
                        position_cost=8.0, orientation_cost=50.0),
    ],

    # FR: mirror of f2w — swing +45° (CW from above) to bring right foot toward +X wall.
    # wall_normal, face_axis, wall_face_pos identical to f2w.
    "f2w_fr": [
        lift_phase(height=0.10, duration=3.0),
        swing_phase(angle_deg=+45.0, duration=1.5),   # mirrored
        f2w_orient_phase(raise_z=0.40,       wall_normal=[+1.0, 0.0, 0.0], duration=3.0,
                         position_cost=20.0, orientation_cost=30.0),
        f2w_measure_phase(wall_normal=[+1.0, 0.0, 0.0], wall_face_pos=[0.500, 0.0, 0.5],
                          foot_standoff=0.008, wall_offset=0.02,
                          position_cost=8.0, orientation_cost=50.0),
        f2w_reach_phase(wall_normal=[+1.0, 0.0, 0.0], duration=2.0,
                        position_cost=8.0, orientation_cost=50.0),
    ],

    # ── direct joint-position tuning test (no IK) ────────────────────────────
    # Uses joint_phase: bypasses IK entirely, drives joints straight to
    # contracted_pose('FL') via PID. No EE Cartesian constraint — pure joint
    # position control. After 3 s the leg holds at the final angles.
    # Once the pose looks right, copy the angles into config.py and switch
    # back to lift_phase with joint_targets for the full blended version.
    #
    # Run with:  python sim.py --sequence f2w_test
    "f2w_test": [
        joint_phase(joint_targets=contracted_pose('FL'), duration=3.0,
                    disable_magnet=True, unbounded=True),
    ],
}


# ── runner ────────────────────────────────────────────────────────────────────

class SequenceRunner:
    """Drives a list of Phase objects, returning an IKTarget each step.

    Call start() once after settle, then step() every IK tick.
    The live ee_pos_fn / hip_pivot_fn are refreshed on every step() call,
    so they always reflect the current physical state without recreating the ctx.
    """

    def __init__(self, sequence: list[Phase]):
        self.sequence  = sequence
        self.phase_idx = -1
        self.phase_t0  = 0.0
        self.done      = False
        self._ctx: Optional[PhaseContext] = None
        self._last_target: Optional[IKTarget] = None

    @property
    def current_phase(self) -> Optional[Phase]:
        if 0 <= self.phase_idx < len(self.sequence):
            return self.sequence[self.phase_idx]
        return None

    def start(self, t: float, ctx: PhaseContext):
        """Call once after settle, before first step(). ctx is owned by the runner."""
        self._ctx = ctx
        self._enter(0, t)

    def step(self, t: float,
             ee_pos_fn: Callable[[], np.ndarray],
             hip_pivot_fn: Callable[[], np.ndarray],
             wall_dist_fn: Optional[Callable] = None) -> IKTarget:
        """Advance the sequence and return the current IKTarget.

        Only the three live-closure arguments are updated each tick;
        everything else (foot, ee_home, magnet fns, shared state) persists from start().
        """
        self._ctx.ee_pos_fn    = ee_pos_fn
        self._ctx.hip_pivot_fn = hip_pivot_fn
        if wall_dist_fn is not None:
            self._ctx.wall_dist_fn = wall_dist_fn

        if self.done or self.phase_idx < 0:
            fallback = self._last_target or IKTarget.position_only(self._ctx.ee_home)
            return fallback

        ph    = self.sequence[self.phase_idx]
        t_rel = t - self.phase_t0

        target = ph.step(t_rel, self._ctx)
        self._last_target = target

        if t_rel >= ph.duration and not ph.unbounded:
            if self.phase_idx + 1 < len(self.sequence):
                self._enter(self.phase_idx + 1, t)
            else:
                self.done = True
                print("\n── sequence complete ──")

        return target

    def progress(self, t: float) -> tuple[str, float]:
        """Return (phase_name, completion 0–1) for telemetry."""
        if self.done:          return "DONE", 1.0
        if self.phase_idx < 0: return "IDLE", 0.0
        ph  = self.sequence[self.phase_idx]
        pct = np.clip((t - self.phase_t0) / ph.duration, 0.0, 1.0) if ph.duration > 0 else 1.0
        return ph.name, pct

    def force_complete(self):
        """Mark the runner done without waiting for the last phase to expire.

        The last evaluated target is preserved in _last_target so subsequent
        step() calls return the frozen position — the foot stays locked at its
        wall contact point and IK treats it as a stance foot.
        Called by sim.py at the FL→FR handoff once F2W_REACH has dwelt long enough.
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