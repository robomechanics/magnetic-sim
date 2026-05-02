"""
combined_config.py — Single source of truth for all sim optimizers and sims.

Replaces:
  sim_opt_config.py      (floor-lift optimizer)
  sim_wallopt_config.py  (wall-hold optimizer)
  sim_pulloff_config.py  (pull-off optimizer)

Import map after migration:
  combined_optimizer.py  → from combined_config import (N_CALLS, BATCH_SIZE, ...)
  sim_opt_sim.py         → from combined_config import (TIMESTEP, MAGNET_BODY_NAMES, ...)
  sim_wallopt_sim.py     → same
  sim_pulloff_sim.py     → same

Cost functions (lower = better):
  floor_calculate_cost(mean_norm, mean_neg_z, zero_contact_frac)
    30% — mean XYZ drift norm of FR/BL/BR EE from settled baselines (m)
    30% — mean max(0, FLOOR_Z - ee_z) of FR/BL/BR; absolute ref from scene.xml
    40% — mean fraction of hold steps where each stance foot had zero mag force

  wall_calculate_cost(fl_norm, fl_into_x, zero_contact_frac)
    30% — mean XYZ drift norm of FR/BL/BR EE from settled baselines (m)
    30% — mean max(0, ee_x - WALL_X) of FR/BL/BR; absolute ref from scene.xml
    40% — mean fraction of hold steps where each stance foot had zero mag force

  pulloff_calculate_cost(pulloff_force, target_force)
    One-sided shortfall below target_force, normalised to [0, 1].
    Cost = 0 when force >= target; cost = 1 when force = 0.

NOTE: this module rewrites mwc_mjcf/robot.xml on every import (see bottom of file).
In a parallel optimization context, worker processes each import this module.
The per-worker write is safe only because _setup_model in each sim uses an
exclusive file lock (fcntl.LOCK_EX) around bake_joint_angles + model load.
"""

from dataclasses import dataclass

import numpy as np

# ── Scene surface positions (from mwc_mjcf/scene.xml) ────────────────────────
# These never move — use as absolute setpoints rather than captured baselines.
#   floor geom: pos="-1.0 0 -0.05" size="1.5 0.5 0.05" → top face at Z = -0.05 + 0.05
#   wall  geom: pos="0.55 0 0.5"   size="0.05 0.5 0.5"  → inner face at X = 0.55 - 0.05
FLOOR_Z = 0.0   # m — floor top surface (world Z)
WALL_X  = 0.5   # m — wall inner surface (world X)


MU_0          = 4 * np.pi * 1e-7
MAGNET_VOLUME = np.pi * (0.0315 ** 2) * 0.025   # r=31.5mm, h=25mm

MAGNET_BODY_NAMES = [
    "electromagnet_FL",
    "electromagnet_FR",
    "electromagnet_BL",
    "electromagnet_BR",
]

# ── Shared simulation timing ───────────────────────────────────────────────────

TIMESTEP    = 0.0005   # s — 2000 Hz
SETTLE_TIME = 2.0      # s — settle phase (all magnets ON)

# ── Pull-off sim timing and targets ───────────────────────────────────────────

PULL_RATE          = 20.0    # N/s — default ramp rate (viewer / manual runs)
PULL_RATE_OPT      = 40.0    # N/s — fixed ramp rate used during optimization
SIM_DURATION       = 40.0    # s  — hard stop
DETACH_HOLD        = 1.0     # s  — must stay above force threshold for this long

# EML63mm-12 — Round Permanent Electromagnet 63mm Dia. 12V DC — Holding 215 lbs
GOAL_FORCE         = 956.37  # N — standalone pull-off optimisation target

# ── Viewer constants (pull-off) ────────────────────────────────────────────────

REAL_TIME_FACTOR   = 2.0
ARROW_RADIUS       = 0.004
MAG_ARROW_SCALE    = 0.001   # m per N  — blue magnetic arrow length
FORCE_ARROW_SCALE  = 0.005   # m per 10N — red pull arrow length
TELEMETRY_INTERVAL = 0.1     # s — terminal print interval

# ── Optimization settings ─────────────────────────────────────────────────────

N_CALLS                = 300
BATCH_SIZE             = 16
CMAES_SIGMA0           = 0.3
OPTIMIZER_RANDOM_STATE = 42

# ── Search space ──────────────────────────────────────────────────────────────
# 14 dimensions, shared across floor, wall, and combined optimizers.
# max_force_per_wheel uses floor/combined bounds [800, 1500]; the standalone
# pull-off optimizer historically used [300, 1200] — override via PULLOFF_SPACE
# if needed.

@dataclass
class Dim:
    """Minimal search-space dimension.
    Drop-in replacement for skopt.space.Real — exposes .low, .high, .prior, .name.
    """
    low:   float
    high:  float
    prior: str    # "log-uniform" or "uniform"
    name:  str


space: list[Dim] = [
    Dim(0.01,  2.0,   "log-uniform", name="sliding_friction"),
    Dim(1e-6,  10.0,  "log-uniform", name="torsional_friction"),
    Dim(1e-6,  1e-3,  "log-uniform", name="rolling_friction"),
    Dim(1e-5,  5.0,   "log-uniform", name="solref_timeconst"),   # was [1e-5, 1.0]
    Dim(0.001, 0.999, "uniform",     name="solimp_dmin"),
    Dim(1e-7,  1.0,   "log-uniform", name="solimp_width"),
    Dim(0.01,  0.99,  "uniform",     name="solimp_midpoint"),
    Dim(1.0,   10.0,  "uniform",     name="solimp_power"),        # was [2.0, 7.0]
    Dim(0,     100,   "uniform",     name="noslip_iterations"),   # was [0, 60]
    Dim(1e-6,  1e-3,  "log-uniform", name="noslip_tolerance"),
    Dim(0.0,   0.02,  "uniform",     name="margin"),              # was [0.0, 0.005]
    Dim(0.5,   2.0,   "log-uniform", name="Br"),
    Dim(0.001, 0.2,   "log-uniform", name="max_magnetic_distance"), # was [0.012, 0.1]
    Dim(800.0, 1500,  "log-uniform", name="max_force_per_wheel"),
]

# Pull-off standalone optimizer uses a wider lower bound on max_force_per_wheel.
PULLOFF_SPACE: list[Dim] = [
    *space[:-1],
    Dim(300.0, 1200, "log-uniform", name="max_force_per_wheel"),
]

# Fixed solref damping ratio (not tuned in any optimizer)
SOLREF_DAMPRATIO_FIXED = 10.0


# ── Space → PARAMS ────────────────────────────────────────────────────────────

def point_to_params(point: list) -> dict:
    """Map a flat CMA-ES point (14 values) to a PARAMS dict.

    solimp layout:         [dmin, dmax=0.9999, width, midpoint, power]
    solref layout:         [timeconst, dampratio=SOLREF_DAMPRATIO_FIXED]
    ground_friction layout:[sliding, torsional, rolling]
    """
    (
        sliding_friction, torsional_friction, rolling_friction,
        solref_timeconst,
        solimp_dmin, solimp_width, solimp_midpoint, solimp_power,
        noslip_iterations, noslip_tolerance,
        margin,
        Br, max_magnetic_distance, max_force_per_wheel,
    ) = point

    return {
        'ground_friction':       [sliding_friction, torsional_friction, rolling_friction],
        'solref':                [solref_timeconst, SOLREF_DAMPRATIO_FIXED],
        'solimp':                [solimp_dmin, 0.9999, solimp_width, solimp_midpoint, solimp_power],
        'noslip_iterations':     int(round(noslip_iterations)),
        'noslip_tolerance':      noslip_tolerance,
        'margin':                margin,
        'Br':                    Br,
        'max_magnetic_distance': max_magnetic_distance,
        'max_force_per_wheel':   max_force_per_wheel,
    }


# ── Cost functions ────────────────────────────────────────────────────────────

def floor_calculate_cost(mean_norm: float, mean_neg_z: float,
                         zero_contact_frac: float = 0.0) -> dict:
    """
    Weighted cost for the floor-lift sim (FL foot held at lift height).

      30% — mean XYZ drift norm of FR/BL/BR EE from settled baselines (metres)
      30% — mean max(0, FLOOR_Z - ee_z) of FR/BL/BR; absolute ref, scene.xml FLOOR_Z=0.0
      40% — mean fraction of hold steps where each stance foot had zero mag force
    """
    norm_cost         = 0.30 * mean_norm
    neg_z_cost        = 0.30 * mean_neg_z
    zero_contact_cost = 0.40 * zero_contact_frac
    total             = norm_cost + neg_z_cost + zero_contact_cost
    return {
        "total_cost":        total,
        "mean_norm":         mean_norm,
        "mean_neg_z":        mean_neg_z,
        "zero_contact_frac": zero_contact_frac,
        "norm_cost":         norm_cost,
        "neg_z_cost":        neg_z_cost,
        "zero_contact_cost": zero_contact_cost,
    }


def wall_calculate_cost(fl_norm: float, fl_into_x: float,
                        zero_contact_frac: float = 0.0) -> dict:
    """
    Weighted cost for the wall-hold sim (FL foot planted on wall).

    The wall is vertical at +X. The failure modes penalised are:
      30% — mean XYZ drift norm of FR/BL/BR EE from settled baselines (metres)
      30% — mean max(0, ee_x - WALL_X) of FR/BL/BR; absolute ref, scene.xml WALL_X=0.5
      40% — mean fraction of hold steps where each stance foot had zero mag force
    """
    norm_cost         = 0.30 * fl_norm
    into_x_cost       = 0.30 * fl_into_x
    zero_contact_cost = 0.40 * zero_contact_frac
    total             = norm_cost + into_x_cost + zero_contact_cost
    return {
        "total_cost":        total,
        "fl_norm":           fl_norm,
        "fl_into_x":         fl_into_x,
        "zero_contact_frac": zero_contact_frac,
        "norm_cost":         norm_cost,
        "into_x_cost":       into_x_cost,
        "zero_contact_cost": zero_contact_cost,
    }


def pulloff_calculate_cost(pulloff_force: float, target_force: float) -> dict:
    """
    One-sided shortfall cost for the pull-off sim.

    Used by the combined optimizer where target_force = params['max_force_per_wheel'].
    For the standalone pull-off optimizer, pass GOAL_FORCE as target_force.

      Cost = 0   when pulloff_force >= target_force
      Cost → 1   as pulloff_force → 0
      Cost = 9999 when pulloff_force == 0 (no adhesion at all)
    """
    if pulloff_force == 0.0:
        return {
            "total_cost":     9999.0,
            "pulloff_target": target_force,
            "shortfall_n":    target_force,
        }

    shortfall_n  = max(0.0, target_force - pulloff_force)
    total_cost   = shortfall_n / target_force   # normalised to [0, 1]
    return {
        "total_cost":      total_cost,
        "pulloff_target":  target_force,
        "shortfall_n":     shortfall_n,
    }


# ── Default / warm-start PARAMS ───────────────────────────────────────────────
# Floor and wall optimizers share the same default (carry-over from last best run).
# Pull-off defaults are separate because it targets a different force regime.

PARAMS = {
    'ground_friction':       [0.379298, 0.0175887, 0.000171233],
    'solref':                [0.0180959, SOLREF_DAMPRATIO_FIXED],
    'solimp':                [0.942108, 0.9999, 7.00088e-05, 0.741898, 3.73896],
    'noslip_iterations':     30,
    'noslip_tolerance':      1.44833e-05,
    'margin':                0.000542304,
    'Br':                    1.25377,
    'max_magnetic_distance': 0.0628915,
    'max_force_per_wheel':   1160.94,
}

PULLOFF_PARAMS = {
    'ground_friction':       [0.130563, 0.00256843, 1.13878e-05],
    'solref':                [0.00192329, SOLREF_DAMPRATIO_FIXED],
    'solimp':                [0.361806, 0.9999, 0.00065636, 0.603088, 3.97107],
    'noslip_iterations':     31,
    'noslip_tolerance':      2.42366e-05,
    'margin':                0.00149908,
    'Br':                    1.32665,
    'max_magnetic_distance': 0.0706161,
    'max_force_per_wheel':   917.216,
}

# ── Magnet array geometry ──────────────────────────────────────────────────────
# All values derived from robot_original.xml.
#   EE cylinder:  fromto="0 0 0  0.066 0 0"  → 66.0 mm long, 11.5 mm radius
#   EM body:      pos="0.066075 0 0" in EE frame → origin 66.075 mm from EE origin
#   EM +Y axis  = EE +X axis (from quat rotation), so EM_Y directly maps to EE_X offset
#
# Limits: 10 % from each end of the enclosure.
# Offset: array placed 10 mm from the far (tip) end of the enclosure.

EE_ENCLOSURE_LENGTH_MM          = 66.0    # mm — CF cylinder axial length
EE_ENCLOSURE_RADIUS_MM          = 11.5    # mm — CF cylinder radius
EM_BODY_ORIGIN_EE_X_MM          = 66.075  # mm — electromagnet body origin in EE-frame X

_EE_LIMIT_PCT                   = 0.10
ARRAY_LOWER_LIMIT_EE_X_MM       = _EE_LIMIT_PCT * EE_ENCLOSURE_LENGTH_MM           #  6.6 mm from near end
ARRAY_UPPER_LIMIT_EE_X_MM       = (1.0 - _EE_LIMIT_PCT) * EE_ENCLOSURE_LENGTH_MM  # 59.4 mm from near end

ARRAY_TARGET_DIST_FROM_FAR_END_MM = 10.0                                            # mm — design intent
_ARRAY_EE_X_MM                  = EE_ENCLOSURE_LENGTH_MM - ARRAY_TARGET_DIST_FROM_FAR_END_MM  # 56.0 mm
ARRAY_EM_Y_M                    = (_ARRAY_EE_X_MM - EM_BODY_ORIGIN_EE_X_MM) / 1000 # -0.010075 m
# ──────────────────────────────────────────────────────────────────────────────

PARAM_PRESETS = {
    'hold': {
        'ground_friction':       [0.9, 0.034585, 0.001734],
        'solref':                [0.000572, SOLREF_DAMPRATIO_FIXED],
        'solimp':                [0.860956, 0.9999, 0.000100, 0.5, 1.0],
        'noslip_iterations':     20,
        'noslip_tolerance':      1e-6,
        'margin':                0.0,
        'Br':                    1.332000,
        'max_magnetic_distance': 0.011413,
        'max_force_per_wheel':   139.324024,
    },
    'drive_sideways': {
        'ground_friction':       [0.975785, 0.000342, 0.000372],
        'solref':                [0.000129, SOLREF_DAMPRATIO_FIXED],
        'solimp':                [0.875461, 0.9999, 0.002193, 0.5, 1.0],
        'noslip_iterations':     20,
        'noslip_tolerance':      1e-6,
        'margin':                0.0,
        'Br':                    1.476441,
        'max_magnetic_distance': 0.029736,
        'max_force_per_wheel':   264.173934,
    },
    'drive_up': {
        'ground_friction':       [0.973464, 0.000202, 0.000010],
        'solref':                [0.000100, SOLREF_DAMPRATIO_FIXED],
        'solimp':                [0.926007, 0.9999, 0.001311, 0.5, 1.0],
        'noslip_iterations':     23,
        'noslip_tolerance':      1e-6,
        'margin':                0.0,
        'Br':                    1.599175,
        'max_magnetic_distance': 0.018389,
        'max_force_per_wheel':   205.270447,
    },
    'pull_off': PULLOFF_PARAMS,
}

# ── Magnet array XML adjustment ───────────────────────────────────────────────
# Runs on every import. Reads robot_original.xml (clean reference — never
# written to) and writes a fresh robot.xml each time.
#
# WARNING: each worker process in a parallel optimization run will execute this
# block on import. The write itself is not locked here — safety relies on the
# exclusive file lock (fcntl.LOCK_EX) inside _setup_model in each sim file,
# which serialises bake_joint_angles + model load. Do not remove that lock.

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_SRC  = Path(__file__).parent / "mwc_mjcf" / "robot_original.xml"
_DEST = Path(__file__).parent / "mwc_mjcf" / "robot.xml"

_ELECTROMAGNET_BODIES = {
    "electromagnet_FL", "electromagnet_FR",
    "electromagnet_BL", "electromagnet_BR",
}

if not _SRC.exists():
    sys.exit(f"ERROR: {_SRC} not found.")

_ee_x_mm = 66.075 + ARRAY_EM_Y_M * 1000
if not (6.6 <= _ee_x_mm <= 59.4):
    sys.exit(
        f"ERROR: target EE-X {_ee_x_mm:.3f} mm is outside the safe range [6.6, 59.4] mm."
    )

_tree = ET.parse(_SRC)
_root = _tree.getroot()

_modified = 0
for _body in _root.iter("body"):
    if _body.get("name") not in _ELECTROMAGNET_BODIES:
        continue
    for _geom in _body:
        if _geom.tag != "geom" or _geom.get("type") != "sphere":
            continue
        _parts = _geom.get("pos", "").split()
        if len(_parts) != 3:
            continue
        _parts[1] = f"{ARRAY_EM_Y_M:+.6f}"
        _geom.set("pos", f"{_parts[0]} {_parts[1]} {_parts[2]}")
        _modified += 1

print(f"[combined_config] Overwriting {_DEST} from {_SRC} ...")
_tree.write(_DEST, xml_declaration=False, encoding="unicode")

print(f"[combined_config] robot.xml updated: "
      f"EM-Y={ARRAY_EM_Y_M:+.6f} m  |  "
      f"EE-X={_ee_x_mm:.1f} mm ({EE_ENCLOSURE_LENGTH_MM - _ee_x_mm:.1f} mm from far end)  |  "
      f"{_modified} spheres")