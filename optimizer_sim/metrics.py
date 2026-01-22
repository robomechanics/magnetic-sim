"""
metrics.py

STAGE 2 — METRICS & REWARD

Computes rollout quality metrics from logged trajectory data.
Independent of MuJoCo - operates purely on recorded step data.

Objective (continuous, NO pass/fail):
- Only two terms are used for scoring:
  (1) detachment fraction = 1 - contact_percentage
  (2) start/end displacement-based term:
      - HOLD (sideways): penalize start/end XY drift
      - DRIVE (drive_up): reward start/end progress along progress_axis

We use the config fields `hold_success_max_change_m` and `drive_success_min_up_m`
as NORMALIZATION SCALES (not thresholds) so the objective is dimensionless and stable.

Reward convention:
- We compute a cost (lower is better) and return reward = -cost
  so optimizers that maximize reward work naturally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass
class MetricConfig:
    """Configuration for metrics computation."""
    # Evaluation mode
    mode: str = "drive"  # "drive" or "hold"

    # Contact threshold used ONLY for detached flag / termination labeling
    contact_time_threshold: float = 0.2

    # Axis used for "drive_up" progress; default Z
    progress_axis: int = 2

    # ---------------------------
    # Objective weights (ONLY these are used)
    # ---------------------------
    # cost = w_detach * detachment_fraction  (+/-) w_progress * normalized_progress_term
    # reward = -cost
    w_detach: float = 1000.0
    w_progress: float = 1.0

    # ---------------------------
    # Normalization scales (USED; not pass/fail)
    # ---------------------------
    # Used as denominators; must be > 0 to avoid blow-ups.
    hold_success_max_change_m: float = 0.2  # scale for sideways XY drift (meters)
    drive_success_min_up_m: float = 0.2     # scale for upward progress (meters)

    # ---------------------------
    # Legacy fields (kept so callers that pass these won't break)
    # Not used for scoring anymore.
    # ---------------------------
    stuck_window_duration: float = 2.0
    stuck_progress_threshold: float = 0.005
    reward_progress_scale: float = 100.0
    reward_failure_penalty: float = -1000.0
    reward_speed_penalty_scale: float = 1.0
    target_speed_mps: float = 0.1
    reward_hold_base: float = 100.0
    reward_slip_penalty_scale: float = 500.0


def _compute_contact_percentage(step_records: List[Dict[str, Any]]) -> float:
    """
    Compute fraction of steps where at least one wheel is in contact.
    Uses wheel_dists: if any distance != inf, we count as contact.
    """
    if not step_records:
        return 0.0
    if "wheel_dists" not in step_records[0]:
        # If wheel_dists not provided, treat as fully detached (worst-case)
        return 0.0

    contact_steps = 0
    total_steps = len(step_records)

    for rec in step_records:
        wheel_dists = rec.get("wheel_dists", [])
        if isinstance(wheel_dists, (list, tuple)) and len(wheel_dists) == 4:
            has_contact = any(d != float("inf") for d in wheel_dists)
            if has_contact:
                contact_steps += 1

    contact_percentage = contact_steps / max(total_steps, 1)
    print(f"[METRICS] Contact: {contact_steps}/{total_steps} steps ({contact_percentage*100:.1f}%)")
    return float(contact_percentage)


def compute_metrics_drive(
    step_records: List[Dict[str, Any]],
    config: MetricConfig
) -> Dict[str, Any]:
    """
    DRIVE / drive_up mode:
    - progress is computed only from start/end along config.progress_axis
    - objective (continuous, no pass/fail):
        cost = w_detach * detachment_fraction - w_progress * (up_progress / drive_success_min_up_m)
        reward = -cost
    """
    if not step_records:
        return {
            "reward": float(-config.w_detach * 1.0),
            "progress_m": 0.0,
            "progress_rate_mps": 0.0,
            "detached": True,
            "stuck": False,
            "slip_m": 0.0,
            "contact_percentage": 0.0,
            "detachment_fraction": 1.0,
            "termination_reason": "no_data",
        }

    positions = np.array([rec["base_pos"] for rec in step_records], dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 2:
        return {
            "reward": float(-config.w_detach * 1.0),
            "progress_m": 0.0,
            "progress_rate_mps": 0.0,
            "detached": True,
            "stuck": False,
            "slip_m": 0.0,
            "contact_percentage": 0.0,
            "detachment_fraction": 1.0,
            "termination_reason": "bad_positions",
        }

    # Start/end progress along climbing axis (ONLY start/end)
    start_pos = float(positions[0, config.progress_axis])
    end_pos = float(positions[-1, config.progress_axis])
    total_progress = float(end_pos - start_pos)

    # progress rate is kept for logging compatibility (not used in scoring)
    times = np.array([rec["time"] for rec in step_records], dtype=float)
    total_time = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
    progress_rate = float(total_progress / total_time) if total_time > 1e-9 else 0.0

    # Detachment fraction from contact percentage
    contact_percentage = _compute_contact_percentage(step_records)
    detachment_fraction = float(1.0 - contact_percentage)

    # Detached flag (for termination labeling only)
    detached = bool(contact_percentage < config.contact_time_threshold)

    # Continuous objective (NO pass/fail). Use config variable as a NORMALIZATION SCALE.
    eps = 1e-9
    drive_scale = float(max(config.drive_success_min_up_m, eps))
    progress_term = float(total_progress / drive_scale)

    cost = float(config.w_detach * detachment_fraction - config.w_progress * progress_term)
    reward = float(-cost)

    termination = "detached" if detached else "ok"

    return {
        "reward": reward,
        "progress_m": float(total_progress),
        "progress_rate_mps": float(progress_rate),
        "detached": bool(detached),
        "stuck": False,  # legacy; no longer used
        "slip_m": 0.0,   # not relevant in drive mode
        "contact_percentage": float(contact_percentage),
        "detachment_fraction": float(detachment_fraction),
        "termination_reason": termination,
    }


def compute_metrics_hold(
    step_records: List[Dict[str, Any]],
    config: MetricConfig
) -> Dict[str, Any]:
    """
    HOLD / sideways mode:
    - drift is computed only from start/end in XY plane
    - objective (continuous, no pass/fail):
        cost = w_detach * detachment_fraction + w_progress * (xy_drift / hold_success_max_change_m)
        reward = -cost
    """
    if not step_records:
        return {
            "reward": float(-config.w_detach * 1.0),
            "progress_m": 0.0,
            "progress_rate_mps": 0.0,
            "detached": True,
            "stuck": False,
            "slip_m": 0.0,
            "contact_percentage": 0.0,
            "detachment_fraction": 1.0,
            "termination_reason": "no_data",
        }

    positions = np.array([rec["base_pos"] for rec in step_records], dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 2:
        return {
            "reward": float(-config.w_detach * 1.0),
            "progress_m": 0.0,
            "progress_rate_mps": 0.0,
            "detached": True,
            "stuck": False,
            "slip_m": 0.0,
            "contact_percentage": 0.0,
            "detachment_fraction": 1.0,
            "termination_reason": "bad_positions",
        }

    # Start/end drift in XY (ONLY start/end)
    start_xy = positions[0, :2]
    end_xy = positions[-1, :2]
    slip_distance = float(np.linalg.norm(end_xy - start_xy))

    # Detachment fraction from contact percentage
    contact_percentage = _compute_contact_percentage(step_records)
    detachment_fraction = float(1.0 - contact_percentage)

    # Detached flag (for termination labeling only)
    detached = bool(contact_percentage < config.contact_time_threshold)

    # Continuous objective (NO pass/fail). Use config variable as a NORMALIZATION SCALE.
    eps = 1e-9
    hold_scale = float(max(config.hold_success_max_change_m, eps))
    slip_term = float(slip_distance / hold_scale)

    cost = float(config.w_detach * detachment_fraction + config.w_progress * slip_term)
    reward = float(-cost)

    termination = "detached" if detached else "ok"

    return {
        "reward": reward,
        "progress_m": 0.0,          # not relevant in hold mode
        "progress_rate_mps": 0.0,   # not relevant
        "detached": bool(detached),
        "stuck": False,            # not relevant
        "slip_m": float(slip_distance),  # start/end drift only
        "contact_percentage": float(contact_percentage),
        "detachment_fraction": float(detachment_fraction),
        "termination_reason": termination,
    }


def compute_metrics(step_records: List[Dict[str, Any]], config: MetricConfig) -> Dict[str, Any]:
    if config.mode == "hold":
        return compute_metrics_hold(step_records, config)
    if config.mode == "drive":
        return compute_metrics_drive(step_records, config)
    raise ValueError(f"Unknown metrics mode: {config.mode}")
