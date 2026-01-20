"""
metrics.py

STAGE 2 — METRICS & REWARD

Computes rollout quality metrics from logged trajectory data.
Independent of MuJoCo - operates purely on recorded step data.

Supports two evaluation modes:
- "hold": Robot should stay in place (penalize slip/drift)
- "drive": Robot should climb upward (reward progress, penalize stuck/detached)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass
class MetricConfig:
    """Configuration for metrics computation"""
    # Evaluation mode
    mode: str = "drive"  # "drive" or "hold"
    
    # Failure detection thresholds
    bounce_tolerance: float = 1.0              # 1m - allow bouncing before considering detached
    detach_distance_threshold: float = 2.0      # meters (kept for legacy, now using bounce_tolerance)
    detach_duration_threshold: float = 4.0      # seconds
    stuck_window_duration: float = 8.0          # seconds
    stuck_progress_threshold: float = 0.005     # meters
    
    # Progress direction (world frame axis)
    progress_axis: int = 2  # Z-axis for upward climbing
    
    # Reward weights - DRIVE mode
    reward_progress_scale: float = 100.0
    reward_failure_penalty: float = -1000.0
    reward_speed_penalty_scale: float = 1.0
    target_speed_mps: float = 0.1  # target climbing speed
    
    # Reward weights - HOLD mode
    reward_hold_base: float = 100.0  # base reward for staying attached
    reward_slip_penalty_scale: float = 500.0  # penalty per meter of slip


def compute_metrics_drive(
    step_records: List[Dict[str, Any]],
    config: MetricConfig
) -> Dict[str, Any]:
    """
    Compute metrics for DRIVE mode (climbing upward).
    
    Rewards:
    - Positive for upward progress
    - Large penalty for detachment or getting stuck
    - Small penalty for speed deviation from target
    """
    if not step_records:
        return {
            "reward": config.reward_failure_penalty,
            "progress_m": 0.0,
            "progress_rate_mps": 0.0,
            "detached": True,
            "stuck": False,
            "slip_m": 0.0,
            "termination_reason": "no_data"
        }
    
    # Extract trajectory data
    times = np.array([rec["time"] for rec in step_records])
    positions = np.array([rec["base_pos"] for rec in step_records])
    
    # Compute progress along climbing axis
    progress = positions[:, config.progress_axis] - positions[0, config.progress_axis]
    total_progress = float(progress[-1])
    
    duration = times[-1] - times[0]
    avg_speed = total_progress / max(duration, 1e-6)
    
    # Failure detection: DETACHED (fell off wall)
    detached = False
    if "wheel_dists" in step_records[0]:
        for rec in step_records:
            wheel_dists = rec.get("wheel_dists", [])
            if len(wheel_dists) == 4:
                # Check if any valid wheel distance exists within tolerance
                has_contact = any(
                    d != float("inf") and d <= config.bounce_tolerance 
                    for d in wheel_dists
                )
                
                # If no wheels in contact, mark as detached
                if not has_contact:
                    detached = True
                    break
    
    # Failure detection: STUCK (no progress)
    stuck = False
    if duration >= config.stuck_window_duration:
        if abs(total_progress) < config.stuck_progress_threshold:
            stuck = True
    
    # Compute reward
    if detached or stuck:
        reward = config.reward_failure_penalty
        termination = "detached" if detached else "stuck"
    else:
        # Progress reward
        reward = total_progress * config.reward_progress_scale
        
        # Speed error penalty (quadratic)
        speed_error = avg_speed - config.target_speed_mps
        reward -= config.reward_speed_penalty_scale * (speed_error ** 2)
        
        termination = "ok"
    
    return {
        "reward": float(reward),
        "progress_m": float(total_progress),
        "progress_rate_mps": float(avg_speed),
        "detached": bool(detached),
        "stuck": bool(stuck),
        "slip_m": 0.0,  # Not relevant in drive mode
        "termination_reason": termination,
    }


def compute_metrics_hold(
    step_records: List[Dict[str, Any]],
    config: MetricConfig
) -> Dict[str, Any]:
    """
    Compute metrics for HOLD mode (staying in place, sideways orientation).
    
    Rewards:
    - Base reward for staying attached
    - Penalty for slipping/drifting from origin
    - Large penalty for detachment
    """
    if not step_records:
        return {
            "reward": config.reward_failure_penalty,
            "progress_m": 0.0,
            "progress_rate_mps": 0.0,
            "detached": True,
            "stuck": False,
            "slip_m": 0.0,
            "termination_reason": "no_data"
        }
    
    # Extract trajectory data
    times = np.array([rec["time"] for rec in step_records])
    positions = np.array([rec["base_pos"] for rec in step_records])
    
    # Compute slip (horizontal drift from starting position)
    # In sideways mode, we care about X-Y plane movement
    start_pos = positions[0, :2]  # X, Y only
    end_pos = positions[-1, :2]
    slip_distance = float(np.linalg.norm(end_pos - start_pos))
    
    # Also compute max slip during entire trajectory
    max_slip = 0.0
    for pos in positions:
        slip = np.linalg.norm(pos[:2] - start_pos)
        max_slip = max(max_slip, slip)
    
    duration = times[-1] - times[0]
    
    # Failure detection: DETACHED (fell off wall)
    detached = False
    if "wheel_dists" in step_records[0]:
        for rec in step_records:
            wheel_dists = rec.get("wheel_dists", [])
            if len(wheel_dists) == 4:
                # Check if any valid wheel distance exists within tolerance
                has_contact = any(
                    d != float("inf") and d <= config.bounce_tolerance 
                    for d in wheel_dists
                )
                
                # If no wheels in contact, mark as detached
                if not has_contact:
                    detached = True
                    break
    
    # Compute reward
    if detached:
        reward = config.reward_failure_penalty
        termination = "detached"
    else:
        # Base reward for staying attached
        reward = config.reward_hold_base
        
        # Penalize slip (we want minimal movement)
        reward -= config.reward_slip_penalty_scale * max_slip
        
        termination = "ok"
    
    return {
        "reward": float(reward),
        "progress_m": 0.0,  # Not relevant in hold mode
        "progress_rate_mps": 0.0,  # Not relevant
        "detached": bool(detached),
        "stuck": False,  # Not relevant in hold mode
        "slip_m": float(max_slip),
        "termination_reason": termination,
    }


def compute_metrics(
    step_records: List[Dict[str, Any]],
    config: MetricConfig
) -> Dict[str, Any]:
    """
    Dispatcher function that calls the appropriate metrics function based on mode.
    """
    if config.mode == "hold":
        return compute_metrics_hold(step_records, config)
    elif config.mode == "drive":
        return compute_metrics_drive(step_records, config)
    else:
        raise ValueError(f"Unknown metrics mode: {config.mode}")