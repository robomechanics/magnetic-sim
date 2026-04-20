"""
metrics.py

Reward computation for magnetic wall-climbing robot optimization.

Computes quality metrics from simulation trajectory data.
Independent of MuJoCo - operates purely on recorded step data.

Reward Function:
    reward = -cost
    
    cost = w_detach * detachment_fraction ± w_progress * normalized_displacement
    
    Where:
    - detachment_fraction = 1 - contact_percentage (0 = always attached, 1 = always detached)
    - normalized_displacement = measured_displacement / normalization_scale
    - Sign of progress term: negative for DRIVE (reward upward motion), positive for HOLD (penalize drift)

Two Test Modes:
    DRIVE (climb up):  Maximize upward progress while staying attached
    HOLD (sideways):   Minimize drift while staying attached
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MetricConfig:
    """Configuration for metrics computation."""
    
    # Test mode
    mode: str = "drive"  # "drive" or "hold"
    
    # Detachment detection
    contact_time_threshold: float = 0.2  # Contact <20% of time = detached
    
    # Progress measurement
    progress_axis: int = 2  # Z-axis (0=X, 1=Y, 2=Z)
    
    # Objective function weights
    w_detach: float = 1000.0  # Penalty weight for detachment
    w_progress: float = 1.0   # Weight for progress/drift term
    
    # Normalization scales (make cost dimensionless)
    hold_success_max_change_m: float = 0.2  # Scale for sideways drift (m)
    drive_success_min_up_m: float = 0.2     # Scale for upward progress (m)
    
    # Legacy fields (kept for backward compatibility, not used)
    stuck_window_duration: float = 2.0
    stuck_progress_threshold: float = 0.005
    reward_progress_scale: float = 100.0
    reward_failure_penalty: float = -1000.0
    reward_speed_penalty_scale: float = 1.0
    target_speed_mps: float = 0.1
    reward_hold_base: float = 100.0
    reward_slip_penalty_scale: float = 500.0


# =============================================================================
# CONTACT DETECTION
# =============================================================================

def _compute_contact_percentage(step_records: List[Dict[str, Any]]) -> float:
    """
    Compute fraction of timesteps where robot is in contact with wall.
    
    Contact is detected when at least one wheel has finite distance to wall.
    
    Args:
        step_records: List of step data with "wheel_dists" field
        
    Returns:
        Contact percentage in range [0, 1]
    """
    if not step_records:
        return 0.0
    
    if "wheel_dists" not in step_records[0]:
        # No distance data = assume fully detached (conservative)
        return 0.0
    
    contact_steps = 0
    total_steps = len(step_records)
    
    for rec in step_records:
        wheel_dists = rec.get("wheel_dists", [])
        
        # Check if valid wheel distance data exists
        if isinstance(wheel_dists, (list, tuple)) and len(wheel_dists) == 4:
            # Any wheel in contact? (finite distance)
            has_contact = any(d != float("inf") for d in wheel_dists)
            if has_contact:
                contact_steps += 1
    
    contact_percentage = contact_steps / max(total_steps, 1)
    
    # print(f"[METRICS] Contact: {contact_steps}/{total_steps} steps ({contact_percentage*100:.1f}%)")
    
    return float(contact_percentage)


# =============================================================================
# DRIVE MODE METRICS
# =============================================================================

def compute_metrics_drive(
    step_records: List[Dict[str, Any]],
    config: MetricConfig
) -> Dict[str, Any]:
    """
    Compute metrics for DRIVE mode (climbing upward).
    
    Objective: Maximize upward progress while staying attached.
    
    Formula:
        progress = max(0, end_z - start_z)  # Only count upward movement
        cost = w_detach * detachment_fraction - w_progress * (progress / scale)
        reward = -cost
    
    Args:
        step_records: Trajectory data with "base_pos" and "wheel_dists"
        config: Metric configuration
        
    Returns:
        Dictionary with reward, progress, contact metrics, termination reason
    """
    # Handle empty or invalid data
    if not step_records:
        return _empty_result(config, termination="no_data")
    
    positions = np.array([rec["base_pos"] for rec in step_records], dtype=float)
    
    if positions.ndim != 2 or positions.shape[0] < 2:
        return _empty_result(config, termination="bad_positions")
    
    # Measure progress along climbing axis (start to end only)
    start_pos = float(positions[0, config.progress_axis])
    end_pos = float(positions[-1, config.progress_axis])
    total_progress_raw = float(end_pos - start_pos)
    
    # ✅ FIX: Only count positive progress (upward movement)
    # If robot slipped down, treat it as zero progress (not negative reward)
    total_progress = float(max(0.0, total_progress_raw))
    
    # Progress rate (for logging only, not used in reward)
    times = np.array([rec["time"] for rec in step_records], dtype=float)
    total_time = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
    progress_rate = float(total_progress / total_time) if total_time > 1e-9 else 0.0
    
    # Contact analysis
    contact_percentage = _compute_contact_percentage(step_records)
    detachment_fraction = float(1.0 - contact_percentage)
    detached = bool(contact_percentage < config.contact_time_threshold)
    
    # Compute continuous reward
    eps = 1e-9
    drive_scale = float(max(config.drive_success_min_up_m, eps))
    progress_term = float(total_progress / drive_scale)
    
    cost = float(config.w_detach * detachment_fraction - config.w_progress * progress_term)
    reward = float(-cost)
    
    termination = "detached" if detached else "ok"
    
    return {
        "reward": reward,
        "progress_m": float(total_progress),  # Returns clamped value
        "progress_m_raw": float(total_progress_raw),  # NEW: Track actual displacement
        "progress_rate_mps": float(progress_rate),
        "detached": bool(detached),
        "stuck": False,  # Legacy field
        "slip_m": 0.0,   # Not relevant in drive mode
        "contact_percentage": float(contact_percentage),
        "detachment_fraction": float(detachment_fraction),
        "termination_reason": termination,
    }


# =============================================================================
# HOLD MODE METRICS
# =============================================================================

def compute_metrics_hold(
    step_records: List[Dict[str, Any]],
    config: MetricConfig
) -> Dict[str, Any]:
    """
    Compute metrics for HOLD mode (no sideways drift).
    
    Objective: Minimize sideways drift while staying attached.
    
    Formula:
        drift = ||end_xy - start_xy||
        cost = w_detach * detachment_fraction + w_progress * (drift / scale)
        reward = -cost
    
    Args:
        step_records: Trajectory data with "base_pos" and "wheel_dists"
        config: Metric configuration
        
    Returns:
        Dictionary with reward, slip distance, contact metrics, termination reason
    """
    # Handle empty or invalid data
    if not step_records:
        return _empty_result(config, termination="no_data")
    
    positions = np.array([rec["base_pos"] for rec in step_records], dtype=float)
    
    if positions.ndim != 2 or positions.shape[0] < 2:
        return _empty_result(config, termination="bad_positions")
    
    # Measure sideways drift in XY plane (start to end only)
    start_xy = positions[0, :2]
    end_xy = positions[-1, :2]
    slip_distance = float(np.linalg.norm(end_xy - start_xy))
    
    # Contact analysis
    contact_percentage = _compute_contact_percentage(step_records)
    detachment_fraction = float(1.0 - contact_percentage)
    detached = bool(contact_percentage < config.contact_time_threshold)
    
    # Compute continuous reward
    eps = 1e-9
    hold_scale = float(max(config.hold_success_max_change_m, eps))
    slip_term = float(slip_distance / hold_scale)
    
    cost = float(config.w_detach * detachment_fraction + config.w_progress * slip_term)
    reward = float(-cost)
    
    termination = "detached" if detached else "ok"
    
    return {
        "reward": reward,
        "progress_m": 0.0,          # Not relevant in hold mode
        "progress_rate_mps": 0.0,   # Not relevant
        "detached": bool(detached),
        "stuck": False,             # Legacy field
        "slip_m": float(slip_distance),
        "contact_percentage": float(contact_percentage),
        "detachment_fraction": float(detachment_fraction),
        "termination_reason": termination,
    }


# =============================================================================
# HELPERS
# =============================================================================

def _empty_result(config: MetricConfig, termination: str) -> Dict[str, Any]:
    """
    Return empty/failed result for invalid data.
    
    Args:
        config: Metric configuration
        termination: Reason for failure
        
    Returns:
        Dictionary with worst-case values
    """
    return {
        "reward": float(-config.w_detach * 1.0),  # Worst case: fully detached
        "progress_m": 0.0,
        "progress_rate_mps": 0.0,
        "detached": True,
        "stuck": False,
        "slip_m": 0.0,
        "contact_percentage": 0.0,
        "detachment_fraction": 1.0,
        "termination_reason": termination,
    }


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def compute_metrics(step_records: List[Dict[str, Any]], config: MetricConfig) -> Dict[str, Any]:
    """
    Compute metrics based on configured mode.
    
    Args:
        step_records: Trajectory data from simulation
        config: Metric configuration with mode selection
        
    Returns:
        Metrics dictionary with reward and diagnostic info
        
    Raises:
        ValueError: If mode is not "hold" or "drive"
    """
    if config.mode == "hold":
        return compute_metrics_hold(step_records, config)
    elif config.mode == "drive":
        return compute_metrics_drive(step_records, config)
    else:
        raise ValueError(f"Unknown metrics mode: {config.mode}. Must be 'hold' or 'drive'")