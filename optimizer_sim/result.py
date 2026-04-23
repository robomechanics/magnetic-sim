"""
result.py - Post-optimization results bundle for Sally magnetic wall-climbing sim.

Generates:
  results/latest/baseline_trajectory.csv   - per-timestep state with baseline params
  results/latest/optimized_trajectory.csv  - per-timestep state with optimized params
  results/latest/metrics_summary.json      - scalar metrics for both runs (for MATLAB)
  results/latest/results_plot.png          - 4-panel matplotlib figure

Called automatically at the end of tune_params.py, or standalone:
  python result.py --mode drive_up --run-dir results/<tag>
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import traceback
from typing import Any

import numpy as np

# ── matplotlib (non-interactive backend so it works headless) ─────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import sim_optimizer
from config import BASELINE_PARAMS, DEFAULT_MODE, DEFAULT_PARAMS, MODES

# ── colour palette ─────────────────────────────────────────────────────────────
C_BASELINE  = "#e05c5c"   # red-ish  → baseline (often bad)
C_OPTIMIZED = "#4caf7d"   # green    → optimized
C_TARGET    = "#aaaaaa"   # grey     → reference / target
AXIS_COLORS = {"X": "#5b9bd5", "Y": "#ed7d31", "Z": "#a5a5a5"}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

_TRAJ_FIELDS = ["time", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
                "quat_w", "quat_x", "quat_y", "quat_z", "failed"]


def _save_trajectory_csv(path: pathlib.Path, trajectory, mode_cfg: dict) -> None:
    """Write per-timestep data to CSV.  If trajectory is None, writes a single
    'failed' row so downstream tools can detect the run status."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TRAJ_FIELDS)
        w.writeheader()
        if trajectory is None:
            w.writerow({k: "nan" for k in _TRAJ_FIELDS} | {"failed": "1", "time": "0"})
            return
        target_vel = np.asarray(mode_cfg["target_velocity_xyz"], dtype=float)
        settle_time = mode_cfg["settle_time"]
        p0 = None
        t0 = None
        for s in trajectory:
            if p0 is None and s["time"] >= settle_time:
                p0, t0 = s["pos"].copy(), s["time"]
            row = {
                "time":   f"{s['time']:.6f}",
                "pos_x":  f"{s['pos'][0]:.8f}",
                "pos_y":  f"{s['pos'][1]:.8f}",
                "pos_z":  f"{s['pos'][2]:.8f}",
                "vel_x":  f"{s['vel'][0]:.8f}",
                "vel_y":  f"{s['vel'][1]:.8f}",
                "vel_z":  f"{s['vel'][2]:.8f}",
                "quat_w": f"{s['quat'][0]:.8f}",
                "quat_x": f"{s['quat'][1]:.8f}",
                "quat_y": f"{s['quat'][2]:.8f}",
                "quat_z": f"{s['quat'][3]:.8f}",
                "failed": "0",
            }
            w.writerow(row)


def _save_metrics_json(path: pathlib.Path, baseline_metrics: dict, opt_metrics: dict,
                       mode: str) -> None:
    bundle = {
        "mode": mode,
        "baseline": baseline_metrics,
        "optimized": opt_metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(bundle, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory → numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

def _traj_arrays(trajectory):
    """Return (time, pos [N,3], vel [N,3]) or None if trajectory is None."""
    if not trajectory:
        return None
    time = np.array([s["time"] for s in trajectory])
    pos  = np.array([s["pos"]  for s in trajectory])
    vel  = np.array([s["vel"]  for s in trajectory])
    return time, pos, vel


def _target_pos_array(time, pos, mode_cfg):
    settle_time = mode_cfg["settle_time"]
    idx = int(np.searchsorted(time, settle_time, side="left"))
    idx = min(idx, len(time) - 1)
    p0 = pos[idx]
    t0 = time[idx]
    target_vel = np.asarray(mode_cfg["target_velocity_xyz"], dtype=float)
    return p0[None, :] + (time - t0)[:, None] * target_vel[None, :]


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_xyz_position(ax, time, pos, target_pos, color, label_prefix, alpha=1.0):
    labels = ["X", "Y", "Z"]
    styles = ["-", "--", ":"]
    for i in range(3):
        ax.plot(time, pos[:, i], linestyle=styles[i], color=color,
                alpha=alpha * (1.0 - 0.15 * i),
                label=f"{label_prefix} {labels[i]}")
    for i in range(3):
        ax.plot(time, target_pos[:, i], linestyle=styles[i],
                color=C_TARGET, alpha=0.45, linewidth=0.8)


def _plot_tracking_error(ax, time, pos, target_pos, color, label_prefix):
    err = pos - target_pos
    labels = ["X", "Y", "Z"]
    styles = ["-", "--", ":"]
    for i in range(3):
        ax.plot(time, np.abs(err[:, i]), linestyle=styles[i], color=color,
                alpha=0.85, label=f"{label_prefix} |err {labels[i]}|")


def _rms_bar_chart(ax, baseline_metrics, opt_metrics):
    keys = ["rms_err_x", "rms_err_y", "rms_err_z", "tracking_error_axis_rms"]
    labels = ["RMS X", "RMS Y", "RMS Z", "RMS (axis)"]
    x = np.arange(len(keys))
    w = 0.35

    b_vals = [baseline_metrics.get(k, np.nan) for k in keys]
    o_vals = [opt_metrics.get(k, np.nan)      for k in keys]

    bars_b = ax.bar(x - w / 2, b_vals, w, label="Baseline",  color=C_BASELINE,  alpha=0.85)
    bars_o = ax.bar(x + w / 2, o_vals, w, label="Optimized", color=C_OPTIMIZED, alpha=0.85)

    # Annotate bars with values
    for bar, val in zip(list(bars_b) + list(bars_o), b_vals + o_vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RMS error (m)", fontsize=8)
    ax.set_title("Tracking Error Summary", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)


def _velocity_panel(ax, baseline_traj, opt_traj, mode_cfg):
    axis = mode_cfg["tracking_axis"]  # None for hold mode
    hold_mode = axis is None

    if hold_mode:
        # Hold mode: no tracking axis — plot velocity magnitude (target = zero)
        for traj, color, name in [(baseline_traj, C_BASELINE, "Baseline"),
                                   (opt_traj,      C_OPTIMIZED, "Optimized")]:
            arr = _traj_arrays(traj)
            if arr is None:
                continue
            time, _, vel = arr
            ax.plot(time, np.linalg.norm(vel, axis=1), color=color, alpha=0.85, label=name)
        ax.axhline(0.0, color=C_TARGET, linewidth=1.2, linestyle="--", label="Target (0 m/s)")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Velocity magnitude (m/s)", fontsize=8)
        ax.set_title("Velocity Magnitude — Hold (want zero)", fontsize=9, fontweight="bold")
    else:
        axis = int(axis)
        label  = ["X", "Y", "Z"][axis]
        target = float(mode_cfg["target_velocity_xyz"][axis])
        for traj, color, name in [(baseline_traj, C_BASELINE, "Baseline"),
                                   (opt_traj,      C_OPTIMIZED, "Optimized")]:
            arr = _traj_arrays(traj)
            if arr is None:
                continue
            time, _, vel = arr
            ax.plot(time, vel[:, axis], color=color, alpha=0.85, label=name)
        ax.axhline(target, color=C_TARGET, linewidth=1.2, linestyle="--", label=f"Target ({target} m/s)")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel(f"Velocity {label} (m/s)", fontsize=8)
        ax.set_title(f"Velocity Tracking — {label} axis", fontsize=9, fontweight="bold")

    ax.legend(fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────

def _build_figure(
    baseline_traj, opt_traj,
    baseline_metrics: dict, opt_metrics: dict,
    mode: str, mode_cfg: dict,
    run_dir: pathlib.Path,
) -> pathlib.Path:

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"Sally Magnetic Wall-Climbing — Optimization Results  [mode: {mode}]",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    ax_pos  = fig.add_subplot(gs[0, 0])   # XYZ position
    ax_err  = fig.add_subplot(gs[0, 1])   # Tracking error magnitude
    ax_vel  = fig.add_subplot(gs[1, 0])   # Velocity on tracking axis
    ax_bar  = fig.add_subplot(gs[1, 1])   # RMS bar summary

    # ── Panel 1 : XYZ position ────────────────────────────────────────────────
    ax_pos.set_title("Body Position (XYZ)", fontsize=9, fontweight="bold")
    ax_pos.set_xlabel("Time (s)", fontsize=8)
    ax_pos.set_ylabel("Position (m)", fontsize=8)
    any_data = False
    for traj, color, name in [(baseline_traj, C_BASELINE, "Baseline"),
                               (opt_traj,      C_OPTIMIZED, "Optimized")]:
        arr = _traj_arrays(traj)
        if arr is None:
            ax_pos.text(0.5, 0.5 if name == "Baseline" else 0.35,
                        f"{name}: SIMULATION FAILED", transform=ax_pos.transAxes,
                        ha="center", color=color, fontsize=9, style="italic")
            continue
        any_data = True
        time, pos, _ = arr
        tgt = _target_pos_array(time, pos, mode_cfg)
        _plot_xyz_position(ax_pos, time, pos, tgt, color, name)
    if any_data:
        # Legend: custom entries for target dashes
        handles, labels_ = ax_pos.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color=C_TARGET, alpha=0.6, linewidth=0.9, label="Target"))
        ax_pos.legend(handles=handles + [], fontsize=7, ncol=2)

    # ── Panel 2 : Tracking error ──────────────────────────────────────────────
    ax_err.set_title("Absolute Tracking Error (XYZ)", fontsize=9, fontweight="bold")
    ax_err.set_xlabel("Time (s)", fontsize=8)
    ax_err.set_ylabel("|Position error| (m)", fontsize=8)
    for traj, color, name in [(baseline_traj, C_BASELINE, "Baseline"),
                               (opt_traj,      C_OPTIMIZED, "Optimized")]:
        arr = _traj_arrays(traj)
        if arr is None:
            continue
        time, pos, _ = arr
        tgt = _target_pos_array(time, pos, mode_cfg)
        _plot_tracking_error(ax_err, time, pos, tgt, color, name)
    ax_err.legend(fontsize=7, ncol=2)
    ax_err.set_yscale("symlog", linthresh=1e-3)

    # ── Panel 3 : Velocity ────────────────────────────────────────────────────
    _velocity_panel(ax_vel, baseline_traj, opt_traj, mode_cfg)

    # ── Panel 4 : Bar chart ───────────────────────────────────────────────────
    _rms_bar_chart(ax_bar, baseline_metrics, opt_metrics)

    # ── Improvement annotation ────────────────────────────────────────────────
    b_rms = baseline_metrics.get("tracking_error_axis_rms", np.nan)
    o_rms = opt_metrics.get("tracking_error_axis_rms", np.nan)
    if np.isfinite(b_rms) and np.isfinite(o_rms) and b_rms > 1e-9:
        pct = 100.0 * (b_rms - o_rms) / b_rms
        sign = "↓" if pct >= 0 else "↑"
        color = C_OPTIMIZED if pct >= 0 else C_BASELINE
        fig.text(0.5, 0.005,
                 f"Axis tracking RMS: {b_rms:.4f} m (baseline) → {o_rms:.4f} m (optimized)  "
                 f"[{sign} {abs(pct):.1f}% improvement]",
                 ha="center", fontsize=9, color=color, fontweight="bold")

    out_path = run_dir / "results_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [result] Plot saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_results_bundle(
    mode: str = DEFAULT_MODE,
    baseline_params: dict[str, Any] | None = None,
    optimized_params: dict[str, Any] | None = None,
    run_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """
    Run both simulations, save CSVs + JSON metrics, render matplotlib figure.

    Parameters
    ----------
    mode             : simulation mode key (see config.MODES)
    baseline_params  : physics params for the 'before' run (defaults to BASELINE_PARAMS)
    optimized_params : physics params for the 'after' run  (defaults to DEFAULT_PARAMS)
    run_dir          : directory to write outputs into; defaults to results/latest/

    Returns
    -------
    run_dir path
    """
    baseline_params  = baseline_params  or BASELINE_PARAMS
    optimized_params = optimized_params or DEFAULT_PARAMS
    mode_cfg = MODES[mode]

    if run_dir is None:
        run_dir = pathlib.Path("results") / "latest"
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Result bundle  |  mode={mode}  |  dir={run_dir}")
    print(f"{'='*60}")

    # ── Run baseline ──────────────────────────────────────────────────────────
    print("\n[1/2] Running BASELINE simulation …")
    try:
        baseline_traj = sim_optimizer.run_simulation(baseline_params, mode=mode)
    except Exception:
        print("  BASELINE run raised an exception:")
        traceback.print_exc()
        baseline_traj = None

    baseline_status = "ok" if baseline_traj else "FAILED"
    print(f"  → status: {baseline_status}"
          + (f"  ({len(baseline_traj)} steps)" if baseline_traj else ""))

    # ── Run optimized ─────────────────────────────────────────────────────────
    print("\n[2/2] Running OPTIMIZED simulation …")
    try:
        opt_traj = sim_optimizer.run_simulation(optimized_params, mode=mode)
    except Exception:
        print("  OPTIMIZED run raised an exception:")
        traceback.print_exc()
        opt_traj = None

    opt_status = "ok" if opt_traj else "FAILED"
    print(f"  → status: {opt_status}"
          + (f"  ({len(opt_traj)} steps)" if opt_traj else ""))

    # ── Compute metrics ───────────────────────────────────────────────────────
    baseline_metrics = sim_optimizer.compute_tracking_metrics(baseline_traj, mode_cfg)
    opt_metrics      = sim_optimizer.compute_tracking_metrics(opt_traj,      mode_cfg)

    _print_metrics_comparison(baseline_metrics, opt_metrics)

    # ── Save files ────────────────────────────────────────────────────────────
    b_csv  = run_dir / "baseline_trajectory.csv"
    o_csv  = run_dir / "optimized_trajectory.csv"
    m_json = run_dir / "metrics_summary.json"

    _save_trajectory_csv(b_csv, baseline_traj, mode_cfg)
    _save_trajectory_csv(o_csv, opt_traj,      mode_cfg)
    _save_metrics_json(m_json, baseline_metrics, opt_metrics, mode)

    print(f"\n  [result] baseline_trajectory.csv  → {b_csv}")
    print(f"  [result] optimized_trajectory.csv → {o_csv}")
    print(f"  [result] metrics_summary.json     → {m_json}")

    # ── Matplotlib figure ─────────────────────────────────────────────────────
    _build_figure(baseline_traj, opt_traj, baseline_metrics, opt_metrics,
                  mode, mode_cfg, run_dir)

    return run_dir


def _print_metrics_comparison(bm: dict, om: dict) -> None:
    keys = [
        ("tracking_error_axis_rms",   "Axis RMS error (m)"),
        ("tracking_error_axis_final", "Final axis error (m)"),
        ("rms_err_x",                 "RMS X (m)"),
        ("rms_err_y",                 "RMS Y (m)"),
        ("rms_err_z",                 "RMS Z (m)"),
        ("settled_total_motion",      "Total motion (m)"),
        ("avg_vel_x",                 "Avg vel X (m/s)"),
        ("avg_vel_y",                 "Avg vel Y (m/s)"),
        ("avg_vel_z",                 "Avg vel Z (m/s)"),
    ]
    print(f"\n{'Metric':<30} {'Baseline':>12} {'Optimized':>12} {'Δ':>10}")
    print("-" * 68)
    for key, label in keys:
        b = bm.get(key, np.nan)
        o = om.get(key, np.nan)
        delta_str = ""
        if np.isfinite(b) and np.isfinite(o):
            delta = o - b
            delta_str = f"{delta:+.4f}"
        print(f"  {label:<28} {_fmt(b):>12} {_fmt(o):>12} {delta_str:>10}")


def _fmt(v) -> str:
    if not np.isfinite(v):
        return "FAILED"
    return f"{v:.5f}"


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sally optimization results bundle")
    parser.add_argument("--mode",    type=str, default=DEFAULT_MODE, choices=MODES.keys())
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Output directory (default: results/latest)")
    args = parser.parse_args()
    generate_results_bundle(mode=args.mode, run_dir=args.run_dir)