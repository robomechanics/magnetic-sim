"""
result.py — Post-optimization results bundle for Sally legged magnetic gaiting.

Produces two figures into run_dir/:
    figure_lift_drift.png   — stance-leg drift + body-height comparison,
                              floor and wall, baseline vs optimized.
    figure_pulloff.png      — pull-off force vs expected adhesion range.

Called automatically by combined_optimizer.py after CMA-ES finishes,
or standalone:
    python result.py [--run-dir results/latest] [--baseline-json b.json]
                     [--optimized-json o.json]  [--skip floor|wall|pulloff]
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import sys
import time
import traceback
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup (mirrors the sim modules) ─────────────────────────────────────
_OPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_LEGGED_DIR = os.path.abspath(os.path.join(_OPT_DIR, ".."))
for _p in (_OPT_DIR, _LEGGED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco
import sim_opt_sim
import sim_vertopt_sim
import sim_pulloff_sim
from config import TIMESTEP

# ── Colours ───────────────────────────────────────────────────────────────────
C_BASE = "#e05c5c"    # red   — baseline (pre-opt)
C_OPT  = "#4caf7d"   # green — optimized
C_REF  = "#888888"   # grey  — reference / expected range


# ─────────────────────────────────────────────────────────────────────────────
# Param helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path) as f:
        p = json.load(f)
    for k in ("ground_friction", "solref", "solimp"):
        if k in p:
            p[k] = list(p[k])
    if "noslip_iterations" in p:
        p["noslip_iterations"] = int(p["noslip_iterations"])
    return p


# Pre-optimization reference params.
# (rocker_stiffness / rocker_damping / wheel_kp / wheel_kv removed -- wheeled sim only)
BASELINE_PARAMS = {
    "ground_friction":       [0.95, 0.01, 0.01],
    "solref":                [0.0004, 25.0],
    "solimp":                [0.9, 0.95, 0.001, 0.5, 1.0],
    "noslip_iterations":     15,
    "noslip_tolerance":      1e-6,
    "margin":                0.0,
    "Br":                    1.48,
    "max_magnetic_distance": 0.010,
    "max_force_per_wheel":   50.0,
}


def _get_params(baseline_json: str | None,
                optimized_json: str | None) -> tuple[dict, dict]:
    cfg = importlib.import_module("sim_opt_config")
    opt = _load_json(optimized_json) if optimized_json else deepcopy(cfg.PARAMS)
    if baseline_json:
        baseline = _load_json(baseline_json)
    elif hasattr(cfg, "BASELINE_PARAMS"):
        baseline = deepcopy(cfg.BASELINE_PARAMS)
    else:
        baseline = deepcopy(BASELINE_PARAMS)
    return baseline, opt


# ─────────────────────────────────────────────────────────────────────────────
# Lift-trace runner
# Re-runs the sim loop with per-timestep recording. Imports each module's own
# _setup_model / _apply_mag / _IK / _PID so physics is identical to the CMA-ES
# evaluations. Returns None on failure.
# ─────────────────────────────────────────────────────────────────────────────

def _run_lift_trace(sim_mod, params: dict, *, is_wall: bool) -> dict | None:
    ST  = sim_mod.SETTLE_TIME
    LH  = sim_mod.LIFT_HOLD
    LMS = sim_mod.LIFT_MEASURE_START
    LDZ = sim_mod.LIFT_DZ
    IKN = sim_mod.IK_EVERY_N
    SW  = sim_mod.SWING_FOOT

    try:
        model, data, plate_ids, magnet_ids, sphere_gids = sim_mod._setup_model(params)
    except Exception:
        traceback.print_exc()
        return None

    swing_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                   f"electromagnet_{SW}")
    stance_feet = [f for f in sim_mod.FEET if f != SW]
    stance_bids = {f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                          f"electromagnet_{f}")
                   for f in stance_feet}
    body_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "main_frame")

    ik  = sim_mod._IK(model)
    pid = sim_mod._PID(model)
    ctrl = np.array([data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
                     for i in range(model.nu)])

    settled = False
    body_base, feet_base, tgt = None, {}, np.zeros(3)
    ik_ctr = 0
    t_list, body_list = [], []
    feet_list = {f: [] for f in stance_feet}

    try:
        while data.time < ST + LH:
            t = data.time
            if t < ST:
                data.xfrc_applied[:] = 0
                sim_mod._apply_mag(model, data, sphere_gids, plate_ids,
                                   magnet_ids, params)
                data.ctrl[:] = pid.compute(model, data, ctrl, TIMESTEP)
                mujoco.mj_step(model, data)
                continue

            if not settled:
                settled = True
                ik.record_stance(data)
                ee0 = ik.ee_pos(data, SW).copy()
                tgt = ee0 + (np.array([-LDZ, 0., 0.]) if is_wall
                             else np.array([0., 0., LDZ]))
                body_base = data.xpos[body_bid].copy()
                for f, bid in stance_bids.items():
                    feet_base[f] = data.xpos[bid].copy()

            data.xfrc_applied[:] = 0
            sim_mod._apply_mag(model, data, sphere_gids, plate_ids,
                               magnet_ids, params, off_mids={swing_bid})
            ik_ctr += 1
            if ik_ctr >= IKN:
                ik_ctr = 0
                ctrl = ik.solve(tgt, data, IKN * TIMESTEP)
            data.ctrl[:] = pid.compute(model, data, ctrl, TIMESTEP)
            mujoco.mj_step(model, data)

            t_list.append(data.time)
            body_list.append(data.xpos[body_bid].copy())
            for f, bid in stance_bids.items():
                feet_list[f].append(data.xpos[bid].copy())

    except Exception:
        traceback.print_exc()
        return None

    if not t_list:
        return None

    t_rel = np.asarray(t_list) - ST
    return dict(
        t_rel       = t_rel,
        body_pos    = np.asarray(body_list),
        body_base   = body_base,
        feet        = {f: np.asarray(v) for f, v in feet_list.items()},
        feet_base   = feet_base,
        mask        = t_rel >= LMS,
        stance_feet = stance_feet,
        mstart      = LMS,
        is_wall     = is_wall,
    )


def _run_pulloff_trace(params: dict) -> dict | None:
    try:
        records, pf, dt = sim_pulloff_sim.run_headless(
            pull_rate=sim_pulloff_sim.PULL_RATE, params=params)
    except Exception:
        traceback.print_exc()
        return None
    if not records:
        return None
    return dict(
        t             = np.array([r["t"]      for r in records]),
        f_pull        = np.array([r["f_pull"] for r in records]),
        f_mag         = np.array([r["f_mag"]  for r in records]),
        z_disp        = np.array([r["z_disp"] for r in records]),
        pulloff_force = float(pf) if pf else 0.0,
        detach_time   = None if dt is None else float(dt),
        max_adhesion  = float(params["max_force_per_wheel"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — match the optimizer's cost computation exactly
# ─────────────────────────────────────────────────────────────────────────────

def _lift_metrics(tr: dict | None) -> dict:
    if tr is None or not tr["mask"].any():
        return {}
    drifts = np.stack([
        np.abs(tr["feet"][f][tr["mask"]] - tr["feet_base"][f])
        for f in tr["stance_feet"]
    ])                                              # (3_feet, N, 3)
    mean_axis = drifts.mean(axis=(0, 1))           # (3,)
    body_peak = np.abs(
        tr["body_pos"][tr["mask"]] - tr["body_base"]
    ).max(axis=0)                                  # (3,)
    return dict(
        mean_x    = float(mean_axis[0]),
        mean_y    = float(mean_axis[1]),
        mean_z    = float(mean_axis[2]),
        bpeak_x   = float(body_peak[0]),
        bpeak_y   = float(body_peak[1]),
        bpeak_z   = float(body_peak[2]),
        rms       = float(np.sqrt((np.linalg.norm(drifts, axis=2)**2).mean())),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(arr, n=200):
    return np.convolve(arr, np.ones(n)/n, mode="same")


def _shade_window(ax, tr):
    """Grey shading over the optimizer's measurement window."""
    if tr is not None:
        ax.axvspan(tr["mstart"], tr["t_rel"][-1], color=C_REF, alpha=0.10,
                   label="Optimizer window")


def _plot_body_ts(ax, tb, to, axis: int, *, ylabel, title):
    """Body COM deviation along one world axis, mm."""
    for tr, c, name in [(tb, C_BASE, "Baseline"), (to, C_OPT, "Optimized")]:
        if tr is None:
            continue
        dev = (tr["body_pos"][:, axis] - tr["body_base"][axis]) * 1e3
        ax.plot(tr["t_rel"], dev, color=c, lw=1.8, label=name)
    _shade_window(ax, tb or to)
    ax.axhline(0, color=C_REF, lw=0.7, ls="--")
    ax.set(xlabel="Time since lift (s)", ylabel=ylabel); ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)


def _plot_drift_ts(ax, tb, to, axes_idx: list[int], *, ylabel, title):
    """Stance-foot mean drift magnitude over the given world axes, mm.
    Shaded band = min/max across FR / BL / BR."""
    for tr, c, name in [(tb, C_BASE, "Baseline"), (to, C_OPT, "Optimized")]:
        if tr is None:
            continue
        per = np.stack([
            np.sqrt(sum((tr["feet"][f][:, i] - tr["feet_base"][f][i])**2
                        for i in axes_idx))
            for f in tr["stance_feet"]
        ]) * 1e3                                     # (n_feet, N) mm
        ax.fill_between(tr["t_rel"], per.min(0), per.max(0), color=c, alpha=0.15)
        ax.plot(tr["t_rel"], per.mean(0), color=c, lw=1.8,
                label=f"{name} (mean FR/BL/BR)")
    _shade_window(ax, tb or to)
    ax.set_ylim(bottom=0)
    ax.set(xlabel="Time since lift (s)", ylabel=ylabel); ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)


def _bar_panel(ax, mb, mo, keys_labels, *, title):
    if not mb and not mo:
        ax.axis("off"); ax.text(.5,.5,"No data",ha="center",transform=ax.transAxes); return
    labels = [l for _, l in keys_labels]
    x = np.arange(len(keys_labels)); w = 0.35
    bv = np.array([mb.get(k, np.nan) for k, _ in keys_labels]) * 1e3
    ov = np.array([mo.get(k, np.nan) for k, _ in keys_labels]) * 1e3
    ax.bar(x - w/2, bv, w, color=C_BASE, alpha=0.85, label="Baseline")
    ax.bar(x + w/2, ov, w, color=C_OPT,  alpha=0.85, label="Optimized")
    ymax = np.nanmax(np.concatenate([bv, ov])) if np.isfinite(bv).any() else 1
    for i, (b, o) in enumerate(zip(bv, ov)):
        if np.isfinite(b) and np.isfinite(o) and b > 1e-6:
            pct = 100*(b-o)/b
            ax.text(i + w/2, o + .04*ymax,
                    f"{'↓' if pct>=0 else '↑'}{abs(pct):.0f}%",
                    ha="center", fontsize=7,
                    color=C_OPT if pct >= 0 else C_BASE, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set(ylabel="Drift (mm)"); ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7); ax.set_ylim(bottom=0); ax.grid(alpha=0.25, axis="y")


# ─────────────────────────────────────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_lift_figure(fb, fo, wb, wo, out: pathlib.Path):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Single-leg-lift stance stability — baseline vs optimized",
                 fontsize=12, fontweight="bold")

    # ── Floor (row 0) ─────────────────────────────────────────────────────────
    _plot_body_ts(ax[0,0], fb, fo, axis=2,
                  ylabel="Body height deviation (mm)",
                  title="Floor — body Z during FL lift")
    _plot_drift_ts(ax[0,1], fb, fo, [0, 1],
                   ylabel="Stance foot |ΔXY| (mm)",
                   title="Floor — horizontal stance drift √(ΔX²+ΔY²)")
    _bar_panel(ax[0,2], _lift_metrics(fb), _lift_metrics(fo),
               [("mean_x","mean|ΔX|"),("mean_y","mean|ΔY|"),
                ("mean_z","mean|ΔZ|"),("rms","RMS |drift|")],
               title="Floor drift summary")

    # ── Wall (row 1): wall normal = X, gravity = −Z ───────────────────────────
    _plot_body_ts(ax[1,0], wb, wo, axis=0,
                  ylabel="Body stand-off deviation (mm)",
                  title="Wall — body X (wall-normal) during FL lift")
    _plot_drift_ts(ax[1,1], wb, wo, [2],
                   ylabel="Stance foot |ΔZ| gravity drop (mm)",
                   title="Wall — gravity-direction stance drift")
    _bar_panel(ax[1,2], _lift_metrics(wb), _lift_metrics(wo),
               [("mean_x","|ΔX| pull-off"),("mean_y","|ΔY|"),
                ("mean_z","|ΔZ| gravity"),("rms","RMS |drift|")],
               title="Wall drift summary")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [result] {out.name} saved")


def _build_pulloff_figure(pb, po, out: pathlib.Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [2.2, 2.2, 1.0]})
    fig.suptitle("Magnetic pull-off vs expected adhesion range",
                 fontsize=12, fontweight="bold")

    # Expected-adhesion band (grey) — from max_force_per_wheel of each run.
    targets = [p["max_adhesion"] for p in (pb, po) if p is not None]
    tlo = min(targets) if targets else 0.
    thi = max(targets) if targets else 0.
    if tlo == thi > 0:                    # expand to visible band if equal
        tlo, thi = tlo * 0.9, thi * 1.1

    for ax, xkey, xlabel in [
        (axes[0], "t",      "Ramp time (s)"),
        (axes[1], "z_disp", "COM displacement (mm)"),
    ]:
        if targets:
            ax.axhspan(tlo, thi, color=C_REF, alpha=0.18,
                       label=f"Expected adhesion ({tlo:.0f}–{thi:.0f} N)")
        for tr, c, name in [(pb, C_BASE, "Baseline"), (po, C_OPT, "Optimized")]:
            if tr is None:
                continue
            step = max(1, len(tr["t"]) // 1500)
            x = tr[xkey][::step]
            ax.plot(x, tr["f_pull"][::step], color=c, lw=1.0, ls="--",
                    alpha=0.55, label=f"{name}: applied")
            ax.plot(x, _smooth(tr["f_mag"])[::step], color=c, lw=1.8,
                    label=f"{name}: mag. attraction")
            if tr["detach_time"] is not None:
                di = int(np.argmin(np.abs(tr["t"] - tr["detach_time"])))
                ax.axvline(tr[xkey][di], color=c, lw=1., ls=":", alpha=0.8)
                ax.annotate(
                    f"pull-off\n{tr['pulloff_force']:.1f} N",
                    xy=(tr[xkey][di], tr["pulloff_force"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=7, color=c,
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.7))
        ax.set(xlabel=xlabel, ylabel="Force (N)")
        ax.set_title(("Force vs time" if xkey == "t" else "Force vs displacement"),
                     fontsize=10, fontweight="bold")
        ax.set_ylim(bottom=0); ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # Bar chart: expected target vs measured pull-off
    ax = axes[2]
    names, tgts, meas = [], [], []
    for tr, name in [(pb, "Baseline"), (po, "Optimized")]:
        if tr:
            names.append(name); tgts.append(tr["max_adhesion"])
            meas.append(tr["pulloff_force"])
    if names:
        x = np.arange(len(names)); w = 0.38
        ax.bar(x - w/2, tgts, w, color=C_REF,  alpha=0.65, label="Expected target")
        colors = [C_BASE if n == "Baseline" else C_OPT for n in names]
        ax.bar(x + w/2, meas, w, color=colors, alpha=0.90, label="Measured pull-off")
        ymax = max(max(tgts), max(meas))
        for i, (tgt, m) in enumerate(zip(tgts, meas)):
            ax.text(i, -0.12*ymax, f"{100*m/tgt:.0f}%\nof target",
                    ha="center", fontsize=7, color=colors[i], fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
        ax.set(ylabel="Force (N)")
        ax.set_title("Pull-off vs target", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7); ax.set_ylim(bottom=-0.18*ymax)
        ax.grid(alpha=0.25, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [result] {out.name} saved")


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called by combined_optimizer.py or standalone
# ─────────────────────────────────────────────────────────────────────────────

def generate_results_bundle(
    run_dir: pathlib.Path | str | None = None,
    skip: list[str] | None = None,
    baseline_json: str | None = None,
    optimized_json: str | None = None,
) -> pathlib.Path:
    skip = set(skip or [])
    run_dir = pathlib.Path(run_dir or "results/latest")
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline, optimized = _get_params(baseline_json, optimized_json)

    print(f"\n{'='*56}\n  Sally result bundle → {run_dir}\n{'='*56}")

    fb = fo = wb = wo = pb = po = None

    if "floor" not in skip:
        print("\n[floor/baseline] ...")
        t0 = time.time(); fb = _run_lift_trace(sim_opt_sim, baseline, is_wall=False)
        print(f"  {'ok' if fb else 'FAILED'}  ({time.time()-t0:.1f}s)")
        print("[floor/optimized] ...")
        t0 = time.time(); fo = _run_lift_trace(sim_opt_sim, optimized, is_wall=False)
        print(f"  {'ok' if fo else 'FAILED'}  ({time.time()-t0:.1f}s)")

    if "wall" not in skip:
        print("\n[wall/baseline] ...")
        t0 = time.time(); wb = _run_lift_trace(sim_vertopt_sim, baseline, is_wall=True)
        print(f"  {'ok' if wb else 'FAILED'}  ({time.time()-t0:.1f}s)")
        print("[wall/optimized] ...")
        t0 = time.time(); wo = _run_lift_trace(sim_vertopt_sim, optimized, is_wall=True)
        print(f"  {'ok' if wo else 'FAILED'}  ({time.time()-t0:.1f}s)")

    if "pulloff" not in skip:
        print("\n[pulloff/baseline] ...")
        t0 = time.time(); pb = _run_pulloff_trace(baseline)
        print(f"  {'ok' if pb else 'FAILED'}  ({time.time()-t0:.1f}s)"
              + (f"  pull-off={pb['pulloff_force']:.1f} N  "
                 f"target={pb['max_adhesion']:.1f} N" if pb else ""))
        print("[pulloff/optimized] ...")
        t0 = time.time(); po = _run_pulloff_trace(optimized)
        print(f"  {'ok' if po else 'FAILED'}  ({time.time()-t0:.1f}s)"
              + (f"  pull-off={po['pulloff_force']:.1f} N  "
                 f"target={po['max_adhesion']:.1f} N" if po else ""))

    if "floor" not in skip or "wall" not in skip:
        _build_lift_figure(fb, fo, wb, wo, run_dir / "figure_lift_drift.png")
    if "pulloff" not in skip:
        _build_pulloff_figure(pb, po, run_dir / "figure_pulloff.png")

    # metrics_summary.json — consumed by results_3.m
    def _pull_meta(tr):
        if tr is None: return None
        return {"pulloff_force_n": tr["pulloff_force"],
                "max_adhesion_n":  tr["max_adhesion"],
                "detach_time_s":   tr["detach_time"],
                "pct_of_target":   100.*tr["pulloff_force"]/tr["max_adhesion"]
                                   if tr["max_adhesion"] > 0 else None}
    summary = {
        "floor":   {"baseline": _lift_metrics(fb), "optimized": _lift_metrics(fo)},
        "wall":    {"baseline": _lift_metrics(wb), "optimized": _lift_metrics(wo)},
        "pulloff": {"baseline": _pull_meta(pb),    "optimized": _pull_meta(po)},
    }
    with open(run_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [result] metrics_summary.json saved")

    print(f"\n{'='*56}")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",        default=None)
    ap.add_argument("--skip",           action="append", default=[],
                    choices=["floor", "wall", "pulloff"])
    ap.add_argument("--baseline-json",  default=None)
    ap.add_argument("--optimized-json", default=None)
    a = ap.parse_args()
    generate_results_bundle(a.run_dir, a.skip, a.baseline_json, a.optimized_json)