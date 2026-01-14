"""
sim_sally_magnet_wall.py

Stage 1: Headless, modular simulation rollout for parameter optimization.

Purpose
-------
This module provides a fast, deterministic MuJoCo rollout function that:
- Applies physics and magnetic parameters
- Runs a fixed-input simulation (no control optimization)
- Returns trajectory data and summary metrics
- Can be called repeatedly by an optimizer or inspection tool

Visualization, keyboard input, and real-time synchronization are intentionally
excluded from this file. Rendering is handled separately.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple


# =============================================================================
# DEFAULT SIMULATION CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = {
    # MJCF scene file
    "xml_file": "scene.xml",

    # Simulation timestep
    "timestep": 0.001,

    # Solver configuration (can become tunable later)
    "integrator": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "o_solref": [4e-4, 25],
    "o_solimp": [0.99, 0.99, 0.001, 0.5, 2],

    # Magnetic model parameters
    "Br": 1.48 / 1.5,
    "magnet_volume": np.pi * ((0.025 / 2) ** 2 - (0.016 / 2) ** 2) * 0.025,
    "max_total_force": 200.0 * 4,
    "MU_0": 4 * np.pi * 1e-7,

    # Geometry and actuator naming (must match MJCF)
    "wall_geom_name": "wall_geom",
    "wheel_names": ["FL_cyl", "FR_cyl", "BL_cyl", "BR_cyl"],
    "actuator_names": [
        "FL_wheel_motor",
        "FR_wheel_motor",
        "BL_wheel_motor",
        "BR_wheel_motor",
    ],
}

def run_xml_generator(mode: str = "sideways") -> None:
    """
    Runs the XML generator script to produce robot_sally_patched.xml.

    Notes:
    - This does NOT modify scene.xml, so any wall euler rotation in scene.xml stays as-is.
    - This assumes generate_test_magnet_wall_env.py writes robot_sally_patched.xml.
    """
    script_path = Path(__file__).parent / "generate_test_magnet_wall_env.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find generator script: {script_path}")

    env = os.environ.copy()
    # The generator currently hard-codes MODE in-file.
    # If you later add MODE=os.environ.get("MODE"), this will start working automatically.
    env["MODE"] = mode

    print(f"[INFO] Generating patched robot XML (MODE={mode}) ...")
    env["NO_AUTOLAUNCH"] = "1"  # prevent the generator from launching sim
    subprocess.run([sys.executable, str(script_path)], check=True, env=env)



# =============================================================================
# ROLLOUT RESULT STRUCTURE
# =============================================================================
@dataclass
class RolloutResult:
    """
    Container for rollout output.

    termination:
        Reason the rollout ended ("ok" or "unstable")
    steps:
        Number of simulation steps executed
    sim_time:
        Final simulation time in seconds
    trajectory:
        Downsampled state history
    summary:
        Aggregate metrics for evaluation
    params_applied:
        Readback of parameters actually applied to the model
    """
    termination: str
    steps: int
    sim_time: float
    trajectory: List[Dict[str, Any]]
    summary: Dict[str, Any]
    params_applied: Dict[str, Any]


# =============================================================================
# MAGNETIC FORCE MODEL
# =============================================================================
def calculate_magnetic_force(distance: float, Br: float, V: float, MU_0: float) -> float:
    """
    Computes magnetic attraction force magnitude as a function of distance.

    Args:
        distance: Separation between magnet and wall
        Br: Residual flux density
        V: Magnet volume
        MU_0: Magnetic permeability of free space

    Returns:
        Scalar magnetic force magnitude
    """
    if distance <= 0.0:
        return 0.0
    m = (Br * V) / MU_0
    return (3 * MU_0 * m ** 2) / (2 * np.pi * (2 * distance) ** 4)

# =============================================================================
# MODEL CONSTRUCTION AND RESET
# =============================================================================
def build_model(config: Dict[str, Any]) -> Tuple[mujoco.MjModel, Dict[str, Any]]:
    """
    Builds the MuJoCo model and resolves all geometry/body/actuator IDs once.

    Resolving names outside the rollout loop improves performance and
    guarantees consistent indexing during optimization.

    Returns:
        model: MuJoCo model
        ids: Dictionary of resolved IDs needed during rollout
    """
    model = mujoco.MjModel.from_xml_path(config["xml_file"])

    # Deterministic solver configuration
    model.opt.timestep = float(config["timestep"])
    model.opt.integrator = config["integrator"]
    model.opt.o_solref[:] = np.array(config["o_solref"], dtype=np.float64)
    model.opt.o_solimp[:] = np.array(config["o_solimp"], dtype=np.float64)

    # Resolve wall geometry
    wall_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, config["wall_geom_name"]
    )
    if wall_id < 0:
        raise ValueError(f"Wall geometry '{config['wall_geom_name']}' not found")

    # Resolve wheel geometries and corresponding bodies
    wheel_geom_ids = []
    wheel_body_ids = []
    wheel_names = []

    for name in config["wheel_names"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        body_name = name.replace("_cyl", "_wheel_geom")
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        if gid >= 0 and bid >= 0:
            wheel_names.append(name)
            wheel_geom_ids.append(gid)
            wheel_body_ids.append(bid)

    if not wheel_geom_ids:
        raise ValueError("No valid wheel geometries found")

    # Resolve wheel actuators
    wheel_actuator_ids = []
    actuator_names = []
    for name in config["actuator_names"]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            actuator_names.append(name)
            wheel_actuator_ids.append(aid)

    ids = {
        "wall_id": wall_id,
        "wheel_names": wheel_names,
        "wheel_geom_ids": wheel_geom_ids,
        "wheel_body_ids": wheel_body_ids,
        "actuator_names": actuator_names,
        "wheel_actuator_ids": wheel_actuator_ids,
    }
    return model, ids


def reset_data(model: mujoco.MjModel) -> mujoco.MjData:
    """
    Creates a fresh simulation state with cleared forces and controls.

    This ensures deterministic rollouts across optimizer calls.
    """
    data = mujoco.MjData(model)
    data.xfrc_applied[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0
    return data


def apply_params(
    model: mujoco.MjModel, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Applies tunable simulation parameters to the model.

    Returns a readback dictionary to verify that parameters were applied
    correctly (important for optimization sanity checks).
    """
    if "timestep" in params:
        model.opt.timestep = float(params["timestep"])
    if "o_solref" in params:
        model.opt.o_solref[:] = np.array(params["o_solref"], dtype=np.float64)
    if "o_solimp" in params:
        model.opt.o_solimp[:] = np.array(params["o_solimp"], dtype=np.float64)

    return {
        "timestep": float(model.opt.timestep),
        "o_solref": model.opt.o_solref.copy().tolist(),
        "o_solimp": model.opt.o_solimp.copy().tolist(),
    }


# =============================================================================
# HEADLESS ROLLOUT FUNCTION
# =============================================================================
def rollout(
    params: Optional[Dict[str, Any]] = None,
    *,
    sim_duration: float = 5.0,
    config: Optional[Dict[str, Any]] = None,
    fixed_torque: float = 0.5,
    settle_time: float = 0.0,
    log_stride: int = 10,
) -> RolloutResult:
    """
    Runs a headless MuJoCo rollout with fixed actuation.

    Args:
        params: Tunable physics parameters (optimizer input)
        sim_duration: Total simulation time in seconds
        config: Optional override of DEFAULT_CONFIG
        fixed_torque: Constant torque applied to wheel actuators
        settle_time: Optional initial settling time (no logging)
        log_stride: Log every N steps to reduce data volume

    Returns:
        RolloutResult object
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    params = params or {}

    # Regenerate the patched robot XML before loading the model.
    # (scene.xml is not modified here, so your wall euler rotation stays as-is.)
    run_xml_generator(mode=params.get("mode", "sideways"))

    model, ids = build_model(cfg)
    params_applied = apply_params(model, params)
    data = reset_data(model)


    dt = model.opt.timestep
    n_steps = int(np.ceil(sim_duration / dt))

    Br = params.get("Br", cfg["Br"])
    magnet_volume = params.get("magnet_volume", cfg["magnet_volume"])
    MU_0 = params.get("MU_0", cfg["MU_0"])
    n_wheels = max(1, len(ids["wheel_body_ids"]))
    max_force_per_wheel = cfg["max_total_force"] / n_wheels


    fromto = np.zeros(6)
    trajectory = []
    termination = "ok"
    total_force_accum = 0.0
    force_samples = 0

    # Optional settling phase (use ceil so the requested settle_time is honored)
    k = -1
    settle_steps = int(np.ceil(settle_time / dt))
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    try:
        for k in range(n_steps):
            # Clear applied forces to avoid accumulation
            for bid in ids["wheel_body_ids"]:
                data.xfrc_applied[bid, :3] = 0.0

            total_force = np.zeros(3)
            wheel_forces = {}

            # Magnetic force computation
            for name, gid, bid in zip(
                ids["wheel_names"], ids["wheel_geom_ids"], ids["wheel_body_ids"]
            ):
                dist = mujoco.mj_geomDistance(
                    model, data, gid, ids["wall_id"], 50, fromto
                )
                if dist < 0:
                    wheel_forces[name] = 0.0
                    continue

                n = fromto[3:6] - fromto[0:3]
                norm = np.linalg.norm(n)
                if norm < 1e-9:
                    wheel_forces[name] = 0.0
                    continue

                fmag = calculate_magnetic_force(dist, Br, magnet_volume, MU_0)
                fmag = np.clip(fmag, 0.0, max_force_per_wheel)
                wheel_forces[name] = float(fmag)

                fvec = fmag * (n / norm)
                data.xfrc_applied[bid, :3] = fvec
                total_force += fvec

            # Fixed actuation
            for aid in ids["wheel_actuator_ids"]:
                data.ctrl[aid] = fixed_torque

            mujoco.mj_step(model, data)

            # Instability detection
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                termination = "unstable"
                break

            # Downsampled logging
            if k % log_stride == 0 or k == n_steps - 1:
                trajectory.append({
                    "time": float(data.time),
                    "qpos": data.qpos[:7].copy(),
                    "qvel": data.qvel[:6].copy(),
                    "total_force": total_force.copy(),
                    "wheel_forces": wheel_forces,
                })

            total_force_accum += np.linalg.norm(total_force)
            force_samples += 1

    except Exception:
        termination = "unstable"

    summary = {
        "final_base_position": data.qpos[:3].copy().tolist(),
        "avg_total_magnetic_force": (
            total_force_accum / max(1, force_samples)
        ),
    }

    return RolloutResult(
        termination=termination,
        steps=max(0, k + 1),
        sim_time=float(data.time),
        trajectory=trajectory,
        summary=summary,
        params_applied={
            **params_applied,
            "Br": Br,
            "magnet_volume": magnet_volume,
            "MU_0": MU_0,
            "fixed_torque": fixed_torque,
        },
    )


# =============================================================================
# SIMPLE HEADLESS TEST
# =============================================================================
if __name__ == "__main__":
    result = rollout(sim_duration=2.0)
    print("Termination:", result.termination)
    print("Summary:", result.summary)
