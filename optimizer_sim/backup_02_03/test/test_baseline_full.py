"""
test_baseline_full.py

Evaluate baseline with full metrics computation AND visual playback.
Run from project root directory.
"""
import os
from contextlib import contextmanager
import sys
from pathlib import Path

# Add parent directory to path to import sim_sally modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer
import numpy as np
import sim_sally_magnet_wall as ssmw
from sim_sally_magnet_wall import (
    rollout,
    run_xml_generator,
    generate_scene_with_robot,
    build_model,
    apply_params,
    reset_data,
    calculate_magnetic_force,
    DEFAULT_CONFIG,
)
from metrics import MetricConfig
from pprint import pprint


@contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))


# Resolve sim asset directory once (CWD-proof baseline script)
SIM_DIR = Path(ssmw.__file__).resolve().parent

# Baseline parameters
BASELINE_PARAMS = {
    "Br": 1.48,
    "o_solref": [4e-4, 25],
    "o_solimp": [0.99, 0.99, 0.001, 0.5, 2.0],
    "wheel_friction": [0.95, 0.05, 0.05],
    "rocker_stiffness": 30.0,
    "rocker_damping": 1.0,
    "wheel_kp": 10.0,
    "wheel_kv": 1.0,
    "max_magnetic_distance": 0.01,
}
# BASELINE_PARAMS = {
#     "Br": 1.609306,
#     "o_solref": [0.0003, 38.6676],
#     "o_solimp": [0.9900, 0.9900, 0.0010, 0.5, 2.0],
#     "wheel_friction": [0.9338, 0.0736, 0.0924],
#     "rocker_stiffness": 51.874403,
#     "rocker_damping": 1.845923,
#     "wheel_kp": 1.212559,
#     "wheel_kv": 1.541354,
#     "max_magnetic_distance": 0.041651,
# }


def run_baseline_with_viewer(mode="drive_up", duration=10.0):
    """Run baseline parameters with visual playback."""
    print("=" * 60)
    print(f"BASELINE VISUAL - {mode.upper()} MODE")
    print("=" * 60)

    # Generate XML files (CWD-proof without modifying sim/gen)
    with pushd(SIM_DIR):
        patched_robot_file_rel = run_xml_generator(mode=mode)
        temp_scene_file_rel = generate_scene_with_robot(patched_robot_file_rel)

    patched_robot_file = str((SIM_DIR / patched_robot_file_rel).resolve())
    temp_scene_file = str((SIM_DIR / temp_scene_file_rel).resolve())

    try:
        # Build configuration
        cfg = dict(DEFAULT_CONFIG)
        cfg["xml_file"] = temp_scene_file

        # Build model and apply parameters
        model, ids = build_model(cfg)
        _params_applied = apply_params(model, BASELINE_PARAMS, cfg)
        data = reset_data(model, cfg)

        # Extract parameters
        dt = model.opt.timestep
        Br = BASELINE_PARAMS.get("Br", cfg["Br"])
        magnet_volume = cfg["magnet_volume"]
        MU_0 = cfg["MU_0"]
        max_magnetic_distance = BASELINE_PARAMS.get(
            "max_magnetic_distance", cfg["max_magnetic_distance"]
        )
        n_wheels = len(ids["wheel_body_ids"])
        max_force_per_wheel = cfg["max_total_force"] / n_wheels
        target_speed = cfg.get("target_wheel_speed_rad_s", 5.0)
        distance_threshold = 0.03

        # Settling phase
        print(f"\nSettling for 1 second...")
        settle_steps = int(1.0 / dt)
        fromto = np.zeros(6)

        for _ in range(settle_steps):
            for name, gid, bid in zip(
                ids["wheel_names"], ids["wheel_geom_ids"], ids["wheel_body_ids"]
            ):
                data.xfrc_applied[bid, :3] = 0.0
                dist = mujoco.mj_geomDistance(
                    model, data, gid, ids["wall_id"], 50, fromto
                )

                if dist >= 0 and dist <= max_magnetic_distance:
                    n = fromto[3:6] - fromto[0:3]
                    norm = np.linalg.norm(n)
                    if norm > 1e-9:
                        fmag = calculate_magnetic_force(dist, Br, magnet_volume, MU_0)
                        fmag = np.clip(fmag, 0.0, max_force_per_wheel)
                        fvec = fmag * (n / norm)
                        data.xfrc_applied[bid, :3] = fvec

            mujoco.mj_step(model, data)

        print("Launching viewer...")
        print("\nCONTROLS: Mouse drag=Rotate, Scroll=Zoom, ESC=Exit\n")

        # Track metrics
        contact_steps = 0
        total_steps = 0
        start_pos = data.qpos[2].copy()

        # Launch viewer with simulation loop
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20

            n_steps = int(duration / dt)

            for k in range(n_steps):
                for bid in ids["wheel_body_ids"]:
                    data.xfrc_applied[bid, :3] = 0.0

                has_contact = False

                for name, gid, bid in zip(
                    ids["wheel_names"], ids["wheel_geom_ids"], ids["wheel_body_ids"]
                ):
                    dist = mujoco.mj_geomDistance(
                        model, data, gid, ids["wall_id"], 50, fromto
                    )

                    if dist >= 0 and dist <= max_magnetic_distance:
                        n = fromto[3:6] - fromto[0:3]
                        norm = np.linalg.norm(n)
                        if norm > 1e-9:
                            fmag = calculate_magnetic_force(
                                dist, Br, magnet_volume, MU_0
                            )
                            fmag = np.clip(fmag, 0.0, max_force_per_wheel)
                            fvec = fmag * (n / norm)
                            data.xfrc_applied[bid, :3] = fvec

                    if dist >= 0 and dist <= distance_threshold:
                        has_contact = True

                if has_contact:
                    contact_steps += 1
                total_steps += 1

                for aid in ids["wheel_actuator_ids"]:
                    data.ctrl[aid] = data.time * target_speed

                mujoco.mj_step(model, data)
                viewer.sync()

                if not viewer.is_running():
                    break

        # Print results
        end_pos = data.qpos[2]
        progress = end_pos - start_pos
        contact_pct = (contact_steps / total_steps * 100) if total_steps > 0 else 0

        print(f"\nContact: {contact_pct:.1f}% ({contact_steps}/{total_steps} steps)")
        if mode == "drive_up":
            print(f"Vertical progress: {progress:.4f} m")

    finally:
        # Clean up generated files from viewer path
        for f in (patched_robot_file, temp_scene_file):
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass


# Main execution
print("=" * 70)
print("BASELINE EVALUATION - FULL METRICS")
print("=" * 70)

# Test HOLD mode
print("\n--- HOLD MODE (Sideways) ---")
params_hold = dict(BASELINE_PARAMS)
params_hold["mode"] = "sideways"
params_hold["rollout_mode"] = "hold"

with pushd(SIM_DIR):
    result_hold = rollout(
        params=params_hold,
        sim_duration=5.0,
        settle_time=1.0,
        log_stride=10,
    )

print(f"\nTermination: {result_hold.termination}")
print(f"Reward: {result_hold.summary['reward']:.3f}")
print(f"Contact: {result_hold.summary.get('contact_percentage', 0)*100:.1f}%")
print(f"Sideways slip: {result_hold.summary.get('slip_m', 0):.4f} m")

# Test DRIVE mode
print("\n--- DRIVE MODE (Climbing) ---")
params_drive = dict(BASELINE_PARAMS)
params_drive["mode"] = "drive_up"
params_drive["rollout_mode"] = "drive"

with pushd(SIM_DIR):
    result_drive = rollout(
        params=params_drive,
        sim_duration=5.0,
        settle_time=1.0,
        log_stride=10,
    )

print(f"\nTermination: {result_drive.termination}")
print(f"Reward: {result_drive.summary['reward']:.3f}")
print(f"Contact: {result_drive.summary.get('contact_percentage', 0)*100:.1f}%")
print(f"Vertical progress: {result_drive.summary.get('progress_m', 0):.4f} m")

# Combined results
reward_hold = result_hold.summary.get("reward", -1e9)
reward_drive = result_drive.summary.get("reward", -1e9)
combined_reward = min(reward_hold, reward_drive)

print("\n" + "=" * 70)
print("BASELINE SUMMARY")
print("=" * 70)
print(f"{'Metric':<25} {'Hold':<20} {'Drive':<20}")
print("-" * 70)
print(f"{'Reward':<25} {reward_hold:<20.3f} {reward_drive:<20.3f}")
print(
    f"{'Contact %':<25} "
    f"{result_hold.summary.get('contact_percentage', 0)*100:<20.1f} "
    f"{result_drive.summary.get('contact_percentage', 0)*100:<20.1f}"
)
print(
    f"{'Detached?':<25} "
    f"{str(result_hold.summary.get('detached', True)):<20} "
    f"{str(result_drive.summary.get('detached', True)):<20}"
)
print(
    f"{'Slip/Progress (m)':<25} "
    f"{result_hold.summary.get('slip_m', 0):<20.4f} "
    f"{result_drive.summary.get('progress_m', 0):<20.4f}"
)
print(f"{'Termination':<25} {result_hold.termination:<20} {result_drive.termination:<20}")
print("-" * 70)
print(f"{'Combined Reward':<25} {combined_reward:<20.3f}")
print("=" * 70)

# Ask if user wants visual playback
response = input("\nWatch visual playback? (y/n): ").strip().lower()

if response == "y":
    print("\n🎬 Playing DRIVE mode...")
    run_baseline_with_viewer(mode="drive_up", duration=10.0)

    response2 = input("\nWatch SIDEWAYS mode? (y/n): ").strip().lower()
    if response2 == "y":
        print("\n🎬 Playing SIDEWAYS mode...")
        run_baseline_with_viewer(mode="sideways", duration=10.0)

print("\n✅ Baseline evaluation complete!")
