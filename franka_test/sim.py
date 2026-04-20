import time
import numpy as np
import mujoco
import mujoco.viewer
import mink

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "mujoco_menagerie/franka_fr3/fr3.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

# ── Mink configuration ────────────────────────────────────────────────────────
configuration = mink.Configuration(model)

EE_SITE = "attachment_site"

tasks = [
    end_effector_task := mink.FrameTask(
        frame_name=EE_SITE,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.1,
        lm_damping=1e-3,
    ),
    posture_task := mink.PostureTask(model, cost=1e-3),
]

limits = [mink.ConfigurationLimit(model)]

posture_task.set_target_from_configuration(configuration)

SOLVER      = "quadprog"
DT          = model.opt.timestep
MAX_IK_ITER = 3

TELEMETRY_INTERVAL = 0.1    # seconds between prints

# ── Motion parameters ─────────────────────────────────────────────────────────
START_POS    = np.array([0.4, 0.0, 0.5])
END_POS      = np.array([0.4, 0.0, 0.7])   # +20cm in Z
SIM_DURATION = 10.0                          # seconds to complete the lift
target_quat  = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz

# ── Joint names for telemetry ─────────────────────────────────────────────────
JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]

def _quintic(t):
    """Smooth quintic interpolation — zero velocity and acceleration at endpoints."""
    return t**3 * (6*t**2 - 15*t + 10)

def get_ee_pos(model, data):
    sid = model.site(EE_SITE).id
    return data.site_xpos[sid].copy()

def print_telemetry(model, data, target_pos):
    ee_pos  = get_ee_pos(model, data)
    ee_err  = (ee_pos - target_pos) * 1000     # mm
    err_mag = np.linalg.norm(ee_err)
    flag    = "!!!" if err_mag > 20 else "ok "

    print(f"t={data.time:6.2f}s | "
          f"EE=({ee_pos[0]:+.3f},{ee_pos[1]:+.3f},{ee_pos[2]:+.3f}) "
          f"tgt=({target_pos[0]:+.3f},{target_pos[1]:+.3f},{target_pos[2]:+.3f}) "
          f"err=({ee_err[0]:+.0f},{ee_err[1]:+.0f},{ee_err[2]:+.0f})mm "
          f"{err_mag:5.1f}mm {flag}")

    parts = []
    for i, jname in enumerate(JOINT_NAMES):
        try:
            jid  = model.joint(jname).id
            qidx = model.jnt_qposadr[jid]
            ctrl = np.degrees(data.ctrl[i])
            qpos = np.degrees(data.qpos[qidx])
            parts.append(f"j{i+1}={ctrl:+5.1f}/{qpos:+5.1f}")
        except Exception:
            pass
    print(f"  joints  {' | '.join(parts)}")

# ── Pause/play state ──────────────────────────────────────────────────────────
_key_state = {"paused": True, "step_once": False}

def _key_callback(keycode):
    if keycode in (32, 257):    # SPACE or ENTER
        _key_state["paused"] = not _key_state["paused"]
        print(f"[sim] {'PAUSED' if _key_state['paused'] else 'RUNNING'}")
    elif keycode == 262:         # RIGHT ARROW — single step
        _key_state["step_once"] = True

# ── Viewer loop ───────────────────────────────────────────────────────────────
print("Press SPACE or ENTER to start. RIGHT ARROW to single-step.")

with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    configuration.update(data.qpos)

    last_print = -TELEMETRY_INTERVAL

    while viewer.is_running():
        # ── Pause handling ────────────────────────────────────────────────────
        if _key_state["paused"] and not _key_state["step_once"]:
            viewer.sync()
            time.sleep(0.02)
            continue
        _key_state["step_once"] = False

        # ── Interpolate target over SIM_DURATION (quintic) ────────────────────
        t_norm     = _quintic(float(np.clip(data.time / SIM_DURATION, 0.0, 1.0)))
        target_pos = (1 - t_norm) * START_POS + t_norm * END_POS
        target_pose = mink.SE3.from_rotation_and_translation(
            mink.SO3(target_quat), target_pos
        )
        end_effector_task.set_target(target_pose)

        # ── IK solve ──────────────────────────────────────────────────────────
        configuration.update(data.qpos)

        for _ in range(MAX_IK_ITER):
            vel = mink.solve_ik(
                configuration,
                tasks,
                DT,
                solver=SOLVER,
                limits=limits,
            )
            configuration.integrate_inplace(vel, DT)

        data.ctrl[:7] = configuration.q[:7]

        # ── Telemetry ─────────────────────────────────────────────────────────
        if data.time - last_print >= TELEMETRY_INTERVAL:
            last_print = data.time
            print_telemetry(model, data, target_pos)

        mujoco.mj_step(model, data)
        viewer.sync()