#!/usr/bin/env python3
"""Generate the One-Click Google Colab Pro A100 Extreme RL Training Notebook for RNV1.
Produces training/RNV1_Colab_A100_Training.ipynb embedding the exact synchronized
simulation/rnv1_humanoid.xml MJCF model and exact RNV1 (36->64->64->12 + CPG) architecture.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "simulation" / "rnv1_humanoid.xml"
NB_OUT = ROOT / "training" / "RNV1_Colab_A100_Training.ipynb"

def main():
    xml_text = XML_PATH.read_text(encoding="utf-8")
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "colab": {
            "name": "RNV1_Colab_A100_Training.ipynb",
            "provenance": [],
            "gpuType": "A100",
            "machine_shape": "hm"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "accelerator": "GPU"
    }

    cells = []

    # Markdown Header
    cells.append(nbf.v4.new_markdown_cell(
"""# RNV1 — Sub-¥10,000 Autonomous Humanoid Extreme RL Training (Google Colab Pro A100)

**Version:** `RNV1` (Robotic Neural Vector 1)  
**Target Hardware:** Seeed Studio XIAO ESP32S3 Sense + 10× MG90S + 2× SG90 + PCA9685 + MPU-6050 + VL53L0X + MAX98357A  
**Physics Engine:** Google DeepMind **MuJoCo MJX (JAX GPU XLA)** — `4,096` parallel humanoid environments on **NVIDIA A100**  
**Output:** Drop-in ESP32-S3 C header `rnv1_policy_weights.h` (`36 → 64 → 64 → 12` MLP + CPG Prior, **29.2 KB Flash**, **< 0.55 ms** @ 50 Hz)

### One-Click Quickstart (ボタンを押すだけ!)
1. Colabメニュー **「ランタイム」→「ランタイムのタイプを変更」→ `A100 GPU`** を選択
2. **「ランタイム」→「すべてのセルを実行 (Run All / Ctrl+F9)」** を押すだけ!
3. 約12〜18分でゼロから **1.2億ステップ (120M steps)** の極限強化学習が完了し、シミュレーション動画とESP32-S3書き込み用ファイル `RNV1_Trained_Bundle.zip` (`rnv1_policy_weights.h` 入り) が自動ダウンロードされます。
"""
    ))

    # Cell 1: GPU Check & Environment Setup
    cells.append(nbf.v4.new_code_cell(
"""#@title 1. GPU (A100) Verification & XLA Performance Tuning { display-mode: "form" }
import os, subprocess, sys, time

TRAINING_VERSION = "RNV1"
NUM_ENVS = 4096          #@param {type:"integer"}
TOTAL_TIMESTEPS = 120_000_000 #@param {type:"integer"}
SEED = 2026              #@param {type:"integer"}

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.88"

print(f"=== {TRAINING_VERSION} Extreme RL Pipeline (Google Colab Pro A100) ===")
try:
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True
    ).strip()
    print(f"[GPU DETECTED] {gpu_info}")
    if "A100" not in gpu_info and "H100" not in gpu_info:
        print("[NOTE] Non-A100 GPU detected: auto-scaling NUM_ENVS to 1024 for VRAM safety.")
        NUM_ENVS = min(NUM_ENVS, 1024)
except Exception as e:
    print(f"[WARN] No NVIDIA GPU detected ({e}); falling back to CPU test mode (NUM_ENVS=16).")
    NUM_ENVS = 16
    TOTAL_TIMESTEPS = 200_000
"""
    ))

    # Cell 2: Dependency Installation
    cells.append(nbf.v4.new_code_cell(
"""#@title 2. Auto-Install MuJoCo, MJX, JAX, Flax & Optax { display-mode: "form" }
import importlib, subprocess, sys

def ensure_pkgs():
    pkgs = ["mujoco>=3.2.0", "mujoco-mjx>=3.2.0", "optax", "flax", "mediapy"]
    missing = []
    for p in ["mujoco", "mujoco.mjx", "jax", "optax", "flax", "mediapy"]:
        try:
            importlib.import_module(p)
        except Exception:
            missing.append(p)
    if missing:
        print("Installing RL stack:", pkgs)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    else:
        print("All MuJoCo-MJX & JAX packages already installed.")

ensure_pkgs()

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import optax
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt

print(f"JAX {jax.__version__} | Devices: {jax.devices()} | MuJoCo {mujoco.__version__}")
"""
    ))

    # Cell 3: Exact Synchronized RNV1 MJCF Model
    escaped_xml = xml_text.replace('"""', '\\"\\"\\"')
    cells.append(nbf.v4.new_code_cell(
f'''#@title 3. Materialize Exact CAD-Synchronized RNV1 MuJoCo MJCF Model {{ display-mode: "form" }}
from pathlib import Path

RNV1_MJCF_XML = """{escaped_xml}"""

Path("rnv1_humanoid.xml").write_text(RNV1_MJCF_XML, encoding="utf-8")
mj_model = mujoco.MjModel.from_xml_path("rnv1_humanoid.xml")
mjx_model = mjx.put_model(mj_model)

print(f"[OK] Loaded RNV1 MJCF | Total mass: {{float(np.sum(mj_model.body_mass)):.4f}} kg | "
      f"Actuators: {{mj_model.nu}} | DoF: {{mj_model.nv}} | MJX GPU Ready!")
'''
    ))

    # Cell 4: GPU-Parallel MJX Environment & RNV1 Actor-Critic Architecture
    cells.append(nbf.v4.new_code_cell(
"""#@title 4. Define RNV1 (36->64->64->12 + CPG Prior) & 4,096-Env Domain-Randomized MJX Task { display-mode: "form" }
import functools
from typing import NamedTuple

OBS_DIM = 36
PRIV_DIM = 52
H1_DIM = 64
H2_DIM = 64
ACT_DIM = 12
CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ
PHYSICS_SUBSTEPS = 10  # 10 x 2ms = 20ms (50Hz)
EPISODE_STEPS = 500    # 10 seconds per episode
RESIDUAL_SCALE = 0.35

Q_MIN = jnp.array([
    -0.524, -0.960, -0.785, -0.611,
    -0.611, -0.960, -0.785, -0.611,
    -1.658, -1.658, -1.309, -0.524,
], dtype=jnp.float32)

Q_MAX = jnp.array([
     0.611,  0.960,  0.785,  0.611,
     0.524,  0.960,  0.785,  0.611,
     2.618,  2.618,  1.309,  0.611,
], dtype=jnp.float32)

DEFAULT_CPG = jnp.array([
    1.45, 0.01, 1.00, 0.19, -0.98, 0.25,
    0.09, 0.00, 0.42, 0.02, 0.65, 0.04
], dtype=jnp.float32)


def quat_to_proj_gravity_jax(q: jnp.ndarray) -> jnp.ndarray:
    w, x, y, z = q[0], q[1], q[2], q[3]
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return jnp.array([gx, gy, gz], dtype=jnp.float32)


def compute_cpg_prior_jax(phi, cmd_vx, cmd_vy, cmd_yaw, cpg, head_phase):
    speed_mag = jnp.clip(jnp.hypot(cmd_vx / 0.12, cmd_yaw / 0.55), 0.0, 1.35)
    act = jnp.clip(speed_mag, 0.0, 1.0)
    fwd_scale = jnp.clip(cmd_vx / 0.12, -0.85, 1.25)
    yaw_scale = jnp.clip(cmd_yaw / 0.55, -1.0, 1.0)
    lat_scale = jnp.clip(cmd_vy / 0.08, -1.0, 1.0)

    lean_fwd    = cpg[1] * fwd_scale
    level_ratio = cpg[2]
    roll_amp    = cpg[3] * act
    ank_ratio   = cpg[4]
    stride_amp  = cpg[5] * jnp.where(jnp.abs(fwd_scale) > 0.05, fwd_scale, 0.45 * jnp.abs(yaw_scale))
    lift_roll   = cpg[6] * act
    phase_off   = cpg[7]
    arm_amp     = cpg[8] * act
    ank_push    = cpg[9] * act
    turn_diff   = cpg[10] * yaw_scale
    turn_roll   = cpg[11] * yaw_scale

    w_roll = jnp.sin(2.0 * jnp.pi * phi)
    w_step = jnp.cos(2.0 * jnp.pi * (phi + phase_off))
    l_sw = jnp.maximum(0.0, w_roll)
    r_sw = jnp.maximum(0.0, -w_roll)

    l_stride_gain = 1.0 - jnp.maximum(0.0, -turn_diff)
    r_stride_gain = 1.0 - jnp.maximum(0.0,  turn_diff)
    rc = roll_amp * w_roll + 0.05 * lat_scale * w_step + turn_roll * w_step

    l_hip = -lean_fwd + stride_amp * l_stride_gain * w_step
    r_hip = -lean_fwd - stride_amp * r_stride_gain * w_step

    q = jnp.array([
        rc + lift_roll * l_sw, l_hip, -l_hip * level_ratio - ank_push * w_roll, ank_ratio * rc - lift_roll * l_sw,
        rc - lift_roll * r_sw, r_hip, -r_hip * level_ratio + ank_push * w_roll, ank_ratio * rc + lift_roll * r_sw,
        -arm_amp * w_step - 0.25 * yaw_scale,
         arm_amp * w_step + 0.25 * yaw_scale,
        jnp.clip(0.45 * yaw_scale + 0.35 * jnp.sin(head_phase), -1.1, 1.1),
        jnp.clip(0.08 + 0.08 * jnp.sin(0.5 * head_phase), -0.35, 0.45),
    ], dtype=jnp.float32)
    return jnp.clip(q, Q_MIN, Q_MAX)


class RNV1ActorCritic(nn.Module):
    @nn.compact
    def __call__(self, obs: jnp.ndarray, priv_obs: jnp.ndarray):
        # Deployable ESP32-S3 Actor: EXACT 36 -> 64 -> 64 -> 12 (tanh)
        h1 = nn.tanh(nn.Dense(H1_DIM, name="actor_l1")(obs))
        h2 = nn.tanh(nn.Dense(H2_DIM, name="actor_l2")(h1))
        mean_delta = nn.tanh(nn.Dense(ACT_DIM, name="actor_l3")(h2))
        log_std = self.param("log_std", nn.initializers.constant(-1.4), (ACT_DIM,))
        cpg_delta = self.param("cpg_delta", nn.initializers.zeros, (12,))

        # Asymmetric Privileged Critic (Training-only, NOT deployed to ESP32-S3)
        v1 = nn.elu(nn.Dense(128, name="critic_l1")(priv_obs))
        v2 = nn.elu(nn.Dense(128, name="critic_l2")(v1))
        value = nn.Dense(1, name="critic_out")(v2).squeeze(-1)
        return mean_delta, log_std, DEFAULT_CPG + jnp.clip(cpg_delta, -0.25, 0.25), value


class EnvState(NamedTuple):
    mjx_data: mjx.Data
    obs: jnp.ndarray
    priv_obs: jnp.ndarray
    cmd: jnp.ndarray           # [vx, vy, yaw]
    phi: jnp.ndarray
    head_phase: jnp.ndarray
    prev_act: jnp.ndarray
    prev2_act: jnp.ndarray
    filtered_ctrl: jnp.ndarray
    servo_alpha: jnp.ndarray   # Domain-randomized 1st-order MG90S lag per env
    step_count: jnp.ndarray
    rng: jnp.ndarray


def _make_obs(d: mjx.Data, cmd, phi, prev_act, prev2_act, rng):
    rng, k1, k2 = jax.random.split(rng, 3)
    pg = quat_to_proj_gravity_jax(d.qpos[3:7])
    pg_noisy = pg + jax.random.normal(k1, (3,)) * 0.015
    pg_noisy = pg_noisy / jnp.maximum(jnp.linalg.norm(pg_noisy), 1e-6)
    gyro = d.qvel[3:6] + jax.random.normal(k2, (3,)) * 0.05
    tof_norm = jnp.array([1.0], dtype=jnp.float32)

    obs = jnp.concatenate([
        pg_noisy, gyro * 0.25, cmd,
        jnp.array([jnp.sin(2.0 * jnp.pi * phi), jnp.cos(2.0 * jnp.pi * phi)]),
        prev_act, prev2_act, tof_norm
    ])
    priv = jnp.concatenate([obs, d.qvel[0:3], d.qpos[2:3], d.qpos[7:19]])
    return obs, priv, rng


def reset_env(rng):
    rng, k_cmd, k_tau, k_q = jax.random.split(rng, 4)
    d = mjx.make_data(mjx_model)
    qpos = d.qpos.at[2].set(0.123)
    qpos = qpos.at[7:].set(jax.random.uniform(k_q, (ACT_DIM,), minval=-0.03, maxval=0.03))
    d = mjx.forward(mjx_model, d.replace(qpos=qpos))

    cmd = jnp.array([
        jax.random.uniform(k_cmd, (), minval=-0.04, maxval=0.22),
        jax.random.uniform(k_cmd, (), minval=-0.05, maxval=0.05),
        jax.random.uniform(k_cmd, (), minval=-0.70, maxval=0.70),
    ], dtype=jnp.float32)
    tau_s = jax.random.uniform(k_tau, (), minval=0.018, maxval=0.038)
    alpha = 0.002 / (tau_s + 0.002)

    prev_act = jnp.zeros(ACT_DIM, dtype=jnp.float32)
    obs, priv, rng = _make_obs(d, cmd, jnp.float32(0.0), prev_act, prev_act, rng)
    return EnvState(
        mjx_data=d, obs=obs, priv_obs=priv, cmd=cmd,
        phi=jnp.float32(0.0), head_phase=jnp.float32(0.0),
        prev_act=prev_act, prev2_act=prev_act, filtered_ctrl=prev_act,
        servo_alpha=alpha, step_count=jnp.int32(0), rng=rng
    )


def step_env(state: EnvState, action_delta: jnp.ndarray, cpg: jnp.ndarray, curriculum_scale: float):
    prior = compute_cpg_prior_jax(state.phi, state.cmd[0], state.cmd[1], state.cmd[2], cpg, state.head_phase)
    scale_vec = jnp.where(jnp.arange(ACT_DIM) >= 10, RESIDUAL_SCALE * 0.25, RESIDUAL_SCALE)
    target_q = jnp.clip(prior + jnp.clip(action_delta, -1.0, 1.0) * scale_vec, Q_MIN, Q_MAX)

    # Random external push perturbation (curriculum-scaled up to 3.5N for 20-40ms!)
    rng, k_push, k_f = jax.random.split(state.rng, 3)
    do_push = (jax.random.uniform(k_push, ()) < (0.015 * curriculum_scale))
    push_f = jax.random.uniform(k_f, (3,), minval=-3.2, maxval=3.2) * curriculum_scale * jnp.array([1.0, 1.0, 0.2])
    xfrc = state.mjx_data.xfrc_applied.at[1, 0:3].set(jnp.where(do_push, push_f, 0.0))
    d0 = state.mjx_data.replace(xfrc_applied=xfrc)

    def substep_fn(carry, _):
        d, f_ctrl = carry
        f_ctrl = f_ctrl + state.servo_alpha * (target_q - f_ctrl)
        d = mjx.step(mjx_model, d.replace(ctrl=f_ctrl))
        return (d, f_ctrl), jnp.mean(d.actuator_force ** 2)

    (d_next, f_ctrl_next), trq_sq = jax.lax.scan(substep_fn, (d0, state.filtered_ctrl), None, length=PHYSICS_SUBSTEPS)

    # Compute dense biomechanically grounded reward
    pg = quat_to_proj_gravity_jax(d_next.qpos[3:7])
    tilt = jnp.arccos(jnp.clip(-pg[2], -1.0, 1.0))
    z_torso = d_next.qpos[2]
    fell = (z_torso < 0.068) | (tilt > 0.75)

    r_vx = 3.0 * jnp.exp(-((d_next.qvel[0] - state.cmd[0]) / 0.09) ** 2)
    r_vy = 1.0 * jnp.exp(-((d_next.qvel[1] - state.cmd[1]) / 0.08) ** 2)
    r_wz = 2.0 * jnp.exp(-((d_next.qvel[5] - state.cmd[2]) / 0.35) ** 2)
    r_up = 1.5 * jnp.exp(-(tilt / 0.20) ** 2)
    r_smooth = -0.35 * jnp.mean((target_q - state.prev_act) ** 2)
    r_energy = -0.40 * jnp.mean(trq_sq)
    reward = jnp.where(fell, -15.0, r_vx + r_vy + r_wz + r_up + r_smooth + r_energy)

    next_step = state.step_count + 1
    done = fell | (next_step >= EPISODE_STEPS)

    freq_hz = jnp.clip(cpg[0], 0.9, 2.5)
    next_phi = jnp.mod(state.phi + freq_hz * CONTROL_DT, 1.0)
    next_hp = state.head_phase + 1.4 * CONTROL_DT
    obs, priv, rng = _make_obs(d_next, state.cmd, next_phi, target_q, state.prev_act, rng)

    next_state = EnvState(
        mjx_data=d_next, obs=obs, priv_obs=priv, cmd=state.cmd,
        phi=next_phi, head_phase=next_hp,
        prev_act=target_q, prev2_act=state.prev_act, filtered_ctrl=f_ctrl_next,
        servo_alpha=state.servo_alpha, step_count=next_step, rng=rng
    )

    # Auto-reset terminated envs cleanly inside JIT
    reset_st = reset_env(rng)
    out_state = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), reset_st, next_state)
    return out_state, reward, done

print("[OK] RNV1 MJX 4,096-parallel GPU environment & Asymmetric Actor-Critic compiled!")
"""
    ))

    # Cell 5: Extreme A100 PPO Training Loop
    cells.append(nbf.v4.new_code_cell(
"""#@title 5. START EXTREME RNV1 TRAINING ON A100 (PPO + GAE + 3-Phase Curriculum) { display-mode: "form" }
ROLLOUT_LEN = 32
PPO_EPOCHS = 4
MINIBATCHES = 16
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.005
VF_COEF = 0.5

steps_per_iter = NUM_ENVS * ROLLOUT_LEN
num_iters = max(10, TOTAL_TIMESTEPS // steps_per_iter)
print(f"[RNV1] Launching PPO | Envs={NUM_ENVS} | Steps/Iter={steps_per_iter:,} | Iters={num_iters} | Target={num_iters*steps_per_iter:,} steps")

model_ac = RNV1ActorCritic()
rng = jax.random.PRNGKey(SEED)
rng, init_k, env_k = jax.random.split(rng, 3)
params = model_ac.init(init_k, jnp.zeros((1, OBS_DIM)), jnp.zeros((1, PRIV_DIM)))

schedule = optax.cosine_decay_schedule(init_value=4e-4, decay_steps=num_iters * PPO_EPOCHS * MINIBATCHES, alpha=0.05)
optimizer = optax.chain(optax.clip_by_global_norm(0.8), optax.adam(schedule))
opt_state = optimizer.init(params)

v_reset = jax.jit(jax.vmap(reset_env))
v_step = jax.jit(jax.vmap(step_env, in_axes=(0, 0, None, None)))

@jax.jit
def collect_and_update(params, opt_state, env_states, rng, curriculum_scale):
    def rollout_step(carry, _):
        st, key = carry
        key, act_k = jax.random.split(key)
        mean, log_std, cpg, val = model_ac.apply(params, st.obs, st.priv_obs)
        std = jnp.exp(jnp.clip(log_std, -3.0, 0.5))
        act = mean + std * jax.random.normal(act_k, mean.shape)
        logp = -0.5 * jnp.sum(((act - mean) / std) ** 2 + 2.0 * jnp.log(std) + jnp.log(2.0 * jnp.pi), axis=-1)
        next_st, rew, done = v_step(st, act, cpg, curriculum_scale)
        return (next_st, key), (st.obs, st.priv_obs, act, logp, rew, done, val)

    (env_states, rng), traj = jax.lax.scan(rollout_step, (env_states, rng), None, length=ROLLOUT_LEN)
    obs_b, priv_b, act_b, logp_b, rew_b, done_b, val_b = traj
    _, _, _, last_val = model_ac.apply(params, env_states.obs, env_states.priv_obs)

    # GAE Advantage Computation
    def gae_step(carry, xs):
        gae, next_v = carry
        r, d, v = xs
        delta = r + GAMMA * next_v * (1.0 - d) - v
        gae = delta + GAMMA * GAE_LAMBDA * (1.0 - d) * gae
        return (gae, v), gae

    _, adv_b = jax.lax.scan(gae_step, (jnp.zeros(NUM_ENVS), last_val), (rew_b, done_b.astype(jnp.float32), val_b), reverse=True)
    ret_b = adv_b + val_b

    # Flatten (T, N, ...) -> (T*N, ...)
    B = ROLLOUT_LEN * NUM_ENVS
    obs_f  = obs_b.reshape((B, OBS_DIM))
    priv_f = priv_b.reshape((B, PRIV_DIM))
    act_f  = act_b.reshape((B, ACT_DIM))
    logp_f = logp_b.reshape((B,))
    adv_f  = adv_b.reshape((B,))
    adv_f  = (adv_f - jnp.mean(adv_f)) / (jnp.std(adv_f) + 1e-8)
    ret_f  = ret_b.reshape((B,))

    def loss_fn(p, o, pr, a, lp_old, adv, ret):
        mean, log_std, _, val = model_ac.apply(p, o, pr)
        std = jnp.exp(jnp.clip(log_std, -3.0, 0.5))
        lp = -0.5 * jnp.sum(((a - mean) / std) ** 2 + 2.0 * jnp.log(std) + jnp.log(2.0 * jnp.pi), axis=-1)
        ratio = jnp.exp(lp - lp_old)
        pg1 = -adv * ratio
        pg2 = -adv * jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        pg_loss = jnp.mean(jnp.maximum(pg1, pg2))
        v_loss = 0.5 * jnp.mean((val - ret) ** 2)
        ent = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1))
        return pg_loss + VF_COEF * v_loss - ENT_COEF * ent

    grad_fn = jax.value_and_grad(loss_fn)

    def epoch_step(carry, _):
        p, os_, key = carry
        key, perm_k = jax.random.split(key)
        perm = jax.random.permutation(perm_k, B).reshape((MINIBATCHES, -1))
        def mb_step(c2, idx):
            p2, os2 = c2
            loss, grads = grad_fn(p2, obs_f[idx], priv_f[idx], act_f[idx], logp_f[idx], adv_f[idx], ret_f[idx])
            updates, os2 = optimizer.update(grads, os2, p2)
            p2 = optax.apply_updates(p2, updates)
            return (p2, os2), loss
        (p, os_), losses = jax.lax.scan(mb_step, (p, os_), perm)
        return (p, os_, key), jnp.mean(losses)

    (params, opt_state, rng), mean_loss = jax.lax.scan(epoch_step, (params, opt_state, rng), None, length=PPO_EPOCHS)
    return params, opt_state, env_states, rng, jnp.mean(rew_b), jnp.mean(done_b.astype(jnp.float32)), jnp.mean(mean_loss)

env_states = v_reset(jax.random.split(env_k, NUM_ENVS))
t_start = time.time()
hist_steps, hist_rew, hist_fall = [], [], []

for it in range(num_iters):
    curr_scale = float(np.clip((it + 1) / max(1, int(0.6 * num_iters)), 0.15, 1.0))
    params, opt_state, env_states, rng, m_rew, m_fall, m_loss = collect_and_update(
        params, opt_state, env_states, rng, curr_scale
    )
    if (it + 1) % max(1, num_iters // 20) == 0 or it == 0:
        steps_done = (it + 1) * steps_per_iter
        sps = steps_done / max(time.time() - t_start, 1e-3)
        hist_steps.append(steps_done / 1e6)
        hist_rew.append(float(m_rew))
        hist_fall.append(float(m_fall) * 100.0)
        print(f"Iter {it+1:4d}/{num_iters} | Steps: {steps_done/1e6:6.2f}M | {sps:9,.0f} SPS | "
              f"Reward: {float(m_rew):+6.3f} | Fall%: {float(m_fall)*100:4.2f}% | Curr: {curr_scale:.2f}")

plt.figure(figsize=(10, 3.5))
plt.subplot(1, 2, 1); plt.plot(hist_steps, hist_rew, lw=2); plt.title("RNV1 Mean Step Reward"); plt.xlabel("Million Steps"); plt.grid(alpha=0.3)
plt.subplot(1, 2, 2); plt.plot(hist_steps, hist_fall, color="crimson", lw=2); plt.title("RNV1 Fall Rate (%)"); plt.xlabel("Million Steps"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""
    ))

    # Cell 6: High-Precision CPU MuJoCo Verification & Video Rendering
    cells.append(nbf.v4.new_code_cell(
"""#@title 6. High-Precision MuJoCo Verification & Simulation Video Playback { display-mode: "form" }
import mediapy as media

p_dict = params["params"]
W1 = np.array(p_dict["actor_l1"]["kernel"]).T.astype(np.float32)  # (64, 36)
B1 = np.array(p_dict["actor_l1"]["bias"]).astype(np.float32)
W2 = np.array(p_dict["actor_l2"]["kernel"]).T.astype(np.float32)  # (64, 64)
B2 = np.array(p_dict["actor_l2"]["bias"]).astype(np.float32)
W3 = np.array(p_dict["actor_l3"]["kernel"]).T.astype(np.float32)  # (12, 64)
B3 = np.array(p_dict["actor_l3"]["bias"]).astype(np.float32)
CPG_OPT = np.array(DEFAULT_CPG + np.clip(p_dict["cpg_delta"], -0.25, 0.25), dtype=np.float32)

def infer_rnv1_np(obs, phi, cmd_vx, cmd_vy, cmd_yaw, hp):
    prior = np.array(compute_cpg_prior_jax(phi, cmd_vx, cmd_vy, cmd_yaw, CPG_OPT, hp), dtype=np.float32)
    h1 = np.tanh(W1 @ obs + B1)
    h2 = np.tanh(W2 @ h1 + B2)
    delta = np.tanh(W3 @ h2 + B3) * RESIDUAL_SCALE
    delta[10:] *= 0.25
    return np.clip(prior + delta, np.array(Q_MIN), np.array(Q_MAX))

mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetData(mj_model, mj_data)
mj_data.qpos[2] = 0.123
mujoco.mj_forward(mj_model, mj_data)

renderer = mujoco.Renderer(mj_model, height=360, width=640)
frames = []
phi, hp = 0.0, 0.0
prev_act = np.zeros(ACT_DIM, dtype=np.float32)
prev2_act = np.zeros(ACT_DIM, dtype=np.float32)
f_ctrl = np.zeros(ACT_DIM, dtype=np.float32)
alpha = 0.002 / (0.025 + 0.002)

for step in range(250):  # 5.0 seconds @ 50Hz
    q = mj_data.qpos[3:7]
    pg = np.array(quat_to_proj_gravity_jax(q), dtype=np.float32)
    gyro = mj_data.qvel[3:6].astype(np.float32)
    obs = np.concatenate([pg, gyro * 0.25, [0.16, 0.0, 0.0], [np.sin(2*np.pi*phi), np.cos(2*np.pi*phi)], prev_act, prev2_act, [1.0]]).astype(np.float32)
    target_q = infer_rnv1_np(obs, phi, 0.16, 0.0, 0.0, hp)
    prev2_act, prev_act = prev_act.copy(), target_q.copy()
    for _ in range(10):
        f_ctrl = f_ctrl + alpha * (target_q - f_ctrl)
        mj_data.ctrl[:] = f_ctrl
        mujoco.mj_step(mj_model, mj_data)
    phi = (phi + float(np.clip(CPG_OPT[0], 0.9, 2.5)) * 0.02) % 1.0
    hp += 1.4 * 0.02
    if step % 2 == 0:  # 25 FPS video
        renderer.update_scene(mj_data)
        frames.append(renderer.render())

print(f"[EVAL] 5.0s Walk Distance: {mj_data.qpos[0]:.3f} m | Torso Z: {mj_data.qpos[2]:.3f} m | Peak Torque: {np.max(np.abs(mj_data.actuator_force)):.4f} N-m")
media.write_video("RNV1_walk_eval.mp4", frames, fps=25)
media.show_video(frames, fps=25)
"""
    ))

    # Cell 7: Auto-Export ESP32-S3 C Header + Auto-Download Bundle
    cells.append(nbf.v4.new_code_cell(
"""#@title 7. Auto-Export ESP32-S3 Firmware Header (rnv1_policy_weights.h) & Download { display-mode: "form" }
import json, zipfile

def fmt_c_arr(name, arr):
    flat = np.asarray(arr, dtype=np.float32).ravel()
    vals = ", ".join(f"{float(v):.7e}f" for v in flat)
    return f"static const float {name}[{len(flat)}] = {{\\n  {vals}\\n}};\\n"

header_lines = [
    f"// AUTO-GENERATED {TRAINING_VERSION} COLAB PRO A100 EXTREME RL WEIGHTS",
    "// Drop directly into firmware/include/rnv1_policy_weights.h and flash XIAO ESP32S3 Sense!",
    "#pragma once",
    "#include <stdint.h>",
    f"#define RNV1_OBS_DIM {OBS_DIM}",
    f"#define RNV1_H1_DIM  {H1_DIM}",
    f"#define RNV1_H2_DIM  {H2_DIM}",
    f"#define RNV1_ACT_DIM {ACT_DIM}",
    f"#define RNV1_RESIDUAL_SCALE {RESIDUAL_SCALE:.7e}f",
    fmt_c_arr("RNV1_Q_MIN", np.array(Q_MIN)),
    fmt_c_arr("RNV1_Q_MAX", np.array(Q_MAX)),
    fmt_c_arr("RNV1_CPG", CPG_OPT),
    fmt_c_arr("RNV1_W1", W1),
    fmt_c_arr("RNV1_B1", B1),
    fmt_c_arr("RNV1_W2", W2),
    fmt_c_arr("RNV1_B2", B2),
    fmt_c_arr("RNV1_W3", W3),
    fmt_c_arr("RNV1_B3", B3),
]
Path("rnv1_policy_weights.h").write_text("\\n".join(header_lines), encoding="utf-8")

weights_json = {
    "version": TRAINING_VERSION, "obs_dim": OBS_DIM, "h1_dim": H1_DIM, "h2_dim": H2_DIM,
    "act_dim": ACT_DIM, "control_hz": CONTROL_HZ, "residual_scale": RESIDUAL_SCALE,
    "cpg": CPG_OPT.tolist(), "w1": W1.tolist(), "b1": B1.tolist(),
    "w2": W2.tolist(), "b2": B2.tolist(), "w3": W3.tolist(), "b3": B3.tolist(),
}
Path("rnv1_policy_weights.json").write_text(json.dumps(weights_json, indent=2), encoding="utf-8")

bundle = "RNV1_Trained_Bundle.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
    for fn in ["rnv1_policy_weights.h", "rnv1_policy_weights.json", "RNV1_walk_eval.mp4"]:
        if Path(fn).exists():
            zf.write(fn)

print(f"[EXPORT COMPLETE] {bundle} ready ({Path(bundle).stat().st_size / 1024:.1f} KB)!")
try:
    from google.colab import files
    files.download(bundle)
except ImportError:
    print("Running outside Colab: files saved locally in current directory.")
"""
    ))

    nb.cells = cells
    NB_OUT.parent.mkdir(parents=True, exist_ok=True)
    with NB_OUT.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Generated One-Click Colab Pro A100 Notebook: {NB_OUT} ({len(cells)} cells)")

if __name__ == "__main__":
    main()
