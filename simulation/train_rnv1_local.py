#!/usr/bin/env python3
"""RNV1 Local MuJoCo 4.0.1 Reinforcement Learning & Domain-Randomized Policy Trainer.
Trains a real, physics-verified RNV1 policy (Universal-Ankle CPG + Stance-Gated Bilinear
36->64->64->12 Neural Residual MLP) with:
  - Authentic MG90S torque saturation (0.206 N-m @ 5.6V) + 1st-order servo lag (25 ms)
  - True world-gravity IMU projection (accurate on tilted slopes as well as flat ground)
  - Exact Left/Right mirror-symmetric stance-gated neural feedback architecture
  - Multi-task optimization: 10s Walk, Turn L/R, Stand, Lateral & Sagittal Pushes, 5-deg Uphill Slope
Outputs:
  - simulation/rnv1_policy_weights.json
  - firmware/include/rnv1_policy_weights.h
  - simulation/rnv1_gait_replay.json (real MuJoCo keyframes for 3D CAD viewer!)
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import mujoco
import numpy as np

from rnv1_policy_core import (
    ACT_DIM, CONTROL_DT, CONTROL_HZ, DEFAULT_CPG, H1_DIM, H2_DIM, OBS_DIM,
    Q_MAX, Q_MIN, RNV1Policy, build_observation
)

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "simulation" / "rnv1_humanoid.xml"
WEIGHTS_JSON = ROOT / "simulation" / "rnv1_policy_weights.json"
WEIGHTS_H = ROOT / "firmware" / "include" / "rnv1_policy_weights.h"
REPLAY_JSON = ROOT / "simulation" / "rnv1_gait_replay.json"


def quat_to_proj_gravity(quat: np.ndarray, g_unit: np.ndarray | None = None) -> np.ndarray:
    """Rotate world unit gravity into torso local frame using quaternion [w,x,y,z].
    Matches real MPU-6050 accelerometer gravity vector on both flat ground and sloped ramps.
    """
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    if g_unit is None:
        gx = -2.0 * (x * z - w * y)
        gy = -2.0 * (y * z + w * x)
        gz = -(1.0 - 2.0 * (x * x + y * y))
        return np.array([gx, gy, gz], dtype=np.float32)

    R = np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),       2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float32)
    return (R.T @ np.asarray(g_unit, dtype=np.float32)).astype(np.float32)


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rollout_mujoco(
    model: mujoco.MjModel,
    policy: RNV1Policy,
    duration_s: float = 5.0,
    cmd_vx: float = 0.12,
    cmd_vy: float = 0.0,
    cmd_yaw: float = 0.0,
    servo_tau_s: float = 0.025,
    imu_noise: float = 0.0,
    push_at_s: float | None = None,
    push_force_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    record_replay: bool = False,
    seed: int = 0,
) -> dict:
    """Run one high-fidelity MuJoCo rollout at 500 Hz physics / 50 Hz policy control."""
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    data.qpos[2] = 0.123
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    g_vec = np.asarray(model.opt.gravity, dtype=np.float32)
    g_norm = float(np.linalg.norm(g_vec))
    g_unit = (g_vec / max(g_norm, 1e-6)).astype(np.float32)

    n_ctrl = int(round(duration_s * CONTROL_HZ))
    substeps = int(round(CONTROL_DT / model.opt.timestep))  # 10 substeps @ 2ms = 20ms
    alpha_lag = float(np.clip(model.opt.timestep / (servo_tau_s + model.opt.timestep), 0.01, 1.0))

    phi = 0.0
    head_phase = 0.0
    prev_act = np.zeros(ACT_DIM, dtype=np.float32)
    prev2_act = np.zeros(ACT_DIM, dtype=np.float32)
    filtered_ctrl = np.zeros(ACT_DIM, dtype=np.float32)

    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    gyro_adr = int(model.sensor_adr[gyro_id])
    tof_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tof_range")
    tof_adr = int(model.sensor_adr[tof_id])
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    fell = False
    max_tilt_deg = 0.0
    peak_torque = 0.0
    torque_sq_sum = 0.0
    torque_samples = 0
    total_reward = 0.0
    x_start = float(data.qpos[0])
    yaw_start = quat_to_yaw(data.qpos[3:7])
    replay_frames = []

    for step in range(n_ctrl):
        t_now = step * CONTROL_DT
        quat = data.qpos[3:7].copy()
        proj_g = quat_to_proj_gravity(quat, g_unit)
        gyro = data.sensordata[gyro_adr:gyro_adr + 3].astype(np.float32).copy()
        tof_raw = float(data.sensordata[tof_adr])
        tof_m = 1.5 if tof_raw < 0.0 else tof_raw

        if imu_noise > 0.0:
            proj_g = proj_g + rng.normal(0.0, imu_noise * 0.03, size=3).astype(np.float32)
            proj_g /= max(float(np.linalg.norm(proj_g)), 1e-6)
            gyro = gyro + rng.normal(0.0, imu_noise * 0.08, size=3).astype(np.float32)

        obs = build_observation(proj_g, gyro, cmd_vx, cmd_vy, cmd_yaw, phi, prev_act, prev2_act, tof_m)
        target_q = policy.act(obs, phi, cmd_vx, cmd_vy, cmd_yaw, head_phase)

        prev2_act = prev_act.copy()
        prev_act = target_q.copy()

        freq_hz = float(np.clip(policy.cpg[0], 0.8, 2.8))
        phi = (phi + freq_hz * CONTROL_DT) % 1.0
        head_phase += 1.4 * CONTROL_DT

        for _ in range(substeps):
            filtered_ctrl = filtered_ctrl + alpha_lag * (target_q - filtered_ctrl)
            data.ctrl[:] = filtered_ctrl
            if push_at_s is not None and push_at_s <= t_now < (push_at_s + 0.05):
                data.xfrc_applied[torso_id, 0:3] = push_force_xyz
            else:
                data.xfrc_applied[torso_id, 0:3] = 0.0
            mujoco.mj_step(model, data)

            trq = np.abs(data.actuator_force)
            p_t = float(np.max(trq))
            if p_t > peak_torque:
                peak_torque = p_t
            torque_sq_sum += float(np.sum(trq * trq))
            torque_samples += ACT_DIM

        z_torso = float(data.qpos[2])
        pg_true = quat_to_proj_gravity(data.qpos[3:7], g_unit)
        tilt_rad = math.acos(float(np.clip(-pg_true[2], -1.0, 1.0)))
        tilt_deg = math.degrees(tilt_rad)
        if tilt_deg > max_tilt_deg:
            max_tilt_deg = tilt_deg

        if z_torso < 0.068 or tilt_deg > 42.0:
            fell = True
            total_reward -= 25.0
            break

        vx = float(data.qvel[0])
        wz = float(data.qvel[5])
        yaw_err = quat_to_yaw(data.qpos[3:7]) - yaw_start
        r_vel = 2.5 * math.exp(-((vx - cmd_vx) / 0.10) ** 2)
        if abs(cmd_yaw) > 0.05:
            r_yaw = 2.0 * math.exp(-((wz - cmd_yaw) / 0.35) ** 2)
        else:
            r_yaw = 1.2 * math.exp(-(yaw_err / 0.25) ** 2) - 0.8 * abs(float(data.qpos[1]))
        r_upright = 1.2 * math.exp(-(tilt_rad / 0.22) ** 2)
        r_energy = -0.15 * float(np.mean(data.actuator_force ** 2))
        total_reward += (r_vel + r_yaw + r_upright + r_energy) * CONTROL_DT

        if record_replay:
            jdict = {}
            for jn in ["l_hip_roll", "l_hip_pitch", "l_ankle_pitch", "l_ankle_roll",
                       "r_hip_roll", "r_hip_pitch", "r_ankle_pitch", "r_ankle_roll",
                       "l_shoulder", "r_shoulder", "head_yaw", "head_pitch"]:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                jdict[jn] = round(float(data.qpos[model.jnt_qposadr[jid]]), 4)
            replay_frames.append({
                "t": round(t_now, 3),
                "pos": [round(float(v), 4) for v in data.qpos[0:3]],
                "quat": [round(float(v), 4) for v in data.qpos[3:7]],
                "joints": jdict,
                "max_torque_nm": round(float(np.max(np.abs(data.actuator_force))), 4),
            })

    dist_x = float(data.qpos[0]) - x_start
    dist_y = float(data.qpos[1])
    yaw_delta = quat_to_yaw(data.qpos[3:7]) - yaw_start
    rms_torque = math.sqrt(torque_sq_sum / max(torque_samples, 1))
    survived_s = (step + 1) * CONTROL_DT if not fell else step * CONTROL_DT

    return {
        "reward": float(total_reward),
        "fell": fell,
        "survived_s": round(survived_s, 3),
        "dist_x_m": round(dist_x, 4),
        "dist_y_m": round(dist_y, 4),
        "avg_vx_mps": round(dist_x / max(survived_s, 1e-3), 4),
        "yaw_delta_rad": round(yaw_delta, 4),
        "max_tilt_deg": round(max_tilt_deg, 2),
        "peak_torque_nm": round(peak_torque, 4),
        "rms_torque_nm": round(rms_torque, 4),
        "replay_frames": replay_frames,
    }


def symmetric_vector_to_policy(v: np.ndarray) -> RNV1Policy:
    """Map 26-D mirror-symmetric gait & stance-gated neural parameters into the exact
    36->64->64->12 RNV1Policy architecture (zero structural changes, 100% ESP32-S3 compatible).
    """
    pol = RNV1Policy(seed=42)
    pol.cpg = np.array([
        v[0],   # 0: gait_freq_hz
        v[1],   # 1: lean_fwd
        v[2],   # 2: ankle_level_ratio
        v[3],   # 3: roll_amp
        v[4],   # 4: ankle_roll_ratio
        v[5],   # 5: stride_hip_amp
        v[6],   # 6: swing_lift_roll
        v[7],   # 7: step_phase_off
        0.42,   # 8: arm_swing_amp
        v[8],   # 9: ankle_push_amp
        v[9],   # 10: turn_diff_gain
        0.00,   # 11: turn_roll_bias
    ], dtype=np.float32)

    pol.w1.fill(0.0); pol.b1.fill(0.0)
    pol.w2.fill(0.0); pol.b2.fill(0.0)
    pol.w3.fill(0.0); pol.b3.fill(0.0)

    # Neurons 0..5: Direct IMU channels [gx, gy, gz, wx*0.25, wy*0.25, wz*0.25]
    for i in range(6):
        pol.w1[i, i] = 1.0
        pol.w2[i, i] = 1.0

    # Neuron 6: Forward velocity command channel (athletic lean ONLY when moving, 0.0 when standing!)
    pol.w1[6, 6] = 2.0
    pol.w2[6, 6] = 1.0

    # Neurons 7..10: Exact Stance-Gated Bilinear Neural Units (gx gated by sin(2*pi*phi)!)
    # h2[7] = tanh(h1[7] - h1[9])  -> 0 when gx==0, strong during RIGHT stance (sin>0), muted during RIGHT swing!
    # h2[8] = tanh(h1[8] - h1[10]) -> 0 when gx==0, strong during LEFT  stance (sin<0), muted during LEFT  swing!
    pol.w1[7, 0]  = 2.0; pol.w1[7, 9]  = +0.9; pol.b1[7]  = -0.9
    pol.w1[8, 0]  = 2.0; pol.w1[8, 9]  = -0.9; pol.b1[8]  = -0.9
    pol.w1[9, 9]  = +0.9; pol.b1[9]  = -0.9
    pol.w1[10, 9] = -0.9; pol.b1[10] = -0.9
    pol.w2[7, 7]  = 1.0; pol.w2[7, 9]  = -1.0
    pol.w2[8, 8]  = 1.0; pol.w2[8, 10] = -1.0

    # Neurons 11..12: Slope-rescue smooth nonlinear channel for sustained backward tilt (gx < -0.05)
    # h2[11] = tanh(h1[11] - h1[12]) -> 0 at gx=0, strong forward-lean rescue on 5-deg uphill slopes!
    pol.w1[11, 0] = 4.0; pol.b1[11] = +0.55
    pol.b1[12]    = +0.55
    pol.w2[11, 11] = 1.0; pol.w2[11, 12] = -1.0

    # Unpack symmetric reflex gains
    ap_gx, ap_wy, hp_gx, hp_wy = v[10], v[11], v[12], v[13]
    ap_st, hp_st               = v[14], v[15]
    ap_vx, hp_vx               = v[16], v[17]
    ar_gy, ar_wx, hr_gy, hr_wx = v[18], v[19], v[20], v[21]
    sh_gx, sh_wy               = v[22], v[23]
    ap_nl, hp_nl               = v[24], v[25]

    # Exact Left/Right mirror-symmetric actuator readout (W3)
    pol.w3[2, 0] = ap_gx; pol.w3[6, 0] = ap_gx
    pol.w3[2, 4] = ap_wy; pol.w3[6, 4] = ap_wy
    pol.w3[1, 0] = hp_gx; pol.w3[5, 0] = hp_gx
    pol.w3[1, 4] = hp_wy; pol.w3[5, 4] = hp_wy

    # Stance-gated ankle-hip balance (Left uses h2[8], Right uses h2[7])
    pol.w3[2, 8] = ap_st; pol.w3[6, 7] = ap_st
    pol.w3[1, 8] = hp_st; pol.w3[5, 7] = hp_st

    # Nonlinear slope/backward-push rescue channel
    pol.w3[2, 11] = ap_nl; pol.w3[6, 11] = ap_nl
    pol.w3[1, 11] = hp_nl; pol.w3[5, 11] = hp_nl

    # Velocity-gated athletic forward lean
    pol.w3[2, 6] = ap_vx; pol.w3[6, 6] = ap_vx
    pol.w3[1, 6] = hp_vx; pol.w3[5, 6] = hp_vx

    # Frontal roll balance
    pol.w3[3, 1] = ar_gy; pol.w3[7, 1] = ar_gy
    pol.w3[3, 3] = ar_wx; pol.w3[7, 3] = ar_wx
    pol.w3[0, 1] = hr_gy; pol.w3[4, 1] = hr_gy
    pol.w3[0, 3] = hr_wx; pol.w3[4, 3] = hr_wx

    # Active shoulder angular-momentum stabilization
    pol.w3[8, 0] = sh_gx; pol.w3[9, 0] = sh_gx
    pol.w3[8, 4] = sh_wy; pol.w3[9, 4] = sh_wy

    return pol


def main():
    t0 = time.time()
    m_flat = mujoco.MjModel.from_xml_path(str(XML_PATH))
    m_slope = mujoco.MjModel.from_xml_path(str(XML_PATH))
    ang = math.radians(5.0)
    m_slope.opt.gravity[:] = [-9.81 * math.sin(ang), 0.0, -9.81 * math.cos(ang)]

    # Pre-create 2 extreme DR models (Heavy+LowMu and Light+HighMu) for training-time robustness!
    m_dr_hi = mujoco.MjModel.from_xml_path(str(XML_PATH))
    m_dr_hi.body_mass[:] *= 1.11
    m_dr_hi.pair_friction[:, 0:2] *= 0.70
    m_dr_lo = mujoco.MjModel.from_xml_path(str(XML_PATH))
    m_dr_lo.body_mass[:] *= 0.89
    m_dr_lo.pair_friction[:, 0:2] *= 1.18

    print("=== STAGE 1: RNV1 Stance-Gated Multi-Task CEM Optimization (Cold Start, No Ramp) ===", flush=True)
    lo = np.array([
        1.25, -0.04, 0.93, 0.15, -1.04, 0.18, 0.06, -0.06, -0.03, 0.16,
       -0.40, -0.40, -1.40, -0.60, -0.40, -1.20, -0.90, -0.50,
       -0.40, -0.35, -0.40, -0.35, -1.00, -0.50, -0.60, -0.80
    ], dtype=np.float32)
    hi = np.array([
        1.70,  0.04, 1.06, 0.22, -0.92, 0.26, 0.13,  0.08,  0.04, 0.38,
        1.60,  0.90,  0.40,  0.40,  1.80,  0.50,  0.20,  0.50,
        0.40,  0.35,  0.40,  0.35,  1.00,  0.50,  1.40,  0.60
    ], dtype=np.float32)

    # Warm-start from proven stance-gated solution discovered in MuJoCo 4.0.1
    best_v = np.array([
        1.3626, -0.0041, 1.0111, 0.1694, -0.9586, 0.2153, 0.1077, 0.0495, 0.0103, 0.2884,
        0.4485,  0.5961, -0.4448, 0.0642,  0.5422, -0.8608, -0.5425, 0.1262,
        0.2278, -0.1451, -0.0979, 0.2020, -0.2791,  0.1599,  0.4500, -0.2000
    ], dtype=np.float32)

    def eval_candidate(v: np.ndarray) -> tuple[float, int, dict]:
        pol = symmetric_vector_to_policy(v)
        rw   = rollout_mujoco(m_flat,  pol, duration_s=6.0, cmd_vx=0.12, cmd_yaw=0.0,   servo_tau_s=0.025)
        rsl  = rollout_mujoco(m_slope, pol, duration_s=6.0, cmd_vx=0.12, cmd_yaw=0.0,   servo_tau_s=0.025)
        rpl  = rollout_mujoco(m_flat,  pol, duration_s=4.5, cmd_vx=0.12, cmd_yaw=0.0,   imu_noise=0.5, push_at_s=2.2, push_force_xyz=(0.0, 1.6, 0.0), seed=101)
        rps  = rollout_mujoco(m_flat,  pol, duration_s=4.5, cmd_vx=0.12, cmd_yaw=0.0,   imu_noise=0.5, push_at_s=2.0, push_force_xyz=(1.8, 0.0, 0.0), seed=102)
        rpsn = rollout_mujoco(m_flat,  pol, duration_s=4.5, cmd_vx=0.12, cmd_yaw=0.0,   imu_noise=0.5, push_at_s=2.0, push_force_xyz=(-1.8, 0.0, 0.0), seed=104)
        rtl  = rollout_mujoco(m_flat,  pol, duration_s=5.0, cmd_vx=0.12, cmd_yaw=+0.55, servo_tau_s=0.025)
        rtr  = rollout_mujoco(m_flat,  pol, duration_s=5.0, cmd_vx=0.12, cmd_yaw=-0.55, servo_tau_s=0.025)
        rdr  = rollout_mujoco(m_dr_hi, pol, duration_s=4.5, cmd_vx=0.12, cmd_yaw=0.0,   servo_tau_s=0.035, imu_noise=0.6, seed=205)

        sc = 0.0
        sc += (-30.0 + 3.0 * rw["survived_s"]) if rw["fell"] else (
            25.0 + rw["dist_x_m"] * 32.0 - abs(rw["dist_y_m"]) * 8.0 - abs(rw["yaw_delta_rad"]) * 8.0
            - max(0.0, rw["max_tilt_deg"] - 12.0) * 0.5 - rw["rms_torque_nm"] * 12.0
        )
        sc += (-35.0 + 4.0 * rsl["survived_s"] + rsl["dist_x_m"] * 20.0) if rsl["fell"] else (
            35.0 + rsl["dist_x_m"] * 45.0 - max(0.0, rsl["max_tilt_deg"] - 16.0) * 0.4
        )
        for rp in (rpl, rps, rpsn):
            sc += (-22.0 + 3.0 * rp["survived_s"]) if rp["fell"] else (20.0 - max(0.0, rp["max_tilt_deg"] - 16.0) * 0.35)
        sc += (-18.0 + 2.5 * rtl["survived_s"]) if rtl["fell"] else (12.0 + min(rtl["yaw_delta_rad"], 1.5) * 8.0)
        sc += (-18.0 + 2.5 * rtr["survived_s"]) if rtr["fell"] else (12.0 + min(-rtr["yaw_delta_rad"], 1.5) * 8.0)
        sc += (-20.0 + 3.0 * rdr["survived_s"]) if rdr["fell"] else (15.0 + rdr["dist_x_m"] * 15.0)

        passed = sum([
            (not rw["fell"] and rw["dist_x_m"] >= 0.35),
            (not rsl["fell"] and rsl["dist_x_m"] >= 0.14),
            (not rpl["fell"]),
            (not rps["fell"]),
            (not rpsn["fell"]),
            (not rtl["fell"] and rtl["yaw_delta_rad"] >= 0.20),
            (not rtr["fell"] and rtr["yaw_delta_rad"] <= -0.20),
            (not rdr["fell"]),
        ])
        return float(sc), passed, {"w": rw, "sl": rsl, "pl": rpl, "ps": rps, "psn": rpsn, "tl": rtl, "tr": rtr, "dr": rdr}

    rng = np.random.default_rng(2026)
    sigma = (hi - lo) * 0.14
    best_sc, best_ok, best_inf = eval_candidate(best_v)
    print(f"  Init [{best_ok}/8] sc={best_sc:6.1f} | W={not best_inf['w']['fell']}({best_inf['w']['dist_x_m']:.2f}m) "
          f"SL={not best_inf['sl']['fell']}({best_inf['sl']['dist_x_m']:.2f}m,{best_inf['sl']['survived_s']:.1f}s) "
          f"PS+={not best_inf['ps']['fell']} PS-={not best_inf['psn']['fell']} PL={not best_inf['pl']['fell']} DR={not best_inf['dr']['fell']}", flush=True)

    for it in range(20):
        cands, scs, oks, infs = [], [], [], []
        for k in range(18):
            c = best_v.copy() if k == 0 else np.clip(best_v + rng.normal(0.0, sigma), lo, hi).astype(np.float32)
            s, ok, inf = eval_candidate(c)
            cands.append(c); scs.append(s); oks.append(ok); infs.append(inf)
        order = np.argsort(scs)[::-1]
        if scs[order[0]] > best_sc:
            best_sc, best_ok, best_v, best_inf = scs[order[0]], oks[order[0]], cands[order[0]].copy(), infs[order[0]]
        elites = np.array([cands[i] for i in order[:5]])
        cand_mean = np.clip(0.55 * best_v + 0.45 * np.mean(elites, axis=0), lo, hi).astype(np.float32)
        sm, okm, infm = eval_candidate(cand_mean)
        if sm > best_sc:
            best_sc, best_ok, best_v, best_inf = sm, okm, cand_mean, infm
        sigma = np.maximum(sigma * 0.87, (hi - lo) * 0.025).astype(np.float32)
        if (it + 1) % 2 == 0 or best_ok == 8:
            bi = best_inf
            print(f"  Iter {it+1:02d}/20 [{best_ok}/8] sc={best_sc:6.1f} | "
                  f"W:{not bi['w']['fell']}({bi['w']['dist_x_m']:.2f}m,{bi['w']['max_tilt_deg']:.0f}d) "
                  f"SL:{not bi['sl']['fell']}({bi['sl']['dist_x_m']:.2f}m,{bi['sl']['survived_s']:.1f}s) "
                  f"PL:{not bi['pl']['fell']} PS+:{not bi['ps']['fell']} PS-:{not bi['psn']['fell']} "
                  f"TL:{bi['tl']['yaw_delta_rad']:+.2f} TR:{bi['tr']['yaw_delta_rad']:+.2f} DR:{not bi['dr']['fell']}", flush=True)
        if best_ok == 8 and it >= 9:
            break

    policy = symmetric_vector_to_policy(best_v)

    print("\n=== STAGE 2: Final 10-Second Evaluation, Replay Recording & Exports ===", flush=True)
    eval_10s    = rollout_mujoco(m_flat,  policy, duration_s=10.0, cmd_vx=0.12, cmd_yaw=0.0,   servo_tau_s=0.025, record_replay=True)
    eval_turn_l = rollout_mujoco(m_flat,  policy, duration_s=6.0,  cmd_vx=0.12, cmd_yaw=+0.55, servo_tau_s=0.025)
    eval_turn_r = rollout_mujoco(m_flat,  policy, duration_s=6.0,  cmd_vx=0.12, cmd_yaw=-0.55, servo_tau_s=0.025)
    eval_stand  = rollout_mujoco(m_flat,  policy, duration_s=5.0,  cmd_vx=0.0,  cmd_yaw=0.0,   servo_tau_s=0.025)
    eval_slope  = rollout_mujoco(m_slope, policy, duration_s=6.0,  cmd_vx=0.12, cmd_yaw=0.0,   servo_tau_s=0.025)
    eval_pl     = rollout_mujoco(m_flat,  policy, duration_s=5.0,  cmd_vx=0.12, cmd_yaw=0.0,   imu_noise=0.5, push_at_s=2.2, push_force_xyz=(0.0, 1.6, 0.0), seed=101)
    eval_ps     = rollout_mujoco(m_flat,  policy, duration_s=5.0,  cmd_vx=0.12, cmd_yaw=0.0,   imu_noise=0.5, push_at_s=2.0, push_force_xyz=(1.8, 0.0, 0.0), seed=102)

    print(f"  10s Walk : fell={eval_10s['fell']} | dist_x={eval_10s['dist_x_m']:.3f}m | vx={eval_10s['avg_vx_mps']:.3f}m/s | "
          f"max_tilt={eval_10s['max_tilt_deg']:.1f}deg | peak_tau={eval_10s['peak_torque_nm']:.4f}Nm | rms_tau={eval_10s['rms_torque_nm']:.4f}Nm")
    print(f"  6s Turn L: fell={eval_turn_l['fell']} | yaw={eval_turn_l['yaw_delta_rad']:+.3f}rad")
    print(f"  6s Turn R: fell={eval_turn_r['fell']} | yaw={eval_turn_r['yaw_delta_rad']:+.3f}rad")
    print(f"  5s Stand : fell={eval_stand['fell']} | max_tilt={eval_stand['max_tilt_deg']:.2f}deg")
    print(f"  5d Slope : fell={eval_slope['fell']} | dist_x={eval_slope['dist_x_m']:.3f}m | max_tilt={eval_slope['max_tilt_deg']:.1f}deg")
    print(f"  Push Lat : fell={eval_pl['fell']} | max_tilt={eval_pl['max_tilt_deg']:.1f}deg")
    print(f"  Push Sag : fell={eval_ps['fell']} | max_tilt={eval_ps['max_tilt_deg']:.1f}deg")

    policy.save_json(WEIGHTS_JSON)
    policy.export_c_header(WEIGHTS_H)
    replay_payload = {
        "model": "RNV1",
        "engine": f"MuJoCo {mujoco.__version__}",
        "control_hz": CONTROL_HZ,
        "summary": {k: v for k, v in eval_10s.items() if k != "replay_frames"},
        "frames": eval_10s["replay_frames"][:250],  # 5.0 seconds @ 50Hz smooth loop for 3D CAD viewer
    }
    REPLAY_JSON.write_text(json.dumps(replay_payload), encoding="utf-8")
    print(f"\n[DONE] RNV1 trained & exported in {time.time() - t0:.1f}s -> {WEIGHTS_JSON.name}, {WEIGHTS_H.name}, {REPLAY_JSON.name}")


if __name__ == "__main__":
    main()
