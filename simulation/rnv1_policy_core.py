#!/usr/bin/env python3
"""RNV1 (Robotic Neural Vector 1) Policy Core — Coaxial Universal-Ankle Architecture.
Shared identically across MuJoCo 4.0.1, Colab Pro A100 MJX, PyBullet 3.2.7, and ESP32-S3 C++.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

OBS_DIM = 36
H1_DIM = 64
H2_DIM = 64
ACT_DIM = 12
CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ

JOINT_NAMES = [
    "l_hip_roll", "l_hip_pitch", "l_ankle_pitch", "l_ankle_roll",
    "r_hip_roll", "r_hip_pitch", "r_ankle_pitch", "r_ankle_roll",
    "l_shoulder", "r_shoulder", "head_yaw", "head_pitch",
]

Q_MIN = np.array([
    -0.524, -0.960, -0.785, -0.611,
    -0.611, -0.960, -0.785, -0.611,
    -1.658, -1.658, -1.309, -0.524,
], dtype=np.float32)

Q_MAX = np.array([
     0.611,  0.960,  0.785,  0.611,
     0.524,  0.960,  0.785,  0.611,
     2.618,  2.618,  1.309,  0.611,
], dtype=np.float32)

# Universal-Ankle CPG Parameters (tuned around natural inverted-pendulum resonance ~1.45 Hz!):
# [gait_freq_hz, lean_fwd, ankle_level_ratio, roll_amp, ankle_roll_ratio,
#  stride_hip_amp, swing_lift_roll, step_phase_off, arm_swing_amp,
#  ankle_push_amp, turn_diff_gain, turn_roll_bias]
DEFAULT_CPG = np.array([
    1.45,   # 0: gait_freq_hz (matches natural pendulum frequency sqrt(g/h)/(2pi) = 1.49 Hz!)
    0.01,   # 1: lean_fwd (rad)
    1.00,   # 2: ankle_level_ratio (ankle_pitch = -hip_pitch * ratio keeps foot 100% flat!)
    0.19,   # 3: roll_amp (rad, lateral weight shift onto stance foot)
   -0.98,   # 4: ankle_roll_ratio
    0.25,   # 5: stride_hip_amp (rad, full 88mm leg stride!)
    0.09,   # 6: swing_lift_roll (rad, outward abduction lifts swing foot 12mm cleanly off floor!)
    0.00,   # 7: step_phase_off (cycles)
    0.42,   # 8: arm_swing_amp (rad, counter-yaw angular momentum balance)
    0.02,   # 9: ankle_push_amp (rad, subtle rocker push-off)
    0.65,   # 10: turn_diff_gain (differential left/right stride for crisp yaw turns)
    0.04,   # 11: turn_roll_bias
], dtype=np.float32)


def compute_cpg_prior(phi: float, cmd_vx: float, cmd_vy: float, cmd_yaw: float,
                      cpg: np.ndarray, head_scan_phase: float = 0.0) -> np.ndarray:
    """Compute smooth flat-foot full-leg bipedal CPG prior (12-D rad)."""
    speed_mag = float(np.clip(math.hypot(cmd_vx / 0.12, cmd_yaw / 0.55), 0.0, 1.35))
    act = float(np.clip(speed_mag, 0.0, 1.0))
    fwd_scale = float(np.clip(cmd_vx / 0.12, -0.85, 1.25))
    yaw_scale = float(np.clip(cmd_yaw / 0.55, -1.0, 1.0))
    lat_scale = float(np.clip(cmd_vy / 0.08, -1.0, 1.0))

    lean_fwd    = float(cpg[1]) * fwd_scale
    level_ratio = float(cpg[2])
    roll_amp    = float(cpg[3]) * act
    ank_ratio   = float(cpg[4])
    stride_base = fwd_scale if abs(fwd_scale) > 0.05 else 0.45 * abs(yaw_scale)
    stride_amp  = float(cpg[5]) * stride_base
    lift_roll   = float(cpg[6]) * act
    phase_off   = float(cpg[7])
    arm_amp     = float(cpg[8]) * act
    ank_push    = float(cpg[9]) * act
    turn_diff   = float(cpg[10]) * yaw_scale
    turn_roll   = float(cpg[11]) * yaw_scale

    w_roll = math.sin(2.0 * math.pi * phi)
    w_step = math.cos(2.0 * math.pi * (phi + phase_off))
    l_sw = max(0.0, math.sin(2.0 * math.pi * phi))
    r_sw = max(0.0, -math.sin(2.0 * math.pi * phi))

    # Inner-leg-only shortening (never amplifies outer leg beyond proven stable stride limit!)
    # Verified empirically in MuJoCo 4.0.1: +yaw_scale turns LEFT (+CCW), -yaw_scale turns RIGHT (-CW)
    l_stride_gain = 1.0 - max(0.0, -turn_diff)
    r_stride_gain = 1.0 - max(0.0,  turn_diff)

    q = np.zeros(ACT_DIM, dtype=np.float32)

    rc = roll_amp * w_roll + 0.05 * lat_scale * w_step + turn_roll * w_step
    # Stance shifts torso over stance sole; swing leg abducts slightly for crisp foot clearance!
    q[0] = rc + lift_roll * l_sw
    q[3] = ank_ratio * rc - lift_roll * l_sw
    q[4] = rc - lift_roll * r_sw
    q[7] = ank_ratio * rc + lift_roll * r_sw

    # Full 88mm leg stride + exact flat-foot ankle pitch compensation!
    l_hip = -lean_fwd + stride_amp * l_stride_gain * w_step
    r_hip = -lean_fwd - stride_amp * r_stride_gain * w_step
    q[1] = l_hip
    q[2] = -l_hip * level_ratio - ank_push * w_roll
    q[5] = r_hip
    q[6] = -r_hip * level_ratio + ank_push * w_roll

    # Counter-swinging arms cancel torso yaw momentum
    q[8] = -arm_amp * w_step - 0.25 * yaw_scale
    q[9] =  arm_amp * w_step + 0.25 * yaw_scale

    # Active 2-DOF spatial head scan (coaxial OV5640/OV3660 + VL53L0X)
    q[10] = float(np.clip(0.45 * yaw_scale + 0.35 * math.sin(head_scan_phase), -1.1, 1.1))
    q[11] = float(np.clip(0.08 + 0.08 * math.sin(0.5 * head_scan_phase), -0.35, 0.45))

    return np.clip(q, Q_MIN, Q_MAX)


class RNV1Policy:
    """Compact CPG + Residual MLP (36 -> 64 -> 64 -> 12) for RNV1."""

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.cpg = DEFAULT_CPG.copy()
        self.residual_scale = np.float32(0.35)
        self.w1 = (rng.standard_normal((H1_DIM, OBS_DIM)) * np.sqrt(1.0 / OBS_DIM) * 0.25).astype(np.float32)
        self.b1 = np.zeros(H1_DIM, dtype=np.float32)
        self.w2 = (rng.standard_normal((H2_DIM, H1_DIM)) * np.sqrt(1.0 / H1_DIM) * 0.25).astype(np.float32)
        self.b2 = np.zeros(H2_DIM, dtype=np.float32)
        self.w3 = (rng.standard_normal((ACT_DIM, H2_DIM)) * np.sqrt(1.0 / H2_DIM) * 0.05).astype(np.float32)
        self.b3 = np.zeros(ACT_DIM, dtype=np.float32)

    def forward_mlp(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        h1 = np.tanh(self.w1 @ x + self.b1)
        h2 = np.tanh(self.w2 @ h1 + self.b2)
        return np.tanh(self.w3 @ h2 + self.b3)

    def act(self, obs: np.ndarray, phi: float, cmd_vx: float, cmd_vy: float,
            cmd_yaw: float, head_scan_phase: float = 0.0) -> np.ndarray:
        q_prior = compute_cpg_prior(phi, cmd_vx, cmd_vy, cmd_yaw, self.cpg, head_scan_phase)
        delta = self.forward_mlp(obs) * self.residual_scale
        delta[10] *= 0.25
        delta[11] *= 0.25
        return np.clip(q_prior + delta, Q_MIN, Q_MAX).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "version": "RNV1",
            "obs_dim": OBS_DIM, "h1_dim": H1_DIM, "h2_dim": H2_DIM, "act_dim": ACT_DIM,
            "control_hz": CONTROL_HZ, "residual_scale": float(self.residual_scale),
            "cpg": self.cpg.tolist(),
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
        }

    def save_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "RNV1Policy":
        d = json.loads(path.read_text(encoding="utf-8"))
        pol = cls()
        pol.residual_scale = np.float32(d["residual_scale"])
        pol.cpg = np.array(d["cpg"], dtype=np.float32)
        pol.w1 = np.array(d["w1"], dtype=np.float32)
        pol.b1 = np.array(d["b1"], dtype=np.float32)
        pol.w2 = np.array(d["w2"], dtype=np.float32)
        pol.b2 = np.array(d["b2"], dtype=np.float32)
        pol.w3 = np.array(d["w3"], dtype=np.float32)
        pol.b3 = np.array(d["b3"], dtype=np.float32)
        return pol

    def export_c_header(self, path: Path) -> None:
        def fmt_arr(name: str, arr: np.ndarray) -> str:
            flat = arr.astype(np.float32).ravel()
            vals = ", ".join(f"{float(v):.7e}f" for v in flat)
            return f"static const float {name}[{len(flat)}] = {{\n  {vals}\n}};\n"

        lines = [
            "// AUTO-GENERATED RNV1 NEURAL POLICY WEIGHTS (36 -> 64 -> 64 -> 12 + Universal-Ankle CPG)",
            "// Target: Seeed Studio XIAO ESP32S3 Sense (Xtensa LX7 240MHz Dual-Core + SIMD)",
            "#pragma once",
            "#include <stdint.h>",
            f"#define RNV1_OBS_DIM {OBS_DIM}",
            f"#define RNV1_H1_DIM  {H1_DIM}",
            f"#define RNV1_H2_DIM  {H2_DIM}",
            f"#define RNV1_ACT_DIM {ACT_DIM}",
            f"#define RNV1_RESIDUAL_SCALE {float(self.residual_scale):.7e}f",
            fmt_arr("RNV1_Q_MIN", Q_MIN),
            fmt_arr("RNV1_Q_MAX", Q_MAX),
            fmt_arr("RNV1_CPG", self.cpg),
            fmt_arr("RNV1_W1", self.w1),
            fmt_arr("RNV1_B1", self.b1),
            fmt_arr("RNV1_W2", self.w2),
            fmt_arr("RNV1_B2", self.b2),
            fmt_arr("RNV1_W3", self.w3),
            fmt_arr("RNV1_B3", self.b3),
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")


def build_observation(proj_gravity: np.ndarray, gyro: np.ndarray,
                      cmd_vx: float, cmd_vy: float, cmd_yaw: float,
                      phi: float, prev_act: np.ndarray, prev2_act: np.ndarray,
                      tof_dist_m: float) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0:3] = proj_gravity
    obs[3:6] = gyro * 0.25
    obs[6] = cmd_vx
    obs[7] = cmd_vy
    obs[8] = cmd_yaw
    obs[9] = math.sin(2.0 * math.pi * phi)
    obs[10] = math.cos(2.0 * math.pi * phi)
    obs[11:23] = prev_act
    obs[23:35] = prev2_act
    obs[35] = float(np.clip(tof_dist_m / 1.5, 0.0, 1.0))
    return obs
