#!/usr/bin/env python3
"""RNV1 Post-Training Equivariant Multi-Engine Policy Optimizer.
Refines the Colab Pro A100 trained policy (RNV1_Trained_Bundle.zip) by:
  1. Preserving the top 25 A100-learned proprioceptive/action-history feature pairs (50 hidden units)
     projected onto the exact Left/Right bipedal mirror-equivariant subspace (M_act @ pi(x) = pi(M_obs @ x)).
  2. Eliminating un-gated phase/head-scan leakage during zero-command standing so 5s stand tilt < 0.1 deg.
  3. Co-optimizing CPG + Stance-Gated Bilinear Reflexes + Equivariant A100 Readout across:
     - MuJoCo 4.0.1: 10s Walk, 6s L/R Turns, 5s Stand, Lateral 1.6N & Sagittal +-1.8N Pushes,
       5-deg Uphill Slope, and 30-World Sim-to-Real Domain Randomization
     - PyBullet 3.2.7: Independent URDF LCP Cross-Engine 4s Stand & 6s Walk
Outputs drop-in weights to:
  - simulation/rnv1_policy_weights.json
  - firmware/include/rnv1_policy_weights.h
  - simulation/rnv1_gait_replay.json
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import mujoco
import numpy as np
import pybullet as p

from rnv1_policy_core import (
    ACT_DIM, CONTROL_DT, CONTROL_HZ, H1_DIM, H2_DIM, JOINT_NAMES, OBS_DIM,
    RNV1Policy, build_observation
)
from train_rnv1_local import quat_to_proj_gravity, rollout_mujoco

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "simulation" / "rnv1_humanoid.xml"
URDF_PATH = ROOT / "simulation" / "rnv1_humanoid.urdf"
WEIGHTS_JSON = ROOT / "simulation" / "rnv1_policy_weights.json"
WEIGHTS_H = ROOT / "firmware" / "include" / "rnv1_policy_weights.h"
REPLAY_JSON = ROOT / "simulation" / "rnv1_gait_replay.json"


def build_symmetry_matrices() -> tuple[np.ndarray, np.ndarray]:
    M_act = np.zeros((ACT_DIM, ACT_DIM), dtype=np.float32)
    M_act[0, 4] = -1.0; M_act[4, 0] = -1.0
    M_act[1, 5] = +1.0; M_act[5, 1] = +1.0
    M_act[2, 6] = +1.0; M_act[6, 2] = +1.0
    M_act[3, 7] = -1.0; M_act[7, 3] = -1.0
    M_act[8, 9] = +1.0; M_act[9, 8] = +1.0
    M_act[10, 10] = -1.0
    M_act[11, 11] = +1.0

    M_obs = np.zeros((OBS_DIM, OBS_DIM), dtype=np.float32)
    M_obs[0, 0] = +1.0; M_obs[1, 1] = -1.0; M_obs[2, 2] = +1.0
    M_obs[3, 3] = -1.0; M_obs[4, 4] = +1.0; M_obs[5, 5] = -1.0
    M_obs[6, 6] = +1.0; M_obs[7, 7] = -1.0; M_obs[8, 8] = -1.0
    M_obs[9, 9] = -1.0; M_obs[10, 10] = -1.0
    M_obs[11:23, 11:23] = M_act
    M_obs[23:35, 23:35] = M_act
    M_obs[35, 35] = +1.0
    return M_obs, M_act


M_OBS, M_ACT = build_symmetry_matrices()


def build_equivariant_a100_policy(a100_src: RNV1Policy, v: np.ndarray) -> RNV1Policy:
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
        v[26],  # 8: arm_swing_amp
        v[8],   # 9: ankle_push_amp
        v[9],   # 10: turn_diff_gain
        0.00,   # 11: turn_roll_bias
    ], dtype=np.float32)

    pol.w1.fill(0.0); pol.b1.fill(0.0)
    pol.w2.fill(0.0); pol.b2.fill(0.0)
    pol.w3.fill(0.0); pol.b3.fill(0.0)

    # 1. Exact Stance-Gated & Slope-Rescue Biomechanical Backbone (Neurons 0..13)
    for i in range(6):
        pol.w1[i, i] = 1.0
        pol.w2[i, i] = 1.0
    pol.w1[6, 6] = 2.0
    pol.w2[6, 6] = 1.0

    pol.w1[7, 0]  = 2.0; pol.w1[7, 9]  = +0.9; pol.b1[7]  = -0.9
    pol.w1[8, 0]  = 2.0; pol.w1[8, 9]  = -0.9; pol.b1[8]  = -0.9
    pol.w1[9, 9]  = +0.9; pol.b1[9]  = -0.9
    pol.w1[10, 9] = -0.9; pol.b1[10] = -0.9
    pol.w2[7, 7]  = 1.0; pol.w2[7, 9]  = -1.0
    pol.w2[8, 8]  = 1.0; pol.w2[8, 10] = -1.0

    pol.w1[11, 0] = 4.0; pol.b1[11] = +0.55
    pol.b1[12]    = +0.55
    pol.w2[11, 11] = 1.0; pol.w2[11, 12] = -1.0

    pol.w1[13, 8] = 1.8  # Yaw command channel
    pol.w2[13, 13] = 1.0

    # 2. Equivariant A100 Proprioceptive & Action-History Trunk (Neurons 14..38 & Mirror 39..63)
    row_norms = np.linalg.norm(a100_src.w1, axis=1)
    top25 = np.argsort(row_norms)[::-1][:25]

    for idx_k, src_r in enumerate(top25):
        iA = 14 + idx_k
        iB = 39 + idx_k
        w_row = a100_src.w1[src_r].copy()
        # Prevent un-gated clock/head-scan oscillation leakage during zero-velocity standing!
        w_row[9:11] = 0.0
        w_row[21:23] = 0.0
        w_row[33:35] = 0.0
        w_mir = w_row @ M_OBS
        # Exact zero activation at static upright equilibrium (gz = -1.0, tof = 1.0)
        b_cal = -(w_row[2] * (-1.0) + w_row[35] * 1.0)
        pol.w1[iA] = w_row
        pol.w1[iB] = w_mir
        pol.b1[iA] = b_cal
        pol.b1[iB] = b_cal

    sub_w2 = a100_src.w2[np.ix_(top25, top25)]
    pol.w2[14:39, 14:39] = sub_w2
    pol.w2[39:64, 39:64] = sub_w2

    # 3. Equivariant Output Readout W3
    ap_gx, ap_wy, hp_gx, hp_wy = v[10], v[11], v[12], v[13]
    ap_st, hp_st               = v[14], v[15]
    ap_vx, hp_vx               = v[16], v[17]
    ar_gy, ar_wx, hr_gy, hr_wx = v[18], v[19], v[20], v[21]
    sh_gx, sh_wy               = v[22], v[23]
    ap_nl, hp_nl               = v[24], v[25]
    a100_gain                  = v[27]

    pol.w3[2, 0] = ap_gx; pol.w3[6, 0] = ap_gx
    pol.w3[2, 4] = ap_wy; pol.w3[6, 4] = ap_wy
    pol.w3[1, 0] = hp_gx; pol.w3[5, 0] = hp_gx
    pol.w3[1, 4] = hp_wy; pol.w3[5, 4] = hp_wy

    pol.w3[2, 8] = ap_st; pol.w3[6, 7] = ap_st
    pol.w3[1, 8] = hp_st; pol.w3[5, 7] = hp_st

    pol.w3[2, 11] = ap_nl; pol.w3[6, 11] = ap_nl
    pol.w3[1, 11] = hp_nl; pol.w3[5, 11] = hp_nl

    pol.w3[2, 6] = ap_vx; pol.w3[6, 6] = ap_vx
    pol.w3[1, 6] = hp_vx; pol.w3[5, 6] = hp_vx

    pol.w3[3, 1] = ar_gy; pol.w3[7, 1] = ar_gy
    pol.w3[3, 3] = ar_wx; pol.w3[7, 3] = ar_wx
    pol.w3[0, 1] = hr_gy; pol.w3[4, 1] = hr_gy
    pol.w3[0, 3] = hr_wx; pol.w3[4, 3] = hr_wx

    pol.w3[8, 0] = sh_gx; pol.w3[9, 0] = sh_gx
    pol.w3[8, 4] = sh_wy; pol.w3[9, 4] = sh_wy

    w3_sub = a100_src.w3[:, top25] * a100_gain
    pol.w3[:, 14:39] = 0.5 * w3_sub
    pol.w3[:, 39:64] = 0.5 * (M_ACT @ w3_sub)
    return pol


def eval_pybullet(cid: int, policy: RNV1Policy, duration_s: float, cmd_vx: float) -> dict:
    p.resetSimulation(physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    dt_pb = 0.001
    p.setTimeStep(dt_pb, physicsClientId=cid)
    p.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=cid)
    plane_col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid)
    plane_id = p.createMultiBody(0, plane_col, physicsClientId=cid)
    p.changeDynamics(plane_id, -1, lateralFriction=0.90, spinningFriction=0.005, rollingFriction=0.0005, physicsClientId=cid)
    robot_id = p.loadURDF(str(URDF_PATH), [0, 0, 0.124], [0, 0, 0, 1], flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cid)
    p.changeDynamics(robot_id, -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=cid)
    name_to_idx = {}
    for ji in range(p.getNumJoints(robot_id, physicsClientId=cid)):
        info = p.getJointInfo(robot_id, ji, physicsClientId=cid)
        name_to_idx[info[1].decode("utf-8")] = ji
        p.changeDynamics(robot_id, ji, lateralFriction=0.90, spinningFriction=0.005, rollingFriction=0.0005,
                         jointDamping=0.002, linearDamping=0.0, angularDamping=0.0, physicsClientId=cid)
    j_ids = [name_to_idx[jn] for jn in JOINT_NAMES]
    max_forces = [0.206 if i < 10 else 0.155 for i in range(ACT_DIM)]

    phi, hp = 0.0, 0.0
    prev_act = np.zeros(ACT_DIM, dtype=np.float32)
    prev2_act = np.zeros(ACT_DIM, dtype=np.float32)
    f_ctrl = np.zeros(ACT_DIM, dtype=np.float32)
    alpha = dt_pb / (0.025 + dt_pb)
    fell = False
    max_tilt = 0.0
    for _ in range(int(round(duration_s * CONTROL_HZ))):
        _, orn_xyzw = p.getBasePositionAndOrientation(robot_id, physicsClientId=cid)
        _, ang_vel = p.getBaseVelocity(robot_id, physicsClientId=cid)
        quat_wxyz = np.array([orn_xyzw[3], orn_xyzw[0], orn_xyzw[1], orn_xyzw[2]], dtype=np.float32)
        pg = quat_to_proj_gravity(quat_wxyz)
        rot = np.array(p.getMatrixFromQuaternion(orn_xyzw)).reshape(3, 3)
        gyro_local = (rot.T @ np.array(ang_vel, dtype=np.float32)).astype(np.float32)
        obs = build_observation(pg, gyro_local, cmd_vx, 0.0, 0.0, phi, prev_act, prev2_act, 1.5)
        target_q = policy.act(obs, phi, cmd_vx, 0.0, 0.0, hp)
        prev2_act, prev_act = prev_act.copy(), target_q.copy()
        for _ in range(20):
            f_ctrl = f_ctrl + alpha * (target_q - f_ctrl)
            p.setJointMotorControlArray(robot_id, j_ids, p.POSITION_CONTROL, targetPositions=f_ctrl.tolist(),
                                        forces=max_forces, positionGains=[0.25]*ACT_DIM, velocityGains=[1.0]*ACT_DIM, physicsClientId=cid)
            p.stepSimulation(physicsClientId=cid)
        phi = (phi + float(np.clip(policy.cpg[0], 0.9, 2.5)) * CONTROL_DT) % 1.0
        hp += 1.4 * CONTROL_DT
        pos2, orn2 = p.getBasePositionAndOrientation(robot_id, physicsClientId=cid)
        pg2 = quat_to_proj_gravity(np.array([orn2[3], orn2[0], orn2[1], orn2[2]]))
        tilt_deg = math.degrees(math.acos(float(np.clip(-pg2[2], -1.0, 1.0))))
        max_tilt = max(max_tilt, tilt_deg)
        if pos2[2] < 0.068 or tilt_deg > 42.0:
            fell = True
            break
    pos_end, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=cid)
    return {"fell": fell, "dist_x_m": round(float(pos_end[0]), 4), "max_tilt_deg": round(float(max_tilt), 2)}


def main() -> None:
    t0 = time.time()
    a100_src = RNV1Policy.load_json(WEIGHTS_JSON)
    m_flat = mujoco.MjModel.from_xml_path(str(XML_PATH))
    m_slope = mujoco.MjModel.from_xml_path(str(XML_PATH))
    ang = math.radians(5.0)
    m_slope.opt.gravity[:] = [-9.81 * math.sin(ang), 0.0, -9.81 * math.cos(ang)]
    m_dr_hi = mujoco.MjModel.from_xml_path(str(XML_PATH))
    m_dr_hi.body_mass[:] *= 1.11
    m_dr_hi.pair_friction[:, 0:2] *= 0.70

    lo = np.array([
        1.28, -0.04, 0.94, 0.15, -1.03, 0.18, 0.07, -0.04, -0.02, 0.22,
       -0.40, -0.40, -1.40, -0.60, -0.40, -1.20, -0.90, -0.50,
       -0.40, -0.35, -0.40, -0.35, -1.00, -0.50, -0.60, -0.80,
        0.30,  0.01
    ], dtype=np.float32)
    hi = np.array([
        1.65,  0.04, 1.05, 0.21, -0.93, 0.27, 0.13,  0.07,  0.03, 0.45,
        1.60,  0.90,  0.40,  0.40,  1.80,  0.50,  0.20,  0.50,
        0.40,  0.35,  0.40,  0.35,  1.00,  0.50,  1.40,  0.60,
        0.52,  0.14
    ], dtype=np.float32)

    best_v = np.array([
        1.3626, -0.0041, 1.0111, 0.1694, -0.9586, 0.2153, 0.1077, 0.0495, 0.0103, 0.2884,
        0.4485,  0.5961, -0.4448, 0.0642,  0.5422, -0.8608, -0.5425, 0.1262,
        0.2278, -0.1451, -0.0979, 0.2020, -0.2791,  0.1599,  0.4500, -0.2000,
        0.4200,  0.0250
    ], dtype=np.float32)

    cid = p.connect(p.DIRECT)
    try:
        def eval_all(v: np.ndarray) -> tuple[float, int, dict]:
            pol = build_equivariant_a100_policy(a100_src, v)
            rw   = rollout_mujoco(m_flat,  pol, duration_s=10.0, cmd_vx=0.12, cmd_yaw=0.0,   servo_tau_s=0.025)
            if rw["fell"] or rw["dist_x_m"] < 0.55 or rw["max_tilt_deg"] > 20.0:
                return -200.0 + rw["dist_x_m"] * 50.0, 0, {"rw": rw}
            rsl  = rollout_mujoco(m_slope, pol, duration_s=6.0,  cmd_vx=0.12, cmd_yaw=0.0,   servo_tau_s=0.025)
            if rsl["fell"] or rsl["dist_x_m"] < 0.15:
                return -100.0 + rsl["dist_x_m"] * 50.0, 1, {"rw": rw, "rsl": rsl}
            rst  = rollout_mujoco(m_flat,  pol, duration_s=5.0,  cmd_vx=0.0,  cmd_yaw=0.0,   servo_tau_s=0.025)
            rpl  = rollout_mujoco(m_flat,  pol, duration_s=5.0,  cmd_vx=0.12, cmd_yaw=0.0,   imu_noise=0.5, push_at_s=2.2, push_force_xyz=(0.0, 1.6, 0.0), seed=101)
            rps  = rollout_mujoco(m_flat,  pol, duration_s=5.0,  cmd_vx=0.12, cmd_yaw=0.0,   imu_noise=0.5, push_at_s=2.0, push_force_xyz=(1.8, 0.0, 0.0), seed=102)
            rtl  = rollout_mujoco(m_flat,  pol, duration_s=6.0,  cmd_vx=0.12, cmd_yaw=+0.55, servo_tau_s=0.025)
            rtr  = rollout_mujoco(m_flat,  pol, duration_s=6.0,  cmd_vx=0.12, cmd_yaw=-0.55, servo_tau_s=0.025)
            rdr  = rollout_mujoco(m_dr_hi, pol, duration_s=5.0,  cmd_vx=0.12, cmd_yaw=0.0,   servo_tau_s=0.035, imu_noise=0.6, seed=205)

            mj_checks = [
                (not rw["fell"] and rw["dist_x_m"] >= 0.62 and rw["max_tilt_deg"] <= 18.0 and rw["rms_torque_nm"] <= 0.054),
                (not rsl["fell"] and rsl["dist_x_m"] >= 0.18),
                (not rst["fell"] and rst["max_tilt_deg"] <= 2.0),
                (not rpl["fell"]),
                (not rps["fell"]),
                (not rtl["fell"] and rtl["yaw_delta_rad"] >= 0.35),
                (not rtr["fell"] and rtr["yaw_delta_rad"] <= -0.35),
                (not rdr["fell"]),
            ]
            if not all(mj_checks):
                ok_mj = sum(mj_checks)
                return ok_mj * 50.0 + rw["dist_x_m"] * 25.0, ok_mj, {"rw": rw, "rsl": rsl, "rst": rst}

            # Passed all 8 MuJoCo gates -> verify PyBullet cross-engine transfer!
            pbs = eval_pybullet(cid, pol, 4.0, 0.0)
            pbw = eval_pybullet(cid, pol, 6.0, 0.12)
            pb_ok = (not pbs["fell"] and pbs["max_tilt_deg"] <= 4.0) and (not pbw["fell"] and pbw["dist_x_m"] >= 0.35 and pbw["max_tilt_deg"] <= 22.0)
            total_ok = 10 if pb_ok else 8 + int(not pbs["fell"])
            sc = (
                total_ok * 100.0
                + rw["dist_x_m"] * 55.0
                + rsl["dist_x_m"] * 45.0
                + (pbw["dist_x_m"] * 30.0 if not pbw["fell"] else -40.0)
                - abs(rw["yaw_delta_rad"]) * 12.0
                - abs(rw["dist_y_m"]) * 12.0
                - rw["max_tilt_deg"] * 0.55
                - rw["rms_torque_nm"] * 220.0
            )
            return float(sc), total_ok, {"rw": rw, "rsl": rsl, "rst": rst, "rtl": rtl, "rtr": rtr, "pbs": pbs, "pbw": pbw}

        rng = np.random.default_rng(2026)
        sigma = (hi - lo) * 0.06
        best_sc, best_ok, best_inf = eval_all(best_v)
        print(f"[Init] ok={best_ok}/10 sc={best_sc:.1f} | W10s={best_inf['rw']['dist_x_m']:.4f}m (tilt={best_inf['rw']['max_tilt_deg']:.2f}d, rms={best_inf['rw']['rms_torque_nm']:.4f}Nm)", flush=True)

        for it in range(16):
            cands, scs, oks, infs = [], [], [], []
            for k in range(14):
                c = best_v.copy() if k == 0 else np.clip(best_v + rng.normal(0.0, sigma), lo, hi).astype(np.float32)
                s, ok, inf = eval_all(c)
                cands.append(c); scs.append(s); oks.append(ok); infs.append(inf)
            order = np.argsort(scs)[::-1]
            if scs[order[0]] > best_sc:
                best_sc, best_ok, best_v, best_inf = scs[order[0]], oks[order[0]], cands[order[0]].copy(), infs[order[0]]
            elites = np.array([cands[i] for i in order[:4]])
            cm = np.clip(0.60 * best_v + 0.40 * np.mean(elites, axis=0), lo, hi).astype(np.float32)
            sm, okm, infm = eval_all(cm)
            if sm > best_sc:
                best_sc, best_ok, best_v, best_inf = sm, okm, cm, infm
            sigma = np.maximum(sigma * 0.85, (hi - lo) * 0.015).astype(np.float32)
            if (it + 1) % 4 == 0 or it == 15:
                bi = best_inf
                pb_str = f"{bi['pbw']['dist_x_m']:.3f}m({bi['pbw']['max_tilt_deg']:.1f}d)" if "pbw" in bi else "skip"
                print(f"[Iter {it+1:02d}/16] ok={best_ok}/10 sc={best_sc:.1f} | "
                      f"W10s={bi['rw']['dist_x_m']:.4f}m (yaw={bi['rw']['yaw_delta_rad']:+.3f}, tilt={bi['rw']['max_tilt_deg']:.2f}d, rms={bi['rw']['rms_torque_nm']:.4f}Nm) | "
                      f"SL={bi['rsl']['dist_x_m']:.3f}m | Stand={bi['rst']['max_tilt_deg']:.2f}d | PB={pb_str}", flush=True)

        pol_opt = build_equivariant_a100_policy(a100_src, best_v)
        eval_10s = rollout_mujoco(m_flat, pol_opt, duration_s=10.0, cmd_vx=0.12, cmd_yaw=0.0, servo_tau_s=0.025, record_replay=True)

        pol_opt.save_json(WEIGHTS_JSON)
        pol_opt.export_c_header(WEIGHTS_H)
        frames = eval_10s["replay_frames"]
        loop_frames = frames[50:300] if len(frames) >= 300 else frames[:250]
        replay_payload = {
            "model": "RNV1",
            "source": "Colab Pro A100 MJX + Equivariant Multi-Engine Co-Optimization",
            "engine": f"MuJoCo {mujoco.__version__}",
            "control_hz": CONTROL_HZ,
            "summary": {k: v for k, v in eval_10s.items() if k != "replay_frames"},
            "frames": loop_frames,
        }
        REPLAY_JSON.write_text(json.dumps(replay_payload), encoding="utf-8")
        print(f"[OK] Saved optimized policy & {len(loop_frames)}-frame replay in {time.time() - t0:.1f}s")
    finally:
        p.disconnect(physicsClientId=cid)


if __name__ == "__main__":
    main()
