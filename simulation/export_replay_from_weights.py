#!/usr/bin/env python3
"""Evaluate simulation/rnv1_policy_weights.json in MuJoCo and export
simulation/rnv1_gait_replay.json for the 3D CAD Web Viewer.

Use this after dropping Colab Pro A100 trained weights (RNV1_Trained_Bundle.zip)
into simulation/rnv1_policy_weights.json and firmware/include/rnv1_policy_weights.h.
"""
from __future__ import annotations
import json
from pathlib import Path
import mujoco

from rnv1_policy_core import CONTROL_HZ, RNV1Policy
from train_rnv1_local import rollout_mujoco

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "simulation" / "rnv1_humanoid.xml"
WEIGHTS_JSON = ROOT / "simulation" / "rnv1_policy_weights.json"
REPLAY_JSON = ROOT / "simulation" / "rnv1_gait_replay.json"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    policy = RNV1Policy.load_json(WEIGHTS_JSON)

    eval_10s = rollout_mujoco(
        model,
        policy,
        duration_s=10.0,
        cmd_vx=0.12,
        cmd_yaw=0.0,
        servo_tau_s=0.025,
        record_replay=True,
    )
    print(
        f"[MuJoCo {mujoco.__version__}] 10s Walk: fell={eval_10s['fell']} | "
        f"survived={eval_10s['survived_s']:.2f}s | dist_x={eval_10s['dist_x_m']:.4f}m | "
        f"vx={eval_10s['avg_vx_mps']:.4f}m/s | max_tilt={eval_10s['max_tilt_deg']:.2f}deg | "
        f"peak_tau={eval_10s['peak_torque_nm']:.4f}Nm | rms_tau={eval_10s['rms_torque_nm']:.4f}Nm"
    )

    frames = eval_10s["replay_frames"]
    # Export a clean 5.0s (250 frames @ 50Hz) loop segment once gait reaches steady stride
    if len(frames) >= 300:
        loop_frames = frames[50:300]
    else:
        loop_frames = frames[:250]

    replay_payload = {
        "model": "RNV1",
        "source": "Colab Pro A100 MJX Pure-JAX PPO (RNV1_Trained_Bundle.zip)",
        "engine": f"MuJoCo {mujoco.__version__}",
        "control_hz": CONTROL_HZ,
        "summary": {k: v for k, v in eval_10s.items() if k != "replay_frames"},
        "frames": loop_frames,
    }
    REPLAY_JSON.write_text(json.dumps(replay_payload), encoding="utf-8")
    print(f"[OK] Exported {len(loop_frames)} replay frames -> {REPLAY_JSON}")


if __name__ == "__main__":
    main()
