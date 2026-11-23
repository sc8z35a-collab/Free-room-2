#!/usr/bin/env python3
"""Generate exact, single-source-of-truth MuJoCo MJCF (MJX-compatible) and PyBullet URDF
models directly from compiled OpenSCAD STL manifest + verified BOM component masses.
Architecture: 12-DOF RNV1 with Coaxial 2-DOF Universal Ankle (Hip Roll, Hip Pitch,
Ankle Pitch, Ankle Roll per leg + 2 Arms + 2-DOF Active Spatial Head).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "cad" / "stl" / "cad_manifest.json"
BOM_PATH = ROOT / "bom" / "rnv1_bom.json"
XML_OUT = ROOT / "simulation" / "rnv1_humanoid.xml"
URDF_OUT = ROOT / "simulation" / "rnv1_humanoid.urdf"

def box_inertia(m, lx, ly, lz):
    ixx = m * (ly**2 + lz**2) / 12.0
    iyy = m * (lx**2 + lz**2) / 12.0
    izz = m * (lx**2 + ly**2) / 12.0
    return (max(ixx, 1e-6), max(iyy, 1e-6), max(izz, 1e-6))

def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    parts_by_name = {p["name"]: p for p in manifest["parts"]}

    m_torso_cad = (parts_by_name["torso_chassis"]["unit_mass_g"] +
                   parts_by_name["torso_back_cover"]["unit_mass_g"]) / 1000.0
    m_head_cad  = (parts_by_name["head_shell"]["unit_mass_g"] +
                   parts_by_name["head_visor"]["unit_mass_g"]) / 1000.0
    m_neck_cad  = parts_by_name["neck_gimbal"]["unit_mass_g"] / 1000.0
    m_hip_cad   = parts_by_name["hip_clevis_left"]["unit_mass_g"] / 1000.0
    m_leg_cad   = parts_by_name["leg_link"]["unit_mass_g"] / 1000.0
    m_ank_cad   = parts_by_name["ankle_clevis"]["unit_mass_g"] / 1000.0
    m_foot_cad  = parts_by_name["foot_sole"]["unit_mass_g"] / 1000.0
    m_arm_cad   = parts_by_name["arm_link_left"]["unit_mass_g"] / 1000.0

    M_MG90S = 0.0134
    M_SG90  = 0.0090

    m_torso = (
        m_torso_cad
        + 0.0510  # 2S 14500 Li-ion pack + BMS + holder + switch (pelvis bay)
        + 0.0085  # PCA9685
        + 0.0021  # MPU-6050 IMU
        + 0.0115  # MAX98357A + 28mm 4ohm 3W speaker
        + 0.0220  # Dual Bucks + 1000uF cap + wiring
        + 4 * M_MG90S  # 2x Hip Roll + 2x Shoulder MG90S
        + 1 * M_SG90   # 1x Neck Pan SG90
        + 0.0060       # M2 torso hardware
    )
    m_neck  = m_neck_cad + 1 * M_SG90 + 0.0006
    m_head  = m_head_cad + 0.0065 + 0.0018 + 0.0010
    m_arm   = m_arm_cad + 0.0008
    m_hip   = m_hip_cad + M_MG90S + 0.0008   # Houses Hip Pitch MG90S
    m_leg   = m_leg_cad + M_MG90S + 0.0008   # Houses Ankle Pitch MG90S at lower end -> ULTRA LOW CoM!
    m_ankle = m_ank_cad + M_MG90S + 0.0008   # Houses Ankle Roll MG90S coaxially at Z=18mm!
    m_foot  = m_foot_cad + 0.0030 + 0.0008   # Foot sole + 2mm silicone anti-slip pad

    total_mass_kg = m_torso + m_neck + m_head + 2 * (m_arm + m_hip + m_leg + m_ankle + m_foot)
    total_printed_g = manifest["total_printed_mass_g"]

    bom = json.loads(BOM_PATH.read_text(encoding="utf-8"))
    for item in bom["components"]:
        if item["id"] == "structure_fasteners_grip":
            hw_mass_g = 17.0
            item["unit_mass_g"] = round(total_printed_g + hw_mass_g, 2)
            item["total_mass_g"] = round(total_printed_g + hw_mass_g, 2)
            item["specs"] = (
                f"{total_printed_g:.1f}g PETG/PLA+ thin-wall monocoque (exact Trimesh STL volume @ 0.85g/cm3) "
                f"+ 44x M2 screws + 2mm silicone sole pads (mu>=0.90) + 22AWG silicone wiring"
            )
    bom["summary"]["total_mass_g"] = round(total_mass_kg * 1000.0, 2)
    bom["summary"]["printed_frame_mass_g"] = round(total_printed_g, 2)
    BOM_PATH.write_text(json.dumps(bom, indent=2, ensure_ascii=False), encoding="utf-8")

    I_torso = box_inertia(m_torso, 0.054, 0.080, 0.068)
    I_neck  = box_inertia(m_neck,  0.020, 0.024, 0.020)
    I_head  = box_inertia(m_head,  0.044, 0.058, 0.042)
    I_arm   = box_inertia(m_arm,   0.016, 0.012, 0.072)
    I_hip   = box_inertia(m_hip,   0.024, 0.022, 0.024)
    I_leg   = box_inertia(m_leg,   0.019, 0.024, 0.088)
    I_ankle = box_inertia(m_ankle, 0.022, 0.024, 0.022)
    I_foot  = box_inertia(m_foot,  0.072, 0.046, 0.016)

    xml = f"""<mujoco model="rnv1_humanoid">
  <compiler angle="radian" coordinate="local" inertiafromgeom="false"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" iterations="25" ls_iterations="10">
    <flag eulerdamp="enable"/>
  </option>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <rgba haze="1 1 1 1"/>
  </visual>

  <default>
    <joint damping="0.019" armature="0.00045" frictionloss="0.003" limited="true"/>
    <geom contype="0" conaffinity="0" condim="3" friction="0.90 0.015 0.004" solref="0.012 1.0" solimp="0.9 0.95 0.001"/>
    <default class="mg90s_leg">
      <position kp="2.8" kv="0.065" forcerange="-0.206 0.206"/>
    </default>
    <default class="mg90s_arm">
      <position kp="2.4" kv="0.055" forcerange="-0.206 0.206"/>
    </default>
    <default class="sg90_head">
      <position kp="1.6" kv="0.035" forcerange="-0.155 0.155"/>
    </default>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.96 0.96 0.97" rgb2="0.90 0.91 0.93" width="256" height="256"/>
    <material name="floor_mat" texture="grid" texrepeat="12 12" reflectance="0.05"/>
    <material name="shell_white" rgba="0.95 0.96 0.97 1"/>
    <material name="servo_blue" rgba="0.16 0.36 0.68 1"/>
    <material name="visor_dark" rgba="0.09 0.11 0.15 1"/>
    <material name="sole_rubber" rgba="0.18 0.19 0.21 1"/>
  </asset>

  <worldbody>
    <light pos="0.5 -0.8 1.5" dir="-0.3 0.5 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="20 20 0.05" material="floor_mat" contype="1" conaffinity="1"/>

    <!-- TORSO + PELVIS UNIBODY (Origin at Z_HIP_ROLL = 0.124 m) -->
    <body name="torso" pos="0 0 0.124">
      <freejoint name="root"/>
      <inertial pos="0.001 0 0.022" mass="{m_torso:.5f}" diaginertia="{I_torso[0]:.7f} {I_torso[1]:.7f} {I_torso[2]:.7f}"/>
      <geom name="torso_geom" type="box" pos="0 0 0.026" size="0.027 0.040 0.034" material="shell_white" contype="2" conaffinity="1"/>
      <site name="imu_site" pos="0.002 0 0.012" size="0.004" rgba="1 0.2 0.2 1"/>

      <!-- 2-DOF ACTIVE SPATIAL HEAD -->
      <body name="neck_gimbal" pos="0 0 0.064">
        <inertial pos="0 0 0.009" mass="{m_neck:.5f}" diaginertia="{I_neck[0]:.7f} {I_neck[1]:.7f} {I_neck[2]:.7f}"/>
        <joint name="head_yaw" type="hinge" axis="0 0 1" range="-1.309 1.309" damping="0.012"/>
        <geom name="neck_geom" type="cylinder" pos="0 0 0.008" size="0.010 0.009" material="servo_blue"/>

        <body name="head" pos="0 0 0.016">
          <inertial pos="0.003 0 0.013" mass="{m_head:.5f}" diaginertia="{I_head[0]:.7f} {I_head[1]:.7f} {I_head[2]:.7f}"/>
          <joint name="head_pitch" type="hinge" axis="0 1 0" range="-0.524 0.611" damping="0.012"/>
          <geom name="head_geom" type="box" pos="0 0 0.013" size="0.022 0.029 0.021" material="shell_white" contype="2" conaffinity="1"/>
          <geom name="visor_geom" type="box" pos="0.0215 0 0.0135" size="0.002 0.021 0.012" material="visor_dark"/>
          <camera name="head_cam" pos="0.023 0 0.015" xyaxes="0 -1 0 0 0 1" fovy="62"/>
          <site name="tof_site" pos="0.023 0 0.005" zaxis="1 0 0" size="0.002" rgba="0 0.8 1 1"/>
        </body>
      </body>

      <!-- LEFT & RIGHT ARMS -->
      <body name="arm_L" pos="0 0.046 0.046">
        <inertial pos="0.004 0 -0.032" mass="{m_arm:.5f}" diaginertia="{I_arm[0]:.7f} {I_arm[1]:.7f} {I_arm[2]:.7f}"/>
        <joint name="l_shoulder" type="hinge" axis="0 1 0" range="-1.658 2.618"/>
        <geom name="arm_L_geom" type="capsule" fromto="0 0 0 0.011 0 -0.068" size="0.006" material="shell_white" contype="2" conaffinity="1"/>
      </body>

      <body name="arm_R" pos="0 -0.046 0.046">
        <inertial pos="0.004 0 -0.032" mass="{m_arm:.5f}" diaginertia="{I_arm[0]:.7f} {I_arm[1]:.7f} {I_arm[2]:.7f}"/>
        <joint name="r_shoulder" type="hinge" axis="0 1 0" range="-1.658 2.618"/>
        <geom name="arm_R_geom" type="capsule" fromto="0 0 0 0.011 0 -0.068" size="0.006" material="shell_white" contype="2" conaffinity="1"/>
      </body>

      <!-- LEFT LEG (4-DOF: Hip Roll, Hip Pitch, Coaxial Universal Ankle Pitch + Ankle Roll) -->
      <body name="hip_clevis_L" pos="0 0.023 0">
        <inertial pos="0 0.002 -0.010" mass="{m_hip:.5f}" diaginertia="{I_hip[0]:.7f} {I_hip[1]:.7f} {I_hip[2]:.7f}"/>
        <joint name="l_hip_roll" type="hinge" axis="1 0 0" range="-0.524 0.611"/>
        <geom name="hip_L_geom" type="box" pos="0 0.002 -0.009" size="0.011 0.010 0.011" material="servo_blue"/>

        <body name="leg_L" pos="0 0.003 -0.018">
          <!-- Ankle Pitch MG90S mounted at bottom of 88mm leg strut -> lowers CoM! -->
          <inertial pos="0 0 -0.056" mass="{m_leg:.5f}" diaginertia="{I_leg[0]:.7f} {I_leg[1]:.7f} {I_leg[2]:.7f}"/>
          <joint name="l_hip_pitch" type="hinge" axis="0 1 0" range="-0.960 0.960"/>
          <geom name="leg_L_geom" type="box" pos="0 0 -0.044" size="0.010 0.011 0.044" material="shell_white"/>

          <body name="ankle_L" pos="0 0 -0.088">
            <inertial pos="0 0 0.002" mass="{m_ankle:.5f}" diaginertia="{I_ankle[0]:.7f} {I_ankle[1]:.7f} {I_ankle[2]:.7f}"/>
            <joint name="l_ankle_pitch" type="hinge" axis="0 1 0" range="-0.785 0.785"/>
            <geom name="ankle_L_geom" type="box" pos="0 0 0.002" size="0.010 0.010 0.010" material="servo_blue"/>

            <body name="foot_L" pos="0 0 0">
              <inertial pos="0.003 0 -0.013" mass="{m_foot:.5f}" diaginertia="{I_foot[0]:.7f} {I_foot[1]:.7f} {I_foot[2]:.7f}"/>
              <joint name="l_ankle_roll" type="hinge" axis="1 0 0" range="-0.611 0.611"/>
              <geom name="foot_L_sole" type="box" pos="0.003 0 -0.015" size="0.024 0.023 0.003" material="sole_rubber" contype="4" conaffinity="1"/>
              <geom name="foot_L_toe" type="box" pos="0.032 0 -0.0135" euler="0 -0.14 0" size="0.008 0.021 0.003" material="shell_white" contype="4" conaffinity="1"/>
              <geom name="foot_L_heel" type="box" pos="-0.025 0 -0.0135" euler="0 0.14 0" size="0.007 0.021 0.003" material="shell_white" contype="4" conaffinity="1"/>
              <site name="l_foot_site" pos="0.003 0 -0.018" size="0.024 0.023 0.002" type="box" rgba="0 1 0 0.0"/>
            </body>
          </body>
        </body>
      </body>

      <!-- RIGHT LEG (4-DOF: Hip Roll, Hip Pitch, Coaxial Universal Ankle Pitch + Ankle Roll) -->
      <body name="hip_clevis_R" pos="0 -0.023 0">
        <inertial pos="0 -0.002 -0.010" mass="{m_hip:.5f}" diaginertia="{I_hip[0]:.7f} {I_hip[1]:.7f} {I_hip[2]:.7f}"/>
        <joint name="r_hip_roll" type="hinge" axis="1 0 0" range="-0.611 0.524"/>
        <geom name="hip_R_geom" type="box" pos="0 -0.002 -0.009" size="0.011 0.010 0.011" material="servo_blue"/>

        <body name="leg_R" pos="0 -0.003 -0.018">
          <inertial pos="0 0 -0.056" mass="{m_leg:.5f}" diaginertia="{I_leg[0]:.7f} {I_leg[1]:.7f} {I_leg[2]:.7f}"/>
          <joint name="r_hip_pitch" type="hinge" axis="0 1 0" range="-0.960 0.960"/>
          <geom name="leg_R_geom" type="box" pos="0 0 -0.044" size="0.010 0.011 0.044" material="shell_white"/>

          <body name="ankle_R" pos="0 0 -0.088">
            <inertial pos="0 0 0.002" mass="{m_ankle:.5f}" diaginertia="{I_ankle[0]:.7f} {I_ankle[1]:.7f} {I_ankle[2]:.7f}"/>
            <joint name="r_ankle_pitch" type="hinge" axis="0 1 0" range="-0.785 0.785"/>
            <geom name="ankle_R_geom" type="box" pos="0 0 0.002" size="0.010 0.010 0.010" material="servo_blue"/>

            <body name="foot_R" pos="0 0 0">
              <inertial pos="0.003 0 -0.013" mass="{m_foot:.5f}" diaginertia="{I_foot[0]:.7f} {I_foot[1]:.7f} {I_foot[2]:.7f}"/>
              <joint name="r_ankle_roll" type="hinge" axis="1 0 0" range="-0.611 0.611"/>
              <geom name="foot_R_sole" type="box" pos="0.003 0 -0.015" size="0.024 0.023 0.003" material="sole_rubber" contype="4" conaffinity="1"/>
              <geom name="foot_R_toe" type="box" pos="0.032 0 -0.0135" euler="0 -0.14 0" size="0.008 0.021 0.003" material="shell_white" contype="4" conaffinity="1"/>
              <geom name="foot_R_heel" type="box" pos="-0.025 0 -0.0135" euler="0 0.14 0" size="0.007 0.021 0.003" material="shell_white" contype="4" conaffinity="1"/>
              <site name="r_foot_site" pos="0.003 0 -0.018" size="0.024 0.023 0.002" type="box" rgba="0 1 0 0.0"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <pair geom1="floor" geom2="foot_L_sole" friction="0.90 0.90 0.015 0.004 0.004"/>
    <pair geom1="floor" geom2="foot_L_toe"  friction="0.80 0.80 0.015 0.004 0.004"/>
    <pair geom1="floor" geom2="foot_L_heel" friction="0.80 0.80 0.015 0.004 0.004"/>
    <pair geom1="floor" geom2="foot_R_sole" friction="0.90 0.90 0.015 0.004 0.004"/>
    <pair geom1="floor" geom2="foot_R_toe"  friction="0.80 0.80 0.015 0.004 0.004"/>
    <pair geom1="floor" geom2="foot_R_heel" friction="0.80 0.80 0.015 0.004 0.004"/>
    <pair geom1="floor" geom2="torso_geom"  friction="0.60 0.60 0.01 0.002 0.002"/>
    <pair geom1="floor" geom2="head_geom"   friction="0.60 0.60 0.01 0.002 0.002"/>
    <pair geom1="floor" geom2="arm_L_geom"  friction="0.80 0.80 0.01 0.002 0.002"/>
    <pair geom1="floor" geom2="arm_R_geom"  friction="0.80 0.80 0.01 0.002 0.002"/>
  </contact>

  <actuator>
    <position name="act_l_hip_roll"    joint="l_hip_roll"    class="mg90s_leg" ctrlrange="-0.524 0.611"/>
    <position name="act_l_hip_pitch"   joint="l_hip_pitch"   class="mg90s_leg" ctrlrange="-0.960 0.960"/>
    <position name="act_l_ankle_pitch" joint="l_ankle_pitch" class="mg90s_leg" ctrlrange="-0.785 0.785"/>
    <position name="act_l_ankle_roll"  joint="l_ankle_roll"  class="mg90s_leg" ctrlrange="-0.611 0.611"/>
    <position name="act_r_hip_roll"    joint="r_hip_roll"    class="mg90s_leg" ctrlrange="-0.611 0.524"/>
    <position name="act_r_hip_pitch"   joint="r_hip_pitch"   class="mg90s_leg" ctrlrange="-0.960 0.960"/>
    <position name="act_r_ankle_pitch" joint="r_ankle_pitch" class="mg90s_leg" ctrlrange="-0.785 0.785"/>
    <position name="act_r_ankle_roll"  joint="r_ankle_roll"  class="mg90s_leg" ctrlrange="-0.611 0.611"/>
    <position name="act_l_shoulder"    joint="l_shoulder"    class="mg90s_arm" ctrlrange="-1.658 2.618"/>
    <position name="act_r_shoulder"    joint="r_shoulder"    class="mg90s_arm" ctrlrange="-1.658 2.618"/>
    <position name="act_head_yaw"      joint="head_yaw"      class="sg90_head" ctrlrange="-1.309 1.309"/>
    <position name="act_head_pitch"    joint="head_pitch"    class="sg90_head" ctrlrange="-0.524 0.611"/>
  </actuator>

  <sensor>
    <gyro name="imu_gyro" site="imu_site"/>
    <accelerometer name="imu_accel" site="imu_site"/>
    <framequat name="imu_quat" objtype="site" objname="imu_site"/>
    <rangefinder name="tof_range" site="tof_site"/>
    <touch name="l_foot_touch" site="l_foot_site"/>
    <touch name="r_foot_touch" site="r_foot_site"/>
  </sensor>

  <keyframe>
    <key name="home" qpos="0 0 0.124 1 0 0 0  0 0  0 0  0 0 0 0  0 0 0 0"/>
  </keyframe>
</mujoco>
"""
    XML_OUT.write_text(xml, encoding="utf-8")

    urdf = f"""<?xml version="1.0"?>
<robot name="rnv1_humanoid">
  <link name="torso">
    <inertial>
      <origin xyz="0.001 0 0.022" rpy="0 0 0"/>
      <mass value="{m_torso:.5f}"/>
      <inertia ixx="{I_torso[0]:.7f}" ixy="0" ixz="0" iyy="{I_torso[1]:.7f}" iyz="0" izz="{I_torso[2]:.7f}"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.026" rpy="0 0 0"/>
      <geometry><box size="0.054 0.080 0.068"/></geometry>
    </collision>
  </link>

  <link name="neck_gimbal">
    <inertial><origin xyz="0 0 0.009"/><mass value="{m_neck:.5f}"/>
      <inertia ixx="{I_neck[0]:.7f}" ixy="0" ixz="0" iyy="{I_neck[1]:.7f}" iyz="0" izz="{I_neck[2]:.7f}"/>
    </inertial>
  </link>
  <joint name="head_yaw" type="revolute">
    <parent link="torso"/><child link="neck_gimbal"/>
    <origin xyz="0 0 0.064"/><axis xyz="0 0 1"/>
    <limit lower="-1.309" upper="1.309" effort="0.155" velocity="10.0"/>
    <dynamics damping="0.012" friction="0.002"/>
  </joint>

  <link name="head">
    <inertial><origin xyz="0.003 0 0.013"/><mass value="{m_head:.5f}"/>
      <inertia ixx="{I_head[0]:.7f}" ixy="0" ixz="0" iyy="{I_head[1]:.7f}" iyz="0" izz="{I_head[2]:.7f}"/>
    </inertial>
    <collision><origin xyz="0 0 0.013"/><geometry><box size="0.044 0.058 0.042"/></geometry></collision>
  </link>
  <joint name="head_pitch" type="revolute">
    <parent link="neck_gimbal"/><child link="head"/>
    <origin xyz="0 0 0.016"/><axis xyz="0 1 0"/>
    <limit lower="-0.524" upper="0.611" effort="0.155" velocity="10.0"/>
    <dynamics damping="0.012" friction="0.002"/>
  </joint>

  <link name="arm_L">
    <inertial><origin xyz="0.004 0 -0.032"/><mass value="{m_arm:.5f}"/>
      <inertia ixx="{I_arm[0]:.7f}" ixy="0" ixz="0" iyy="{I_arm[1]:.7f}" iyz="0" izz="{I_arm[2]:.7f}"/>
    </inertial>
    <collision><origin xyz="0.005 0 -0.034"/><geometry><box size="0.012 0.010 0.068"/></geometry></collision>
  </link>
  <joint name="l_shoulder" type="revolute">
    <parent link="torso"/><child link="arm_L"/>
    <origin xyz="0 0.046 0.046"/><axis xyz="0 1 0"/>
    <limit lower="-1.658" upper="2.618" effort="0.206" velocity="11.0"/>
    <dynamics damping="0.019" friction="0.003"/>
  </joint>

  <link name="arm_R">
    <inertial><origin xyz="0.004 0 -0.032"/><mass value="{m_arm:.5f}"/>
      <inertia ixx="{I_arm[0]:.7f}" ixy="0" ixz="0" iyy="{I_arm[1]:.7f}" iyz="0" izz="{I_arm[2]:.7f}"/>
    </inertial>
    <collision><origin xyz="0.005 0 -0.034"/><geometry><box size="0.012 0.010 0.068"/></geometry></collision>
  </link>
  <joint name="r_shoulder" type="revolute">
    <parent link="torso"/><child link="arm_R"/>
    <origin xyz="0 -0.046 0.046"/><axis xyz="0 1 0"/>
    <limit lower="-1.658" upper="2.618" effort="0.206" velocity="11.0"/>
    <dynamics damping="0.019" friction="0.003"/>
  </joint>

  <!-- LEFT LEG -->
  <link name="hip_clevis_L">
    <inertial><origin xyz="0 0.002 -0.010"/><mass value="{m_hip:.5f}"/>
      <inertia ixx="{I_hip[0]:.7f}" ixy="0" ixz="0" iyy="{I_hip[1]:.7f}" iyz="0" izz="{I_hip[2]:.7f}"/>
    </inertial>
  </link>
  <joint name="l_hip_roll" type="revolute">
    <parent link="torso"/><child link="hip_clevis_L"/>
    <origin xyz="0 0.023 0"/><axis xyz="1 0 0"/>
    <limit lower="-0.524" upper="0.611" effort="0.206" velocity="11.0"/>
    <dynamics damping="0.019" friction="0.003"/>
  </joint>

  <link name="leg_L">
    <inertial><origin xyz="0 0 -0.056"/><mass value="{m_leg:.5f}"/>
      <inertia ixx="{I_leg[0]:.7f}" ixy="0" ixz="0" iyy="{I_leg[1]:.7f}" iyz="0" izz="{I_leg[2]:.7f}"/>
    </inertial>
  </link>
  <joint name="l_hip_pitch" type="revolute">
    <parent link="hip_clevis_L"/><child link="leg_L"/>
    <origin xyz="0 0.003 -0.018"/><axis xyz="0 1 0"/>
    <limit lower="-0.960" upper="0.960" effort="0.206" velocity="11.0"/>
    <dynamics damping="0.019" friction="0.003"/>
  </joint>

  <link name="ankle_L">
    <inertial><origin xyz="0 0 0.002"/><mass value="{m_ankle:.5f}"/>
      <inertia ixx="{I_ankle[0]:.7f}" ixy="0" ixz="0" iyy="{I_ankle[1]:.7f}" iyz="0" izz="{I_ankle[2]:.7f}"/>
    </inertial>
  </link>
  <joint name="l_ankle_pitch" type="revolute">
    <parent link="leg_L"/><child link="ankle_L"/>
    <origin xyz="0 0 -0.088"/><axis xyz="0 1 0"/>
    <limit lower="-0.785" upper="0.785" effort="0.206" velocity="11.0"/>
    <dynamics damping="0.019" friction="0.003"/>
  </joint>

  <link name="foot_L">
    <inertial><origin xyz="0.003 0 -0.013"/><mass value="{m_foot:.5f}"/>
      <inertia ixx="{I_foot[0]:.7f}" ixy="0" ixz="0" iyy="{I_foot[1]:.7f}" iyz="0" izz="{I_foot[2]:.7f}"/>
    </inertial>
    <collision><origin xyz="0.003 0 -0.015"/><geometry><box size="0.048 0.046 0.006"/></geometry></collision>
    <collision><origin xyz="0.032 0 -0.0135" rpy="0 -0.14 0"/><geometry><box size="0.016 0.042 0.006"/></geometry></collision>
    <collision><origin xyz="-0.025 0 -0.0135" rpy="0 0.14 0"/><geometry><box size="0.014 0.042 0.006"/></geometry></collision>
  </link>
  <joint name="l_ankle_roll" type="revolute">
    <parent link="ankle_L"/><child link="foot_L"/>
    <origin xyz="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-0.611" upper="0.611" effort="0.206" velocity="11.0"/>
    <dynamics damping="0.019" friction="0.003"/>
  </joint>

  <!-- RIGHT LEG -->
  <link name="hip_clevis_R">
    <inertial><origin xyz="0 -0.002 -0.010"/><mass value="{m_hip:.5f}"/>
      <inertia ixx="{I_hip[0]:.7f}" ixy="0" ixz="0" iyy="{I_hip[1]:.7f}" iyz="0" izz="{I_hip[2]:.7f}"/>
    </inertial>
  </link>
  <joint name="r_hip_roll" type="revolute">
    <parent link="torso"/><child link="hip_clevis_R"/>
    <origin xyz="0 -0.023 0"/><axis xyz="1 0 0"/>
    <limit lower="-0.611" upper="0.524" effort="0.206" velocity="11.0"/>
    <dynamics damping="0.019" friction="0.003"/>
  </joint>

  <link name="leg_R">
    <inertial><origin xyz="0 0 -0.056"/><mass value="{m_leg:.5f}"/>
      <inertia ixx="{I_leg[0]:.7f}" ixy="0" ixz="0" iyy="{I_leg[1]:.7f}" iyz="0" izz="{I_leg[2]:.7f}"/>
    </inertial>
  </link>
  <joint name="r_hip_pitch" type="revolute">
    <parent link="hip_clevis_R"/><child link="leg_R"/>
    <origin xyz="0 -0.003 -0.018"/><axis xyz="0 1 0"/>
    <limit lower="-0.960" upper="0.960" effort="0.206" velocity="11.0"/>
    <dynamics damping="0.019" friction="0.003"/>
  </joint>

  <link name="ankle_R">
    <inertial><origin xyz="0 0 0.002"/><mass value="{m_ankle:.5f}"/>
      <inertia ixx="{I_ankle[0]:.7f}" ixy="0" ixz="0" iyy="{I_ankle[1]:.7f}" iyz="0" izz="{I_ankle[2]:.7f}"/>
    </inertial>
  </link>
  <joint name="r_ankle_pitch" type="revolute">
    <parent link="leg_R"/><child link="ankle_R"/>
    <origin xyz="0 0 -0.088"/><axis xyz="0 1 0"/>
    <limit lower="-0.785" upper="0.785" effort="0.206" velocity="11.0"/>
    <dynamics damping="0.019" friction="0.003"/>
  </joint>

  <link name="foot_R">
    <inertial><origin xyz="0.003 0 -0.013"/><mass value="{m_foot:.5f}"/>
      <inertia ixx="{I_foot[0]:.7f}" ixy="0" ixz="0" iyy="{I_foot[1]:.7f}" iyz="0" izz="{I_foot[2]:.7f}"/>
    </inertial>
    <collision><origin xyz="0.003 0 -0.015"/><geometry><box size="0.048 0.046 0.006"/></geometry></collision>
    <collision><origin xyz="0.032 0 -0.0135" rpy="0 -0.14 0"/><geometry><box size="0.016 0.042 0.006"/></geometry></collision>
    <collision><origin xyz="-0.025 0 -0.0135" rpy="0 0.14 0"/><geometry><box size="0.014 0.042 0.006"/></geometry></collision>
  </link>
  <joint name="r_ankle_roll" type="revolute">
    <parent link="ankle_R"/><child link="foot_R"/>
    <origin xyz="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-0.611" upper="0.611" effort="0.206" velocity="11.0"/>
    <dynamics damping="0.019" friction="0.003"/>
  </joint>
</robot>
"""
    URDF_OUT.write_text(urdf, encoding="utf-8")
    print(f"[OK] Built Universal-Ankle MuJoCo MJCF & PyBullet URDF | Printed={total_printed_g:.2f}g | Total={total_mass_kg*1000:.2f}g")

if __name__ == "__main__":
    main()
