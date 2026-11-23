// RNV1 Hardware Configuration & Zero-Conflict XIAO ESP32S3 Sense Pinout
// Single-source-of-truth synchronized with bom/wiring_pinout.json
#pragma once
#include <stdint.h>

// --- I2C Bus (400 kHz Fast Mode) ---
#define PIN_I2C_SDA         5   // XIAO D4 (GPIO5)
#define PIN_I2C_SCL         6   // XIAO D5 (GPIO6)
#define I2C_ADDR_PCA9685    0x40
#define I2C_ADDR_MPU6050    0x68
#define I2C_ADDR_VL53L0X    0x29

// --- I2S_NUM_1 Audio Output -> MAX98357A Class-D 3W Amp ---
#define PIN_I2S_SPK_BCLK    1   // XIAO D0 (GPIO1)
#define PIN_I2S_SPK_LRC     2   // XIAO D1 (GPIO2)
#define PIN_I2S_SPK_DOUT    4   // XIAO D3 (GPIO4)
#define PIN_AMP_SD_MODE     9   // XIAO D10 (GPIO9) - High=Active, Low=Mute/Standby

// --- I2S_NUM_0 Audio Input -> Onboard PDM Digital Mic (B2B Sense Board) ---
#define PIN_PDM_MIC_CLK     42  // Internal Sense B2B
#define PIN_PDM_MIC_DATA    41  // Internal Sense B2B

// --- Control & Safety GPIO ---
#define PIN_TOF_XSHUT       3   // XIAO D2 (GPIO3)
#define PIN_VBAT_ADC        7   // XIAO D8 (GPIO7, ADC1_CH6 via 200k/100k divider: Vbat = Vadc * 3.0)
#define PIN_PCA9685_OE      8   // XIAO D9 (GPIO8, Active-Low; 10k pull-up disables servos at boot!)

// --- Battery & Safety Thresholds (2S 14500 Li-ion 7.4V Nominal) ---
#define VBAT_DIVIDER_RATIO  3.0f    // 200kOhm / 100kOhm divider -> 8.4V maps to 2.80V
#define VBAT_LOW_WARN_V     6.80f   // Speak low-battery warning
#define VBAT_CUTOFF_V       6.40f   // Safe posture sit-down + disable PCA9685 OE#
#define FALL_TILT_MAX_RAD   0.785f  // 45 deg safety fall detector

// --- Servo Calibration (MG90S / SG90 @ 50Hz PWM, 500..2500 us = -90..+90 deg) ---
#define SERVO_PWM_FREQ_HZ   50.0f
#define SERVO_NEUTRAL_US    1500.0f
#define SERVO_US_PER_RAD    636.61977f  // 1000 us / (pi/2)

// PCA9685 Channel Assignment (matches MuJoCo actuator indices 0..11)
enum RNV1JointIndex : uint8_t {
    J_L_HIP_ROLL    = 0,
    J_L_HIP_PITCH   = 1,
    J_L_ANKLE_PITCH = 2,
    J_L_ANKLE_ROLL  = 3,
    J_R_HIP_ROLL    = 4,
    J_R_HIP_PITCH   = 5,
    J_R_ANKLE_PITCH = 6,
    J_R_ANKLE_ROLL  = 7,
    J_L_SHOULDER   = 8,
    J_R_SHOULDER   = 9,
    J_HEAD_YAW     = 10,
    J_HEAD_PITCH   = 11
};
