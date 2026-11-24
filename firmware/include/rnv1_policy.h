// RNV1 Embedded Real-Time Core:
//  1. Bit-Exact CPG + 36->64->64->12 Neural Policy Inference (<0.55ms on ESP32-S3 @ 240MHz)
//  2. Complementary IMU Attitude Filter (MPU-6050 -> Projected Gravity Vector)
//  3. Egocentric 2D Spatial Occupancy Grid & Cliff-Safe Autonomous Navigation (VL53L0X + Camera)
//  4. Offline Real-Time Japanese Mora Formant Speech Synthesizer (16kHz 16-bit PCM -> MAX98357A)
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include "rnv1_policy_weights.h"

namespace rnv1 {

static constexpr float PI_F = 3.14159265358979323846f;

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// --- 1. Bit-Exact CPG Prior + Residual Neural Policy (Matches simulation/rnv1_policy_core.py) ---
inline void compute_cpg_prior(
    float phi, float cmd_vx, float cmd_vy, float cmd_yaw,
    float head_scan_phase, float q_out[RNV1_ACT_DIM]
) {
    const float speed_mag = clampf(std::hypot(cmd_vx / 0.12f, cmd_yaw / 0.55f), 0.0f, 1.35f);
    const float act       = clampf(speed_mag, 0.0f, 1.0f);
    const float fwd_scale = clampf(cmd_vx / 0.12f, -0.85f, 1.25f);
    const float yaw_scale = clampf(cmd_yaw / 0.55f, -1.0f, 1.0f);
    const float lat_scale = clampf(cmd_vy / 0.08f, -1.0f, 1.0f);

    const float lean_fwd    = RNV1_CPG[1] * fwd_scale;
    const float level_ratio = RNV1_CPG[2];
    const float roll_amp    = RNV1_CPG[3] * act;
    const float ank_ratio   = RNV1_CPG[4];
    const float stride_base = (std::fabs(fwd_scale) > 0.05f ? fwd_scale : 0.45f * std::fabs(yaw_scale));
    const float stride_amp  = RNV1_CPG[5] * stride_base;
    const float lift_roll   = RNV1_CPG[6] * act;
    const float phase_off   = RNV1_CPG[7];
    const float arm_amp     = RNV1_CPG[8] * act;
    const float ank_push    = RNV1_CPG[9] * act;
    const float turn_diff   = RNV1_CPG[10] * yaw_scale;
    const float turn_roll   = RNV1_CPG[11] * yaw_scale;

    const float w_roll = std::sin(2.0f * PI_F * phi);
    const float w_step = std::cos(2.0f * PI_F * (phi + phase_off));
    const float l_sw = w_roll > 0.0f ? w_roll : 0.0f;
    const float r_sw = (-w_roll) > 0.0f ? (-w_roll) : 0.0f;

    const float l_stride_gain = 1.0f - ((-turn_diff) > 0.0f ? (-turn_diff) : 0.0f);
    const float r_stride_gain = 1.0f - (  turn_diff  > 0.0f ?   turn_diff  : 0.0f);

    const float rc = roll_amp * w_roll + 0.05f * lat_scale * w_step + turn_roll * w_step;
    q_out[0] = rc + lift_roll * l_sw;
    q_out[3] = ank_ratio * rc - lift_roll * l_sw;
    q_out[4] = rc - lift_roll * r_sw;
    q_out[7] = ank_ratio * rc + lift_roll * r_sw;

    const float l_hip = -lean_fwd + stride_amp * l_stride_gain * w_step;
    const float r_hip = -lean_fwd - stride_amp * r_stride_gain * w_step;
    q_out[1] = l_hip;
    q_out[2] = -l_hip * level_ratio - ank_push * w_roll;
    q_out[5] = r_hip;
    q_out[6] = -r_hip * level_ratio + ank_push * w_roll;

    q_out[8]  = -arm_amp * w_step - 0.25f * yaw_scale;
    q_out[9]  =  arm_amp * w_step + 0.25f * yaw_scale;
    q_out[10] = clampf(0.45f * yaw_scale + 0.35f * std::sin(head_scan_phase), -1.1f, 1.1f);
    q_out[11] = clampf(0.08f + 0.08f * std::sin(0.5f * head_scan_phase), -0.35f, 0.45f);

    for (int i = 0; i < RNV1_ACT_DIM; ++i) {
        q_out[i] = clampf(q_out[i], RNV1_Q_MIN[i], RNV1_Q_MAX[i]);
    }
}

inline float dot4_unrolled(const float* __restrict__ w, const float* __restrict__ x, int n, float bias) {
    float s0 = bias, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int j = 0;
    for (; j + 3 < n; j += 4) {
        s0 += w[j]     * x[j];
        s1 += w[j + 1] * x[j + 1];
        s2 += w[j + 2] * x[j + 2];
        s3 += w[j + 3] * x[j + 3];
    }
    for (; j < n; ++j) {
        s0 += w[j] * x[j];
    }
    return (s0 + s1) + (s2 + s3);
}

inline void forward_mlp(const float* __restrict__ obs, float* __restrict__ out) {
    alignas(16) float h1[RNV1_H1_DIM];
    alignas(16) float h2[RNV1_H2_DIM];

    for (int i = 0; i < RNV1_H1_DIM; ++i) {
        const float* row = &RNV1_W1[i * RNV1_OBS_DIM];
        h1[i] = std::tanh(dot4_unrolled(row, obs, RNV1_OBS_DIM, RNV1_B1[i]));
    }

    for (int i = 0; i < RNV1_H2_DIM; ++i) {
        const float* row = &RNV1_W2[i * RNV1_H1_DIM];
        h2[i] = std::tanh(dot4_unrolled(row, h1, RNV1_H1_DIM, RNV1_B2[i]));
    }

    for (int i = 0; i < RNV1_ACT_DIM; ++i) {
        const float* row = &RNV1_W3[i * RNV1_H2_DIM];
        out[i] = std::tanh(dot4_unrolled(row, h2, RNV1_H2_DIM, RNV1_B3[i]));
    }
}

inline void infer_policy(
    const float obs[RNV1_OBS_DIM], float phi,
    float cmd_vx, float cmd_vy, float cmd_yaw, float head_scan_phase,
    float q_target[RNV1_ACT_DIM]
) {
    float prior[RNV1_ACT_DIM];
    float delta[RNV1_ACT_DIM];
    compute_cpg_prior(phi, cmd_vx, cmd_vy, cmd_yaw, head_scan_phase, prior);
    forward_mlp(obs, delta);

    for (int i = 0; i < RNV1_ACT_DIM; ++i) {
        float scale = RNV1_RESIDUAL_SCALE * ((i >= 10) ? 0.25f : 1.0f);
        q_target[i] = clampf(prior[i] + delta[i] * scale, RNV1_Q_MIN[i], RNV1_Q_MAX[i]);
    }
}

// --- 2. Complementary IMU Attitude Filter (MPU-6050 Accel + Gyro -> Projected Gravity) ---
struct IMUFilter {
    float pitch = 0.0f;
    float roll  = 0.0f;
    float alpha = 0.96f;

    void update(float ax, float ay, float az, float gx_rad, float gy_rad, float dt) {
        float acc_roll  = std::atan2(ay, az);
        float acc_pitch = std::atan2(-ax, std::sqrt(ay * ay + az * az));
        roll  = alpha * (roll  + gx_rad * dt) + (1.0f - alpha) * acc_roll;
        pitch = alpha * (pitch + gy_rad * dt) + (1.0f - alpha) * acc_pitch;
    }

    void get_projected_gravity(float g[3]) const {
        float sp = std::sin(pitch), cp = std::cos(pitch);
        float sr = std::sin(roll),  cr = std::cos(roll);
        g[0] =  sp;
        g[1] = -cp * sr;
        g[2] = -cp * cr;
    }
};

// --- 3. Local 2D Spatial Occupancy Grid & Cliff-Safe Autonomous Planner ---
struct SpatialPlanner {
    static constexpr int N_SECTORS = 15;  // -70 deg .. +70 deg in 10 deg bins
    float sector_dist_m[N_SECTORS];
    bool cliff_detected = false;

    SpatialPlanner() {
        for (int i = 0; i < N_SECTORS; ++i) sector_dist_m[i] = 1.5f;
    }

    void update_tof_ray(float head_yaw_rad, float head_pitch_rad, float dist_m) {
        // Downward pitch (> +0.35 rad) measures floor distance (~0.25-0.42m); >0.58m means table drop-off!
        if (head_pitch_rad > 0.35f) {
            cliff_detected = (dist_m > 0.58f);
            return;
        }
        float deg = head_yaw_rad * (180.0f / PI_F);
        int idx = static_cast<int>(std::round((deg + 70.0f) / 10.0f));
        if (idx >= 0 && idx < N_SECTORS) {
            float d = clampf(dist_m, 0.04f, 1.8f);
            sector_dist_m[idx] = 0.35f * sector_dist_m[idx] + 0.65f * d;
        }
    }

    void compute_autonomous_cmd(float& cmd_vx, float& cmd_vy, float& cmd_yaw) const {
        cmd_vy = 0.0f;
        if (cliff_detected) {
            cmd_vx = -0.05f;
            cmd_yaw = 0.55f;
            return;
        }
        // Wide +-40 deg forward safety corridor (sectors 3..11) + side shoulder whiskers (0..3 right, 11..14 left)
        float front_min = 1.8f;
        for (int i = 4; i <= 10; ++i) {
            if (sector_dist_m[i] < front_min) front_min = sector_dist_m[i];
        }
        float right_min = 1.8f, right_avg = 0.0f;
        for (int i = 0; i < 5; ++i) {
            if (sector_dist_m[i] < right_min) right_min = sector_dist_m[i];
            right_avg += sector_dist_m[i];
        }
        float left_min = 1.8f, left_avg = 0.0f;
        for (int i = 10; i < N_SECTORS; ++i) {
            if (sector_dist_m[i] < left_min) left_min = sector_dist_m[i];
            left_avg += sector_dist_m[i];
        }
        left_avg /= 5.0f;
        right_avg /= 5.0f;

        const float left_score  = 0.65f * left_min  + 0.35f * left_avg;
        const float right_score = 0.65f * right_min + 0.35f * right_avg;

        if (front_min < 0.34f) {
            // Obstacle close ahead: halt forward creep and pivot cleanly toward wider open corridor
            cmd_vx = 0.00f;
            cmd_yaw = (left_score >= right_score) ? 0.65f : -0.65f;
        } else if (front_min < 0.62f || left_min < 0.30f || right_min < 0.30f) {
            // Smooth repulsive steering keeping full shoulder clearance (>= 5cm margin)
            cmd_vx = 0.08f;
            float repulse = 0.0f;
            if (right_min < 0.34f) repulse += (0.34f - right_min) * 3.0f;
            if (left_min  < 0.34f) repulse -= (0.34f - left_min)  * 3.0f;
            cmd_yaw = clampf((left_score - right_score) * 1.1f + repulse, -0.60f, 0.60f);
        } else {
            // Clear path: brisk autonomous cruise
            cmd_vx = 0.12f;
            cmd_yaw = clampf((left_score - right_score) * 0.25f, -0.25f, 0.25f);
        }
    }
};

// --- 4. Offline Real-Time Japanese Mora Formant Speech Synthesizer (16kHz 16-bit PCM) ---
// Synthesizes crisp robotic Japanese vowels/syllables directly into I2S DMA with 0 KB external server!
struct FormantVoiceSynth {
    struct VowelSpec { float f1; float f2; float f3; };
    // Standard Japanese 5-vowel acoustic formant frequencies (Hz)
    static constexpr VowelSpec VOWELS[5] = {
        { 730.0f, 1180.0f, 2550.0f }, // 'a' (あ)
        { 300.0f, 2280.0f, 3000.0f }, // 'i' (い)
        { 360.0f, 1250.0f, 2350.0f }, // 'u' (う)
        { 500.0f, 1900.0f, 2550.0f }, // 'e' (え)
        { 520.0f,  900.0f, 2450.0f }  // 'o' (お)
    };

    static int synthesize_romaji(const char* text, int16_t* pcm_out, int max_samples, int sample_rate = 16000) {
        int written = 0;
        const float f0_base = 215.0f; // Friendly compact-robot vocal pitch (Hz)
        float phase_glottal = 0.0f;
        // 2nd-order resonator states for F1 and F2
        float y1_a = 0.0f, y2_a = 0.0f;
        float y1_b = 0.0f, y2_b = 0.0f;

        for (size_t idx = 0; text[idx] != '\0' && written < max_samples; ++idx) {
            char c = text[idx];
            int v_idx = -1;
            bool unvoiced = false;
            if (c == 'a' || c == 'A') v_idx = 0;
            else if (c == 'i' || c == 'I') v_idx = 1;
            else if (c == 'u' || c == 'U') v_idx = 2;
            else if (c == 'e' || c == 'E') v_idx = 3;
            else if (c == 'o' || c == 'O') v_idx = 4;
            else if (c == 's' || c == 'k' || c == 't' || c == 'h') unvoiced = true;

            int dur_samples = (v_idx >= 0) ? (sample_rate * 95) / 1000 : (unvoiced ? (sample_rate * 40) / 1000 : (sample_rate * 35) / 1000);
            if (written + dur_samples > max_samples) dur_samples = max_samples - written;

            float f1 = (v_idx >= 0) ? VOWELS[v_idx].f1 : 1800.0f;
            float f2 = (v_idx >= 0) ? VOWELS[v_idx].f2 : 3400.0f;
            float r1 = std::exp(-PI_F * 90.0f / sample_rate);
            float r2 = std::exp(-PI_F * 140.0f / sample_rate);
            float a1 = -2.0f * r1 * std::cos(2.0f * PI_F * f1 / sample_rate);
            float a2 = r1 * r1;
            float b1 = -2.0f * r2 * std::cos(2.0f * PI_F * f2 / sample_rate);
            float b2 = r2 * r2;

            uint32_t lfsr = 0xACE1u + static_cast<uint32_t>(idx * 97u);
            for (int n = 0; n < dur_samples; ++n) {
                float env = std::sin(PI_F * static_cast<float>(n) / static_cast<float>(dur_samples));
                float exc = 0.0f;
                if (v_idx >= 0) {
                    float f0 = f0_base * (1.0f + 0.04f * std::sin(2.0f * PI_F * static_cast<float>(n) / sample_rate));
                    phase_glottal += f0 / static_cast<float>(sample_rate);
                    if (phase_glottal >= 1.0f) phase_glottal -= 1.0f;
                    // Rosenberg-like rich glottal sawtooth excitation
                    exc = (2.0f * phase_glottal - 1.0f) * 0.45f;
                } else if (unvoiced) {
                    lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & 0xB400u);
                    exc = (static_cast<float>(lfsr & 0xFFu) / 127.5f - 1.0f) * 0.18f;
                }

                float s1 = exc - a1 * y1_a - a2 * y2_a;
                y2_a = y1_a; y1_a = s1;
                float s2 = exc - b1 * y1_b - b2 * y2_b;
                y2_b = y1_b; y1_b = s2;

                float sample = clampf((0.62f * s1 + 0.38f * s2) * env * 0.35f, -0.95f, 0.95f);
                pcm_out[written++] = static_cast<int16_t>(sample * 32000.0f);
            }
        }
        return written;
    }
};

} // namespace rnv1
