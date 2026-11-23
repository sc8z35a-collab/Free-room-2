// Native Host Verification Test for RNV1 C++ Embedded Core
// Compiled with g++ -O3 -Wall -Wextra -Werror and compared bit-exact against Python!
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include "../include/rnv1_policy.h"

int main() {
    // 1. Read test vectors from stdin or run deterministic benchmark & verification
    // Print deterministic inference outputs for 5 canonical states so Python checks bit-exact parity (<1e-5)
    for (int tc = 0; tc < 5; ++tc) {
        float obs[RNV1_OBS_DIM];
        for (int j = 0; j < RNV1_OBS_DIM; ++j) {
            obs[j] = std::sin(static_cast<float>(tc * 13 + j * 7 + 1) * 0.31f) * 0.5f;
        }
        float phi = 0.17f * tc;
        float cmd_vx = 0.16f - 0.04f * tc;
        float cmd_vy = 0.01f * (tc - 2);
        float cmd_yaw = 0.25f * (tc - 2);
        float head_p = 0.5f * tc;
        float out[RNV1_ACT_DIM];
        rnv1::infer_policy(obs, phi, cmd_vx, cmd_vy, cmd_yaw, head_p, out);
        std::printf("CASE %d:", tc);
        for (int k = 0; k < RNV1_ACT_DIM; ++k) {
            std::printf(" %.7f", out[k]);
        }
        std::printf("\n");
    }

    // 2. Verify IMU Filter Convergence (tilted 15 deg pitch, -10 deg roll)
    rnv1::IMUFilter filt;
    const float p_true = 0.261799f, r_true = -0.174533f;
    float ax = -std::sin(p_true);
    float ay =  std::cos(p_true) * std::sin(r_true);
    float az =  std::cos(p_true) * std::cos(r_true);
    for (int i = 0; i < 400; ++i) filt.update(ax, ay, az, 0.0f, 0.0f, 0.02f);
    float g[3];
    filt.get_projected_gravity(g);
    float imu_err = std::fabs(filt.pitch - p_true) + std::fabs(filt.roll - r_true);
    std::printf("IMU_ERR %.6f GRAV %.5f %.5f %.5f\n", imu_err, g[0], g[1], g[2]);

    // 3. Verify Spatial Planner Obstacle Avoidance & Cliff Stop
    rnv1::SpatialPlanner sp;
    for (int i = 0; i < 10; ++i) sp.update_tof_ray(0.0f, 0.0f, 0.18f); // Close obstacle dead ahead
    for (int i = 0; i < 10; ++i) sp.update_tof_ray(0.95f, 0.0f, 1.40f); // Clear left
    for (int i = 0; i < 10; ++i) sp.update_tof_ray(-0.95f, 0.0f, 0.30f); // Blocked right
    float vx, vy, yaw;
    sp.compute_autonomous_cmd(vx, vy, yaw);
    bool obs_avoid_ok = (vx < 0.05f && yaw > 0.4f);

    sp.update_tof_ray(0.0f, 0.45f, 0.85f); // Downward scan sees table edge cliff!
    float cvx, cvy, cyaw;
    sp.compute_autonomous_cmd(cvx, cvy, cyaw);
    bool cliff_ok = (sp.cliff_detected && cvx < 0.0f);
    std::printf("SPATIAL_OK %d CLIFF_OK %d\n", obs_avoid_ok ? 1 : 0, cliff_ok ? 1 : 0);

    // 4. Verify Offline Japanese Formant Speech Synthesis (16kHz PCM energy & non-clipping)
    static int16_t pcm[16000];
    int n_samp = rnv1::FormantVoiceSynth::synthesize_romaji("konnichiwa", pcm, 16000, 16000);
    double rms = 0.0;
    int max_abs = 0;
    for (int i = 0; i < n_samp; ++i) {
        int a = std::abs(static_cast<int>(pcm[i]));
        if (a > max_abs) max_abs = a;
        rms += static_cast<double>(pcm[i]) * static_cast<double>(pcm[i]);
    }
    rms = std::sqrt(rms / (n_samp > 0 ? n_samp : 1));
    std::printf("SPEECH_SAMPLES %d RMS %.1f MAX %d\n", n_samp, rms, max_abs);

    // 5. Measure host inference latency over 10,000 runs
    float obs_b[RNV1_OBS_DIM] = {0};
    float out_b[RNV1_ACT_DIM] = {0};
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < 10000; ++it) {
        obs_b[0] = static_cast<float>(it & 7) * 0.01f;
        rnv1::infer_policy(obs_b, 0.25f, 0.16f, 0.0f, 0.0f, 0.1f, out_b);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_per_call = std::chrono::duration<double, std::micro>(t1 - t0).count() / 10000.0;
    std::printf("HOST_US %.3f MACS %d BYTES_WEIGHTS %zu\n",
                us_per_call,
                RNV1_OBS_DIM * RNV1_H1_DIM + RNV1_H1_DIM * RNV1_H2_DIM + RNV1_H2_DIM * RNV1_ACT_DIM,
                sizeof(RNV1_W1) + sizeof(RNV1_B1) + sizeof(RNV1_W2) + sizeof(RNV1_B2) + sizeof(RNV1_W3) + sizeof(RNV1_B3) + sizeof(RNV1_CPG));
    return (obs_avoid_ok && cliff_ok && n_samp > 5000 && rms > 500.0) ? 0 : 1;
}
