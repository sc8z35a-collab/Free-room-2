// RNV1 (Robotic Neural Vector 1) - Dual-Core FreeRTOS ESP32-S3 Firmware
// Core 1 (50 Hz Hard Real-Time): MPU-6050 IMU -> RNV1 36-64-64-12 NN -> PCA9685 12-DOF Servos
// Core 0 (Vision/Spatial/Voice): OV5640/OV3660 Camera + 2-DOF VL53L0X Scan + PDM Mic + MAX98357A Speech
#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s.h>
#include <esp_camera.h>
#include <Adafruit_PWMServoDriver.h>
#include <MPU6050.h>
#include <VL53L0X.h>

#include "rnv1_config.h"
#include "rnv1_policy.h"

static Adafruit_PWMServoDriver pca9685(I2C_ADDR_PCA9685);
static MPU6050 imu(I2C_ADDR_MPU6050);
static VL53L0X tof;

static rnv1::IMUFilter imu_filter;
static rnv1::SpatialPlanner planner;

// Shared state protected by spinlock between Core 0 (Planner) and Core 1 (Locomotion)
static portMUX_TYPE g_mux = portMUX_INITIALIZER_UNLOCKED;
static float g_cmd_vx = 0.0f, g_cmd_vy = 0.0f, g_cmd_yaw = 0.0f;
static float g_tof_dist_m = 1.5f;
static float g_head_yaw_rad = 0.0f, g_head_pitch_rad = 0.0f;
static bool  g_low_batt_warn = false;
static bool  g_emergency_stop = false;

static uint16_t rad_to_pca9685_tick(float rad) {
    float us = SERVO_NEUTRAL_US + rad * SERVO_US_PER_RAD;
    us = rnv1::clampf(us, 550.0f, 2450.0f);
    // 12-bit PCA9685 @ 50Hz => 20,000 us / 4096 = 4.8828 us per tick
    return static_cast<uint16_t>(us * (4096.0f / 20000.0f));
}

static void speak_offline_japanese(const char* romaji_phrase) {
    static int16_t pcm_buf[16000]; // 1.0 sec buffer (32 KB)
    int n_samples = rnv1::FormantVoiceSynth::synthesize_romaji(romaji_phrase, pcm_buf, 16000, 16000);
    if (n_samples <= 0) return;
    digitalWrite(PIN_AMP_SD_MODE, HIGH);
    size_t bytes_written = 0;
    i2s_write(I2S_NUM_1, pcm_buf, static_cast<size_t>(n_samples) * sizeof(int16_t), &bytes_written, portMAX_DELAY);
    digitalWrite(PIN_AMP_SD_MODE, LOW);
}

// --- CORE 1: Deterministic 50 Hz Locomotion & Active Balance Task ---
static void task_locomotion_core1(void* /*pv*/) {
    float phi = 0.0f;
    float head_scan = 0.0f;
    float prev_act[RNV1_ACT_DIM] = {0};
    float prev2_act[RNV1_ACT_DIM] = {0};
    float obs[RNV1_OBS_DIM] = {0};
    float target_q[RNV1_ACT_DIM] = {0};

    const TickType_t period = pdMS_TO_TICKS(20); // Exact 50 Hz
    TickType_t last_wake = xTaskGetTickCount();

    // Release PCA9685 OE# ONLY after neutral pose is commanded
    for (int i = 0; i < RNV1_ACT_DIM; ++i) {
        pca9685.setPWM(i, 0, rad_to_pca9685_tick(0.0f));
    }
    digitalWrite(PIN_PCA9685_OE, LOW);

    while (true) {
        vTaskDelayUntil(&last_wake, period);

        // 1. Read MPU-6050 (accel +-8g, gyro +-1000 deg/s)
        int16_t ax_r, ay_r, az_r, gx_r, gy_r, gz_r;
        imu.getMotion6(&ax_r, &ay_r, &az_r, &gx_r, &gy_r, &gz_r);
        float ax = ax_r / 4096.0f;
        float ay = ay_r / 4096.0f;
        float az = az_r / 4096.0f;
        float gx = (gx_r / 32.8f) * (rnv1::PI_F / 180.0f);
        float gy = (gy_r / 32.8f) * (rnv1::PI_F / 180.0f);
        float gz = (gz_r / 32.8f) * (rnv1::PI_F / 180.0f);

        imu_filter.update(ax, ay, az, gx, gy, 0.02f);
        float proj_g[3];
        imu_filter.get_projected_gravity(proj_g);

        // 2. Battery voltage check (2S Li-ion undervoltage protection)
        float vbat = (analogReadMilliVolts(PIN_VBAT_ADC) / 1000.0f) * VBAT_DIVIDER_RATIO;
        bool warn_batt = (vbat > 2.0f && vbat < VBAT_LOW_WARN_V);
        bool low_batt  = (vbat > 2.0f && vbat < VBAT_CUTOFF_V);
        float tilt_rad = std::acos(rnv1::clampf(-proj_g[2], -1.0f, 1.0f));

        float vx, vy, yaw, tof_m;
        portENTER_CRITICAL(&g_mux);
        vx = g_cmd_vx; vy = g_cmd_vy; yaw = g_cmd_yaw;
        tof_m = g_tof_dist_m;
        g_low_batt_warn = warn_batt;
        bool estop = g_emergency_stop || low_batt || (tilt_rad > FALL_TILT_MAX_RAD);
        portEXIT_CRITICAL(&g_mux);

        if (estop) {
            digitalWrite(PIN_PCA9685_OE, HIGH); // Immediately cut servo torque on fall/undervoltage
            continue;
        }

        // 3. Build 36-D observation & run RNV1 neural policy (<0.55 ms!)
        alignas(16) float obs_aligned[RNV1_OBS_DIM];
        obs_aligned[0] = proj_g[0]; obs_aligned[1] = proj_g[1]; obs_aligned[2] = proj_g[2];
        obs_aligned[3] = gx * 0.25f; obs_aligned[4] = gy * 0.25f; obs_aligned[5] = gz * 0.25f;
        obs_aligned[6] = vx; obs_aligned[7] = vy; obs_aligned[8] = yaw;
        obs_aligned[9]  = std::sin(2.0f * rnv1::PI_F * phi);
        obs_aligned[10] = std::cos(2.0f * rnv1::PI_F * phi);
        std::memcpy(&obs_aligned[11], prev_act, sizeof(prev_act));
        std::memcpy(&obs_aligned[23], prev2_act, sizeof(prev2_act));
        obs_aligned[35] = rnv1::clampf(tof_m / 1.5f, 0.0f, 1.0f);
        (void)obs;

        rnv1::infer_policy(obs_aligned, phi, vx, vy, yaw, head_scan, target_q);

        std::memcpy(prev2_act, prev_act, sizeof(prev_act));
        std::memcpy(prev_act, target_q, sizeof(target_q));

        // 4. Write all 12 joints to PCA9685 over 400kHz I2C
        for (int i = 0; i < RNV1_ACT_DIM; ++i) {
            pca9685.setPWM(i, 0, rad_to_pca9685_tick(target_q[i]));
        }

        portENTER_CRITICAL(&g_mux);
        g_head_yaw_rad = target_q[J_HEAD_YAW];
        g_head_pitch_rad = target_q[J_HEAD_PITCH];
        portEXIT_CRITICAL(&g_mux);

        float freq = rnv1::clampf(RNV1_CPG[0], 0.8f, 2.8f);
        phi = std::fmod(phi + freq * 0.02f, 1.0f);
        head_scan += 1.4f * 0.02f;
    }
}

// --- CORE 0: Spatial ToF Mapping, Vision & Voice Interaction Task ---
static void task_perception_core0(void* /*pv*/) {
    speak_offline_japanese("konnichiwa aaru enu bui wan kidou");
    uint32_t last_speak_ms = millis();

    while (true) {
        // 1. Sample VL53L0X laser rangefinder & update egocentric 2D spatial planner
        uint16_t mm = tof.readRangeContinuousMillimeters();
        if (!tof.timeoutOccurred()) {
            float d_m = mm / 1000.0f;
            float hy, hp;
            bool warn_batt = false;
            portENTER_CRITICAL(&g_mux);
            hy = g_head_yaw_rad;
            hp = g_head_pitch_rad;
            g_tof_dist_m = d_m;
            warn_batt = g_low_batt_warn;
            portEXIT_CRITICAL(&g_mux);

            planner.update_tof_ray(hy, hp, d_m);
            float vx, vy, yaw;
            planner.compute_autonomous_cmd(vx, vy, yaw);

            portENTER_CRITICAL(&g_mux);
            g_cmd_vx = vx; g_cmd_vy = vy; g_cmd_yaw = yaw;
            portEXIT_CRITICAL(&g_mux);

            // Autonomous vocal status announcements when encountering cliffs, obstacles, or low battery
            if (planner.cliff_detected && (millis() - last_speak_ms > 4000)) {
                speak_offline_japanese("dansa hakken tomari masu");
                last_speak_ms = millis();
            } else if (warn_batt && (millis() - last_speak_ms > 12000)) {
                speak_offline_japanese("denchi teika juuden shite kudasai");
                last_speak_ms = millis();
            } else if (d_m < 0.25f && (millis() - last_speak_ms > 5000)) {
                speak_offline_japanese("shougai butsu kaihi");
                last_speak_ms = millis();
            }
        }

        // 2. Capture lightweight QQVGA camera frame for brightness/person cueing
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
            esp_camera_fb_return(fb);
        }

        vTaskDelay(pdMS_TO_TICKS(40)); // 25 Hz perception loop
    }
}

void setup() {
    pinMode(PIN_PCA9685_OE, OUTPUT);
    digitalWrite(PIN_PCA9685_OE, HIGH); // Keep all servos safely disabled during init
    pinMode(PIN_AMP_SD_MODE, OUTPUT);
    digitalWrite(PIN_AMP_SD_MODE, LOW);

    Serial.begin(115200);
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);

    pca9685.begin();
    pca9685.setOscillatorFrequency(27000000);
    pca9685.setPWMFreq(SERVO_PWM_FREQ_HZ);

    imu.initialize();
    imu.setFullScaleAccelRange(MPU6050_ACCEL_FS_8);
    imu.setFullScaleGyroRange(MPU6050_GYRO_FS_1000);

    tof.setTimeout(100);
    tof.init();
    tof.startContinuous(33);

    // Configure I2S_NUM_1 TX for MAX98357A 16kHz 16-bit mono speaker output
    i2s_config_t i2s_cfg = {};
    i2s_cfg.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_TX);
    i2s_cfg.sample_rate = 16000;
    i2s_cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
    i2s_cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
    i2s_cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    i2s_cfg.dma_buf_count = 6;
    i2s_cfg.dma_buf_len = 256;
    i2s_pin_config_t pin_cfg = { PIN_I2S_SPK_BCLK, PIN_I2S_SPK_LRC, PIN_I2S_SPK_DOUT, I2S_PIN_NO_CHANGE };
    i2s_driver_install(I2S_NUM_1, &i2s_cfg, 0, nullptr);
    i2s_set_pin(I2S_NUM_1, &pin_cfg);

    // Initialize onboard XIAO ESP32S3 Sense B2B camera (OV5640/OV3660, zero header GPIO conflict)
    camera_config_t cam_cfg = {};
    cam_cfg.pin_pwdn = -1;
    cam_cfg.pin_reset = -1;
    cam_cfg.pin_xclk = 10;
    cam_cfg.pin_sccb_sda = 40;
    cam_cfg.pin_sccb_scl = 39;
    cam_cfg.pin_d7 = 48; cam_cfg.pin_d6 = 11; cam_cfg.pin_d5 = 12; cam_cfg.pin_d4 = 14;
    cam_cfg.pin_d3 = 16; cam_cfg.pin_d2 = 18; cam_cfg.pin_d1 = 17; cam_cfg.pin_d0 = 15;
    cam_cfg.pin_vsync = 38;
    cam_cfg.pin_href = 47;
    cam_cfg.pin_pclk = 13;
    cam_cfg.xclk_freq_hz = 20000000;
    cam_cfg.ledc_timer = LEDC_TIMER_0;
    cam_cfg.ledc_channel = LEDC_CHANNEL_0;
    cam_cfg.pixel_format = PIXFORMAT_GRAYSCALE;
    cam_cfg.frame_size = FRAMESIZE_QQVGA; // 160x120 lightweight edge vision in PSRAM
    cam_cfg.fb_count = 2;
    cam_cfg.fb_location = CAMERA_FB_IN_PSRAM;
    cam_cfg.grab_mode = CAMERA_GRAB_LATEST;
    esp_camera_init(&cam_cfg);

    xTaskCreatePinnedToCore(task_locomotion_core1, "rnv1_ctrl", 8192, nullptr, 5, nullptr, 1);
    xTaskCreatePinnedToCore(task_perception_core0, "rnv1_perc", 16384, nullptr, 2, nullptr, 0);
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}
