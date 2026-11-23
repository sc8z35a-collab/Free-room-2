// RNV1 (Robotic Neural Vector 1) - Master Parametric Mechanical Specification
// Units: millimeters (mm)
// Coordinate Frame: +X Forward, +Y Left, +Z Up (Standing Neutral Pose, Floor = Z=0)

$fn = 36;

// --- FDM 3D Print Engineering Tolerances (0.4mm nozzle, PETG / PLA+) ---
WALL_T       = 2.0;    // Primary structural wall thickness (>= 1.6mm requirement)
RIB_T        = 1.8;    // Structural stiffening rib thickness
CLEARANCE    = 0.25;   // Standard FDM press/slip fit clearance
M2_PILOT_D   = 1.75;   // M2 self-tapping pilot hole diameter
M2_CLEAR_D   = 2.3;    // M2 through clearance hole diameter
PIVOT_BOSS_D = 4.0;    // Dual-shear opposing coaxial pivot pin diameter

// --- MG90S Metal-Gear Micro Servo Envelope (TowerPro/Generic standard) ---
MG90S_L      = 22.8;   // Body length along flange axis
MG90S_W      = 12.2;   // Body width
MG90S_H      = 28.5;   // Total housing height (bottom to top gear dome)
MG90S_FLANGE_L = 32.2; // Mounting ear span
MG90S_HOLE_DX  = 27.6; // Mounting hole center-to-center
MG90S_SHAFT_OFF = 5.9; // Output shaft offset from end of 22.8mm body

// --- SG90 9g Micro Servo Envelope ---
SG90_L       = 22.5;
SG90_W       = 12.0;
SG90_H       = 22.5;

// --- Master Kinematic Skeleton Coordinates (mm, Z from Floor = 0) ---
SOLE_PAD_H   = 2.0;    // 2mm silicone anti-slip pad (1mm recessed)
FOOT_L       = 72.0;   // X: -32mm heel to +40mm toe
FOOT_HEEL_X  = -32.0;
FOOT_TOE_X   = 40.0;
FOOT_W       = 46.0;   // Y width per foot
FOOT_PLATE_H = 6.5;    // Rigid foot sole plate thickness

FOOT_Y_CTR   = 26.0;   // Left=+26mm, Right=-26mm (6mm gap between inner edges!)
HIP_ROLL_Y   = 23.0;   // Left=+23mm, Right=-23mm

Z_ANKLE_ROLL = 18.0;   // Ankle roll axis height
Z_KNEE_PITCH = 62.0;   // Knee pitch axis height  (L_shin = 44.0mm)
Z_HIP_PITCH  = 106.0;  // Hip pitch axis height   (L_thigh = 44.0mm)
Z_HIP_ROLL   = 124.0;  // Hip roll axis height    (dZ = 18.0mm)

L_SHIN       = Z_KNEE_PITCH - Z_ANKLE_ROLL; // 44.0 mm
L_THIGH      = Z_HIP_PITCH - Z_KNEE_PITCH;  // 44.0 mm
L_HIP_CLEVIS = Z_HIP_ROLL - Z_HIP_PITCH;    // 18.0 mm

// --- Torso & Pelvis Envelope ---
TORSO_Z_BOT  = 116.0;
TORSO_Z_TOP  = 184.0;
TORSO_H      = TORSO_Z_TOP - TORSO_Z_BOT;   // 68.0 mm
TORSO_W      = 80.0;   // Y: -40 to +40mm
TORSO_D      = 54.0;   // X: -27 to +27mm

// --- Arms ---
SHOULDER_Y   = 46.0;   // Left=+46mm, Right=-46mm
SHOULDER_Z   = 170.0;
ARM_LEN      = 72.0;   // Shoulder to palm tip

// --- 2-DOF Active Spatial Head ---
Z_HEAD_YAW   = 188.0;  // Pan axis Z
Z_HEAD_PITCH = 204.0;  // Tilt axis Z
HEAD_W       = 58.0;   // Y width
HEAD_D       = 44.0;   // X depth
HEAD_H       = 42.0;   // Z height (Z = 196..238mm -> Total Robot Height = 238mm)
HEAD_Z_CTR   = 217.0;

// --- Utility Rounded Box Module ---
module rbox(size, r=2.0, center=true) {
    sx = size[0]; sy = size[1]; sz = size[2];
    rc = min(r, sx/2 - 0.1, sy/2 - 0.1, sz/2 - 0.1);
    translate(center ? [0,0,0] : [sx/2, sy/2, sz/2])
    hull() {
        for (dx = [-sx/2 + rc, sx/2 - rc])
        for (dy = [-sy/2 + rc, sy/2 - rc])
        for (dz = [-sz/2 + rc, sz/2 - rc])
            translate([dx, dy, dz]) sphere(r=rc, $fn=16);
    }
}

module rbox_xy(size, r=2.5, center=true) {
    sx = size[0]; sy = size[1]; sz = size[2];
    rc = min(r, sx/2 - 0.1, sy/2 - 0.1);
    translate(center ? [0,0,0] : [sx/2, sy/2, sz/2])
    linear_extrude(height=sz, center=true)
    offset(r=rc) offset(r=-rc)
        square([sx, sy], center=true);
}
