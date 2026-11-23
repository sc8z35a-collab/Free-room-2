// RNV1 Part: Thin-Wall Expressive Humanoid Sensor Head Shell (2.0mm shell, ~10g)
// Local Origin: Head Pitch axis at (0, 0, 0) [Z=204mm], top at Z=+34mm [Z=238mm]
include <../rnv1_params.scad>

module head_shell() {
    difference() {
        union() {
            // 1.6mm Thin-Wall Sculpted Cranium (4x 0.4mm FDM perimeters)
            difference() {
                translate([0, 0, 13.0])
                    rbox([HEAD_D, HEAD_W, HEAD_H], r=11.0, center=true);
                translate([0, 0, 13.0])
                    rbox([HEAD_D - 3.2, HEAD_W - 3.2, HEAD_H - 3.2], r=9.4, center=true);
            }

            // Subtle Side Ear Accent Rings
            for (sy = [-HEAD_W/2, HEAD_W/2]) {
                translate([-1.0, sy, 14.0])
                    rotate([90, 0, 0])
                        cylinder(d=16.0, h=2.6, center=true);
            }

            // Internal Dual-Shear Neck Tilt Mounting Ears (Y = +-13.2mm)
            for (sy = [-13.2, 13.2]) {
                translate([0, sy, 3.0])
                    rbox_xy([13.0, 3.2, 13.5], r=1.5, center=true);
            }
        }

        // Bottom Neck Gimbal Clearance Pocket
        translate([0, 0, -4.5])
            rbox_xy([26.0, 23.0, 17.0], r=3.0, center=true);

        // Front Visor Bezel Opening
        translate([HEAD_D/2 - 0.5, 0, 13.5])
            rbox([5.5, HEAD_W - 14.4, 25.4], r=5.6, center=true);

        // PDM Digital Microphone Acoustic Port
        for (sy = [-4.0, 0.0, 4.0]) {
            translate([HEAD_D/2 - 3.0, sy, 26.5])
                rotate([0, 70, 0])
                    cylinder(d=1.8, h=8.0, center=true);
        }

        // Rear Chimney Thermal Exhaust Vents (direct natural convection over XIAO dual heatsinks)
        for (dz = [8, 13, 18, 23]) {
            translate([-HEAD_D/2, 0, dz])
                rbox_xy([10.0, 28.0, 2.0], r=0.8, center=true);
        }

        // Neck Tilt Axis Bores
        translate([0, 13.2, 0])
            rotate([90, 0, 0])
                cylinder(d=6.2, h=6.0, center=true);
        translate([0, -13.2, 0])
            rotate([90, 0, 0])
                cylinder(d=PIVOT_BOSS_D + CLEARANCE, h=6.0, center=true);
    }

    // Internal Precision Snap-Rails for XIAO ESP32S3 Sense (21 x 17.8mm)
    for (sy = [-10.5, 10.5]) {
        translate([9.0, sy, 15.0])
            cube([12.0, 1.8, 16.0], center=true);
    }
}

head_shell();
