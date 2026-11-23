// RNV1 Part: Hip Roll-to-Pitch Clevis (Connects Hip Roll at Z=+18mm to Hip Pitch at Z=0)
// Local Origin: Hip Pitch axis at (0, 0, 0), Hip Roll axis at (0, -3mm [for L], +18mm)
include <../rnv1_params.scad>

module hip_clevis(side = 1) {
    // side = +1 for Left leg, -1 for Right leg
    y_roll = -side * (FOOT_Y_CTR - HIP_ROLL_Y); // -3mm for Left, +3mm for Right
    difference() {
        union() {
            // Lower Hip Pitch MG90S housing (shaft along +Y at Z=0)
            translate([0, 0, 2.0])
                rbox_xy([20.0, 19.6, 22.0], r=2.5, center=true);

            // Opposing coaxial dual-shear pivot boss at Y = -9.8mm (for Left)
            translate([0, -9.8, 0])
                rotate([90, 0, 0])
                    cylinder(d=PIVOT_BOSS_D, h=3.6);

            // Upper Dual-Shear Hip Roll Clevis Bridge (connects to Pelvis Hip Roll MG90S at Z=+18mm)
            for (sx = [-12.5, 12.5]) {
                hull() {
                    translate([sx, 0, 9.5])
                        rbox_xy([4.2, 18.0, 4.0], r=1.2, center=true);
                    translate([sx, y_roll, L_HIP_CLEVIS])
                        rotate([0, 90, 0])
                            cylinder(d=13.0, h=4.2, center=true);
                }
            }
        }

        // MG90S Hip Pitch Servo Pocket
        translate([0, 0.4, 2.5])
            cube([MG90S_W + CLEARANCE, 18.8, 20.0], center=true);
        translate([0, 9.8, 0])
            rotate([90, 0, 0])
                cylinder(d=12.6, h=5.0, center=true);

        // Hip Roll Clevis Bores (front +X = servo horn, rear -X = opposing pivot boss)
        translate([12.5, y_roll, L_HIP_CLEVIS])
            rotate([0, 90, 0])
                cylinder(d=6.2, h=8.0, center=true);
        translate([-12.5, y_roll, L_HIP_CLEVIS])
            rotate([0, 90, 0])
                cylinder(d=PIVOT_BOSS_D + CLEARANCE, h=8.0, center=true);
    }
}

hip_clevis(1);
