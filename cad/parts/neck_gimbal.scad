// RNV1 Part: 2-DOF Pan/Tilt Neck Gimbal Yoke
// Local Origin: Head Yaw axis at (0, 0, 0) [Z=188mm], Head Pitch axis at (0, 0, 16mm) [Z=204mm]
include <../rnv1_params.scad>

module neck_gimbal() {
    difference() {
        union() {
            // Circular Yaw Turntable Hub (connects to Torso Neck SG90 horn)
            cylinder(d=20.0, h=5.0, center=true);

            // Tilt SG90 Mounting Cradle (Z = +2.5 to +19mm)
            translate([0, 0, 11.0])
                rbox_xy([16.5, 21.0, 16.5], r=2.0, center=true);

            // Opposing Dual-Shear Tilt Pivot Boss on -Y side at Z = +16mm
            translate([0, -10.5, 16.0])
                rotate([90, 0, 0])
                    cylinder(d=PIVOT_BOSS_D, h=3.5);
        }

        // Bottom Yaw SG90 Horn Recess
        translate([0, 0, -1.6])
            hull() {
                cylinder(d=7.2, h=2.4, center=true);
                translate([11.0, 0, 0]) cylinder(d=4.4, h=2.4, center=true);
            }
        cylinder(d=3.2, h=8.0, center=true);

        // Head Tilt SG90 Servo Cavity (shaft pointing +Y at Z=+16mm)
        translate([0, 0.5, 11.5])
            cube([SG90_W + CLEARANCE, 18.5, 15.0], center=true);
        translate([0, 10.5, 16.0])
            rotate([90, 0, 0])
                cylinder(d=11.8, h=4.0, center=true);
    }
}

neck_gimbal();
