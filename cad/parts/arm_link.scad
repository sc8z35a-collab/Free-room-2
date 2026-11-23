// RNV1 Part: Lightweight Ribbed Sculpted Humanoid Arm Link (~5.2g each)
include <../rnv1_params.scad>

module arm_link() {
    difference() {
        union() {
            // Shoulder hub
            rotate([90, 0, 0])
                cylinder(d=15.5, h=9.6, center=true);

            // Upper arm tapered strut (Z = 0 to -35mm)
            hull() {
                rotate([90, 0, 0])
                    cylinder(d=14.0, h=9.2, center=true);
                translate([2.0, 0, -35.0])
                    rbox_xy([11.5, 8.8, 7.0], r=2.6, center=true);
            }

            // Forearm ergonomic 15-deg forward curve (Z = -35 to -65mm)
            hull() {
                translate([2.0, 0, -35.0])
                    rbox_xy([11.5, 8.8, 7.0], r=2.6, center=true);
                translate([10.5, 0, -64.0])
                    rbox_xy([9.8, 8.0, 7.0], r=2.5, center=true);
            }

            // Rounded tactile palm bumper
            translate([12.5, 0, -68.0])
                scale([1.15, 0.82, 1.0])
                    sphere(r=6.2, $fn=24);
        }

        // MG90S Horn Recess Pocket on inner shoulder face
        translate([0, -4.0, 0])
            rotate([90, 0, 0])
                hull() {
                    cylinder(d=7.4, h=2.6, center=true);
                    translate([0, -12.0, 0]) cylinder(d=4.4, h=2.6, center=true);
                }
        rotate([90, 0, 0])
            cylinder(d=3.6, h=14.0, center=true);

        // Dual-side I-beam weight-relief pockets along arm
        for (sy = [-4.2, 4.2]) {
            translate([4.5, sy, -34.0])
                rotate([0, -12, 0])
                    rbox_xy([5.5, 4.2, 34.0], r=1.6, center=true);
        }
    }
}

arm_link();
