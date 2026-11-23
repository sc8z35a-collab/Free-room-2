// RNV1 Part: Lightweight Truss-Ribbed Foot Sole with 8-deg Rocker Bevel & Dual-Shear Ankle Clevis
// Local Origin: Foot bottom Z=0, Ankle Roll axis at (0, 0, Z_ANKLE_ROLL=18mm)
include <../rnv1_params.scad>

module foot_sole() {
    difference() {
        union() {
            // Thin-wall 4.0mm rocker-beveled sole deck (flat central 48x46mm zone + 8deg front/rear rockers)
            hull() {
                translate([3.0, 0, 1.8])
                    rbox_xy([48.0, FOOT_W, 3.6], r=5.0, center=true);
                translate([FOOT_HEEL_X + 4.5, 0, 4.2])
                    rbox_xy([9.0, FOOT_W - 4.0, 3.0], r=4.0, center=true);
                translate([FOOT_TOE_X - 4.5, 0, 4.2])
                    rbox_xy([9.0, FOOT_W - 4.0, 3.0], r=4.5, center=true);
            }

            // Perimeter Stiffening Rim + Diagonal X-Truss Spine Ribs (high bending stiffness, low mass)
            for (sy = [-15.5, 15.5]) {
                translate([3.0, sy, 5.2])
                    rbox_xy([52.0, 2.6, 3.6], r=0.8, center=true);
            }
            for (sx = [-12.5, 12.5]) {
                translate([sx, 0, 5.2])
                    rbox_xy([3.2, 34.0, 3.6], r=0.8, center=true);
            }

            // Dual-Shear Ankle Roll Clevis Towers (front X=+12.5mm & rear X=-12.5mm)
            for (sx = [-12.5, 12.5]) {
                hull() {
                    translate([sx, 0, 5.5])
                        rbox_xy([3.8, 20.0, 3.0], r=1.2, center=true);
                    translate([sx, 0, Z_ANKLE_ROLL])
                        rotate([0, 90, 0])
                            cylinder(d=12.6, h=3.8, center=true);
                }
            }
        }

        // 1.0mm recessed pocket under central flat stance zone for 2mm silicone grip pad
        translate([3.0, 0, 0.5])
            rbox_xy([44.0, FOOT_W - 6.0, 1.2], r=3.5, center=true);

        // Upper deck truss weight-relief pockets (leaves solid 2.2mm floor + thick ribs)
        translate([27.5, 0, 4.5])
            rbox_xy([13.0, 26.0, 4.0], r=3.0, center=true);
        translate([-21.5, 0, 4.5])
            rbox_xy([11.0, 26.0, 4.0], r=3.0, center=true);
        translate([0.0, 0, 4.8])
            rbox_xy([18.0, 24.0, 3.6], r=3.0, center=true);

        // Front tower: MG90S horn spline clearance + recessed horn slot
        translate([12.5, 0, Z_ANKLE_ROLL])
            rotate([0, 90, 0])
                cylinder(d=6.2, h=8.0, center=true);
        translate([13.5, 0, Z_ANKLE_ROLL])
            rotate([0, 90, 0])
                hull() {
                    cylinder(d=7.2, h=2.2, center=true);
                    translate([0, 11.0, 0]) cylinder(d=4.4, h=2.2, center=true);
                }

        // Rear tower: Opposing coaxial dual-shear pivot bore
        translate([-12.5, 0, Z_ANKLE_ROLL])
            rotate([0, 90, 0])
                cylinder(d=PIVOT_BOSS_D + CLEARANCE, h=8.0, center=true);
    }
}

foot_sole();
