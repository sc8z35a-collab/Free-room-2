// RNV1 Part: Universal 2-DOF Dual-Shear Ankle Clevis (Connects Ankle Pitch Y-axis to Ankle Roll X-axis!)
// Local Origin: Coaxial Ankle Center at (0, 0, 0) [Global Z=18mm]
// Houses Ankle Roll MG90S horizontally along X + dual-shear Ankle Pitch clevis ears along Y
include <../rnv1_params.scad>

module ankle_clevis() {
    difference() {
        union() {
            // Compact Ankle Roll MG90S Housing Block (shaft along +X at Z=0)
            translate([0, 0, 1.5])
                rbox_xy([20.0, 18.8, 22.0], r=2.2, center=true);

            // Rear Opposing Coaxial Ankle Roll Dual-Shear Pivot Boss (X = -10.0 -> -13.6mm)
            translate([-10.0, 0, 0])
                rotate([0, -90, 0])
                    cylinder(d=PIVOT_BOSS_D, h=3.6);

            // Outer Dual-Shear Ankle Pitch Clevis Arms (Y = +-12.2mm, connecting up to leg_link Ankle Pitch MG90S)
            for (sy = [-12.2, 12.2]) {
                hull() {
                    translate([0, sy, -2.0])
                        rbox_xy([13.0, 3.8, 8.0], r=1.2, center=true);
                    translate([0, sy, 0.0])
                        rotate([90, 0, 0])
                            cylinder(d=13.0, h=3.8, center=true);
                }
            }
        }

        // Ankle Roll MG90S Cavity (shaft along +X)
        translate([0.5, 0, 2.0])
            cube([19.2, MG90S_W + CLEARANCE, 19.6], center=true);
        translate([10.0, 0, 0])
            rotate([0, 90, 0])
                cylinder(d=12.4, h=5.0, center=true);

        // Ankle Pitch Dual-Shear Bores (+Y servo horn side, -Y opposing pivot boss)
        translate([0, 12.2, 0])
            rotate([90, 0, 0])
                cylinder(d=6.2, h=8.0, center=true);
        translate([0, -12.2, 0])
            rotate([90, 0, 0])
                cylinder(d=PIVOT_BOSS_D + CLEARANCE, h=8.0, center=true);
    }
}

ankle_clevis();
