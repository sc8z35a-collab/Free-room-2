// RNV1 Part: Sculpted 88mm Monocoque I-Beam Leg Strut (Hip Pitch Z=106mm to Ankle Pitch Z=18mm)
// Local Origin: Ankle Pitch axis at (0, 0, 0) [Global Z=18mm], Hip Pitch axis at (0, 0, 88mm) [Global Z=106mm]
// Features: Athletic human-proportioned knee contour, I-beam weight-relief truss, dual-shear clevis forks at top
//           + lower Ankle Pitch MG90S servo pocket with coaxial opposing dual-shear pivot boss!
include <../rnv1_params.scad>

L_LEG_FULL = Z_HIP_PITCH - Z_ANKLE_ROLL; // 88.0 mm

module leg_link() {
    difference() {
        union() {
            // Lower Ankle Pitch MG90S Housing Block (shaft along +Y at Z=0)
            translate([0, 0, 6.0])
                rbox_xy([18.5, 19.6, 25.0], r=2.4, center=true);

            // Opposing coaxial dual-shear pivot boss at Y = -9.8mm -> extends to -13.2mm
            translate([0, -9.8, 0])
                rotate([90, 0, 0])
                    cylinder(d=PIVOT_BOSS_D, h=3.6);

            // Athletic Sculpted Leg Column with subtle forward knee contour (Z = 18 to 76mm)
            hull() {
                translate([0, 0, 18.0])
                    rbox_xy([17.5, 19.6, 4.0], r=2.4, center=true);
                translate([3.5, 0, 46.0])  // Subtle sculpted humanoid knee cap forward contour
                    rbox_xy([19.0, 21.0, 10.0], r=3.0, center=true);
            }
            hull() {
                translate([3.5, 0, 46.0])
                    rbox_xy([19.0, 21.0, 10.0], r=3.0, center=true);
                translate([0, 0, 76.0])
                    rbox_xy([16.5, 27.6, 4.0], r=2.4, center=true);
            }

            // Upper Dual-Shear Hip Pitch Clevis Forks (Y = +-12.2mm at Z = 88mm)
            for (sy = [-12.2, 12.2]) {
                hull() {
                    translate([0, sy, 76.0])
                        rbox_xy([15.5, 4.0, 4.0], r=1.2, center=true);
                    translate([0, sy, L_LEG_FULL])
                        rotate([90, 0, 0])
                            cylinder(d=13.5, h=4.0, center=true);
                }
            }
        }

        // Ankle Pitch MG90S Servo Cavity (Z = 6mm)
        translate([0, 0.4, 6.5])
            cube([MG90S_W + CLEARANCE, 18.8, 22.8 + CLEARANCE], center=true);
        translate([0, 9.8, 0])
            rotate([90, 0, 0])
                cylinder(d=12.6, h=5.0, center=true);

        // Two-bay I-Beam Truss Weight-Relief Pockets (leaves 3.0mm outer flanges + 2.2mm central web!)
        for (sx = [-8.5, 8.5]) {
            translate([sx + 1.5, 0, 32.0])
                rbox_xy([8.0, 12.5, 18.0], r=2.2, center=true);
            translate([sx + 1.5, 0, 60.0])
                rbox_xy([8.0, 13.0, 18.0], r=2.2, center=true);
        }

        // Upper Hip Pitch Dual-Shear Bores
        translate([0, 12.2, L_LEG_FULL])
            rotate([90, 0, 0])
                cylinder(d=6.2, h=8.0, center=true);
        translate([0, -12.2, L_LEG_FULL])
            rotate([90, 0, 0])
                cylinder(d=PIVOT_BOSS_D + CLEARANCE, h=8.0, center=true);

        // Internal rear servo cable routing channel
        translate([-6.8, 0, 46.0])
            cylinder(d=4.4, h=58.0, center=true);
    }
}

leg_link();
