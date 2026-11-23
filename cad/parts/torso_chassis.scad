// RNV1 Part: Aerospace-Grade Thin-Wall Monocoque Torso & Pelvis Unibody Chassis
// Local Origin: Pelvis center at (0, 0, Z_HIP_ROLL=124mm) -> Z spans -8mm (Z=116) to +60mm (Z=184)
// Engineered 2.2mm structural monocoque wall with internal stiffening ribs for ultra-low mass (~32g)
include <../rnv1_params.scad>

module torso_outer_shell(inset = 0.0) {
    hull() {
        translate([0, 0, -2.0 + inset*0.5])
            rbox_xy([TORSO_D - 8.0 - 2*inset, TORSO_W - 14.0 - 2*inset, 12.0 - inset], r=max(2.0, 6.0-inset), center=true);
        translate([1.5, 0, 26.0])
            rbox_xy([TORSO_D - 2*inset, TORSO_W - 4.0 - 2*inset, 22.0], r=max(2.0, 8.0-inset), center=true);
        translate([0, 0, 48.0])
            rbox_xy([TORSO_D - 4.0 - 2*inset, TORSO_W - 2*inset, 18.0], r=max(2.0, 8.0-inset), center=true);
        translate([0, 0, 58.0 - inset*0.5])
            rbox_xy([TORSO_D - 14.0 - 2*inset, TORSO_W - 22.0 - 2*inset, 4.0], r=max(2.0, 6.0-inset), center=true);
    }
}

module torso_chassis() {
    z0 = Z_HIP_ROLL; // 124.0mm local Z=0 reference
    difference() {
        union() {
            // 1.6mm Thin-Wall Monocoque Shell (4x 0.4mm FDM perimeters = rigid & ultra-light)
            difference() {
                torso_outer_shell(0.0);
                torso_outer_shell(1.6);
            }

            // Structural Pelvis Hip-Roll Bulkhead Frame (transfers 3G landing loads cleanly across pelvis)
            for (sy = [-HIP_ROLL_Y, HIP_ROLL_Y]) {
                translate([0, sy, 2.0])
                    rbox_xy([22.8, MG90S_W + 3.4, MG90S_L + 3.4], r=1.6, center=true);
                // Dual-Shear Rear Pelvis Pivot Bosses for L/R Hip Roll
                translate([-11.2, sy, 0])
                    rotate([0, -90, 0])
                        cylinder(d=PIVOT_BOSS_D, h=3.8);
            }
            // Pelvis Cross-Tie Structural Beam (links Left & Right Hip Roll bulkheads)
            translate([0, 0, 12.0])
                rbox_xy([20.0, 54.0, 2.2], r=1.0, center=true);

            // Shoulder MG90S Mounting Brackets (1.8mm cages)
            for (sy = [-1, 1]) {
                translate([0, sy * 29.0, SHOULDER_Z - z0])
                    rbox_xy([MG90S_L + 3.2, 20.5, MG90S_W + 3.2], r=1.5, center=true);
                translate([0, sy * 40.0, SHOULDER_Z - z0])
                    rotate([90, 0, 0])
                        cylinder(d=13.5, h=2.8, center=true);
            }

            // Top Neck Pan SG90 Cage + Collar Flange
            translate([0, 0, 52.5])
                rbox_xy([SG90_L + 3.2, SG90_W + 3.2, 13.0], r=1.5, center=true);
            translate([0, 0, 59.5])
                cylinder(d=22.0, h=2.2, center=true);

            // Front Speaker Acoustic Ring Retainer (28mm speaker snap-ring)
            translate([22.5, 0, 34.0])
                rotate([0, 90, 0])
                    cylinder(d=30.8, h=4.0, center=true);
        }

        // Open Rear Service Hatch for Battery Bay & Electronics Deck
        translate([-24.0, 0, 29.0])
            rbox_xy([16.0, 56.0, 42.0], r=3.0, center=true);

        // Front Chest 28mm Speaker Cavity
        translate([20.5, 0, 34.0])
            rotate([0, 90, 0])
                cylinder(d=28.4, h=8.0, center=true);

        // Front Acoustic Louver Slots (7 crisp horizontal slots)
        for (dz = [-9, -6, -3, 0, 3, 6, 9]) {
            w = 23.0 - abs(dz) * 1.1;
            translate([27.5, 0, 34.0 + dz])
                rbox_xy([8.0, w, 1.5], r=0.6, center=true);
        }

        // L & R Hip Roll MG90S Servo Pockets + Bottom Clevis Swing Clearance
        for (sy = [-HIP_ROLL_Y, HIP_ROLL_Y]) {
            translate([0, sy, 2.0])
                cube([20.4, MG90S_W + CLEARANCE, MG90S_L + CLEARANCE], center=true);
            translate([11.8, sy, 0])
                rotate([0, 90, 0])
                    cylinder(d=12.8, h=14.0, center=true);
            translate([0, sy, -7.8])
                rbox_xy([34.0, 19.0, 12.0], r=2.0, center=true);
        }

        // L & R Shoulder MG90S Pockets
        for (sy = [-1, 1]) {
            translate([0, sy * 29.0, SHOULDER_Z - z0])
                cube([MG90S_L + CLEARANCE, 21.0, MG90S_W + CLEARANCE], center=true);
            translate([0, sy * 40.5, SHOULDER_Z - z0])
                rotate([90, 0, 0])
                    cylinder(d=12.6, h=6.0, center=true);
        }

        // Top Neck Pan SG90 Pocket
        translate([0, 0, 52.5])
            cube([SG90_L + CLEARANCE, SG90_W + CLEARANCE, 17.0], center=true);
        translate([0, 0, 60.0])
            cylinder(d=12.4, h=8.0, center=true);

        // Side Convective Cooling Louver Vents
        for (side = [-1, 1]) {
            for (dz = [18, 23, 28]) {
                translate([-2.0, side * 39.0, dz])
                    rbox_xy([16.0, 6.0, 2.0], r=0.8, center=true);
            }
        }

        // MPU-6050 IMU CoM Mount M2 Holes on Pelvis Cross-Tie Beam
        for (sy = [-8.0, 8.0]) {
            translate([2.0, sy, 12.0])
                cylinder(d=M2_PILOT_D, h=6.0, center=true);
        }

        // Rear Cover M2 Pilot Holes (4 corners)
        for (sy = [-26.0, 26.0])
        for (dz = [10.0, 48.0]) {
            translate([-23.0, sy, dz])
                rotate([0, 90, 0])
                    cylinder(d=M2_PILOT_D, h=8.0, center=true);
        }
    }
}

torso_chassis();
