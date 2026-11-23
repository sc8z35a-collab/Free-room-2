// RNV1 Part: Thin-Wall Ventilated Rear Service Cover (2.0mm shell, ~5.5g)
include <../rnv1_params.scad>

module torso_back_cover() {
    difference() {
        union() {
            translate([-25.2, 0, 29.0])
                rbox_xy([4.4, 60.0, 44.0], r=4.5, center=true);
            translate([-22.5, 0, 29.0])
                rbox_xy([2.2, 54.5, 40.5], r=2.8, center=true);
        }
        // Hollow thin-wall interior (leaves 2.0mm backplate)
        translate([-22.4, 0, 29.0])
            rbox_xy([4.2, 51.0, 37.0], r=2.2, center=true);

        // USB-C 2S Charging Port Cutout
        translate([-27.0, -14.0, 14.5])
            rbox_xy([8.0, 10.0, 4.5], r=1.5, center=true);

        // Main 6A Power Slide Switch Cutout
        translate([-27.0, 14.0, 14.5])
            rbox_xy([8.0, 12.0, 5.5], r=1.0, center=true);

        // Convective Cooling Louver Vents
        for (dz = [31, 36, 41]) {
            translate([-27.0, 0, dz])
                rbox_xy([8.0, 32.0, 2.2], r=0.8, center=true);
        }

        // 4x M2 Mounting Clearance Holes
        for (sy = [-26.0, 26.0])
        for (dz = [10.0, 48.0]) {
            translate([-25.0, sy, dz])
                rotate([0, 90, 0])
                    cylinder(d=M2_CLEAR_D, h=12.0, center=true);
        }
    }
}

torso_back_cover();
