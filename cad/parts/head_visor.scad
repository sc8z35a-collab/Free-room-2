// RNV1 Part: Smoked Optical Visor Bezel Plate (Snaps into head_shell front recess)
// Local Origin: Same as head_shell (Head Pitch axis at Z=0 [Global Z=204mm])
include <../rnv1_params.scad>

module head_visor() {
    difference() {
        // Curved sleek visor faceplate
        translate([HEAD_D/2 - 1.2, 0, 13.5])
            rbox([3.2, HEAD_W - 15.2, 24.8], r=5.4, center=true);

        // Trim back half flush
        translate([HEAD_D/2 - 4.0, 0, 13.5])
            cube([4.0, HEAD_W, 30.0], center=true);

        // Precision OV5640 / OV3660 Camera Lens Window
        translate([HEAD_D/2, 0, 15.0])
            rotate([0, 90, 0])
                cylinder(d=8.6, h=10.0, center=true);

        // Precision VL53L0X Laser ToF Emitter/Receiver Window
        translate([HEAD_D/2, 0, 5.0])
            rbox_xy([10.0, 8.2, 4.8], r=1.0, center=true);
    }
}

head_visor();
