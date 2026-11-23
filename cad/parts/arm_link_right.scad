// RNV1 Part: Right Arm Link (exact mirror of left arm link along Y)
use <arm_link.scad>
mirror([0, 1, 0]) arm_link();
