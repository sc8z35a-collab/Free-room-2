#!/usr/bin/env python3
"""Compile all parametric OpenSCAD parts of RNV1 into watertight 3D-printable STL meshes
and validate manifoldness, build volume, and printed PETG/PLA+ mass budget.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import trimesh

ROOT = Path(__file__).resolve().parent
PARTS_DIR = ROOT / "parts"
STL_DIR = ROOT / "stl"
STL_DIR.mkdir(parents=True, exist_ok=True)

# Realistic FDM effective density for 0.4mm nozzle (2 perimeters = 0.8mm skin + 15% gyroid infill inside bosses/ribs)
# Solid PLA+/PETG = 1.24 g/cm^3 -> calibrated PrusaSlicer/OrcaSlicer effective bulk factor ~0.685 = 0.85 g/cm^3
FDM_EFFECTIVE_DENSITY_G_CM3 = 0.85

# Parts list: (scad_filename, stl_basename, qty_on_robot)
PARTS = [
    ("torso_chassis.scad",    "torso_chassis",    1),
    ("torso_back_cover.scad", "torso_back_cover", 1),
    ("head_shell.scad",       "head_shell",       1),
    ("head_visor.scad",       "head_visor",       1),
    ("neck_gimbal.scad",      "neck_gimbal",      1),
    ("hip_clevis.scad",       "hip_clevis_left",  1),
    ("hip_clevis_right.scad", "hip_clevis_right", 1),
    ("leg_link.scad",         "leg_link",         2),
    ("ankle_clevis.scad",     "ankle_clevis",     2),
    ("foot_sole.scad",        "foot_sole",        2),
    ("arm_link.scad",         "arm_link_left",    1),
    ("arm_link_right.scad",   "arm_link_right",   1),
]

def main():
    manifest = {"parts": [], "total_printed_mass_g": 0.0, "total_volume_cm3": 0.0, "all_watertight": True}
    for scad_name, stl_base, qty in PARTS:
        scad_path = PARTS_DIR / scad_name
        stl_path = STL_DIR / f"{stl_base}.stl"
        cmd = ["openscad", "-o", str(stl_path), str(scad_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not stl_path.exists():
            print(f"[FAIL] OpenSCAD failed for {scad_name}:\n{res.stderr}", file=sys.stderr)
            sys.exit(1)

        mesh = trimesh.load(str(stl_path), force="mesh")
        if not mesh.is_watertight:
            # Repair minor floating-point winding/stitching if needed, then re-verify
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_winding(mesh)
            mesh.export(str(stl_path))
            mesh = trimesh.load(str(stl_path), force="mesh")

        vol_cm3 = abs(float(mesh.volume)) / 1000.0
        mass_g = vol_cm3 * FDM_EFFECTIVE_DENSITY_G_CM3
        extents = [round(float(x), 2) for x in mesh.extents]
        fits_bed = all(e <= 200.0 for e in extents)
        wt = bool(mesh.is_watertight)
        if not wt or not fits_bed or vol_cm3 <= 0.01:
            manifest["all_watertight"] = False

        entry = {
            "name": stl_base,
            "stl_file": f"cad/stl/{stl_base}.stl",
            "qty": qty,
            "triangles": int(len(mesh.faces)),
            "watertight": wt,
            "fits_200mm_bed": fits_bed,
            "extents_mm": extents,
            "unit_volume_cm3": round(vol_cm3, 3),
            "unit_mass_g": round(mass_g, 2),
            "total_mass_g": round(mass_g * qty, 2),
        }
        manifest["parts"].append(entry)
        manifest["total_volume_cm3"] += vol_cm3 * qty
        manifest["total_printed_mass_g"] += mass_g * qty
        status = "PASS" if (wt and fits_bed) else "FAIL"
        print(f"[{status}] {stl_base:18s} | tris={entry['triangles']:5d} | wt={str(wt):5s} | "
              f"ext={extents} mm | {entry['unit_mass_g']:5.2f}g x{qty} = {entry['total_mass_g']:5.2f}g")

    manifest["total_volume_cm3"] = round(manifest["total_volume_cm3"], 2)
    manifest["total_printed_mass_g"] = round(manifest["total_printed_mass_g"], 2)
    out_json = STL_DIR / "cad_manifest.json"
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("-" * 80)
    print(f"Total 3D-printed mass: {manifest['total_printed_mass_g']} g | "
          f"All watertight: {manifest['all_watertight']}")
    if not manifest["all_watertight"]:
        sys.exit(2)

if __name__ == "__main__":
    main()
