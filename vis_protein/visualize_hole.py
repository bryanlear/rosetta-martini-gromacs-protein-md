"""
PyMOL script to visualize HOLE pore analysis for SCN5A.
Highlights mutation site and the pore-forming domain (358-378).

Usage (from PyMOL command line):
    run visualize_hole.py

Or from terminal:
    pymol -r visualize_hole.py
"""
from pymol import cmd, stored
from pymol.cgo import (
    CYLINDER, SPHERE, COLOR, BEGIN, END,
    VERTEX, LINES, LINEWIDTH, ALPHA
)
import os, math

# -- Configuration --
WORK_DIR = "/protein_simulation/vis_protein"
os.chdir(WORK_DIR)

PDB_FILE = os.path.join(WORK_DIR, "scn5a_af.pdb")
SPH_FILE = os.path.join(WORK_DIR, "hole.sph")

# Residue of interest
MUTATION_RESI = 376
PORE_DOMAIN = (358, 378)  # Pore-forming intramembrane domain
HELIX_RANGE = (374, 385)  # Helix containing mutation site

# Radius cutoffs (Angstroms) for coloring
TIGHT = 1.15    # red: too narrow for water
MID   = 2.30    # green: fits water but not ions well
                 # blue: wide enough for hydrated ions
MAX_RADIUS = 5.0  # cap display radius to avoid giant balloons

# ---- Load protein ----
cmd.load(PDB_FILE, "scn5a")
cmd.hide("everything", "scn5a")
cmd.show("cartoon", "scn5a")
cmd.color("gray80", "scn5a")
cmd.set("cartoon_transparency", 0.55, "scn5a")

# ---- Highlight pore-forming domain (358-378) ----
pore_sel = f"scn5a and resi {PORE_DOMAIN[0]}-{PORE_DOMAIN[1]}"
cmd.color("palecyan", pore_sel)
cmd.set("cartoon_transparency", 0.15, pore_sel)
cmd.show("sticks", pore_sel)
cmd.set("stick_transparency", 0.5, pore_sel)
cmd.color("palecyan", f"{pore_sel} and name C*")
cmd.color("red", f"{pore_sel} and name O*")
cmd.color("blue", f"{pore_sel} and name N*")

# ---- Highlight residue 376 (mutation site) prominently ----
mut_sel = f"scn5a and resi {MUTATION_RESI}"
cmd.show("sticks", mut_sel)
cmd.set("stick_radius", 0.25, mut_sel)
cmd.set("stick_transparency", 0.0, mut_sel)
cmd.color("magenta", f"{mut_sel} and name C*")
cmd.color("red", f"{mut_sel} and name O*")
cmd.color("blue", f"{mut_sel} and name N*")
# Add a transparent surface around residue 376
cmd.create("res376_surface", mut_sel)
cmd.show("surface", "res376_surface")
cmd.color("magenta", "res376_surface")
cmd.set("transparency", 0.65, "res376_surface")

# ---- Highlight the helix (374-385) containing the mutation ----
helix_sel = f"scn5a and resi {HELIX_RANGE[0]}-{HELIX_RANGE[1]}"
cmd.color("lightblue", helix_sel)
cmd.set("cartoon_transparency", 0.2, helix_sel)

# ---- Load HOLE spheres as a PDB (the .sph file IS PDB format) ----
cmd.load(SPH_FILE, "hole_spheres", format="pdb")

# Cap the VDW radius and apply
cmd.alter("hole_spheres", f"vdw = min(b, {MAX_RADIUS})")
cmd.rebuild("hole_spheres")

# Show as spheres with smaller scale for a cleaner look
cmd.hide("everything", "hole_spheres")
cmd.show("spheres", "hole_spheres")
cmd.set("sphere_scale", 1.0, "hole_spheres")
cmd.set("sphere_transparency", 0.45, "hole_spheres")

# ---- Color using spectrum for a smooth gradient ----
cmd.spectrum("b", "red_white_blue", "hole_spheres", minimum=0, maximum=5)

# ---- Build a smooth CGO pore surface (cylinder segments) ----
stored.data = []
cmd.iterate_state(1, "hole_spheres", "stored.data.append((x, y, z, b))")

if stored.data:
    cgo = []
    for i in range(len(stored.data) - 1):
        x1, y1, z1, r1 = stored.data[i]
        x2, y2, z2, r2 = stored.data[i + 1]
        # Cap radii for display
        r1 = min(r1, MAX_RADIUS)
        r2 = min(r2, MAX_RADIUS)
        # Color based on average radius
        ravg = (r1 + r2) / 2.0
        if ravg < TIGHT:
            cr, cg, cb = 1.0, 0.2, 0.2   # red
        elif ravg < MID:
            frac = (ravg - TIGHT) / (MID - TIGHT)
            cr, cg, cb = 1.0 - frac, 0.5 + 0.5 * frac, 0.2  # red->green
        else:
            frac = min((ravg - MID) / (MAX_RADIUS - MID), 1.0)
            cr, cg, cb = 0.2, 1.0 - 0.6 * frac, 0.3 + 0.7 * frac  # green->blue
        # Draw cylinder segment
        cgo.extend([
            CYLINDER, x1, y1, z1, x2, y2, z2,
            min(r1, r2) * 0.5,  # use smaller radius for smooth tube
            cr, cg, cb,  # color1
            cr, cg, cb,  # color2
        ])
    cmd.load_cgo(cgo, "pore_tube")
    cmd.set("cgo_transparency", 0.35, "pore_tube")

# ---- Draw centre-line ----
if stored.data:
    line_cgo = [LINEWIDTH, 2.0, BEGIN, LINES, COLOR, 1.0, 1.0, 0.0]
    for i in range(len(stored.data) - 1):
        x1, y1, z1, _ = stored.data[i]
        x2, y2, z2, _ = stored.data[i + 1]
        line_cgo.extend([VERTEX, x1, y1, z1, VERTEX, x2, y2, z2])
    line_cgo.extend([END])
    cmd.load_cgo(line_cgo, "pore_axis")

# ---- Optionally hide the raw spheres (pore_tube is cleaner) ----
cmd.disable("hole_spheres")  # toggle on with: enable hole_spheres

# ---- Nice view settings ----
cmd.bg_color("white")
cmd.set("ray_opaque_background", 0)
cmd.set("ray_shadow", "off")
cmd.set("antialias", 2)
cmd.set("depth_cue", 0)
cmd.set("spec_reflect", 0.3)

# Orient and zoom on the pore region near residue 376
cmd.orient(pore_sel)
cmd.zoom(pore_sel, 15)

print("\n=== HOLE + Mutation Site Visualization ===")
print(f"  Magenta sticks + surface: Residue {MUTATION_RESI} (mutation site)")
print(f"  Cyan sticks:  Pore-forming domain ({PORE_DOMAIN[0]}-{PORE_DOMAIN[1]})")
print(f"  Light blue:   Helix ({HELIX_RANGE[0]}-{HELIX_RANGE[1]})")
print(f"  Pore tube:    red (< {TIGHT} A) -> green -> blue (> {MID} A)")
print(f"  Max display radius capped at {MAX_RADIUS} A")
print("  Objects: scn5a, pore_tube, pore_axis, res376_surface")
print("  Toggle raw spheres: enable hole_spheres")
print("==========================================\n")
