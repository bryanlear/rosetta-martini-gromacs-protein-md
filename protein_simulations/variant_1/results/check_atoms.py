#!/usr/bin/env python3
import MDAnalysis as mda

BASE = "/backmap_cg2at/backmap_charmm36/FINAL"
MT_GRO = f"{BASE}/production/first_frame.gro"
WT_GRO = f"{BASE}/production_for_WT/first_frame.gro"

mt = mda.Universe(MT_GRO)
wt = mda.Universe(WT_GRO)

print(f"MT total atoms: {mt.atoms.n_atoms}")
print(f"WT total atoms: {wt.atoms.n_atoms}")
print(f"MT Cα atoms: {mt.select_atoms('name CA').n_atoms}")
print(f"WT Cα atoms: {wt.select_atoms('name CA').n_atoms}")
print(f"MT backbone: {mt.select_atoms('backbone').n_atoms}")
print(f"WT backbone: {wt.select_atoms('backbone').n_atoms}")
print(f"MT protein residues: {mt.select_atoms('protein').residues.n_residues}")
print(f"WT protein residues: {wt.select_atoms('protein').residues.n_residues}")
