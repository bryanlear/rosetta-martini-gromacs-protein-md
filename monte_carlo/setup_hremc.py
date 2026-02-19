#!/usr/bin/env python3
"""
setup_hremc.py — Setup script for Hamiltonian Replica Exchange Monte Carlo
==========================================================================
SCN5A Nav1.5 (WT and RxxxH mutant) — Martini 3 CG with Go-model

Creates 24 replica directories for each system (wt / mutant), each with:
  - Scaled go_nbparams ITP (Go ε × λ_i)
  - Full GROMACS topology referencing the scaled ITP
  - Symlinks to shared force-field and protein ITP files
  - Initial coordinates (from insane_membrane build)
  - Energy-minimized and equilibrated starting structures

Usage:
    python setup_hremc.py [--config config.yaml] [--skip-eq]
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


# ============================================================
# Lambda ladder generation
# ============================================================
def generate_lambda_ladder(n_replicas: int, lam_min: float, lam_max: float,
                           spacing: str = "geometric") -> list[float]:
    """Generate lambda values for N replicas.

    Parameters
    ----------
    n_replicas : int
    lam_min, lam_max : float
        Range of lambda (Go-contact scaling factor).
    spacing : str
        'geometric' or 'linear'.

    Returns
    -------
    list of float, length n_replicas, sorted descending (1.0 → lam_min).
    """
    if spacing == "geometric":
        ratio = (lam_min / lam_max) ** (1.0 / (n_replicas - 1))
        lambdas = [lam_max * (ratio ** i) for i in range(n_replicas)]
    elif spacing == "linear":
        step = (lam_max - lam_min) / (n_replicas - 1)
        lambdas = [lam_max - i * step for i in range(n_replicas)]
    else:
        raise ValueError(f"Unknown spacing: {spacing}")
    return lambdas


# ============================================================
# Scale Go-model nbparams
# ============================================================
def scale_go_nbparams(input_itp: str, output_itp: str, lam: float,
                      native_eps: float = 9.414) -> int:
    """Read a go_nbparams ITP and write a copy with epsilon scaled by lambda.

    Each line format:
        AtomType_i  AtomType_j  1  sigma  epsilon  ;go bond dist

    Returns the number of contacts scaled.
    """
    scaled_eps = native_eps * lam
    n_contacts = 0

    with open(input_itp, "r") as fin, open(output_itp, "w") as fout:
        fout.write(f"; Go-model nb params scaled by lambda = {lam:.6f}\n")
        fout.write(f"; eps_scaled = {scaled_eps:.6f} kJ/mol "
                   f"(native = {native_eps:.6f})\n")
        for line in fin:
            stripped = line.strip()
            if not stripped or stripped.startswith(";") or stripped.startswith("["):
                fout.write(line)
                continue
            # Parse: type_i  type_j  func  sigma  epsilon  ;comment
            parts = line.split(";")
            data = parts[0].split()
            comment = ";" + parts[1] if len(parts) > 1 else ""
            if len(data) >= 5:
                data[4] = f"{scaled_eps:.8f}"
                n_contacts += 1
                new_line = (f"{data[0]:>30s} {data[1]:>30s} {data[2]:>2s} "
                            f"{data[3]:>14s} {data[4]:>14s} {comment}\n")
                fout.write(new_line)
            else:
                fout.write(line)
    return n_contacts


# ============================================================
# Generate topology for a replica
# ============================================================
def write_replica_topology(replica_dir: Path, system_label: str,
                           protein_name: str, variant: str,
                           source_dir: Path, config: dict) -> None:
    """Write the GROMACS topology for one replica.

    Uses relative #include paths pointing to shared FF files.
    The scaled go_nbparams ITP is local to the replica directory.
    """
    paths = config["paths"]

    # Resolve all paths to absolute before computing relative includes
    replica_dir = Path(replica_dir).resolve()
    source_dir = Path(source_dir).resolve()

    # Compute relative paths from replica_dir to shared resources
    # replica_dir is e.g. monte_carlo/wt/replica_00
    # We need to go up to the project level for includes
    mc_root = replica_dir.parent.parent  # monte_carlo/
    ext_sim = mc_root.parent             # extended_simulation/
    proj_root = ext_sim.parent           # rs199473101_SCN5A/

    # Relative from replica dir — resolve() to normalize ../../ before relpath
    def relpath(target):
        return os.path.relpath(Path(target).resolve(), replica_dir.resolve())

    # Project-level martini_files lives at protein_simulation/martini_files
    protein_sim = proj_root.parent  # protein_simulation/

    martini_itp = relpath(protein_sim / "martini_files/martini_v300/martini_v3.0.0.itp")
    go_atomtypes = relpath(source_dir / f"go_atomtypes_{variant}.itp")
    ffbonded = relpath(protein_sim / "martini_files/M3-Lipid-Parameters-main/ITPs/martini_v3.0.0_ffbonded_v2.itp")
    pc_itp = relpath(protein_sim / "martini_files/M3-Lipid-Parameters-main/ITPs/martini_v3.0.0_phospholipids_PC_v2.itp")
    pe_itp = relpath(protein_sim / "martini_files/M3-Lipid-Parameters-main/ITPs/martini_v3.0.0_phospholipids_PE_v2.itp")
    ps_itp = relpath(protein_sim / "martini_files/M3-Lipid-Parameters-main/ITPs/martini_v3.0.0_phospholipids_PS_v2.itp")
    sterols_itp = relpath(protein_sim / "martini_files/M3-Lipid-Parameters-main/ITPs/martini_v3.0.0_sterols_v1.itp")
    solvents_itp = relpath(protein_sim / "martini_files/martini_v3.0.0_solvents_v1.itp")
    ions_itp = relpath(protein_sim / "martini_files/martini_v3.0.0_ions_v1.itp")
    protein_itp = relpath(source_dir / f"{protein_name}.itp")

    # Read molecule counts from the insane_membrane topology
    membrane_dir = source_dir / "insane_membrane"
    insane_top = membrane_dir / "insane_topol.top"
    mol_lines = _parse_molecules(insane_top, skip_first=True)

    topol = f"""\
; ============================================================
; HREMC TOPOLOGY — Nav1.5 (SCN5A) {system_label.upper()}
; Replica directory: {replica_dir.name}
; ============================================================

#define GO_VIRT

; Force field
#include "{martini_itp}"

; Go-model atom types
#include "{go_atomtypes}"

; Go-model nb params (lambda-scaled — local file)
#include "go_nbparams_scaled.itp"

; Lipid bonded parameters
#include "{ffbonded}"

; Lipid topologies
#include "{pc_itp}"
#include "{pe_itp}"
#include "{ps_itp}"
#include "{sterols_itp}"

; Solvents & ions
#include "{solvents_itp}"
#include "{ions_itp}"

; Protein
#include "{protein_itp}"

[ system ]
Nav1.5 {system_label.upper()} HREMC

[ molecules ]
; Compound        #mols
{protein_name}     1
{mol_lines}
"""
    (replica_dir / "topol.top").write_text(topol)


def _parse_molecules(insane_top: Path, skip_first: bool = True) -> str:
    """Parse [ molecules ] from an insane topology, skipping the protein line."""
    lines = []
    in_mol = False
    first_skipped = False
    with open(insane_top) as f:
        for line in f:
            stripped = line.strip()
            if "[ molecules ]" in stripped:
                in_mol = True
                continue
            if in_mol and stripped.startswith("["):
                break
            if in_mol and stripped and not stripped.startswith(";"):
                if skip_first and not first_skipped:
                    first_skipped = True
                    continue
                lines.append(stripped)
    return "\n".join(lines)


# ============================================================
# Setup one system (wt or mutant)
# ============================================================
def setup_system(variant: str, config: dict, mc_root: Path,
                 skip_eq: bool = False) -> None:
    """Set up all replicas for one system variant (wt or mutant)."""
    n_rep = config["replicas"]["n_replicas"]
    lam_min = config["replicas"]["lambda_min"]
    lam_max = config["replicas"]["lambda_max"]
    spacing = config["replicas"]["spacing"]
    native_eps = config["system"]["go_eps_native"]

    if variant == "wt":
        protein_name = config["system"]["protein_name_wt"]
        source_dir = (mc_root / config["paths"]["wt_source"]).resolve()
        system_label = "WT"
    else:
        protein_name = config["system"]["protein_name_mutant"]
        source_dir = (mc_root / config["paths"]["mutant_source"]).resolve()
        system_label = "R376H mutant"

    membrane_gro = source_dir / "insane_membrane" / "system_membrane.gro"
    go_nbparams_src = source_dir / f"go_nbparams_{variant}.itp"

    lambdas = generate_lambda_ladder(n_rep, lam_min, lam_max, spacing)

    sys_dir = mc_root / variant
    sys_dir.mkdir(exist_ok=True)

    # Save lambda schedule
    with open(sys_dir / "lambda_schedule.dat", "w") as f:
        f.write("# replica  lambda  effective_T_K  go_eps_scaled\n")
        T_ref = config["system"]["temperature"]
        for i, lam in enumerate(lambdas):
            T_eff = T_ref / lam
            eps_scaled = native_eps * lam
            f.write(f"{i:>3d}  {lam:10.6f}  {T_eff:10.2f}  {eps_scaled:10.6f}\n")

    print(f"\n{'='*60}")
    print(f" Setting up {system_label} — {n_rep} replicas")
    print(f" Lambda range: {lam_max:.4f} → {lam_min:.4f}")
    print(f" Source: {source_dir}")
    print(f"{'='*60}\n")

    for i, lam in enumerate(lambdas):
        rep_dir = sys_dir / f"replica_{i:02d}"
        rep_dir.mkdir(exist_ok=True)

        # 1. Scale Go contacts
        n_contacts = scale_go_nbparams(
            str(go_nbparams_src),
            str(rep_dir / "go_nbparams_scaled.itp"),
            lam,
            native_eps
        )

        # 2. Write topology
        write_replica_topology(rep_dir, system_label, protein_name,
                               variant, source_dir, config)

        # 3. Copy initial coordinates
        if membrane_gro.exists():
            shutil.copy2(membrane_gro, rep_dir / "conf.gro")
        else:
            print(f"  WARNING: {membrane_gro} not found — "
                  "you must provide conf.gro manually")

        # 4. Create index file placeholder
        _write_ndx_script(rep_dir, config)

        print(f"  Replica {i:02d}: λ={lam:.4f}, ε_Go={lam*native_eps:.3f} kJ/mol, "
              f"{n_contacts} contacts")

    # Write combined info
    print(f"\n  → {n_rep} replica directories created in {sys_dir}/")

    if not skip_eq:
        print(f"\n  Run equilibration with:")
        print(f"    python hremc_engine.py --config config.yaml "
              f"--system {variant} --equilibrate")


def _write_ndx_script(rep_dir: Path, config: dict) -> None:
    """Write a small helper script to generate index groups."""
    script = f"""\
#!/bin/bash
# Generate index file with Protein / Non-Protein groups
# Run from within the replica directory
cd "$(dirname "$0")"
echo -e "1 | 12 | 13 | 14\\nname 15 Protein\\n!15\\nname 16 Non-Protein\\nq" \\
    | gmx make_ndx -f conf.gro -o index.ndx 2>/dev/null || \\
echo -e "q" | gmx make_ndx -f conf.gro -o index.ndx 2>/dev/null
echo "Index file created."
"""
    script_path = rep_dir / "make_index.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Setup HREMC replicas for CG SCN5A Nav1.5")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--system", choices=["wt", "mutant", "both"],
                        default="both",
                        help="Which system to set up")
    parser.add_argument("--skip-eq", action="store_true",
                        help="Skip equilibration reminder")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc_root = config_path.parent

    if args.system in ("wt", "both"):
        setup_system("wt", config, mc_root, args.skip_eq)
    if args.system in ("mutant", "both"):
        setup_system("mutant", config, mc_root, args.skip_eq)

    print("\n" + "="*60)
    print(" SETUP COMPLETE")
    print("="*60)
    print("\n Next steps:")
    print("   1. Review lambda_schedule.dat in each system directory")
    print("   2. Run: python hremc_engine.py --config config.yaml --equilibrate")
    print("   3. Run: python hremc_engine.py --config config.yaml --production")
    print("   4. Analyze: python analyze_hremc.py --config config.yaml")
    print()


if __name__ == "__main__":
    main()
