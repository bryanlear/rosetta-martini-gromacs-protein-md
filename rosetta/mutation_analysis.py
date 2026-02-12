#!/usr/bin/env python3
"""
Mutation analysis script: Assess structural impact of point mutations.

Uses PyRosetta to calculate ΔΔG (change in stability) for mutations,
which can help predict pathogenicity of missense variants.

Uses LOCAL repacking (not FastRelax) so large structures (>1000 residues)
complete in seconds–minutes rather than hours.

Usage:
    python mutation_analysis.py --pdb protein.pdb --chain A --position 376 --mutation H
    python mutation_analysis.py --pdb protein.pdb --chain A --position 376 --mutation His
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional
import pyrosetta
from pyrosetta import rosetta


# ── Amino acid code conversion ──────────────────────────────────────────────
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}   # e.g. "H" → "HIS"
ONE_LETTER_AA = set(THREE_TO_ONE.values())


def normalize_aa(token: str) -> str:
    """Accept single-letter (H), three-letter (His/HIS), or full name and return single-letter."""
    t = token.strip().upper()
    if len(t) == 1 and t in ONE_LETTER_AA:
        return t
    if len(t) == 3 and t in THREE_TO_ONE:
        return THREE_TO_ONE[t]
    raise ValueError(
        f"Unrecognised amino acid '{token}'. "
        f"Use single-letter (e.g. H) or three-letter (e.g. His)."
    )


def aa1_to_rosetta(one_letter: str) -> str:
    """Convert single-letter AA to Rosetta residue type name (three-letter, e.g. 'HIS')."""
    return ONE_TO_THREE[one_letter]


def setup_pyrosetta(verbose=False):
    """Initialize PyRosetta with appropriate settings."""
    options = (
        "-ignore_unrecognized_res "
        "-ignore_zero_occupancy false "
        "-ex1 -ex2aro "                          # extra rotamers
        "-no_his_his_pairE "                      # avoid HIS-HIS pair energy crash
        "-detect_disulf false "                   # skip disulfide detection (AlphaFold)
        "-missing_density_to_jump "               # handle missing density gracefully
    )
    if not verbose:
        options += "-mute all "
    pyrosetta.init(extra_options=options)


def _make_local_task_factory(
    pose_position: int,
    radius: float = 8.0,
) -> rosetta.core.pack.task.TaskFactory:
    """Build a TaskFactory that only repacks residues within *radius* Å of *pose_position*."""
    tf = rosetta.core.pack.task.TaskFactory()

    # Keep current rotamers as an option
    tf.push_back(rosetta.core.pack.task.operation.IncludeCurrent())

    # Restrict everything outside neighbourhood to no repacking
    focus = rosetta.core.select.residue_selector.ResidueIndexSelector(pose_position)
    neighbourhood = rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    neighbourhood.set_focus_selector(focus)
    neighbourhood.set_distance(radius)
    neighbourhood.set_include_focus_in_subset(True)

    # Residues *outside* neighbourhood: prevent repacking entirely
    prevent = rosetta.core.pack.task.operation.OperateOnResidueSubset(
        rosetta.core.pack.task.operation.PreventRepackingRLT(),
        neighbourhood,
        flip_subset=True,         # apply to residues NOT in neighbourhood
    )
    tf.push_back(prevent)

    # Residues *inside* neighbourhood (except the mutated one): restrict to repacking
    restrict = rosetta.core.pack.task.operation.OperateOnResidueSubset(
        rosetta.core.pack.task.operation.RestrictToRepackingRLT(),
        neighbourhood,
        flip_subset=False,
    )
    tf.push_back(restrict)

    return tf


def calculate_ddg(
    pose: pyrosetta.Pose,
    chain: str,
    position: int,
    mutant_aa: str,
    scorefxn=None,
    repack_radius: float = 8.0,
    repack_rounds: int = 3,
) -> Dict:
    """
    Calculate ΔΔG for a point mutation using **local repacking** (fast).

    Protocol (based on Kellogg et al. 2011 "cartesian_ddg"-lite):
      1. Locally repack WT around the site → score
      2. Mutate residue → locally repack mutant → score
      3. ΔΔG = best_mutant – best_wt  (averaged over *repack_rounds*)

    Args:
        pose: PyRosetta Pose object
        chain: Chain identifier
        position: Residue position (PDB numbering)
        mutant_aa: Target amino acid (single letter code)
        scorefxn: Scoring function (default: ref2015)
        repack_radius: Å radius for local repacking (default 8)
        repack_rounds: Number of independent repack trajectories

    Returns:
        Dictionary with ddg, wildtype, mutant info
    """
    if scorefxn is None:
        scorefxn = pyrosetta.create_score_function("ref2015")

    # ── Resolve PDB → Rosetta numbering ────────────────────────────────────
    pose_position = pose.pdb_info().pdb2pose(chain, position)
    if pose_position == 0:
        raise ValueError(f"Position {chain}:{position} not found in structure")

    wildtype_aa = pose.residue(pose_position).name1()
    if wildtype_aa == mutant_aa:
        return {
            "wildtype": wildtype_aa, "mutant": mutant_aa,
            "position": position, "chain": chain,
            "ddg": 0.0, "wt_energy": 0.0, "mut_energy": 0.0,
            "interpretation": "Same amino acid (no mutation)",
        }

    print(f"  Mutation: {wildtype_aa}{position}{mutant_aa}")
    print(f"  Repack radius: {repack_radius} Å, rounds: {repack_rounds}")

    # ── Build local task factory ────────────────────────────────────────────
    tf = _make_local_task_factory(pose_position, repack_radius)

    best_wt = float("inf")
    best_mut = float("inf")

    for r in range(1, repack_rounds + 1):
        t0 = time.time()

        try:
            # WT repacking
            wt_pose = pose.clone()
            packer_wt = rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
            packer_wt.task_factory(tf)
            packer_wt.apply(wt_pose)
            wt_e = scorefxn(wt_pose)
            if wt_e < best_wt:
                best_wt = wt_e

            # Mutant: mutate + local repack
            mut_pose = pose.clone()
            rosetta.protocols.simple_moves.MutateResidue(pose_position, aa1_to_rosetta(mutant_aa)).apply(mut_pose)
            packer_mut = rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
            packer_mut.task_factory(tf)
            packer_mut.apply(mut_pose)
            mut_e = scorefxn(mut_pose)
            if mut_e < best_mut:
                best_mut = mut_e

            elapsed = time.time() - t0
            print(f"  Round {r}/{repack_rounds}: WT={wt_e:.1f}  MUT={mut_e:.1f}  ({elapsed:.1f}s)")
        except RuntimeError as exc:
            print(f"  Round {r}/{repack_rounds}: Rosetta error: {exc}")
            # Fallback: score without repacking
            if best_wt == float("inf"):
                best_wt = scorefxn(pose)
            if best_mut == float("inf"):
                mut_pose = pose.clone()
                rosetta.protocols.simple_moves.MutateResidue(pose_position, aa1_to_rosetta(mutant_aa)).apply(mut_pose)
                best_mut = scorefxn(mut_pose)
            break

    ddg = best_mut - best_wt

    if ddg > 1.0:
        interpretation = "Destabilizing (likely pathogenic)"
    elif ddg < -1.0:
        interpretation = "Stabilizing (likely benign)"
    else:
        interpretation = "Neutral"

    return {
        "wildtype": wildtype_aa, "mutant": mutant_aa,
        "position": position, "chain": chain,
        "ddg": ddg, "wt_energy": best_wt, "mut_energy": best_mut,
        "interpretation": interpretation,
    }


def analyze_mutation(
    pdb_file: Path,
    chain: str,
    residue: int,
    mutation_to: str,
    output_mutant: Optional[Path] = None,
    repack_radius: float = 8.0,
    repack_rounds: int = 3,
) -> Dict:
    """
    Analyze a single mutation.

    Args:
        pdb_file: Path to PDB file
        chain: Chain identifier
        residue: Residue position
        mutation_to: Target amino acid (single-letter code)
        output_mutant: Optional path to save mutant structure
        repack_radius: Å radius for local repacking
        repack_rounds: Independent repack trajectories

    Returns:
        Dictionary with analysis results
    """
    t0 = time.time()

    # Load structure
    pose = pyrosetta.pose_from_pdb(str(pdb_file))
    print(f"Loaded structure: {pose.total_residue()} residues")

    # Calculate ΔΔG
    result = calculate_ddg(
        pose, chain, residue, mutation_to,
        repack_radius=repack_radius,
        repack_rounds=repack_rounds,
    )

    # Save mutant structure if requested
    if output_mutant:
        pose_position = pose.pdb_info().pdb2pose(chain, residue)
        mutant_pose = pose.clone()
        rosetta.protocols.simple_moves.MutateResidue(
            pose_position, aa1_to_rosetta(mutation_to)
        ).apply(mutant_pose)
        mutant_pose.dump_pdb(str(output_mutant))
        print(f"Saved mutant structure to: {output_mutant}")

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")

    return result


def batch_analyze_mutations(pdb_file: Path, 
                            mutations_file: Path) -> list:
    """
    Analyze multiple mutations from a file.
    
    Format: chain position wildtype mutant (e.g., A 156 V L)
    
    Args:
        pdb_file: Path to PDB file
        mutations_file: Path to mutations list
    
    Returns:
        List of result dictionaries
    """
    
    pose = pyrosetta.pose_from_pdb(str(pdb_file))
    results = []
    
    with open(mutations_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            
            chain, position, wt, mut = parts
            position = int(position)
            
            print(f"\nAnalyzing {chain}:{wt}{position}{mut}...")
            
            try:
                result = calculate_ddg(pose, chain, position, mut)
                results.append(result)
                
                print(f"  ΔΔG: {result['ddg']:+.2f} REU")
                print(f"  {result['interpretation']}")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze structural impact of mutations"
    )
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--chain", required=True, help="Chain identifier")
    parser.add_argument("--position", type=int, required=True, 
                       help="Residue position")
    parser.add_argument("--mutation", required=True, 
                       help="Target amino acid: single-letter (H) or three-letter (His)")
    parser.add_argument("--output", help="Save mutant structure")
    parser.add_argument("--batch", help="Batch mode: mutations file")
    parser.add_argument("--radius", type=float, default=8.0,
                       help="Repack radius in Å (default: 8.0)")
    parser.add_argument("--rounds", type=int, default=3,
                       help="Repack rounds (default: 3)")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    pdb_file = Path(args.pdb)
    if not pdb_file.exists():
        print(f"Error: PDB file not found: {pdb_file}")
        return 1
    
    # Normalize amino acid input
    if not args.batch:
        try:
            args.mutation = normalize_aa(args.mutation)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        print(f"Target amino acid: {args.mutation}")
    
    # Setup PyRosetta
    setup_pyrosetta(verbose=args.verbose)
    
    if args.batch:
        # Batch mode
        mutations_file = Path(args.batch)
        results = batch_analyze_mutations(pdb_file, mutations_file)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for r in results:
            print(f"{r['chain']}:{r['wildtype']}{r['position']}{r['mutant']}: "
                  f"ΔΔG = {r['ddg']:+.2f} ({r['interpretation']})")
    else:
        # Single mutation mode
        output = Path(args.output) if args.output else None
        result = analyze_mutation(
            pdb_file, args.chain, args.position, args.mutation,
            output_mutant=output, repack_radius=args.radius,
            repack_rounds=args.rounds,
        )
        
        print("\n" + "="*60)
        print("MUTATION ANALYSIS RESULTS")
        print("="*60)
        print(f"Structure: {pdb_file}")
        print(f"Mutation: {result['chain']}:{result['wildtype']}{result['position']}{result['mutant']}")
        print(f"\nEnergies (Rosetta Energy Units):")
        print(f"  Wildtype:  {result['wt_energy']:.2f} REU")
        print(f"  Mutant:    {result['mut_energy']:.2f} REU")
        print(f"  ΔΔG:       {result['ddg']:+.2f} REU")
        print(f"\nInterpretation: {result['interpretation']}")
        print("\nGuidelines:")
        print("  ΔΔG > +1.0 REU: Destabilizing (possibly pathogenic)")
        print("  ΔΔG < -1.0 REU: Stabilizing (likely benign)")
        print("  -1.0 < ΔΔG < +1.0: Neutral effect")
    
    return 0


if __name__ == "__main__":
    exit(main())
