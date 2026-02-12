#!/usr/bin/env python3
"""
FBN1 (Fibrillin-1) analysis example using AlphaFold structure.

FBN1 mutations are associated with Marfan syndrome and related disorders.
This script demonstrates how to analyze specific residues in FBN1.

Usage:
    python fbn1_example.py --residue 2220
"""

import argparse
import urllib.request
from pathlib import Path
import pyrosetta
from pyrosetta import rosetta


def download_fbn1_structure(fragment=2, cache_dir="."):
    """
    Download FBN1 AlphaFold structure.
    
    Args:
        fragment: AlphaFold fragment number (1-5)
        cache_dir: Directory to cache downloaded structures
    
    Returns:
        Path to downloaded PDB file
    """
    # FBN1 UniProt: P35555
    url = f"https://alphafold.ebi.ac.uk/files/AF-P35555-F{fragment}-model_v4.pdb"
    filename = Path(cache_dir) / f"fbn1_fragment{fragment}.pdb"
    
    if filename.exists():
        print(f"Using cached structure: {filename}")
        return filename
    
    print(f"Downloading FBN1 fragment {fragment} from AlphaFold...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded to: {filename}")
        return filename
    except Exception as e:
        print(f"Error downloading: {e}")
        raise


def analyze_fbn1_residue(pdb_file: Path, residue: int, scan_mutations=True):
    """
    Analyze a specific residue in FBN1 structure.
    
    Args:
        pdb_file: Path to FBN1 PDB file
        residue: Residue number to analyze
        scan_mutations: If True, scan common pathogenic mutations
    """
    
    # Load structure
    pose = pyrosetta.pose_from_pdb(str(pdb_file))
    print(f"\nLoaded FBN1 structure: {pose.total_residue()} residues")
    
    # Get residue info
    chain = "A"  # AlphaFold structures use chain A
    try:
        pose_position = pose.pdb_info().pdb2pose(chain, residue)
    except:
        print(f"Warning: Using sequential numbering")
        pose_position = residue
    
    if pose_position == 0 or pose_position > pose.total_residue():
        print(f"Error: Residue {residue} not found")
        return
    
    res = pose.residue(pose_position)
    wildtype_aa = res.name1()
    
    print(f"\nResidue {residue}:")
    print(f"  Wildtype: {wildtype_aa} ({res.name3()})")
    print(f"  Rosetta position: {pose_position}")
    
    # Get structural context
    scorefxn = pyrosetta.create_score_function("ref2015")
    energy = scorefxn.residue_total_energy(pose, pose_position)
    print(f"  Energy: {energy:.2f} REU")
    
    # Check for secondary structure
    dssp = rosetta.protocols.moves.DsspMover()
    dssp.apply(pose)
    ss = pose.secstruct(pose_position)
    ss_name = {"H": "Helix", "E": "Strand", "L": "Loop"}.get(ss, "Unknown")
    print(f"  Secondary structure: {ss_name} ({ss})")
    
    # Count neighbors (burial)
    neighbor_count = 0
    for i in range(1, pose.total_residue() + 1):
        if i == pose_position:
            continue
        ca1 = pose.residue(pose_position).xyz("CA")
        ca2 = pose.residue(i).xyz("CA")
        distance = ca1.distance(ca2)
        if distance <= 10.0:  # 10Å cutoff
            neighbor_count += 1
    
    burial = "Buried" if neighbor_count > 15 else "Surface" if neighbor_count < 8 else "Intermediate"
    print(f"  Burial: {burial} ({neighbor_count} neighbors within 10Å)")
    
    # Scan mutations if requested
    if scan_mutations:
        print(f"\n{'='*60}")
        print(f"MUTATION SCAN FOR RESIDUE {residue}")
        print(f"{'='*60}")
        
        # Common pathogenic amino acids to test
        test_aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        results = []
        for mutant_aa in test_aa:
            if mutant_aa == wildtype_aa:
                continue
            
            try:
                # Quick ΔΔG estimate using repacking
                mutant_pose = pose.clone()
                mutator = rosetta.protocols.simple_moves.MutateResidue(
                    pose_position, mutant_aa
                )
                mutator.apply(mutant_pose)
                
                # Repack around mutation
                task_factory = rosetta.core.pack.task.TaskFactory()
                packer = rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
                packer.task_factory(task_factory)
                packer.apply(mutant_pose)
                
                mut_energy = scorefxn.residue_total_energy(mutant_pose, pose_position)
                ddg = mut_energy - energy
                
                results.append((mutant_aa, ddg))
                
            except Exception as e:
                print(f"  Error with {wildtype_aa}{residue}{mutant_aa}: {e}")
        
        # Sort by ΔΔG (most destabilizing first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nMutation predictions for {wildtype_aa}{residue}:")
        print(f"{'Mutation':<12} {'ΔΔG (REU)':<12} {'Prediction'}")
        print("-" * 50)
        
        for mutant_aa, ddg in results[:10]:  # Show top 10 most destabilizing
            mutation = f"{wildtype_aa}{residue}{mutant_aa}"
            prediction = "Destabilizing" if ddg > 1.0 else "Stabilizing" if ddg < -1.0 else "Neutral"
            print(f"{mutation:<12} {ddg:>+7.2f}        {prediction}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FBN1 structure and mutations"
    )
    parser.add_argument("--residue", type=int, default=2220,
                       help="Residue position to analyze (default: 2220)")
    parser.add_argument("--fragment", type=int, default=2,
                       help="AlphaFold fragment (1-5, default: 2)")
    parser.add_argument("--no-scan", action="store_true",
                       help="Skip mutation scanning")
    parser.add_argument("--pdb", help="Use local PDB file instead of downloading")
    parser.add_argument("--cache-dir", default=".",
                       help="Cache directory for downloads")
    
    args = parser.parse_args()
    
    # Setup PyRosetta
    print("Initializing PyRosetta...")
    pyrosetta.init(extra_options="-mute all -ignore_unrecognized_res")
    
    # Get structure
    if args.pdb:
        pdb_file = Path(args.pdb)
        if not pdb_file.exists():
            print(f"Error: PDB file not found: {pdb_file}")
            return 1
    else:
        try:
            pdb_file = download_fbn1_structure(args.fragment, args.cache_dir)
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    # Analyze
    analyze_fbn1_residue(
        pdb_file=pdb_file,
        residue=args.residue,
        scan_mutations=not args.no_scan
    )
    
    print("\n✅ Analysis complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
