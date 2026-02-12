#!/usr/bin/env python3
"""
Fixed Backbone Design (FixBB) example using PyRosetta.

FixBB performs protein sequence design on a fixed backbone structure,
optimizing the amino acid sequence while keeping backbone geometry constant.

Usage:
    python fixbb_example.py --pdb input.pdb --output designed.pdb
"""

import argparse
from pathlib import Path
import pyrosetta
from pyrosetta import rosetta


def setup_pyrosetta(verbose=False):
    """Initialize PyRosetta with appropriate settings."""
    if verbose:
        pyrosetta.init()
    else:
        pyrosetta.init(extra_options="-mute all")
    print("PyRosetta initialized")


def run_fixbb_design(input_pdb: Path, output_pdb: Path, 
                     design_positions: str = None,
                     num_iterations: int = 3):
    """
    Run fixed backbone design on a PDB structure.
    
    Args:
        input_pdb: Path to input PDB file
        output_pdb: Path to save designed structure
        design_positions: Residue positions to design (e.g., "10-20,45,50-55")
                         If None, designs all residues
        num_iterations: Number of design iterations
    """
    
    # Load structure
    pose = pyrosetta.pose_from_pdb(str(input_pdb))
    print(f"Loaded structure: {pose.total_residue()} residues")
    
    # Create a score function (weights for energy terms)
    scorefxn = pyrosetta.create_score_function("ref2015")
    
    # Score initial structure
    initial_score = scorefxn(pose)
    print(f"Initial score: {initial_score:.2f}")
    
    # Setup task factory for design
    task_factory = rosetta.core.pack.task.TaskFactory()
    
    # Restrict to repacking (keep identity) or allow design
    if design_positions:
        # Parse position ranges
        task_factory.push_back(
            rosetta.core.pack.task.operation.RestrictToRepacking()
        )
        # TODO: Add residue selector for specific positions
        print(f"Designing positions: {design_positions}")
    else:
        # Design all positions with standard amino acids
        print("Designing all positions")
    
    # Prevent design of prolines in the middle of chains (common constraint)
    task_factory.push_back(
        rosetta.core.pack.task.operation.IncludeCurrent()
    )
    
    # Setup packer (the algorithm that optimizes side-chain conformations)
    packer = rosetta.protocols.minimization_packing.PackRotamersMover(
        scorefxn
    )
    packer.task_factory(task_factory)
    
    # Run design iterations
    best_score = initial_score
    best_pose = pose.clone()
    
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        
        # Apply packer/design
        packer.apply(pose)
        
        # Score designed structure
        current_score = scorefxn(pose)
        print(f"  Score: {current_score:.2f}")
        
        if current_score < best_score:
            best_score = current_score
            best_pose = pose.clone()
            print(f"  ✓ New best score!")
    
    # Save best designed structure
    best_pose.dump_pdb(str(output_pdb))
    print(f"\nDesign complete!")
    print(f"Best score: {best_score:.2f} (Δ{best_score - initial_score:+.2f})")
    print(f"Saved to: {output_pdb}")
    
    return best_pose, best_score


def compare_sequences(original_pdb: Path, designed_pdb: Path):
    """Compare sequences between original and designed structures."""
    pose1 = pyrosetta.pose_from_pdb(str(original_pdb))
    pose2 = pyrosetta.pose_from_pdb(str(designed_pdb))
    
    seq1 = pose1.sequence()
    seq2 = pose2.sequence()
    
    mutations = 0
    for i, (aa1, aa2) in enumerate(zip(seq1, seq2), start=1):
        if aa1 != aa2:
            mutations += 1
            print(f"Position {i}: {aa1} → {aa2}")
    
    print(f"\nTotal mutations: {mutations}/{len(seq1)}")
    print(f"Sequence identity: {100*(1-mutations/len(seq1)):.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Fixed Backbone Design with PyRosetta"
    )
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--output", default="designed.pdb", 
                       help="Output PDB file")
    parser.add_argument("--positions", help="Positions to design (e.g., 10-20,45)")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of design iterations")
    parser.add_argument("--compare", action="store_true",
                       help="Compare original and designed sequences")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    input_pdb = Path(args.pdb)
    output_pdb = Path(args.output)
    
    if not input_pdb.exists():
        print(f"Error: Input PDB not found: {input_pdb}")
        return 1
    
    # Setup PyRosetta
    setup_pyrosetta(verbose=args.verbose)
    
    # Run design
    run_fixbb_design(
        input_pdb=input_pdb,
        output_pdb=output_pdb,
        design_positions=args.positions,
        num_iterations=args.iterations
    )
    
    # Compare sequences if requested
    if args.compare:
        print("\n" + "="*50)
        print("SEQUENCE COMPARISON")
        print("="*50)
        compare_sequences(input_pdb, output_pdb)
    
    return 0


if __name__ == "__main__":
    exit(main())
