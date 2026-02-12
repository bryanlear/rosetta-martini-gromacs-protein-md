# Rosetta for Protein Simulations

This directory contains tools and examples for protein structure analysis and design using Rosetta/PyRosetta.

## Overview

Rosetta is a comprehensive software suite for protein structure prediction, design, and analysis. This workspace focuses on:

- **Fixed Backbone Design (FixBB)**: Optimize amino acid sequences on fixed backbone structures
- **Protein-protein docking**: Model protein interactions
- **Structure refinement**: Improve model quality
- **Mutation analysis**: Assess structural impact of variants

## Slow Start

### 1. Install PyRosetta

Follow the instructions in [INSTALLATION.md](INSTALLATION.md). With student email:

```bash
# Get license from https://www.pyrosetta.org/downloads
# Then install:
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

### 2. Test Installation

```bash
python test_pyrosetta.py
```

### 3. Run FixBB Design Example

```bash
# Basic usage
python fixbb_example.py --pdb your_structure.pdb

# Design specific positions with comparison
python fixbb_example.py --pdb your_structure.pdb --output designed.pdb --compare

# Advanced: design only certain residues
python fixbb_example.py --pdb your_structure.pdb --positions "10-20,45,50-55" --iterations 5
```

## Files

- **INSTALLATION.md**: Detailed installation instructions
- **test_pyrosetta.py**: Simple test script to verify installation
- **fixbb_example.py**: Fixed backbone design example
- **mutation_analysis.py**: Analyze structural impact of mutations
- **fbn1_example.py**: Example using FBN1 structure from AlphaFold
- **requirements.txt**: Python dependencies

## 🧬 Use Cases

### Fixed Backbone Design (FixBB)

Design optimal amino acid sequences for a given backbone structure:

```python
import pyrosetta
from fixbb_example import run_fixbb_design

pyrosetta.init()
designed_pose, score = run_fixbb_design(
    input_pdb="structure.pdb",
    output_pdb="designed.pdb",
    num_iterations=5
)
```

### Mutation Analysis

Assess the structural impact of point mutations:

```python
from mutation_analysis import analyze_mutation

# Analyze a specific mutation
result = analyze_mutation(
    pdb_file="wildtype.pdb",
    chain="A",
    residue=156,
    mutation_to="L"
)

print(f"ΔΔG: {result['ddg']:.2f} kcal/mol")
print(f"Stability: {result['interpretation']}")
```

### Working with AlphaFold Structures

```python
# Example: Load and analyze FBN1 structure
python fbn1_example.py --residue 2220
```

## Common Rosetta Protocols

### Energy Minimization

```python
import pyrosetta
from pyrosetta import rosetta

pose = pyrosetta.pose_from_pdb("structure.pdb")
scorefxn = pyrosetta.create_score_function("ref2015")

# Setup minimizer
min_mover = rosetta.protocols.minimization_packing.MinMover()
min_mover.score_function(scorefxn)

# Minimize
min_mover.apply(pose)
pose.dump_pdb("minimized.pdb")
```

### Relax (Local Refinement)

```python
relax = rosetta.protocols.relax.FastRelax()
relax.set_scorefxn(scorefxn)
relax.apply(pose)
```

### Packing (Optimize Side Chains)

```python
packer = rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
packer.apply(pose)
```

## Scoring Functions

Rosetta uses scoring functions to evaluate protein structures:

- **ref2015**: Default all-atom scoring function (recommended)
- **beta_nov16**: Beta version with improved parameters
- **talaris2014**: Older standard function

```python
# Create scoring function
scorefxn = pyrosetta.create_score_function("ref2015")

# Score a pose
energy = scorefxn(pose)
print(f"Total energy: {energy:.2f} REU")

# Get individual energy terms
scorefxn.show(pose)
```

## 🧪 Example Workflow: Variant Classification

Assess if a missense mutation is destabilizing:

```bash
# 1. Get AlphaFold structure
wget https://alphafold.ebi.ac.uk/files/AF-P35555-F1-model_v4.pdb -O fbn1.pdb

# 2. Analyze mutation
python mutation_analysis.py --pdb fbn1.pdb --chain A --position 2220 --mutation A

# 3. Interpret results:
#    ΔΔG > +1.0: Likely destabilizing (pathogenic)
#    ΔΔG < -1.0: Likely stabilizing (benign)
#    -1.0 < ΔΔG < +1.0: Neutral
```

## Resources

### Documentation
- [PyRosetta Website](https://www.pyrosetta.org/)
- [Rosetta Commons](https://www.rosettacommons.org/)
- [PyRosetta Notebooks](https://github.com/RosettaCommons/PyRosetta.notebooks)

### Tutorials
- [FixBB Protocol](https://www.rosettacommons.org/docs/latest/application_documentation/design/fixbb)
- [ddG Prediction](https://www.rosettacommons.org/docs/latest/application_documentation/analysis/ddg-monomer)
- [PyRosetta Tutorials](https://graylab.jhu.edu/PyRosetta.documentation/pyrosetta.rosetta.protocols.html)

### Papers
- Alford et al. (2017). "The Rosetta All-Atom Energy Function for Macromolecular Modeling and Design." *J Chem Theory Comput* 13(6):3031-3048.
- Leaver-Fay et al. (2011). "ROSETTA3: an object-oriented software suite for the simulation and design of macromolecules." *Methods Enzymol* 487:545-574.

## ⚠️ Important Notes

1. **Academic License Required**: PyRosetta requires a free academic license
2. **Computational Cost**: Some protocols are CPU-intensive
3. **Memory Usage**: Large proteins may require significant RAM
4. **Energy Units**: Rosetta Energy Units (REU) are relative, not kcal/mol
5. **Statistical Significance**: Run multiple trajectories for robust results

## 🍆 Troubleshooting

### PyRosetta won't import
```bash
# Check Python version (3.8-3.11 supported)
python --version

# Reinstall
pip uninstall pyrosetta
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

### Segmentation fault on init
```python
# Use quiet mode
import pyrosetta
pyrosetta.init(extra_options="-mute all -ex1 -ex2aro")
```

### Out of memory
```python
# Limit pose size or use lower resolution
pyrosetta.init(extra_options="-in:file:centroid")
```

## 💩 License

PyRosetta is free for academic use. See https://www.pyrosetta.org/home/licensing for details.
