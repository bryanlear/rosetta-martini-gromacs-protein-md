# Rosetta Quick Reference

## Installation

```bash
# Step 1: Get academic license (actually no license is needed as long as it is used for research purposes and whatnot)
# Visit: https://www.pyrosetta.org/downloads

# Step 2: Install PyRosetta
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'

# Step 3: Test
python test_pyrosetta.py
```

## Common Commands

### Basic Structure Loading
```python
import pyrosetta
pyrosetta.init()
pose = pyrosetta.pose_from_pdb("structure.pdb")
```

### Energy Calculation
```python
scorefxn = pyrosetta.create_score_function("ref2015")
energy = scorefxn(pose)
```

### Mutation Analysis
```bash
python mutation_analysis.py --pdb structure.pdb --chain A --position 156 --mutation L
```

### Fixed Backbone Design
```bash
python fixbb_example.py --pdb structure.pdb --output designed.pdb --compare
```

### FBN1 Analysis
```bash
python fbn1_example.py --residue 2220
```

## Key Functions

### Create Scoring Function
```python
scorefxn = pyrosetta.create_score_function("ref2015")
```

### Mutate Residue
```python
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
mutator = MutateResidue(target_position, 'A')  # Mutate to Alanine
mutator.apply(pose)
```

### Energy Minimization
```python
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
min_mover = MinMover()
min_mover.score_function(scorefxn)
min_mover.apply(pose)
```

### Relax (Refinement)
```python
from pyrosetta.rosetta.protocols.relax import FastRelax
relax = FastRelax()
relax.set_scorefxn(scorefxn)
relax.apply(pose)
```

### Packing (Side Chain Optimization)
```python
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
packer = PackRotamersMover(scorefxn)
packer.apply(pose)
```

## Scoring Functions

| Name | Description | Use Case |
|------|-------------|----------|
| ref2015 | Default all-atom | General use (recommended) |
| beta_nov16 | Beta version | Testing new parameters |
| talaris2014 | Older standard | Legacy compatibility |

## Energy Interpretation

| ΔΔG Range | Interpretation | Likely Impact |
|-----------|----------------|---------------|
| > +2.0 | Highly destabilizing | Pathogenic |
| +1.0 to +2.0 | Destabilizing | Possibly pathogenic |
| -1.0 to +1.0 | Neutral | VUS |
| -1.0 to -2.0 | Stabilizing | Likely benign |
| < -2.0 | Highly stabilizing | Benign |

**Note:** These are guidelines. Multiple lines of evidence should be considered.

## Common Residue Selectors

```python
from pyrosetta.rosetta.core.select.residue_selector import *

# Select single residue
selector = ResidueIndexSelector(42)

# Select by chain
selector = ChainSelector("A")

# Select neighbors within distance
selector = NeighborhoodResidueSelector()
selector.set_distance(8.0)

# Select by secondary structure
selector = SecondaryStructureSelector("H")  # H=helix, E=strand, L=loop
```

## Troubleshooting

### ImportError: No module named pyrosetta
```bash
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

### Segmentation fault
```python
# Use quiet initialization
pyrosetta.init(extra_options="-mute all")
```

### Memory issues
```python
# Use centroid mode for large structures
pyrosetta.init(extra_options="-in:file:centroid")
```

### Ignore missing residues
```python
pyrosetta.init(extra_options="-ignore_unrecognized_res -ignore_zero_occupancy false")
```

## File Formats

### Input
- `.pdb` - Protein Data Bank format (most common)
- `.cif` - Crystallographic Information File
- `.silent` - Rosetta-specific format

### Output
```python
# Save PDB
pose.dump_pdb("output.pdb")

# Save score file
scorefxn.show(pose)  # Print to console
```

## Useful Links

- **PyRosetta Docs**: https://www.pyrosetta.org/documentation
- **Tutorials**: https://www.rosettacommons.org/demos/latest/Home
- **Forum**: https://www.rosettacommons.org/forum
- **API Reference**: https://graylab.jhu.edu/PyRosetta.documentation/

## Examples in This Directory

| Script | Purpose | Command |
|--------|---------|---------|
| test_pyrosetta.py | Test installation | `python test_pyrosetta.py` |
| fixbb_example.py | Fixed backbone design | `python fixbb_example.py --pdb input.pdb` |
| mutation_analysis.py | ΔΔG calculations | `python mutation_analysis.py --pdb input.pdb --chain A --position 156 --mutation L` |
| fbn1_example.py | FBN1-specific analysis | `python fbn1_example.py --residue 2220` |
