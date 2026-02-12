# Rosetta Installation Guide

## PyRosetta Installation (Recommended for Python integration)

PyRosetta provides Python bindings to Rosetta's core functionality including fixbb for protein design.

### Option 1: Install via Academic License (Free with Student Email)

1. **Get Academic License**
   - Visit: https://www.pyrosetta.org/downloads

2. **Install using pip with credentials**
   ```bash
   # Activate your virtual environment first
   source ../.venv/bin/activate
   
   # Install PyRosetta (Python 3.8-3.11 supported)
   pip install pyrosetta-installer
   python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
   ```

3. **Alternative: Direct download and install**
   ```bash
   # After getting credentials from pyrosetta.org
   pip install \
     --index-url=https://USERNAME:PASSWORD@pypi.rosettacommons.org/simple/ \
     pyrosetta
   ```

### Option 2: Conda Installation (Easier, but may have version lag)

```bash
# Create a dedicated conda environment
conda create -n rosetta python=3.10
conda activate rosetta

# Install PyRosetta from conda-forge
conda install -c conda-forge pyrosetta
```

### Option 3: Full Rosetta Suite (C++ binaries)

For the complete Rosetta suite with all protocols:

1. **Register for academic license**
   - Visit: https://www.rosettacommons.org/software/license-and-download
   - Fill out the form with your academic email
   - Download the Rosetta source or binaries

2. **Extract and compile** (if using source)
   ```bash
   cd rosetta
   tar -xvzf rosetta_src_*.tar.gz
   cd main/source
   
   # Compile (takes 1-2 hours)
   ./scons.py -j8 mode=release bin
   
   # Binaries will be in: main/source/bin/
   ```

3. **Add to PATH**
   ```bash
   echo 'export ROSETTA_HOME=/Users/bry_lee/ft-eng-embd/rosetta/main' >> ~/.zshrc
   echo 'export PATH=$ROSETTA_HOME/source/bin:$PATH' >> ~/.zshrc
   source ~/.zshrc
   ```

## Verify Installation

```python
import pyrosetta
pyrosetta.init()
print("PyRosetta installed successfully!")
```

## Quick Test

```bash
cd /Users/bry_lee/ft-eng-embd/rosetta
python test_pyrosetta.py
```

## Resources

- **PyRosetta Documentation**: https://www.pyrosetta.org/documentation
- **Rosetta Tutorials**: https://www.rosettacommons.org/demos/latest/Home
- **FixBB Protocol**: https://www.rosettacommons.org/docs/latest/application_documentation/design/fixbb
- **Forum**: https://www.rosettacommons.org/forum
