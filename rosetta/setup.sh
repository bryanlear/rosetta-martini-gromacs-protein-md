#!/bin/bash
# Quick setup script for Rosetta workspace

set -e

echo "=========================================="
echo "Rosetta/PyRosetta Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo ""
    echo "⚠️  Warning: Not in a virtual environment"
    echo "Recommended: activate your venv first"
    echo "  source ../.venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if PyRosetta is installed
echo ""
echo "Checking for PyRosetta..."
if python3 -c "import pyrosetta" 2>/dev/null; then
    echo "✓ PyRosetta is already installed"
    python3 -c "import pyrosetta; print(f'Version: {pyrosetta.__version__}')" 2>/dev/null || echo "Version check failed"
else
    echo "✗ PyRosetta not found"
    echo ""
    echo "PyRosetta installation options:"
    echo ""
    echo "Option 1: Use pyrosetta-installer (recommended)"
    echo "  pip install pyrosetta-installer"
    echo "  python3 -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'"
    echo ""
    echo "Option 2: Install with credentials"
    echo "  Get credentials from: https://www.pyrosetta.org/downloads"
    echo "  pip install --index-url=https://USERNAME:PASSWORD@pypi.rosettacommons.org/simple/ pyrosetta"
    echo ""
    echo "See INSTALLATION.md for detailed instructions"
    echo ""
    read -p "Install pyrosetta-installer now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing pyrosetta-installer..."
        pip install pyrosetta-installer
        echo ""
        echo "Now run this to install PyRosetta:"
        echo "  python3 -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'"
    fi
fi

# Install other requirements
echo ""
echo "Installing additional requirements..."
pip install numpy pandas matplotlib seaborn biopython

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Ensure PyRosetta is installed (see above)"
echo "2. Test installation: python3 test_pyrosetta.py"
echo "3. Try examples:"
echo "   - python3 fbn1_example.py --residue 2220"
echo "   - python3 fixbb_example.py --pdb structure.pdb"
echo ""
echo "Documentation:"
echo "  - See README.md for usage examples"
echo "  - See INSTALLATION.md for PyRosetta setup"
echo ""
