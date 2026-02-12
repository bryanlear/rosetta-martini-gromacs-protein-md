#!/usr/bin/env python3
"""
Quick test to verify PyRosetta installation.
"""

def test_pyrosetta():
    """Test basic PyRosetta functionality."""
    try:
        import pyrosetta
        print("✓ PyRosetta import successful")
        
        # Initialize PyRosetta
        pyrosetta.init(extra_options="-mute all")
        print("✓ PyRosetta initialized successfully")
        
        # Test basic pose creation
        from pyrosetta import pose_from_sequence
        test_pose = pose_from_sequence("TESTSEQUENCE")
        print(f"✓ Created test pose with {test_pose.total_residue()} residues")
        
        print("\n✅ PyRosetta is working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ PyRosetta not installed: {e}")
        print("\nPlease follow INSTALLATION.md to install PyRosetta")
        return False
    except Exception as e:
        print(f"❌ Error testing PyRosetta: {e}")
        return False


if __name__ == "__main__":
    test_pyrosetta()
