#!/usr/bin/env python3
"""
Test script to verify PyDimension 2.0 environment setup.

This script checks that all required packages are installed and can be imported.
"""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: {e}")
        return False

def main():
    """Run all environment tests."""
    print("=" * 70)
    print("PyDimension 2.0 Environment Test")
    print("=" * 70)
    print()
    
    # Core dependencies
    print("Core Dependencies:")
    print("-" * 70)
    core_packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sympy", "SymPy"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
    ]
    
    core_results = []
    for module, name in core_packages:
        core_results.append(test_import(module, name))
    
    print()
    print("Machine Learning Dependencies:")
    print("-" * 70)
    ml_packages = [
        ("sklearn", "scikit-learn"),
        ("torch", "PyTorch"),
    ]
    
    ml_results = []
    for module, name in ml_packages:
        ml_results.append(test_import(module, name))
    
    print()
    print("PyDimension Modules:")
    print("-" * 70)
    pydimension_modules = [
        ("pydimension.data_generation", "Data Generation"),
        ("pydimension.data_preprocessing", "Data Preprocessing"),
        ("pydimension.intrinsic_coordinate", "Intrinsic Coordinate"),
        ("pydimension.symmetry_discovery", "Symmetry Discovery"),
        ("legacy.pydimension_v2.dimensional_analysis", "Dimensional Analysis (legacy)"),
        ("legacy.pydimension_v2.constraint_filtering", "Constraint Filtering (legacy)"),
        ("legacy.pydimension_v2.optimization_discovery", "Optimization Discovery (legacy)"),
    ]
    
    module_results = []
    for module, name in pydimension_modules:
        module_results.append(test_import(module, name))
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    all_core = all(core_results)
    all_ml = all(ml_results)
    all_modules = all(module_results)
    
    if all_core:
        print("✅ All core dependencies installed")
    else:
        print("❌ Some core dependencies missing")
    
    if all_ml:
        print("✅ All ML dependencies installed")
    else:
        print("⚠️  Some ML dependencies missing (optimization_discovery may not work)")
    
    if all_modules:
        print("✅ All PyDimension modules can be imported")
    else:
        print("❌ Some PyDimension modules cannot be imported")
    
    print()
    
    if all_core and all_modules:
        print("🎉 Environment setup successful!")
        print()
        print("You can now run:")
        print("  python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot")
        return 0
    else:
        print("⚠️  Environment setup incomplete. Please install missing packages.")
        print()
        print("Install using:")
        print("  pip install -r requirements.txt")
        print()
        print("Or using conda:")
        print("  conda env create -f environment.yml")
        print("  conda activate pydimension2.0")
        return 1

if __name__ == "__main__":
    sys.exit(main())

