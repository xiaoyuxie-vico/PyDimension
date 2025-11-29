#!/usr/bin/env python3
"""Quick test script - run after making code changes."""

import sys
import time

def test_imports():
    """Quick import test."""
    try:
        import pydimension
        from pydimension import (
            DataGenerator, DataPreprocessor, DimensionalAnalyzer,
            ConstraintFilterer, OptimizationDiscoverer
        )
        return True, f"✅ Imports OK (v{pydimension.__version__})"
    except Exception as e:
        return False, f"❌ Import failed: {e}"

def test_command_line_tools():
    """Quick CLI test."""
    import subprocess
    try:
        result = subprocess.run(
            ['pydimension-generate', '--help'],
            capture_output=True,
            timeout=3
        )
        return True, "✅ CLI tools OK"
    except FileNotFoundError:
        return False, "❌ CLI tools not found (run: pip install -e .)"
    except Exception as e:
        return False, f"❌ CLI test failed: {e}"

def test_config_files():
    """Quick config files test."""
    try:
        import pydimension
        from pathlib import Path
        config_dir = Path(pydimension.__file__).parent / 'configs'
        config_files = list(config_dir.glob('*.json'))
        if config_files:
            return True, f"✅ Config files OK ({len(config_files)} found)"
        else:
            return False, "❌ No config files found"
    except Exception as e:
        return False, f"❌ Config test failed: {e}"

def main():
    print("🧪 Quick Test - Testing code changes...")
    print("-" * 50)
    
    start = time.time()
    
    # Test imports
    passed, msg = test_imports()
    print(msg)
    if not passed:
        print("\n❌ Quick test failed - fix imports first")
        return 1
    
    # Test CLI
    passed, msg = test_command_line_tools()
    print(msg)
    
    # Test config files
    passed, msg = test_config_files()
    print(msg)
    
    elapsed = time.time() - start
    print(f"\n✅ Quick tests passed in {elapsed:.2f}s")
    print("💡 For full tests, run: python test_package_installation.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())

