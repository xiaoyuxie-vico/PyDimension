#!/usr/bin/env python3
"""Quick test script for PyDimension package installation."""

import sys
from pathlib import Path

def test_imports():
    """Test that all main classes can be imported."""
    print("Testing imports...")
    try:
        import pydimension
        print(f"✅ Package version: {pydimension.__version__}")
        
        from pydimension import (
            DataGenerator, DataGenerationConfig,
            DataPreprocessor, DataPreprocessingConfig,
            DimensionalAnalyzer, DimensionalAnalysisConfig,
            ConstraintFilterer, ConstraintFilteringConfig,
            OptimizationDiscoverer, OptimizationDiscoveryConfig
        )
        print("✅ All classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_files():
    """Test that config files are accessible."""
    print("\nTesting config files...")
    try:
        import pydimension
        config_dir = Path(pydimension.__file__).parent / 'configs'
        config_files = list(config_dir.glob('*.json'))
        if config_files:
            print(f"✅ Found {len(config_files)} config files")
            for cf in config_files:
                print(f"   - {cf.name}")
            return True
        else:
            print("❌ No config files found")
            return False
    except Exception as e:
        print(f"❌ Error checking config files: {e}")
        return False

def test_command_line_tools():
    """Test that command-line tools are available."""
    print("\nTesting command-line tools...")
    import subprocess
    tools = [
        'pydimension-generate',
        'pydimension-preprocess',
        'pydimension-analyze',
        'pydimension-filter',
        'pydimension-optimize'
    ]
    
    all_found = True
    for tool in tools:
        try:
            result = subprocess.run(
                [tool, '--help'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0 or 'usage' in result.stdout.decode().lower() or 'help' in result.stderr.decode().lower():
                print(f"✅ {tool} is available")
            else:
                print(f"⚠️ {tool} returned non-zero exit code")
                all_found = False
        except FileNotFoundError:
            print(f"❌ {tool} not found in PATH")
            all_found = False
        except Exception as e:
            print(f"⚠️ {tool} error: {e}")
            all_found = False
    
    return all_found

def main():
    """Run all tests."""
    print("=" * 60)
    print("PyDimension 2.0 Package Installation Test")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Config Files", test_config_files()))
    results.append(("Command-Line Tools", test_command_line_tools()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All tests passed! Package is correctly installed.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

