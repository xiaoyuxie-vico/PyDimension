# Development Workflow - Checking Code Changes

Quick guide for checking and testing code changes during development.

## Prerequisites

```bash
# Install in editable mode (one time)
pip install -e .
```

With editable install, **code changes are immediately available** - no reinstall needed!

## Quick Check Workflow

### 1. Make Code Changes

Edit any file in `pydimension/` directory.

### 2. Quick Syntax Check

```bash
# Check Python syntax
python -m py_compile pydimension/your_module.py

# Or check entire package
python -c "import pydimension"
```

### 3. Quick Functionality Check

```bash
# Fast test (~5 seconds)
python quick_test.py
```

This verifies:
- ✅ Imports work
- ✅ Config files accessible  
- ✅ Command-line tools available

### 4. Test Specific Changes

```bash
# Test specific module
python -c "from pydimension.data_generation import DataGenerator; print('✅ OK')"

# Test command-line tool
pydimension-generate --help

# Test functionality
python -c "
from pydimension import DataGenerator, DataGenerationConfig
config = DataGenerationConfig.from_dict({
    'DATA_GENERATION': {'enabled': True, 'N': 7, 'M': 10, 'ndim': 1, 'poly_order': 1, 'coefficients': [2.0, 1.0], 'random_seed': 32},
    'OUTPUT': {'output_dir': 'test_output', 'data_dir': 'data', 'figures_dir': 'figures', 'results_dir': 'results'}
})
gen = DataGenerator(config)
gen.generate(verbose=False)
print('✅ Works')
"
```

## Development Checklist

### Before Committing

```bash
# 1. Syntax check
python -m py_compile pydimension/your_module.py

# 2. Import test
python -c "import pydimension; from pydimension import YourClass"

# 3. Quick test
python quick_test.py

# 4. Functional test (if changed core logic)
python run_pipeline.py --config pydimension/configs/config_synthetic.json
```

### After Code Changes

| Change Type | Check Command | Time |
|------------|--------------|------|
| Any code | `python quick_test.py` | ~5s |
| Import/export | `python -c "import pydimension"` | ~1s |
| CLI tool | `pydimension-generate --help` | ~1s |
| Functionality | `python run_pipeline.py --config ...` | ~30s |
| setup.py | `pip install -e . --force-reinstall` | ~10s |

## Common Checks

### Check Import Structure

```bash
# Test all main imports
python -c "
from pydimension import (
    DataGenerator, DataPreprocessor, DimensionalAnalyzer,
    ConstraintFilterer, OptimizationDiscoverer
)
print('✅ All imports OK')
"
```

### Check Command-Line Tools

```bash
# Test all CLI tools
pydimension-generate --help
pydimension-preprocess --help
pydimension-analyze --help
pydimension-filter --help
pydimension-optimize --help
```

### Check Package Structure

```bash
# Verify package location
python -c "import pydimension; print(pydimension.__file__)"

# Check config files
python -c "
import pydimension
from pathlib import Path
config_dir = Path(pydimension.__file__).parent / 'configs'
print(f'Configs: {list(config_dir.glob(\"*.json\"))}')
"
```

### Check from Different Directory

```bash
# Verify package works from anywhere
cd /tmp
python -c "import pydimension; print('✅ Accessible')"
cd /Users/xie/projects/PyDimension
```

## When to Reinstall

**Only reinstall if you changed:**
- `setup.py` (package structure, entry points)
- Package structure (added/removed modules)
- Entry points (CLI tool definitions)

```bash
pip install -e . --force-reinstall
```

**For all other changes:** Just test - no reinstall needed!

## Testing Workflow

### Minimal Test (Fastest)

```bash
python quick_test.py
```

### Full Test

```bash
python test_package_installation.py
```

### Functional Test

```bash
# Test complete pipeline
python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot
```

## Troubleshooting

### Changes Not Reflected?

```bash
# Check if editable install
pip show pydimension | grep Location

# Force reinstall if needed
pip install -e . --force-reinstall --no-cache-dir
```

### Import Errors?

```bash
# Check syntax first
python -m py_compile pydimension/your_file.py

# Then test import
python -c "import pydimension.your_module"
```

### CLI Not Working?

```bash
# Reinstall (entry points need reinstall)
pip install -e . --force-reinstall
pydimension-generate --help
```

## Quick Reference

```bash
# After ANY code change:
python quick_test.py

# After changing setup.py:
pip install -e . --force-reinstall
python quick_test.py

# Before committing:
python quick_test.py && python run_pipeline.py --config pydimension/configs/config_synthetic.json
```

## Best Practices

1. **Always use editable install**: `pip install -e .`
2. **Test immediately**: Run `python quick_test.py` after changes
3. **Test before commit**: Verify everything works
4. **Only reinstall when needed**: setup.py, structure, or entry points
5. **Use quick tests**: Don't run full pipeline for every small change

## See Also

- [TESTING.md](TESTING.md) - Detailed testing guide
- [SETUP.md](SETUP.md) - Installation and setup
- [USAGE.md](USAGE.md) - Usage examples
- [README.md](../README.md) - Project overview

