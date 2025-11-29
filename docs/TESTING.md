# Testing PyDimension

Quick guide for testing the package and code changes.

## Quick Test (After Code Changes)

**With editable install (`pip install -e .`), changes are immediate - no reinstall needed!**

```bash
# Run quick test script
python quick_test.py

# Or test manually
python -c "import pydimension; print('✅ OK')"
pydimension-generate --help
```

## Installation Verification

```bash
# Run full test suite
python test_package_installation.py
```

This verifies:
- ✅ All classes can be imported
- ✅ Config files are accessible
- ✅ Command-line tools are available

## Testing Code Changes

### Quick Workflow

1. **Make code changes** (any `.py` file in `pydimension/`)
2. **Test immediately** (no reinstall needed with editable install):
   ```bash
   python quick_test.py
   ```
3. **Test specific functionality**:
   ```bash
   python -c "from pydimension.data_generation import DataGenerator; print('✅ OK')"
   ```

### When Reinstall is Needed

Only if you changed:
- `setup.py` (package structure, entry points)
- Package structure (added/removed modules)
- Entry points (CLI tool definitions)

```bash
pip install -e . --force-reinstall
```

## Functional Tests

### Test Individual Modules

```bash
# Data generation
python -c "
from pydimension import DataGenerator, DataGenerationConfig
config = DataGenerationConfig.from_dict({
    'DATA_GENERATION': {'enabled': True, 'N': 7, 'M': 10, 'ndim': 1, 'poly_order': 1, 'coefficients': [2.0, 1.0], 'random_seed': 32},
    'OUTPUT': {'output_dir': 'test_output', 'data_dir': 'data', 'figures_dir': 'figures', 'results_dir': 'results'}
})
gen = DataGenerator(config)
gen.generate(verbose=False)
print('✅ Data generation works')
"
```

### Test Command-Line Tools

```bash
pydimension-generate --config pydimension/configs/config_synthetic.json
pydimension-preprocess --config pydimension/configs/config_synthetic.json
pydimension-analyze --config pydimension/configs/config_synthetic.json
pydimension-filter --config pydimension/configs/config_synthetic.json
pydimension-optimize --config pydimension/configs/config_synthetic.json --plot
```

### Test Complete Pipeline

```bash
python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot
```

## Testing from Different Locations

```bash
# Test that package works from anywhere
cd /tmp
python -c "import pydimension; print('✅ Package accessible')"
cd /Users/xie/projects/PyDimension
```

## Troubleshooting

**Changes not reflected?**
```bash
pip install -e . --force-reinstall --no-cache-dir
```

**Import errors?**
```bash
python -m py_compile pydimension/your_module.py  # Check syntax
pip install -e . --force-reinstall
```

**CLI not working?**
```bash
pip install -e . --force-reinstall  # Entry points need reinstall
```

## Test Scripts

- `quick_test.py` - Fast test after code changes (~5 seconds)
- `test_package_installation.py` - Full installation verification
- `test_environment.py` - Environment and dependency check

See the [Workflow section](../README.md#workflow) in README.md for module-specific test commands and examples.

## Next Steps

- [README.md](../README.md) - Project overview
- [USAGE.md](USAGE.md) - How to use the package
- [SETUP.md](SETUP.md) - Installation guide

