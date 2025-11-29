# Using PyDimension

Quick reference for using the package.

## Installation

```bash
pip install -e .  # Editable install (recommended)
```

## Usage Methods

### 1. Command-Line Tools

```bash
# Generate data
pydimension-generate --config pydimension/configs/config_synthetic.json

# Preprocess
pydimension-preprocess --config pydimension/configs/config_synthetic.json

# Analyze dimensions
pydimension-analyze --config pydimension/configs/config_synthetic.json

# Filter constraints
pydimension-filter --config pydimension/configs/config_synthetic.json

# Optimize discovery
pydimension-optimize --config pydimension/configs/config_synthetic.json --plot
```

### 2. Python API

```python
from pydimension import DataGenerator, DataGenerationConfig

# Load config
config = DataGenerationConfig.from_json('pydimension/configs/config_synthetic.json')

# Use module
generator = DataGenerator(config)
generator.generate()
generator.save_datasets()
```

### 3. Convenience Scripts

```bash
python generate_data.py --config pydimension/configs/config_synthetic.json
python preprocess_data.py --config pydimension/configs/config_synthetic.json
python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot
```

### 4. Module Execution

```bash
python -m pydimension.data_generation --config pydimension/configs/config_synthetic.json
```

## Complete Pipeline

```bash
# Automated (recommended)
python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot

# Manual step-by-step
python generate_data.py --config pydimension/configs/config_synthetic.json --plot
python preprocess_data.py --config pydimension/configs/config_synthetic.json --plot
python analyze_dimensions.py --config pydimension/configs/config_synthetic.json --plot
python filter_constraints.py --config pydimension/configs/config_synthetic.json --plot
python optimize_discovery.py --config pydimension/configs/config_synthetic.json --plot

# For the keyhole problem
python run_pipeline.py --config pydimension/configs/config_keyhole.json --plot

python preprocess_data.py --config pydimension/configs/config_keyhole.json --plot
python analyze_dimensions.py --config pydimension/configs/config_keyhole.json --plot
python filter_constraints.py --config pydimension/configs/config_keyhole.json --plot
python optimize_discovery.py --config pydimension/configs/config_keyhole.json --plot
```

## Finding Config Files

```python
import pydimension
from pathlib import Path

config_dir = Path(pydimension.__file__).parent / 'configs'
print(f"Configs: {config_dir}")
```

## Examples

### Example: Generate and Preprocess

```python
from pydimension import DataGenerator, DataPreprocessor
from pydimension import DataGenerationConfig, DataPreprocessingConfig

# Generate
gen_config = DataGenerationConfig.from_json('pydimension/configs/config_synthetic.json')
generator = DataGenerator(gen_config)
generator.generate()
generator.save_datasets()

# Preprocess
prep_config = DataPreprocessingConfig.from_json('pydimension/configs/config_synthetic.json')
preprocessor = DataPreprocessor(prep_config)
preprocessor.process()
preprocessor.save_results()
```

## Next Steps

- [SETUP.md](SETUP.md) - Installation guide
- [TESTING.md](TESTING.md) - Testing guide
- [README.md](../README.md#workflow) - Workflow and command examples
- [README.md](../README.md) - Project overview
