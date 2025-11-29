# Data Preprocessing Module

Preprocess datasets by selecting variables, normalizing data, and generating dimension matrices for dimensionless learning pipelines.

## Overview

The data preprocessing module prepares datasets for dimensional analysis by:
- Loading CSV datasets (from data generation or pre-prepared files)
- Selecting input and output variables
- Loading or generating dimension matrices
- Normalizing data (dividing by maximum values)
- Saving processed data for downstream modules

## Quick Start

### Using the Convenience Script (Recommended)

The easiest way to preprocess data is using the convenience script:

```bash
# Using the unified config
python preprocess_data.py --config pydimension/configs/config_synthetic.json

# Using command-line arguments directly
python preprocess_data.py --input_file dataset.csv

# With specific variables
python preprocess_data.py --input_file dataset.csv \
    --input_variables p1 p2 p3 p4 p5 p6 p7 \
    --output_variables "p*"

# With visualization
python preprocess_data.py --input_file dataset.csv --plot
```

### Using the Module Directly

You can also use the module directly:

```bash
# Using the unified config
python -m pydimension.data_preprocessing --config pydimension/configs/config_synthetic.json
```

### Using Command-Line Arguments

```bash
# Basic usage - auto-detect variables
python -m pydimension.data_preprocessing --input_file dataset.csv

# Specify input and output variables
python -m pydimension.data_preprocessing --input_file dataset.csv \
    --input_variables p1 p2 p3 p4 p5 p6 p7 \
    --output_variables "p*"

# Use dimension matrix from file
python -m pydimension.data_preprocessing --input_file dataset.csv \
    --dimension_matrix_file dimension_matrix.csv

# Disable normalization
python -m pydimension.data_preprocessing --input_file dataset.csv --no-normalize

# Generate with visualization
python -m pydimension.data_preprocessing --input_file dataset.csv --plot
```

### Using Python API

```python
from pydimension.data_preprocessing import DataPreprocessor, DataPreprocessingConfig

# Create config
config = DataPreprocessingConfig(
    input_file='dataset.csv',
    input_variables=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
    output_variables=['p*'],
    normalize=True
)

# Process data
preprocessor = DataPreprocessor(config)
results = preprocessor.process()

# Save results
normalized_path, matrix_path = preprocessor.save_results()
print(f"Normalized data: {normalized_path}")
print(f"Dimension matrix: {matrix_path}")

# Create visualization
plot_path = preprocessor.create_visualization()
print(f"Plot: {plot_path}")
```

## Visualization

The data preprocessing module can generate comprehensive visualization plots to help you understand your data.

### When to Use Visualization

✅ **Use `--plot` when:**
- **First-time preprocessing**: Verify your data preprocessing is working correctly
- **Quality assurance**: Check normalization and variable distributions
- **Data exploration**: Understand correlations and relationships between variables
- **Documentation**: Create plots for reports, papers, or presentations
- **Debugging**: Troubleshoot issues with variable selection or normalization

❌ **Skip `--plot` when:**
- **Batch processing**: Processing many datasets automatically (faster without plots)
- **Production runs**: When you're confident in your preprocessing pipeline
- **Large datasets**: Very large datasets where plotting might be slow

### What the Visualization Shows

The visualization creates a 2×3 grid of plots:

1. **Original vs Normalized Comparison** (top-left)
   - Scatter plot showing original values vs normalized values
   - Shows first 5 variables for clarity
   - Diagonal line (y=x) for reference

2. **Input Variables Distribution - Original** (top-middle)
   - Histograms of input variables before normalization
   - Shows first 3 input variables

3. **Input Variables Distribution - Normalized** (top-right)
   - Histograms of input variables after normalization
   - All values should be ≤ 1.0 if normalization is enabled
   - Shows first 3 input variables

4. **Correlation Matrix** (bottom-left)
   - Heatmap showing correlations between input variables
   - Color-coded from -1 (blue) to +1 (red)
   - Correlation values displayed as text

5. **Output Variable Distribution** (bottom-middle)
   - Histograms of output variable(s)
   - Shows both original and normalized distributions

6. **Summary Statistics** (bottom-right)
   - Text summary with:
     - Total number of variables (input/output breakdown)
     - Data shape
     - Normalization status
     - Normalized value range
     - List of selected variables

### Usage

```bash
# Using convenience script with visualization
python preprocess_data.py --input_file dataset.csv --plot

# Using module directly with visualization
python -m pydimension.data_preprocessing --input_file dataset.csv --plot

# Custom plot filename
python preprocess_data.py --input_file dataset.csv --plot --plot_filename my_plot.png
```

### Python API

```python
from pydimension.data_preprocessing import DataPreprocessor, DataPreprocessingConfig

config = DataPreprocessingConfig(
    input_file='dataset.csv',
    input_variables=['p1', 'p2', 'p3'],
    output_variables=['p*']
)

preprocessor = DataPreprocessor(config)
preprocessor.process()

# Create visualization
plot_path = preprocessor.create_visualization(
    filename='my_preprocessing_plot.png',
    show=False  # Set to True to display instead of saving
)
print(f"Plot saved to: {plot_path}")
```

**File location**: `output/figures/data_preprocessing_plots.png` (or custom filename)

## Configuration

### Config File Format

The unified config file contains a `DATA_PREPROCESSING` section:

```json
{
  "DATA_PREPROCESSING": {
    "enabled": true,
    "input_file": "output/data/dataset_synthetic.csv",
    "input_variables": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
    "output_variables": ["p*"],
    "dimension_matrix_file": null,
    "variable_units": null,
    "normalize": true
  },
  "OUTPUT": {
    "output_dir": "output",
    "data_dir": "data"
  },
  "DATA_PREPROCESSING_OUTPUT": {
    "normalized_data_filename": "normalized_data.csv",
    "dimension_matrix_filename": "dimension_matrix.csv",
    "plot_filename": "data_preprocessing_plots.png"
  }
}
```

### Parameter Descriptions

#### Input/Output Settings

- **input_file** (str, required): Path to input CSV file
  - Can be absolute or relative path
  - If `null`, will try to find default dataset from data generation module
  - Default locations checked: `output/data/dataset_synthetic.csv`, `dataset_synthetic.csv`

- **input_variables** (list of str, optional): List of input variable names
  - If `null`, will auto-detect variables (looks for p1-p7 pattern or all except outputs)
  - Example: `["p1", "p2", "p3", "p4", "p5", "p6", "p7"]`

- **output_variables** (list of str, optional): List of output variable names
  - If `null`, will auto-detect variables (looks for p*, e*, Ke, or variables ending with *)
  - Example: `["p*"]` or `["e*"]`

#### Dimension Matrix Settings

- **dimension_matrix_file** (str, optional): Path to dimension matrix CSV file
  - If provided, will load dimension matrix from this file
  - If `null`, will try default locations or generate from units
  - Default locations checked:
    - `output/data/dimension_matrix_synthetic.csv`
    - `output/data/dimension_matrix.csv`
    - `dimension_matrix_synthetic.csv`
    - `dimension_matrix.csv`

- **variable_units** (dict, optional): Dictionary mapping variable names to units
  - If provided, will use these units to generate dimension matrix
  - If `null`, will infer units from variable names
  - Example: `{"p1": "dimensionless", "Vs": "m/s", "rho": "kg/m³"}`

#### Normalization Settings

- **normalize** (bool): Whether to normalize data (divide by maximum)
  - Default: `true`
  - Normalized values will be ≤ 1.0
  - If `false`, original data values are preserved

#### Output Settings

- **output_dir** (str): Base output directory
  - Default: `"output"`
  - Files are saved to `output_dir/data_dir/`

- **normalized_data_filename** (str): Filename for normalized data
  - Default: `"normalized_data.csv"`

- **dimension_matrix_filename** (str): Filename for dimension matrix
  - Default: `"dimension_matrix.csv"`

## Variable Auto-Detection

If `input_variables` or `output_variables` are not specified, the module will auto-detect:

### Input Variables
- Looks for variables matching pattern `p1`, `p2`, `p3`, etc.
- If no pattern found, uses all variables except output candidates

### Output Variables
- Looks for variables: `p*`, `e*`, `Ke`
- Looks for variables ending with `*`
- If no pattern found, uses the last variable (excluding inputs)

### Metadata Columns
The following columns are automatically excluded:
- `case`
- `source`

## Dimension Matrix

The dimension matrix maps variables to their fundamental dimensions:
- **Mass** (M)
- **Length** (L)
- **Time** (T)
- **Temperature** (θ)
- **Current** (I)
- **Amount** (N)
- **Luminous** (J)

### Loading from File

If a dimension matrix file is provided, it should have:
- A `Dimension` or `Variable` column with dimension names
- Variable columns with integer exponents

Example:
```csv
Dimension,p1,p2,p3,p4,p5,p6,p7
Mass,0,0,0,0,1,0,0
Length,1,1,1,1,0,0,0
Time,0,-1,-1,-1,0,0,0
...
```

### Generating from Units

If no dimension matrix file is found, the module will:
1. Use `variable_units` from config if provided
2. Otherwise, infer units from variable names using common patterns
3. Parse unit strings to extract fundamental dimensions

Supported unit patterns:
- `W` (Watt) → [1, 2, -3, 0, 0, 0, 0]
- `m/s` (velocity) → [0, 1, -1, 0, 0, 0, 0]
- `kg/m³` (density) → [1, -3, 0, 0, 0, 0, 0]
- `J/(kg·K)` (specific heat) → [0, 2, -2, -1, 0, 0, 0]
- `dimensionless` → [0, 0, 0, 0, 0, 0, 0]

## Output Files

### normalized_data.csv

Contains the preprocessed dataset with:
- Selected input and output variables only
- Normalized values (if normalization enabled)
- Values ≤ 1.0 (if normalized)

Example:
```csv
p1,p2,p3,p4,p5,p6,p7,p*
0.623,0.789,0.654,0.712,0.801,0.567,0.698,0.523
...
```

### dimension_matrix.csv

Contains the dimension matrix with:
- `Dimension` column: dimension names
- Variable columns: integer exponents for each dimension

Example:
```csv
Dimension,p1,p2,p3,p4,p5,p6,p7,p*
Mass,0,0,0,0,1,0,0,0
Length,1,1,1,1,0,0,0,0
Time,0,-1,-1,-1,0,0,0,0
...
```

## Examples

### Example 1: Preprocess Synthetic Data

After generating data with the data generation module:

```bash
# Generate data first
python generate_data.py --config pydimension/configs/config_synthetic.json

# Preprocess the generated data
python -m pydimension.data_preprocessing \
    --input_file output/data/dataset_synthetic.csv \
    --dimension_matrix_file output/data/dimension_matrix_synthetic.csv
```

### Example 2: Preprocess Pre-prepared Dataset

```bash
python -m pydimension.data_preprocessing \
    --input_file dataset_keyhole.csv \
    --input_variables etaP Vs r0 alpha rho cp "Tv-T0" Lv "Tl-T0" Lm e Ke \
    --output_variables "e*"
```

### Example 3: Using Config File

Create or update `pydimension/configs/config_synthetic.json`:

```json
{
  "DATA_PREPROCESSING": {
    "enabled": true,
    "input_file": "dataset_keyhole.csv",
    "input_variables": ["etaP", "Vs", "r0", "alpha", "rho", "cp", "Tv-T0", "Lv", "Tl-T0", "Lm", "e", "Ke"],
    "output_variables": ["e*"],
    "normalize": true
  }
}
```

Then run:
```bash
python preprocess_data.py --config pydimension/configs/config_synthetic.json --plot
```

### Example 4: Python API with Custom Units

```python
from pydimension.data_preprocessing import DataPreprocessor, DataPreprocessingConfig

config = DataPreprocessingConfig(
    input_file='dataset_keyhole.csv',
    input_variables=['etaP', 'Vs', 'r0', 'alpha', 'rho', 'cp'],
    output_variables=['e*'],
    variable_units={
        'etaP': 'W',
        'Vs': 'm/s',
        'r0': 'm',
        'alpha': 'm²/s',
        'rho': 'kg/m³',
        'cp': 'J/(kg·K)',
        'e*': 'dimensionless'
    },
    normalize=True
)

preprocessor = DataPreprocessor(config)
results = preprocessor.process()
preprocessor.save_results()
```

## Command-Line Interface

### All Options

```bash
python -m pydimension.data_preprocessing \
    --config pydimension/configs/config_synthetic.json \              # Config file path
    --input_file dataset.csv \          # Input CSV file
    --input_variables p1 p2 p3 \       # Input variable names
    --output_variables "p*" \           # Output variable names
    --dimension_matrix_file dim.csv \  # Dimension matrix file
    --no-normalize \                    # Disable normalization
    --output_dir output \               # Output directory
    --normalized_data_filename norm.csv \ # Normalized data filename
    --dimension_matrix_filename dim.csv \ # Dimension matrix filename
    --plot \                            # Generate visualization plots
    --plot_filename plot.png \          # Plot filename
    --quiet                             # Suppress progress messages
```

## Python API

### DataPreprocessingConfig

Configuration class for data preprocessing parameters.

```python
from pydimension.data_preprocessing import DataPreprocessingConfig

# Create from dictionary
config = DataPreprocessingConfig.from_dict({
    'DATA_PREPROCESSING': {
        'input_file': 'dataset.csv',
        'input_variables': ['p1', 'p2', 'p3'],
        'output_variables': ['p*']
    }
})

# Create from JSON file
config = DataPreprocessingConfig.from_json('pydimension/configs/config_synthetic.json')

# Create directly
config = DataPreprocessingConfig(
    input_file='dataset.csv',
    input_variables=['p1', 'p2', 'p3'],
    output_variables=['p*'],
    normalize=True
)

# Validate configuration
errors = config.validate()
if errors:
    print("Configuration errors:", errors)

# Save to JSON (saved to configs directory)
config.to_json('pydimension/configs/my_config_synthetic.json')
```

### DataPreprocessor

Main preprocessor class for data preprocessing.

```python
from pydimension.data_preprocessing import DataPreprocessor, DataPreprocessingConfig

# Create preprocessor
config = DataPreprocessingConfig(
    input_file='dataset.csv',
    input_variables=['p1', 'p2', 'p3'],
    output_variables=['p*']
)
preprocessor = DataPreprocessor(config)

# Process data
results = preprocessor.process(verbose=True)

# Access results
print(f"Input variables: {results['input_variables']}")
print(f"Output variables: {results['output_variables']}")
print(f"Dimension matrix: {results['dimension_matrix']}")

# Access processed data
print(f"Normalized data shape: {preprocessor.normalized_data.shape}")
print(f"Dimension matrix: {preprocessor.dimension_matrix}")

# Save results
normalized_path, matrix_path = preprocessor.save_results()

# Create visualization
plot_path = preprocessor.create_visualization()
print(f"Visualization saved to: {plot_path}")
```

### Results Dictionary

The `process()` method returns a dictionary with:

```python
results = {
    'timestamp': '2025-01-01T12:00:00',
    'input_file': 'dataset.csv',
    'input_variables': ['p1', 'p2', 'p3', ...],
    'output_variables': ['p*'],
    'dimension_matrix': {'p1': [0, 1, 0, ...], ...},
    'variable_units': {'p1': 'dimensionless', ...},
    'normalized': True,
    'data_shape': (100, 8)
}
```

## Integration with Other Modules

### Workflow

1. **Data Generation** → Generates `dataset_synthetic.csv` and `dimension_matrix_synthetic.csv`
2. **Data Preprocessing** → Loads data, selects variables, normalizes, saves `normalized_data.csv` and `dimension_matrix.csv`
3. **Dimensional Analysis** → Uses normalized data and dimension matrix, saves `afterDA_data.csv` and `normalized_lg_afterDA_data.csv`
4. **Dimensional Filtering** → Uses normalized lg afterDA data, saves `suggested_dominant_count.json`
5. **Optimization and Discovery** → Uses normalized lg afterDA data and suggested count, discovers dimensionless scaling laws

### Typical Pipeline

```bash
# Step 1: Generate synthetic data
python generate_data.py --config pydimension/configs/config_synthetic.json

# Step 2: Preprocess the data (with visualization)
python preprocess_data.py \
    --input_file output/data/dataset_synthetic.csv \
    --dimension_matrix_file output/data/dimension_matrix_synthetic.csv \
    --plot

# Step 3: Dimensional analysis (normalized lg data saved by default)
python analyze_dimensions.py --config pydimension/configs/config_synthetic.json

# Step 4: Dimensional filtering (outputs suggested_dominant_count.json)
python filter_constraints.py --config pydimension/configs/config_synthetic.json

# Step 5: Optimization and discovery (automatically uses suggested count)
python optimize_discovery.py --config pydimension/configs/config_synthetic.json --plot
```

## See Also

- Main package README: `../../README.md`
- Data Generation Module: `../data_generation/README.md`
- Dimensional Analysis Module: `../dimensional_analysis/README.md`
- Dimensional Filtering Module: `../constraint_filtering/README.md`
- Optimization and Discovery Module: `../optimization_discovery/README.md`
- Config files: `../../configs/README.md`
- Example config: `../../configs/config_synthetic.json`

