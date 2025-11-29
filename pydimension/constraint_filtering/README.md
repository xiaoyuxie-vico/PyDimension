# Dimensional Filtering Module

Identify dominant dimensionless groups using Principal Component Analysis (PCA) and Sliced Inverse Regression (SIR) analysis.

## Overview

The dimensional filtering module helps identify the most important dimensionless groups by:
- Running PCA on standardized data (inputs + output) to find principal components
- Running SIR analysis to find directions in input space that best predict the output
- Suggesting the number of dominant components/directions based on explained variance
- Providing visualizations to understand the relationships

## Quick Start

### Using the Convenience Script (Recommended)

The easiest way to perform dimensional filtering is using the convenience script:

```bash
# Using the unified config
python filter_constraints.py --config pydimension/configs/config_synthetic.json

# Using command-line arguments directly
python filter_constraints.py \
    --input_file output/data/normalized_lg_afterDA_data.csv

# With visualization
python filter_constraints.py --config config_synthetic.json --plot
```

### Using the Module Directly

You can also use the module directly:

```bash
# Using the unified config
python -m pydimension.constraint_filtering --config pydimension/configs/config_synthetic.json
```

### Using Command-Line Arguments

```bash
# Basic usage - auto-detect from default locations
python -m pydimension.constraint_filtering

# Specify input file directly
python -m pydimension.constraint_filtering \
    --input_file output/data/normalized_lg_afterDA_data.csv

# Run only PCA
python -m pydimension.constraint_filtering \
    --input_file normalized_lg_afterDA_data.csv \
    --no-sir

# Run only SIR
python -m pydimension.constraint_filtering \
    --input_file normalized_lg_afterDA_data.csv \
    --no-pca

# Custom SIR parameters
python -m pydimension.constraint_filtering \
    --input_file normalized_lg_afterDA_data.csv \
    --n_sir_slices 20 \
    --n_sir_directions 5

# With visualization
python -m pydimension.constraint_filtering --config config_synthetic.json --plot
```

### Using Python API

```python
from pydimension.constraint_filtering import ConstraintFilterer, ConstraintFilteringConfig

# Create config
config = ConstraintFilteringConfig(
    input_file='output/data/normalized_lg_afterDA_data.csv',
    run_pca=True,
    run_sir=True,
    n_sir_slices=10,
    n_sir_directions=3
)

# Perform analysis
filterer = ConstraintFilterer(config)
results = filterer.process()

# Save results
pca_path, sir_path = filterer.save_results()
print(f"PCA results: {pca_path}")
print(f"SIR results: {sir_path}")

# Create visualization
plot_path = filterer.create_visualization()
print(f"Plot: {plot_path}")
```

## Visualization

The dimensional filtering module can generate comprehensive visualization plots to help you understand the results.

### When to Use Visualization

✅ **Use `--plot` when:**
- **First-time analysis**: Verify your constraint filtering is working correctly
- **Understanding relationships**: See how SIR directions relate to output
- **Result verification**: Verify suggested number of components/directions
- **Documentation**: Create plots for reports, papers, or presentations
- **Debugging**: Troubleshoot issues with PCA or SIR analysis

❌ **Skip `--plot` when:**
- **Batch processing**: Processing many datasets automatically (faster without plots)
- **Production runs**: When you're confident in your analysis pipeline
- **Large datasets**: Very large datasets where plotting might be slow

### What the Visualization Shows

The visualization creates a grid of plots:

1. **PCA Eigenvalues** (if PCA enabled)
   - Bar chart showing eigenvalues for each principal component
   - Higher eigenvalues indicate more important components

2. **PCA Explained Variance** (if PCA enabled)
   - Bar chart showing individual explained variance
   - Line plot showing cumulative explained variance
   - 75% threshold line for reference
   - Suggested number of components

3. **SIR Eigenvalues** (if SIR enabled)
   - Bar chart showing eigenvalues for each SIR direction
   - Higher eigenvalues indicate more important directions

4. **SIR Explained Variance** (if SIR enabled)
   - Bar chart showing individual explained variance
   - Line plot showing cumulative explained variance
   - 75% threshold line for reference
   - Suggested number of directions (⭐ if 1D hidden variable detected)

5. **SIR Directions vs Output** (if SIR enabled)
   - Scatter plots for each SIR direction vs output
   - Color-coded by output value
   - Trend line showing relationship
   - If nonlinear pattern visible → SIR successfully found it!

### Usage

```bash
# Using convenience script with visualization
python filter_constraints.py --config config_synthetic.json --plot

# Using module directly with visualization
python -m pydimension.constraint_filtering --config config_synthetic.json --plot

# Custom plot filename
python filter_constraints.py --config config_synthetic.json --plot --plot_filename my_plot.png
```

### Python API

```python
from pydimension.constraint_filtering import ConstraintFilterer, ConstraintFilteringConfig

config = ConstraintFilteringConfig(
    input_file='output/data/normalized_lg_afterDA_data.csv',
    run_pca=True,
    run_sir=True
)

filterer = ConstraintFilterer(config)
filterer.process()

# Create visualization
plot_path = filterer.create_visualization(
    filename='my_filtering_plot.png',
    show=False  # Set to True to display instead of saving
)
print(f"Plot saved to: {plot_path}")
```

**File location**: `output/figures/constraint_filtering_plots.png` (or custom filename)

## How It Works

### Step 1: Load Data

The module loads:
- **Normalized lg afterDA data**: CSV file with lgπ variables and output
- Assumes last column is output, all others are inputs
- Automatically handles non-numeric values and NaN

### Step 2: PCA Analysis (Optional)

If `run_pca=True`:
- Standardizes data (mean=0, std=1) for all columns (inputs + output)
- Performs SVD decomposition
- Computes eigenvalues and explained variance ratio
- Suggests number of dominant components (75% cumulative variance threshold)

### Step 3: SIR Analysis (Optional)

If `run_sir=True`:
- Standardizes inputs (mean=0, std=1)
- Slices output into bins (configurable number of slices)
- Computes mean of inputs in each slice
- Analyzes variation of slice means to find important directions
- Projects data onto SIR directions
- Suggests number of important directions (75% cumulative variance threshold)

**Key advantage of SIR**: Works even when relationship is nonlinear (exp, log, etc.)
- If 1 direction dominates → you found your 1D hidden variable! ⭐

### Step 4: Save Results

Saves:
- **pca_results.json**: PCA eigenvalues, explained variance, suggested count
- **sir_results.json**: SIR eigenvalues, eigenvectors, explained variance, suggested directions

## Configuration

### Unified Config Format

The unified config file contains a `CONSTRAINT_FILTERING` section:

```json
{
  "CONSTRAINT_FILTERING": {
    "enabled": true,
    "input_file": "output/data/normalized_lg_afterDA_data.csv",
    "run_pca": true,
    "run_sir": true,
    "n_sir_slices": 10,
    "n_sir_directions": 3
  },
  "OUTPUT": {
    "output_dir": "output",
    "data_dir": "data",
    "figures_dir": "figures",
    "results_dir": "results"
  },
  "CONSTRAINT_FILTERING_OUTPUT": {
    "pca_results_filename": "pca_results.json",
    "sir_results_filename": "sir_results.json",
    "plot_filename": "constraint_filtering_plots.png"
  }
}
```

### Parameter Descriptions

#### Input Settings

- **input_file** (str, optional): Path to normalized lg afterDA data CSV file
  - Can be absolute or relative path
  - If `null`, will try to find default files from dimensional analysis module
  - Default locations checked:
    - `output/data/normalized_lg_afterDA_data.csv`
    - `normalized_lg_afterDA_data.csv`

#### Analysis Settings

- **run_pca** (bool): Whether to run PCA analysis
  - Default: `true`
  - PCA finds principal components in standardized data

- **run_sir** (bool): Whether to run SIR analysis
  - Default: `true`
  - SIR finds directions that best predict output (works for nonlinear relationships)

- **n_sir_slices** (int): Number of slices for SIR
  - Default: `10`
  - Must be at least 2
  - Output is divided into this many bins
  - Automatically adjusted if too large for dataset

- **n_sir_directions** (int): Number of SIR directions to compute
  - Default: `3`
  - Must be at least 1
  - Number of directions to extract and visualize

#### Output Settings

- **output_dir** (str): Base output directory
  - Default: `"output"`
  - Files are saved to `output_dir/results_dir/` and `output_dir/figures_dir/`

- **pca_results_filename** (str): Filename for PCA results
  - Default: `"pca_results.json"`

- **sir_results_filename** (str): Filename for SIR results
  - Default: `"sir_results.json"`

- **plot_filename** (str): Filename for visualization plot
  - Default: `"constraint_filtering_plots.png"`

## Output Files

### pca_results.json

Contains PCA analysis results:

```json
{
  "timestamp": "2025-01-01T12:00:00",
  "eigenvalues": [2.5, 1.2, 0.8, ...],
  "explained_variance_ratio": [0.45, 0.22, 0.15, ...],
  "cumulative_variance": [0.45, 0.67, 0.82, ...],
  "suggested_dominant_count": 3
}
```

### sir_results.json

Contains SIR analysis results:

```json
{
  "timestamp": "2025-01-01T12:00:00",
  "eigenvalues": [0.85, 0.12, 0.03, ...],
  "eigenvectors": [[0.5, 0.3, ...], [0.2, -0.4, ...], ...],
  "explained_variance": [0.85, 0.12, 0.03, ...],
  "cumulative_variance": [0.85, 0.97, 1.0, ...],
  "suggested_directions": 1,
  "n_slices": 10,
  "n_directions": 3
}
```

## Examples

### Example 1: Complete Pipeline

After dimensional analysis:

```bash
# Step 1: Generate data
python generate_data.py --config pydimension/configs/config_synthetic.json

# Step 2: Preprocess data
python preprocess_data.py --config pydimension/configs/config_synthetic.json

# Step 3: Dimensional analysis (with normalized lg data)
python analyze_dimensions.py --config pydimension/configs/config_synthetic.json --save-normalized-lg

# Step 4: Dimensional filtering (outputs suggested_dominant_count.json)
python filter_constraints.py --config pydimension/configs/config_synthetic.json --plot

# Step 5: Optimization and discovery (automatically uses suggested count)
python optimize_discovery.py --config pydimension/configs/config_synthetic.json --plot
```

### Example 2: Only PCA

```bash
python filter_constraints.py \
    --input_file output/data/normalized_lg_afterDA_data.csv \
    --no-sir \
    --plot
```

### Example 3: Only SIR

```bash
python filter_constraints.py \
    --input_file output/data/normalized_lg_afterDA_data.csv \
    --no-pca \
    --plot
```

### Example 4: Custom SIR Parameters

```bash
python filter_constraints.py \
    --input_file output/data/normalized_lg_afterDA_data.csv \
    --n_sir_slices 20 \
    --n_sir_directions 5 \
    --plot
```

## Python API

### ConstraintFilteringConfig

Configuration class for dimensional filtering:

```python
from pydimension.constraint_filtering import ConstraintFilteringConfig

# Create from dict
config_dict = {
    'CONSTRAINT_FILTERING': {
        'input_file': 'normalized_lg_afterDA_data.csv',
        'run_pca': True,
        'run_sir': True,
        'n_sir_slices': 10,
        'n_sir_directions': 3
    },
    'OUTPUT': {
        'output_dir': 'output',
        'results_dir': 'results',
        'figures_dir': 'figures'
    }
}
config = ConstraintFilteringConfig.from_dict(config_dict)

# Create from JSON
config = ConstraintFilteringConfig.from_json('config_synthetic.json')

# Create directly
config = ConstraintFilteringConfig(
    input_file='output/data/normalized_lg_afterDA_data.csv',
    run_pca=True,
    run_sir=True,
    n_sir_slices=10,
    n_sir_directions=3
)

# Validate
errors = config.validate()
if errors:
    print("Errors:", errors)
```

### ConstraintFilterer

Main filterer class:

```python
from pydimension.constraint_filtering import ConstraintFilterer, ConstraintFilteringConfig

# Create config
config = ConstraintFilteringConfig(
    input_file='output/data/normalized_lg_afterDA_data.csv',
    run_pca=True,
    run_sir=True,
    n_sir_slices=10,
    n_sir_directions=3
)

# Create filterer
filterer = ConstraintFilterer(config)

# Load data
data = filterer.load_data()

# Run PCA
if config.run_pca:
    pca_results = filterer.run_pca()

# Run SIR
if config.run_sir:
    sir_results = filterer.run_sir()

# Or run complete pipeline
results = filterer.process(verbose=True)

# Save results
pca_path, sir_path = filterer.save_results()

# Create visualization
plot_path = filterer.create_visualization()
```

### Results Dictionary

The `process()` method returns a dictionary with:

```python
results = {
    'timestamp': '2025-01-01T12:00:00',
    'input_file': 'output/data/normalized_lg_afterDA_data.csv',
    'input_columns': ['lgπ1', 'lgπ2', ...],
    'output_column': 'output',
    'data_shape': (100, 4),
    'pca': {
        'eigenvalues': [2.5, 1.2, ...],
        'explained_variance_ratio': [0.45, 0.22, ...],
        'cumulative_variance': [0.45, 0.67, ...],
        'suggested_dominant_count': 3,
        'n_samples': 100,
        'n_features': 4
    },
    'sir': {
        'eigenvalues': [0.85, 0.12, ...],
        'eigenvectors': [[0.5, 0.3, ...], ...],
        'explained_variance': [0.85, 0.12, ...],
        'cumulative_variance': [0.85, 0.97, ...],
        'suggested_directions': 1,
        'n_slices': 10,
        'n_samples': 100,
        'n_features': 3,
        'top_directions': [...]
    }
}
```

## Integration with Other Modules

### Workflow

1. **Data Generation** → Generates `dataset_synthetic.csv` and `dimension_matrix_synthetic.csv`
2. **Data Preprocessing** → Loads data, normalizes, saves `normalized_data.csv` and `dimension_matrix.csv`
3. **Dimensional Analysis** → Uses normalized data, saves `afterDA_data.csv` and `normalized_lg_afterDA_data.csv`
4. **Dimensional Filtering** → Uses normalized lg afterDA data, saves PCA/SIR results and `suggested_dominant_count.json`
5. **Optimization and Discovery** → Uses normalized lg afterDA data and suggested count, discovers dimensionless scaling laws

### Typical Pipeline

```bash
# Step 1: Generate synthetic data
python generate_data.py --config pydimension/configs/config_synthetic.json

# Step 2: Preprocess the data
python preprocess_data.py \
    --input_file output/data/dataset_synthetic.csv \
    --dimension_matrix_file output/data/dimension_matrix_synthetic.csv

# Step 3: Perform dimensional analysis (with normalized lg data)
python analyze_dimensions.py \
    --normalized_data_file output/data/normalized_data.csv \
    --dimension_matrix_file output/data/dimension_matrix.csv \
    --save-normalized-lg

# Step 4: Dimensional filtering
python filter_constraints.py \
    --input_file output/data/normalized_lg_afterDA_data.csv \
    --plot
```

## Mathematical Background

### Principal Component Analysis (PCA)

PCA finds directions of maximum variance in the data:

1. Standardize data: X_standardized = (X - mean) / std
2. Compute SVD: X = U × S × V^T
3. Eigenvalues: λ = S² / (n-1)
4. Explained variance: λ / Σλ

**Interpretation**: Components with high eigenvalues explain more variance and are more important.

### Sliced Inverse Regression (SIR)

SIR finds directions in input space that best predict the output:

1. Slice output into bins (e.g., low, medium, high values)
2. Compute mean of inputs in each slice
3. Analyze variation of these means across slices
4. Directions with most variation are most important

**Key advantage**: Works even when relationship is nonlinear (exp, log, etc.)

**If 1 direction dominates** → you found your 1D hidden variable! ⭐

## Troubleshooting

### Error: "Not enough columns in input file"

The input file must have at least 2 columns (1 input + 1 output).

**Solution**: Check your input file format.

### Warning: "Too many slices for samples"

SIR automatically adjusts the number of slices if it's too large for the dataset.

**Solution**: This is handled automatically. Reduce `n_sir_slices` if you want fewer slices.

### Error: "No analysis results available"

You need to run at least one analysis (PCA or SIR) before creating visualization.

**Solution**: Ensure `run_pca=True` or `run_sir=True` in config.

### Error: "input_file must be specified or found in default locations"

The dimensional filtering module requires `normalized_lg_afterDA_data.csv` as input, which is generated by the dimensional analysis module.

**Solution**: Run dimensional analysis with the `--save-normalized-lg` flag first:
```bash
python analyze_dimensions.py --config pydimension/configs/config_synthetic.json --save-normalized-lg
```

Then run dimensional filtering:
```bash
python filter_constraints.py --config pydimension/configs/config_synthetic.json
```

Alternatively, you can specify the input file explicitly:
```bash
python filter_constraints.py --input_file path/to/normalized_lg_afterDA_data.csv
```

## See Also

- Main package README: `../../README.md`
- Data Generation Module: `../data_generation/README.md`
- Data Preprocessing Module: `../data_preprocessing/README.md`
- Dimensional Analysis Module: `../dimensional_analysis/README.md`
- Optimization and Discovery Module: `../optimization_discovery/README.md`
- Config files: `../../configs/README.md`
- Example config: `../../configs/config_synthetic.json`

