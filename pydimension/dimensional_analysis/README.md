# Dimensional Analysis Module

Perform dimensional analysis to find basis vectors of the null space and create dimensionless variables (π groups) for dimensionless learning pipelines.

## Overview

The dimensional analysis module performs the core dimensional analysis operations:
- Loads normalized data and dimension matrix from CSV files
- Computes the null space of the dimension matrix
- Simplifies basis vectors to sparse, simple components using SymPy
- Optionally normalizes basis vectors to unit length
- Creates dimensionless variables (π1, π2, ...) using basis vectors
- Saves afterDA data and basis vectors for downstream modules
- Optionally computes normalized log10 versions (lgπ = log10(π/max(π)))

## Quick Start

### Using the Convenience Script (Recommended)

The easiest way to perform dimensional analysis is using the convenience script:

```bash
# Using the unified config
python analyze_dimensions.py --config pydimension/configs/config_synthetic.json

# Using command-line arguments directly
python analyze_dimensions.py \
    --normalized_data_file output/data/normalized_data.csv \
    --dimension_matrix_file output/data/dimension_matrix.csv

# Also save normalized log10 data
python analyze_dimensions.py --config config_synthetic.json --save-normalized-lg
```

### Using the Module Directly

You can also use the module directly:

```bash
# Using the unified config
python -m pydimension.dimensional_analysis --config pydimension/configs/config_synthetic.json
```

### Using Command-Line Arguments

```bash
# Basic usage - auto-detect from default locations
python -m pydimension.dimensional_analysis

# Specify input files directly
python -m pydimension.dimensional_analysis \
    --normalized_data_file output/data/normalized_data.csv \
    --dimension_matrix_file output/data/dimension_matrix.csv

# Disable basis vector normalization
python -m pydimension.dimensional_analysis \
    --normalized_data_file normalized_data.csv \
    --dimension_matrix_file dimension_matrix.csv \
    --no-normalize-basis

# Also save normalized log10 data
python -m pydimension.dimensional_analysis --config config_synthetic.json --save-normalized-lg
```

### Using Python API

```python
from pydimension.dimensional_analysis import DimensionalAnalyzer, DimensionalAnalysisConfig

# Create config
config = DimensionalAnalysisConfig(
    normalized_data_file='output/data/normalized_data.csv',
    dimension_matrix_file='output/data/dimension_matrix.csv',
    normalize_basis=True
)

# Perform analysis
analyzer = DimensionalAnalyzer(config)
results = analyzer.process()

# Save results
afterDA_path, basis_path = analyzer.save_results()
print(f"AfterDA data: {afterDA_path}")
print(f"Basis vectors: {basis_path}")

# Optionally save normalized log10 data
lg_path = analyzer.save_normalized_lg_data()
print(f"Normalized lg data: {lg_path}")

# Create visualization
plot_path = analyzer.create_visualization()
print(f"Plot: {plot_path}")
```

## Visualization

The dimensional analysis module can generate comprehensive visualization plots to help you verify and understand the results.

### When to Use Visualization

✅ **Use `--plot` when:**
- **First-time analysis**: Verify your dimensional analysis is working correctly
- **Quality assurance**: Check basis vectors and dimensionless variables
- **Result verification**: Verify that π groups are correctly created
- **Documentation**: Create plots for reports, papers, or presentations
- **Debugging**: Troubleshoot issues with basis vectors or dimensionless variables

❌ **Skip `--plot` when:**
- **Batch processing**: Processing many datasets automatically (faster without plots)
- **Production runs**: When you're confident in your analysis pipeline
- **Large datasets**: Very large datasets where plotting might be slow

### What the Visualization Shows

The visualization creates a 2×3 grid of plots:

1. **Dimension Matrix Heatmap** (top-left)
   - Shows the dimension matrix for input variables
   - Color-coded from -3 (blue) to +3 (red)
   - Dimension names as rows, variables as columns
   - Integer values displayed as text

2. **Basis Vectors Visualization** (top-middle)
   - Shows the basis vectors (null space) as a heatmap
   - Color-coded from -1 (blue) to +1 (red)
   - Basis vectors (w1, w2, ...) as rows, variables as columns
   - Values displayed as text (formatted for readability)

3. **Dimensionless Variables (π) Distributions** (top-right)
   - Histograms showing the distribution of π groups
   - Shows first 3 π groups for clarity
   - Helps verify that π groups are well-distributed

4. **π vs Output Scatter Plot** (bottom-left)
   - Scatter plot showing relationship between π groups and output
   - Shows first 3 π groups
   - Helps identify which π groups are most correlated with output

5. **Correlation Matrix of π Groups** (bottom-middle)
   - Heatmap showing correlations between π groups
   - Color-coded from -1 (blue) to +1 (red)
   - Correlation values displayed as text
   - Helps identify independent π groups

6. **Summary Statistics** (bottom-right)
   - Text summary with:
     - Number of input variables and output variable
     - Active dimensions and matrix rank
     - Null space dimension
     - Number of dimensionless groups
     - Basis normalization status
     - Dimensionless expressions (first 3)
     - List of input variables (first 5)

### Usage

```bash
# Using convenience script with visualization
python analyze_dimensions.py --config config_synthetic.json --plot

# Using module directly with visualization
python -m pydimension.dimensional_analysis --config config_synthetic.json --plot

# Custom plot filename
python analyze_dimensions.py --config config_synthetic.json --plot --plot_filename my_plot.png
```

### Python API

```python
from pydimension.dimensional_analysis import DimensionalAnalyzer, DimensionalAnalysisConfig

config = DimensionalAnalysisConfig(
    normalized_data_file='output/data/normalized_data.csv',
    dimension_matrix_file='output/data/dimension_matrix.csv',
    normalize_basis=True
)

analyzer = DimensionalAnalyzer(config)
analyzer.process()

# Create visualization
plot_path = analyzer.create_visualization(
    filename='my_analysis_plot.png',
    show=False  # Set to True to display instead of saving
)
print(f"Plot saved to: {plot_path}")
```

**File location**: `output/figures/dimensional_analysis_plots.png` (or custom filename)

## How It Works

### Step 1: Load Data

The module loads:
- **Normalized data**: Preprocessed dataset with normalized input variables and output variable
- **Dimension matrix**: Matrix mapping variables to fundamental dimensions

The module automatically:
- Identifies input variables (all columns except the last)
- Identifies output variable (last column)
- Filters dimension matrix to only include input variables
- Removes unused dimensions (rows with all zeros)

### Step 2: Find Null Space

The module computes the null space of the dimension matrix:
- Uses `scipy.linalg.null_space()` to find basis vectors
- The null space dimension = number of variables - matrix rank
- Each basis vector represents a dimensionless group

### Step 3: Simplify Basis Vectors

The module simplifies basis vectors using SymPy:
- Converts to exact rational arithmetic
- Clears denominators using LCM
- Makes vectors primitive using GCD
- Normalizes sign (first non-zero element positive)
- Results in sparse vectors with simple integer/fractional components

### Step 4: Normalize Basis Vectors (Optional)

If `normalize_basis=True`:
- Normalizes each basis vector to unit length (magnitude = 1)
- Ensures numerical stability and consistency across modules
- Uses Euclidean norm (L2 norm)

### Step 5: Create Dimensionless Variables

For each basis vector w_i, creates a dimensionless variable:
```
πi = p1^w_i1 × p2^w_i2 × ... × pn^w_in
```

Where:
- p1, p2, ..., pn are input variables
- w_i1, w_i2, ..., w_in are components of basis vector w_i

The calculation uses logarithms to avoid overflow:
```
log(πi) = w_i1 × log(p1) + w_i2 × log(p2) + ... + w_in × log(pn)
πi = exp(log(πi))
```

### Step 6: Save Results

Saves:
- **afterDA_data.csv**: Dimensionless variables (π1, π2, ...) and output variable
- **basis_vectors.csv**: Basis vectors with variables as rows and w1, w2, ... as columns

### Step 7: Normalized Log10 Data (Optional)

If `--save-normalized-lg` is used:
- Divides each π by its maximum: π_norm = π / max(π)
- Computes log10: lgπ = log10(π_norm)
- Normalizes output: output_norm = output / max(output)
- **Note**: Output is NOT logged, only normalized
- Saves to **normalized_lg_afterDA_data.csv**

## Configuration

### Unified Config Format

The unified config file contains a `DIMENSIONAL_ANALYSIS` section:

```json
{
  "DIMENSIONAL_ANALYSIS": {
    "enabled": true,
    "normalized_data_file": "output/data/normalized_data.csv",
    "dimension_matrix_file": "output/data/dimension_matrix.csv",
    "normalize_basis": true
  },
  "OUTPUT": {
    "output_dir": "output",
    "data_dir": "data",
    "figures_dir": "figures"
  },
  "DIMENSIONAL_ANALYSIS_OUTPUT": {
    "afterDA_data_filename": "afterDA_data.csv",
    "basis_vectors_filename": "basis_vectors.csv",
    "normalized_lg_data_filename": "normalized_lg_afterDA_data.csv"
  }
}
```

### Parameter Descriptions

#### Input Settings

- **normalized_data_file** (str, optional): Path to normalized data CSV file
  - Can be absolute or relative path
  - If `null`, will try to find default files from data preprocessing module
  - Default locations checked:
    - `output/data/normalized_data.csv`
    - `normalized_data.csv`

- **dimension_matrix_file** (str, optional): Path to dimension matrix CSV file
  - Can be absolute or relative path
  - If `null`, will try to find default files from data preprocessing module
  - Default locations checked:
    - `output/data/dimension_matrix.csv`
    - `dimension_matrix.csv`

#### Analysis Settings

- **normalize_basis** (bool): Whether to normalize basis vectors to unit length
  - Default: `true`
  - If `true`, each basis vector is normalized: v_normalized = v / ||v||
  - Ensures numerical stability and consistency across modules
  - If `false`, basis vectors keep their original magnitudes

#### Output Settings

- **output_dir** (str): Base output directory
  - Default: `"output"`
  - Files are saved to `output_dir/data_dir/`

- **afterDA_data_filename** (str): Filename for afterDA data
  - Default: `"afterDA_data.csv"`

- **basis_vectors_filename** (str): Filename for basis vectors
  - Default: `"basis_vectors.csv"`

- **normalized_lg_data_filename** (str): Filename for normalized log10 data
  - Default: `"normalized_lg_afterDA_data.csv"`

- **plot_filename** (str): Filename for visualization plot
  - Default: `"dimensional_analysis_plots.png"`

## Output Files

### afterDA_data.csv

Contains dimensionless variables and output:
- Columns: `π1`, `π2`, ..., `πm`, `output_variable`
- Each row represents one data point
- π values are dimensionless (no units)

Example:
```csv
π1,π2,output
0.5234567890,0.1234567890,0.4567890123
0.6123456789,0.2345678901,0.5678901234
...
```

### basis_vectors.csv

Contains basis vectors:
- Rows: Input variables
- Columns: `Variable`, `w1`, `w2`, ..., `wm`
- Each column w_i is a basis vector
- Values are exponents for creating dimensionless variables

Example:
```csv
Variable,w1,w2
p1,1.0,0.0
p2,1.0,1.0
p3,0.0,1.0
...
```

### normalized_lg_afterDA_data.csv (Optional)

Contains normalized log10 versions:
- Columns: `lgπ1`, `lgπ2`, ..., `lgπm`, `output_variable`
- lgπ values are log10(π/max(π))
- Output is normalized (output/max(output)) but NOT logged

Example:
```csv
lgπ1,lgπ2,output
-0.2801234567,-0.6123456789,0.4567890123
-0.2123456789,-0.5234567890,0.5678901234
...
```

## Dimensionless Variable Expressions

The module generates human-readable expressions for each dimensionless variable:

Example output:
```
π1 = p1 × p2
π2 = p2 × p3^(-1)
π3 = p4^(2.000) × p5^(-1.000)
```

These expressions show how each π is constructed from input variables.

## Examples

### Example 1: Complete Pipeline

After preprocessing data:

```bash
# Step 1: Generate data
python generate_data.py --config pydimension/configs/config_synthetic.json

# Step 2: Preprocess data
python preprocess_data.py --config pydimension/configs/config_synthetic.json

# Step 3: Perform dimensional analysis
python analyze_dimensions.py --config pydimension/configs/config_synthetic.json
```

### Example 2: With Normalized Log10 Data

```bash
python analyze_dimensions.py \
    --config pydimension/configs/config_synthetic.json \
    --save-normalized-lg
```

### Example 3: Disable Basis Normalization

```bash
python analyze_dimensions.py \
    --normalized_data_file output/data/normalized_data.csv \
    --dimension_matrix_file output/data/dimension_matrix.csv \
    --no-normalize-basis
```

### Example 4: Using Config File

Create or update `pydimension/configs/config_synthetic.json`:

```json
{
  "DIMENSIONAL_ANALYSIS": {
    "enabled": true,
    "normalized_data_file": "output/data/normalized_data.csv",
    "dimension_matrix_file": "output/data/dimension_matrix.csv",
    "normalize_basis": true
  },
  "OUTPUT": {
    "output_dir": "output",
    "data_dir": "data"
  },
  "DIMENSIONAL_ANALYSIS_OUTPUT": {
    "afterDA_data_filename": "afterDA_data.csv",
    "basis_vectors_filename": "basis_vectors.csv",
    "normalized_lg_data_filename": "normalized_lg_afterDA_data.csv"
  }
}
```

Then run:
```bash
python analyze_dimensions.py --config pydimension/configs/config_synthetic.json
```

## Python API

### DimensionalAnalysisConfig

Configuration class for dimensional analysis:

```python
from pydimension.dimensional_analysis import DimensionalAnalysisConfig

# Create from dict
config_dict = {
    'DIMENSIONAL_ANALYSIS': {
        'normalized_data_file': 'normalized_data.csv',
        'dimension_matrix_file': 'dimension_matrix.csv',
        'normalize_basis': True
    },
    'OUTPUT': {
        'output_dir': 'output',
        'data_dir': 'data'
    }
}
config = DimensionalAnalysisConfig.from_dict(config_dict)

# Create from JSON
config = DimensionalAnalysisConfig.from_json('config_synthetic.json')

# Create directly
config = DimensionalAnalysisConfig(
    normalized_data_file='normalized_data.csv',
    dimension_matrix_file='dimension_matrix.csv',
    normalize_basis=True,
    output_dir='output'
)

# Validate
errors = config.validate()
if errors:
    print("Errors:", errors)
```

### DimensionalAnalyzer

Main analyzer class:

```python
from pydimension.dimensional_analysis import DimensionalAnalyzer, DimensionalAnalysisConfig

# Create config
config = DimensionalAnalysisConfig(
    normalized_data_file='output/data/normalized_data.csv',
    dimension_matrix_file='output/data/dimension_matrix.csv',
    normalize_basis=True
)

# Create analyzer
analyzer = DimensionalAnalyzer(config)

# Load data
normalized_data, dimension_matrix = analyzer.load_data()

# Find basis vectors
basis_vectors = analyzer.find_basis_vectors()

# Create dimensionless variables
afterDA_data = analyzer.create_dimensionless_variables()

# Or run complete pipeline
results = analyzer.process(verbose=True)

# Save results
afterDA_path, basis_path = analyzer.save_results()

# Optionally save normalized lg data
lg_path = analyzer.save_normalized_lg_data()

# Create visualization
plot_path = analyzer.create_visualization()
print(f"Plot: {plot_path}")
```

### Results Dictionary

The `process()` method returns a dictionary with:

```python
results = {
    'timestamp': '2025-01-01T12:00:00',
    'normalized_data_file': 'output/data/normalized_data.csv',
    'dimension_matrix_file': 'output/data/dimension_matrix.csv',
    'input_variables': ['p1', 'p2', 'p3', ...],
    'output_variable': 'p*',
    'dimension_names': ['Mass', 'Length', 'Time', ...],
    'matrix_shape': (7, 7),
    'matrix_rank': 4,
    'null_space_dimension': 3,
    'basis_vectors': [[1.0, 0.0, ...], [0.0, 1.0, ...], ...],
    'dimensionless_expressions': ['π1 = p1 × p2', 'π2 = p2 × p3^(-1)', ...],
    'normalize_basis': True,
    'data_shape': (100, 4)
}
```

## Integration with Other Modules

### Workflow

1. **Data Generation** → Generates `dataset_synthetic.csv` and `dimension_matrix_synthetic.csv`
2. **Data Preprocessing** → Loads data, normalizes, saves `normalized_data.csv` and `dimension_matrix.csv`
3. **Dimensional Analysis** → Uses normalized data and dimension matrix, saves `afterDA_data.csv`, `basis_vectors.csv`, and `normalized_lg_afterDA_data.csv`
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

# Step 3: Perform dimensional analysis
python analyze_dimensions.py \
    --normalized_data_file output/data/normalized_data.csv \
    --dimension_matrix_file output/data/dimension_matrix.csv

# Step 4: Dimensional filtering (outputs suggested_dominant_count.json)
python filter_constraints.py --config pydimension/configs/config_synthetic.json --plot

# Step 5: Optimization and discovery (automatically uses suggested count)
python optimize_discovery.py --config pydimension/configs/config_synthetic.json --plot
```

## Mathematical Background

### Dimension Matrix

The dimension matrix maps variables to fundamental dimensions:

```
D = [d_ij]
```

Where:
- Rows: Fundamental dimensions (Mass, Length, Time, Temperature, Current, Amount, Luminous)
- Columns: Variables (p1, p2, ..., pn)
- d_ij: Exponent of dimension i for variable j

### Null Space

The null space of D contains vectors w such that:

```
D × w = 0
```

Each vector w in the null space represents a dimensionless group:
- If w = [w1, w2, ..., wn], then π = p1^w1 × p2^w2 × ... × pn^wn is dimensionless

### Basis Vectors

The null space has dimension m = n - rank(D), where:
- n = number of variables
- rank(D) = rank of dimension matrix

The module finds m linearly independent basis vectors w1, w2, ..., wm.

### Dimensionless Variables

For each basis vector wi, creates:
```
πi = p1^(wi1) × p2^(wi2) × ... × pn^(win)
```

These π variables are dimensionless and can be used for dimensionless learning.

## Troubleshooting

### Error: "No null space found"

This means the dimension matrix is full rank (rank = number of variables). Possible causes:
- All variables are already dimensionless
- Dimension matrix is incorrect
- Variables are linearly dependent in dimension space

**Solution**: Check your dimension matrix and variables.

### Error: "All basis vectors are zero"

This means the dimension matrix is full rank. The null space dimension is 0.

**Solution**: Verify your dimension matrix has rank < number of variables.

### Warning: "Simplified vectors not exactly in null space"

This can occur after normalization due to floating-point precision. The error should be < 1e-10.

**Solution**: This is usually acceptable. If the error is large, check your dimension matrix.

### SymPy Not Available

If SymPy is not installed, the module will use the original null space vectors (may not be simplified).

**Solution**: Install SymPy: `pip install sympy`

## See Also

- Main package README: `../../README.md`
- Data Generation Module: `../data_generation/README.md`
- Data Preprocessing Module: `../data_preprocessing/README.md`
- Dimensional Filtering Module: `../constraint_filtering/README.md`
- Optimization and Discovery Module: `../optimization_discovery/README.md`
- Config files: `../../configs/README.md`
- Example config: `../../configs/config_synthetic.json`

