# Data Generation Module

Generate synthetic datasets with known dimensionless relationships for testing dimensionless learning pipelines.

## Overview

The data generation module creates synthetic datasets where the relationship between input variables and output is known through dimensionless groups (π₁, π₂, ...). This is useful for:
- Testing dimensionless learning algorithms
- Validating discovery pipelines
- Creating benchmark datasets
- Understanding dimensionless relationships

## Quick Start

### Using a Config File (Recommended)

The unified config file (`pydimension/configs/config_synthetic.json`) contains settings for all modules and is designed to be extensible:

```bash
# Using generate_data.py convenience script
python generate_data.py --config pydimension/configs/config_synthetic.json --plot

# Or using the module directly
python -m pydimension.data_generation --config pydimension/configs/config_synthetic.json --plot

# With noise enabled
python generate_data.py --config pydimension/configs/config_synthetic_with_noise.json --plot
```

**Note**: See [pydimension/configs/README.md](../configs/README.md) for details on the unified config structure.

### Using Command-Line Arguments

```bash
# Basic usage
python -m pydimension.data_generation --N 7 --M 100 --ndim 1 --random_seed 42

# With noise (5% noise level)
python -m pydimension.data_generation --N 7 --M 100 --ndim 1 --noise_level 5.0 --random_seed 42

# With noise and visualization
python -m pydimension.data_generation --N 7 --M 100 --ndim 1 --noise_level 5.0 --plot
```

**When to use `--plot`:**
- **Visual verification**: Check if the generated data matches the expected relationship
- **Debugging**: Verify that coefficients, noise, or other parameters are working correctly
- **Documentation**: Generate plots for reports or presentations
- **Quality control**: Visually inspect data before using it in downstream analysis
- **Noise assessment**: See how noise affects the data distribution (especially useful with `--noise_level`)

The `--plot` flag generates a visualization saved to `output/figures/data_generation_plots.png` showing:
- **Left plot**: p* vs π₁ (and π₂, π₃ for multi-dimensional cases) with theoretical curve
- **Right plot**: Validation plot (actual vs predicted p*) with R² score

**Note**: Without `--plot`, only CSV data files are generated (faster for batch processing).

### Using Python API

```python
from pydimension.data_generation import DataGenerator, DataGenerationConfig

# Create config
config = DataGenerationConfig(
    N=7,
    M=100,
    ndim=1,
    poly_order=1,
    random_seed=32,
    coefficients=[2.0, 1.0]
)

# Generate data
generator = DataGenerator(config)
results = generator.generate(verbose=True)

# Save datasets
dataset_path, dim_matrix_path = generator.save_datasets()
print(f"Dataset saved to: {dataset_path}")
print(f"Dimension matrix saved to: {dim_matrix_path}")
```

**With noise:**
```python
# Add 5% noise to the output
config = DataGenerationConfig(
    N=7,
    M=100,
    ndim=1,
    coefficients=[2.0, 1.0],
    noise_level=5.0  # 5% noise
)

generator = DataGenerator(config)
generator.generate()
generator.save_datasets()

# Create visualization to see noise effect
plot_path = generator.create_visualization()
print(f"Plot saved to: {plot_path}")
```

## Configuration

### Config File Format

The unified config file (`pydimension/configs/config_synthetic.json`) contains settings for all modules and is designed to be extensible:

```json
{
  "DATA_GENERATION": {
    "N": 7,
    "M": 100,
    "ndim": 1,
    "poly_order": 1,
    "random_seed": 32,
    "noise_level": 0.0,
    "n_discrete": 0,
    "n_fix": 5,
    "coefficients": [2.0, 1.0],
    "gamma_vectors": [[1.0, 1.0, 0.0]]
  },
  "OUTPUT": {
    "output_dir": "output",
    "data_dir": "data",
    "figures_dir": "figures"
  },
  "DATA_GENERATION_OUTPUT": {
    "dataset_filename": "dataset_synthetic.csv",
    "dimension_matrix_filename": "dimension_matrix_synthetic.csv",
    "plot_filename": "data_generation_plots.png"
  }
}
```

**Example config files:**
- `pydimension/configs/config_synthetic.json` - Main unified config with all modules
- `pydimension/configs/config_synthetic_with_noise.json` - Unified config with 5% noise enabled

**Note**: See [pydimension/configs/README.md](../configs/README.md) for more details on the unified config structure.

### Parameter Descriptions

#### Core Parameters

- **N** (int, required): Number of input variables
  - Minimum: 5 (to ensure rank=4 dimension matrix)
  - Default: 7
  - Must satisfy: N ≥ 4 + ndim

- **M** (int, required): Number of datapoints
  - Minimum: 10
  - Default: 100

- **ndim** (int, required): Number of dimensionless groups
  - Range: 1-3 (higher values not supported)
  - Default: 1
  - Determines the output relationship type:
    - `ndim=1`: Polynomial relationship p* = f(π₁)
    - `ndim=2`: Nonlinear relationship p* = exp(A×π₁) + π₂^B
    - `ndim=3`: Nonlinear relationship p* = exp(A×π₁) + π₂^B + log(1+C×π₃)

#### Output Relationship Parameters

- **poly_order** (int): Polynomial order for ndim=1
  - Range: 1-10
  - Default: 1 (linear)
  - Only used when ndim=1
  - Determines number of coefficients needed: poly_order + 1

- **coefficients** (list of float): Coefficients for output relationship
  - For **ndim=1** (polynomial):
    - Format: `[A, B, C, D, ...]`
    - Relationship: `p* = A + B×π₁ + C×π₁² + D×π₁³ + ...`
    - Number needed: `poly_order + 1`
    - Example: `[2.0, 1.0]` → `p* = 2 + 1×π₁`
    - Example: `[2.0, 1.0, 0.5]` → `p* = 2 + π₁ + 0.5×π₁²`
  
  - For **ndim>1** (nonlinear):
    - Format: `[A, B, C]`
    - Relationship: `p* = exp(A×π₁) + π₂^B + log(1+C×π₃)`
    - Number needed: at least `ndim`
    - Example (ndim=2): `[2.0, -0.5]` → `p* = exp(2×π₁) + π₂^(-0.5)`
    - Example (ndim=3): `[2.0, -0.5, 0.5]` → `p* = exp(2×π₁) + π₂^(-0.5) + log(1+0.5×π₃)`

#### Sampling Parameters

- **random_seed** (int): Random seed for reproducibility
  - Default: 32
  - The generator will automatically retry with incremented seeds if vector components exceed [-5, 5]
  - The final seed used is reported in the results

- **noise_level** (float): Noise level as percentage of output range
  - Range: 0.0-100.0
  - Default: 0.0 (no noise)
  - Example: `5.0` adds 5% Gaussian noise

- **n_discrete** (int): Number of discretely sampled variables
  - Range: 0 to N
  - Default: 0 (all continuous)
  - Variables are randomly selected for discrete sampling

- **n_fix** (int): Number of fixed values for discrete variables
  - Minimum: 2
  - Default: 5
  - Only used when n_discrete > 0

#### Advanced Parameters

- **gamma_vectors** (list of lists, optional): Gamma vectors for combining basis vectors
  - Each gamma vector has dimension N-4
  - Number of vectors should equal ndim
  - If not provided, will be auto-generated to ensure linear independence
  - Format: `[[γ₁₁, γ₁₂, ...], [γ₂₁, γ₂₂, ...], ...]`
  - Example for N=7 (gamma_dim=3), ndim=2: `[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]`

#### Output Parameters

- **output_dir** (str): Directory to save output files
  - Default: "." (current directory)
  - Will be created if it doesn't exist

- **dataset_filename** (str): Filename for generated dataset
  - Default: "dataset_synthetic.csv"

- **dimension_matrix_filename** (str): Filename for dimension matrix
  - Default: "dimension_matrix_synthetic.csv"

## Output Files

### dataset_synthetic.csv

Contains the complete generated dataset with columns:
- Input variables: `p1, p2, ..., pN`
- Dimensionless variables: `π1, π2, ..., π_ndim`
- Output: `p*`

Example (N=7, ndim=1):
```
p1,p2,p3,p4,p5,p6,p7,π1,p*
0.623,0.789,0.654,0.712,0.801,0.567,0.698,0.523,2.523
...
```

### dimension_matrix_synthetic.csv

Contains the dimension matrix (7 dimensions × N variables):
- Rows: Mass, Length, Time, Temperature, Current, Amount, Luminous
- Columns: p1, p2, ..., pN
- Values: Integer exponents (-2 to 2 for first 4 dimensions, 0 for last 3)

## Examples

### Example 1: Simple Linear Relationship

Generate data with a linear relationship: `p* = 2 + 1×π₁`

**Config file:**
```json
{
  "DATA_GENERATION": {
    "N": 7,
    "M": 100,
    "ndim": 1,
    "poly_order": 1,
    "random_seed": 32,
    "coefficients": [2.0, 1.0]
  }
}
```

**Command:**
```bash
python -m pydimension.data_generation --config pydimension/configs/config_synthetic.json
```

**Python:**
```python
from pydimension.data_generation import DataGenerator, DataGenerationConfig

config = DataGenerationConfig(
    N=7, M=100, ndim=1, poly_order=1,
    coefficients=[2.0, 1.0]
)
generator = DataGenerator(config)
generator.generate()
generator.save_datasets()
```

### Example 2: Quadratic Relationship

Generate data with a quadratic relationship: `p* = 2 + 1×π₁ + 0.5×π₁²`

**Config file:**
```json
{
  "DATA_GENERATION": {
    "N": 7,
    "M": 200,
    "ndim": 1,
    "poly_order": 2,
    "random_seed": 42,
    "coefficients": [2.0, 1.0, 0.5]
  }
}
```

### Example 3: Multi-dimensional Nonlinear

Generate data with two dimensionless groups: `p* = exp(2×π₁) + π₂^(-0.5)`

**Config file:**
```json
{
  "DATA_GENERATION": {
    "N": 8,
    "M": 200,
    "ndim": 2,
    "random_seed": 32,
    "coefficients": [2.0, -0.5]
  }
}
```

### Example 4: With Noise and Discrete Variables

Generate data with 5% noise and 2 discrete variables:

**Config file:**
```json
{
  "DATA_GENERATION": {
    "N": 7,
    "M": 500,
    "ndim": 1,
    "poly_order": 2,
    "random_seed": 32,
    "noise_level": 5.0,
    "n_discrete": 2,
    "n_fix": 5,
    "coefficients": [2.0, 1.0, 0.5]
  }
}
```

### Example 5: Custom Gamma Vectors

Specify custom gamma vectors for combining basis vectors:

**Config file:**
```json
{
  "DATA_GENERATION": {
    "N": 7,
    "M": 100,
    "ndim": 2,
    "random_seed": 32,
    "coefficients": [2.0, -0.5],
    "gamma_vectors": [
      [1.0, 1.0, 0.0],
      [0.0, 1.0, 1.0]
    ]
  }
}
```

**Command-line:**
```bash
python -m pydimension.data_generation --N 7 --M 100 --ndim 2 \
    --coefficients 2.0 -0.5 \
    --gamma_vectors "1,1,0;0,1,1"
```

## Command-Line Interface

### Basic Usage

```bash
# Using config file
python -m pydimension.data_generation --config pydimension/configs/config_synthetic.json

# Override config parameters
python -m pydimension.data_generation --config pydimension/configs/config_synthetic.json --M 500 --noise_level 5.0

# Command-line only (no config file)
python -m pydimension.data_generation --N 7 --M 100 --ndim 1 --coefficients 2.0 1.0

# Generate with visualization
python -m pydimension.data_generation --config pydimension/configs/config_synthetic.json --plot
```

### When to Use `--plot`

The `--plot` flag generates visualization plots saved to `output/figures/data_generation_plots.png`. Use it when:

✅ **Use `--plot` when:**
- **First-time setup**: Verify your configuration produces expected results
- **Parameter tuning**: Visualize how different coefficients or noise levels affect the data
- **Quality assurance**: Check data quality before using in downstream modules
- **Debugging**: Troubleshoot issues with data generation
- **Documentation**: Create plots for reports, papers, or presentations
- **Noise analysis**: See the effect of noise on the dimensionless relationship
- **Validation**: Verify the theoretical relationship matches generated data

❌ **Skip `--plot` when:**
- **Batch processing**: Generating many datasets automatically (faster without plots)
- **Production runs**: When you're confident in your configuration
- **Large datasets**: Very large M values where plotting might be slow
- **Automated pipelines**: Scripts that don't need visual output

**What the plot shows:**
- **Left subplot(s)**: p* vs πᵢ scatter plots with theoretical curve overlay
- **Right subplot**: Validation plot (actual vs predicted p*) with R² score
- **File location**: `output/figures/data_generation_plots.png`

### All Options

```bash
python -m pydimension.data_generation \
    --config pydimension/configs/config_synthetic.json \              # Config file path
    --N 7 \                              # Number of variables
    --M 100 \                            # Number of datapoints
    --ndim 1 \                           # Number of dimensionless groups
    --poly_order 1 \                     # Polynomial order
    --random_seed 32 \                   # Random seed
    --noise_level 0.0 \                  # Noise level (%)
    --n_discrete 0 \                     # Number of discrete variables
    --n_fix 5 \                          # Fixed values per discrete variable
    --coefficients 2.0 1.0 \             # Coefficients (space-separated)
    --gamma_vectors "1,1,0;0,1,1" \      # Gamma vectors (semicolon-separated)
    --output_dir "." \                   # Output directory
    --dataset_filename dataset.csv \     # Dataset filename
    --dimension_matrix_filename dim.csv \ # Dimension matrix filename
    --max_trials 10 \                    # Max retry trials
    --quiet                              # Suppress progress messages
```

## Python API

### DataGenerationConfig

Configuration class for data generation parameters.

```python
from pydimension.data_generation import DataGenerationConfig

# Create from dictionary
config = DataGenerationConfig.from_dict({
    'DATA_GENERATION': {
        'N': 7,
        'M': 100,
        'ndim': 1,
        'coefficients': [2.0, 1.0]
    }
})

# Create from JSON file
config = DataGenerationConfig.from_json('pydimension/configs/config_synthetic.json')

# Create directly
config = DataGenerationConfig(
    N=7,
    M=100,
    ndim=1,
    coefficients=[2.0, 1.0]
)

# Validate configuration
errors = config.validate()
if errors:
    print("Configuration errors:", errors)

# Save to JSON (saved to configs directory)
config.to_json('pydimension/configs/my_config.json')
```

### DataGenerator

Main generator class for creating synthetic datasets.

```python
from pydimension.data_generation import DataGenerator, DataGenerationConfig

# Create generator
config = DataGenerationConfig(N=7, M=100, ndim=1, coefficients=[2.0, 1.0])
generator = DataGenerator(config)

# Generate data
results = generator.generate(verbose=True, max_trials=10)

# Access generated data
print(f"Input data shape: {generator.input_data.shape}")
print(f"Output values: {generator.output_values[:5]}")
print(f"π₁ values: {generator.pi_values[0][:5]}")
print(f"π₁ expression: {generator.pi_expressions[0]}")

# Save datasets
dataset_path, dim_matrix_path = generator.save_datasets(output_dir='./output')
```

### Results Dictionary

The `generate()` method returns a dictionary with:

```python
results = {
    'timestamp': '2025-01-01T12:00:00',
    'config': {...},                    # Full configuration
    'random_seed_used': 32,             # Final seed used
    'max_vector_component': 4.2,        # Max |wi| component
    'trials_needed': 1,                 # Number of trials
    'variable_names': ['p1', 'p2', ...],
    'dimension_matrix': [[...], ...],    # Dimension matrix
    'dimension_names': [...],
    'basis_vectors': [[...], ...],      # Basis vectors
    'gamma_vectors': [[...], ...],      # Gamma vectors used
    'final_vectors': [[...], ...],      # Final vectors w1, w2, ...
    'pi_expressions': ['π1 = p1 × p2', ...],
    'pi_ranges': {'π1': [0.5, 2.0], ...},
    'output_range': [1.5, 3.5],
    'output_mean': 2.5
}
```

## How It Works

The data generation process follows these steps:

1. **Generate Input Variables**: Create N input variables (p₁, p₂, ..., pₙ) with M datapoints
   - Values range from 0.5 to 1.0
   - Can be continuous (uniform) or discrete (n_fix fixed values)

2. **Generate Dimension Matrix**: Create a 7×N dimension matrix with rank=4
   - First 4 dimensions (Mass, Length, Time, Temperature): random integers from -2 to 2
   - Last 3 dimensions (Current, Amount, Luminous): all zeros
   - Automatically retries until rank=4 is achieved

3. **Compute Basis Vectors**: Solve dimension_matrix × w = 0 to get null space
   - Uses SymPy for exact rational arithmetic
   - Normalizes to unit vectors (magnitude = 1)
   - Number of basis vectors = N - 4

4. **Generate Final Vectors**: Combine basis vectors using gamma vectors
   - wᵢ = basis_vectors × γᵢ
   - Automatically retries with new seeds if components exceed [-5, 5]
   - Ensures linear independence of gamma vectors

5. **Calculate Dimensionless Variables**: Compute πᵢ = p₁^wᵢ[0] × p₂^wᵢ[1] × ... × pₙ^wᵢ[n-1]
   - Uses log-space computation to avoid overflow
   - Generates human-readable expressions

6. **Calculate Output**: Compute p* from dimensionless variables
   - For ndim=1: Polynomial p* = A + B×π₁ + C×π₁² + ...
   - For ndim>1: Nonlinear p* = exp(A×π₁) + π₂^B + log(1+C×π₃)
   - Optionally adds Gaussian noise

7. **Save Datasets**: Export to CSV files
   - Complete dataset with all variables
   - Dimension matrix for downstream analysis

## Troubleshooting

### "Configuration errors: N must be at least 5"
- Increase N to at least 5 (or 4 + ndim if ndim > 1)

### "Failed to generate rank=4 matrix"
- This is rare but can happen. Try a different random seed.

### "After X trials, could not generate vectors within [-5, 5]"
- The generator will proceed anyway, but you may want to:
  - Increase max_trials
  - Try a different starting seed
  - Adjust N or ndim

### "Invalid coefficient value"
- For ndim=1: Need at least (poly_order + 1) coefficients
- For ndim>1: Need at least ndim coefficients

### Import errors
- Make sure you're in the PyDimension directory or have installed the package
- Check that all dependencies are installed: `pip install -r requirements.txt`

## Integration with Other Modules

The generated datasets are designed to work seamlessly with other PyDimension modules:

- **Data Preprocessing**: Use `dataset_synthetic.csv` as input
- **Dimensional Analysis**: Use `dimension_matrix_synthetic.csv` for basis vector computation
- **Dimensional Filtering**: Analyze dimensionless groups to identify dominant ones
- **Neural Network Training**: Train models to discover the known relationships

## See Also

- Main package README: `../../README.md`
- Example config: `../../pydimension/configs/config_synthetic.json`
- Package documentation: `../../README.md`

