# Optimization and Discovery Module

Train neural networks to discover dimensionless scaling laws from preprocessed data.

## Overview

The optimization and discovery module uses neural networks with a special architecture to discover dimensionless relationships:
- **Linear Combination Layer**: Learns gamma vectors (dimensionless group coefficients)
- **Hidden Layers**: Learn nonlinear relationships between dimensionless groups and output
- **Ensemble Learning**: Trains multiple models for robustness and uncertainty quantification
- **Gamma Regularization**: Encourages simple integer/half-integer coefficients (soft quantization)
- **Fixed Gamma Option**: Can use basis vectors from dimensional analysis as fixed gamma values

## Quick Start

### Using the Convenience Script (Recommended)

The easiest way to run optimization and discovery is using the convenience script:

```bash
# Using the unified config (automatically uses suggested count from dimensional filtering)
python optimize_discovery.py --config pydimension/configs/config_synthetic.json --plot

# Using command-line arguments directly
python optimize_discovery.py \
    --input_file output/data/normalized_lg_afterDA_data.csv \
    --num_linear 1 \
    --epochs 1000 \
    --plot
```

**Important**: The `num_linear` parameter (number of dimensionless groups) is **automatically loaded** from the dimensional filtering results if not explicitly set. This means:

1. **Run dimensional filtering first** to get the suggested dominant count:
   ```bash
   python filter_constraints.py --config pydimension/configs/config_synthetic.json
   ```
   This creates `output/results/suggested_dominant_count.json`.

2. **Then run optimization discovery** - it will automatically use the suggested count:
   ```bash
   python optimize_discovery.py --config pydimension/configs/config_synthetic.json --plot
   ```

3. **Or explicitly set `num_linear`** if you want to override:
   ```bash
   python optimize_discovery.py --config config_synthetic.json --num_linear 2 --plot
   ```

### Using the Module Directly

You can also use the module directly:

```bash
# Using the unified config
python -m pydimension.optimization_discovery --config pydimension/configs/config_synthetic.json --plot

# Using command-line arguments
python -m pydimension.optimization_discovery \
    --input_file output/data/normalized_lg_afterDA_data.csv \
    --num_linear 1 \
    --epochs 1000 \
    --plot
```

### Using Python API

```python
from pydimension.optimization_discovery import OptimizationDiscoverer, OptimizationDiscoveryConfig

# Create config
config = OptimizationDiscoveryConfig(
    input_file='output/data/normalized_lg_afterDA_data.csv',
    num_linear=1,
    num_hidden_layers=4,
    num_hidden_nodes=10,
    epochs=1000,
    num_ensembles=5,
    use_gamma_regularization=True,
    gamma_reg_strength=0.01,
    gamma_reg_resolution='half-integers'
)

# Train models
optimizer = OptimizationDiscoverer(config)
results = optimizer.process()

# Save results
results_path, results_dir = optimizer.save_results()
print(f"Results saved to: {results_path}")

# Create visualization
plot_path = optimizer.create_visualization()
print(f"Plot saved to: {plot_path}")
```

## Visualization

The optimization and discovery module can generate visualization plots to help you understand the training results.

### When to Use `--plot`

Use the `--plot` flag when you want to:
- Visualize training history (loss curves for all ensemble models)
- See predictions vs actual values with R² score
- Verify that models are learning correctly
- Compare performance across ensemble members

### What the Plots Show

The visualization includes:

1. **Summary Overview** (Top row):
   - **All Models Training History**: Combined view of all ensemble models' loss curves (train and test)
   - **Ensemble Predictions vs Actual**: Scatter plot showing ensemble-averaged predictions against actual values, with R² score

2. **Individual Model Details** (One row per model):
   - **Model Loss History**: Individual training and test loss curves for each model
     - Shows final train and test loss values
     - Displays R² score in the title
     - Marks the best model with ⭐
   - **Model Predictions vs Actual**: Individual predictions for each model
     - Shows how well each model performs
     - Displays R² score in the title
     - Marks the best model with ⭐

This layout makes it easy to:
- Compare training progress across all models (summary view)
- Analyze individual model performance (detailed view)
- Identify the best performing model
- Spot any models that didn't converge properly

### Usage Examples

```bash
# Generate plots with default filename
python optimize_discovery.py --config config_synthetic.json --plot

# Generate plots with custom filename
python optimize_discovery.py --config config_synthetic.json --plot --plot_filename my_plots.png
```

### Python API

```python
# Create visualization
plot_path = optimizer.create_visualization(filename='my_plots.png')
print(f"Plot saved to: {plot_path}")
```

## Configuration

The optimization and discovery module uses the unified config format. The relevant sections are:

### `OPTIMIZATION_DISCOVERY` Section

```json
{
  "OPTIMIZATION_DISCOVERY": {
    "enabled": false,
    "input_file": null,
    "basis_vectors_file": null,
    "num_linear": 1,
    "num_hidden_layers": 4,
    "num_hidden_nodes": 10,
    "random_seed": 49,
    "epochs": 1000,
    "learning_rate": 0.001,
    "train_percent": 0.8,
    "num_ensembles": 5,
    "use_fixed_gamma": false,
    "use_gamma_regularization": true,
    "gamma_reg_strength": 0.01,
    "gamma_reg_resolution": "half-integers"
  }
}
```

### Parameters

- **`input_file`** (str, optional): Path to normalized lg afterDA data CSV. If not specified, searches default locations.
- **`basis_vectors_file`** (str, optional): Path to basis_vectors.csv. Used for fixed gamma or expression discovery.
- **`num_linear`** (int, default=1): Number of linear nodes (gamma vectors). Should match suggested dominant count from dimensional filtering.
- **`num_hidden_layers`** (int, default=4): Number of hidden layers in the neural network.
- **`num_hidden_nodes`** (int, default=10): Number of nodes per hidden layer.
- **`random_seed`** (int, default=49): Random seed for reproducibility.
- **`epochs`** (int, default=1000): Number of training epochs.
- **`learning_rate`** (float, default=0.001): Learning rate for Adam optimizer.
- **`train_percent`** (float, default=0.8): Train/test split ratio (0-1).
- **`num_ensembles`** (int, default=5): Number of ensemble models to train.
- **`use_fixed_gamma`** (bool, default=false): Use fixed gamma values from basis vectors (not learned).
- **`use_gamma_regularization`** (bool, default=true): Enable gamma regularization (soft quantization).
- **`gamma_reg_strength`** (float, default=0.01): Strength of gamma regularization penalty.
- **`gamma_reg_resolution`** (str, default="half-integers"): Target resolution for gamma values. Options: "integers", "half-integers", "quarter-integers".

### `OPTIMIZATION_DISCOVERY_OUTPUT` Section

```json
{
  "OPTIMIZATION_DISCOVERY_OUTPUT": {
    "model_results_filename": "optimization_discovery_results.json",
    "plot_filename": "optimization_discovery_plots.png"
  }
}
```

## Neural Network Architecture

The neural network has a special architecture designed for discovering dimensionless relationships:

1. **Linear Combination Layer**: 
   - Input: log-transformed dimensionless variables (lgπ₁, lgπ₂, ...)
   - Output: Linear combinations (gamma vectors)
   - No activation, no bias
   - This layer learns the dimensionless group coefficients

2. **Hidden Layers**:
   - ReLU activation
   - Learn nonlinear relationships between dimensionless groups and output

3. **Output Layer**:
   - Single output node
   - Predicts the target variable

## Gamma Regularization

Gamma regularization (soft quantization) encourages learned coefficients to be simple values:

- **Integers**: 0, ±1, ±2, ±3, ±4, ±5
- **Half-integers**: 0, ±0.5, ±1, ±1.5, ±2, ±2.5, ±3 (default)
- **Quarter-integers**: 0, ±0.25, ±0.5, ±0.75, ±1, ±1.25, ..., ±3

The regularization penalty is the minimum distance from each gamma value to the nearest simple value, weighted by `gamma_reg_strength`.

## Ensemble Learning

The module trains multiple models (ensemble) to:
- **Robustness**: Reduce sensitivity to random initialization
- **Uncertainty Quantification**: Estimate prediction uncertainty from ensemble variance
- **Model Selection**: Choose best model based on R² score

## Output Files

### Results JSON

Saved to `output/results/optimization_discovery_results.json`:

```json
{
  "timestamp": "2025-01-01T12:00:00",
  "config": {...},
  "training_history": {
    "train_loss": [[...], [...]],
    "test_loss": [[...], [...]]
  },
  "model_performance": {
    "model_r2_scores": [0.95, 0.94, ...],
    "final_train_loss": [0.001, 0.002, ...],
    "final_test_loss": [0.001, 0.002, ...]
  },
  "learned_gamma_vectors": {
    "best_model_index": 0,
    "gamma_vectors": [[...], [...]],
    "all_models_gamma_vectors": [[[...], [...]]], ...]
  },
  "input_columns": ["lgπ1", "lgπ2", ...],
  "output_column": "output"
}
```

### Visualization Plot

Saved to `output/figures/optimization_discovery_plots.png`:
- Training history for all ensemble models
- Predictions vs actual values with R² score

## Integration with Other Modules

### Workflow

1. **Data Generation** → Generates `dataset_synthetic.csv` and `dimension_matrix_synthetic.csv`
2. **Data Preprocessing** → Loads data, normalizes, saves `normalized_data.csv` and `dimension_matrix.csv`
3. **Dimensional Analysis** → Uses normalized data, saves `afterDA_data.csv`, `basis_vectors.csv`, and `normalized_lg_afterDA_data.csv`
4. **Dimensional Filtering** → Uses normalized lg afterDA data, suggests dominant count
5. **Optimization and Discovery** → Uses normalized lg afterDA data, discovers dimensionless scaling laws

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

# Step 4: Dimensional filtering (get suggested dominant count)
python filter_constraints.py --config pydimension/configs/config_synthetic.json --plot
# This creates: output/results/suggested_dominant_count.json

# Step 5: Optimization and discovery (automatically uses suggested count)
python optimize_discovery.py --config pydimension/configs/config_synthetic.json --plot
# No need to specify --num_linear - it's loaded from step 4 automatically!

# Or explicitly override if needed:
python optimize_discovery.py \
    --config pydimension/configs/config_synthetic.json \
    --num_linear 2 \
    --epochs 1000 \
    --plot
```

## Troubleshooting

### Error: "PyTorch is required for neural network training"

PyTorch is required for this module. Install it with:
```bash
pip install torch
```

### Error: "scikit-learn not available"

scikit-learn is required for data scaling and metrics. Install it with:
```bash
pip install scikit-learn
```

### Error: "input_file must be specified or found in default locations"

The optimization and discovery module requires `normalized_lg_afterDA_data.csv` as input, which is generated by the dimensional analysis module.

**Solution**: Run dimensional analysis with the `--save-normalized-lg` flag first:
```bash
python analyze_dimensions.py --config pydimension/configs/config_synthetic.json --save-normalized-lg
```

Then run dimensional filtering to get the suggested count:
```bash
python filter_constraints.py --config pydimension/configs/config_synthetic.json
```

Finally, run optimization and discovery (it will automatically use the suggested count):
```bash
python optimize_discovery.py --config pydimension/configs/config_synthetic.json
```

Alternatively, you can specify the input file explicitly:
```bash
python optimize_discovery.py --input_file path/to/normalized_lg_afterDA_data.csv
```

### Warning: "No suggested count file found, using default num_linear = 1"

The optimization discovery module tries to automatically load the suggested dominant count from dimensional filtering results. If this file is not found, it defaults to `num_linear = 1`.

**Solution**: Run dimensional filtering first to generate `output/results/suggested_dominant_count.json`:
```bash
python filter_constraints.py --config pydimension/configs/config_synthetic.json
```

Or explicitly set `num_linear` if you know the value:
```bash
python optimize_discovery.py --config config_synthetic.json --num_linear 2
```

### Warning: "Models not converging"

If training loss doesn't decrease:
- Increase `epochs`
- Adjust `learning_rate` (try 0.0001 or 0.01)
- Check that input data is properly normalized
- Verify that `num_linear` matches the suggested dominant count from dimensional filtering

### High Prediction Uncertainty

If ensemble uncertainty is high:
- Increase `num_ensembles` (more models = better uncertainty estimate)
- Check for data quality issues
- Verify that the relationship is learnable (may need more data)

## See Also

- Main package README: `../../README.md`
- Data Generation Module: `../data_generation/README.md`
- Data Preprocessing Module: `../data_preprocessing/README.md`
- Dimensional Analysis Module: `../dimensional_analysis/README.md`
- Dimensional Filtering Module: `../constraint_filtering/README.md`
- Config files: `../../configs/README.md`
- Example config: `../../configs/config_synthetic.json`

