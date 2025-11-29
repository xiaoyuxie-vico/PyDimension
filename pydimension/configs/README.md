# Configuration Files

**All configuration files for PyDimension should be maintained in this directory** (`pydimension/configs/`).

This directory contains example configuration files for PyDimension modules.

## Unified Config File Structure

The recommended approach is to use a **unified config file** (`config_synthetic.json`) that contains settings for all modules. This allows you to:

- Configure the entire pipeline in one place
- Easily enable/disable modules
- Maintain consistent output directories
- Extend with new modules without breaking existing configs

### Unified Config Format

```json
{
  "DATA_GENERATION": {
    "N": 7,
    "M": 100,
    "ndim": 1,
    "poly_order": 1,
    "random_seed": 32,
    "noise_level": 0.0,
    "coefficients": [2.0, 1.0]
  },
  "DATA_PREPROCESSING": {
    "enabled": false,
    ...
  },
  "DIMENSIONAL_ANALYSIS": {
    "enabled": false,
    ...
  },
  "CONSTRAINT_FILTERING": {
    "enabled": false,
    ...
  },
  "NEURAL_NETWORK_TRAINING": {
    "enabled": false,
    ...
  },
  "OPTIMIZATION_DISCOVERY": {
    "enabled": false
  },
  "OUTPUT": {
    "output_dir": "output",
    "data_dir": "data",
    "figures_dir": "figures",
    "results_dir": "results",
    "logs_dir": "logs"
  },
  "DATA_GENERATION_OUTPUT": {
    "dataset_filename": "dataset_synthetic.csv",
    "dimension_matrix_filename": "dimension_matrix_synthetic.csv",
    "plot_filename": "data_generation_plots.png"
  },
  "DATA_PREPROCESSING_OUTPUT": {
    "normalized_data_filename": "normalized_data.csv",
    "dimension_matrix_filename": "dimension_matrix.csv",
    "plot_filename": "data_preprocessing_plots.png"
  },
  "DIMENSIONAL_ANALYSIS_OUTPUT": {
    "afterDA_data_filename": "afterDA_data.csv",
    "basis_vectors_filename": "basis_vectors.csv",
    "normalized_lg_data_filename": "normalized_lg_afterDA_data.csv",
    "plot_filename": "dimensional_analysis_plots.png"
  },
  "CONSTRAINT_FILTERING_OUTPUT": {
    "pca_results_filename": "pca_results.json",
    "sir_results_filename": "sir_results.json",
    "plot_filename": "constraint_filtering_plots.png",
    "suggested_count_filename": "suggested_dominant_count.json"
  },
  "OPTIMIZATION_DISCOVERY_OUTPUT": {
    "model_results_filename": "optimization_discovery_results.json",
    "plot_filename": "optimization_discovery_plots.png"
  }
}
```

### Config Sections

1. **Module Sections** (e.g., `DATA_GENERATION`, `DATA_PREPROCESSING`, etc.):
   - Each module has its own section
   - Contains module-specific parameters
   - Can have an `enabled` flag to control execution

2. **OUTPUT Section**:
   - Shared output directory structure
   - Used by all modules
   - Defines: `output_dir`, `data_dir`, `figures_dir`, `results_dir`, `logs_dir`

3. **Module-Specific Output Sections** (e.g., `DATA_GENERATION_OUTPUT`):
   - Module-specific filenames
   - Overrides or supplements the OUTPUT section

## Available Config Files

- **`config_synthetic.json`**: Main unified config file with all modules (data generation enabled)
- **`config_synthetic_with_noise.json`**: Unified config with 5% noise enabled

## Usage

### Using Config Files

```bash
# With generate_data.py
python generate_data.py --config pydimension/configs/config_synthetic.json --plot

# With module CLI
python -m pydimension.data_generation --config pydimension/configs/config_synthetic.json --plot
```

## Extending the Config

When adding a new module:

1. Add a new section (e.g., `YOUR_MODULE`) with module-specific parameters
2. Add an `enabled` flag to control execution
3. Optionally add a `YOUR_MODULE_OUTPUT` section for module-specific filenames
4. Update the OUTPUT section if new directories are needed

Example:
```json
{
  "YOUR_MODULE": {
    "enabled": true,
    "param1": value1,
    "param2": value2
  },
  "YOUR_MODULE_OUTPUT": {
    "output_filename": "your_output.csv"
  }
}
```

## Config File Priority

When loading configs, the system checks in this order:
1. Unified OUTPUT section
2. Module-specific OUTPUT section (e.g., DATA_GENERATION_OUTPUT)

This ensures consistent configuration across all modules.

