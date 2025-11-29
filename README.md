---
title: PyDimension
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
app_file: streamlit_app.py
pinned: false
---

# PyDimension

A modular Python package for discovering **dimensionless relationships** (dimensionless numbers and scaling laws) in physical systems using **data and machine learning**.

## 🚀 Try It Online

**🌐 [Streamlit Web App](https://huggingface.co/spaces/xiaoyuxie-vico/PyDimension)** - Run PyDimension in your browser (no installation required)


> **What is this?** PyDimension takes experimental or synthetic data with physical units and automatically finds a small set of meaningful **dimensionless groups** and **scaling laws** that explain the system.
> It is the production-ready, modular implementation of the method from the Nature Communications paper *“Data-driven discovery of dimensionless numbers and governing laws from scarce measurements”*.


## Relationship to PyDimension v1.0

- **This branch (`main`) – PyDimension**  
  Modern, modular Python package (`pydimension/*`) with command-line tools, Python API, and a Streamlit web app for running full pipelines.

- **Legacy research code (`v1.0` branch) – PyDimension v1.0**  
  Original research notebooks and scripts for PDE discovery and case studies used in the paper (kept for reproducibility).

You can browse the legacy materials on GitHub:

- Legacy README and tutorials: `https://github.com/xiaoyuxie-vico/PyDimension/blob/v1.0/README.md`
- Scaling-law examples: `https://github.com/xiaoyuxie-vico/PyDimension/tree/v1.0/scaling_law`
- PDE discovery examples: `https://github.com/xiaoyuxie-vico/PyDimension/tree/v1.0/PDE_discovery`

## Who is PyDimension for?

- **Experimentalists and engineers** who want to understand which combinations of parameters control their system (e.g., flow, heat transfer, additive manufacturing).
- **Data and ML practitioners** looking for **interpretable, physics-aware models** instead of black-box predictors.
- **Students and researchers** interested in dimensional analysis, scaling laws, and data-driven discovery of governing laws.

If you can provide a table of measurements (inputs with units and outputs), PyDimension can help you:

- Reduce many parameters to a few **dimensionless numbers**.
- Learn **simple scaling laws** between those numbers and outputs.
- Visualize and interpret how your system behaves across conditions.


## Research Paper

**Title:** Data-driven discovery of dimensionless numbers and governing laws from scarce measurements

**Authors:** Xiaoyu Xie, Arash Samaei, Jiachen Guo, Wing Kam Liu, Zhengtao Gan

**Journal:** Nature Communications, Volume 13, Article 7569 (2022)


**Links:**
- [Main manuscript](https://www.nature.com/articles/s41467-022-35084-w) (Nature Communications)
- [Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-35084-w/MediaObjects/41467_2022_35084_MOESM1_ESM.pdf) (PDF)

**Note:** This repository (PyDimension) is a refactored, production-ready implementation of the methodology described in the paper. The original research code and notebooks used in the paper are preserved on the [`v1.0` branch](https://github.com/xiaoyuxie-vico/PyDimension/tree/v1.0).

### Video Talks

- **[CMU Scientific Machine Learning Webinar](https://www.youtube.com/watch?v=b3y4ksYzcig)** (March 30, 2023)  
  Presentation on dimensionless learning methodology and applications

- **[Brown University - Math + Machine Learning + X](https://www.youtube.com/watch?v=R6pJleczQr4&t=55s)** (January 27, 2023)  
  Talk on data-driven discovery of dimensionless numbers and governing laws
  
## Quick Start

### Installation

#### Install as a Python Package (Recommended)

```bash
# Clone the repository
git clone https://github.com/xiaoyuxie-vico/PyDimension.git
cd PyDimension

# Install in editable mode (recommended for development)
pip install -e .

# Or install regularly
pip install .
```

After installation, you can use the package from anywhere:
```bash
# Use command-line tools
pydimension-generate --config pydimension/configs/config_synthetic.json
pydimension-preprocess --config pydimension/configs/config_synthetic.json

# Or use Python API
python -c "from pydimension import DataGenerator; print('✅ Installed successfully')"
```

#### Environment Setup (Alternative)

You can use **Conda**, **Mamba**, or **Micromamba** to create the environment. All three work with the same `environment.yml` file:

**Option 1: Micromamba (Fastest, Recommended)**
```bash
micromamba env create -f environment.yml
micromamba activate pydimension
python test_environment.py
python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot
```

**Option 2: Mamba (Faster than Conda)**
```bash
mamba env create -f environment.yml
conda activate pydimension  # Use conda activate even with mamba
python test_environment.py
python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot
```

**Option 3: Conda (Standard)**
```bash
conda env create -f environment.yml
conda activate pydimension
python test_environment.py
python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot
```

**Option 4: pip (Alternative)**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python test_environment.py
python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot
```

> **Note**: Conda, Mamba, and Micromamba are all compatible package managers. Choose based on your preference:
> - **Micromamba**: Fastest, standalone (no conda installation needed)
> - **Mamba**: Faster drop-in replacement for conda
> - **Conda**: Standard Anaconda/Miniconda package manager

See [docs/SETUP.md](docs/SETUP.md) for detailed instructions and troubleshooting.

### Web Interface

```bash
streamlit run streamlit_app.py
```

See [docs/STREAMLIT_README.md](docs/STREAMLIT_README.md) for details.

## Overview

PyDimension discovers dimensionless scaling laws from experimental or synthetic data through a modular pipeline:

1. **Data Generation** - Generate synthetic datasets with known relationships
2. **Data Preprocessing** - Load, clean, and normalize data
3. **Dimensional Analysis** - Compute basis vectors and dimensionless variables
4. **Dimensional Filtering** - Identify dominant groups via PCA/SIR
5. **Optimization Discovery** - Train neural networks to discover scaling laws

### Key Features

- **Modular Architecture** - Each module is independent and can be used standalone
- **Config-Based** - JSON configuration files for reproducibility
- **Multiple Interfaces** - Command-line, Python API, and web interface
- **Extensible Design** - Easy to add new modules

## Package Structure

```
PyDimension/
├── pydimension/
│   ├── data_generation/          # Module 1: Synthetic data generation
│   ├── data_preprocessing/        # Module 2: Data preprocessing
│   ├── dimensional_analysis/      # Module 3: Dimensional analysis
│   ├── constraint_filtering/      # Module 4: Dimensional filtering
│   ├── optimization_discovery/    # Module 5: Neural network training
│   └── configs/                   # Configuration files
├── generate_data.py              # Convenience scripts
├── preprocess_data.py
├── analyze_dimensions.py
├── filter_constraints.py
├── optimize_discovery.py
├── run_pipeline.py               # Run complete pipeline
└── streamlit_app.py              # Web application
```

## Modules

| Module | Status | Documentation | Quick Start |
|--------|--------|---------------|-------------|
| **Data Generation** | ✅ | [README](pydimension/data_generation/README.md) | `python generate_data.py --config pydimension/configs/config_synthetic.json` |
| **Data Preprocessing** | ✅ | [README](pydimension/data_preprocessing/README.md) | `python preprocess_data.py --config pydimension/configs/config_synthetic.json` |
| **Dimensional Analysis** | ✅ | [README](pydimension/dimensional_analysis/README.md) | `python analyze_dimensions.py --config pydimension/configs/config_synthetic.json` |
| **Dimensional Filtering** | ✅ | [README](pydimension/constraint_filtering/README.md) | `python filter_constraints.py --config pydimension/configs/config_synthetic.json` |
| **Optimization Discovery** | ✅ | [README](pydimension/optimization_discovery/README.md) | `python optimize_discovery.py --config pydimension/configs/config_synthetic.json --plot` |

## Workflow

**Note**: Optimization Discovery automatically uses the suggested dominant count from Dimensional Filtering. Run Dimensional Filtering first.

### Automated Pipeline (Recommended)

```bash
# Run all modules with visualization
python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot

# Without visualization
python run_pipeline.py --config pydimension/configs/config_synthetic.json

# Custom output directory
python run_pipeline.py --config pydimension/configs/config_synthetic.json --output_dir my_output --plot

# Continue on error
python run_pipeline.py --config pydimension/configs/config_synthetic.json --continue-on-error

# Skip specific steps
python run_pipeline.py --config pydimension/configs/config_synthetic.json --skip data_generation --skip data_preprocessing

# Stop after a specific step
python run_pipeline.py --config pydimension/configs/config_synthetic.json --stop-after dimensional_analysis
```

This runs all 5 modules sequentially with automatic data flow between steps.

### Manual Step-by-Step

#### Synthetic Data Example

```bash
# Step 1: Generate data
python generate_data.py --config pydimension/configs/config_synthetic.json --plot

# Step 2: Preprocess data
python preprocess_data.py --config pydimension/configs/config_synthetic.json --plot

# Step 3: Dimensional analysis (normalized lg data saved by default)
python analyze_dimensions.py --config pydimension/configs/config_synthetic.json --plot

# Step 4: Dimensional filtering (outputs suggested_dominant_count.json)
python filter_constraints.py --config pydimension/configs/config_synthetic.json --plot

# Step 5: Optimization discovery (automatically uses suggested count from step 4)
python optimize_discovery.py --config pydimension/configs/config_synthetic.json --plot
```

#### Keyhole Problem Example

```bash
# Step 1: Preprocess data (using keyhole dataset)
python preprocess_data.py --config pydimension/configs/config_keyhole.json --plot

# Step 2: Dimensional analysis
python analyze_dimensions.py --config pydimension/configs/config_keyhole.json --plot

# Step 3: Dimensional filtering
python filter_constraints.py --config pydimension/configs/config_keyhole.json --plot

# Step 4: Optimization discovery
python optimize_discovery.py --config pydimension/configs/config_keyhole.json --plot
```

## Configuration

All modules use a unified JSON configuration file (`pydimension/configs/config_synthetic.json`):

```json
{
  "DATA_GENERATION": { ... },
  "DATA_PREPROCESSING": { "enabled": false, ... },
  "DIMENSIONAL_ANALYSIS": { "enabled": false, ... },
  "CONSTRAINT_FILTERING": { "enabled": false, ... },
  "OPTIMIZATION_DISCOVERY": { "enabled": false, ... },
  "OUTPUT": { "output_dir": "output", ... }
}
```

**All configuration files must be in `pydimension/configs/` directory.**

## Interfaces

### Web Interface (Streamlit)

```bash
streamlit run streamlit_app.py
```

Features: Interactive configuration, real-time execution, inline visualization, results viewer.

### Command-Line Interface

```bash
# Run any module
python -m pydimension.module_name --config pydimension/configs/config_synthetic.json

# Or use convenience scripts
python generate_data.py --config pydimension/configs/config_synthetic.json
```

### Python API

```python
from pydimension.data_generation import DataGenerator, DataGenerationConfig

config = DataGenerationConfig.from_json('pydimension/configs/config_synthetic.json')
generator = DataGenerator(config)
generator.generate()
generator.save_datasets()
```

## Documentation

- **[docs/SETUP.md](docs/SETUP.md)** - Installation and setup guide
- **[docs/USAGE.md](docs/USAGE.md)** - How to use the package (CLI, API, scripts)
- **[docs/TESTING.md](docs/TESTING.md)** - Testing guide and verification
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Development workflow and code checking
- **[docs/WORKFLOW.md](docs/WORKFLOW.md)** - Quick push workflow and deployment shortcuts
- **[docs/STREAMLIT_README.md](docs/STREAMLIT_README.md)** - Web interface guide
- **[docs/HUGGINGFACE_DEPLOY.md](docs/HUGGINGFACE_DEPLOY.md)** - Hugging Face Spaces deployment guide
- **Module READMEs**: Detailed docs in `pydimension/*/README.md`
- **Config Files**: [pydimension/configs/README.md](pydimension/configs/README.md)

## Output Structure

```
output/
├── data/              # Generated datasets (CSV)
├── figures/           # Visualizations (PNG)
└── results/          # Analysis results (JSON)
```

## Requirements

- Python 3.7+
- NumPy >= 1.21.0, Pandas >= 1.3.0, SciPy >= 1.7.0
- SymPy >= 1.9.0 (for symbolic simplification)
- Matplotlib >= 3.4.0, Seaborn >= 0.12.0 (for visualization)
- scikit-learn >= 1.0.0 (for scaling and metrics)
- PyTorch >= 1.9.0 (for neural network training)

## Contributing

When adding new modules:
1. Follow the established module structure
2. Create comprehensive documentation (README.md)
3. Provide example configuration files
4. Implement both CLI and Python API
5. Add validation and error handling

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Citations

```

@article{xie2022data,
  title={Data-driven discovery of dimensionless numbers and governing laws from scarce measurements},
  author={Xie, Xiaoyu and Samaei, Arash and Guo, Jiachen and Liu, Wing Kam and Gan, Zhengtao},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={1--11},
  year={2022},
  publisher={Nature Publishing Group}
}
```

## Contact

If you have any questions, suggestions, or would like to contribute to or collaborate on this repository, please contact:

- Xiaoyu Xie
  - Email: xiaoyuxie.vico@gmail.com
  - Website: xiaoyuxie.top
- Zhengtao Gan
  - Email: Zhengtao.Gan@asu.edu
  - Profile: ASU Profile