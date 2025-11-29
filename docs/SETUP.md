# PyDimension 2.0 Setup Guide

Complete guide for installing and setting up PyDimension 2.0.

## Prerequisites

- Python 3.7+ (3.8-3.11 recommended)
- Conda, Mamba, Micromamba, or pip

## Installation Methods

### Method 1: Install as Python Package (Recommended)

**For Development (Editable Install):**
```bash
git clone https://github.com/xiaoyuxie-vico/PyDimension2.0.git
cd PyDimension2.0
pip install -e .  # Changes are immediately available
```

**For Regular Use:**
```bash
pip install .
```

**Verify Installation:**
```bash
python -c "import pydimension; print(pydimension.__version__)"
pydimension-generate --help
```

### Method 2: Conda/Mamba Environment

**Option A: Micromamba (Fastest)**
```bash
micromamba env create -f environment.yml
micromamba activate pydimension2.0
```

**Option B: Mamba**
```bash
mamba env create -f environment.yml
conda activate pydimension2.0
```

**Option C: Conda**
```bash
conda env create -f environment.yml
conda activate pydimension2.0
```

### Method 3: pip with Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Verify Installation

```bash
# Check Python version
python --version

# Verify packages
python -c "import numpy, pandas, scipy, sympy, matplotlib, seaborn, sklearn, torch; print('✅ All packages OK')"

# Test PyDimension
python test_environment.py
```

## Quick Test

```bash
python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot
```

## When to Reinstall

### With Editable Install (`pip install -e .`)

**✅ NO reinstall needed for:**
- Code changes to `.py` files
- Bug fixes and feature additions
- Config file changes

**🔄 Reinstall needed only for:**
- Changes to `setup.py` (package structure, entry points)
- Changes to package structure (adding/removing modules)
- Changes to `__init__.py` exports

```bash
pip install -e . --force-reinstall
```

### With Regular Install (`pip install .`)

**Must reinstall for every change:**
```bash
pip install . --upgrade
```

## Troubleshooting

### Package Manager Differences

- **Micromamba**: Fastest, standalone. Use `micromamba` for all commands.
- **Mamba**: Faster than conda. Use `mamba` for installs, `conda activate` for activation.
- **Conda**: Standard package manager.

**Installation:**
- Micromamba: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
- Mamba: `conda install mamba -n base -c conda-forge`

### Common Issues

**PyTorch Installation:**
```bash
# CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU: https://pytorch.org/get-started/locally/
```

**Import Errors:**
- Ensure you're in the correct environment: `conda activate pydimension2.0`
- Reinstall if needed: `pip install -e . --force-reinstall`

**Command-Line Tools Not Found:**
- Reinstall: `pip install -e . --force-reinstall`
- Check PATH includes Python scripts directory

## Package Versions

Tested with:
- Python: 3.7-3.11
- NumPy >=1.21.0, Pandas >=1.3.0, SciPy >=1.7.0
- SymPy >=1.9.0, Matplotlib >=3.4.0, Seaborn >=0.12.0
- scikit-learn >=1.0.0, PyTorch >=1.9.0

## Next Steps

- [README.md](../README.md) - Project overview
- [USAGE.md](USAGE.md) - How to use the package
- [TESTING.md](TESTING.md) - Testing guide
