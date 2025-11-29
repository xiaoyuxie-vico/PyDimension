# Streamlit Web Application

Interactive web interface for PyDimension 2.0.

## Quick Start

```bash
# Install streamlit (if needed)
pip install streamlit

# Run the app
streamlit run streamlit_app.py
```

App opens at `http://localhost:8501`.

## Features

### Navigation Pages

1. **🏠 Home** - Overview and status
2. **📊 Data Generation** - Generate synthetic datasets
3. **🔧 Data Preprocessing** - Preprocess and normalize data
4. **📐 Dimensional Analysis** - Compute basis vectors
5. **🔍 Dimensional Filtering** - PCA and SIR analysis
6. **🧠 Optimization Discovery** - Neural network training
7. **🚀 Complete Pipeline** - Run all modules sequentially
8. **📁 Results Viewer** - Browse outputs

### Key Features

- **Interactive Configuration** - Configure each module through forms
- **Real-time Execution** - See progress and results immediately
- **Inline Visualization** - View plots directly in the app
- **Dataset Selection** - Choose from available datasets
- **Results Browser** - View all outputs in one place

## Usage

### Running Individual Modules

1. Navigate to the module page (e.g., "📊 Data Generation")
2. Configure parameters in the form
3. Click the run button
4. View results and visualizations

### Running Complete Pipeline

1. Navigate to "🚀 Complete Pipeline"
2. Configure settings (skip data generation if using existing dataset)
3. Select dataset if starting from preprocessing
4. Click "Run Complete Pipeline"
5. Monitor progress and view results

### Using Pre-prepared Datasets

1. Place dataset in `dataset/<problem_name>/`
2. Include `dataset_*.csv` and `dimension_matrix.csv`
3. Select from dropdown in Data Preprocessing or Complete Pipeline pages

## Testing the Application

### Basic Test

```bash
# Launch app
streamlit run streamlit_app.py

# Verify:
# ✅ Browser opens
# ✅ Home page displays
# ✅ All 8 pages accessible via sidebar
```

### Module Tests

**Data Generation:**
1. Navigate to "📊 Data Generation"
2. Set: M=50, N=7, ndim=1
3. Click "Generate Data"
4. Verify: Success message and visualization appear

**Data Preprocessing:**
1. Navigate to "🔧 Data Preprocessing"
2. Select a dataset
3. Click "Preprocess Data"
4. Verify: Input data preview, preprocessing results, visualization

**Complete Pipeline:**
1. Navigate to "🚀 Complete Pipeline"
2. Configure settings
3. Click "Run Complete Pipeline"
4. Verify: All steps complete, results available

## Command-Line Options

```bash
# Custom port
streamlit run streamlit_app.py --server.port 8502

# Headless mode
streamlit run streamlit_app.py --server.headless true

# Dark theme
streamlit run streamlit_app.py --theme.base dark
```

## Troubleshooting

**App won't start:**
```bash
pip install streamlit
streamlit --version
```

**Import errors:**
```bash
# Ensure package is installed
pip install -e .
python -c "import pydimension"
```

**Port already in use:**
```bash
streamlit run streamlit_app.py --server.port 8502
```

## Next Steps

- [USAGE.md](USAGE.md) - Command-line and Python API usage
- [TEST_COMMANDS.md](TEST_COMMANDS.md) - Test commands reference
- [README.md](../README.md) - Project overview
