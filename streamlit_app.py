#!/usr/bin/env python3
"""
Streamlit Web Application for PyDimension 2.0

A comprehensive web interface for running the complete PyDimension pipeline
with visualization and interactive configuration.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add pydimension to path
sys.path.insert(0, str(Path(__file__).parent))

from pydimension.data_generation import DataGenerator, DataGenerationConfig
from pydimension.data_preprocessing import DataPreprocessor, DataPreprocessingConfig
from pydimension.dimensional_analysis import DimensionalAnalyzer, DimensionalAnalysisConfig
from pydimension.constraint_filtering import ConstraintFilterer, ConstraintFilteringConfig
from pydimension.optimization_discovery import OptimizationDiscoverer, OptimizationDiscoveryConfig

# Page configuration
st.set_page_config(
    page_title="PyDimension 2.0",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = {}
if 'current_step' not in st.session_state:
    st.session_state.current_step = None
if 'config' not in st.session_state:
    st.session_state.config = None

def get_available_configs():
    """Get list of available config files, with config_synthetic.json first."""
    config_dir = Path('pydimension/configs')
    if not config_dir.exists():
        return []
    config_files = list(config_dir.glob('config*.json'))
    # Sort configs, but put config_synthetic.json first
    config_files_sorted = sorted(config_files, key=lambda x: (x.name != 'config_synthetic.json', x.name))
    return [str(f) for f in config_files_sorted]

def load_default_config(config_path: Path = None):
    """Load default configuration from specified path or default."""
    if config_path is None:
        config_path = Path('pydimension/configs/config_synthetic.json')
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load config from {config_path}: {e}")
            return {}
    return {}

def save_config_to_session(config_dict):
    """Save configuration to session state."""
    st.session_state.config = config_dict

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🔬 PyDimension 2.0</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["🏠 Home", "📊 Data Generation", "🔧 Data Preprocessing", "📐 Dimensional Analysis", 
         "🔍 Dimensional Filtering", "🧠 Optimization Discovery", "📁 Results Viewer"]
    )
    
    # Config file selector in sidebar
    st.sidebar.markdown("---")
    st.sidebar.title("⚙️ Configuration")
    available_configs = get_available_configs()
    
    if available_configs:
        # Get config file names for display
        config_names = [Path(c).name for c in available_configs]
        # Find index of config_synthetic.json (default), or use 0
        default_index = 0
        if 'config_synthetic.json' in config_names:
            default_index = config_names.index('config_synthetic.json')
        
        selected_config_name = st.sidebar.selectbox(
            "Select config file",
            options=config_names,
            index=default_index,
            help=(
                "Use a config that matches your dataset when possible. For pre-loaded "
                "example datasets (e.g., keyhole), choose the matching config (such as "
                "config_keyhole.json). For new or synthetic datasets, "
                "`config_synthetic.json` is a safe default starting point."
            ),
        )
        # Find the full path
        selected_config_path = next((c for c in available_configs if Path(c).name == selected_config_name), None)
    else:
        selected_config_path = None
        st.sidebar.warning("No config files found in pydimension/configs/")
    
    # Load selected config
    if selected_config_path:
        default_config = load_default_config(selected_config_path)
        st.sidebar.success(f"Loaded config: {Path(selected_config_path).name}")
    else:
        default_config = load_default_config()
        st.sidebar.info("Using default: config_synthetic.json")
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Generation":
        show_data_generation_page(default_config)
    elif page == "🔧 Data Preprocessing":
        show_data_preprocessing_page(default_config)
    elif page == "📐 Dimensional Analysis":
        show_dimensional_analysis_page(default_config)
    elif page == "🔍 Dimensional Filtering":
        show_constraint_filtering_page(default_config)
    elif page == "🧠 Optimization Discovery":
        show_optimization_discovery_page(default_config)
    elif page == "📁 Results Viewer":
        show_results_viewer_page()

def show_home_page():
    """Display home page with overview."""
    # Show logo centered above the title
    logo_col_left, logo_col_center, logo_col_right = st.columns([1, 2, 1])
    with logo_col_center:
        st.image("docs/media/logo.png", use_container_width=True)
    
    st.markdown("## Welcome to PyDimension 2.0")
    st.markdown("### A Comprehensive Tool for Discovering Dimensionless Relationships")
    
    st.markdown("""
    **PyDimension 2.0** is an end-to-end environment for discovering dimensionless
    relationships and scaling laws from data.
    
    Use the **left-hand navigation** to walk through the pipeline, or follow the
    quick recipes below depending on whether you are working with **synthetic data**
    or **your own experiments**.
    """)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **▶️ Start with Synthetic Data**
        
        1. Open **📊 Data Generation**  
           - Configure the number of variables, samples, and polynomial order  
           - Click **Generate Data** to create a benchmark dataset and dimension matrix
        2. Then run **🔧 Data Preprocessing**, **📐 Dimensional Analysis**,  
           **🔍 Dimensional Filtering**, and **🧠 Optimization Discovery** in order.
        3. Inspect the generated figures and JSON results to see how well the
           discovered groups match the known synthetic law.
        """)
    
    with col2:
        st.markdown("""
        **📂 Start with Your Own Data**
        
        1. Go to **🔧 Data Preprocessing**  
           - Upload your **dataset CSV** and **dimension matrix CSV**  
           - (Use the **Dimensional Analysis** page help to build the dimension matrix)
        2. Run **📐 Dimensional Analysis** to build basis vectors and π-groups.  
        3. Run **🔍 Dimensional Filtering** to select dominant π-groups.  
        4. Finish with **🧠 Optimization Discovery** to train the ensemble model
           and extract candidate scaling laws.
        """)
    
    st.markdown("---")
    
    st.markdown("### 📋 Pipeline at a Glance")
    
    st.markdown("""
    The full workflow consists of **5 modules**:
    
    1. **📊 Data Generation** *(optional)*  
       Create controlled synthetic datasets with known π-groups and target laws.
    
    2. **🔧 Data Preprocessing**
       - Load and clean your dataset (CSV)  
       - Select input & output variables  
       - Attach a **dimension matrix** for each variable  
       - Produce normalized data for downstream steps
    
    3. **📐 Dimensional Analysis**
       - Compute basis vectors of the dimension matrix  
       - Construct π-groups from your original variables  
       - Optionally export normalized log10(π) data
    
    4. **🔍 Dimensional Filtering**  
       - Run PCA / SIR on the π-groups  
       - Suggest the effective number of dominant input π-groups  
       - Reduce dimensionality before optimization
    
    5. **🧠 Optimization Discovery**
       - Train an ensemble of neural networks on the dominant π-groups  
       - Apply gamma regularization to obtain interpretable exponents  
       - Export discovered π-groups and candidate scaling laws, plus diagnostics
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Configuration (Sidebar)")
    
    st.markdown("""
    - Use the **⚙️ Configuration** section in the sidebar to choose a JSON config
      (for example `config_synthetic.json` or a problem-specific config).  
    - Each module reads sensible defaults from that config but you can **override
      parameters interactively** on each page.  
    - Your choices live in the current Streamlit session, so you can experiment
      with settings and rerun modules without editing JSON by hand.
    """)
    
    st.markdown("---")
    
    st.markdown("### 📁 Output & Results")
    
    st.markdown("""
    Every module writes its outputs under the **`output/`** folder:
    
    - **`output/data/`** – cleaned datasets, dimension matrices, π-data, basis vectors  
    - **`output/figures/`** – high-resolution PNG figures for each step  
    - **`output/results/`** – JSON files with numerical results and discovered equations  
    
    Use the **📁 Results Viewer** page to quickly preview and download these files,
    or open them directly in your favorite analysis tools.
    """)
    
    st.markdown("---")
    
    # Contact / credits
    st.markdown("### 📬 Contact")
    st.markdown("""
    For questions, feedback, or collaboration opportunities:
    
    - **Xiaoyu Xie**
      - Email: [xiaoyuxie.vico@gmail.com](mailto:xiaoyuxie.vico@gmail.com)  
      - Website: [xiaoyuxie.top](https://xiaoyuxie.top/)
    
    - **Zhengtao Gan** 
      - Email: [Zhengtao.Gan@asu.edu](mailto:Zhengtao.Gan@asu.edu)  
      - Profile: [ASU Profile](https://search.asu.edu/profile/5142230)
    """)
    
    st.markdown("---")
    
    # Show current status
    st.markdown("### 📊 Current Session Status")
    if st.session_state.pipeline_results:
        completed_steps = list(st.session_state.pipeline_results.keys())
        st.success(f"✅ **{len(completed_steps)} step(s) completed:** {', '.join(completed_steps)}")
        with st.expander("View detailed results"):
            st.json(st.session_state.pipeline_results)
    else:
        st.info("👈 **No analysis run yet.** Use the navigation sidebar to start with any module!")

def show_data_generation_page(default_config):
    """Display data generation page."""
    st.markdown("## 📊 Data Generation")
    st.markdown("Generate synthetic datasets with known dimensionless relationships.")
    
    with st.expander("ℹ️ How to use the Data Generation module", expanded=False):
        st.markdown("""
        ### What this page does
        
        The **Data Generation** module creates a *synthetic* dataset with a known
        underlying dimensionless law. This is useful for:
        
        - Testing that the full PyDimension pipeline works end-to-end  
        - Building intuition about π-groups and discovered scaling laws  
        - Comparing the **true** law to what Optimization Discovery recovers
        
        ### Typical workflow
        
        1. Choose the number of input variables (**N**), samples (**M**), and
           the number of true dimensionless groups (**ndim**).  
        2. Set the polynomial order and coefficients to define the hidden law
           between π-groups and the dimensionless output.  
        3. (Optional) Add noise, discrete variables, or fixed levels to mimic
           more realistic experimental designs.  
        4. Click **Generate Data**. This will:
           - Create a synthetic dataset `dataset_synthetic.csv`  
           - Create a matching `dimension_matrix_synthetic.csv`  
           - Save both under `output/data/` and produce a figure in `output/figures/`.
        5. Move on to **Data Preprocessing** → **Dimensional Analysis** →
           **Dimensional Filtering** → **Optimization Discovery** using the
           generated synthetic data.
        
        ### Tips
        
        - Start with the default settings (N=7, M=100, ndim=1, no noise) to
          verify the workflow, then increase complexity.  
        - Higher polynomial order and more π-groups will generally make the
          discovery task harder but more realistic.  
        - The saved figure on this page shows **p\*** vs the first π-group,
          overlaid with the theoretical curve defined by your coefficients.
        """)
    
    with st.expander("Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            N = st.number_input("Number of input variables (N)", min_value=5, max_value=20, value=default_config.get('DATA_GENERATION', {}).get('N', 7))
            M = st.number_input("Number of datapoints (M)", min_value=10, max_value=10000, value=default_config.get('DATA_GENERATION', {}).get('M', 100))
            ndim = st.number_input("Number of dimensionless groups (ndim)", min_value=1, max_value=5, value=default_config.get('DATA_GENERATION', {}).get('ndim', 1))
            poly_order = st.number_input("Polynomial order", min_value=1, max_value=5, value=default_config.get('DATA_GENERATION', {}).get('poly_order', 1))
        
        with col2:
            random_seed = st.number_input("Random seed", min_value=0, value=default_config.get('DATA_GENERATION', {}).get('random_seed', 32))
            noise_level = st.number_input("Noise level (%)", min_value=0.0, max_value=100.0, value=default_config.get('DATA_GENERATION', {}).get('noise_level', 0.0))
            n_discrete = st.number_input("Number of discrete variables", min_value=0, max_value=N, value=default_config.get('DATA_GENERATION', {}).get('n_discrete', 0))
            n_fix = st.number_input("Fixed values for discrete variables", min_value=2, value=default_config.get('DATA_GENERATION', {}).get('n_fix', 5))
        
        coefficients_input = st.text_input("Coefficients (comma-separated)", 
                                          value=",".join(map(str, default_config.get('DATA_GENERATION', {}).get('coefficients', [2.0, 1.0]))))
        coefficients = [float(x.strip()) for x in coefficients_input.split(',') if x.strip()]
    
    output_dir = st.text_input("Output directory", value="output")
    
    if st.button("Generate Data", type="primary"):
        try:
            with st.spinner("Generating data..."):
                # Create config
                config_dict = {
                    'DATA_GENERATION': {
                        'N': N,
                        'M': M,
                        'ndim': ndim,
                        'poly_order': poly_order,
                        'random_seed': random_seed,
                        'noise_level': noise_level,
                        'n_discrete': n_discrete,
                        'n_fix': n_fix,
                        'coefficients': coefficients
                    },
                    'OUTPUT': {
                        'output_dir': output_dir,
                        'data_dir': 'data',
                        'figures_dir': 'figures',
                        'results_dir': 'results'
                    }
                }
                
                config = DataGenerationConfig.from_dict(config_dict)
                generator = DataGenerator(config)
                results = generator.generate(verbose=False)
                
                # Save datasets
                dataset_path, dim_matrix_path = generator.save_datasets()
                
                # Create visualization
                plot_path = generator.create_visualization()
                
                st.success("✅ Data generation completed!")
                st.session_state.pipeline_results['data_generation'] = {
                    'dataset_path': dataset_path,
                    'dimension_matrix_path': dim_matrix_path,
                    'plot_path': plot_path
                }
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Generated Dataset")
                    st.info(f"Saved to: `{dataset_path}`")
                    if Path(dataset_path).exists():
                        df = pd.read_csv(dataset_path)
                        st.dataframe(df.head(10))
                        st.caption(f"Shape: {df.shape}")
                
                with col2:
                    st.markdown("### Dimension Matrix")
                    st.info(f"Saved to: `{dim_matrix_path}`")
                    if Path(dim_matrix_path).exists():
                        dim_df = pd.read_csv(dim_matrix_path)
                        st.dataframe(dim_df)
                
                st.markdown("### Visualization")
                if Path(plot_path).exists():
                    st.image(plot_path)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

def get_available_datasets():
    """Get list of available datasets from dataset folder, with Synthetic (Generated) first."""
    datasets = []
    dataset_dir = Path('dataset')
    
    # First, check for synthetic data in output folder (prioritize this)
    synthetic_path = Path('output/data/dataset_synthetic.csv')
    if synthetic_path.exists():
        datasets.append({
            'name': 'Synthetic (Generated)',
            'dataset_file': str(synthetic_path),
            'dimension_matrix_file': str(Path('output/data/dimension_matrix_synthetic.csv'))
        })
    
    # Then, check for problem-specific folders
    if dataset_dir.exists():
        # Check for problem-specific folders
        for item in dataset_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Look for CSV files in this folder
                csv_files = list(item.glob('*.csv'))
                if csv_files:
                    # Find dataset file (prefer files with 'dataset' in name)
                    dataset_file = None
                    dim_matrix_file = None
                    
                    for csv_file in csv_files:
                        if 'dataset' in csv_file.name.lower():
                            dataset_file = csv_file
                        elif 'dimension' in csv_file.name.lower() or 'matrix' in csv_file.name.lower():
                            dim_matrix_file = csv_file
                    
                    if dataset_file:
                        datasets.append({
                            'name': item.name,
                            'dataset_file': str(dataset_file),
                            'dimension_matrix_file': str(dim_matrix_file) if dim_matrix_file else None
                        })
    
    return datasets

def show_data_preprocessing_page(default_config):
    """Display data preprocessing page."""
    st.markdown("## 🔧 Data Preprocessing")
    st.markdown("Load, clean, and normalize data for analysis.")
    
    with st.expander("ℹ️ How to use the Data Preprocessing module", expanded=False):
        st.markdown("""
        ### What this page does
        
        The **Data Preprocessing** module takes a raw dataset and prepares it for
        dimensional analysis and discovery:
        
        - Selects **input** and **output** variables  
        - Associates each variable with a **dimension matrix** (if provided)  
        - Handles normalization / scaling and saves standardized data files
        
        ### Typical workflow
        
        1. **Choose your data source**
           - Select an existing dataset (e.g., *Synthetic (Generated)* or a folder
             under `dataset/`), **or**
           - Upload your own CSV file(s) for the dataset and dimension matrix.
        
        2. **Attach a dimension matrix**
           - If you generated data using the **Data Generation** module, the
             correct dimension matrix is auto-detected.  
           - For your own experiments, upload a CSV whose columns match the
             dataset variables and whose rows are base dimensions (M, L, T, Θ, ...).
        
        3. **Variable selection**
           - Let PyDimension **auto-detect** input/output variables, *or*  
           - Turn off auto-detect and specify variable names manually
             (e.g., `etaP,Vs,r0,...` for inputs and `e*` for output).
        
        4. **Normalization**
           - Enable **Normalize data** to standardize variables, which is
             recommended for most workflows and required for downstream steps.
        
        5. Click **Preprocess Data**
           - This saves normalized datasets and any intermediate files under
             `output/data/` and generates a figure under `output/figures/`.
        
        ### Tips
        
        - Start by verifying that the preview table and basic statistics look
          reasonable (no obvious unit or scaling mistakes).  
        - Ensure that the **dimension matrix** covers all selected input/output
          variables that will later be used in Dimensional Analysis.  
        - If you see missing variables or mismatched names, fix them here before
          proceeding; later modules assume this mapping is correct.
        """)
    
    # Get available datasets
    available_datasets = get_available_datasets()
    
    # Try to load dataset-specific config
    dataset_config = None
    config_input_vars = None
    config_output_vars = None
    
    with st.expander("Configuration", expanded=True):
        # Data source selection
        data_source = st.radio(
            "Data source",
            options=["Select from available datasets", "Upload your own dataset"],
            help="Choose to use an existing dataset or upload your own CSV files"
        )
        
        input_file = None
        dim_matrix_file = None
        uploaded_dataset_path = None
        uploaded_dim_matrix_path = None
        
        if data_source == "Select from available datasets":
            # Dataset selection from available datasets
            if available_datasets:
                dataset_names = [d['name'] for d in available_datasets]
                # Find index of "Synthetic (Generated)" (default), or use 0
                default_dataset_index = 0
                if 'Synthetic (Generated)' in dataset_names:
                    default_dataset_index = dataset_names.index('Synthetic (Generated)')
                
                selected_dataset = st.selectbox(
                    "Select dataset",
                    options=dataset_names,
                    index=default_dataset_index,
                    help="Choose from available datasets. Default: Synthetic (Generated)"
                )
                
                # Get selected dataset info
                selected_info = next((d for d in available_datasets if d['name'] == selected_dataset), None)
                
                if selected_info:
                    input_file = st.text_input("Input file path", value=selected_info['dataset_file'])
                    
                    # Try to load dataset-specific config (e.g., config_keyhole.json for keyhole dataset)
                    if selected_info['name'] != 'Synthetic (Generated)':
                        config_path = Path(f"pydimension/configs/config_{selected_info['name'].lower()}.json")
                        if config_path.exists():
                            try:
                                with open(config_path, 'r') as f:
                                    dataset_config = json.load(f)
                                config_input_vars = dataset_config.get('DATA_PREPROCESSING', {}).get('input_variables')
                                config_output_vars = dataset_config.get('DATA_PREPROCESSING', {}).get('output_variables')
                                if config_input_vars or config_output_vars:
                                    st.info(f"📋 Loaded variables from config: {config_path.name}")
                            except Exception as e:
                                st.warning(f"Could not load config: {e}")
                    
                    # Auto-fill dimension matrix if available
                    if selected_info['dimension_matrix_file']:
                        dim_matrix_file = st.text_input(
                            "Dimension matrix file", 
                            value=selected_info['dimension_matrix_file'],
                            help="Dimension matrix for the selected dataset"
                        )
                    else:
                        dim_matrix_file = st.text_input(
                            "Dimension matrix file", 
                            value="",
                            help="Path to dimension matrix CSV file (optional, will try to auto-detect)"
                        )
                else:
                    input_file = st.text_input("Input file path", value="output/data/dataset_synthetic.csv")
                    dim_matrix_file = st.text_input("Dimension matrix file", value="")
            else:
                input_file = st.text_input("Input file path", value="output/data/dataset_synthetic.csv")
                dim_matrix_file = st.text_input("Dimension matrix file", value="")
        
        else:  # Upload your own dataset
            st.markdown("**📁 Step 1: Upload Dataset CSV File (Required)**")
            uploaded_file = st.file_uploader(
                "Choose a CSV file for your dataset",
                type=['csv'],
                help="Upload your dataset as a CSV file. The file should have columns for input and output variables.",
                key="dataset_uploader"
            )
            
            if uploaded_file is not None:
                # Create temp directory for uploaded files
                temp_dir = Path('output/temp')
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                # Save uploaded file
                uploaded_dataset_path = temp_dir / f"uploaded_dataset_{uploaded_file.name}"
                with open(uploaded_dataset_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                input_file = str(uploaded_dataset_path)
                st.success(f"✅ Dataset file uploaded: {uploaded_file.name}")
                
                # Try to load and preview the uploaded file
                try:
                    preview_df = pd.read_csv(uploaded_dataset_path)
                    st.info(f"📊 Loaded {len(preview_df)} rows and {len(preview_df.columns)} columns")
                    st.dataframe(preview_df.head(5))
                except Exception as e:
                    st.error(f"Error reading uploaded file: {str(e)}")
            
            st.markdown("**📐 Step 2: Upload Dimension Matrix CSV File (Required)**")
            uploaded_dim_matrix = st.file_uploader(
                "Choose a CSV file for your dimension matrix",
                type=['csv'],
                key="dim_matrix_uploader",
                help="Upload your dimension matrix as a CSV file. This is required for dimensional analysis. The matrix should have dimensions as rows and variables as columns."
            )
            
            if uploaded_dim_matrix is not None:
                # Create temp directory if not exists
                temp_dir = Path('output/temp')
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                # Save uploaded dimension matrix file
                uploaded_dim_matrix_path = temp_dir / f"uploaded_dim_matrix_{uploaded_dim_matrix.name}"
                with open(uploaded_dim_matrix_path, 'wb') as f:
                    f.write(uploaded_dim_matrix.getbuffer())
                
                dim_matrix_file = str(uploaded_dim_matrix_path)
                st.success(f"✅ Dimension matrix uploaded: {uploaded_dim_matrix.name}")
                
                # Try to load and preview the dimension matrix
                try:
                    preview_dim_df = pd.read_csv(uploaded_dim_matrix_path)
                    st.info(f"📐 Dimension matrix: {preview_dim_df.shape[0]} dimensions × {preview_dim_df.shape[1]} variables")
                    st.dataframe(preview_dim_df)
                except Exception as e:
                    st.error(f"Error reading dimension matrix: {str(e)}")
            
            # Show warning if files are missing
            if data_source == "Upload your own dataset":
                if not uploaded_file:
                    st.warning("⚠️ Please upload a dataset CSV file (Step 1)")
                if not uploaded_dim_matrix:
                    st.warning("⚠️ Please upload a dimension matrix CSV file (Step 2)")
                if uploaded_file and uploaded_dim_matrix:
                    st.success("✅ Both required files have been uploaded!")
        
        normalize = st.checkbox(
            "Normalize data",
            value=default_config.get('DATA_PREPROCESSING', {}).get('normalize', True),
            help=(
                "If enabled, each selected variable is divided by its own maximum value "
                "(simple max-scaling so most values fall in [0, 1] if nonnegative). "
                "No mean-centering or standard-deviation (z-score) scaling is applied."
            ),
        )
        
        st.markdown("### Variable Selection")
        
        # Determine default values: use config if available, otherwise use default_config, otherwise auto-detect
        has_config_vars = config_input_vars is not None or config_output_vars is not None
        has_default_vars = (default_config.get('DATA_PREPROCESSING', {}).get('input_variables') is not None or
                           default_config.get('DATA_PREPROCESSING', {}).get('output_variables') is not None)
        
        # Default to auto-detect only if no config variables are available
        auto_detect_default = not (has_config_vars or has_default_vars)
        auto_detect = st.checkbox("Auto-detect variables", value=auto_detect_default,
                                 help="If unchecked, you can specify custom variables")
        
        if not auto_detect:
            # Pre-populate with config variables if available
            default_input_str = ""
            default_output_str = ""
            
            if config_input_vars:
                default_input_str = ",".join(config_input_vars)
            elif default_config.get('DATA_PREPROCESSING', {}).get('input_variables'):
                default_input_str = ",".join(default_config.get('DATA_PREPROCESSING', {}).get('input_variables'))
            
            if config_output_vars:
                default_output_str = ",".join(config_output_vars)
            elif default_config.get('DATA_PREPROCESSING', {}).get('output_variables'):
                default_output_str = ",".join(default_config.get('DATA_PREPROCESSING', {}).get('output_variables'))
            
            input_vars = st.text_input(
                "Input variables (comma-separated)", 
                value=default_input_str,
                help="e.g., etaP,Vs,r0,alpha,rho,cp,Tv-T0,Lv,Tl-T0,Lm"
            )
            output_vars = st.text_input(
                "Output variables (comma-separated)", 
                value=default_output_str,
                help="e.g., e*"
            )
        else:
            input_vars = None
            output_vars = None
    
    # Input Data Preview and Visualization
    if input_file and Path(input_file).exists():
        try:
            with st.expander("📊 Input Data Preview", expanded=False):
                # Load and display data
                input_df = pd.read_csv(input_file)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(input_df))
                with col2:
                    st.metric("Total Columns", len(input_df.columns))
                with col3:
                    st.metric("Memory Usage", f"{input_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                st.markdown("### Data Preview")
                st.dataframe(input_df.head(10), use_container_width=True)
                
                st.markdown("### Basic Statistics")
                st.dataframe(input_df.describe(), use_container_width=True)
                
                # Visualizations
                st.markdown("### Data Visualizations")
                
                # Select columns for visualization (numeric only)
                numeric_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) > 0:
                    # Distribution plots
                    n_cols_to_plot = min(6, len(numeric_cols))
                    cols_to_plot = numeric_cols[:n_cols_to_plot]
                    
                    if len(cols_to_plot) > 0:
                        st.markdown("#### Variable Distributions")
                        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                        axes = axes.flatten()
                        
                        for idx, col in enumerate(cols_to_plot):
                            if idx < len(axes):
                                axes[idx].hist(input_df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                                axes[idx].set_title(f'{col}', fontsize=10)
                                axes[idx].set_xlabel('Value')
                                axes[idx].set_ylabel('Frequency')
                                axes[idx].grid(True, alpha=0.3)
                        
                        # Hide unused subplots
                        for idx in range(len(cols_to_plot), len(axes)):
                            axes[idx].axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Correlation matrix
                    if len(numeric_cols) > 1:
                        st.markdown("#### Correlation Matrix")
                        corr_matrix = input_df[numeric_cols].corr()
                        
                        fig, ax = plt.subplots(figsize=(12, 10))
                        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                                   ax=ax, xticklabels=True, yticklabels=True)
                        ax.set_title('Correlation Matrix of Numeric Variables', fontsize=12, pad=20)
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Pair plot for first few variables
                    if len(numeric_cols) >= 2:
                        st.markdown("#### Pairwise Relationships (First 4 Variables)")
                        pair_cols = numeric_cols[:min(4, len(numeric_cols))]
                        if len(pair_cols) >= 2:
                            pair_df = input_df[pair_cols].dropna()
                            if len(pair_df) > 0:
                                fig = sns.pairplot(pair_df, diag_kind='hist', plot_kws={'alpha': 0.6, 's': 20})
                                st.pyplot(fig.fig)
                                plt.close(fig.fig)
                else:
                    st.info("No numeric columns found for visualization.")
                
        except Exception as e:
            st.warning(f"Could not load input data for preview: {str(e)}")
    elif input_file:
        st.warning(f"⚠️ Input file not found: {input_file}")
    
    output_dir = st.text_input("Output directory", value="output")
    
    if st.button("Preprocess Data", type="primary"):
        # Validate that input file is provided
        if not input_file:
            st.error("❌ Please select a dataset or upload a CSV file first.")
        elif not Path(input_file).exists():
            st.error(f"❌ Input file not found: {input_file}")
        # Validate dimension matrix for uploaded datasets
        elif data_source == "Upload your own dataset" and not dim_matrix_file:
            st.error("❌ Please upload both the dataset file and the dimension matrix file. Both are required for custom datasets.")
        elif data_source == "Upload your own dataset" and dim_matrix_file and not Path(dim_matrix_file).exists():
            st.error(f"❌ Dimension matrix file not found: {dim_matrix_file}")
        else:
            try:
                with st.spinner("Preprocessing data..."):
                    # Parse input/output variables (strip whitespace and filter empty strings)
                    parsed_input_vars = None
                    parsed_output_vars = None
                    
                    if not auto_detect:
                        # User specified variables manually
                        if input_vars and input_vars.strip():
                            parsed_list = [v.strip() for v in input_vars.split(',') if v.strip()]
                            if parsed_list:  # Only set if list is not empty
                                parsed_input_vars = parsed_list
                        if output_vars and output_vars.strip():
                            parsed_list = [v.strip() for v in output_vars.split(',') if v.strip()]
                            if parsed_list:  # Only set if list is not empty
                                parsed_output_vars = parsed_list
                    # If auto_detect is True, parsed_input_vars and parsed_output_vars remain None
                    # which will trigger auto-detection in detect_variables()
                    
                    config_dict = {
                        'DATA_PREPROCESSING': {
                            'enabled': True,
                            'input_file': input_file,
                            'input_variables': parsed_input_vars,  # None triggers auto-detection
                            'output_variables': parsed_output_vars,  # None triggers auto-detection
                            'dimension_matrix_file': dim_matrix_file if dim_matrix_file else None,
                            'normalize': normalize
                        },
                        'OUTPUT': {
                            'output_dir': output_dir,
                            'data_dir': 'data',
                            'figures_dir': 'figures',
                            'results_dir': 'results'
                        }
                    }
                    
                    config = DataPreprocessingConfig.from_dict(config_dict)
                    preprocessor = DataPreprocessor(config)
                    preprocessor.process(verbose=False)
                    preprocessor.save_results()
                    
                    plot_path = preprocessor.create_visualization()
                    
                    st.success("✅ Data preprocessing completed!")
                    st.session_state.pipeline_results['data_preprocessing'] = {
                        'plot_path': plot_path
                    }
                    
                    st.markdown("### Visualization")
                    if Path(plot_path).exists():
                        st.image(plot_path)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

def show_dimensional_analysis_page(default_config):
    """Display dimensional analysis page."""
    st.markdown("## 📐 Dimensional Analysis")
    st.markdown("Compute basis vectors and create dimensionless variables.")
    
    # High-level guidance for this module
    with st.expander("ℹ️ What the Dimensional Analysis module does", expanded=False):
        st.markdown("""
        ### Overview
        
        The **Dimensional Analysis** module takes your **normalized data** and the
        **dimension matrix** and produces:
        
        - Basis vectors spanning the null space of the dimension matrix  
        - Corresponding **π-groups** (dimensionless variables) for each sample  
        - Optional normalized log10(π) data for downstream filtering and discovery
        
        ### Typical workflow
        
        1. Make sure you have already run **Data Preprocessing**, which saves:
           - `normalized_data.csv` under `output/data/`  
           - A matching `dimension_matrix.csv` (or `dimension_matrix_synthetic.csv`)
        2. On this page, point **Normalized data file** and **Dimension matrix file**
           to those outputs (defaults usually work if you followed the pipeline).  
        3. Keep **Normalize basis vectors** checked to scale each basis vector to
           unit length (this often improves interpretability and numerical stability).  
        4. Enable **Save normalized log10 data** if you plan to run
           **Dimensional Filtering** and **Optimization Discovery** afterwards.
        5. Click **Run Dimensional Analysis** to:
           - Compute basis vectors  
           - Build π-groups from your original variables  
           - Save results and a summary figure under `output/`.
        
        ### When to revisit this page
        
        - If you modify the **dimension matrix** (e.g., fix units or add variables),
          rerun Dimensional Analysis so that all π-groups are updated.  
        - If you change which variables are inputs/outputs in **Data Preprocessing**,
          rerun this module to keep the basis and π-groups consistent.
        """)
    
    # Guidance: how to prepare a dimension matrix
    with st.expander("ℹ️ How to prepare the dimension matrix (recommended reading)", expanded=False):
        st.markdown("""
        ### What is a dimension matrix?
        
        A **dimension matrix** encodes the physical dimensions of each variable in your dataset.
        
        - **Rows**: base dimensions (e.g., Mass \\(M\\), Length \\(L\\), Time \\(T\\), Temperature \\(\\Theta\\), etc.)
        - **Columns**: variables in your dataset (e.g., `rho`, `U`, `D`, `mu`, `p`, `T`, ...)
        - **Entries**: integer (or rational) exponents of each base dimension in that variable.
        
        For example, if a variable has dimensions \\(M^a L^b T^c \\Theta^d\\), then in the dimension matrix:
        
        - Row `M` gets value `a`
        - Row `L` gets value `b`
        - Row `T` gets value `c`
        - Row `Θ` gets value `d`
        
        The CSV should look like:
        
        - First column: base-dimension names (e.g., `M`, `L`, `T`, `Theta`)
        - Remaining columns: one column per variable in your dataset (names must match the dataset header).
        """)
        
        st.markdown("""
        ### Example layout (CSV)
        
        ```text
        Dimension,rho,U,D,mu,p,T
        M,1,0,0,1,1,0
        L,-3,1,1,-1,-1,0
        T,0,-1,0,-1,-2,0
        Theta,0,0,0,0,0,1
        ```
        
        - `rho` (density) has \\(M^1 L^{-3} T^0\\)
        - `U` (velocity) has \\(M^0 L^{1} T^{-1}\\)
        - `D` (length / diameter) has \\(L^1\\)
        - `mu` (dynamic viscosity) has \\(M^1 L^{-1} T^{-1}\\)
        - `p` (pressure) has \\(M^1 L^{-1} T^{-2}\\)
        - `T` (temperature) has \\(\\Theta^1\\)
        """)
        
        st.markdown("""
        ### Common base dimensions
        
        - **M**: Mass  
        - **L**: Length  
        - **T**: Time  
        - **Θ**: Temperature  
        - *(Optional for your problem)*: Electric current (I), Amount of substance (N), Luminous intensity (J)
        """)
        
        st.markdown("""
        ### Common variables and their dimensions
        
        | Variable (example name) | Physical meaning           | Dimensions (M, L, T, Θ)          |
        |-------------------------|----------------------------|----------------------------------|
        | `L`, `D`, `r`           | Length / Diameter / Radius | $M^0 L^1 T^0 \\Theta^0$          |
        | `U`, `V`, `c`           | Velocity                   | $M^0 L^1 T^{-1} \\Theta^0$       |
        | `a`                     | Acceleration               | $M^0 L^1 T^{-2} \\Theta^0$       |
        | `rho`                   | Density                    | $M^1 L^{-3} T^0 \\Theta^0$       |
        | `mu`                    | Dynamic viscosity          | $M^1 L^{-1} T^{-1} \\Theta^0$    |
        | `nu`                    | Kinematic viscosity        | $M^0 L^{2} T^{-1} \\Theta^0$     |
        | `p`                     | Pressure                   | $M^1 L^{-1} T^{-2} \\Theta^0$    |
        | `F`                     | Force                      | $M^1 L^{1} T^{-2} \\Theta^0$     |
        | `Q`                     | Volumetric flow rate       | $M^0 L^{3} T^{-1} \\Theta^0$     |
        | `T`, `Temp`             | Temperature                | $M^0 L^0 T^0 \\Theta^1$          |
        
        You can extend this table with your own variables, as long as you express them consistently in terms of base dimensions.
        """)
    
    # Try to find the dimension matrix from preprocessing output
    default_dim_matrix = "output/data/dimension_matrix.csv"
    if not Path(default_dim_matrix).exists():
        # Fallback to synthetic if preprocessing hasn't run
        default_dim_matrix = "output/data/dimension_matrix_synthetic.csv"
    
    with st.expander("Configuration", expanded=True):
        normalized_data_file = st.text_input("Normalized data file", value="output/data/normalized_data.csv")
        dimension_matrix_file = st.text_input("Dimension matrix file", value=default_dim_matrix,
                                             help="Should match the dimension matrix saved by preprocessing")
        normalize_basis = st.checkbox(
            "Normalize basis vectors",
            value=default_config.get('DIMENSIONAL_ANALYSIS', {}).get('normalize_basis', True),
            help=(
                "If enabled, each basis vector in the null space of the dimension matrix "
                "is rescaled to have unit Euclidean length (‖w‖₂ = 1). This keeps all "
                "π-groups on a comparable scale without changing which combinations of "
                "variables are dimensionless."
            ),
        )
        save_normalized_lg = st.checkbox(
            "Save normalized log10 data",
            value=True,
            help=(
                "If enabled, computes log10 of each π-group, normalizes these values, "
                "and saves them to 'normalized_lg_afterDA_data.csv' in output/data/. "
                "This file is used as the input for dimensional filtering and optimization discovery."
            ),
        )
    
    output_dir = st.text_input("Output directory", value="output")
    
    if st.button("Run Dimensional Analysis", type="primary"):
        try:
            with st.spinner("Running dimensional analysis..."):
                config_dict = {
                    'DIMENSIONAL_ANALYSIS': {
                        'enabled': True,
                        'normalized_data_file': normalized_data_file,
                        'dimension_matrix_file': dimension_matrix_file,
                        'normalize_basis': normalize_basis
                    },
                    'OUTPUT': {
                        'output_dir': output_dir,
                        'data_dir': 'data',
                        'figures_dir': 'figures',
                        'results_dir': 'results'
                    }
                }
                
                config = DimensionalAnalysisConfig.from_dict(config_dict)
                analyzer = DimensionalAnalyzer(config)
                analyzer.process(verbose=False)
                analyzer.save_results()
                
                if save_normalized_lg:
                    analyzer.save_normalized_lg_data()
                
                plot_path = analyzer.create_visualization()
                
                st.success("✅ Dimensional analysis completed!")
                st.session_state.pipeline_results['dimensional_analysis'] = {
                    'plot_path': plot_path
                }
                
                st.markdown("### Visualization")
                if Path(plot_path).exists():
                    st.image(plot_path)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

def show_constraint_filtering_page(default_config):
    """Display dimensional filtering page."""
    st.markdown("## 🔍 Dimensional Filtering")
    st.markdown("Identify dominant dimensionless groups using PCA and SIR analysis.")
    
    # High-level guidance for this module
    with st.expander("ℹ️ What the Dimensional Filtering module does", expanded=False):
        st.markdown("""
        ### Overview
        
        The **Dimensional Filtering** module takes the π-groups produced by
        Dimensional Analysis (typically from `normalized_lg_afterDA_data.csv`)
        and identifies how many of them are truly *dominant* for explaining the output.
        
        It uses two complementary methods:
        
        - **Principal Component Analysis (PCA)** – Unsupervised, finds directions of
          maximum variance in the π-space.  
        - **Sliced Inverse Regression (SIR)** – Supervised, finds directions in π-space
          that are most predictive of the output variable.
        
        ### Typical workflow
        
        1. Ensure you have run **Dimensional Analysis** with
           “Save normalized log10 data” enabled, so
           `output/data/normalized_lg_afterDA_data.csv` exists.  
        2. On this page, set **Input file** to that normalized log10 π dataset.  
        3. Choose whether to run **Principal Component Analysis (PCA)**,
           **Sliced Inverse Regression (SIR)**, or both.  
        4. Adjust the PCA and SIR thresholds to control how many dominant directions
           are retained.  
        5. Click **Run Dimensional Filtering** to compute suggested counts and
           generate summary plots and `suggested_dominant_count.json` in `output/results/`.
        
        The suggested dominant count is then used as the default number of input
        π-groups in the **Optimization Discovery** module.
        """)
    
    with st.expander("Configuration", expanded=True):
        input_file = st.text_input("Input file", value="output/data/normalized_lg_afterDA_data.csv")
        run_pca = st.checkbox(
            "Run PCA",
            value=default_config.get('CONSTRAINT_FILTERING', {}).get('run_pca', True),
            help=(
                "Run Principal Component Analysis (PCA) on the π-groups to find directions "
                "of maximum variance and estimate how many independent input π-groups are needed."
            ),
        )
        run_sir = st.checkbox(
            "Run SIR",
            value=default_config.get('CONSTRAINT_FILTERING', {}).get('run_sir', True),
            help=(
                "Run SIR to identify directions in π-space that are most predictive of the "
                "output, providing a supervised estimate of the effective number of dominant "
                "π-groups."
            ),
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if run_pca:
                pca_threshold = st.number_input("PCA threshold", min_value=0.01, max_value=1.0, 
                                               value=float(default_config.get('CONSTRAINT_FILTERING', {}).get('pca_threshold', 0.75)),
                                               step=0.05, help="Cumulative variance threshold for PCA (0.0-1.0)")
            else:
                pca_threshold = default_config.get('CONSTRAINT_FILTERING', {}).get('pca_threshold', 0.75)
        
        with col2:
            if run_sir:
                sir_threshold = st.number_input(
                    "SIR threshold",
                    min_value=0.01,
                    max_value=1.0,
                                               value=float(default_config.get('CONSTRAINT_FILTERING', {}).get('sir_threshold', 0.75)),
                    step=0.05,
                    help=("Cumulative variance threshold for SIR. "
                          "Higher values keep more SIR directions; typical range is 0.7–0.9."),
                )
                n_sir_slices = st.number_input(
                    "SIR slices",
                    min_value=2,
                    max_value=50,
                    value=default_config.get('CONSTRAINT_FILTERING', {}).get('n_sir_slices', 10),
                    help=("Number of slices used by SIR when partitioning the output range. "
                          "More slices can capture finer structure but require more data."),
                )
                n_sir_directions = st.number_input(
                    "SIR directions",
                    min_value=1,
                    max_value=10,
                    value=default_config.get('CONSTRAINT_FILTERING', {}).get('n_sir_directions', 3),
                    help=("Number of SIR directions to retain as candidate dominant "
                          "π-group combinations."),
                )
            else:
                sir_threshold = default_config.get('CONSTRAINT_FILTERING', {}).get('sir_threshold', 0.75)
                n_sir_slices = 10
                n_sir_directions = 3
    
    output_dir = st.text_input("Output directory", value="output")
    
    if st.button("Run Dimensional Filtering", type="primary"):
        try:
            with st.spinner("Running dimensional filtering..."):
                config_dict = {
                    'CONSTRAINT_FILTERING': {
                        'enabled': True,
                        'input_file': input_file,
                        'run_pca': run_pca,
                        'run_sir': run_sir,
                        'pca_threshold': pca_threshold,
                        'sir_threshold': sir_threshold,
                        'n_sir_slices': n_sir_slices if run_sir else 10,
                        'n_sir_directions': n_sir_directions if run_sir else 3
                    },
                    'OUTPUT': {
                        'output_dir': output_dir,
                        'data_dir': 'data',
                        'figures_dir': 'figures',
                        'results_dir': 'results'
                    }
                }
                
                config = ConstraintFilteringConfig.from_dict(config_dict)
                filterer = ConstraintFilterer(config)
                filterer.process(verbose=False)
                filterer.save_results()
                filterer.save_suggested_count()
                
                plot_path = filterer.create_visualization()
                
                st.success("✅ Constraint filtering completed!")
                st.session_state.pipeline_results['constraint_filtering'] = {
                    'plot_path': plot_path
                }
                
                st.markdown("### Visualization")
                if Path(plot_path).exists():
                    st.image(plot_path)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

def show_optimization_discovery_page(default_config):
    """Display optimization discovery page."""
    st.markdown("## 🧠 Optimization Discovery")
    st.markdown("Train neural networks to discover dimensionless scaling laws.")
    
    # Try to load suggested count from constraint filtering
    # Check multiple possible locations for the suggested count file
    suggested_count_paths = [
        Path('output/results/suggested_dominant_count.json'),
        Path('results/suggested_dominant_count.json'),
        Path('suggested_dominant_count.json'),
        Path(default_config.get('OUTPUT', {}).get('output_dir', 'output')) / 'results' / 'suggested_dominant_count.json'
    ]
    
    suggested_count = None
    for path in suggested_count_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    suggested_data = json.load(f)
                    suggested_count = suggested_data.get('suggested_dominant_count')
                    if suggested_count is not None:
                        break
            except Exception:
                continue
    
    # Use suggested count if available, otherwise use config default, otherwise 1
    default_num_linear = suggested_count if suggested_count is not None else default_config.get('OPTIMIZATION_DISCOVERY', {}).get('num_linear', 1)
    
    with st.expander("Configuration", expanded=True):
        # Data / input configuration
        st.markdown("#### Data & Input")
        input_file = st.text_input(
            "Input file",
            value="output/data/normalized_lg_afterDA_data.csv",
            help="Path to the normalized log10 π data produced by Dimensional Analysis.",
        )
        
        # Neural network architecture
        st.markdown("#### Neural Network Architecture")
        col_arch1, col_arch2 = st.columns(2)
        with col_arch1:
            num_linear = st.number_input(
                "Number of linear (γ) nodes",
                min_value=1,
                max_value=10,
                                        value=default_num_linear,
                help="Typically set to the suggested dominant π-count from Dimensional Filtering.",
            )
            if suggested_count is not None:
                st.info(f"📋 Suggested: {suggested_count} input π from dimensional filtering results")
        with col_arch2:
            num_hidden_layers = st.number_input(
                "Hidden layers",
                min_value=0,
                max_value=10,
                value=default_config.get('OPTIMIZATION_DISCOVERY', {}).get('num_hidden_layers', 4),
            )
            num_hidden_nodes = st.number_input(
                "Hidden nodes per layer",
                min_value=1,
                max_value=100,
                value=default_config.get('OPTIMIZATION_DISCOVERY', {}).get('num_hidden_nodes', 10),
            )
        
        # Training hyperparameters
        st.markdown("#### Training Hyperparameters")
        col_train1, col_train2 = st.columns(2)
        with col_train1:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=10000,
                value=default_config.get('OPTIMIZATION_DISCOVERY', {}).get('epochs', 1000),
                help="Total number of training epochs for each ensemble member.",
            )
        with col_train2:
            learning_rate = st.number_input(
                "Learning rate",
                min_value=0.0001,
                max_value=1.0,
                                           value=default_config.get('OPTIMIZATION_DISCOVERY', {}).get('learning_rate', 0.001), 
                format="%.4f",
                help="Learning rate for the Adam optimizer.",
            )
            num_ensembles = st.number_input(
                "Number of ensembles",
                min_value=1,
                max_value=20,
                value=default_config.get('OPTIMIZATION_DISCOVERY', {}).get('num_ensembles', 5),
                help="Number of independently initialized neural networks to train.",
            )
        
        # Regularization settings
        st.markdown("#### Regularization")
        use_gamma_regularization = st.checkbox(
            "Use gamma regularization",
            value=default_config.get('OPTIMIZATION_DISCOVERY', {}).get('use_gamma_regularization', True),
            help="Encourage γ-coefficients to be simple values (integers / half-integers / quarter-integers).",
        )
        
        # Gamma regularization settings (only show if enabled)
        if use_gamma_regularization:
            st.markdown("#### Gamma Regularization Settings")
            col3, col4 = st.columns(2)
            with col3:
                gamma_reg_strength = st.number_input("Gamma regularization strength", 
                                                     min_value=0.0, max_value=1.0, 
                                                     value=float(default_config.get('OPTIMIZATION_DISCOVERY', {}).get('gamma_reg_strength', 0.01)),
                                                     step=0.01, format="%.2f",
                                                     help="Strength of gamma regularization penalty (0.0-1.0)")
            with col4:
                resolution_options = ['integers', 'half-integers', 'quarter-integers']
                default_resolution = default_config.get('OPTIMIZATION_DISCOVERY', {}).get('gamma_reg_resolution', 'half-integers')
                default_index = resolution_options.index(default_resolution) if default_resolution in resolution_options else 1
                gamma_reg_resolution = st.selectbox("Gamma regularization resolution",
                                                   options=resolution_options,
                                                   index=default_index,
                                                   help="Target resolution for gamma values")
        else:
            gamma_reg_strength = default_config.get('OPTIMIZATION_DISCOVERY', {}).get('gamma_reg_strength', 0.01)
            gamma_reg_resolution = default_config.get('OPTIMIZATION_DISCOVERY', {}).get('gamma_reg_resolution', 'half-integers')
    
    output_dir = st.text_input("Output directory", value="output")
    
    if st.button("Run Optimization Discovery", type="primary"):
        try:
            with st.spinner("Training neural networks (this may take a while)..."):
                config_dict = {
                    'OPTIMIZATION_DISCOVERY': {
                        'enabled': True,
                        'input_file': input_file,
                        'num_linear': num_linear,
                        'num_hidden_layers': num_hidden_layers,
                        'num_hidden_nodes': num_hidden_nodes,
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'num_ensembles': num_ensembles,
                        'use_gamma_regularization': use_gamma_regularization,
                        'gamma_reg_strength': gamma_reg_strength if use_gamma_regularization else 0.01,
                        'gamma_reg_resolution': gamma_reg_resolution if use_gamma_regularization else 'half-integers'
                    },
                    'OUTPUT': {
                        'output_dir': output_dir,
                        'data_dir': 'data',
                        'figures_dir': 'figures',
                        'results_dir': 'results'
                    }
                }
                
                config = OptimizationDiscoveryConfig.from_dict(config_dict)
                optimizer = OptimizationDiscoverer(config)
                optimizer.process(verbose=False)
                results_path, results_dir = optimizer.save_results()
                
                plot_path = optimizer.create_visualization()
                
                st.success("✅ Optimization discovery completed!")
                st.session_state.pipeline_results['optimization_discovery'] = {
                    'plot_path': plot_path,
                    'results_path': results_path
                }
                
                # Load results JSON to display information
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    
                    # Display Setup and Configuration Information
                    st.markdown("---")
                    st.markdown("### 📋 Setup and Configuration")
                    with st.expander("View Setup Information", expanded=False):
                        # Config source
                        st.markdown("#### Configuration")
                        config_source = input_file if 'config' in str(input_file) else 'Streamlit inputs'
                        st.text(f"Config source: {config_source}")
                        
                        # Suggested count
                        if suggested_count is not None:
                            st.markdown(f"**Suggested dominant count:** {suggested_count} (from constraint filtering)")
                            st.markdown(f"**Using num_linear:** {num_linear}")
                        
                        # Data loading info
                        st.markdown("#### Data Loading")
                        if hasattr(optimizer, 'data') and optimizer.data is not None:
                            st.text(f"✅ Loaded data from: {input_file}")
                            st.text(f"   Shape: {optimizer.data.shape}")
                            if 'input_columns' in results:
                                st.text(f"   Input columns: {', '.join(results['input_columns'])}")
                            if 'output_column' in results:
                                st.text(f"   Output column: {results['output_column']}")
                        
                        # Basis vectors info
                        if hasattr(optimizer, 'basis_vectors') and optimizer.basis_vectors is not None:
                            st.markdown("#### Basis Vectors")
                            st.text(f"✅ Loaded basis vectors")
                            if hasattr(optimizer, 'original_parameter_names') and optimizer.original_parameter_names:
                                st.text(f"   Original parameters: {', '.join(optimizer.original_parameter_names)}")
                            st.text(f"   Basis vectors shape: {optimizer.basis_vectors.shape}")
                        
                        # Data splits
                        st.markdown("#### Data Splits")
                        if hasattr(optimizer, 'X_train') and optimizer.X_train is not None:
                            train_samples = len(optimizer.X_train)
                            test_samples = len(optimizer.X_test) if hasattr(optimizer, 'X_test') and optimizer.X_test is not None else 0
                            st.text(f"   Training: {train_samples} samples")
                            st.text(f"   Testing: {test_samples} samples")
                            st.text("   Applied StandardScaler to inputs and outputs")
                        
                        # Training configuration
                        st.markdown("#### Training Configuration")
                        st.text(f"Architecture: {num_linear} linear nodes, {num_hidden_layers} hidden layers, {num_hidden_nodes} nodes per layer")
                        st.text(f"Ensemble size: {num_ensembles}")
                        st.text(f"Epochs: {epochs}")
                        st.text(f"Learning rate: {learning_rate}")
                        if use_gamma_regularization:
                            gamma_strength = default_config.get('OPTIMIZATION_DISCOVERY', {}).get('gamma_reg_strength', 0.01)
                            gamma_resolution = default_config.get('OPTIMIZATION_DISCOVERY', {}).get('gamma_reg_resolution', 'half-integers')
                            st.text(f"Gamma regularization: {gamma_strength} ({gamma_resolution})")
                        else:
                            st.text("Gamma regularization: Disabled")
                    
                    # Display Ensemble Evaluation Results
                    st.markdown("### 📊 Ensemble Evaluation")
                    if 'model_performance' in results and results['model_performance']:
                        perf = results['model_performance']
                        if perf.get('model_r2_scores'):
                            r2_scores = perf['model_r2_scores']
                            best_idx = int(np.argmax(r2_scores))
                            best_r2 = r2_scores[best_idx]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Best Model", f"Model {best_idx + 1}", f"R² = {best_r2:.6f}")
                            with col2:
                                if perf.get('final_test_loss'):
                                    best_test_loss = perf['final_test_loss'][best_idx]
                                    st.metric("Best Test Loss", f"{best_test_loss:.6f}")
                            
                            # Show all model R² scores
                            st.markdown("**All Model R² Scores:**")
                            model_scores_df = pd.DataFrame({
                                'Model': [f"Model {i+1}" for i in range(len(r2_scores))],
                                'R² Score': [f"{r2:.6f}" for r2 in r2_scores],
                                'Test Loss': [f"{loss:.6f}" for loss in perf.get('final_test_loss', [0]*len(r2_scores))]
                            })
                            st.dataframe(model_scores_df, use_container_width=True, hide_index=True)
                    
                    # Display Target Equation (if synthetic data)
                    if 'target_equation' in results and results['target_equation']:
                        st.markdown("### 📊 Target Equation (Synthetic Data)")
                        target_eq = results['target_equation']
                        if 'target_equation' in target_eq:
                            st.code(target_eq['target_equation'], language=None)
                    
                    # Display Discovered Equation
                    st.markdown("### 🔍 Discovered Equation")
                    if 'discovered_equation' in results and results['discovered_equation']:
                        discovered_eq = results['discovered_equation']
                        
                        # Display discovered input dimensionless numbers (in terms of basis π)
                        if discovered_eq.get('dimensionless_groups'):
                            st.markdown("#### Discovered Input Dimensionless Numbers (in terms of basis π)")
                            for pi_expr in discovered_eq['dimensionless_groups']:
                                # Round coefficients to 4 decimal places
                                def round_coeffs(match):
                                    num = float(match.group(1))
                                    return f"({round(num, 4)})"
                                formatted_expr = re.sub(r'\(([-+]?\d*\.?\d+)\)', round_coeffs, pi_expr)
                                st.code(formatted_expr, language=None)
                        
                        # Display discovered input dimensionless numbers (in terms of original parameters)
                        if discovered_eq.get('dimensionless_groups_original_params'):
                            st.markdown("#### Discovered Input Dimensionless Numbers (in terms of original parameters)")
                            for pi_expr in discovered_eq['dimensionless_groups_original_params']:
                                st.code(pi_expr, language=None)
                    
                    # Display Gamma Coefficients
                    st.markdown("### 📊 Gamma Coefficients (γ vectors)")
                    if 'learned_gamma_vectors' in results and results['learned_gamma_vectors']:
                        gamma_data = results['learned_gamma_vectors']
                        gamma_vectors = gamma_data.get('gamma_vectors', [])
                        best_model_idx = gamma_data.get('best_model_index', 0)
                        
                        if gamma_vectors:
                            for idx, gamma in enumerate(gamma_vectors):
                                # Round to 4 decimal places
                                gamma_str = ", ".join([f"{round(g, 4)}" for g in gamma])
                                is_best = (idx == best_model_idx)
                                label = f"γ_{idx+1}" + (" (Best Model)" if is_best else "")
                                st.markdown(f"**{label}:**")
                                st.code(f"[{gamma_str}]", language=None)
                                
                                # Show expression if available (round coefficients to 4 decimals)
                                if 'discovered_equation' in results and results['discovered_equation']:
                                    detailed_eqs = results['discovered_equation'].get('detailed_equations', [])
                                    if idx < len(detailed_eqs):
                                        expr = detailed_eqs[idx].get('expression', 'N/A')
                                        # Round coefficients in expression to 4 decimal places
                                        def round_coeffs_expr(match):
                                            num = float(match.group(1))
                                            return f"({round(num, 4)})"
                                        formatted_expr = re.sub(r'\(([-+]?\d*\.?\d+)\)', round_coeffs_expr, expr)
                                        st.text(f"   → {formatted_expr}")
                    
                    # Display Basis Vectors
                    if hasattr(optimizer, 'basis_vectors') and optimizer.basis_vectors is not None:
                        st.markdown("### 🔢 Basis Vectors")
                        if hasattr(optimizer, 'original_parameter_names') and optimizer.original_parameter_names:
                            basis_df = pd.DataFrame(
                                optimizer.basis_vectors,
                                columns=[f"π{i+1}" for i in range(optimizer.basis_vectors.shape[1])],
                                index=optimizer.original_parameter_names
                            )
                        else:
                            basis_df = pd.DataFrame(
                                optimizer.basis_vectors,
                                columns=[f"π{i+1}" for i in range(optimizer.basis_vectors.shape[1])]
                            )
                        st.dataframe(basis_df.style.format("{:.4f}"), use_container_width=True)
                    
                    # Display Gamma Vectors from all models
                    if 'learned_gamma_vectors' in results and results['learned_gamma_vectors']:
                        gamma_data = results['learned_gamma_vectors']
                        all_models_gamma = gamma_data.get('all_models_gamma_vectors', None)
                        
                        if all_models_gamma is not None:
                            st.markdown("### 📊 Gamma Vectors (All Models)")
                            st.info(
                                "Gamma vectors identified by each ensemble model. "
                                "Each row represents one model, and each column represents one gamma vector component."
                            )
                            
                            # Get input column names for better labeling
                            input_cols = results.get('input_columns', [])
                            
                            # Create a list to store all rows
                            gamma_rows = []
                            
                            # Process each model's gamma vectors
                            for model_idx, model_gammas in enumerate(all_models_gamma):
                                # model_gammas is a list of gamma vectors for this model
                                # Each gamma vector is a list of components
                                row_data = {'Model': f'Model {model_idx + 1}'}
                                
                                # Add R² score if available
                                if 'model_performance' in results and results['model_performance']:
                                    perf = results['model_performance']
                                    if perf.get('model_r2_scores') and model_idx < len(perf['model_r2_scores']):
                                        r2_val = perf['model_r2_scores'][model_idx]
                                        row_data['R² Score'] = round(r2_val, 6)
                                
                                # Add each gamma vector's components
                                for gamma_idx, gamma_vector in enumerate(model_gammas):
                                    for comp_idx, component in enumerate(gamma_vector):
                                        # Create column name
                                        if input_cols and comp_idx < len(input_cols):
                                            col_name = f'γ{gamma_idx+1}_{input_cols[comp_idx]}'
                                        else:
                                            col_name = f'γ{gamma_idx+1}_π{comp_idx+1}'
                                        row_data[col_name] = round(component, 4)
                                
                                gamma_rows.append(row_data)
                            
                            # Create DataFrame
                            if gamma_rows:
                                gamma_df = pd.DataFrame(gamma_rows)
                                
                                # Format numeric columns (excluding Model and R² Score)
                                format_dict = {}
                                for col in gamma_df.columns:
                                    if col not in ['Model', 'R² Score']:
                                        format_dict[col] = "{:.4f}"
                                
                                if format_dict:
                                    st.dataframe(gamma_df.style.format(format_dict), use_container_width=True)
                                else:
                                    st.dataframe(gamma_df, use_container_width=True)
                                
                                # Show best model indicator
                                best_model_idx = gamma_data.get('best_model_index', 0)
                                st.caption(f"* Model {best_model_idx + 1} is the best model (highest R² score)")
                    
                except Exception as e:
                    st.warning(f"Could not load detailed results: {e}")
                
                # Display Visualization
                st.markdown("---")
                st.markdown("### 📈 Visualization")
                
                # Neural network architecture schematic
                arch_plot_path = Path(output_dir) / 'figures' / 'optimization_discovery_architecture.png'
                if arch_plot_path.exists():
                    st.markdown("#### Neural Network Architecture")
                    st.caption(
                        "Schematic of the neural network used in Optimization Discovery, "
                        "showing input π-groups, linear (γ) nodes, hidden layers, and the output node."
                    )
                    arch_col_left, arch_col_center, arch_col_right = st.columns([1, 2, 1])
                    with arch_col_center:
                        st.image(str(arch_plot_path), use_container_width=True)
                
                # Display original scale π vs output plot (if available)
                original_scale_pi_plot_path = Path(output_dir) / 'figures' / 'original_scale_pi_vs_output.png'
                if original_scale_pi_plot_path.exists():
                    st.markdown("#### Input π vs Output π (Original Data Scale)")
                    st.info(
                        "**Note:** Data shown in original (unnormalized) scale. "
                        "The input π is calculated from original parameters using the identified power indices "
                        "(from basis vectors and gamma vectors). Training and test data are shown in different colors."
                    )
                    # Center the image using columns
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(str(original_scale_pi_plot_path), width=600)
                elif 'original_scale_pi_plot_path' in results:
                    # Try to load from results JSON if path is stored there
                    stored_path = results.get('original_scale_pi_plot_path')
                    if stored_path and Path(stored_path).exists():
                        st.markdown("#### Input π vs Output π (Original Data Scale)")
                        st.info(
                            "**Note:** Data shown in original (unnormalized) scale. "
                            "The input π is calculated from original parameters using the identified power indices "
                            "(from basis vectors and gamma vectors). Training and test data are shown in different colors."
                        )
                        # Center the image using columns
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.image(stored_path, width=600)
                
                # Display main training plots
                if Path(plot_path).exists():
                    st.markdown("#### Training Loss and Prediction Comparison")
                    st.image(plot_path)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

def show_results_viewer_page():
    """Display results viewer page."""
    st.markdown("## 📁 Results Viewer")
    st.markdown("View outputs from previous pipeline runs.")
    
    output_dir = Path(st.text_input("Output directory", value="output"))
    
    if st.button("Load Results"):
        try:
            # List available files
            data_dir = output_dir / "data"
            figures_dir = output_dir / "figures"
            results_dir = output_dir / "results"
            
            tabs = st.tabs(["Data Files", "Figures", "Results"])
            
            with tabs[0]:
                if data_dir.exists():
                    data_files = list(data_dir.glob("*.csv"))
                    for file in data_files:
                        with st.expander(f"📄 {file.name}"):
                            df = pd.read_csv(file)
                            st.dataframe(df)
                            st.download_button("Download CSV", df.to_csv(index=False), file.name, "text/csv")
                else:
                    st.info("No data files found.")
            
            with tabs[1]:
                if figures_dir.exists():
                    figure_files = list(figures_dir.glob("*.png"))
                    for file in figure_files:
                        st.markdown(f"### {file.name}")
                        st.image(str(file))
                else:
                    st.info("No figure files found.")
            
            with tabs[2]:
                if results_dir.exists():
                    result_files = list(results_dir.glob("*.json"))
                    for file in result_files:
                        with st.expander(f"📄 {file.name}"):
                            with open(file, 'r') as f:
                                data = json.load(f)
                            st.json(data)
                else:
                    st.info("No result files found.")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()

