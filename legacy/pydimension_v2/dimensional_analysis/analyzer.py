"""
Core dimensional analysis functionality.
"""

import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank
from scipy.linalg import null_space
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import os

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from .config import DimensionalAnalysisConfig


class DimensionalAnalyzer:
    """Dimensional analysis class for finding basis vectors and creating dimensionless variables."""
    
    def __init__(self, config: DimensionalAnalysisConfig):
        """Initialize analyzer with configuration."""
        self.config = config
        self.normalized_data: Optional[pd.DataFrame] = None
        self.dimension_matrix: Optional[np.ndarray] = None
        self.dimension_names: List[str] = []
        self.input_variables: List[str] = []
        self.output_variable: Optional[str] = None
        self.basis_vectors: Optional[np.ndarray] = None
        self.afterDA_data: Optional[pd.DataFrame] = None
        self.dimensionless_expressions: List[str] = []
        
    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load normalized data and dimension matrix from CSV files."""
        # Load normalized data
        if self.config.normalized_data_file is None:
            raise ValueError("normalized_data_file must be specified in config")
        
        normalized_path = Path(self.config.normalized_data_file)
        if not normalized_path.exists():
            raise FileNotFoundError(f"Normalized data file not found: {self.config.normalized_data_file}")
        
        self.normalized_data = pd.read_csv(normalized_path)
        print(f"✅ Loaded normalized data from: {normalized_path}")
        print(f"   Shape: {self.normalized_data.shape}")
        print(f"   Columns: {list(self.normalized_data.columns)}")
        
        # Simple approach: assume last column is output
        all_columns = list(self.normalized_data.columns)
        self.output_variable = all_columns[-1]
        self.input_variables = all_columns[:-1]
        
        print(f"   Input variables: {self.input_variables}")
        print(f"   Output variable: {self.output_variable}")
        
        # Load dimension matrix
        if self.config.dimension_matrix_file is None:
            raise ValueError("dimension_matrix_file must be specified in config")
        
        dim_path = Path(self.config.dimension_matrix_file)
        if not dim_path.exists():
            raise FileNotFoundError(f"Dimension matrix file not found: {self.config.dimension_matrix_file}")
        
        dim_df = pd.read_csv(dim_path)
        print(f"✅ Loaded dimension matrix from: {dim_path}")
        
        # Ensure cp (specific heat capacity) has correct dimensions if present
        if 'cp' in dim_df.columns and 'Dimension' in dim_df.columns:
            try:
                cp_dims = {
                    'Mass': 0,
                    'Length': 2,
                    'Time': -2,
                    'Temperature': -1,
                    'Current': 0,
                    'Amount': 0,
                    'Luminous': 0,
                }
                for dim_name, val in cp_dims.items():
                    dim_df.loc[dim_df['Dimension'] == dim_name, 'cp'] = val
            except Exception:
                pass
        
        # Extract only input variables from dimension matrix
        available_vars = [var for var in self.input_variables if var in dim_df.columns]
        
        if not available_vars:
            raise ValueError(f"No input variables found in dimension matrix. "
                           f"Input variables: {self.input_variables}, "
                           f"Matrix columns: {list(dim_df.columns)}")
        
        self.input_variables = available_vars
        full_matrix = dim_df[self.input_variables].values
        
        # Remove rows that are all zeros (unused dimensions)
        non_zero_rows = ~np.all(full_matrix == 0, axis=1)
        self.dimension_matrix = full_matrix[non_zero_rows, :]
        self.dimension_names = dim_df['Dimension'].values[non_zero_rows].tolist()
        
        # Calculate rank and expected null space
        rank = matrix_rank(self.dimension_matrix)
        n = len(self.input_variables)
        m = n - rank
        
        print(f"   Active dimensions: {self.dimension_names}")
        print(f"   Matrix shape: {self.dimension_matrix.shape}")
        print(f"   Matrix rank: {rank}")
        print(f"   Variables: {n}")
        print(f"   Expected null space dimension: {m}")
        
        return self.normalized_data, self.dimension_matrix
    
    def find_basis_vectors(self) -> np.ndarray:
        """Find basis vectors of the null space."""
        if self.dimension_matrix is None:
            raise ValueError("Dimension matrix must be loaded first. Call load_data() before find_basis_vectors().")
        
        print(f"\n=== Finding Null Space ===")
        print(f"Dimension matrix shape: {self.dimension_matrix.shape}")
        
        # Get basis vectors of null space
        null_sp = null_space(self.dimension_matrix)
        print(f"Null space shape: {null_sp.shape}")
        
        if null_sp.shape[1] == 0:
            raise ValueError("No null space found. Variables may already be dimensionless or dimension matrix is full rank.")
        
        # Check if null space vectors are all zero
        if np.all(np.abs(null_sp) < 1e-10):
            rank = matrix_rank(self.dimension_matrix)
            raise ValueError(
                f"All basis vectors are zero!\n"
                f"Matrix shape: {self.dimension_matrix.shape}\n"
                f"Matrix rank: {rank}\n"
                f"Variables: {len(self.input_variables)}\n"
                f"Dimensions: {len(self.dimension_names)}\n\n"
                "This means the dimension matrix is FULL RANK!\n"
                "For dimensional analysis, we need rank < number of variables."
            )
        
        # Simplify basis vectors to be sparse with simple components
        print(f"\n=== Simplifying Basis Vectors ===")
        simplified = self.simplify_basis_vectors(null_sp)
        print(f"Simplified basis vectors shape: {simplified.shape}")
        
        # Normalize basis vectors to unit vectors if requested
        if self.config.normalize_basis:
            print(f"\n=== Normalizing Basis Vectors ===")
            normalized = self.normalize_to_unit_vectors(simplified)
            
            # Verify that basis vectors are still in null space
            verification = self.dimension_matrix @ normalized
            max_error = np.max(np.abs(verification))
            print(f"Verification error: {max_error:.2e}")
            
            if max_error > 1e-10:
                print("⚠️ Warning: Simplified vectors not exactly in null space (may be due to normalization)")
            
            self.basis_vectors = normalized
        else:
            self.basis_vectors = simplified
        
        return self.basis_vectors
    
    def simplify_basis_vectors(self, null_sp: np.ndarray) -> np.ndarray:
        """Create sparse basis vectors using SymPy exact rational arithmetic.
        
        This method ensures consistent basis vectors across modules.
        """
        try:
            from sympy import Matrix, ilcm, igcd
        except ImportError:
            print("  ⚠️ Warning: SymPy not available, using original vectors")
            return null_sp
        
        # Convert numpy array to SymPy Matrix and get exact null space
        M = Matrix(self.dimension_matrix)
        nullspace_vectors = M.nullspace()
        
        if not nullspace_vectors:
            return null_sp
        
        # Convert to primitive integer vectors efficiently
        primitive_vectors = []
        for v in nullspace_vectors:
            # Clear denominators: multiply by LCM of denominators
            denominators = [x.as_numer_denom()[1] for x in v if x != 0]
            L = denominators[0] if denominators else 1
            for d in denominators[1:]:
                L = ilcm(L, d)
            
            w = (v * L)
            
            # Make primitive: divide by GCD of all elements
            elements = [abs(int(x)) for x in w if x != 0]
            g = elements[0] if elements else 1
            for elem in elements[1:]:
                g = igcd(g, elem)
            
            if g > 1:
                w = w // g
            
            # Normalize sign (make first non-zero element positive)
            for x in w:
                if x != 0:
                    if x < 0:
                        w = -w
                    break
            
            primitive_vectors.append(np.array([float(x) for x in w]))
        
        return np.column_stack(primitive_vectors)
    
    def normalize_to_unit_vectors(self, basis_vectors: np.ndarray) -> np.ndarray:
        """Normalize each basis vector to unit vector (magnitude = 1).
        
        For each basis vector v, compute: v_normalized = v / ||v||
        where ||v|| is the Euclidean norm (L2 norm) of the vector.
        """
        normalized_vectors = []
        
        for j in range(basis_vectors.shape[1]):
            vec = basis_vectors[:, j].copy()
            
            # Compute Euclidean norm (magnitude)
            norm = np.linalg.norm(vec)
            
            if norm > 1e-10:
                # Normalize to unit vector
                vec = vec / norm
            else:
                print(f"  ⚠️ Warning: Basis vector w{j+1} has zero magnitude")
            
            normalized_vectors.append(vec)
            
            # Print magnitude for verification
            magnitude = np.linalg.norm(vec)
            print(f"  w{j+1} magnitude: {magnitude:.6f}")
        
        return np.column_stack(normalized_vectors)
    
    def create_dimensionless_variables(self) -> pd.DataFrame:
        """Create dimensionless variables using basis vectors."""
        if self.basis_vectors is None:
            raise ValueError("Basis vectors must be computed first. Call find_basis_vectors() before create_dimensionless_variables().")
        
        if self.normalized_data is None:
            raise ValueError("Normalized data must be loaded first. Call load_data() before create_dimensionless_variables().")
        
        # Get input data
        input_data = self.normalized_data[self.input_variables].values
        
        # Number of basis vectors (dimensionless groups)
        m = self.basis_vectors.shape[1]
        
        # Create dimensionless variables: Pi_i = p1^w_i1 * p2^w_i2 * ... * pn^w_in
        dimensionless_data = np.zeros((input_data.shape[0], m))
        self.dimensionless_expressions = []
        
        for i in range(m):
            # Get basis vector
            w = self.basis_vectors[:, i]
            
            # Create expression string
            expr_parts = []
            for j, var in enumerate(self.input_variables):
                if abs(w[j]) > 1e-10:  # Only include non-zero exponents
                    if abs(w[j] - 1.0) < 1e-10:
                        expr_parts.append(f"{var}")
                    elif abs(w[j] + 1.0) < 1e-10:
                        expr_parts.append(f"{var}^(-1)")
                    elif abs(w[j] - round(w[j])) < 1e-10:
                        expr_parts.append(f"{var}^{int(round(w[j]))}")
                    else:
                        expr_parts.append(f"{var}^({w[j]:.3f})")
            
            expression = " × ".join(expr_parts) if expr_parts else "1"
            self.dimensionless_expressions.append(f"π{i+1} = {expression}")
            
            # Calculate dimensionless variable
            # Using log to avoid overflow: log(Pi) = sum(w_j * log(p_j))
            log_pi = np.zeros(input_data.shape[0])
            for j in range(len(self.input_variables)):
                if abs(w[j]) > 1e-10:  # Only include non-zero exponents
                    # Handle zero or negative values
                    data_col = input_data[:, j]
                    data_col = np.maximum(data_col, 1e-10)  # Avoid log(0)
                    log_pi += w[j] * np.log(data_col)
            
            dimensionless_data[:, i] = np.exp(log_pi)
        
        # Round for consistent display and saving
        dimensionless_data = np.round(dimensionless_data, 10)
        
        # Create DataFrame with dimensionless variables
        dim_columns = [f"π{i+1}" for i in range(m)]
        
        # Include output variable
        self.afterDA_data = pd.DataFrame(dimensionless_data, columns=dim_columns)
        self.afterDA_data[self.output_variable] = self.normalized_data[self.output_variable].values
        
        print(f"\n=== Created Dimensionless Variables ===")
        print(f"Number of dimensionless groups: {m}")
        for expr in self.dimensionless_expressions:
            print(f"  {expr}")
        
        return self.afterDA_data
    
    def compute_normalized_lg_pis(self) -> pd.DataFrame:
        """Compute normalized log10 versions: lgπ = log10(π/max(π)), output = output/max(output).
        
        Note: Output is NOT logged, only normalized.
        """
        if self.afterDA_data is None:
            raise ValueError("AfterDA data must be created first. Call create_dimensionless_variables() before compute_normalized_lg_pis().")
        
        # Identify pi columns
        pi_cols = [c for c in self.afterDA_data.columns if c.startswith('π')]
        out_col = self.output_variable if self.output_variable in self.afterDA_data.columns else None
        
        work = self.afterDA_data.copy()
        
        # Step 1: divide by max per π column (avoid 0 with epsilon)
        eps = 1e-12
        for c in pi_cols:
            m = work[c].max()
            m = m if m > eps else eps
            work[c] = work[c] / m
        
        # Step 1.5: normalize output column by its maximum
        if out_col and out_col in work.columns:
            m_out = work[out_col].max()
            m_out = m_out if m_out > eps else eps
            work[out_col] = work[out_col] / m_out
        
        # Step 2: log10 (output is NOT logged)
        for c in pi_cols:
            work[c] = np.log10(np.maximum(work[c], eps))
        
        # Rename to lgπi
        rename_map = {c: f"lg{c}" for c in pi_cols}
        work = work.rename(columns=rename_map)
        
        # Reorder columns: lgπ..., output
        cols = list(rename_map.values()) + ([out_col] if out_col else [])
        return work[cols]
    
    def process(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the complete dimensional analysis pipeline."""
        if verbose:
            print("=== Dimensional Analysis ===")
        
        # Load data
        self.load_data()
        
        # Find basis vectors
        self.find_basis_vectors()
        
        # Create dimensionless variables
        self.create_dimensionless_variables()
        
        # Prepare results
        results = {
            'timestamp': datetime.now().isoformat(),
            'normalized_data_file': str(self.config.normalized_data_file),
            'dimension_matrix_file': str(self.config.dimension_matrix_file),
            'input_variables': self.input_variables,
            'output_variable': self.output_variable,
            'dimension_names': self.dimension_names,
            'matrix_shape': self.dimension_matrix.shape,
            'matrix_rank': matrix_rank(self.dimension_matrix),
            'null_space_dimension': self.basis_vectors.shape[1],
            'basis_vectors': self.basis_vectors.tolist(),
            'dimensionless_expressions': self.dimensionless_expressions,
            'normalize_basis': self.config.normalize_basis,
            'data_shape': self.afterDA_data.shape if self.afterDA_data is not None else None
        }
        
        if verbose:
            print("\n=== Dimensional Analysis Complete ===")
        
        return results
    
    def save_results(self) -> Tuple[Path, Path]:
        """Save afterDA data and basis vectors to files."""
        if self.afterDA_data is None:
            raise ValueError("No afterDA data to save. Run process() first.")
        
        if self.basis_vectors is None:
            raise ValueError("No basis vectors to save. Run process() first.")
        
        # Create output directories
        output_dir = Path(self.config.output_dir)
        data_dir = output_dir / self.config.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save afterDA data
        afterDA_path = data_dir / self.config.afterDA_data_filename
        self.afterDA_data.to_csv(afterDA_path, index=False, float_format='%.10f')
        
        # Save basis vectors
        m = self.basis_vectors.shape[1]
        cols = [f"w{i+1}" for i in range(m)]
        basis_df = pd.DataFrame(self.basis_vectors, index=self.input_variables, columns=cols)
        basis_df.insert(0, 'Variable', basis_df.index)
        basis_df.reset_index(drop=True, inplace=True)
        
        basis_path = data_dir / self.config.basis_vectors_filename
        basis_df.to_csv(basis_path, index=False)
        
        print(f"\n=== Files Saved ===")
        print(f"AfterDA data: {afterDA_path}")
        print(f"Basis vectors: {basis_path}")
        
        return afterDA_path, basis_path
    
    def save_normalized_lg_data(self) -> Path:
        """Save normalized log10 data to file."""
        if self.afterDA_data is None:
            raise ValueError("AfterDA data must be created first. Run process() before save_normalized_lg_data().")
        
        normalized_lg_data = self.compute_normalized_lg_pis()
        
        # Create output directories
        output_dir = Path(self.config.output_dir)
        data_dir = output_dir / self.config.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save normalized lg data
        lg_path = data_dir / self.config.normalized_lg_data_filename
        normalized_lg_data.to_csv(lg_path, index=False, float_format='%.10f')
        
        print(f"Normalized lg data: {lg_path}")
        
        return lg_path
    
    def create_visualization(self, output_dir: Optional[str] = None,
                            filename: Optional[str] = None,
                            show: bool = False) -> str:
        """
        Create visualization plots for dimensional analysis results.
        
        Creates multiple plots:
        1. Dimension matrix heatmap
        2. Basis vectors visualization
        3. Dimensionless variables (π) distributions
        4. π vs output scatter plots
        5. Correlation matrix of π groups
        6. Summary statistics
        
        Args:
            output_dir: Base directory to save plots (defaults to config.output_dir)
            filename: Filename for saved plot (default: 'dimensional_analysis_plots.png')
            show: Whether to display plots (default: False, saves to file)
        
        Returns:
            Path to saved plot file
        """
        if self.afterDA_data is None:
            raise ValueError("No afterDA data available. Run process() first.")
        
        if self.basis_vectors is None:
            raise ValueError("No basis vectors available. Run process() first.")
        
        if output_dir is None:
            output_dir = self.config.output_dir
        
        # Create figures subdirectory
        figures_dir = os.path.join(output_dir, self.config.figures_dir)
        os.makedirs(figures_dir, exist_ok=True)
        
        if filename is None:
            filename = 'dimensional_analysis_plots.png'
        
        # Close any existing figures
        plt.close('all')
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        
        # Calculate layout: 2 rows, 3 columns
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        
        # Get π columns
        pi_cols = [c for c in self.afterDA_data.columns if c.startswith('π')]
        n_pi = len(pi_cols)
        
        # Plot 1: Dimension Matrix Heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        if self.dimension_matrix is not None and len(self.dimension_names) > 0:
            # Create DataFrame for heatmap
            dim_df = pd.DataFrame(
                self.dimension_matrix,
                index=self.dimension_names,
                columns=self.input_variables
            )
            im = ax1.imshow(dim_df.values, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
            ax1.set_xticks(range(len(self.input_variables)))
            ax1.set_yticks(range(len(self.dimension_names)))
            ax1.set_xticklabels(self.input_variables, rotation=45, ha='right', fontsize=8)
            ax1.set_yticklabels(self.dimension_names, fontsize=8)
            ax1.set_title('Dimension Matrix\n(Input Variables)', fontsize=11, fontweight='bold')
            
            # Add values as text
            for i in range(len(self.dimension_names)):
                for j in range(len(self.input_variables)):
                    val = int(self.dimension_matrix[i, j])
                    color = 'white' if abs(val) > 1.5 else 'black'
                    ax1.text(j, i, str(val), ha="center", va="center", 
                            color=color, fontsize=7, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        else:
            ax1.text(0.5, 0.5, 'Dimension matrix\nnot available', 
                    ha='center', va='center', fontsize=10)
            ax1.set_title('Dimension Matrix', fontsize=11, fontweight='bold')
        
        # Plot 2: Basis Vectors Visualization
        ax2 = fig.add_subplot(gs[0, 1])
        if self.basis_vectors is not None:
            # Create DataFrame for basis vectors
            basis_df = pd.DataFrame(
                self.basis_vectors.T,
                columns=self.input_variables,
                index=[f'w{i+1}' for i in range(self.basis_vectors.shape[1])]
            )
            im = ax2.imshow(basis_df.values, cmap='coolwarm', aspect='auto', 
                           vmin=-1, vmax=1, interpolation='nearest')
            ax2.set_xticks(range(len(self.input_variables)))
            ax2.set_yticks(range(basis_df.shape[0]))
            ax2.set_xticklabels(self.input_variables, rotation=45, ha='right', fontsize=8)
            ax2.set_yticklabels(basis_df.index, fontsize=8)
            ax2.set_title('Basis Vectors\n(Null Space)', fontsize=11, fontweight='bold')
            
            # Add values as text
            for i in range(basis_df.shape[0]):
                for j in range(len(self.input_variables)):
                    val = self.basis_vectors[j, i]
                    # Format value
                    if abs(val) < 1e-10:
                        text = '0'
                    elif abs(val - round(val)) < 1e-6:
                        text = str(int(round(val)))
                    elif abs(val * 2 - round(val * 2)) < 1e-6:
                        text = f'{val:.1f}'
                    else:
                        text = f'{val:.2f}'
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax2.text(j, i, text, ha="center", va="center", 
                            color=color, fontsize=7, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        else:
            ax2.text(0.5, 0.5, 'Basis vectors\nnot available', 
                    ha='center', va='center', fontsize=10)
            ax2.set_title('Basis Vectors', fontsize=11, fontweight='bold')
        
        # Plot 3: Dimensionless Variables (π) Distributions
        ax3 = fig.add_subplot(gs[0, 2])
        if n_pi > 0:
            n_show = min(3, n_pi)  # Show first 3 π groups
            for i, pi_col in enumerate(pi_cols[:n_show]):
                data = self.afterDA_data[pi_col]
                ax3.hist(data, bins=20, alpha=0.6, label=pi_col, 
                        edgecolor='black', linewidth=0.5)
            ax3.set_xlabel('Value', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax3.set_title('Dimensionless Variables\n(π Distributions)', 
                        fontsize=11, fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No dimensionless\nvariables available', 
                    ha='center', va='center', fontsize=10)
            ax3.set_title('Dimensionless Variables', fontsize=11, fontweight='bold')
        
        # Plot 4: π vs Output Scatter
        ax4 = fig.add_subplot(gs[1, 0])
        if n_pi > 0 and self.output_variable in self.afterDA_data.columns:
            output_data = self.afterDA_data[self.output_variable]
            n_show = min(3, n_pi)  # Show first 3 π groups
            for i, pi_col in enumerate(pi_cols[:n_show]):
                pi_data = self.afterDA_data[pi_col]
                ax4.scatter(pi_data, output_data, alpha=0.6, s=20, label=pi_col)
            ax4.set_xlabel('Dimensionless Variable (π)', fontsize=10, fontweight='bold')
            ax4.set_ylabel(f'Output ({self.output_variable})', fontsize=10, fontweight='bold')
            ax4.set_title('π vs Output\n(Scatter Plot)', fontsize=11, fontweight='bold')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No data available\nfor scatter plot', 
                    ha='center', va='center', fontsize=10)
            ax4.set_title('π vs Output', fontsize=11, fontweight='bold')
        
        # Plot 5: Correlation Matrix of π Groups
        ax5 = fig.add_subplot(gs[1, 1])
        if n_pi > 1:
            pi_data = self.afterDA_data[pi_cols]
            corr_matrix = pi_data.corr()
            im = ax5.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax5.set_xticks(range(len(pi_cols)))
            ax5.set_yticks(range(len(pi_cols)))
            ax5.set_xticklabels(pi_cols, rotation=45, ha='right', fontsize=8)
            ax5.set_yticklabels(pi_cols, fontsize=8)
            ax5.set_title('Correlation Matrix\n(π Groups)', fontsize=11, fontweight='bold')
            
            # Add correlation values as text
            for i in range(len(pi_cols)):
                for j in range(len(pi_cols)):
                    text = ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=7)
            
            # Add colorbar
            plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
        elif n_pi == 1:
            ax5.text(0.5, 0.5, 'Only 1 π group\n(no correlation)', 
                    ha='center', va='center', fontsize=10)
            ax5.set_title('Correlation Matrix\n(π Groups)', fontsize=11, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No π groups\navailable', 
                    ha='center', va='center', fontsize=10)
            ax5.set_title('Correlation Matrix\n(π Groups)', fontsize=11, fontweight='bold')
        
        # Plot 6: Summary Statistics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Create summary statistics
        summary_text = "Dimensional Analysis Summary\n\n"
        summary_text += f"Input Variables: {len(self.input_variables)}\n"
        summary_text += f"Output Variable: {self.output_variable}\n"
        summary_text += f"Active Dimensions: {len(self.dimension_names)}\n"
        
        if self.dimension_matrix is not None:
            rank = matrix_rank(self.dimension_matrix)
            summary_text += f"Matrix Rank: {rank}\n"
            summary_text += f"Null Space Dim: {self.basis_vectors.shape[1] if self.basis_vectors is not None else 0}\n"
        
        summary_text += f"\nDimensionless Groups: {n_pi}\n"
        summary_text += f"Data Shape: {self.afterDA_data.shape}\n"
        summary_text += f"Basis Normalization: {'Enabled' if self.config.normalize_basis else 'Disabled'}\n"
        
        summary_text += f"\nDimensionless Expressions:\n"
        for i, expr in enumerate(self.dimensionless_expressions[:3]):  # Show first 3
            summary_text += f"  {expr}\n"
        if len(self.dimensionless_expressions) > 3:
            summary_text += f"  ... and {len(self.dimensionless_expressions) - 3} more\n"
        
        summary_text += f"\nInput Variables:\n"
        for var in self.input_variables[:5]:  # Show first 5
            summary_text += f"  - {var}\n"
        if len(self.input_variables) > 5:
            summary_text += f"  ... and {len(self.input_variables) - 5} more\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save figure
        plot_path = os.path.join(figures_dir, filename)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close('all')
        
        return plot_path

