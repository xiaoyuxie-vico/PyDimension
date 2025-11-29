"""
Core data preprocessing functionality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import os

# Matplotlib setup for non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from .config import DataPreprocessingConfig


class DataPreprocessor:
    """Data preprocessing class for selecting variables, normalizing data, and generating dimension matrices."""
    
    def __init__(self, config: DataPreprocessingConfig):
        """Initialize preprocessor with configuration."""
        self.config = config
        self.original_data: Optional[pd.DataFrame] = None
        self.normalized_data: Optional[pd.DataFrame] = None
        self.dimension_matrix: Dict[str, List[int]] = {}
        self.input_variables: List[str] = []
        self.output_variables: List[str] = []
        self.variable_units: Dict[str, str] = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        if self.config.input_file is None:
            raise ValueError("input_file must be specified in config")
        
        input_path = Path(self.config.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_file}")
        
        self.original_data = pd.read_csv(input_path)
        print(f"✅ Loaded data from: {input_path}")
        print(f"   Shape: {self.original_data.shape}")
        print(f"   Columns: {list(self.original_data.columns)}")
        
        return self.original_data
    
    def detect_variables(self) -> Tuple[List[str], List[str]]:
        """Auto-detect input and output variables from data."""
        if self.original_data is None:
            raise ValueError("Data must be loaded first. Call load_data() before detect_variables().")
        
        # Exclude metadata columns
        exclude_cols = {'case', 'source'}
        all_vars = [col for col in self.original_data.columns if col not in exclude_cols]
        
        # Auto-detect inputs: typically p1-p7 or similar patterns
        if self.config.input_variables is None:
            # Try to find p1-p7 pattern
            input_vars = [v for v in all_vars if v.startswith('p') and v[1:].isdigit()]
            if not input_vars:
                # If no p1-p7 pattern, use all except output candidates
                output_candidates = [v for v in all_vars if v in ['p*', 'e*', 'Ke'] or v.endswith('*')]
                input_vars = [v for v in all_vars if v not in output_candidates]
        else:
            input_vars = [v for v in self.config.input_variables if v in all_vars]
        
        # Auto-detect outputs: typically p*, e*, Ke, or variables ending with *
        if self.config.output_variables is None:
            output_vars = [v for v in all_vars if v in ['p*', 'e*', 'Ke'] or v.endswith('*')]
            if not output_vars:
                # If no obvious output pattern, use last variable as output
                remaining = [v for v in all_vars if v not in input_vars]
                output_vars = [remaining[-1]] if remaining else []
        else:
            output_vars = [v for v in self.config.output_variables if v in all_vars]
        
        self.input_variables = input_vars
        self.output_variables = output_vars
        
        print(f"✅ Detected variables:")
        print(f"   Input variables ({len(input_vars)}): {input_vars}")
        print(f"   Output variables ({len(output_vars)}): {output_vars}")
        
        return input_vars, output_vars
    
    def load_dimension_matrix(self) -> Dict[str, List[int]]:
        """Load dimension matrix from file or generate from units."""
        dimension_names = ['Mass', 'Length', 'Time', 'Temperature', 'Current', 'Amount', 'Luminous']
        
        # Try to load from file first
        if self.config.dimension_matrix_file:
            matrix_path = Path(self.config.dimension_matrix_file)
            if matrix_path.exists():
                return self._load_matrix_from_file(matrix_path, dimension_names)
        
        # Try default locations
        default_paths = [
            Path(self.config.output_dir) / self.config.data_dir / 'dimension_matrix_synthetic.csv',
            Path(self.config.output_dir) / self.config.data_dir / 'dimension_matrix.csv',
            'output/data/dimension_matrix_synthetic.csv',
            'output/data/dimension_matrix.csv',
            'dimension_matrix_synthetic.csv',
            'dimension_matrix.csv'
        ]
        
        for path in default_paths:
            if Path(path).exists():
                print(f"✅ Found dimension matrix at: {path}")
                return self._load_matrix_from_file(Path(path), dimension_names)
        
        # If no matrix file found, generate from units
        print("⚠️ No dimension matrix file found. Generating from units...")
        return self._generate_matrix_from_units(dimension_names)
    
    def _load_matrix_from_file(self, matrix_path: Path, dimension_names: List[str]) -> Dict[str, List[int]]:
        """Load dimension matrix from CSV file."""
        df = pd.read_csv(matrix_path)
        
        # Check if it has 'Dimension' or 'Variable' column
        if 'Dimension' not in df.columns and 'Variable' not in df.columns:
            raise ValueError(f"Dimension matrix file must have 'Dimension' or 'Variable' column: {matrix_path}")
        
        # Standardize to 'Dimension'
        if 'Variable' in df.columns and 'Dimension' not in df.columns:
            df = df.rename(columns={'Variable': 'Dimension'})
        
        all_vars = self.input_variables + self.output_variables
        matrix = {}
        
        for var in all_vars:
            if var in df.columns:
                # Variable is in the matrix
                var_dims = []
                for dim_name in dimension_names:
                    dim_row = df[df['Dimension'] == dim_name]
                    if not dim_row.empty:
                        val = df.loc[df['Dimension'] == dim_name, var].values[0]
                        var_dims.append(int(float(val)))
                    else:
                        var_dims.append(0)
                matrix[var] = var_dims
            else:
                # Variable not in matrix (e.g., output variable p*)
                # Assume dimensionless
                matrix[var] = [0, 0, 0, 0, 0, 0, 0]
                print(f"   {var}: not in matrix, assuming dimensionless")
        
        print(f"✅ Loaded dimension matrix from file")
        return matrix
    
    def _generate_matrix_from_units(self, dimension_names: List[str]) -> Dict[str, List[int]]:
        """Generate dimension matrix from unit strings."""
        if self.config.variable_units is None:
            # Try to infer units from variable names
            self.variable_units = self._infer_units_from_names()
        
        all_vars = self.input_variables + self.output_variables
        matrix = {}
        
        for var in all_vars:
            unit = self.config.variable_units.get(var, 'dimensionless')
            dimensions = self._parse_dimensions(unit)
            matrix[var] = dimensions
        
        print(f"✅ Generated dimension matrix from units")
        return matrix
    
    def _infer_units_from_names(self) -> Dict[str, str]:
        """Infer units from variable names using common patterns."""
        unit_recommendations = {
            'etaP': 'W',
            'Vs': 'm/s',
            'r0': 'm',
            'alpha': 'm²/s',
            'rho': 'kg/m³',
            'cp': 'J/(kg·K)',
            'Tv-T0': 'K',
            'Lv': 'J/kg',
            'Tl-T0': 'K',
            'Lm': 'J/kg',
            'e': 'dimensionless',
            'Ke': 'dimensionless',
            'e*': 'dimensionless',
            'p*': 'dimensionless'
        }
        
        all_vars = self.input_variables + self.output_variables
        units = {}
        
        for var in all_vars:
            # Check exact match first
            if var in unit_recommendations:
                units[var] = unit_recommendations[var]
            # Check if it's a p1-p7 pattern (assume dimensionless for now, or could be inferred)
            elif var.startswith('p') and var[1:].isdigit():
                units[var] = 'dimensionless'  # Default for synthetic data
            # Check if it ends with * (dimensionless output)
            elif var.endswith('*'):
                units[var] = 'dimensionless'
            else:
                units[var] = 'dimensionless'  # Default fallback
        
        return units
    
    def _parse_dimensions(self, unit: str) -> List[int]:
        """Parse unit string to get fundamental dimensions [Mass, Length, Time, Temperature, Current, Amount, Luminous]."""
        dimensions = [0, 0, 0, 0, 0, 0, 0]
        
        unit_lower = unit.lower().replace(' ', '').replace('·', '').replace('⋅', '').replace('*', '')
        
        # Handle dimensionless
        if 'dimensionless' in unit_lower or unit == '1':
            return dimensions
        
        # Quick exact pattern for specific heat capacity: J/(kg·K)
        cp_patterns = [
            'j/(kgk)', 'j/kg/k', 'jkg^-1k^-1', 'jkg-1k-1', 'j/(kg·k)', 'j/(kg*k)'
        ]
        if any(p in unit_lower for p in cp_patterns):
            return [0, 2, -2, -1, 0, 0, 0]
        
        # Mass (kg)
        if 'kg' in unit_lower:
            if '/kg' in unit_lower:
                dimensions[0] = -1
            else:
                dimensions[0] = 1
        
        # Length (m) - be careful not to count 'mol'
        if 'kg/m³' in unit_lower or 'kg/m^3' in unit_lower:
            dimensions[1] = -3
        elif 'm²/s' in unit_lower or 'm^2/s' in unit_lower:
            dimensions[1] = 2
        elif 'm³' in unit_lower or 'm^3' in unit_lower:
            dimensions[1] = 3
        elif 'm²' in unit_lower or 'm^2' in unit_lower:
            dimensions[1] = 2
        elif 'm/s' in unit_lower:
            dimensions[1] = 1
        elif unit_lower == 'm':
            dimensions[1] = 1
        
        # Time (s)
        if '/s²' in unit_lower or '/s^2' in unit_lower:
            dimensions[2] = -2
        elif '/s³' in unit_lower or '/s^3' in unit_lower:
            dimensions[2] = -3
        elif '/s' in unit_lower:
            dimensions[2] = -1
        
        # Temperature (K)
        if '(kg·k)' in unit_lower or '/(kg·k)' in unit_lower:
            dimensions[3] = -1
        elif unit_lower.endswith('k') or 'k)' in unit_lower or unit_lower == 'k':
            dimensions[3] = 1
        
        # Handle Watts (W = J/s = kg⋅m²/s³)
        if 'w' in unit_lower and 'j' not in unit_lower:
            dimensions[0] = 1
            dimensions[1] = 2
            dimensions[2] = -3
        # Handle Joules
        elif 'j/(kg·k)' in unit_lower or 'j/(kg*k)' in unit_lower or 'j/kg/k' in unit_lower or 'j/(kgk)' in unit_lower:
            dimensions[0] = 0
            dimensions[1] = 2
            dimensions[2] = -2
            dimensions[3] = -1
        elif 'j/kg' in unit_lower:
            dimensions[0] = 0
            dimensions[1] = 2
            dimensions[2] = -2
        elif 'j' in unit_lower:
            dimensions[0] = 1
            dimensions[1] = 2
            dimensions[2] = -2
        
        return dimensions
    
    def normalize_data(self) -> pd.DataFrame:
        """Normalize data by dividing by maximum (values ≤ 1)."""
        if self.original_data is None:
            raise ValueError("Data must be loaded first. Call load_data() before normalize_data().")
        
        if not self.input_variables and not self.output_variables:
            raise ValueError("Variables must be selected first. Call detect_variables() or set input_variables/output_variables.")
        
        selected_columns = self.input_variables + self.output_variables
        selected_data = self.original_data[selected_columns].copy()
        
        # Normalize each column by dividing by its maximum value
        self.normalized_data = selected_data.copy()
        for col in selected_columns:
            max_val = selected_data[col].max()
            if max_val != 0:
                self.normalized_data[col] = selected_data[col] / max_val
            else:
                self.normalized_data[col] = 0
        
        print(f"✅ Normalized data")
        print(f"   Shape: {self.normalized_data.shape}")
        print(f"   Value range: [{self.normalized_data.min().min():.4f}, {self.normalized_data.max().max():.4f}]")
        
        return self.normalized_data
    
    def process(self, verbose: bool = True) -> Dict[str, any]:
        """Run the complete preprocessing pipeline."""
        if verbose:
            print("=== Data Preprocessing ===")
        
        # Load data
        self.load_data()
        
        # Detect variables
        self.detect_variables()
        
        # Load or generate dimension matrix
        self.dimension_matrix = self.load_dimension_matrix()
        
        # Normalize data if requested
        if self.config.normalize:
            self.normalize_data()
        else:
            # Use original data without normalization
            selected_columns = self.input_variables + self.output_variables
            self.normalized_data = self.original_data[selected_columns].copy()
        
        # Prepare results
        results = {
            'timestamp': datetime.now().isoformat(),
            'input_file': str(self.config.input_file),
            'input_variables': self.input_variables,
            'output_variables': self.output_variables,
            'dimension_matrix': self.dimension_matrix,
            'variable_units': self.variable_units,
            'normalized': self.config.normalize,
            'data_shape': self.normalized_data.shape if self.normalized_data is not None else None
        }
        
        if verbose:
            print("\n=== Preprocessing Complete ===")
        
        return results
    
    def save_results(self) -> Tuple[Path, Path, Path]:
        """Save normalized data, original data, and dimension matrix to files."""
        if self.normalized_data is None:
            raise ValueError("No normalized data to save. Run process() first.")
        
        if self.original_data is None:
            raise ValueError("No original data to save. Run process() first.")
        
        if not self.dimension_matrix:
            raise ValueError("No dimension matrix to save. Run process() first.")
        
        # Create output directories
        output_dir = Path(self.config.output_dir)
        data_dir = output_dir / self.config.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save normalized data
        normalized_path = data_dir / self.config.normalized_data_filename
        self.normalized_data.to_csv(normalized_path, index=False)
        
        # Save original data (selected columns only)
        selected_columns = self.input_variables + self.output_variables
        original_selected = self.original_data[selected_columns].copy()
        original_path = data_dir / self.config.original_data_filename
        original_selected.to_csv(original_path, index=False)
        
        # Save dimension matrix
        dimension_names = ['Mass', 'Length', 'Time', 'Temperature', 'Current', 'Amount', 'Luminous']
        matrix_data = {'Dimension': dimension_names}
        
        for var in self.input_variables + self.output_variables:
            dimensions = self.dimension_matrix[var]
            matrix_data[var] = dimensions
        
        df_matrix = pd.DataFrame(matrix_data)
        matrix_path = data_dir / self.config.dimension_matrix_filename
        df_matrix.to_csv(matrix_path, index=False)
        
        print(f"\n=== Files Saved ===")
        print(f"Normalized data: {normalized_path}")
        print(f"Original data: {original_path}")
        print(f"Dimension matrix: {matrix_path}")
        
        return normalized_path, original_path, matrix_path
    
    def create_visualization(self, output_dir: Optional[str] = None,
                            filename: Optional[str] = None,
                            show: bool = False) -> str:
        """
        Create visualization plots for the preprocessed data.
        
        Creates two plots:
        1. Correlation matrix
        2. Summary statistics
        
        Args:
            output_dir: Base directory to save plots (defaults to config.output_dir)
            filename: Filename for saved plot (default: 'data_preprocessing_plots.png')
            show: Whether to display plots (default: False, saves to file)
        
        Returns:
            Path to saved plot file
        """
        if self.normalized_data is None:
            raise ValueError("No normalized data available. Run process() first.")
        
        if not self.input_variables and not self.output_variables:
            raise ValueError("No variables selected. Run process() first.")
        
        if output_dir is None:
            output_dir = self.config.output_dir
        
        # Create figures subdirectory
        figures_dir = os.path.join(output_dir, self.config.figures_dir)
        os.makedirs(figures_dir, exist_ok=True)
        
        if filename is None:
            filename = 'data_preprocessing_plots.png'
        
        # Close any existing figures
        plt.close('all')
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        
        # Create figure with 1 row, 2 columns
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
        
        all_vars = self.input_variables + self.output_variables
        n_vars = len(all_vars)
        
        # Plot 1: Correlation Matrix (Input Variables)
        ax1 = fig.add_subplot(gs[0, 0])
        if len(self.input_variables) > 1:
            corr_data = self.normalized_data[self.input_variables]
            corr_matrix = corr_data.corr()
            
            # Use seaborn heatmap for better visualization
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                       ax=ax1, xticklabels=True, yticklabels=True)
            ax1.set_title('Correlation Matrix (Input Variables)', fontsize=12, fontweight='bold', pad=20)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        else:
            ax1.text(0.5, 0.5, 'Need at least 2\ninput variables\nfor correlation', 
                    ha='center', va='center', fontsize=12)
            ax1.set_title('Correlation Matrix (Input Variables)', fontsize=12, fontweight='bold')
        
        # Plot 2: Summary Statistics Table
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        # Create summary statistics
        summary_text = "Summary Statistics\n\n"
        summary_text += f"Total Variables: {n_vars}\n"
        summary_text += f"  - Input: {len(self.input_variables)}\n"
        summary_text += f"  - Output: {len(self.output_variables)}\n"
        summary_text += f"\nData Shape: {self.normalized_data.shape}\n"
        summary_text += f"Normalization: {'Enabled' if self.config.normalize else 'Disabled'}\n"
        
        if self.config.normalize:
            summary_text += f"\nNormalized Range:\n"
            summary_text += f"  Min: {self.normalized_data.min().min():.4f}\n"
            summary_text += f"  Max: {self.normalized_data.max().max():.4f}\n"
        
        summary_text += f"\nInput Variables:\n"
        for var in self.input_variables[:10]:  # Show first 10
            summary_text += f"  - {var}\n"
        if len(self.input_variables) > 10:
            summary_text += f"  ... and {len(self.input_variables) - 10} more\n"
        
        summary_text += f"\nOutput Variables:\n"
        for var in self.output_variables:
            summary_text += f"  - {var}\n"
        
        ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes, fontsize=10,
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

