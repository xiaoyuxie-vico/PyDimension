"""
Core dimensional filtering functionality using PCA and SIR analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import os
import json

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d

from .config import ConstraintFilteringConfig


class ConstraintFilterer:
    """Dimensional filtering class for PCA and SIR analysis."""
    
    def __init__(self, config: ConstraintFilteringConfig):
        """Initialize filterer with configuration."""
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.input_columns: List[str] = []
        self.output_column: Optional[str] = None
        
        # PCA results
        self.pca_eigenvalues: Optional[np.ndarray] = None
        self.pca_explained_variance_ratio: Optional[np.ndarray] = None
        self.pca_suggested_count: Optional[int] = None
        
        # SIR results
        self.sir_eigenvalues: Optional[np.ndarray] = None
        self.sir_eigenvectors: Optional[np.ndarray] = None
        self.sir_explained_variance: Optional[np.ndarray] = None
        self.sir_projections: Optional[np.ndarray] = None
        self.sir_suggested_count: Optional[int] = None
        
    def load_data(self) -> pd.DataFrame:
        """Load normalized lg afterDA data from CSV file."""
        if self.config.input_file is None:
            raise ValueError("input_file must be specified in config")
        
        input_path = Path(self.config.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_file}")
        
        df = pd.read_csv(input_path)
        print(f"✅ Loaded data from: {input_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Force all columns to numeric, coerce errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for any NaN values introduced by conversion
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            print(f"⚠️ Warning: Non-numeric values converted to NaN in columns: {nan_cols}")
            # Drop rows with NaN
            rows_before = len(df)
            df = df.dropna()
            rows_after = len(df)
            if rows_before != rows_after:
                print(f"   Dropped {rows_before - rows_after} rows with NaN values")
        
        # Assume last column is output
        cols = list(df.columns)
        if len(cols) < 2:
            raise ValueError("Not enough columns in input file (need at least 2)")
        
        self.output_column = cols[-1]
        self.input_columns = cols[:-1]
        self.data = df
        
        print(f"   Input columns: {self.input_columns}")
        print(f"   Output column: {self.output_column}")
        
        return self.data
    
    def run_pca(self) -> Dict[str, Any]:
        """Run Principal Component Analysis on standardized data."""
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data() before run_pca().")
        
        print(f"\n=== Running PCA ===")
        
        # Use all columns (inputs + output)
        X = self.data[self.input_columns + [self.output_column]].values.astype(float)
        
        # Standardize data: center (subtract mean) and scale (divide by std dev)
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_std = np.std(X, axis=0, keepdims=True)
        X_std = np.where(X_std < 1e-10, 1.0, X_std)  # Avoid division by zero
        X_standardized = (X - X_mean) / X_std
        
        print(f"   Data shape: {X_standardized.shape}")
        print(f"   Standardized: mean=0, std=1")
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)
        
        # Eigenvalues of covariance matrix: (S^2)/(n-1)
        n_samples = X_standardized.shape[0]
        eigvals = (S ** 2) / max(n_samples - 1, 1)
        total_var = np.sum(eigvals) if np.sum(eigvals) > 0 else 1.0
        explained = eigvals / total_var
        
        self.pca_eigenvalues = eigvals
        self.pca_explained_variance_ratio = explained
        self.pca_suggested_count = self._suggest_dominant_count(eigvals, threshold=self.config.pca_threshold)
        
        threshold_pct = self.config.pca_threshold * 100
        print(f"   Eigenvalues: {len(eigvals)}")
        print(f"   Suggested dominant count: {self.pca_suggested_count} ({threshold_pct:.1f}% cumulative variance threshold)")
        
        results = {
            'eigenvalues': eigvals.tolist(),
            'explained_variance_ratio': explained.tolist(),
            'cumulative_variance': np.cumsum(explained).tolist(),
            'suggested_dominant_count': self.pca_suggested_count,
            'n_samples': n_samples,
            'n_features': X_standardized.shape[1]
        }
        
        return results
    
    def run_sir(self) -> Dict[str, Any]:
        """Run Sliced Inverse Regression (SIR) analysis."""
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data() before run_sir().")
        
        print(f"\n=== Running SIR ===")
        
        # Get parameters
        n_slices = self.config.n_sir_slices
        n_directions = self.config.n_sir_directions
        
        # Get input and output data
        X = self.data[self.input_columns].values.astype(float)
        y = self.data[self.output_column].values.astype(float)
        
        n_samples, n_features = X.shape
        
        # Validate slices
        if n_slices > n_samples // 2:
            print(f"⚠️ Warning: Too many slices ({n_slices}) for {n_samples} samples. Using {n_samples // 2} slices instead.")
            n_slices = max(2, n_samples // 2)
        
        # Standardize inputs (same as PCA)
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_std = np.std(X, axis=0, keepdims=True)
        X_std = np.where(X_std < 1e-10, 1.0, X_std)
        X_standardized = (X - X_mean) / X_std
        
        print(f"   Data: {n_samples} samples, {n_features} features")
        print(f"   Parameters: {n_slices} slices, {n_directions} directions")
        print(f"   Standardized: mean=0, std=1")
        
        # Step 1: Slice the output into bins
        sorted_indices = np.argsort(y)
        
        # Create slices of approximately equal size
        slice_size = n_samples // n_slices
        slice_means = []
        slice_counts = []
        slice_y_ranges = []
        
        for i in range(n_slices):
            start_idx = i * slice_size
            if i == n_slices - 1:
                # Last slice gets all remaining points
                end_idx = n_samples
            else:
                end_idx = (i + 1) * slice_size
            
            slice_indices = sorted_indices[start_idx:end_idx]
            X_slice = X_standardized[slice_indices]
            y_slice = y[slice_indices]
            
            # Compute mean of inputs in this slice
            slice_mean = np.mean(X_slice, axis=0)
            slice_means.append(slice_mean)
            slice_counts.append(len(slice_indices))
            slice_y_ranges.append((y_slice.min(), y_slice.max()))
        
        slice_means = np.array(slice_means)  # shape: (n_slices, n_features)
        slice_counts = np.array(slice_counts)
        
        # Step 2: Compute covariance matrix of slice means (weighted by slice size)
        weights = slice_counts / n_samples
        weighted_mean = np.sum(slice_means * weights[:, None], axis=0)
        
        # Weighted covariance of slice means
        centered_means = slice_means - weighted_mean
        Sigma_sir = np.zeros((n_features, n_features))
        for i in range(n_slices):
            Sigma_sir += weights[i] * np.outer(centered_means[i], centered_means[i])
        
        # Step 3: Eigendecomposition to find directions
        eigvals, eigvecs = np.linalg.eigh(Sigma_sir)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Keep only positive eigenvalues
        positive_mask = eigvals > 1e-10
        eigvals = eigvals[positive_mask]
        eigvecs = eigvecs[:, positive_mask]
        
        # Compute explained variance
        total_var = np.sum(eigvals) if np.sum(eigvals) > 0 else 1.0
        explained = eigvals / total_var
        
        # Store results
        self.sir_eigenvalues = eigvals
        self.sir_eigenvectors = eigvecs
        self.sir_explained_variance = explained
        self.sir_suggested_count = self._suggest_dominant_count(eigvals, threshold=self.config.sir_threshold)
        
        # Step 4: Project data onto SIR directions
        n_dirs_to_compute = min(n_directions, len(eigvals))
        X_sir = X_standardized @ eigvecs[:, :n_dirs_to_compute]
        self.sir_projections = X_sir
        
        print(f"   Found {len(eigvals)} directions")
        print(f"   Top direction explains {explained[0]*100:.1f}% of variance")
        print(f"   Suggested important directions: {self.sir_suggested_count}")
        
        # Prepare results
        directions_info = []
        for dir_idx in range(min(3, len(eigvals))):
            direction = eigvecs[:, dir_idx]
            coef_dict = {self.input_columns[i]: float(direction[i]) 
                        for i in range(len(self.input_columns))}
            directions_info.append({
                'index': dir_idx + 1,
                'eigenvalue': float(eigvals[dir_idx]),
                'explained_variance': float(explained[dir_idx]),
                'coefficients': coef_dict
            })
        
        results = {
            'eigenvalues': eigvals.tolist(),
            'eigenvectors': eigvecs.tolist(),
            'explained_variance': explained.tolist(),
            'cumulative_variance': np.cumsum(explained).tolist(),
            'suggested_directions': self.sir_suggested_count,
            'n_slices': n_slices,
            'n_samples': n_samples,
            'n_features': n_features,
            'top_directions': directions_info
        }
        
        return results
    
    def _suggest_dominant_count(self, eigvals: np.ndarray, threshold: float = None) -> int:
        """Suggest number of dominant components using configurable cumulative variance threshold."""
        if eigvals is None or len(eigvals) == 0:
            return 0
        if threshold is None:
            threshold = self.config.pca_threshold
        ratios = eigvals / np.sum(eigvals) if np.sum(eigvals) > 0 else np.zeros_like(eigvals)
        cum = np.cumsum(ratios)
        k = int(np.searchsorted(cum, threshold) + 1)
        return max(1, min(k, len(eigvals)))
    
    def process(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the complete dimensional filtering pipeline."""
        if verbose:
            print("=== Dimensional Filtering ===")
        
        # Load data
        self.load_data()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'input_file': str(self.config.input_file),
            'input_columns': self.input_columns,
            'output_column': self.output_column,
            'data_shape': self.data.shape
        }
        
        # Run PCA if requested
        if self.config.run_pca:
            pca_results = self.run_pca()
            results['pca'] = pca_results
        
        # Run SIR if requested
        if self.config.run_sir:
            sir_results = self.run_sir()
            results['sir'] = sir_results
        
        if verbose:
            print("\n=== Dimensional Filtering Complete ===")
        
        return results
    
    def save_results(self) -> Tuple[Path, Optional[Path]]:
        """Save PCA and SIR results to JSON files."""
        # Create output directories
        output_dir = Path(self.config.output_dir)
        results_dir = output_dir / self.config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        
        pca_path = None
        sir_path = None
        
        # Save PCA results
        if self.pca_eigenvalues is not None:
            pca_results = {
                'timestamp': datetime.now().isoformat(),
                'eigenvalues': self.pca_eigenvalues.tolist(),
                'explained_variance_ratio': self.pca_explained_variance_ratio.tolist(),
                'cumulative_variance': np.cumsum(self.pca_explained_variance_ratio).tolist(),
                'suggested_dominant_count': self.pca_suggested_count
            }
            pca_path = results_dir / self.config.pca_results_filename
            with open(pca_path, 'w') as f:
                json.dump(pca_results, f, indent=2)
            print(f"\n=== Files Saved ===")
            print(f"PCA results: {pca_path}")
        
        # Save SIR results
        if self.sir_eigenvalues is not None:
            sir_results = {
                'timestamp': datetime.now().isoformat(),
                'eigenvalues': self.sir_eigenvalues.tolist(),
                'eigenvectors': self.sir_eigenvectors.tolist(),
                'explained_variance': self.sir_explained_variance.tolist(),
                'cumulative_variance': np.cumsum(self.sir_explained_variance).tolist(),
                'suggested_directions': self.sir_suggested_count,
                'n_slices': self.config.n_sir_slices,
                'n_directions': self.config.n_sir_directions
            }
            sir_path = results_dir / self.config.sir_results_filename
            with open(sir_path, 'w') as f:
                json.dump(sir_results, f, indent=2)
            print(f"SIR results: {sir_path}")
        
        return pca_path, sir_path
    
    def save_suggested_count(self) -> Optional[Path]:
        """Save suggested dominant count to JSON file for use by optimization discovery module."""
        # Determine the suggested count (prefer SIR if available, else PCA)
        suggested_count = None
        method = None
        
        if self.sir_suggested_count is not None:
            suggested_count = self.sir_suggested_count
            method = 'SIR'
        elif self.pca_suggested_count is not None:
            suggested_count = self.pca_suggested_count
            method = 'PCA'
        
        if suggested_count is None:
            print("⚠️ No suggested count available (neither PCA nor SIR was run)")
            return None
        
        # Create output directories
        output_dir = Path(self.config.output_dir)
        results_dir = output_dir / self.config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        suggested_count_data = {
            'timestamp': datetime.now().isoformat(),
            'suggested_dominant_count': suggested_count,
            'method': method,
            'note': 'This value should be used as num_linear in optimization_discovery module'
        }
        
        # Add details based on method
        if method == 'PCA' and self.pca_suggested_count is not None:
            suggested_count_data['pca_details'] = {
                'suggested_count': self.pca_suggested_count,
                'explained_variance_ratio': self.pca_explained_variance_ratio.tolist() if self.pca_explained_variance_ratio is not None else None,
                'cumulative_variance': np.cumsum(self.pca_explained_variance_ratio).tolist() if self.pca_explained_variance_ratio is not None else None
            }
        if method == 'SIR' and self.sir_suggested_count is not None:
            suggested_count_data['sir_details'] = {
                'suggested_count': self.sir_suggested_count,
                'explained_variance': self.sir_explained_variance.tolist() if self.sir_explained_variance is not None else None,
                'cumulative_variance': np.cumsum(self.sir_explained_variance).tolist() if self.sir_explained_variance is not None else None
            }
        
        # Save to file
        count_path = results_dir / self.config.suggested_count_filename
        with open(count_path, 'w') as f:
            json.dump(suggested_count_data, f, indent=2)
        
        print(f"Suggested dominant count ({method}): {suggested_count}")
        print(f"Saved to: {count_path}")
        
        return count_path
    
    def create_visualization(self, output_dir: Optional[str] = None,
                            filename: Optional[str] = None,
                            show: bool = False) -> str:
        """
        Create visualization plots for dimensional filtering results.
        
        Creates plots for:
        1. PCA eigenvalues and explained variance
        2. SIR eigenvalues and explained variance
        3. SIR directions vs output scatter plots
        
        Args:
            output_dir: Base directory to save plots (defaults to config.output_dir)
            filename: Filename for saved plot (default: 'constraint_filtering_plots.png')
            show: Whether to display plots (default: False, saves to file)
        
        Returns:
            Path to saved plot file
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        # Create figures subdirectory
        figures_dir = os.path.join(output_dir, self.config.figures_dir)
        os.makedirs(figures_dir, exist_ok=True)
        
        if filename is None:
            filename = self.config.plot_filename
        
        # Close any existing figures
        plt.close('all')
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        
        # Determine number of subplots needed (only subfigures 2-4: skip PCA Eigenvalues)
        n_plots = 0
        if self.config.run_pca and self.pca_eigenvalues is not None:
            n_plots += 1  # Only explained variance (skip eigenvalues)
        if self.config.run_sir and self.sir_eigenvalues is not None:
            n_plots += 2  # Eigenvalues and explained variance
        
        if n_plots == 0:
            raise ValueError("No analysis results available. Run process() first.")
        
        # Create figure with appropriate size (1 row, 3 columns for subfigures 2-4)
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        plot_idx = 0
        
        # PCA plots (skip eigenvalues, only show explained variance - subfigure 2)
        if self.config.run_pca and self.pca_eigenvalues is not None:
            # PCA Explained Variance (subfigure 2)
            ax2 = fig.add_subplot(gs[0, plot_idx])
            explained = self.pca_explained_variance_ratio
            cum_explained = np.cumsum(explained)
            ax2.bar(range(1, len(explained) + 1), explained * 100, alpha=0.7, 
                   label='Individual', edgecolor='black')
            ax2.plot(range(1, len(cum_explained) + 1), cum_explained * 100, 
                    'ro-', linewidth=2, label='Cumulative')
            threshold_pct = self.config.pca_threshold * 100
            ax2.axhline(y=threshold_pct, color='r', linestyle='--', alpha=0.5, label=f'{threshold_pct:.1f}% threshold')
            ax2.set_xlabel('Component', fontsize=10, fontweight='bold')
            ax2.set_ylabel('Explained Variance (%)', fontsize=10, fontweight='bold')
            ax2.set_title(f'PCA Explained Variance\n(Suggested: {self.pca_suggested_count} components)', 
                         fontsize=11, fontweight='bold')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            plot_idx += 1
        
        # SIR plots
        if self.config.run_sir and self.sir_eigenvalues is not None:
            # SIR Eigenvalues (subfigure 3)
            ax3 = fig.add_subplot(gs[0, plot_idx])
            eigvals = self.sir_eigenvalues
            ax3.bar(range(1, len(eigvals) + 1), eigvals, alpha=0.7, edgecolor='black', color='green')
            ax3.set_xlabel('Direction', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Eigenvalue', fontsize=10, fontweight='bold')
            ax3.set_title('SIR Eigenvalues', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            plot_idx += 1
            
            # SIR Explained Variance (subfigure 4)
            ax4 = fig.add_subplot(gs[0, plot_idx])
            explained = self.sir_explained_variance
            cum_explained = np.cumsum(explained)
            ax4.bar(range(1, len(explained) + 1), explained * 100, alpha=0.7, 
                   label='Individual', edgecolor='black', color='green')
            ax4.plot(range(1, len(cum_explained) + 1), cum_explained * 100, 
                    'ro-', linewidth=2, label='Cumulative')
            threshold_pct = self.config.sir_threshold * 100
            ax4.axhline(y=threshold_pct, color='r', linestyle='--', alpha=0.5, label=f'{threshold_pct:.1f}% threshold')
            ax4.set_xlabel('Direction', fontsize=10, fontweight='bold')
            ax4.set_ylabel('Explained Variance (%)', fontsize=10, fontweight='bold')
            title = f'SIR Explained Variance\n(Suggested: {self.sir_suggested_count} direction(s))'
            if self.sir_suggested_count == 1:
                title += ' *'
            ax4.set_title(title, fontsize=11, fontweight='bold')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Save figure
        plot_path = os.path.join(figures_dir, filename)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close('all')
        
        return plot_path

