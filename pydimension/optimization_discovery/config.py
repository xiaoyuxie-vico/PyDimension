"""
Configuration handling for optimization and discovery module.
"""

import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class OptimizationDiscoveryConfig:
    """Configuration for optimization and discovery."""
    
    # Input settings
    input_file: Optional[str] = None  # Path to normalized lg afterDA data CSV
    basis_vectors_file: Optional[str] = None  # Path to basis_vectors.csv (optional)
    
    # Model architecture
    num_linear: int = 1  # Number of linear nodes (gamma vectors)
    num_hidden_layers: int = 4  # Number of hidden layers
    num_hidden_nodes: int = 10  # Number of nodes per hidden layer
    
    # Training settings
    random_seed: int = 49  # Random seed
    epochs: int = 1000  # Training epochs
    learning_rate: float = 0.001  # Learning rate
    train_percent: float = 0.8  # Train/test split ratio
    
    # Ensemble settings
    num_ensembles: int = 5  # Number of ensemble models
    
    # Gamma settings
    use_fixed_gamma: bool = False  # Use fixed gamma values (from basis vectors)
    fixed_gamma_values: Optional[np.ndarray] = None  # Fixed gamma values (shape: num_linear, input_dim)
    use_gamma_regularization: bool = True  # Enable gamma regularization
    gamma_reg_strength: float = 0.01  # Gamma regularization strength
    gamma_reg_resolution: str = "half-integers"  # Resolution: "integers", "half-integers", "quarter-integers"
    
    # Output paths
    output_dir: str = "output"  # Base output directory
    data_dir: str = "data"  # Subdirectory for data files
    figures_dir: str = "figures"  # Subdirectory for figures
    results_dir: str = "results"  # Subdirectory for results
    
    # Output filenames
    model_results_filename: str = "optimization_discovery_results.json"
    plot_filename: str = "optimization_discovery_plots.png"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizationDiscoveryConfig':
        """Create config from dictionary (e.g., from JSON).
        
        Expects unified config format with OPTIMIZATION_DISCOVERY section and OUTPUT section.
        """
        # Extract OPTIMIZATION_DISCOVERY section
        if 'OPTIMIZATION_DISCOVERY' not in config_dict:
            raise ValueError(
                "Config must contain 'OPTIMIZATION_DISCOVERY' section. "
                "Please use the unified config format. "
                "See pydimension/configs/config_synthetic.json for an example."
            )
        
        optimization_discovery = config_dict['OPTIMIZATION_DISCOVERY']
        optimization_discovery_output = config_dict.get('OPTIMIZATION_DISCOVERY_OUTPUT', {})
        
        # Extract output settings from unified OUTPUT section
        output_section = config_dict.get('OUTPUT', {})
        
        # Get output_dir from OUTPUT section
        output_dir = output_section.get('output_dir', 'output')
        data_dir = output_section.get('data_dir', 'data')
        figures_dir = output_section.get('figures_dir', 'figures')
        results_dir = output_section.get('results_dir', 'results')
        
        model_results_filename = optimization_discovery_output.get('model_results_filename', 'optimization_discovery_results.json')
        plot_filename = optimization_discovery_output.get('plot_filename', 'optimization_discovery_plots.png')
        
        # Handle input file - can be relative or absolute
        input_file = optimization_discovery.get('input_file')
        if input_file is None:
            # Try to find default files from dimensional analysis
            base_path = Path(output_dir).resolve()
            default_paths = [
                base_path / data_dir / 'normalized_lg_afterDA_data.csv',
                Path('output') / data_dir / 'normalized_lg_afterDA_data.csv',
                Path('normalized_lg_afterDA_data.csv')
            ]
            # Also check relative to current working directory
            cwd = Path.cwd()
            default_paths.extend([
                cwd / output_dir / data_dir / 'normalized_lg_afterDA_data.csv',
                cwd / 'output' / data_dir / 'normalized_lg_afterDA_data.csv',
                cwd / 'normalized_lg_afterDA_data.csv'
            ])
            for path in default_paths:
                if path.exists():
                    input_file = str(path)
                    break
        
        # Handle basis vectors file
        basis_vectors_file = optimization_discovery.get('basis_vectors_file')
        if basis_vectors_file is None:
            # Try to find default basis_vectors.csv
            base_path = Path(output_dir).resolve()
            default_basis_paths = [
                base_path / data_dir / 'basis_vectors.csv',
                Path('output') / data_dir / 'basis_vectors.csv',
                Path('basis_vectors.csv')
            ]
            cwd = Path.cwd()
            default_basis_paths.extend([
                cwd / output_dir / data_dir / 'basis_vectors.csv',
                cwd / 'output' / data_dir / 'basis_vectors.csv',
                cwd / 'basis_vectors.csv'
            ])
            for path in default_basis_paths:
                if path.exists():
                    basis_vectors_file = str(path)
                    break
        
        # Handle num_linear - try to load from suggested_count file if not explicitly set
        num_linear = optimization_discovery.get('num_linear')
        if num_linear is None:
            # Try to load from suggested_dominant_count.json (from dimensional filtering)
            base_path = Path(output_dir).resolve()
            suggested_count_paths = [
                base_path / results_dir / 'suggested_dominant_count.json',
                Path('output') / results_dir / 'suggested_dominant_count.json',
                Path('suggested_dominant_count.json')
            ]
            cwd = Path.cwd()
            suggested_count_paths.extend([
                cwd / output_dir / results_dir / 'suggested_dominant_count.json',
                cwd / 'output' / results_dir / 'suggested_dominant_count.json',
                cwd / 'suggested_dominant_count.json'
            ])
            for path in suggested_count_paths:
                if path.exists():
                    try:
                        with open(path, 'r') as f:
                            suggested_data = json.load(f)
                            num_linear = suggested_data.get('suggested_dominant_count')
                            if num_linear is not None:
                                method = suggested_data.get('method', 'unknown')
                                print(f"✅ Loaded suggested dominant count from: {path}")
                                print(f"   Method: {method}, Using num_linear = {num_linear}")
                                break
                    except Exception:
                        pass
            
            # If not found, try to read from PCA results file directly
            if num_linear is None:
                pca_results_paths = [
                    base_path / results_dir / 'pca_results.json',
                    Path('output') / results_dir / 'pca_results.json',
                    Path('pca_results.json')
                ]
                pca_results_paths.extend([
                    cwd / output_dir / results_dir / 'pca_results.json',
                    cwd / 'output' / results_dir / 'pca_results.json',
                    cwd / 'pca_results.json'
                ])
                for path in pca_results_paths:
                    if path.exists():
                        try:
                            with open(path, 'r') as f:
                                pca_data = json.load(f)
                                num_linear = pca_data.get('suggested_dominant_count')
                                if num_linear is not None:
                                    print(f"✅ Loaded suggested dominant count from PCA results: {path}")
                                    print(f"   Using num_linear = {num_linear} (from PCA)")
                                    break
                        except Exception:
                            pass
            
            # Default if not found
            if num_linear is None:
                num_linear = 1
                print(f"⚠️ No suggested count file found, using default num_linear = {num_linear}")
                print(f"   Note: Run dimensional filtering first to get suggested count from PCA/SIR")
        
        # Handle fixed gamma values
        fixed_gamma_values = None
        if optimization_discovery.get('use_fixed_gamma', False):
            # Try to load from basis_vectors_file if available
            if basis_vectors_file and Path(basis_vectors_file).exists():
                try:
                    import pandas as pd
                    bv = pd.read_csv(basis_vectors_file)
                    wcols = [c for c in bv.columns if c.startswith('w')]
                    if len(wcols) > 0:
                        num_gamma = min(num_linear, len(wcols))
                        basis_matrix = bv[wcols].values  # Shape: (num_variables, num_basis)
                        fixed_gamma_values = basis_matrix[:, :num_gamma].T  # Shape: (num_gamma, num_variables)
                except Exception:
                    pass
        
        return cls(
            input_file=input_file,
            basis_vectors_file=basis_vectors_file,
            num_linear=num_linear,
            num_hidden_layers=optimization_discovery.get('num_hidden_layers', 4),
            num_hidden_nodes=optimization_discovery.get('num_hidden_nodes', 10),
            random_seed=optimization_discovery.get('random_seed', 49),
            epochs=optimization_discovery.get('epochs', 1000),
            learning_rate=optimization_discovery.get('learning_rate', 0.001),
            train_percent=optimization_discovery.get('train_percent', 0.8),
            num_ensembles=optimization_discovery.get('num_ensembles', 5),
            use_fixed_gamma=optimization_discovery.get('use_fixed_gamma', False),
            fixed_gamma_values=fixed_gamma_values,
            use_gamma_regularization=optimization_discovery.get('use_gamma_regularization', True),
            gamma_reg_strength=optimization_discovery.get('gamma_reg_strength', 0.01),
            gamma_reg_resolution=optimization_discovery.get('gamma_reg_resolution', 'half-integers'),
            output_dir=output_dir,
            data_dir=data_dir,
            figures_dir=figures_dir,
            results_dir=results_dir,
            model_results_filename=model_results_filename,
            plot_filename=plot_filename
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'OptimizationDiscoveryConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (unified format)."""
        fixed_gamma_list = None
        if self.fixed_gamma_values is not None:
            fixed_gamma_list = self.fixed_gamma_values.tolist()
        
        return {
            'OPTIMIZATION_DISCOVERY': {
                'enabled': True,
                'input_file': self.input_file,
                'basis_vectors_file': self.basis_vectors_file,
                'num_linear': self.num_linear,
                'num_hidden_layers': self.num_hidden_layers,
                'num_hidden_nodes': self.num_hidden_nodes,
                'random_seed': self.random_seed,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'train_percent': self.train_percent,
                'num_ensembles': self.num_ensembles,
                'use_fixed_gamma': self.use_fixed_gamma,
                'use_gamma_regularization': self.use_gamma_regularization,
                'gamma_reg_strength': self.gamma_reg_strength,
                'gamma_reg_resolution': self.gamma_reg_resolution
            },
            'OUTPUT': {
                'output_dir': self.output_dir,
                'data_dir': self.data_dir,
                'figures_dir': self.figures_dir,
                'results_dir': self.results_dir,
                'logs_dir': 'logs'
            },
            'OPTIMIZATION_DISCOVERY_OUTPUT': {
                'model_results_filename': self.model_results_filename,
                'plot_filename': self.plot_filename
            }
        }
    
    def to_json(self, json_path: str):
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors (empty if valid)."""
        errors = []
        
        if self.input_file is None:
            # Build list of checked paths for error message
            base_path = Path(self.output_dir).resolve()
            checked_paths = [
                base_path / self.data_dir / 'normalized_lg_afterDA_data.csv',
                Path('output') / self.data_dir / 'normalized_lg_afterDA_data.csv',
                Path('normalized_lg_afterDA_data.csv')
            ]
            checked_paths_str = "\n".join([f"    - {p}" for p in checked_paths])
            errors.append(
                f"input_file must be specified or found in default locations.\n"
                f"  Expected file: normalized_lg_afterDA_data.csv\n"
                f"  Default locations checked:\n{checked_paths_str}\n"
                f"  Note: This file is created by running dimensional analysis with --save-normalized-lg flag.\n"
                f"  Example: python analyze_dimensions.py --config pydimension/configs/config_synthetic.json --save-normalized-lg"
            )
        elif not Path(self.input_file).exists():
            errors.append(f"Input file not found: {self.input_file}")
        
        if self.num_linear < 1:
            errors.append("num_linear must be at least 1")
        
        if self.num_hidden_layers < 0:
            errors.append("num_hidden_layers must be non-negative")
        
        if self.num_hidden_nodes < 1:
            errors.append("num_hidden_nodes must be at least 1")
        
        if self.epochs < 1:
            errors.append("epochs must be at least 1")
        
        if not (0 < self.train_percent < 1):
            errors.append("train_percent must be between 0 and 1")
        
        if self.num_ensembles < 1:
            errors.append("num_ensembles must be at least 1")
        
        if self.gamma_reg_resolution not in ['integers', 'half-integers', 'quarter-integers']:
            errors.append("gamma_reg_resolution must be one of: 'integers', 'half-integers', 'quarter-integers'")
        
        if self.use_fixed_gamma and self.fixed_gamma_values is None:
            errors.append("use_fixed_gamma is True but fixed_gamma_values is None. Load basis vectors first.")
        
        return errors

