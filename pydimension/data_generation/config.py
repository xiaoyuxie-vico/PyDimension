"""
Configuration handling for data generation module.
"""

import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation."""
    
    # Control parameters
    N: int = 7  # Number of input variables
    M: int = 100  # Number of datapoints
    ndim: int = 1  # Number of dimensionless groups
    poly_order: int = 1  # Polynomial order (for ndim=1)
    random_seed: int = 32  # Random seed for reproducibility
    noise_level: float = 0.0  # Noise level in percentage (0-100)
    n_discrete: int = 0  # Number of discretely sampled variables
    n_fix: int = 5  # Number of fixed values for discrete variables
    
    # Coefficients for output relationship
    # For ndim=1: polynomial coefficients [A, B, C, ...] for p* = A + B*π1 + C*π1² + ...
    # For ndim>1: nonlinear coefficients [A, B, C] for p* = exp(A×π1) + π2^B + log(1+C×π3)
    coefficients: List[float] = field(default_factory=lambda: [2.0, 1.0])
    
    # Gamma vectors (one per dimensionless group)
    # Each gamma vector has dimension N-4
    # If not provided, will be auto-generated
    gamma_vectors: Optional[List[List[float]]] = None
    
    # Output paths
    output_dir: str = "output"  # Base output directory for generated files
    dataset_filename: str = "dataset_synthetic.csv"
    dimension_matrix_filename: str = "dimension_matrix_synthetic.csv"
    figures_dir: str = "figures"  # Subdirectory for figures
    data_dir: str = "data"  # Subdirectory for data files
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataGenerationConfig':
        """Create config from dictionary (e.g., from JSON).
        
        Expects unified config format with DATA_GENERATION section and OUTPUT section.
        """
        # Extract DATA_GENERATION section
        if 'DATA_GENERATION' not in config_dict:
            raise ValueError(
                "Config must contain 'DATA_GENERATION' section. "
                "Please use the unified config format. "
                "See pydimension/configs/config_synthetic.json for an example."
            )
        
        data_gen = config_dict['DATA_GENERATION']
        
        # Extract coefficients
        coefficients = data_gen.get('coefficients', None)
        if coefficients is None:
            # Try to extract from coefficient dict
            coeff_dict = data_gen.get('all_coefficients', None)
            if coeff_dict is None:
                # Try to build from A, B, C, ... keys
                coeff_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
                coefficients = []
                for name in coeff_names:
                    if name in data_gen:
                        coefficients.append(data_gen[name])
                    else:
                        break
                if not coefficients:
                    coefficients = [2.0, 1.0]  # Default
            else:
                coefficients = coeff_dict
        
        # Extract output settings from unified OUTPUT section
        output_section = config_dict.get('OUTPUT', {})
        data_gen_output = config_dict.get('DATA_GENERATION_OUTPUT', {})
        
        # Get output_dir from OUTPUT section
        output_dir = output_section.get('output_dir', 'output')
        data_dir = output_section.get('data_dir', 'data')
        figures_dir = output_section.get('figures_dir', 'figures')
        
        dataset_filename = data_gen_output.get('dataset_filename', 'dataset_synthetic.csv')
        dimension_matrix_filename = data_gen_output.get('dimension_matrix_filename', 'dimension_matrix_synthetic.csv')
        
        return cls(
            N=data_gen.get('N', 7),
            M=data_gen.get('M', 100),
            ndim=data_gen.get('ndim', 1),
            poly_order=data_gen.get('poly_order', 1),
            random_seed=data_gen.get('random_seed', 32),
            noise_level=data_gen.get('noise_level', 0.0),
            n_discrete=data_gen.get('n_discrete', 0),
            n_fix=data_gen.get('n_fix', 5),
            coefficients=coefficients,
            gamma_vectors=data_gen.get('gamma_vectors', None),
            output_dir=output_dir,
            dataset_filename=dataset_filename,
            dimension_matrix_filename=dimension_matrix_filename,
            figures_dir=figures_dir,
            data_dir=data_dir
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'DataGenerationConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (unified format)."""
        return {
            'DATA_GENERATION': {
                'N': self.N,
                'M': self.M,
                'ndim': self.ndim,
                'poly_order': self.poly_order,
                'random_seed': self.random_seed,
                'noise_level': self.noise_level,
                'n_discrete': self.n_discrete,
                'n_fix': self.n_fix,
                'coefficients': self.coefficients,
                'gamma_vectors': self.gamma_vectors
            },
            'OUTPUT': {
                'output_dir': self.output_dir,
                'data_dir': self.data_dir,
                'figures_dir': self.figures_dir,
                'results_dir': 'results',
                'logs_dir': 'logs'
            },
            'DATA_GENERATION_OUTPUT': {
                'dataset_filename': self.dataset_filename,
                'dimension_matrix_filename': self.dimension_matrix_filename,
                'plot_filename': 'data_generation_plots.png'
            }
        }
    
    def to_json(self, json_path: str):
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors (empty if valid)."""
        errors = []
        
        if self.N < 5:
            errors.append("N must be at least 5 to ensure rank=4 dimension matrix with null space.")
        
        if self.M < 10:
            errors.append("M must be at least 10.")
        
        if self.ndim < 1:
            errors.append("ndim must be at least 1.")
        
        if self.ndim > 3:
            errors.append(f"ndim={self.ndim} is greater than 3 and is not supported. Please use ndim <= 3.")
        
        if self.N < 4 + self.ndim:
            errors.append(f"N must be at least {4 + self.ndim} for ndim={self.ndim} (need N ≥ 4 + ndim).")
        
        if self.n_discrete < 0 or self.n_discrete > self.N:
            errors.append(f"Number of discrete variables must be between 0 and {self.N}.")
        
        if self.n_discrete > 0 and self.n_fix < 2:
            errors.append("n_fix must be at least 2 for discrete variables.")
        
        if self.noise_level < 0 or self.noise_level > 100:
            errors.append("noise_level must be between 0 and 100.")
        
        if self.poly_order < 1 or self.poly_order > 10:
            errors.append("polynomial order must be between 1 and 10.")
        
        # Check coefficients
        if self.ndim == 1:
            # Need at least poly_order + 1 coefficients
            if len(self.coefficients) < self.poly_order + 1:
                errors.append(f"For ndim=1, need at least {self.poly_order + 1} coefficients (got {len(self.coefficients)}).")
        else:
            # Need at least ndim coefficients
            if len(self.coefficients) < self.ndim:
                errors.append(f"For ndim={self.ndim}, need at least {self.ndim} coefficients (got {len(self.coefficients)}).")
        
        return errors

