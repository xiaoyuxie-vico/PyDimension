"""
Configuration handling for data preprocessing module.
"""

import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataPreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Input/Output settings
    input_file: Optional[str] = None  # Path to input CSV file
    input_variables: Optional[List[str]] = None  # List of input variable names (if None, auto-detect)
    output_variables: Optional[List[str]] = None  # List of output variable names (if None, auto-detect)
    
    # Dimension matrix settings
    dimension_matrix_file: Optional[str] = None  # Path to dimension matrix CSV (optional)
    variable_units: Optional[Dict[str, str]] = None  # Dictionary mapping variable names to units
    
    # Normalization settings
    normalize: bool = True  # Whether to normalize data (divide by maximum)
    
    # Output paths
    output_dir: str = "output"  # Base output directory
    normalized_data_filename: str = "normalized_data.csv"
    original_data_filename: str = "original_data.csv"  # Original input data file
    dimension_matrix_filename: str = "dimension_matrix.csv"
    data_dir: str = "data"  # Subdirectory for data files
    figures_dir: str = "figures"  # Subdirectory for figures
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataPreprocessingConfig':
        """Create config from dictionary (e.g., from JSON).
        
        Expects unified config format with DATA_PREPROCESSING section and OUTPUT section.
        """
        # Extract DATA_PREPROCESSING section
        if 'DATA_PREPROCESSING' not in config_dict:
            raise ValueError(
                "Config must contain 'DATA_PREPROCESSING' section. "
                "Please use the unified config format. "
                "See pydimension/configs/config_synthetic.json for an example."
            )
        
        data_prep = config_dict['DATA_PREPROCESSING']
        
        # Extract output settings from unified OUTPUT section
        output_section = config_dict.get('OUTPUT', {})
        data_prep_output = config_dict.get('DATA_PREPROCESSING_OUTPUT', {})
        
        # Get output_dir from OUTPUT section
        output_dir = output_section.get('output_dir', 'output')
        data_dir = output_section.get('data_dir', 'data')
        
        normalized_data_filename = data_prep_output.get('normalized_data_filename', 'normalized_data.csv')
        original_data_filename = data_prep_output.get('original_data_filename', 'original_data.csv')
        dimension_matrix_filename = data_prep_output.get('dimension_matrix_filename', 'dimension_matrix.csv')
        
        figures_dir = output_section.get('figures_dir', 'figures')
        
        # Handle input_file - can be relative or absolute
        input_file = data_prep.get('input_file')
        if input_file is None:
            # Try to find default dataset from data generation
            default_paths = [
                Path(output_dir) / data_dir / 'dataset_synthetic.csv',
                'output/data/dataset_synthetic.csv',
                'dataset_synthetic.csv'
            ]
            for path in default_paths:
                if Path(path).exists():
                    input_file = str(path)
                    break
        
        return cls(
            input_file=input_file,
            input_variables=data_prep.get('input_variables'),
            output_variables=data_prep.get('output_variables'),
            dimension_matrix_file=data_prep.get('dimension_matrix_file'),
            variable_units=data_prep.get('variable_units'),
            normalize=data_prep.get('normalize', True),
            output_dir=output_dir,
            normalized_data_filename=normalized_data_filename,
            original_data_filename=original_data_filename,
            dimension_matrix_filename=dimension_matrix_filename,
            data_dir=data_dir,
            figures_dir=figures_dir
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'DataPreprocessingConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (unified format)."""
        return {
            'DATA_PREPROCESSING': {
                'enabled': True,
                'input_file': self.input_file,
                'input_variables': self.input_variables,
                'output_variables': self.output_variables,
                'dimension_matrix_file': self.dimension_matrix_file,
                'variable_units': self.variable_units,
                'normalize': self.normalize
            },
            'OUTPUT': {
                'output_dir': self.output_dir,
                'data_dir': self.data_dir,
                'figures_dir': 'figures',
                'results_dir': 'results',
                'logs_dir': 'logs'
            },
            'DATA_PREPROCESSING_OUTPUT': {
                'normalized_data_filename': self.normalized_data_filename,
                'original_data_filename': self.original_data_filename,
                'dimension_matrix_filename': self.dimension_matrix_filename,
                'plot_filename': 'data_preprocessing_plots.png'
            }
        }
    
    def to_json(self, json_path: str):
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors (empty if valid)."""
        errors = []
        
        if self.input_file is None:
            errors.append("input_file must be specified")
        elif not Path(self.input_file).exists():
            errors.append(f"Input file not found: {self.input_file}")
        
        if self.input_variables is not None and len(self.input_variables) == 0:
            errors.append("input_variables cannot be empty list (use None for auto-detect)")
        
        if self.output_variables is not None and len(self.output_variables) == 0:
            errors.append("output_variables cannot be empty list (use None for auto-detect)")
        
        if self.dimension_matrix_file is not None and not Path(self.dimension_matrix_file).exists():
            errors.append(f"Dimension matrix file not found: {self.dimension_matrix_file}")
        
        return errors

