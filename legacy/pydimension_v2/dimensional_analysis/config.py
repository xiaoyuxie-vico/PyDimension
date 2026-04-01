"""
Configuration handling for dimensional analysis module.
"""

import json
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DimensionalAnalysisConfig:
    """Configuration for dimensional analysis."""
    
    # Input settings
    normalized_data_file: Optional[str] = None  # Path to normalized data CSV
    dimension_matrix_file: Optional[str] = None  # Path to dimension matrix CSV
    
    # Analysis settings
    normalize_basis: bool = True  # Normalize basis vectors to unit length
    
    # Output paths
    output_dir: str = "output"  # Base output directory
    afterDA_data_filename: str = "afterDA_data.csv"
    basis_vectors_filename: str = "basis_vectors.csv"
    normalized_lg_data_filename: str = "normalized_lg_afterDA_data.csv"
    data_dir: str = "data"  # Subdirectory for data files
    figures_dir: str = "figures"  # Subdirectory for figures
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DimensionalAnalysisConfig':
        """Create config from dictionary (e.g., from JSON).
        
        Expects unified config format with DIMENSIONAL_ANALYSIS section and OUTPUT section.
        """
        # Extract DIMENSIONAL_ANALYSIS section
        if 'DIMENSIONAL_ANALYSIS' not in config_dict:
            raise ValueError(
                "Config must contain 'DIMENSIONAL_ANALYSIS' section. "
                "Please use the unified config format. "
                "See pydimension/configs/config_synthetic.json for an example."
            )
        
        dim_analysis = config_dict['DIMENSIONAL_ANALYSIS']
        
        # Extract output settings from unified OUTPUT section
        output_section = config_dict.get('OUTPUT', {})
        dim_analysis_output = config_dict.get('DIMENSIONAL_ANALYSIS_OUTPUT', {})
        
        # Get output_dir from OUTPUT section
        output_dir = output_section.get('output_dir', 'output')
        data_dir = output_section.get('data_dir', 'data')
        figures_dir = output_section.get('figures_dir', 'figures')
        
        afterDA_data_filename = dim_analysis_output.get('afterDA_data_filename', 'afterDA_data.csv')
        basis_vectors_filename = dim_analysis_output.get('basis_vectors_filename', 'basis_vectors.csv')
        normalized_lg_data_filename = dim_analysis_output.get('normalized_lg_data_filename', 'normalized_lg_afterDA_data.csv')
        
        # Handle input files - can be relative or absolute
        normalized_data_file = dim_analysis.get('normalized_data_file')
        if normalized_data_file is None:
            # Try to find default files from data preprocessing
            default_paths = [
                Path(output_dir) / data_dir / 'normalized_data.csv',
                'output/data/normalized_data.csv',
                'normalized_data.csv'
            ]
            for path in default_paths:
                if Path(path).exists():
                    normalized_data_file = str(path)
                    break
        
        dimension_matrix_file = dim_analysis.get('dimension_matrix_file')
        if dimension_matrix_file is None:
            # Try to find default dimension matrix
            default_paths = [
                Path(output_dir) / data_dir / 'dimension_matrix.csv',
                'output/data/dimension_matrix.csv',
                'dimension_matrix.csv'
            ]
            for path in default_paths:
                if Path(path).exists():
                    dimension_matrix_file = str(path)
                    break
        
        return cls(
            normalized_data_file=normalized_data_file,
            dimension_matrix_file=dimension_matrix_file,
            normalize_basis=dim_analysis.get('normalize_basis', True),
            output_dir=output_dir,
            afterDA_data_filename=afterDA_data_filename,
            basis_vectors_filename=basis_vectors_filename,
            normalized_lg_data_filename=normalized_lg_data_filename,
            data_dir=data_dir,
            figures_dir=figures_dir
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'DimensionalAnalysisConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (unified format)."""
        return {
            'DIMENSIONAL_ANALYSIS': {
                'enabled': True,
                'normalized_data_file': self.normalized_data_file,
                'dimension_matrix_file': self.dimension_matrix_file,
                'normalize_basis': self.normalize_basis
            },
            'OUTPUT': {
                'output_dir': self.output_dir,
                'data_dir': self.data_dir,
                'figures_dir': self.figures_dir,
                'results_dir': 'results',
                'logs_dir': 'logs'
            },
            'DIMENSIONAL_ANALYSIS_OUTPUT': {
                'afterDA_data_filename': self.afterDA_data_filename,
                'basis_vectors_filename': self.basis_vectors_filename,
                'normalized_lg_data_filename': self.normalized_lg_data_filename,
                'plot_filename': 'dimensional_analysis_plots.png'
            }
        }
    
    def to_json(self, json_path: str):
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors (empty if valid)."""
        errors = []
        
        if self.normalized_data_file is None:
            errors.append("normalized_data_file must be specified or found in default locations")
        elif not Path(self.normalized_data_file).exists():
            errors.append(f"Normalized data file not found: {self.normalized_data_file}")
        
        if self.dimension_matrix_file is None:
            errors.append("dimension_matrix_file must be specified or found in default locations")
        elif not Path(self.dimension_matrix_file).exists():
            errors.append(f"Dimension matrix file not found: {self.dimension_matrix_file}")
        
        return errors

