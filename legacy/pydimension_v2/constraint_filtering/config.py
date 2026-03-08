"""
Configuration handling for dimensional filtering module.
"""

import json
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConstraintFilteringConfig:
    """Configuration for dimensional filtering."""
    
    # Input settings
    input_file: Optional[str] = None  # Path to normalized lg afterDA data CSV
    
    # Analysis settings
    run_pca: bool = True  # Run PCA analysis
    run_sir: bool = True  # Run SIR analysis
    pca_threshold: float = 0.75  # PCA cumulative variance threshold (0.0-1.0) for suggesting dominant count
    sir_threshold: float = 0.75  # SIR cumulative variance threshold (0.0-1.0) for suggesting dominant count
    n_sir_slices: int = 10  # Number of slices for SIR
    n_sir_directions: int = 3  # Number of SIR directions to compute
    
    # Output paths
    output_dir: str = "output"  # Base output directory
    data_dir: str = "data"  # Subdirectory for data files
    figures_dir: str = "figures"  # Subdirectory for figures
    results_dir: str = "results"  # Subdirectory for results
    
    # Output filenames
    pca_results_filename: str = "pca_results.json"
    sir_results_filename: str = "sir_results.json"
    plot_filename: str = "constraint_filtering_plots.png"
    suggested_count_filename: str = "suggested_dominant_count.json"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConstraintFilteringConfig':
        """Create config from dictionary (e.g., from JSON).
        
        Expects unified config format with CONSTRAINT_FILTERING section and OUTPUT section.
        """
        # Extract CONSTRAINT_FILTERING section
        if 'CONSTRAINT_FILTERING' not in config_dict:
            raise ValueError(
                "Config must contain 'CONSTRAINT_FILTERING' section. "
                "Please use the unified config format. "
                "See pydimension/configs/config_synthetic.json for an example."
            )
        
        constraint_filtering = config_dict['CONSTRAINT_FILTERING']
        constraint_filtering_output = config_dict.get('CONSTRAINT_FILTERING_OUTPUT', {})
        
        # Extract output settings from unified OUTPUT section
        output_section = config_dict.get('OUTPUT', {})
        
        # Get output_dir from OUTPUT section
        output_dir = output_section.get('output_dir', 'output')
        data_dir = output_section.get('data_dir', 'data')
        figures_dir = output_section.get('figures_dir', 'figures')
        results_dir = output_section.get('results_dir', 'results')
        
        pca_results_filename = constraint_filtering_output.get('pca_results_filename', 'pca_results.json')
        sir_results_filename = constraint_filtering_output.get('sir_results_filename', 'sir_results.json')
        plot_filename = constraint_filtering_output.get('plot_filename', 'constraint_filtering_plots.png')
        suggested_count_filename = constraint_filtering_output.get('suggested_count_filename', 'suggested_dominant_count.json')
        
        # Handle input file - can be relative or absolute
        input_file = constraint_filtering.get('input_file')
        if input_file is None:
            # Try to find default files from dimensional analysis
            # Use absolute paths based on config output_dir
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
        
        return cls(
            input_file=input_file,
            run_pca=constraint_filtering.get('run_pca', True),
            run_sir=constraint_filtering.get('run_sir', True),
            pca_threshold=constraint_filtering.get('pca_threshold', 0.75),
            sir_threshold=constraint_filtering.get('sir_threshold', 0.75),
            n_sir_slices=constraint_filtering.get('n_sir_slices', 10),
            n_sir_directions=constraint_filtering.get('n_sir_directions', 3),
            output_dir=output_dir,
            data_dir=data_dir,
            figures_dir=figures_dir,
            results_dir=results_dir,
            pca_results_filename=pca_results_filename,
            sir_results_filename=sir_results_filename,
            plot_filename=plot_filename,
            suggested_count_filename=suggested_count_filename
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'ConstraintFilteringConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (unified format)."""
        return {
            'CONSTRAINT_FILTERING': {
                'enabled': True,
                'input_file': self.input_file,
                'run_pca': self.run_pca,
                'run_sir': self.run_sir,
                'pca_threshold': self.pca_threshold,
                'sir_threshold': self.sir_threshold,
                'n_sir_slices': self.n_sir_slices,
                'n_sir_directions': self.n_sir_directions
            },
            'OUTPUT': {
                'output_dir': self.output_dir,
                'data_dir': self.data_dir,
                'figures_dir': self.figures_dir,
                'results_dir': self.results_dir,
                'logs_dir': 'logs'
            },
            'CONSTRAINT_FILTERING_OUTPUT': {
                'pca_results_filename': self.pca_results_filename,
                'sir_results_filename': self.sir_results_filename,
                'plot_filename': self.plot_filename,
                'suggested_count_filename': self.suggested_count_filename
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
            # Build list of checked paths for error message (remove duplicates)
            base_path = Path(self.output_dir).resolve()
            cwd = Path.cwd()
            checked_paths_set = {
                base_path / self.data_dir / 'normalized_lg_afterDA_data.csv',
                cwd / 'output' / self.data_dir / 'normalized_lg_afterDA_data.csv',
                Path('output') / self.data_dir / 'normalized_lg_afterDA_data.csv',
                cwd / 'normalized_lg_afterDA_data.csv',
                Path('normalized_lg_afterDA_data.csv')
            }
            checked_paths = sorted([str(p) for p in checked_paths_set])
            checked_paths_str = "\n".join([f"    - {p}" for p in checked_paths])
            errors.append(
                f"input_file must be specified or found in default locations.\n"
                f"  Expected file: normalized_lg_afterDA_data.csv\n"
                f"  Default locations checked:\n{checked_paths_str}\n"
                f"  \n"
                f"  This file is created by the dimensional analysis module.\n"
                f"  \n"
                f"  Option 1 (Recommended): Use the automated pipeline:\n"
                f"    python run_pipeline.py --config pydimension/configs/config_synthetic.json\n"
                f"  \n"
                f"  Option 2: Run dimensional analysis first with --save-normalized-lg:\n"
                f"    python analyze_dimensions.py --config pydimension/configs/config_synthetic.json --save-normalized-lg\n"
                f"    python filter_constraints.py --config pydimension/configs/config_synthetic.json\n"
                f"  \n"
                f"  Option 3: Specify the input file explicitly:\n"
                f"    python filter_constraints.py --input_file path/to/normalized_lg_afterDA_data.csv"
            )
        elif not Path(self.input_file).exists():
            errors.append(f"Input file not found: {self.input_file}")
        
        if self.n_sir_slices < 2:
            errors.append("n_sir_slices must be at least 2")
        
        if self.n_sir_directions < 1:
            errors.append("n_sir_directions must be at least 1")
        
        if not (0.0 < self.pca_threshold <= 1.0):
            errors.append("pca_threshold must be between 0.0 and 1.0 (exclusive of 0.0)")
        
        if not (0.0 < self.sir_threshold <= 1.0):
            errors.append("sir_threshold must be between 0.0 and 1.0 (exclusive of 0.0)")
        
        if not self.run_pca and not self.run_sir:
            errors.append("At least one of run_pca or run_sir must be True")
        
        return errors

