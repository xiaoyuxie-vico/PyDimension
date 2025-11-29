#!/usr/bin/env python3
"""
Simple script to preprocess datasets using PyDimension 2.0.

This is a convenience script that can be run directly from the package root.
Uses unified config files that contain settings for all modules.
"""

import sys
from pathlib import Path

# Add pydimension to path
sys.path.insert(0, str(Path(__file__).parent))

from pydimension.data_preprocessing import DataPreprocessor, DataPreprocessingConfig


def main():
    """Preprocess data from config file or use command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess datasets: select variables, normalize data, and generate dimension matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use unified config file
  python preprocess_data.py --config pydimension/configs/config_synthetic.json
  
  # Use default config (auto-detect from data generation output)
  python preprocess_data.py
  
  # Specify input file directly
  python preprocess_data.py --input_file dataset.csv
  
  # Specify input and output variables
  python preprocess_data.py --input_file dataset.csv \\
      --input_variables p1 p2 p3 p4 p5 p6 p7 \\
      --output_variables "p*"
  
  # Use dimension matrix from file
  python preprocess_data.py --input_file dataset.csv \\
      --dimension_matrix_file output/data/dimension_matrix_synthetic.csv
  
  # Disable normalization
  python preprocess_data.py --input_file dataset.csv --no-normalize
        """
    )
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to unified JSON config file. '
                            'If not specified, uses default configuration.')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Path to input CSV file')
    parser.add_argument('--input_variables', type=str, nargs='+', default=None,
                       help='List of input variable names (space-separated)')
    parser.add_argument('--output_variables', type=str, nargs='+', default=None,
                       help='List of output variable names (space-separated)')
    parser.add_argument('--dimension_matrix_file', type=str, default=None,
                       help='Path to dimension matrix CSV file (optional)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Disable normalization (default: normalize enabled)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base output directory (overrides config, default: "output")')
    parser.add_argument('--normalized_data_filename', type=str, default=None,
                       help='Filename for normalized data (overrides config)')
    parser.add_argument('--dimension_matrix_filename', type=str, default=None,
                       help='Filename for dimension matrix (overrides config)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save visualization plots')
    parser.add_argument('--plot_filename', type=str, default=None,
                       help='Filename for saved plot (default: from config or "data_preprocessing_plots.png")')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}")
            return 1
        config = DataPreprocessingConfig.from_json(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        # Use default config
        config = DataPreprocessingConfig()
        print("Using default configuration. Use --config to specify a config file.")
        print("Example: python preprocess_data.py --config pydimension/configs/config_synthetic.json")
    
    # Override config with command-line arguments
    if args.input_file:
        config.input_file = args.input_file
    if args.input_variables:
        config.input_variables = args.input_variables
    if args.output_variables:
        config.output_variables = args.output_variables
    if args.dimension_matrix_file:
        config.dimension_matrix_file = args.dimension_matrix_file
    if args.no_normalize:
        config.normalize = False
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.normalized_data_filename:
        config.normalized_data_filename = args.normalized_data_filename
    if args.dimension_matrix_filename:
        config.dimension_matrix_filename = args.dimension_matrix_filename
    
    # Validate config
    errors = config.validate()
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    
    # Create preprocessor and run
    try:
        preprocessor = DataPreprocessor(config)
        results = preprocessor.process(verbose=True)
        
        # Save results
        normalized_path, matrix_path = preprocessor.save_results()
        
        # Create visualization if requested
        if args.plot:
            plot_filename = args.plot_filename
            if plot_filename is None:
                # Try to get from config if available
                try:
                    import json
                    if args.config:
                        with open(args.config, 'r') as f:
                            full_config = json.load(f)
                            plot_filename = full_config.get('DATA_PREPROCESSING_OUTPUT', {}).get('plot_filename')
                except:
                    pass
                if plot_filename is None:
                    plot_filename = 'data_preprocessing_plots.png'
            
            plot_path = preprocessor.create_visualization(filename=plot_filename)
            print(f"Plot: {plot_path}")
        
        print(f"\n=== Files Saved ===")
        print(f"Normalized data: {normalized_path}")
        print(f"Dimension matrix: {matrix_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

