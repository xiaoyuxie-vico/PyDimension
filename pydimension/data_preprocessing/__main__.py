"""
Command-line interface for data preprocessing module.
"""

import argparse
import sys
from pathlib import Path

from .preprocessor import DataPreprocessor
from .config import DataPreprocessingConfig


def main():
    """Main entry point for data preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description='Preprocess datasets: select variables, normalize data, and generate dimension matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using unified config file
  python -m pydimension.data_preprocessing --config pydimension/configs/config_synthetic.json
  
  # Specify input file directly
  python -m pydimension.data_preprocessing --input_file dataset.csv
  
  # Specify input and output variables
  python -m pydimension.data_preprocessing --input_file dataset.csv \\
      --input_variables p1 p2 p3 p4 p5 p6 p7 \\
      --output_variables "p*"
  
  # Use dimension matrix from file
  python -m pydimension.data_preprocessing --input_file dataset.csv \\
      --dimension_matrix_file dimension_matrix.csv
        """
    )
    
    # Config file
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to unified JSON config file')
    
    # Input/Output settings
    parser.add_argument('--input_file', type=str, default=None,
                       help='Path to input CSV file')
    parser.add_argument('--input_variables', type=str, nargs='+', default=None,
                       help='List of input variable names (space-separated)')
    parser.add_argument('--output_variables', type=str, nargs='+', default=None,
                       help='List of output variable names (space-separated)')
    
    # Dimension matrix
    parser.add_argument('--dimension_matrix_file', type=str, default=None,
                       help='Path to dimension matrix CSV file (optional)')
    
    # Normalization
    parser.add_argument('--no-normalize', action='store_true',
                       help='Disable normalization (default: normalize enabled)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base output directory (overrides config)')
    parser.add_argument('--normalized_data_filename', type=str, default=None,
                       help='Filename for normalized data (overrides config)')
    parser.add_argument('--dimension_matrix_filename', type=str, default=None,
                       help='Filename for dimension matrix (overrides config)')
    
    # Visualization
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save visualization plots')
    parser.add_argument('--plot_filename', type=str, default=None,
                       help='Filename for saved plot (default: from config or "data_preprocessing_plots.png")')
    
    # Other
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            return 1
        config = DataPreprocessingConfig.from_json(args.config)
        if not args.quiet:
            print(f"Loaded config from: {args.config}")
    else:
        # Create config from command-line arguments
        config = DataPreprocessingConfig()
    
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
        results = preprocessor.process(verbose=not args.quiet)
        
        # Save results
        normalized_path, original_path, matrix_path = preprocessor.save_results()
        
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
            if not args.quiet:
                print(f"Plot: {plot_path}")
        
        if not args.quiet:
            print(f"\n✅ Preprocessing complete!")
            print(f"   Input variables: {len(results['input_variables'])}")
            print(f"   Output variables: {len(results['output_variables'])}")
            print(f"   Data shape: {results['data_shape']}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        if not args.quiet:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

