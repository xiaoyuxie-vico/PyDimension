"""
Command-line interface for dimensional analysis module.
"""

import argparse
import sys
from pathlib import Path

from .analyzer import DimensionalAnalyzer
from .config import DimensionalAnalysisConfig


def main():
    """Main entry point for dimensional analysis CLI."""
    parser = argparse.ArgumentParser(
        description='Perform dimensional analysis: find basis vectors and create dimensionless variables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using unified config file
  python -m pydimension.dimensional_analysis --config pydimension/configs/config_synthetic.json
  
  # Specify input files directly
  python -m pydimension.dimensional_analysis \\
      --normalized_data_file output/data/normalized_data.csv \\
      --dimension_matrix_file output/data/dimension_matrix.csv
  
  # Disable basis vector normalization
  python -m pydimension.dimensional_analysis \\
      --normalized_data_file normalized_data.csv \\
      --dimension_matrix_file dimension_matrix.csv \\
      --no-normalize-basis
        """
    )
    
    # Config file
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to unified JSON config file')
    
    # Input settings
    parser.add_argument('--normalized_data_file', type=str, default=None,
                       help='Path to normalized data CSV file')
    parser.add_argument('--dimension_matrix_file', type=str, default=None,
                       help='Path to dimension matrix CSV file')
    
    # Analysis settings
    parser.add_argument('--no-normalize-basis', action='store_true',
                       help='Disable normalization of basis vectors to unit length (default: normalize enabled)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base output directory (overrides config)')
    parser.add_argument('--afterDA_data_filename', type=str, default=None,
                       help='Filename for afterDA data (overrides config)')
    parser.add_argument('--basis_vectors_filename', type=str, default=None,
                       help='Filename for basis vectors (overrides config)')
    
    # Normalized lg data
    parser.add_argument('--save-normalized-lg', action='store_true',
                       help='Also save normalized log10 data (lgπ = log10(π/max(π)))')
    
    # Visualization
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save visualization plots')
    parser.add_argument('--plot_filename', type=str, default=None,
                       help='Filename for saved plot (default: from config or "dimensional_analysis_plots.png")')
    
    # Other
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            return 1
        config = DimensionalAnalysisConfig.from_json(args.config)
        if not args.quiet:
            print(f"Loaded config from: {args.config}")
    else:
        # Create config from command-line arguments
        config = DimensionalAnalysisConfig()
    
    # Override config with command-line arguments
    if args.normalized_data_file:
        config.normalized_data_file = args.normalized_data_file
    if args.dimension_matrix_file:
        config.dimension_matrix_file = args.dimension_matrix_file
    if args.no_normalize_basis:
        config.normalize_basis = False
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.afterDA_data_filename:
        config.afterDA_data_filename = args.afterDA_data_filename
    if args.basis_vectors_filename:
        config.basis_vectors_filename = args.basis_vectors_filename
    
    # Validate config
    errors = config.validate()
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    
    # Create analyzer and run
    try:
        analyzer = DimensionalAnalyzer(config)
        results = analyzer.process(verbose=not args.quiet)
        
        # Save results
        afterDA_path, basis_path = analyzer.save_results()
        
        # Save normalized lg data if requested
        if args.save_normalized_lg:
            lg_path = analyzer.save_normalized_lg_data()
            if not args.quiet:
                print(f"Normalized lg data: {lg_path}")
        
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
                            plot_filename = full_config.get('DIMENSIONAL_ANALYSIS_OUTPUT', {}).get('plot_filename')
                except:
                    pass
                if plot_filename is None:
                    plot_filename = 'dimensional_analysis_plots.png'
            
            plot_path = analyzer.create_visualization(filename=plot_filename)
            if not args.quiet:
                print(f"Plot: {plot_path}")
        
        if not args.quiet:
            print(f"\n✅ Dimensional analysis complete!")
            print(f"   Input variables: {len(results['input_variables'])}")
            print(f"   Output variable: {results['output_variable']}")
            print(f"   Null space dimension: {results['null_space_dimension']}")
            print(f"   Dimensionless groups: {len(results['dimensionless_expressions'])}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        if not args.quiet:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

