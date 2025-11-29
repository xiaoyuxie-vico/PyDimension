"""
Command-line interface for dimensional filtering module.
"""

import argparse
import sys
from pathlib import Path

from .filterer import ConstraintFilterer
from .config import ConstraintFilteringConfig


def main():
    """Main entry point for dimensional filtering CLI."""
    parser = argparse.ArgumentParser(
        description='Dimensional filtering: identify dominant dimensionless groups using PCA and SIR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using unified config file
  python -m pydimension.constraint_filtering --config pydimension/configs/config_synthetic.json
  
  # Specify input file directly
  python -m pydimension.constraint_filtering \\
      --input_file output/data/normalized_lg_afterDA_data.csv
  
  # Run only PCA
  python -m pydimension.constraint_filtering \\
      --input_file normalized_lg_afterDA_data.csv \\
      --no-sir
  
  # Run only SIR
  python -m pydimension.constraint_filtering \\
      --input_file normalized_lg_afterDA_data.csv \\
      --no-pca
  
  # Custom SIR parameters
  python -m pydimension.constraint_filtering \\
      --input_file normalized_lg_afterDA_data.csv \\
      --n_sir_slices 20 \\
      --n_sir_directions 5
        """
    )
    
    # Config file
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to unified JSON config file')
    
    # Input settings
    parser.add_argument('--input_file', type=str, default=None,
                       help='Path to normalized lg afterDA data CSV file')
    
    # Analysis settings
    parser.add_argument('--no-pca', action='store_true',
                       help='Disable PCA analysis (default: PCA enabled)')
    parser.add_argument('--no-sir', action='store_true',
                       help='Disable SIR analysis (default: SIR enabled)')
    parser.add_argument('--n_sir_slices', type=int, default=None,
                       help='Number of slices for SIR (overrides config)')
    parser.add_argument('--n_sir_directions', type=int, default=None,
                       help='Number of SIR directions to compute (overrides config)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base output directory (overrides config)')
    
    # Visualization
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save visualization plots')
    parser.add_argument('--plot_filename', type=str, default=None,
                       help='Filename for saved plot (default: from config or "constraint_filtering_plots.png")')
    
    # Other
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            return 1
        config = ConstraintFilteringConfig.from_json(args.config)
        if not args.quiet:
            print(f"Loaded config from: {args.config}")
    else:
        # Create config from command-line arguments
        config = ConstraintFilteringConfig()
        if not args.quiet:
            print("Using default configuration. Use --config to specify a config file.")
    
    # Override config with command-line arguments
    if args.input_file:
        config.input_file = args.input_file
    if args.no_pca:
        config.run_pca = False
    if args.no_sir:
        config.run_sir = False
    if args.n_sir_slices:
        config.n_sir_slices = args.n_sir_slices
    if args.n_sir_directions:
        config.n_sir_directions = args.n_sir_directions
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Validate config (after all overrides)
    errors = config.validate()
    if errors:
        if not args.quiet:
            print("\n❌ Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    
    # Create filterer and run
    try:
        filterer = ConstraintFilterer(config)
        results = filterer.process(verbose=not args.quiet)
        
        # Save results
        pca_path, sir_path = filterer.save_results()
        
        # Save suggested dominant count for optimization discovery
        count_path = filterer.save_suggested_count()
        if count_path and not args.quiet:
            print(f"Suggested count: {count_path}")
        
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
                            plot_filename = full_config.get('CONSTRAINT_FILTERING_OUTPUT', {}).get('plot_filename')
                except:
                    pass
                if plot_filename is None:
                    plot_filename = 'constraint_filtering_plots.png'
            
            plot_path = filterer.create_visualization(filename=plot_filename)
            if not args.quiet:
                print(f"Plot: {plot_path}")
        
        if not args.quiet:
            print(f"\n✅ Dimensional filtering complete!")
            if config.run_pca and 'pca' in results:
                print(f"   PCA: Suggested {results['pca']['suggested_dominant_count']} dominant components")
            if config.run_sir and 'sir' in results:
                print(f"   SIR: Suggested {results['sir']['suggested_directions']} important direction(s)")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        if not args.quiet:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

