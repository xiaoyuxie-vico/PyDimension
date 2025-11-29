"""
Command-line entry point for data generation module.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydimension.data_generation import DataGenerator, DataGenerationConfig


def main():
    """Main entry point for data generation from command line."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic datasets with known dimensionless relationships',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data using a config file
  python -m pydimension.data_generation --config pydimension/configs/config_synthetic.json
  
  # Generate data with custom parameters
  python -m pydimension.data_generation --N 7 --M 100 --ndim 1 --random_seed 42
  
  # Generate multi-dimensional data
  python -m pydimension.data_generation --N 8 --M 200 --ndim 2 --coefficients 2.0 -0.5 0.5
        """
    )
    
    # Config file option
    parser.add_argument('--config', '-c', type=str, help='Path to JSON config file')
    
    # Direct parameter options
    parser.add_argument('--N', type=int, help='Number of input variables')
    parser.add_argument('--M', type=int, help='Number of datapoints')
    parser.add_argument('--ndim', type=int, help='Number of dimensionless groups')
    parser.add_argument('--poly_order', type=int, help='Polynomial order (for ndim=1)')
    parser.add_argument('--random_seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--noise_level', type=float, help='Noise level in percentage (0-100)')
    parser.add_argument('--n_discrete', type=int, help='Number of discretely sampled variables')
    parser.add_argument('--n_fix', type=int, help='Number of fixed values for discrete variables')
    parser.add_argument('--coefficients', nargs='+', type=float, 
                       help='Coefficients for output relationship (space-separated)')
    parser.add_argument('--gamma_vectors', type=str, 
                       help='Gamma vectors as comma-separated lists, semicolon-separated (e.g., "1,1,0;0,1,1")')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Base output directory for generated files (default: from config or "output")')
    parser.add_argument('--dataset_filename', type=str, default='dataset_synthetic.csv',
                       help='Filename for generated dataset')
    parser.add_argument('--dimension_matrix_filename', type=str, 
                       default='dimension_matrix_synthetic.csv',
                       help='Filename for dimension matrix')
    parser.add_argument('--max_trials', type=int, default=10,
                       help='Maximum trials to find valid vectors')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save visualization plots')
    parser.add_argument('--plot_filename', type=str, default='data_generation_plots.png',
                       help='Filename for saved plot (default: data_generation_plots.png)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Load config from file or create from arguments
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        config = DataGenerationConfig.from_json(args.config)
        
        # Override with command-line arguments if provided
        if args.N is not None:
            config.N = args.N
        if args.M is not None:
            config.M = args.M
        if args.ndim is not None:
            config.ndim = args.ndim
        if args.poly_order is not None:
            config.poly_order = args.poly_order
        if args.random_seed is not None:
            config.random_seed = args.random_seed
        if args.noise_level is not None:
            config.noise_level = args.noise_level
        if args.n_discrete is not None:
            config.n_discrete = args.n_discrete
        if args.n_fix is not None:
            config.n_fix = args.n_fix
        if args.coefficients is not None:
            config.coefficients = args.coefficients
        if args.gamma_vectors is not None:
            # Parse gamma vectors: "1,1,0;0,1,1" -> [[1,1,0], [0,1,1]]
            gamma_strs = args.gamma_vectors.split(';')
            config.gamma_vectors = [[float(x) for x in g.split(',')] for g in gamma_strs]
        if args.output_dir is not None:
            config.output_dir = args.output_dir
        if args.dataset_filename is not None:
            config.dataset_filename = args.dataset_filename
        if args.dimension_matrix_filename is not None:
            config.dimension_matrix_filename = args.dimension_matrix_filename
    else:
        # Create config from command-line arguments
        if args.N is None or args.M is None or args.ndim is None:
            print("Error: Must provide --config or --N, --M, and --ndim")
            sys.exit(1)
        
        config = DataGenerationConfig(
            N=args.N,
            M=args.M,
            ndim=args.ndim,
            poly_order=args.poly_order or 1,
            random_seed=args.random_seed or 32,
            noise_level=args.noise_level or 0.0,
            n_discrete=args.n_discrete or 0,
            n_fix=args.n_fix or 5,
            coefficients=args.coefficients or ([2.0, 1.0] if args.ndim == 1 else [2.0, -0.5, 0.5]),
            gamma_vectors=None,  # Will be auto-generated
            output_dir=args.output_dir or "output",  # Default to "output" if not specified
            dataset_filename=args.dataset_filename,
            dimension_matrix_filename=args.dimension_matrix_filename
        )
        
        if args.gamma_vectors is not None:
            # Parse gamma vectors
            gamma_strs = args.gamma_vectors.split(';')
            config.gamma_vectors = [[float(x) for x in g.split(',')] for g in gamma_strs]
    
    # Validate config
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Generate data
    try:
        generator = DataGenerator(config)
        results = generator.generate(max_trials=args.max_trials, verbose=not args.quiet)
        
        # Save datasets
        dataset_path, dim_matrix_path = generator.save_datasets()
        
        # Create visualization if requested
        if args.plot:
            plot_path = generator.create_visualization(
                filename=args.plot_filename,
                show=False
            )
            if not args.quiet:
                print(f"Plot: {os.path.abspath(plot_path)}")
        
        if not args.quiet:
            print(f"\n=== Files Saved ===")
            print(f"Dataset: {os.path.abspath(dataset_path)}")
            print(f"Dimension matrix: {os.path.abspath(dim_matrix_path)}")
            print(f"\nOutput structure:")
            print(f"  {os.path.abspath(config.output_dir)}/")
            print(f"    ├── {config.data_dir}/")
            print(f"    │   ├── {config.dataset_filename}")
            print(f"    │   └── {config.dimension_matrix_filename}")
            if args.plot:
                print(f"    └── {config.figures_dir}/")
                print(f"        └── {args.plot_filename}")
        
    except Exception as e:
        print(f"Error during data generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

