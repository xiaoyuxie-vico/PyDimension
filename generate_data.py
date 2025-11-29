#!/usr/bin/env python3
"""
Simple script to generate synthetic data using PyDimension 2.0.

This is a convenience script that can be run directly from the package root.
Uses unified config files that contain settings for all modules.
"""

import sys
from pathlib import Path

# Add pydimension to path
sys.path.insert(0, str(Path(__file__).parent))

from pydimension.data_generation import DataGenerator, DataGenerationConfig


def main():
    """Generate data from config file or use defaults."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic data with dimensionless relationships',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use unified config file
  python generate_data.py --config pydimension/configs/config_synthetic.json
  
  # Use default config
  python generate_data.py
  
  # Override output directory
  python generate_data.py --config config_synthetic.json --output_dir my_output
  
  # Generate with visualization
  python generate_data.py --config config_synthetic.json --plot
        """
    )
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to unified JSON config file. '
                            'If not specified, uses default configuration.')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Base output directory (overrides config, default: "output")')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate and save visualization plots')
    parser.add_argument('--plot_filename', type=str, default=None,
                       help='Filename for saved plot (default: from config or "data_generation_plots.png")')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}")
            return 1
        config = DataGenerationConfig.from_json(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        # Use default config
        config = DataGenerationConfig()
        print("Using default configuration. Use --config to specify a config file.")
        print("Example: python generate_data.py --config pydimension/configs/config_synthetic.json")
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Override plot filename if specified
    plot_filename = args.plot_filename
    if plot_filename is None and args.plot:
        # Try to get from config if available
        try:
            import json
            if args.config:
                with open(args.config, 'r') as f:
                    full_config = json.load(f)
                    plot_filename = full_config.get('DATA_GENERATION_OUTPUT', {}).get('plot_filename')
        except:
            pass
        if plot_filename is None:
            plot_filename = 'data_generation_plots.png'
    
    # Generate data
    generator = DataGenerator(config)
    results = generator.generate(verbose=True)
    
    # Save datasets
    dataset_path, dim_matrix_path = generator.save_datasets()
    
    # Create visualization if requested
    if args.plot:
        plot_path = generator.create_visualization(filename=plot_filename)
        print(f"Plot: {plot_path}")
    
    print(f"\n=== Files Saved ===")
    print(f"Dataset: {dataset_path}")
    print(f"Dimension matrix: {dim_matrix_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

