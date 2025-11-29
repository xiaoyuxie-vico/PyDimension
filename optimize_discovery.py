#!/usr/bin/env python3
"""
Convenience script for running optimization and discovery module.

This script provides a simple interface to run the optimization and discovery
module directly from the project root.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pydimension.optimization_discovery import OptimizationDiscoverer, OptimizationDiscoveryConfig


def main():
    """Main entry point for convenience script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimization and Discovery: train neural networks to discover dimensionless scaling laws',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using unified config file
  python optimize_discovery.py --config pydimension/configs/config_synthetic.json --plot
  
  # Specify input file directly
  python optimize_discovery.py \\
      --input_file output/data/normalized_lg_afterDA_data.csv \\
      --num_linear 1 \\
      --epochs 1000 \\
      --plot
  
  # Custom architecture with gamma regularization
  python optimize_discovery.py \\
      --input_file normalized_lg_afterDA_data.csv \\
      --num_linear 2 \\
      --num_hidden_layers 4 \\
      --num_hidden_nodes 10 \\
      --num_ensembles 5 \\
      --gamma_reg_strength 0.01 \\
      --plot
        """
    )
    
    # Config file
    parser.add_argument('--config', '-c', type=str, default='pydimension/configs/config_synthetic.json',
                       help='Path to unified JSON config file (default: pydimension/configs/config_synthetic.json)')
    
    # Input settings
    parser.add_argument('--input_file', type=str, default=None,
                       help='Path to normalized lg afterDA data CSV file')
    parser.add_argument('--basis_vectors_file', type=str, default=None,
                       help='Path to basis_vectors.csv (optional, for fixed gamma)')
    
    # Model architecture
    parser.add_argument('--num_linear', type=int, default=None,
                       help='Number of linear nodes (gamma vectors)')
    parser.add_argument('--num_hidden_layers', type=int, default=None,
                       help='Number of hidden layers')
    parser.add_argument('--num_hidden_nodes', type=int, default=None,
                       help='Number of nodes per hidden layer')
    
    # Training settings
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--train_percent', type=float, default=None,
                       help='Train/test split ratio (0-1)')
    
    # Ensemble settings
    parser.add_argument('--num_ensembles', type=int, default=None,
                       help='Number of ensemble models')
    
    # Gamma settings
    parser.add_argument('--use_fixed_gamma', action='store_true',
                       help='Use fixed gamma values from basis vectors')
    parser.add_argument('--use_gamma_regularization', action='store_true', default=None,
                       help='Enable gamma regularization (soft quantization)')
    parser.add_argument('--no_gamma_regularization', action='store_true',
                       help='Disable gamma regularization')
    parser.add_argument('--gamma_reg_strength', type=float, default=None,
                       help='Gamma regularization strength')
    parser.add_argument('--gamma_reg_resolution', type=str, default=None,
                       choices=['integers', 'half-integers', 'quarter-integers'],
                       help='Gamma regularization resolution')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base output directory (overrides config)')
    
    # Visualization
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save visualization plots')
    parser.add_argument('--plot_filename', type=str, default=None,
                       help='Filename for saved plot (default: from config)')
    
    # Other
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        config = OptimizationDiscoveryConfig.from_json(args.config)
        if not args.quiet:
            print(f"Loaded config from: {args.config}")
    else:
        # Create default config
        config = OptimizationDiscoveryConfig()
        if not args.quiet:
            print(f"Config file not found: {args.config}")
            print("Using default configuration.")
    
    # Override config with command-line arguments
    if args.input_file:
        config.input_file = args.input_file
    if args.basis_vectors_file:
        config.basis_vectors_file = args.basis_vectors_file
    if args.num_linear:
        config.num_linear = args.num_linear
    if args.num_hidden_layers:
        config.num_hidden_layers = args.num_hidden_layers
    if args.num_hidden_nodes:
        config.num_hidden_nodes = args.num_hidden_nodes
    if args.random_seed:
        config.random_seed = args.random_seed
    if args.epochs:
        config.epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.train_percent:
        config.train_percent = args.train_percent
    if args.num_ensembles:
        config.num_ensembles = args.num_ensembles
    if args.use_fixed_gamma:
        config.use_fixed_gamma = True
    if args.use_gamma_regularization is not None:
        config.use_gamma_regularization = args.use_gamma_regularization
    if args.no_gamma_regularization:
        config.use_gamma_regularization = False
    if args.gamma_reg_strength:
        config.gamma_reg_strength = args.gamma_reg_strength
    if args.gamma_reg_resolution:
        config.gamma_reg_resolution = args.gamma_reg_resolution
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Validate config
    errors = config.validate()
    if errors:
        if not args.quiet:
            print("\n❌ Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    
    # Create optimizer and run
    try:
        optimizer = OptimizationDiscoverer(config)
        results = optimizer.process(verbose=not args.quiet)
        
        # Save results
        results_path, results_dir = optimizer.save_results()
        
        # Create visualization if requested
        if args.plot:
            plot_filename = args.plot_filename
            if plot_filename is None:
                # Try to get from config if available
                try:
                    import json
                    if Path(args.config).exists():
                        with open(args.config, 'r') as f:
                            full_config = json.load(f)
                            plot_filename = full_config.get('OPTIMIZATION_DISCOVERY_OUTPUT', {}).get('plot_filename')
                except:
                    pass
                if plot_filename is None:
                    plot_filename = 'optimization_discovery_plots.png'
            
            plot_path = optimizer.create_visualization(filename=plot_filename)
            if not args.quiet:
                print(f"Plot: {plot_path}")
        
        if not args.quiet:
            print(f"\n✅ Optimization and discovery complete!")
            import numpy as np
            if optimizer.model_r2_scores is not None:
                best_idx = int(np.argmax(optimizer.model_r2_scores))
                print(f"   Best model: {best_idx + 1} (R² = {optimizer.model_r2_scores[best_idx]:.6f})")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        if not args.quiet:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

