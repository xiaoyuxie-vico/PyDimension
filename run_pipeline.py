#!/usr/bin/env python3
"""
Run the complete PyDimension 2.0 pipeline.

This script runs all modules in sequence:
1. Data Generation
2. Data Preprocessing
3. Dimensional Analysis
4. Dimensional Filtering
5. Optimization and Discovery

Each step uses the output from the previous step automatically.
"""

import sys
from pathlib import Path
import argparse

# Add pydimension to path
sys.path.insert(0, str(Path(__file__).parent))

from pydimension.data_generation import DataGenerator, DataGenerationConfig
from pydimension.data_preprocessing import DataPreprocessor, DataPreprocessingConfig
from pydimension.dimensional_analysis import DimensionalAnalyzer, DimensionalAnalysisConfig
from pydimension.constraint_filtering import ConstraintFilterer, ConstraintFilteringConfig
from pydimension.optimization_discovery import OptimizationDiscoverer, OptimizationDiscoveryConfig


def run_step(step_name, step_func, *args, **kwargs):
    """Run a pipeline step with error handling."""
    print(f"\n{'='*70}")
    print(f"STEP: {step_name}")
    print(f"{'='*70}\n")
    
    try:
        result = step_func(*args, **kwargs)
        print(f"\n✅ {step_name} completed successfully")
        return result, None
    except Exception as e:
        error_msg = f"❌ {step_name} failed: {str(e)}"
        print(f"\n{error_msg}")
        import traceback
        traceback.print_exc()
        return None, error_msg


def main():
    """Run the complete PyDimension 2.0 pipeline."""
    parser = argparse.ArgumentParser(
        description='Run the complete PyDimension 2.0 pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with default config
  python run_pipeline.py --config pydimension/configs/config_synthetic.json
  
  # Run with visualization
  python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot
  
  # Run without stopping on errors
  python run_pipeline.py --config config_synthetic.json --continue-on-error
  
  # Skip specific steps
  python run_pipeline.py --config config_synthetic.json --skip data_generation --skip data_preprocessing
        """
    )
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to unified JSON config file. '
                            'If not specified, uses default configuration.')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base output directory (overrides config, default: "output")')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save visualization plots for all steps')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue to next step even if current step fails')
    parser.add_argument('--skip', action='append', dest='skip_steps',
                       choices=['data_generation', 'data_preprocessing', 'dimensional_analysis',
                               'constraint_filtering', 'optimization_discovery'],
                       help='Skip specific steps (can be used multiple times)')
    parser.add_argument('--stop-after', type=str,
                       choices=['data_generation', 'data_preprocessing', 'dimensional_analysis',
                               'constraint_filtering', 'optimization_discovery'],
                       help='Stop after completing the specified step')
    
    args = parser.parse_args()
    
    skip_steps = set(args.skip_steps or [])
    
    print("="*70)
    print("PyDimension 2.0 - Complete Pipeline")
    print("="*70)
    
    # Load config
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}")
            return 1
        config_path = args.config
        print(f"\n📋 Using config file: {config_path}")
    else:
        config_path = None
        print("\n⚠️  No config file specified. Using default configuration.")
        print("   Example: python run_pipeline.py --config pydimension/configs/config_synthetic.json")
    
    # Track results
    results = {}
    errors = []
    
    # ========================================================================
    # Step 1: Data Generation
    # ========================================================================
    if 'data_generation' not in skip_steps:
        def step1():
            if config_path:
                config = DataGenerationConfig.from_json(config_path)
            else:
                config = DataGenerationConfig()
            
            if args.output_dir:
                config.output_dir = args.output_dir
            
            generator = DataGenerator(config)
            results_data = generator.generate(verbose=True)
            dataset_path, dim_matrix_path = generator.save_datasets()
            
            if args.plot:
                plot_filename = 'data_generation_plots.png'
                generator.create_visualization(filename=plot_filename)
            
            return {'dataset_path': dataset_path, 'dim_matrix_path': dim_matrix_path}
        
        result, error = run_step("Data Generation", step1)
        results['data_generation'] = result
        if error:
            errors.append(error)
            if not args.continue_on_error:
                print("\n❌ Pipeline stopped due to error. Use --continue-on-error to continue.")
                return 1
        
        if args.stop_after == 'data_generation':
            print("\n✅ Pipeline stopped after data generation (--stop-after)")
            return 0
    else:
        print("\n⏭️  Skipping: Data Generation")
    
    # ========================================================================
    # Step 2: Data Preprocessing
    # ========================================================================
    if 'data_preprocessing' not in skip_steps:
        def step2():
            if config_path:
                config = DataPreprocessingConfig.from_json(config_path)
            else:
                config = DataPreprocessingConfig()
            
            if args.output_dir:
                config.output_dir = args.output_dir
            
            preprocessor = DataPreprocessor(config)
            preprocessor.process(verbose=True)
            preprocessor.save_results()
            
            if args.plot:
                plot_filename = 'data_preprocessing_plots.png'
                preprocessor.create_visualization(filename=plot_filename)
            
            return {}
        
        result, error = run_step("Data Preprocessing", step2)
        results['data_preprocessing'] = result
        if error:
            errors.append(error)
            if not args.continue_on_error:
                print("\n❌ Pipeline stopped due to error. Use --continue-on-error to continue.")
                return 1
        
        if args.stop_after == 'data_preprocessing':
            print("\n✅ Pipeline stopped after data preprocessing (--stop-after)")
            return 0
    else:
        print("\n⏭️  Skipping: Data Preprocessing")
    
    # ========================================================================
    # Step 3: Dimensional Analysis
    # ========================================================================
    if 'dimensional_analysis' not in skip_steps:
        def step3():
            if config_path:
                config = DimensionalAnalysisConfig.from_json(config_path)
            else:
                config = DimensionalAnalysisConfig()
            
            if args.output_dir:
                config.output_dir = args.output_dir
            
            analyzer = DimensionalAnalyzer(config)
            analyzer.process(verbose=True)
            analyzer.save_results()
            analyzer.save_normalized_lg_data()  # Important: needed for dimensional filtering
            
            if args.plot:
                plot_filename = 'dimensional_analysis_plots.png'
                analyzer.create_visualization(filename=plot_filename)
            
            return {}
        
        result, error = run_step("Dimensional Analysis", step3)
        results['dimensional_analysis'] = result
        if error:
            errors.append(error)
            if not args.continue_on_error:
                print("\n❌ Pipeline stopped due to error. Use --continue-on-error to continue.")
                return 1
        
        if args.stop_after == 'dimensional_analysis':
            print("\n✅ Pipeline stopped after dimensional analysis (--stop-after)")
            return 0
    else:
        print("\n⏭️  Skipping: Dimensional Analysis")
    
    # ========================================================================
    # Step 4: Dimensional Filtering
    # ========================================================================
    if 'constraint_filtering' not in skip_steps:
        def step4():
            if config_path:
                config = ConstraintFilteringConfig.from_json(config_path)
            else:
                config = ConstraintFilteringConfig()
            
            if args.output_dir:
                config.output_dir = args.output_dir
            
            filterer = ConstraintFilterer(config)
            filterer.process(verbose=True)
            filterer.save_results()
            filterer.save_suggested_count()  # Important: needed for optimization discovery
            
            if args.plot:
                plot_filename = 'constraint_filtering_plots.png'
                filterer.create_visualization(filename=plot_filename)
            
            return {}
        
        result, error = run_step("Dimensional Filtering", step4)
        results['constraint_filtering'] = result
        if error:
            errors.append(error)
            if not args.continue_on_error:
                print("\n❌ Pipeline stopped due to error. Use --continue-on-error to continue.")
                return 1
        
        if args.stop_after == 'constraint_filtering':
            print("\n✅ Pipeline stopped after dimensional filtering (--stop-after)")
            return 0
    else:
        print("\n⏭️  Skipping: Dimensional Filtering")
    
    # ========================================================================
    # Step 5: Optimization and Discovery
    # ========================================================================
    if 'optimization_discovery' not in skip_steps:
        def step5():
            if config_path:
                config = OptimizationDiscoveryConfig.from_json(config_path)
            else:
                config = OptimizationDiscoveryConfig()
            
            if args.output_dir:
                config.output_dir = args.output_dir
            
            optimizer = OptimizationDiscoverer(config)
            optimizer.process(verbose=True)
            optimizer.save_results()
            
            if args.plot:
                plot_filename = 'optimization_discovery_plots.png'
                optimizer.create_visualization(filename=plot_filename)
            
            return {}
        
        result, error = run_step("Optimization and Discovery", step5)
        results['optimization_discovery'] = result
        if error:
            errors.append(error)
            if not args.continue_on_error:
                print("\n❌ Pipeline stopped due to error. Use --continue-on-error to continue.")
                return 1
    else:
        print("\n⏭️  Skipping: Optimization and Discovery")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    completed_steps = [step for step, result in results.items() if result is not None]
    failed_steps = [step for step, result in results.items() if result is None]
    
    print(f"\n✅ Completed steps: {len(completed_steps)}")
    for step in completed_steps:
        print(f"   - {step.replace('_', ' ').title()}")
    
    if failed_steps:
        print(f"\n❌ Failed steps: {len(failed_steps)}")
        for step in failed_steps:
            print(f"   - {step.replace('_', ' ').title()}")
    
    if errors:
        print(f"\n⚠️  Errors encountered: {len(errors)}")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")
    
    if len(completed_steps) == 5:
        print("\n🎉 Pipeline completed successfully!")
        return 0
    elif len(completed_steps) > 0:
        print(f"\n⚠️  Pipeline completed with {len(failed_steps)} failed step(s)")
        return 1
    else:
        print("\n❌ Pipeline failed completely")
        return 1


if __name__ == '__main__':
    sys.exit(main())

