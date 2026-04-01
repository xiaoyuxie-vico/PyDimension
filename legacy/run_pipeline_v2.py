#!/usr/bin/env python3
"""
Run the preserved PyDimension 2.0 benchmark pipeline.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legacy.pydimension_v2 import (
    ConstraintFilterer,
    ConstraintFilteringConfig,
    DataGenerationConfig,
    DataGenerator,
    DataPreprocessingConfig,
    DataPreprocessor,
    DimensionalAnalysisConfig,
    DimensionalAnalyzer,
    OptimizationDiscoverer,
    OptimizationDiscoveryConfig,
)


def _run_step(name, func):
    print(f"\n{'=' * 70}\nSTEP: {name}\n{'=' * 70}\n")
    result = func()
    print(f"✅ {name} completed successfully")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the legacy PyDimension 2.0 benchmark pipeline")
    parser.add_argument(
        "--config",
        "-c",
        default="legacy/pydimension_v2/configs/config_synthetic_v2.json",
        help="Path to the benchmark config file.",
    )
    parser.add_argument("--output_dir", default=None, help="Override the output directory.")
    parser.add_argument("--plot", action="store_true", help="Generate plots when supported.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    print("=" * 70)
    print("PyDimension 2.0 Benchmark Pipeline")
    print("=" * 70)
    print(f"Using config: {config_path}")

    generation_config = DataGenerationConfig.from_json(str(config_path))
    preprocessing_config = DataPreprocessingConfig.from_json(str(config_path))
    analysis_config = DimensionalAnalysisConfig.from_json(str(config_path))
    filtering_config = ConstraintFilteringConfig.from_json(str(config_path))
    discovery_config = OptimizationDiscoveryConfig.from_json(str(config_path))

    if args.output_dir:
        for config in (
            generation_config,
            preprocessing_config,
            analysis_config,
            filtering_config,
            discovery_config,
        ):
            config.output_dir = args.output_dir

    generator = DataGenerator(generation_config)
    generation_artifacts = _run_step(
        "Data Generation",
        lambda: (generator.generate(verbose=True), generator.save_datasets()),
    )
    if args.plot:
        generator.create_visualization()

    dataset_path, dimension_matrix_path = generation_artifacts[1]
    preprocessing_config.input_file = dataset_path
    preprocessing_config.dimension_matrix_file = dimension_matrix_path

    preprocessor = DataPreprocessor(preprocessing_config)
    _run_step("Data Preprocessing", lambda: (preprocessor.process(verbose=True), preprocessor.save_results()))
    if args.plot:
        preprocessor.create_visualization()

    normalized_data_file = str(
        Path(preprocessing_config.output_dir)
        / preprocessing_config.data_dir
        / preprocessing_config.normalized_data_filename
    )
    dimension_matrix_file = str(
        Path(preprocessing_config.output_dir)
        / preprocessing_config.data_dir
        / preprocessing_config.dimension_matrix_filename
    )
    analysis_config.normalized_data_file = normalized_data_file
    analysis_config.dimension_matrix_file = dimension_matrix_file

    analyzer = DimensionalAnalyzer(analysis_config)
    _run_step(
        "Dimensional Analysis",
        lambda: (
            analyzer.process(verbose=True),
            analyzer.save_results(),
            analyzer.save_normalized_lg_data(),
        ),
    )
    if args.plot:
        analyzer.create_visualization()

    normalized_lg_file = str(
        Path(analysis_config.output_dir)
        / analysis_config.data_dir
        / analysis_config.normalized_lg_data_filename
    )
    basis_vectors_file = str(
        Path(analysis_config.output_dir)
        / analysis_config.data_dir
        / analysis_config.basis_vectors_filename
    )
    filtering_config.input_file = normalized_lg_file

    filterer = ConstraintFilterer(filtering_config)
    _run_step(
        "Constraint Filtering",
        lambda: (filterer.process(verbose=True), filterer.save_results(), filterer.save_suggested_count()),
    )
    if args.plot:
        filterer.create_visualization()

    discovery_config.input_file = normalized_lg_file
    discovery_config.basis_vectors_file = basis_vectors_file

    optimizer = OptimizationDiscoverer(discovery_config)
    _run_step("Optimization Discovery", lambda: (optimizer.process(verbose=True), optimizer.save_results()))
    if args.plot:
        optimizer.create_visualization()

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
