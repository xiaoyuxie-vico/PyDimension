#!/usr/bin/env python3
"""
Pipeline dispatcher: run the 3.0 OpenSymmetry pipeline or the preserved 2.0 benchmark.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pydimension.data_generation import DataGenerationConfig, TranslationalSymmetryGenerator
from pydimension.data_preprocessing import DataPreprocessingConfig, DataPreprocessingPipeline
from pydimension.intrinsic_coordinate import IntrinsicCoordinateConfig, IntrinsicCoordinateFinder
from pydimension.symmetry_discovery import SymmetryDiscoveryConfig, SymmetryDiscoveryEngine


def _run_step(name, func):
    print(f"\n{'=' * 70}\nSTEP: {name}\n{'=' * 70}\n")
    result = func()
    print(f"\u2705 {name} completed successfully")
    return result


def run_v3(args) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    print("=" * 70)
    print("PyDimension 3.0 Pipeline")
    print("=" * 70)
    print(f"Using config: {config_path}")

    generation_config = DataGenerationConfig.from_json(str(config_path))
    preprocessing_config = DataPreprocessingConfig.from_json(str(config_path))
    intrinsic_config = IntrinsicCoordinateConfig.from_json(str(config_path))
    discovery_config = SymmetryDiscoveryConfig.from_json(str(config_path))

    if args.output_dir:
        for config in (generation_config, preprocessing_config, intrinsic_config, discovery_config):
            config.output_dir = args.output_dir

    generator = TranslationalSymmetryGenerator(generation_config)
    generation_artifacts = _run_step(
        "Data Generation",
        lambda: (generator.generate(verbose=True), generator.save_datasets()),
    )

    dataset_path, dimension_matrix_path = generation_artifacts[1]
    preprocessing_config.input_file = dataset_path
    preprocessing_config.dimension_matrix_file = dimension_matrix_path

    preprocessing = DataPreprocessingPipeline(
        preprocessing_config,
        method=preprocessing_config.preprocessing_method,
    )
    artifacts = _run_step("Data Preprocessing", lambda: preprocessing.run(verbose=True))

    if artifacts.normalized_lg_data_file is not None:
        intrinsic_config.input_file = artifacts.normalized_lg_data_file
        discovery_config.input_file = artifacts.normalized_lg_data_file
    if artifacts.basis_vectors_file is not None:
        discovery_config.basis_vectors_file = artifacts.basis_vectors_file

    intrinsic = IntrinsicCoordinateFinder(intrinsic_config)
    _run_step("Intrinsic Coordinate", lambda: intrinsic.process(verbose=True))

    discovery = SymmetryDiscoveryEngine(discovery_config)
    _run_step("Symmetry Discovery", lambda: discovery.process(verbose=True))

    print("\nPipeline completed successfully.")
    return 0


def run_v2(args) -> int:
    from legacy.run_pipeline_v2 import main as run_v2_main
    sys.argv = [sys.argv[0], "--config", args.config]
    if args.output_dir:
        sys.argv += ["--output_dir", args.output_dir]
    return run_v2_main()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PyDimension pipeline (v3 default, v2 benchmark)")
    parser.add_argument(
        "--pipeline-version",
        choices=["v2", "v3"],
        default="v3",
        help="Select pipeline version (default: v3).",
    )
    parser.add_argument(
        "--config", "-c",
        default="pydimension/configs/config_translation.json",
        help="Path to config file.",
    )
    parser.add_argument("--output_dir", default=None, help="Override the output directory.")
    args = parser.parse_args()

    if args.pipeline_version == "v2":
        return run_v2(args)
    return run_v3(args)


if __name__ == "__main__":
    raise SystemExit(main())
