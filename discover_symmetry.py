#!/usr/bin/env python3
"""
Run the PyDimension 3.0 symmetry-discovery stage.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pydimension.symmetry_discovery import SymmetryDiscoveryConfig, SymmetryDiscoveryEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the symmetry-discovery stage for PyDimension 3.0")
    parser.add_argument(
        "--config",
        "-c",
        default="pydimension/configs/config_translation_v3.json",
        help="Path to a PyDimension 3.0 config file.",
    )
    args = parser.parse_args()

    config = SymmetryDiscoveryConfig.from_json(args.config)
    errors = config.validate()
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 1

    engine = SymmetryDiscoveryEngine(config)
    artifacts = engine.process(verbose=True)
    print(f"\nSymmetry type: {artifacts.symmetry_type}")
    print(f"Encoder: {artifacts.encoder_name}")
    if artifacts.results_file:
        print(f"Results file: {artifacts.results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
