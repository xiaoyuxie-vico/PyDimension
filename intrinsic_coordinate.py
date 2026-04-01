#!/usr/bin/env python3
"""
Run the PyDimension 3.0 intrinsic-coordinate stage.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pydimension.intrinsic_coordinate import IntrinsicCoordinateConfig, IntrinsicCoordinateFinder


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the intrinsic-coordinate stage for PyDimension 3.0")
    parser.add_argument(
        "--config",
        "-c",
        default="pydimension/configs/config_translation_v3.json",
        help="Path to a PyDimension 3.0 config file.",
    )
    args = parser.parse_args()

    config = IntrinsicCoordinateConfig.from_json(args.config)
    errors = config.validate()
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 1

    finder = IntrinsicCoordinateFinder(config)
    artifacts = finder.process(verbose=True)
    print(f"\nSuggested intrinsic dimension: {artifacts.suggested_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
