"""
PyDimension - Symmetry Discovery from Data

A modular Python package for discovering hidden symmetries in physical
systems using data and machine learning.  Dimensionless learning is one
module within this broader framework (the scaling-symmetry case).
"""

__version__ = "3.0.0-dev"

# -------------------------------------------------------------------
# PyDimension 3.0 (OpenSymmetry) public API
# -------------------------------------------------------------------
from .data_generation import DataGenerator, DataGenerationConfig
from .data_preprocessing import (
    DataPreprocessor,
    DataPreprocessingConfig,
    DataPreprocessingPipeline,
    run_dimensional_analysis_preprocessing,
)
from .intrinsic_coordinate import (
    IntrinsicCoordinateConfig,
    IntrinsicCoordinateFinder,
)
from .symmetry_discovery import (
    SymmetryDiscoveryConfig,
    SymmetryDiscoveryEngine,
)

# -------------------------------------------------------------------
# Legacy 2.0 classes (kept for backward compatibility, now in legacy/)
# -------------------------------------------------------------------
from legacy.pydimension_v2.dimensional_analysis import DimensionalAnalyzer, DimensionalAnalysisConfig
from legacy.pydimension_v2.constraint_filtering import ConstraintFilterer, ConstraintFilteringConfig
from legacy.pydimension_v2.optimization_discovery import OptimizationDiscoverer, OptimizationDiscoveryConfig

__all__ = [
    "__version__",
    # 3.0 data generation
    "DataGenerator",
    "DataGenerationConfig",
    # 3.0 data preprocessing (with DA merged)
    "DataPreprocessor",
    "DataPreprocessingConfig",
    "DataPreprocessingPipeline",
    "run_dimensional_analysis_preprocessing",
    # 3.0 intrinsic coordinate
    "IntrinsicCoordinateConfig",
    "IntrinsicCoordinateFinder",
    # 3.0 symmetry discovery
    "SymmetryDiscoveryConfig",
    "SymmetryDiscoveryEngine",
    # Legacy 2.0
    "DimensionalAnalyzer",
    "DimensionalAnalysisConfig",
    "ConstraintFilterer",
    "ConstraintFilteringConfig",
    "OptimizationDiscoverer",
    "OptimizationDiscoveryConfig",
]
