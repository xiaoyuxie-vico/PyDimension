"""
PyDimension 2.0 benchmark package.

This package provides a stable import surface for the preserved
dimensionless-learning implementation while PyDimension 3.0 is developed.
"""

__version__ = "2.0.0-benchmark"

from pydimension.data_generation import DataGenerator, DataGenerationConfig
from pydimension.data_preprocessing import DataPreprocessor, DataPreprocessingConfig
from .dimensional_analysis import DimensionalAnalyzer, DimensionalAnalysisConfig
from .constraint_filtering import ConstraintFilterer, ConstraintFilteringConfig
from .optimization_discovery import OptimizationDiscoverer, OptimizationDiscoveryConfig

__all__ = [
    "__version__",
    "DataGenerator",
    "DataGenerationConfig",
    "DataPreprocessor",
    "DataPreprocessingConfig",
    "DimensionalAnalyzer",
    "DimensionalAnalysisConfig",
    "ConstraintFilterer",
    "ConstraintFilteringConfig",
    "OptimizationDiscoverer",
    "OptimizationDiscoveryConfig",
]
