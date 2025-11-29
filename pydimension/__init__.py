"""
PyDimension 2.0 - Dimensionless Learning Package

A comprehensive Python package for discovering dimensionless relationships
in physical systems using machine learning and dimensional analysis.
"""

__version__ = "2.0.0"

# Import main classes from each module
from .data_generation import DataGenerator, DataGenerationConfig
from .data_preprocessing import DataPreprocessor, DataPreprocessingConfig
from .dimensional_analysis import DimensionalAnalyzer, DimensionalAnalysisConfig
from .constraint_filtering import ConstraintFilterer, ConstraintFilteringConfig
from .optimization_discovery import OptimizationDiscoverer, OptimizationDiscoveryConfig

__all__ = [
    # Version
    "__version__",
    # Data Generation
    "DataGenerator",
    "DataGenerationConfig",
    # Data Preprocessing
    "DataPreprocessor",
    "DataPreprocessingConfig",
    # Dimensional Analysis
    "DimensionalAnalyzer",
    "DimensionalAnalysisConfig",
    # Dimensional Filtering
    "ConstraintFilterer",
    "ConstraintFilteringConfig",
    # Optimization Discovery
    "OptimizationDiscoverer",
    "OptimizationDiscoveryConfig",
]

