"""
Data Preprocessing Module

Preprocess datasets by selecting variables, normalizing data, generating
dimension matrices, and optionally running dimensional analysis as one
unified stage.
"""

from .preprocessor import DataPreprocessor
from .config import DataPreprocessingConfig
from .base import BasePreprocessingMethod, PreprocessingArtifacts
from .pipeline import DataPreprocessingPipeline, run_dimensional_analysis_preprocessing
from .loader import load_csv_data, detect_variables
from .normalizer import normalize_by_max
from .unit_parser import infer_units, parse_dimensions
from .dimension_matrix import find_dimension_matrix, load_matrix_from_file, generate_matrix_from_units
from .transforms import log10_normalize_pi_groups

__all__ = [
    "DataPreprocessor",
    "DataPreprocessingConfig",
    "BasePreprocessingMethod",
    "PreprocessingArtifacts",
    "DataPreprocessingPipeline",
    "run_dimensional_analysis_preprocessing",
    "load_csv_data",
    "detect_variables",
    "normalize_by_max",
    "infer_units",
    "parse_dimensions",
    "find_dimension_matrix",
    "load_matrix_from_file",
    "generate_matrix_from_units",
    "log10_normalize_pi_groups",
]
