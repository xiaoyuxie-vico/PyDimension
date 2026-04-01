"""
Data Generation Module

Generate synthetic datasets with known dimensionless relationships
for testing dimensionless learning pipelines.
"""

from .generator import DataGenerator
from .config import DataGenerationConfig
from .base import BaseSymmetryGenerator, SymmetryGenerationSpec
from .translational import TranslationalSymmetryGenerator
from .rotational import RotationalSymmetryGenerator
from .scaling import ScalingSymmetryGenerator

__all__ = [
    "DataGenerator",
    "DataGenerationConfig",
    "BaseSymmetryGenerator",
    "SymmetryGenerationSpec",
    "TranslationalSymmetryGenerator",
    "RotationalSymmetryGenerator",
    "ScalingSymmetryGenerator",
]

