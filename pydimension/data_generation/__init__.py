"""
Data Generation Module

Generate synthetic datasets with known dimensionless relationships
for testing dimensionless learning pipelines.
"""

from .generator import DataGenerator
from .config import DataGenerationConfig

__all__ = ['DataGenerator', 'DataGenerationConfig']

