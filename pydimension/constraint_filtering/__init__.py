"""
Dimensional Filtering Module

Identify dominant dimensionless groups using PCA and SIR analysis.
"""

from .filterer import ConstraintFilterer
from .config import ConstraintFilteringConfig

__all__ = ['ConstraintFilterer', 'ConstraintFilteringConfig']

