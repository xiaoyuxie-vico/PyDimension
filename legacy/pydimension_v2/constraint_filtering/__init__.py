"""
Legacy Constraint Filtering Module

Identify dominant dimensionless groups using PCA and SIR analysis.
This package is retained for PyDimension 2.0 compatibility.
"""

from .filterer import ConstraintFilterer
from .config import ConstraintFilteringConfig

__all__ = ['ConstraintFilterer', 'ConstraintFilteringConfig']

