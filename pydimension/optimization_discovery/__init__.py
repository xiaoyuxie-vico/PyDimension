"""
Optimization and Discovery module for PyDimension 2.0.

This module trains neural networks to discover dimensionless scaling laws
from preprocessed data.
"""

from .optimizer import OptimizationDiscoverer
from .config import OptimizationDiscoveryConfig

__all__ = ['OptimizationDiscoverer', 'OptimizationDiscoveryConfig']

