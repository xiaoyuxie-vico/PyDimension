"""
Symmetry discovery module for PyDimension 3.0.
"""

from .base import BaseSymmetryEncoder, SymmetryDiscoveryArtifacts
from .config import SymmetryDiscoveryConfig
from .engine import SymmetryDiscoveryEngine
from .scoring import SymmetryScore, rank_symmetries, r2_score
from .relation_heads import BaseRelationHead, RelationResult

__all__ = [
    "BaseSymmetryEncoder",
    "SymmetryDiscoveryArtifacts",
    "SymmetryDiscoveryConfig",
    "SymmetryDiscoveryEngine",
    "SymmetryScore",
    "rank_symmetries",
    "r2_score",
    "BaseRelationHead",
    "RelationResult",
]
