"""
Relation heads: lightweight output layers that map symmetry-aware latent
representations to compact, interpretable relations.

Scaffold for future development. Planned relation types include:
- ``LinearRelationHead``: y = a * z + b
- ``PowerLawRelationHead``: y = a * z^b
- ``PolynomialRelationHead``: y = sum(a_i * z^i)

These will sit downstream of symmetry encoders and convert the learned
reduced features into human-readable formulas.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RelationResult:
    """Output of a relation head."""

    formula: str
    coefficients: Dict[str, float]
    r2: Optional[float] = None
    metadata: Dict[str, Any] = None


class BaseRelationHead:
    """Interface for relation heads."""

    relation_type = "base"

    def fit(self, z: np.ndarray, y: np.ndarray) -> "BaseRelationHead":
        raise NotImplementedError

    def predict(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def export(self) -> RelationResult:
        raise NotImplementedError
