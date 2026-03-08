"""
Base encoder interface shared by all symmetry-specific encoders.

Every encoder must implement ``discover()`` and return a
``SymmetryDiscoveryArtifacts`` object.  The base class also defines optional
hooks for ``fit``, ``transform``, ``score``, and ``export_parameters`` that
concrete encoders can override.
"""

from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseSymmetryEncoder, SymmetryDiscoveryArtifacts


class SymmetryEncoderBase(BaseSymmetryEncoder):
    """Extended base with optional train / transform / score hooks.

    Subclasses should override whichever hooks make sense for the symmetry
    class they handle.  The ``discover`` method can be left to compose these
    hooks in order.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SymmetryEncoderBase":
        """Fit the encoder on input data (override in subclass)."""
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data into symmetry-aware coordinates (override in subclass)."""
        raise NotImplementedError

    def score(self) -> Dict[str, float]:
        """Return metrics that quantify the discovered symmetry."""
        return {}

    def export_parameters(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of learned parameters."""
        return {}
