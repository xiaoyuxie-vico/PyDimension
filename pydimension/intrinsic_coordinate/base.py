"""
Base interfaces for intrinsic coordinate discovery.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class IntrinsicCoordinateArtifacts:
    """Outputs shared by intrinsic coordinate methods."""

    suggested_count: Optional[int] = None
    pca_results: Optional[Dict[str, Any]] = None
    sir_results: Optional[Dict[str, Any]] = None
    method_results: Optional[Dict[str, Any]] = None


class BaseIntrinsicCoordinateMethod:
    """Common interface for latent-coordinate discovery methods."""

    method_name = "base"

    def run(self, verbose: bool = False) -> IntrinsicCoordinateArtifacts:
        raise NotImplementedError
