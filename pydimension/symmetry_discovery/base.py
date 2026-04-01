"""
Base interfaces for symmetry-discovery encoders.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SymmetryDiscoveryArtifacts:
    """Shared output payload for symmetry-discovery runs."""

    symmetry_type: str
    encoder_name: str
    results_file: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class BaseSymmetryEncoder:
    """Common interface for encoders that transform data before discovery."""

    symmetry_type = "unknown"
    encoder_name = "base"

    def discover(self, verbose: bool = False) -> SymmetryDiscoveryArtifacts:
        raise NotImplementedError
