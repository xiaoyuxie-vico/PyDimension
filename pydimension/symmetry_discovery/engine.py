"""
Symmetry-discovery engine that dispatches to encoder-specific implementations.
"""

from .base import SymmetryDiscoveryArtifacts
from .config import SymmetryDiscoveryConfig
from .encoders import (
    RotationalSymmetryEncoder,
    ScalingSymmetryEncoder,
    TranslationalSymmetryEncoder,
)


class SymmetryDiscoveryEngine:
    """Main entrypoint for encoder-based symmetry discovery."""

    def __init__(self, config: SymmetryDiscoveryConfig):
        self.config = config
        if config.symmetry_type == "rotational":
            self.encoder = RotationalSymmetryEncoder(config)
        elif config.symmetry_type == "scaling":
            self.encoder = ScalingSymmetryEncoder(config)
        else:
            self.encoder = TranslationalSymmetryEncoder(config)

    def process(self, verbose: bool = False) -> SymmetryDiscoveryArtifacts:
        return self.encoder.discover(verbose=verbose)
