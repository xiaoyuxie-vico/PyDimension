"""
Symmetry-discovery engine that dispatches to encoder-specific implementations.
"""

from .base import SymmetryDiscoveryArtifacts
from .config import OptimizationDiscoveryConfig
from .encoders import (
    RotationalSymmetryEncoder,
    ScalingSymmetryEncoder,
    TranslationalSymmetryEncoder,
)


class SymmetryDiscoveryEngine:
    """Main entrypoint for encoder-based symmetry discovery."""

    def __init__(self, config: OptimizationDiscoveryConfig):
        self.config = config
        symmetry_type = getattr(config, "symmetry_type", "translational")
        if symmetry_type == "rotational":
            self.encoder = RotationalSymmetryEncoder(config)
        elif symmetry_type == "scaling":
            self.encoder = ScalingSymmetryEncoder(config)
        else:
            self.encoder = TranslationalSymmetryEncoder(config)

    def process(self, verbose: bool = False) -> SymmetryDiscoveryArtifacts:
        return self.encoder.discover(verbose=verbose)
