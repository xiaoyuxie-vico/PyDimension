"""
Dispatcher for intrinsic coordinate discovery methods.
"""

from .autoencoder import AutoencoderIntrinsicCoordinate
from .base import IntrinsicCoordinateArtifacts
from .config import IntrinsicCoordinateConfig
from .pca import PCAAndSIRIntrinsicCoordinate
from .sir import SIRIntrinsicCoordinate


class IntrinsicCoordinateFinder:
    """Route the renamed stage to its selected backend method."""

    def __init__(self, config: IntrinsicCoordinateConfig):
        self.config = config
        if config.method == "autoencoder":
            self.method = AutoencoderIntrinsicCoordinate(config)
        elif config.method == "sir":
            self.method = SIRIntrinsicCoordinate(config)
        else:
            self.method = PCAAndSIRIntrinsicCoordinate(config)

    def process(self, verbose: bool = False) -> IntrinsicCoordinateArtifacts:
        return self.method.run(verbose=verbose)
