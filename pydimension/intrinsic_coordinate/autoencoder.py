"""
Autoencoder scaffold for future nonlinear intrinsic-coordinate discovery.
"""

from pathlib import Path
import json

from .base import BaseIntrinsicCoordinateMethod, IntrinsicCoordinateArtifacts
from .config import IntrinsicCoordinateConfig


class AutoencoderIntrinsicCoordinate(BaseIntrinsicCoordinateMethod):
    """Initial scaffold that records the requested latent-space configuration."""

    method_name = "autoencoder"

    def __init__(self, config: IntrinsicCoordinateConfig):
        self.config = config

    def run(self, verbose: bool = False) -> IntrinsicCoordinateArtifacts:
        output_path = Path(self.config.output_dir) / self.config.results_dir / self.config.autoencoder_results_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "scaffold",
            "message": "Autoencoder-based intrinsic coordinate discovery is reserved for a later phase.",
            "input_file": self.config.input_file,
            "latent_dim": self.config.latent_dim,
        }
        with open(output_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        if verbose:
            print(f"✅ Wrote autoencoder scaffold results to: {output_path}")
        return IntrinsicCoordinateArtifacts(
            suggested_count=self.config.latent_dim,
            method_results=payload,
        )
