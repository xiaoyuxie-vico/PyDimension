"""
Stub encoder for future scaling symmetry discovery.
"""

import json
from pathlib import Path

from ..base import BaseSymmetryEncoder, SymmetryDiscoveryArtifacts


class ScalingSymmetryEncoder(BaseSymmetryEncoder):
    """Reserved interface for scaling symmetry discovery."""

    symmetry_type = "scaling"
    encoder_name = "scaling"

    def __init__(self, config):
        self.config = config

    def discover(self, verbose: bool = False) -> SymmetryDiscoveryArtifacts:
        results_file = Path(self.config.output_dir) / self.config.results_dir / self.config.model_results_filename
        results_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "stub",
            "symmetry_type": self.symmetry_type,
            "message": "Scaling encoder will be implemented in a later phase.",
        }
        with open(results_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        if verbose:
            print(f"✅ Wrote scaling encoder stub to: {results_file}")
        return SymmetryDiscoveryArtifacts(
            symmetry_type=self.symmetry_type,
            encoder_name=self.encoder_name,
            results_file=str(results_file),
            metrics=payload,
        )
