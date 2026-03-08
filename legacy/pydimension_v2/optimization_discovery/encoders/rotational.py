"""
Stub encoder for future rotational symmetry discovery.
"""

from pathlib import Path
import json

from ..base import BaseSymmetryEncoder, SymmetryDiscoveryArtifacts


class RotationalSymmetryEncoder(BaseSymmetryEncoder):
    """Reserved interface for rotational symmetry discovery."""

    symmetry_type = "rotational"
    encoder_name = "rotational"

    def __init__(self, config):
        self.config = config

    def discover(self, verbose: bool = False) -> SymmetryDiscoveryArtifacts:
        results_file = Path(self.config.output_dir) / self.config.results_dir / "rotational_encoder_stub.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "stub",
            "symmetry_type": self.symmetry_type,
            "message": "Rotational encoder will be implemented in a later phase.",
        }
        with open(results_file, "w") as handle:
            json.dump(payload, handle, indent=2)
        if verbose:
            print(f"✅ Wrote rotational encoder stub to: {results_file}")
        return SymmetryDiscoveryArtifacts(
            symmetry_type=self.symmetry_type,
            encoder_name=self.encoder_name,
            results_file=str(results_file),
            metrics=payload,
        )
