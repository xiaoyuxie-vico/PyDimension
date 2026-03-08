"""
Translational symmetry encoder backed by the original optimizer.
"""

from pathlib import Path

from legacy.pydimension_v2.optimization_discovery import OptimizationDiscoverer
from ..base import BaseSymmetryEncoder, SymmetryDiscoveryArtifacts
from ..config import SymmetryDiscoveryConfig


class TranslationalSymmetryEncoder(BaseSymmetryEncoder):
    """Compatibility path for the first PyDimension 3.0 symmetry-discovery engine."""

    symmetry_type = "translational"
    encoder_name = "translational"

    def __init__(self, config: SymmetryDiscoveryConfig):
        self.config = config
        self.optimizer = OptimizationDiscoverer(config)

    def discover(self, verbose: bool = False) -> SymmetryDiscoveryArtifacts:
        self.optimizer.process(verbose=verbose)
        self.optimizer.save_results()

        legacy_results = (
            Path(self.config.output_dir)
            / self.config.results_dir
            / "optimization_discovery_results.json"
        )
        renamed_results = (
            Path(self.config.output_dir)
            / self.config.results_dir
            / self.config.model_results_filename
        )
        if legacy_results.exists() and legacy_results != renamed_results:
            renamed_results.write_text(legacy_results.read_text(), encoding="utf-8")

        metrics = None
        if self.optimizer.model_r2_scores is not None:
            metrics = {
                "mean_r2": float(self.optimizer.model_r2_scores.mean()),
                "num_ensembles": int(len(self.optimizer.model_r2_scores)),
            }
        return SymmetryDiscoveryArtifacts(
            symmetry_type=self.symmetry_type,
            encoder_name=self.encoder_name,
            results_file=str(renamed_results if renamed_results.exists() else legacy_results),
            metrics=metrics,
        )
