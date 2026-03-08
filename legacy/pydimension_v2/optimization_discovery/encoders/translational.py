"""
Translational symmetry encoder backed by the original optimizer.
"""

from pathlib import Path

from ..base import BaseSymmetryEncoder, SymmetryDiscoveryArtifacts
from ..config import OptimizationDiscoveryConfig
from ..optimizer import OptimizationDiscoverer


class TranslationalSymmetryEncoder(BaseSymmetryEncoder):
    """Compatibility path for the first PyDimension 3.0 symmetry-discovery engine."""

    symmetry_type = "translational"
    encoder_name = "translational"

    def __init__(self, config: OptimizationDiscoveryConfig):
        self.config = config
        self.optimizer = OptimizationDiscoverer(config)

    def discover(self, verbose: bool = False) -> SymmetryDiscoveryArtifacts:
        self.optimizer.process(verbose=verbose)
        self.optimizer.save_results()

        results_file = (
            Path(self.config.output_dir)
            / self.config.results_dir
            / self.config.model_results_filename
        )
        metrics = None
        if self.optimizer.model_r2_scores is not None:
            metrics = {
                "mean_r2": float(self.optimizer.model_r2_scores.mean()),
                "num_ensembles": int(len(self.optimizer.model_r2_scores)),
            }
        return SymmetryDiscoveryArtifacts(
            symmetry_type=self.symmetry_type,
            encoder_name=self.encoder_name,
            results_file=str(results_file),
            metrics=metrics,
        )
