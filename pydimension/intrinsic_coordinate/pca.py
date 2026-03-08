"""
PCA/SIR-based intrinsic coordinate discovery reused from the v2 filterer.
"""

from .base import BaseIntrinsicCoordinateMethod, IntrinsicCoordinateArtifacts
from .config import IntrinsicCoordinateConfig
from legacy.pydimension_v2.constraint_filtering import ConstraintFilterer, ConstraintFilteringConfig


class PCAAndSIRIntrinsicCoordinate(BaseIntrinsicCoordinateMethod):
    """Expose the renamed 3.0 interface while reusing the validated 2.0 backend."""

    method_name = "pca_sir"

    def __init__(self, config: IntrinsicCoordinateConfig):
        self.config = config
        legacy_config = ConstraintFilteringConfig(
            input_file=config.input_file,
            run_pca=config.run_pca,
            run_sir=config.run_sir,
            pca_threshold=config.pca_threshold,
            sir_threshold=config.sir_threshold,
            n_sir_slices=config.n_sir_slices,
            n_sir_directions=config.n_sir_directions,
            output_dir=config.output_dir,
            data_dir=config.data_dir,
            figures_dir=config.figures_dir,
            results_dir=config.results_dir,
            pca_results_filename=config.pca_results_filename,
            sir_results_filename=config.sir_results_filename,
            plot_filename=config.plot_filename,
            suggested_count_filename=config.suggested_count_filename,
        )
        self.filterer = ConstraintFilterer(legacy_config)

    def run(self, verbose: bool = False) -> IntrinsicCoordinateArtifacts:
        self.filterer.process(verbose=verbose)
        self.filterer.save_results()
        self.filterer.save_suggested_count()
        return IntrinsicCoordinateArtifacts(
            suggested_count=self.filterer.pca_suggested_count or self.filterer.sir_suggested_count,
            pca_results={
                "suggested_dominant_count": self.filterer.pca_suggested_count,
                "explained_variance_ratio": (
                    self.filterer.pca_explained_variance_ratio.tolist()
                    if self.filterer.pca_explained_variance_ratio is not None
                    else None
                ),
            },
            sir_results={
                "suggested_dominant_count": self.filterer.sir_suggested_count,
                "explained_variance": (
                    self.filterer.sir_explained_variance.tolist()
                    if self.filterer.sir_explained_variance is not None
                    else None
                ),
            },
        )
