"""
SIR-only intrinsic coordinate discovery via the v2 filterer backend.

This method class wraps the legacy ConstraintFilterer with PCA disabled, so
that only Sliced Inverse Regression (SIR) is used to estimate the number and
orientation of intrinsic coordinates.
"""

from .base import BaseIntrinsicCoordinateMethod, IntrinsicCoordinateArtifacts
from .config import IntrinsicCoordinateConfig
from legacy.pydimension_v2.constraint_filtering import ConstraintFilterer, ConstraintFilteringConfig


class SIRIntrinsicCoordinate(BaseIntrinsicCoordinateMethod):
    """SIR-only intrinsic coordinate method backed by the 2.0 filterer."""

    method_name = "sir"

    def __init__(self, config: IntrinsicCoordinateConfig):
        self.config = config
        legacy_config = ConstraintFilteringConfig(
            input_file=config.input_file,
            run_pca=False,
            run_sir=True,
            sir_threshold=config.sir_threshold,
            n_sir_slices=config.n_sir_slices,
            n_sir_directions=config.n_sir_directions,
            output_dir=config.output_dir,
            data_dir=config.data_dir,
            figures_dir=config.figures_dir,
            results_dir=config.results_dir,
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
            suggested_count=self.filterer.sir_suggested_count,
            sir_results={
                "suggested_dominant_count": self.filterer.sir_suggested_count,
                "explained_variance": (
                    self.filterer.sir_explained_variance.tolist()
                    if self.filterer.sir_explained_variance is not None
                    else None
                ),
            },
        )
