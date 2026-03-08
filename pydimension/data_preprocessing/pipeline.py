"""
Unified preprocessing entrypoint for PyDimension 3.0.
"""

from pathlib import Path

from .base import PreprocessingArtifacts
from .config import DataPreprocessingConfig
from .preprocessor import DataPreprocessor


def run_dimensional_analysis_preprocessing(
    config: DataPreprocessingConfig,
    verbose: bool = False,
) -> PreprocessingArtifacts:
    """Run unified preprocessing plus dimensional analysis without a separate module."""
    preprocessor = DataPreprocessor(config)
    preprocessor.process_with_dimensional_analysis(verbose=verbose)
    preprocessor.save_results()
    preprocessor.save_dimensional_analysis_results()
    preprocessor.save_normalized_lg_data()
    base_dir = Path(preprocessor.config.output_dir) / preprocessor.config.data_dir

    return PreprocessingArtifacts(
        normalized_data_file=str(base_dir / preprocessor.config.normalized_data_filename),
        dimension_matrix_file=str(base_dir / preprocessor.config.dimension_matrix_filename),
        basis_vectors_file=str(base_dir / preprocessor.config.basis_vectors_filename),
        normalized_lg_data_file=str(base_dir / preprocessor.config.normalized_lg_data_filename),
    )


class DataPreprocessingPipeline:
    """Dispatcher for preprocessing methods that share a common output contract."""

    def __init__(self, config: DataPreprocessingConfig, method: str = "dimensional_analysis"):
        self.config = config
        self.method = method

        if method != "dimensional_analysis":
            raise ValueError(f"Unsupported preprocessing method: {method}")

    def run(self, verbose: bool = False) -> PreprocessingArtifacts:
        return run_dimensional_analysis_preprocessing(self.config, verbose=verbose)
