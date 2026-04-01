"""
Base interfaces for the unified 3.0 preprocessing stage.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessingArtifacts:
    """Files emitted by the preprocessing stage."""

    normalized_data_file: Optional[str] = None
    dimension_matrix_file: Optional[str] = None
    basis_vectors_file: Optional[str] = None
    normalized_lg_data_file: Optional[str] = None


class BasePreprocessingMethod:
    """Common interface for v3 preprocessing methods."""

    method_name = "base"

    def run(self, verbose: bool = False) -> PreprocessingArtifacts:
        raise NotImplementedError
