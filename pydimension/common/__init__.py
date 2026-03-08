"""
Shared helpers for the PyDimension 3.0 migration path.
"""

from .artifacts import ensure_directory, output_path
from .io import read_csv, write_csv, read_json, write_json
from .paths import OutputLayout
from .plotting import setup_style, save_figure, close_all
from .validation import require_file, check_config_errors, require_columns
from .types import SymmetryMetadata, VariableInfo, DatasetDescriptor

__all__ = [
    "ensure_directory",
    "output_path",
    "read_csv",
    "write_csv",
    "read_json",
    "write_json",
    "OutputLayout",
    "setup_style",
    "save_figure",
    "close_all",
    "require_file",
    "check_config_errors",
    "require_columns",
    "SymmetryMetadata",
    "VariableInfo",
    "DatasetDescriptor",
]
