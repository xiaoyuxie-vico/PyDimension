"""
Common interfaces for symmetry-aware data generation.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SymmetryGenerationSpec:
    """Minimal symmetry descriptor used by generalized generators."""

    symmetry_type: str
    parameters: Dict[str, Any]


class BaseSymmetryGenerator:
    """Base interface for all symmetry-aware generators."""

    symmetry_type = "unknown"

    def generate(self, verbose: bool = False):
        raise NotImplementedError
