"""
Shared type definitions and dataclasses used across PyDimension modules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SymmetryMetadata:
    """Metadata describing a symmetry class attached to a dataset or result."""

    symmetry_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None


@dataclass
class VariableInfo:
    """Description of a physical variable including name, unit, and dimensions."""

    name: str
    unit: str = "dimensionless"
    dimensions: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0])

    DIMENSION_NAMES: List[str] = field(
        default=None,
        repr=False,
        init=False,
    )

    def __post_init__(self):
        self.DIMENSION_NAMES = [
            "Mass", "Length", "Time", "Temperature",
            "Current", "Amount", "Luminous",
        ]

    @property
    def is_dimensionless(self) -> bool:
        return all(d == 0 for d in self.dimensions)


@dataclass
class DatasetDescriptor:
    """Lightweight summary of a loaded dataset before processing."""

    n_samples: int
    n_features: int
    input_variables: List[str]
    output_variables: List[str]
    source_file: Optional[str] = None
    symmetry: Optional[SymmetryMetadata] = None
