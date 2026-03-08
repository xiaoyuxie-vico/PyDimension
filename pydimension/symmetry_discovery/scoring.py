"""
Symmetry scoring utilities.

After an encoder has been trained or run, these helpers quantify how strongly
a specific symmetry class is expressed in the data.  The scoring contract is
intentionally simple so that new metrics can be added without touching the
encoder implementations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SymmetryScore:
    """Single score for one symmetry class."""

    symmetry_type: str
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def rank_symmetries(scores: List[SymmetryScore]) -> List[SymmetryScore]:
    """Sort symmetry scores from most to least plausible."""
    return sorted(scores, key=lambda s: s.value, reverse=True)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R^2) for measuring fit quality."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)
