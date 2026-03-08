"""
Coordinate transforms used during preprocessing (log, normalization, etc.).
"""

from typing import List, Optional

import numpy as np
import pandas as pd


def log10_normalize_pi_groups(
    afterDA_data: pd.DataFrame,
    output_variable: Optional[str] = None,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Normalize π columns to [0, 1], then take log10.

    Returns a DataFrame with columns renamed to ``lgπ1``, ``lgπ2``, etc., plus
    the (normalized) output column if present.
    """
    pi_cols = [c for c in afterDA_data.columns if c.startswith("π")]
    work = afterDA_data.copy()

    for col in pi_cols:
        mx = max(work[col].max(), epsilon)
        work[col] = work[col] / mx

    if output_variable and output_variable in work.columns:
        mx = max(work[output_variable].max(), epsilon)
        work[output_variable] = work[output_variable] / mx

    for col in pi_cols:
        work[col] = np.log10(np.maximum(work[col], epsilon))

    rename_map = {c: f"lg{c}" for c in pi_cols}
    work = work.rename(columns=rename_map)
    ordered = list(rename_map.values()) + ([output_variable] if output_variable and output_variable in work.columns else [])
    return work[ordered]
