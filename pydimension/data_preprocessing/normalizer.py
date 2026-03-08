"""
Data normalization strategies for the preprocessing stage.
"""

from typing import List

import numpy as np
import pandas as pd


def normalize_by_max(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Normalize selected columns by dividing each by its maximum value."""
    result = df[columns].copy()
    for col in columns:
        max_val = result[col].max()
        if max_val != 0:
            result[col] = result[col] / max_val
        else:
            result[col] = 0
    print(f"✅ Normalized data")
    print(f"   Shape: {result.shape}")
    print(f"   Value range: [{result.min().min():.4f}, {result.max().max():.4f}]")
    return result
