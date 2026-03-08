"""
Data loading utilities for the preprocessing stage.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def load_csv_data(input_file: str) -> pd.DataFrame:
    """Load a CSV dataset with basic validation."""
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    df = pd.read_csv(path)
    print(f"✅ Loaded data from: {path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df


def detect_variables(
    df: pd.DataFrame,
    input_variables: Optional[List[str]] = None,
    output_variables: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Auto-detect or validate input and output variable columns."""
    exclude_cols = {"case", "source"}
    all_vars = [col for col in df.columns if col not in exclude_cols]

    if input_variables is None:
        input_vars = [v for v in all_vars if v.startswith("p") and v[1:].isdigit()]
        if not input_vars:
            output_candidates = [v for v in all_vars if v.endswith("*")]
            input_vars = [v for v in all_vars if v not in output_candidates]
    else:
        input_vars = [v for v in input_variables if v in all_vars]

    if output_variables is None:
        output_vars = [v for v in all_vars if v in ["p*", "e*", "Ke"] or v.endswith("*")]
        if not output_vars:
            remaining = [v for v in all_vars if v not in input_vars]
            output_vars = [remaining[-1]] if remaining else []
    else:
        output_vars = [v for v in output_variables if v in all_vars]

    print(f"✅ Detected variables:")
    print(f"   Input variables ({len(input_vars)}): {input_vars}")
    print(f"   Output variables ({len(output_vars)}): {output_vars}")
    return input_vars, output_vars
