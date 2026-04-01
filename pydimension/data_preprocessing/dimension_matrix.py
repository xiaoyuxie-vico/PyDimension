"""
Dimension-matrix construction and loading for the preprocessing stage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .unit_parser import infer_units, parse_dimensions

DIMENSION_NAMES = ["Mass", "Length", "Time", "Temperature", "Current", "Amount", "Luminous"]


def load_matrix_from_file(
    matrix_path: Path,
    variables: List[str],
) -> Dict[str, List[int]]:
    """Load a dimension matrix CSV and return a dict mapping variable names to dimension vectors."""
    df = pd.read_csv(matrix_path)
    if "Dimension" not in df.columns and "Variable" not in df.columns:
        raise ValueError(f"Dimension matrix must have 'Dimension' or 'Variable' column: {matrix_path}")
    if "Variable" in df.columns and "Dimension" not in df.columns:
        df = df.rename(columns={"Variable": "Dimension"})

    matrix: Dict[str, List[int]] = {}
    for var in variables:
        if var in df.columns:
            var_dims = []
            for dim_name in DIMENSION_NAMES:
                row = df[df["Dimension"] == dim_name]
                if not row.empty:
                    var_dims.append(int(float(row[var].values[0])))
                else:
                    var_dims.append(0)
            matrix[var] = var_dims
        else:
            matrix[var] = [0] * 7
            print(f"   {var}: not in matrix, assuming dimensionless")
    print("✅ Loaded dimension matrix from file")
    return matrix


def generate_matrix_from_units(
    variables: List[str],
    variable_units: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
    """Generate a dimension matrix from unit strings (inferred if not provided)."""
    units = dict(variable_units) if variable_units else infer_units(variables)
    matrix: Dict[str, List[int]] = {}
    for var in variables:
        unit = units.get(var, "dimensionless")
        matrix[var] = parse_dimensions(unit)
    print("✅ Generated dimension matrix from units")
    return matrix, units


def find_dimension_matrix(
    all_variables: List[str],
    *,
    dimension_matrix_file: Optional[str] = None,
    input_file: Optional[str] = None,
    output_dir: str = "output",
    data_dir: str = "data",
    variable_units: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
    """Locate and load (or generate) the dimension matrix.

    Tries in order: explicit path, default locations relative to input_file and
    output_dir, then falls back to unit-based generation.
    """
    if dimension_matrix_file:
        p = Path(dimension_matrix_file)
        if p.exists():
            return load_matrix_from_file(p, all_variables), {}

    default_filenames = ["dimension_matrix_synthetic.csv", "dimension_matrix.csv"]
    search_roots: List[Optional[Path]] = []
    if input_file:
        search_roots.append(Path(input_file).parent)
    search_roots.append(Path(output_dir) / data_dir)
    search_roots.extend([Path("output") / data_dir, Path(".")])

    for root in search_roots:
        if root is None:
            continue
        for name in default_filenames:
            candidate = root / name
            if candidate.exists():
                print(f"✅ Found dimension matrix at: {candidate}")
                return load_matrix_from_file(candidate, all_variables), {}

    print("⚠️ No dimension matrix file found. Generating from units...")
    return generate_matrix_from_units(all_variables, variable_units)
