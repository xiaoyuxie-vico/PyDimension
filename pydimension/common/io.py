"""
I/O helpers for reading and writing common file formats (CSV, JSON).
"""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file, raising a clear error if the path does not exist."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")
    return pd.read_csv(p, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, **kwargs)
    return p


def read_json(path: str | Path) -> Dict[str, Any]:
    """Read a JSON file and return its contents as a dictionary."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with open(p, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Dict[str, Any], path: str | Path, indent: int = 2) -> Path:
    """Write a dictionary to a JSON file, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)
    return p
