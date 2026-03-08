"""
Shared validation helpers used across pipeline modules.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


def require_file(path: Optional[str], label: str = "file") -> Path:
    """Return a resolved Path or raise ValueError / FileNotFoundError."""
    if path is None:
        raise ValueError(f"{label} path must not be None")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def check_config_errors(errors: List[str], context: str = "Config") -> None:
    """Raise ValueError if the error list is non-empty."""
    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"{context} validation failed:\n  - {joined}")


def require_columns(columns: List[str], available: List[str], label: str = "data") -> None:
    """Raise ValueError if any required column is missing from the available list."""
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(f"Missing columns in {label}: {missing}")
