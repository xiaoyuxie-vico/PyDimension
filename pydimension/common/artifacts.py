"""
Filesystem helpers shared by the v3 pipeline modules.
"""

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return the path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def output_path(output_dir: str | Path, *parts: str) -> Path:
    """Build a path inside an output directory and ensure its parent exists."""
    target = Path(output_dir).joinpath(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
