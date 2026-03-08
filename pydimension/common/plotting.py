"""
Shared plotting utilities used by multiple pipeline modules.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    """Apply the default PyDimension plot style."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = 150,
    close: bool = True,
) -> Path:
    """Save a matplotlib figure, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return p


def close_all():
    """Close all open matplotlib figures."""
    plt.close("all")
