"""
Centralized path resolution for output directories used by all pipeline stages.
"""

from pathlib import Path
from typing import Optional


class OutputLayout:
    """Convention for subdirectory layout under a single output root."""

    def __init__(
        self,
        output_dir: str = "output",
        data_dir: str = "data",
        figures_dir: str = "figures",
        results_dir: str = "results",
        logs_dir: str = "logs",
    ):
        self.root = Path(output_dir)
        self._data = data_dir
        self._figures = figures_dir
        self._results = results_dir
        self._logs = logs_dir

    @property
    def data(self) -> Path:
        return self.root / self._data

    @property
    def figures(self) -> Path:
        return self.root / self._figures

    @property
    def results(self) -> Path:
        return self.root / self._results

    @property
    def logs(self) -> Path:
        return self.root / self._logs

    def ensure_all(self) -> "OutputLayout":
        """Create every subdirectory if it does not exist."""
        for d in (self.data, self.figures, self.results, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def from_config_dict(cls, config: dict) -> "OutputLayout":
        """Build from an OUTPUT section of a JSON config."""
        out = config.get("OUTPUT", {})
        return cls(
            output_dir=out.get("output_dir", "output"),
            data_dir=out.get("data_dir", "data"),
            figures_dir=out.get("figures_dir", "figures"),
            results_dir=out.get("results_dir", "results"),
            logs_dir=out.get("logs_dir", "logs"),
        )
