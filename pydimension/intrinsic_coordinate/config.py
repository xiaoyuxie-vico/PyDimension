"""
Configuration handling for the intrinsic coordinate module.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class IntrinsicCoordinateConfig:
    """Configuration for the renamed intrinsic-coordinate stage."""

    input_file: Optional[str] = None
    method: str = "pca_sir"
    run_pca: bool = True
    run_sir: bool = True
    pca_threshold: float = 0.75
    sir_threshold: float = 0.75
    n_sir_slices: int = 10
    n_sir_directions: int = 3
    latent_dim: int = 1
    output_dir: str = "output"
    data_dir: str = "data"
    figures_dir: str = "figures"
    results_dir: str = "results"
    pca_results_filename: str = "pca_results.json"
    sir_results_filename: str = "sir_results.json"
    plot_filename: str = "intrinsic_coordinate_plots.png"
    suggested_count_filename: str = "suggested_dominant_count.json"
    autoencoder_results_filename: str = "autoencoder_results.json"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "IntrinsicCoordinateConfig":
        section = config_dict.get("INTRINSIC_COORDINATE", config_dict.get("CONSTRAINT_FILTERING", {}))
        output = config_dict.get("OUTPUT", {})
        section_output = config_dict.get(
            "INTRINSIC_COORDINATE_OUTPUT",
            config_dict.get("CONSTRAINT_FILTERING_OUTPUT", {}),
        )

        output_dir = output.get("output_dir", "output")
        data_dir = output.get("data_dir", "data")
        figures_dir = output.get("figures_dir", "figures")
        results_dir = output.get("results_dir", "results")

        input_file = section.get("input_file")
        if input_file is None:
            candidate = Path(output_dir) / data_dir / "normalized_lg_afterDA_data.csv"
            if candidate.exists():
                input_file = str(candidate)

        return cls(
            input_file=input_file,
            method=section.get("method", "pca_sir"),
            run_pca=section.get("run_pca", True),
            run_sir=section.get("run_sir", True),
            pca_threshold=section.get("pca_threshold", 0.75),
            sir_threshold=section.get("sir_threshold", 0.75),
            n_sir_slices=section.get("n_sir_slices", 10),
            n_sir_directions=section.get("n_sir_directions", 3),
            latent_dim=section.get("latent_dim", 1),
            output_dir=output_dir,
            data_dir=data_dir,
            figures_dir=figures_dir,
            results_dir=results_dir,
            pca_results_filename=section_output.get("pca_results_filename", "pca_results.json"),
            sir_results_filename=section_output.get("sir_results_filename", "sir_results.json"),
            plot_filename=section_output.get("plot_filename", "intrinsic_coordinate_plots.png"),
            suggested_count_filename=section_output.get(
                "suggested_count_filename", "suggested_dominant_count.json"
            ),
            autoencoder_results_filename=section_output.get(
                "autoencoder_results_filename", "autoencoder_results.json"
            ),
        )

    @classmethod
    def from_json(cls, json_path: str) -> "IntrinsicCoordinateConfig":
        with open(json_path, "r") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "INTRINSIC_COORDINATE": {
                "enabled": True,
                "input_file": self.input_file,
                "method": self.method,
                "run_pca": self.run_pca,
                "run_sir": self.run_sir,
                "pca_threshold": self.pca_threshold,
                "sir_threshold": self.sir_threshold,
                "n_sir_slices": self.n_sir_slices,
                "n_sir_directions": self.n_sir_directions,
                "latent_dim": self.latent_dim,
            },
            "OUTPUT": {
                "output_dir": self.output_dir,
                "data_dir": self.data_dir,
                "figures_dir": self.figures_dir,
                "results_dir": self.results_dir,
                "logs_dir": "logs",
            },
            "INTRINSIC_COORDINATE_OUTPUT": {
                "pca_results_filename": self.pca_results_filename,
                "sir_results_filename": self.sir_results_filename,
                "plot_filename": self.plot_filename,
                "suggested_count_filename": self.suggested_count_filename,
                "autoencoder_results_filename": self.autoencoder_results_filename,
            },
        }

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.input_file is None:
            errors.append("input_file must be specified or discoverable from preprocessing outputs")
        elif not Path(self.input_file).exists():
            errors.append(f"Input file not found: {self.input_file}")
        if self.method not in {"pca_sir", "sir", "autoencoder"}:
            errors.append("method must be 'pca_sir', 'sir', or 'autoencoder'")
        return errors
