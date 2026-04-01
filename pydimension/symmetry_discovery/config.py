"""
Configuration handling for the symmetry discovery module.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SymmetryDiscoveryConfig:
    """Configuration for the renamed 3.0 symmetry-discovery stage."""

    input_file: Optional[str] = None
    basis_vectors_file: Optional[str] = None
    symmetry_type: str = "translational"
    encoder_name: str = "translational"
    num_linear: int = 1
    num_hidden_layers: int = 4
    num_hidden_nodes: int = 10
    random_seed: int = 49
    epochs: int = 1000
    learning_rate: float = 0.001
    train_percent: float = 0.8
    num_ensembles: int = 5
    use_fixed_gamma: bool = False
    fixed_gamma_values: Optional[np.ndarray] = None
    use_gamma_regularization: bool = True
    gamma_reg_strength: float = 0.01
    gamma_reg_resolution: str = "half-integers"
    output_dir: str = "output"
    data_dir: str = "data"
    figures_dir: str = "figures"
    results_dir: str = "results"
    model_results_filename: str = "symmetry_discovery_results.json"
    plot_filename: str = "symmetry_discovery_plots.png"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SymmetryDiscoveryConfig":
        section = config_dict.get("SYMMETRY_DISCOVERY", config_dict.get("OPTIMIZATION_DISCOVERY"))
        if section is None:
            raise ValueError(
                "Config must contain 'SYMMETRY_DISCOVERY' section. "
                "Legacy fallback to 'OPTIMIZATION_DISCOVERY' is still supported."
            )

        section_output = config_dict.get(
            "SYMMETRY_DISCOVERY_OUTPUT",
            config_dict.get("OPTIMIZATION_DISCOVERY_OUTPUT", {}),
        )
        output = config_dict.get("OUTPUT", {})

        output_dir = output.get("output_dir", "output")
        data_dir = output.get("data_dir", "data")
        figures_dir = output.get("figures_dir", "figures")
        results_dir = output.get("results_dir", "results")

        input_file = section.get("input_file")
        if input_file is None:
            base_path = Path(output_dir).resolve()
            cwd = Path.cwd()
            default_paths = [
                base_path / data_dir / "normalized_lg_afterDA_data.csv",
                Path("output") / data_dir / "normalized_lg_afterDA_data.csv",
                Path("normalized_lg_afterDA_data.csv"),
                cwd / output_dir / data_dir / "normalized_lg_afterDA_data.csv",
                cwd / "output" / data_dir / "normalized_lg_afterDA_data.csv",
                cwd / "normalized_lg_afterDA_data.csv",
            ]
            for path in default_paths:
                if path.exists():
                    input_file = str(path)
                    break

        basis_vectors_file = section.get("basis_vectors_file")
        if basis_vectors_file is None:
            base_path = Path(output_dir).resolve()
            cwd = Path.cwd()
            default_paths = [
                base_path / data_dir / "basis_vectors.csv",
                Path("output") / data_dir / "basis_vectors.csv",
                Path("basis_vectors.csv"),
                cwd / output_dir / data_dir / "basis_vectors.csv",
                cwd / "output" / data_dir / "basis_vectors.csv",
                cwd / "basis_vectors.csv",
            ]
            for path in default_paths:
                if path.exists():
                    basis_vectors_file = str(path)
                    break

        num_linear = section.get("num_linear")
        if num_linear is None:
            base_path = Path(output_dir).resolve()
            cwd = Path.cwd()
            suggested_paths = [
                base_path / results_dir / "suggested_dominant_count.json",
                Path("output") / results_dir / "suggested_dominant_count.json",
                cwd / output_dir / results_dir / "suggested_dominant_count.json",
                cwd / "output" / results_dir / "suggested_dominant_count.json",
            ]
            for path in suggested_paths:
                if path.exists():
                    try:
                        with open(path, "r") as handle:
                            payload = json.load(handle)
                        num_linear = payload.get("suggested_dominant_count")
                        if num_linear is not None:
                            break
                    except Exception:
                        pass
            if num_linear is None:
                num_linear = 1

        fixed_gamma_values = None
        if section.get("use_fixed_gamma", False) and basis_vectors_file and Path(basis_vectors_file).exists():
            try:
                import pandas as pd

                basis_frame = pd.read_csv(basis_vectors_file)
                weight_columns = [column for column in basis_frame.columns if column.startswith("w")]
                if weight_columns:
                    num_gamma = min(num_linear, len(weight_columns))
                    basis_matrix = basis_frame[weight_columns].values
                    fixed_gamma_values = basis_matrix[:, :num_gamma].T
            except Exception:
                pass

        return cls(
            input_file=input_file,
            basis_vectors_file=basis_vectors_file,
            symmetry_type=section.get("symmetry_type", "translational"),
            encoder_name=section.get("encoder_name", section.get("symmetry_type", "translational")),
            num_linear=num_linear,
            num_hidden_layers=section.get("num_hidden_layers", 4),
            num_hidden_nodes=section.get("num_hidden_nodes", 10),
            random_seed=section.get("random_seed", 49),
            epochs=section.get("epochs", 1000),
            learning_rate=section.get("learning_rate", 0.001),
            train_percent=section.get("train_percent", 0.8),
            num_ensembles=section.get("num_ensembles", 5),
            use_fixed_gamma=section.get("use_fixed_gamma", False),
            fixed_gamma_values=fixed_gamma_values,
            use_gamma_regularization=section.get("use_gamma_regularization", True),
            gamma_reg_strength=section.get("gamma_reg_strength", 0.01),
            gamma_reg_resolution=section.get("gamma_reg_resolution", "half-integers"),
            output_dir=output_dir,
            data_dir=data_dir,
            figures_dir=figures_dir,
            results_dir=results_dir,
            model_results_filename=section_output.get(
                "model_results_filename", "symmetry_discovery_results.json"
            ),
            plot_filename=section_output.get("plot_filename", "symmetry_discovery_plots.png"),
        )

    @classmethod
    def from_json(cls, json_path: str) -> "SymmetryDiscoveryConfig":
        with open(json_path, "r") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "SYMMETRY_DISCOVERY": {
                "enabled": True,
                "input_file": self.input_file,
                "basis_vectors_file": self.basis_vectors_file,
                "symmetry_type": self.symmetry_type,
                "encoder_name": self.encoder_name,
                "num_linear": self.num_linear,
                "num_hidden_layers": self.num_hidden_layers,
                "num_hidden_nodes": self.num_hidden_nodes,
                "random_seed": self.random_seed,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "train_percent": self.train_percent,
                "num_ensembles": self.num_ensembles,
                "use_fixed_gamma": self.use_fixed_gamma,
                "use_gamma_regularization": self.use_gamma_regularization,
                "gamma_reg_strength": self.gamma_reg_strength,
                "gamma_reg_resolution": self.gamma_reg_resolution,
            },
            "OUTPUT": {
                "output_dir": self.output_dir,
                "data_dir": self.data_dir,
                "figures_dir": self.figures_dir,
                "results_dir": self.results_dir,
                "logs_dir": "logs",
            },
            "SYMMETRY_DISCOVERY_OUTPUT": {
                "model_results_filename": self.model_results_filename,
                "plot_filename": self.plot_filename,
            },
        }

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.input_file is None:
            errors.append("input_file must be specified or found in default preprocessing outputs")
        elif not Path(self.input_file).exists():
            errors.append(f"Input file not found: {self.input_file}")
        if self.num_linear < 1:
            errors.append("num_linear must be at least 1")
        if self.num_hidden_layers < 0:
            errors.append("num_hidden_layers must be non-negative")
        if self.num_hidden_nodes < 1:
            errors.append("num_hidden_nodes must be at least 1")
        if self.epochs < 1:
            errors.append("epochs must be at least 1")
        if not (0 < self.train_percent < 1):
            errors.append("train_percent must be between 0 and 1")
        if self.num_ensembles < 1:
            errors.append("num_ensembles must be at least 1")
        if self.gamma_reg_resolution not in ["integers", "half-integers", "quarter-integers"]:
            errors.append(
                "gamma_reg_resolution must be one of: 'integers', 'half-integers', 'quarter-integers'"
            )
        if self.use_fixed_gamma and self.fixed_gamma_values is None:
            errors.append("use_fixed_gamma is True but fixed_gamma_values is None. Load basis vectors first.")
        if self.symmetry_type not in ["translational", "rotational", "scaling"]:
            errors.append("symmetry_type must be one of: 'translational', 'rotational', 'scaling'")
        return errors
