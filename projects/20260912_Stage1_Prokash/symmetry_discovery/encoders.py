"""
Symmetry-type-specific encoders for symmetry identification.

Each SymmetryEncoder is a SINGLE linear layer (no hidden layers, no bias)
applied to a fixed feature transform of X:

    translational : X          → W · X        (linear)
    rotational    : X²         → W · X²       (quadratic)
    scaling       : log(|X|+ε) → W · log|X|   (log)

The correct transform type naturally aligns with the true symmetry of the data,
so training with the frozen Task-3 decoder reveals the winning type via loss.
"""

import numpy as np
import torch
import torch.nn as nn

_SYMMETRY_TYPES = ("translational", "rotational", "scaling")


class SymmetryEncoder(nn.Module):
    """
    Single-layer linear encoder with a symmetry-specific feature transform.

    Parameters
    ----------
    symmetry_type : str
        One of "translational", "rotational", "scaling".
    n_inputs : int
        Dimensionality of raw input X.
    n_latent : int
        Latent (output) dimensionality.
    """

    def __init__(self, symmetry_type: str, n_inputs: int, n_latent: int):
        super().__init__()
        if symmetry_type not in _SYMMETRY_TYPES:
            raise ValueError(
                f"symmetry_type must be one of {_SYMMETRY_TYPES}, got '{symmetry_type}'"
            )
        self.symmetry_type = symmetry_type
        self.n_inputs      = n_inputs
        self.n_latent      = n_latent

        # Single linear layer, no bias: z = W · transform(X)
        self.linear = nn.Linear(n_inputs, n_latent, bias=False)

    # ------------------------------------------------------------------
    # Feature transform
    # ------------------------------------------------------------------

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the symmetry-specific feature transform."""
        if self.symmetry_type == "translational":
            return x
        elif self.symmetry_type == "rotational":
            return x ** 2
        else:  # scaling
            # Same clamp as Task-3 augmented encoder for consistency;
            # exact log(x) for scaling data where min|x| ≈ 0.135 > 0.1
            return torch.log(x.abs().clamp(min=0.1))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_inputs) → z: (batch, n_latent)"""
        return self.linear(self._transform(x))

    # ------------------------------------------------------------------
    # Coefficient extraction
    # ------------------------------------------------------------------

    @property
    def weight_matrix(self) -> np.ndarray:
        """Weight matrix W, shape (n_latent, n_inputs)."""
        return self.linear.weight.detach().cpu().numpy()

    @property
    def coefficients(self) -> np.ndarray:
        """
        Coefficient vector / matrix for the winning encoder.

        For n_latent == 1 returns a 1-D array of length n_inputs;
        otherwise returns the full (n_latent, n_inputs) weight matrix.
        """
        W = self.weight_matrix
        return W[0] if self.n_latent == 1 else W
