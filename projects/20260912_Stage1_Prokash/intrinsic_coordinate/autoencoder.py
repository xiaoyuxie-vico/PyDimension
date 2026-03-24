"""
Intrinsic Coordinate Autoencoder.

Architecture:
    Encoder (linear, over augmented features):
        augment(X) = [X, X², log(|X|+1)]     shape: (batch, 3*n_inputs)
        z = W_enc · augment(X)                shape: (batch, n_latent)

    Decoder (nonlinear):
        z → Linear → Tanh → Linear → Tanh → Linear → Output(1)

The augmented feature space lets the linear encoder select the right feature
type for each symmetry without needing to know the type in advance:
  - Translational: uses X features        (linear in x)
  - Rotational:    uses X² features       (quadratic in x)
  - Scaling:       uses log(|X|+1) feats  (≈ log x for positive x)

The LINEAR encoder is key: it cannot "cheat" by computing y directly from
nonlinear combinations of inputs, so k=1 will genuinely underfit when the
true latent dimension is > 1.
"""

import torch
import torch.nn as nn


class IntrinsicCoordinateAutoencoder(nn.Module):
    """
    Encoder-decoder network for discovering intrinsic coordinates.

    Parameters
    ----------
    n_inputs : int
        Dimensionality of input X.
    n_latent : int
        Bottleneck dimension to test.
    hidden_dim : int
        Width of hidden layers in the decoder.
    """

    def __init__(self, n_inputs: int, n_latent: int, hidden_dim: int = 64):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_latent = n_latent

        n_aug = 3 * n_inputs  # [X, X², log(|X|+1)]

        # Linear encoder: no hidden layers, no nonlinearity
        self._enc = nn.Linear(n_aug, n_latent, bias=True)

        # Nonlinear decoder: two hidden layers with Tanh
        self._dec = nn.Sequential(
            nn.Linear(n_latent, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        """[X, X², log(max(|X|, 0.1))]

        Clamp to 0.1 before taking log so:
          • scaling data (min |X| ≈ 0.135) gets exact log values
          • translational/rotational data (X near 0) is bounded at log(0.1)=-2.3
        """
        return torch.cat([x, x ** 2, torch.log(x.abs().clamp(min=0.1))], dim=1)

    # ------------------------------------------------------------------
    # Public interface (encoder / decoder as callable modules)
    # ------------------------------------------------------------------

    @property
    def encoder(self) -> "_EncoderWrapper":
        """Returns an encoder that accepts (batch, n_inputs) → (batch, n_latent)."""
        return _EncoderWrapper(self._enc, self.n_inputs, self.n_latent)

    @property
    def decoder(self) -> nn.Module:
        """Returns decoder: (batch, n_latent) → (batch, 1)."""
        return self._dec

    # ------------------------------------------------------------------
    # nn.Module forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_inputs) → y_hat: (batch, 1)"""
        return self._dec(self._enc(self._augment(x)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_inputs) → z: (batch, n_latent)"""
        return self._enc(self._augment(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, n_latent) → y_hat: (batch, 1)"""
        return self._dec(z)


class _EncoderWrapper(nn.Module):
    """Thin wrapper so encoder.parameters() and encoder(x) work correctly."""

    def __init__(self, linear: nn.Linear, n_inputs: int, n_latent: int):
        super().__init__()
        self._linear = linear
        self.n_inputs = n_inputs
        self.n_latent = n_latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(IntrinsicCoordinateAutoencoder._augment(x))
