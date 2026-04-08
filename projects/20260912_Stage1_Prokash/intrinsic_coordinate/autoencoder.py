"""
Intrinsic Coordinate Autoencoder.

Architecture:
    Encoder (over augmented features):
        augment(X) = [X, X², log(|X|+ε), π₁...πₘ]  shape: (batch, 3*n_inputs + n_pi)
        z = Encoder(augment(X))                       shape: (batch, n_latent)

    The encoder can be either:
      - Linear (default): z = W · augment(X)
        Cannot "cheat" → k=1 genuinely underfits when true dim > 1.
      - Multi-layer MLP: augment → h₁ → Tanh → ... → n_latent
        More expressive for latent dimension discovery (Step 1).

    Decoder (nonlinear):
        z → Linear → Tanh → Linear → Tanh → Linear → Output(1)

The augmented feature space lets the encoder select the right feature
type for each symmetry without needing to know the type in advance:
  - Translational: uses X features        (linear in x)
  - Rotational:    uses X² features       (quadratic in x)
  - Scaling:       uses log(|X|+ε) feats  (≈ log x for positive x)
  - Pi groups:     uses π features         (dimensionless combinations)
"""

from typing import List, Optional

import torch
import torch.nn as nn


class IntrinsicCoordinateAutoencoder(nn.Module):
    """
    Encoder-decoder network for discovering intrinsic coordinates.

    Parameters
    ----------
    n_inputs : int
        Dimensionality of input X (raw features, excluding Pi groups).
    n_latent : int
        Bottleneck dimension to test.
    hidden_dim : int
        Width of hidden layers in the decoder.
    encoder_hidden_dims : list of int, optional
        If provided, builds a multi-layer encoder MLP with these hidden
        widths and Tanh activations.  If None (default), uses a single
        linear layer (original behaviour).
    n_pi_groups : int
        Number of dimensionless Pi group features appended after the
        standard augmentation.  Default 0 (no Pi groups).
    """

    def __init__(
        self,
        n_inputs: int,
        n_latent: int,
        hidden_dim: int = 64,
        encoder_hidden_dims: Optional[List[int]] = None,
        n_pi_groups: int = 0,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_latent = n_latent
        self.n_pi_groups = n_pi_groups
        self._encoder_hidden_dims = encoder_hidden_dims

        n_aug = 3 * n_inputs + n_pi_groups  # [X, X², log(|X|+ε), π₁...πₘ]

        # Encoder
        if encoder_hidden_dims is None:
            # Linear encoder: no hidden layers, no nonlinearity
            self._enc = nn.Linear(n_aug, n_latent, bias=True)
        else:
            # Multi-layer MLP encoder with Tanh activations
            layers: List[nn.Module] = []
            in_dim = n_aug
            for h_dim in encoder_hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.Tanh())
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, n_latent))
            self._enc = nn.Sequential(*layers)

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
    def _augment_raw(x: torch.Tensor) -> torch.Tensor:
        """[X, X², log(max(|X|, 0.1))]  — augment raw features only.

        Clamp to 0.1 before taking log so:
          • scaling data (min |X| ≈ 0.135) gets exact log values
          • translational/rotational data (X near 0) is bounded at log(0.1)=-2.3
        """
        return torch.cat([x, x ** 2, torch.log(x.abs().clamp(min=0.1))], dim=1)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Build full augmented feature vector.

        If n_pi_groups > 0, x is expected to have shape
        (batch, n_inputs + n_pi_groups).  The first n_inputs columns get
        the standard [X, X², log|X|] augmentation; the remaining Pi group
        columns are appended as-is (they are already nonlinear combinations).
        """
        if self.n_pi_groups > 0:
            x_raw = x[:, :self.n_inputs]
            pi = x[:, self.n_inputs:]
            return torch.cat([self._augment_raw(x_raw), pi], dim=1)
        return self._augment_raw(x)

    # ------------------------------------------------------------------
    # Public interface (encoder / decoder as callable modules)
    # ------------------------------------------------------------------

    @property
    def encoder(self) -> "_EncoderWrapper":
        """Returns an encoder that accepts (batch, n_inputs[+n_pi]) → (batch, n_latent)."""
        return _EncoderWrapper(self._enc, self.n_inputs, self.n_latent,
                               self.n_pi_groups)

    @property
    def decoder(self) -> nn.Module:
        """Returns decoder: (batch, n_latent) → (batch, 1)."""
        return self._dec

    # ------------------------------------------------------------------
    # nn.Module forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_inputs[+n_pi]) → y_hat: (batch, 1)"""
        return self._dec(self._enc(self._augment(x)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_inputs[+n_pi]) → z: (batch, n_latent)"""
        return self._enc(self._augment(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, n_latent) → y_hat: (batch, 1)"""
        return self._dec(z)


class _EncoderWrapper(nn.Module):
    """Thin wrapper so encoder.parameters() and encoder(x) work correctly."""

    def __init__(self, enc: nn.Module, n_inputs: int, n_latent: int,
                 n_pi_groups: int = 0):
        super().__init__()
        self._enc = enc
        self.n_inputs = n_inputs
        self.n_latent = n_latent
        self.n_pi_groups = n_pi_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_pi_groups > 0:
            x_raw = x[:, :self.n_inputs]
            pi = x[:, self.n_inputs:]
            aug = torch.cat([IntrinsicCoordinateAutoencoder._augment_raw(x_raw), pi], dim=1)
        else:
            aug = IntrinsicCoordinateAutoencoder._augment_raw(x)
        return self._enc(aug)
