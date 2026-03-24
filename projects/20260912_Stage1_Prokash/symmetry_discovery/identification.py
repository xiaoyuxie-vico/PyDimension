"""
Symmetry identification via competitive encoder-decoder training.

For each candidate symmetry type ("translational", "rotational", "scaling"):
  1. Build a SymmetryEncoder that transforms X with the type-specific feature map.
  2. Train encoder + decoder jointly (decoder always fresh) with cosine LR decay.
  3. Record the validation MSE.

Training with a fresh decoder (rather than a Task-3 warm-start) ensures an
unbiased starting point for every type:
  - The correct symmetry type reaches a very low loss (the feature transform
    perfectly captures the signal).
  - Wrong symmetry types plateau at a higher loss (wrong feature space).
  - Encoder weights converge to the true physical coefficients.
"""

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .encoders import SymmetryEncoder

# ------------------------------------------------------------------
# Fresh decoder factory (same architecture as Task-3 decoder)
# ------------------------------------------------------------------

def _make_decoder(n_latent: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_latent, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )


# ------------------------------------------------------------------
# Joint encoder + decoder training
# ------------------------------------------------------------------

def _train_joint(
    encoder: SymmetryEncoder,
    decoder: nn.Module,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> None:
    """Train encoder and decoder jointly to minimise MSE(decoder(encoder(X)), y)."""
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )
    loss_fn   = nn.MSELoss()

    n = X_tr.shape[0]
    if n <= 20_000:
        # Small dataset: pre-load to GPU and batch manually — no worker overhead.
        X_gpu = X_tr.to(device)
        y_gpu = y_tr.to(device)
        encoder.train()
        decoder.train()
        for _ in range(n_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                loss_fn(decoder(encoder(X_gpu[idx])), y_gpu[idx]).backward()
                optimizer.step()
            scheduler.step()
    else:
        num_workers = min(4, torch.multiprocessing.cpu_count())
        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        encoder.train()
        decoder.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad()
                loss_fn(decoder(encoder(xb)), yb).backward()
                optimizer.step()
            scheduler.step()


def _val_mse(
    encoder: SymmetryEncoder,
    decoder: nn.Module,
    X_val: torch.Tensor,
    y_val_np: np.ndarray,
) -> float:
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        pred = decoder(encoder(X_val)).squeeze(1).cpu().numpy()
    return float(np.mean((y_val_np - pred) ** 2))


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def identify_symmetry(
    X: np.ndarray,
    y: np.ndarray,
    n_latent: int,
    decoder: nn.Module,
    n_epochs: int = 1500,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    val_fraction: float = 0.2,
    n_restarts: int = 3,
    seed: int = 0,
    device: str = "auto",
) -> dict:
    """
    Identify the symmetry type of the data by competitive encoder-decoder training.

    For each symmetry type a SymmetryEncoder is trained jointly with a fresh
    randomly-initialised decoder.  The type whose (encoder, decoder) pair
    achieves the lowest validation MSE wins.

    Parameters
    ----------
    X : (n_samples, n_inputs)
    y : (n_samples,)
    n_latent : int
        Latent dimension from Task 3.
    decoder : nn.Module
        Unused (kept for API compatibility). Each restart uses a fresh decoder.
    n_epochs : int
        Training epochs per encoder / restart.
    batch_size : int
    lr : float
        Adam learning rate.
    weight_decay : float
        L2 regularisation on all parameters.
    hidden_dim : int
        Hidden dimension for fresh decoder instances.
    val_fraction : float
        Fraction of data held out for validation.
    n_restarts : int
        Random restarts per symmetry type; keeps the best run.
    seed : int
        Random seed for train/val split and weight init.

    Returns
    -------
    dict with keys:
        symmetry_type : str   — "translational", "rotational", or "scaling"
        coefficients  : np.ndarray — winning encoder's weight_matrix
                        shape (n_inputs,) for n_latent==1, else (n_latent, n_inputs)
        losses        : dict[str → float] — val MSE for each symmetry type
        encoders      : dict[str → SymmetryEncoder] — best encoder per type
    """
    if device == "auto":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_samples, n_inputs = X.shape
    n_val   = int(n_samples * val_fraction)
    idx     = np.random.permutation(n_samples)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_tr     = torch.tensor(X[train_idx], dtype=torch.float32)
    y_tr     = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    X_val    = torch.tensor(X[val_idx],   dtype=torch.float32).to(_device)
    y_val_np = y[val_idx]

    sym_types = ("translational", "rotational", "scaling")
    best_losses   = {}
    best_encoders = {}

    for sym_type in sym_types:
        best_loss    = np.inf
        best_encoder = None

        for restart in range(n_restarts):
            torch.manual_seed(seed + hash(sym_type) % 1000 + restart * 37)

            enc = SymmetryEncoder(sym_type, n_inputs, n_latent).to(_device)
            # Fresh decoder for each restart: avoids bias from Task-3 warm-start,
            # which could favour whichever feature type the Task-3 autoencoder
            # happened to converge to (often log-like for rotational data).
            dec = _make_decoder(n_latent, hidden_dim).to(_device)

            _train_joint(enc, dec, X_tr, y_tr, n_epochs, batch_size, lr, weight_decay, _device)

            val_loss = _val_mse(enc, dec, X_val, y_val_np)
            if val_loss < best_loss:
                best_loss    = val_loss
                best_encoder = enc

        best_losses[sym_type]   = best_loss
        best_encoders[sym_type] = best_encoder

    # Winner = lowest validation MSE
    winner = min(best_losses, key=lambda t: best_losses[t])

    return {
        "symmetry_type": winner,
        "coefficients":  best_encoders[winner].coefficients,
        "losses":        best_losses,
        "encoders":      best_encoders,
    }
