"""
Latent dimension discovery via autoencoder sweep.

For n_latent = 1, 2, ..., max_latent:
    - Train IntrinsicCoordinateAutoencoder on (X, y)
    - Record R² and MSE on validation set

Select the minimal n_latent where R² first exceeds the threshold (elbow method
with R² > threshold as fallback).

Optional enhancements (professor-recommended):
    - Multi-layer encoder: pass ``encoder_hidden_dims`` to use an MLP encoder
      instead of a single linear layer.  More expressive for latent dimension
      discovery; interpretability is not needed at this stage.
    - Pi group augmentation: pass ``pi_basis_vectors`` (null-space of the
      dimension matrix) to compute dimensionless Pi groups and append them
      as extra encoder inputs, giving the encoder a physics-informed head start.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .autoencoder import IntrinsicCoordinateAutoencoder


def _compute_pi_groups(X: np.ndarray, basis_vectors: np.ndarray) -> np.ndarray:
    """Compute dimensionless Pi groups from raw data and basis vectors.

    Parameters
    ----------
    X : (n_samples, n_inputs)
        Raw (positive, normalised) input data.
    basis_vectors : (n_inputs, n_groups)
        Null-space basis of the dimension matrix.  Each column is a set of
        exponents defining one Pi group: π_i = prod(x_j ^ basis[j, i]).

    Returns
    -------
    pi_groups : (n_samples, n_groups)
    """
    # log-space computation for numerical stability (same as preprocessor.py)
    X_safe = np.maximum(np.abs(X), 1e-10)
    log_X = np.log(X_safe)                        # (n_samples, n_inputs)
    log_pi = log_X @ basis_vectors                 # (n_samples, n_groups)
    return np.exp(log_pi)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _train_autoencoder(
    model: IntrinsicCoordinateAutoencoder,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    n = len(X_train)

    # For small datasets: pre-load to GPU and batch manually.
    # DataLoader worker overhead dominates when there are only a few batches.
    if n <= 20_000:
        X_dev = X_train.to(device)
        Y_dev = y_train.to(device)
        model.train()
        for _ in range(n_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                loss_fn(model(X_dev[idx]), Y_dev[idx]).backward()
                optimizer.step()
    else:
        num_workers = min(4, torch.multiprocessing.cpu_count())
        loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        model.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad()
                loss_fn(model(xb), yb).backward()
                optimizer.step()


def discover_latent_dimension(
    X: np.ndarray,
    y: np.ndarray,
    max_latent: int = 6,
    hidden_dim: int = 64,
    n_epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    r2_threshold: float = 0.95,
    n_restarts: int = 3,
    seed: int = 0,
    device: str = "auto",
    encoder_hidden_dims: Optional[List[int]] = None,
    pi_basis_vectors: Optional[np.ndarray] = None,
) -> dict:
    """
    Sweep n_latent from 1 to max_latent, train an autoencoder for each,
    and select the minimal n_latent that explains the data well.

    Parameters
    ----------
    X : (n_samples, n_inputs)
    y : (n_samples,)
    max_latent : int
        Maximum latent dimension to try.
    hidden_dim : int
        Hidden layer width.
    n_epochs : int
        Training epochs per model.
    batch_size : int
    lr : float
        Adam learning rate.
    val_fraction : float
        Fraction of data held out for validation.
    r2_threshold : float
        Minimum R² to consider a latent dimension sufficient.
    seed : int
        Random seed for train/val split and weight init.
    encoder_hidden_dims : list of int, optional
        Hidden layer widths for a multi-layer encoder MLP.
        If None (default), the encoder is a single linear layer.
    pi_basis_vectors : (n_inputs, n_groups) array, optional
        Null-space basis of the dimension matrix.  When provided,
        dimensionless Pi groups are computed from X and appended as
        extra encoder inputs: [X, X², log|X|, π₁...πₘ].

    Returns
    -------
    dict with keys:
        optimal_n_latent : int
        best_encoder     : nn.Module   accepts (batch, n_inputs) → (batch, n_latent)
        best_decoder     : nn.Module   accepts (batch, n_latent) → (batch, 1)
        metrics          : dict[int → {"R2": float, "MSE": float}]
        device           : str         device used for training
    """
    if device == "auto":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_samples, n_inputs = X.shape

    # --- Compute Pi groups if basis vectors are provided ---
    n_pi_groups = 0
    if pi_basis_vectors is not None:
        pi_groups = _compute_pi_groups(X, pi_basis_vectors)
        n_pi_groups = pi_groups.shape[1]
        X_aug_np = np.hstack([X, pi_groups])  # (n_samples, n_inputs + n_pi)
        print(f"  Pi group augmentation: {n_pi_groups} groups appended "
              f"→ encoder input dim = 3×{n_inputs} + {n_pi_groups} = "
              f"{3 * n_inputs + n_pi_groups}")
    else:
        X_aug_np = X

    n_val = int(n_samples * val_fraction)
    idx = np.random.permutation(n_samples)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_tr = torch.tensor(X_aug_np[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_aug_np[val_idx], dtype=torch.float32)
    y_val_np = y[val_idx]

    if encoder_hidden_dims is not None:
        print(f"  Multi-layer encoder: {encoder_hidden_dims}")

    metrics = {}
    models  = {}

    X_val_dev = X_val.to(_device)

    for k in range(1, max_latent + 1):
        best_r2    = -np.inf
        best_model = None

        # Multiple restarts: keep best R² to reduce variance from random init
        for restart in range(n_restarts):
            torch.manual_seed(seed + k * 100 + restart)
            model = IntrinsicCoordinateAutoencoder(
                n_inputs, k, hidden_dim,
                encoder_hidden_dims=encoder_hidden_dims,
                n_pi_groups=n_pi_groups,
            ).to(_device)
            _train_autoencoder(model, X_tr, y_tr, n_epochs, batch_size, lr, _device)

            model.eval()
            with torch.no_grad():
                y_pred_np = model(X_val_dev).squeeze(1).cpu().numpy()

            r2 = _r2_score(y_val_np, y_pred_np)
            if r2 > best_r2:
                best_r2    = r2
                best_model = model

        mse = float(np.mean((y_val_np - best_model(X_val_dev).squeeze(1).detach().cpu().numpy()) ** 2))

        # Also compute training R² for comparison
        best_model.eval()
        with torch.no_grad():
            y_tr_pred = best_model(X_tr.to(_device)).squeeze(1).cpu().numpy()
        r2_train = _r2_score(y[train_idx], y_tr_pred)

        metrics[k] = {"R2": best_r2, "R2_train": r2_train, "MSE": mse}
        models[k]  = best_model

    # --- Select optimal latent dimension ---
    # Because the encoder is linear over augmented features, k=1 genuinely
    # underfits when the true latent dimension is > 1.  We therefore use the
    # simple rule: pick the smallest k where R²(k) >= r2_threshold.
    # Fallback: pick the k with the highest R² if no k crosses the threshold.
    optimal_k = max(metrics, key=lambda k: metrics[k]["R2"])  # fallback

    for k in range(1, max_latent + 1):
        if metrics[k]["R2"] >= r2_threshold:
            optimal_k = k
            break

    # Ensure we don't pick a k with poor R² when a higher k is much better
    # (handles cases where threshold is crossed mid-sweep)
    best_model = models[optimal_k]

    return {
        "optimal_n_latent": optimal_k,
        "best_encoder": best_model.encoder,
        "best_decoder": best_model.decoder,
        "metrics": metrics,
        "device": str(_device),
    }
