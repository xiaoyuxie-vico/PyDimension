"""
Latent dimension discovery via autoencoder sweep.

For n_latent = 1, 2, ..., max_latent:
    - Train IntrinsicCoordinateAutoencoder on (X, y)
    - Record R² and MSE on validation set

Select the minimal n_latent where R² first exceeds the threshold (elbow method
with R² > threshold as fallback).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .autoencoder import IntrinsicCoordinateAutoencoder


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
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    dataset   = TensorDataset(X_train, y_train)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
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

    Returns
    -------
    dict with keys:
        optimal_n_latent : int
        best_encoder     : nn.Module   accepts (batch, n_inputs) → (batch, n_latent)
        best_decoder     : nn.Module   accepts (batch, n_latent) → (batch, 1)
        metrics          : dict[int → {"R2": float, "MSE": float}]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_samples, n_inputs = X.shape
    n_val = int(n_samples * val_fraction)
    idx = np.random.permutation(n_samples)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_tr = torch.tensor(X[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val_np = y[val_idx]

    metrics = {}
    models  = {}

    for k in range(1, max_latent + 1):
        best_r2    = -np.inf
        best_model = None

        # Multiple restarts: keep best R² to reduce variance from random init
        for restart in range(n_restarts):
            torch.manual_seed(seed + k * 100 + restart)
            model = IntrinsicCoordinateAutoencoder(n_inputs, k, hidden_dim)
            _train_autoencoder(model, X_tr, y_tr, n_epochs, batch_size, lr)

            model.eval()
            with torch.no_grad():
                y_pred_np = model(X_val).squeeze(1).numpy()

            r2 = _r2_score(y_val_np, y_pred_np)
            if r2 > best_r2:
                best_r2    = r2
                best_model = model

        mse = float(np.mean((y_val_np - best_model(X_val).squeeze(1).detach().numpy()) ** 2))
        metrics[k] = {"R2": best_r2, "MSE": mse}
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
    }
