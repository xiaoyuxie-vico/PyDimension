"""
Translational symmetry data generator.

y = f(W·x) where rows of W are the measurement directions.
Invariant under x → x + ε·d for any d in the null space of W.

m_orbits sets how many latent (measurement) directions there are.
The true latent dimension equals m_orbits.
"""

import numpy as np
from scipy.linalg import null_space as _null_space


def generate_translational_data(
    n_inputs: int,
    m_orbits: int = 1,
    orbit_directions: np.ndarray = None,
    n_samples: int = 1000,
    noise_level: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic data with translational symmetry.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    m_orbits : int
        Number of latent (measurement) directions. True latent dim = m_orbits.
    orbit_directions : np.ndarray, optional
        Shape (m_orbits, n_inputs). Rows are the measurement directions W.
        If None, random orthonormal directions are generated.
    n_samples : int
        Number of data points.
    noise_level : float
        Additive Gaussian noise on y: noise ~ N(0, noise_level * std(y_clean)).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        X                   : (n_samples, n_inputs)   input data
        y                   : (n_samples,)             output (possibly noisy)
        y_clean             : (n_samples,)             noise-free output
        orbit_directions    : (m_orbits, n_inputs)     measurement matrix W
        orthogonal_directions : (n_inputs, n_inputs-m_orbits)  null space of W
                                (true translational orbit directions)
        n_latent            : m_orbits
        symmetry_type       : "translational"
    """
    rng = np.random.default_rng(seed)

    if orbit_directions is not None:
        W = np.asarray(orbit_directions, dtype=float)
        if W.ndim == 1:
            W = W[np.newaxis, :]
        assert W.shape == (m_orbits, n_inputs), (
            f"orbit_directions must be ({m_orbits}, {n_inputs}), got {W.shape}"
        )
    else:
        # Build m_orbits random orthonormal rows via QR decomposition
        A = rng.standard_normal((n_inputs, n_inputs))
        Q, _ = np.linalg.qr(A)
        W = Q[:, :m_orbits].T  # (m_orbits, n_inputs)

    # Null space of W = actual translational orbit directions (y is invariant along these)
    orth = _null_space(W)  # (n_inputs, n_inputs - m_orbits)

    # Input data: uniform in [-3, 3]
    X = rng.uniform(-3.0, 3.0, (n_samples, n_inputs))

    # Latent representation: Z = X @ W.T  shape (n_samples, m_orbits)
    Z = X @ W.T

    # Nonlinear output that depends only on latent Z
    y_clean = np.sin(Z[:, 0]) + 0.5 * Z[:, 0] ** 2
    for i in range(1, m_orbits):
        y_clean = y_clean + 0.3 * np.cos(Z[:, i])

    # Additive Gaussian noise
    if noise_level > 0.0:
        y = y_clean + noise_level * np.std(y_clean) * rng.standard_normal(n_samples)
    else:
        y = y_clean.copy()

    return {
        "X": X,
        "y": y,
        "y_clean": y_clean,
        "orbit_directions": W,
        "orthogonal_directions": orth,
        "n_latent": m_orbits,
        "symmetry_type": "translational",
    }
