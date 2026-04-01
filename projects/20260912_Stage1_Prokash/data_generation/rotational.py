"""
Rotational symmetry data generator.

y = f(a_1·x_1² + a_2·x_2² + ... + a_n·x_n²)

Invariant under rotations that mix dimensions with equal coefficients.
"""

import numpy as np


def generate_rotational_data(
    n_inputs: int,
    coefficients: np.ndarray = None,
    n_samples: int = 1000,
    noise_level: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic data with rotational symmetry.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    coefficients : array-like, shape (n_inputs,)
        Quadratic coefficients [a_1, ..., a_n]. Dimensions with equal coefficients
        share a rotational symmetry.  Defaults to all-ones.
    n_samples : int
        Number of data points.
    noise_level : float
        Additive Gaussian noise on y: noise ~ N(0, noise_level * std(y_clean)).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        X            : (n_samples, n_inputs)  input data
        y            : (n_samples,)           output (possibly noisy)
        y_clean      : (n_samples,)           noise-free output
        coefficients : (n_inputs,)            ground-truth quadratic coefficients
        n_latent     : 1
        symmetry_type: "rotational"
    """
    rng = np.random.default_rng(seed)

    if coefficients is None:
        coefficients = np.ones(n_inputs, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    assert len(coefficients) == n_inputs, (
        f"len(coefficients)={len(coefficients)} must equal n_inputs={n_inputs}"
    )

    # Input data: standard normal (natural for rotational problems)
    X = rng.standard_normal((n_samples, n_inputs))

    # Rotational invariant: r = sum_i a_i * x_i^2
    r = (X ** 2) @ coefficients  # (n_samples,)

    # Nonlinear output: sin + linear to avoid trivial constant regions
    y_clean = np.sin(r) + 0.1 * r

    # Additive Gaussian noise
    if noise_level > 0.0:
        y = y_clean + noise_level * np.std(y_clean) * rng.standard_normal(n_samples)
    else:
        y = y_clean.copy()

    return {
        "X": X,
        "y": y,
        "y_clean": y_clean,
        "coefficients": coefficients,
        "n_latent": 1,
        "symmetry_type": "rotational",
    }
