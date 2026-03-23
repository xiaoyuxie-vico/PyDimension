"""
Scaling symmetry data generator.

y = f(x_1^a_1 · x_2^a_2 · ..., x_1^b_1 · x_2^b_2 · ..., ...)

All inputs X are strictly positive (log-uniform distribution).
Invariant under x_i → λ^s_i · x_i where scaling_exponents @ s = 0.
"""

import numpy as np
from scipy.linalg import null_space as _null_space


def generate_scaling_data(
    n_inputs: int,
    m_scaling_vars: int = 1,
    scaling_exponents: np.ndarray = None,
    n_samples: int = 1000,
    noise_level: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic data with scaling symmetry.

    Parameters
    ----------
    n_inputs : int
        Number of input features (all strictly positive).
    m_scaling_vars : int
        Number of dimensionless scaling variables (= true latent dimension).
    scaling_exponents : array-like, shape (m_scaling_vars, n_inputs) or (n_inputs,)
        Exponent matrix E. Each row defines one dimensionless group:
            pi_k = prod_i x_i^{E[k,i]}
        If 1-D, treated as a single row (m_scaling_vars=1).
        If None, random integer exponents in [-2, 2] are generated.
    n_samples : int
        Number of data points.
    noise_level : float
        Additive Gaussian noise on y: noise ~ N(0, noise_level * std(y_clean)).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        X                       : (n_samples, n_inputs)         input data, X > 0
        y                       : (n_samples,)                  output (possibly noisy)
        y_clean                 : (n_samples,)                  noise-free output
        scaling_vectors         : (m_scaling_vars, n_inputs)    exponent matrix E
        scaling_orbit_directions: (n_inputs, n_inputs-m_scaling_vars)
                                  null space of E (scaling-invariant directions)
        n_latent                : m_scaling_vars
        symmetry_type           : "scaling"
    """
    rng = np.random.default_rng(seed)

    if scaling_exponents is not None:
        E = np.atleast_2d(np.asarray(scaling_exponents, dtype=float))
        m_scaling_vars = E.shape[0]
    else:
        # Generate random integer exponents, avoid all-zero rows
        E = np.zeros((m_scaling_vars, n_inputs), dtype=float)
        for k in range(m_scaling_vars):
            while True:
                row = rng.integers(-2, 3, size=n_inputs).astype(float)
                if not np.allclose(row, 0):
                    E[k] = row
                    break

    assert E.shape == (m_scaling_vars, n_inputs), (
        f"scaling_exponents must be ({m_scaling_vars}, {n_inputs}), got {E.shape}"
    )

    # Positive inputs: log-uniform (uniform in log space → X = exp(U))
    log_X = rng.uniform(-2.0, 2.0, (n_samples, n_inputs))
    X = np.exp(log_X)  # strictly positive

    # Scaling variables in log space: log(pi_k) = sum_i E[k,i] * log(X_i)
    log_pi = log_X @ E.T  # (n_samples, m_scaling_vars)
    pi = np.exp(log_pi)   # (n_samples, m_scaling_vars)

    # Nonlinear output
    y_clean = np.sin(pi[:, 0]) + 0.1 * pi[:, 0]
    for k in range(1, m_scaling_vars):
        y_clean = y_clean + 0.2 * np.tanh(pi[:, k])

    # Additive Gaussian noise
    if noise_level > 0.0:
        y = y_clean + noise_level * np.std(y_clean) * rng.standard_normal(n_samples)
    else:
        y = y_clean.copy()

    # Null space of E = scaling-invariant directions
    orth = _null_space(E)  # (n_inputs, n_inputs - m_scaling_vars)

    return {
        "X": X,
        "y": y,
        "y_clean": y_clean,
        "scaling_vectors": E,
        "scaling_orbit_directions": orth,
        "n_latent": m_scaling_vars,
        "symmetry_type": "scaling",
    }
