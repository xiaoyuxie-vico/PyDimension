"""
Lie-algebra generator extraction from trained symmetry encoders.

Three symmetry types, three generator families:

    translational : null space of W   →  vectors g  with W·g = 0
    scaling       : null space of W   →  log-space vectors s  with W·s = 0
                    (applied as x → x · exp(ε·s))
    rotational    : antisymmetric n×n matrices A  for each equal-coefficient pair

References
----------
Task spec §5:
  "Translational & Scaling: compute null space of W via SVD.
   Rotational: cluster columns of W by coefficient similarity,
               build antisymmetric matrices for each pair within a cluster."
"""

import numpy as np
from scipy.linalg import null_space, expm


# ------------------------------------------------------------------
# Generator extraction
# ------------------------------------------------------------------

def extract_generators(
    symmetry_type: str,
    encoder,
    cluster_tol: float = 0.25,
) -> list:
    """
    Extract Lie-algebra generators from a trained SymmetryEncoder.

    Parameters
    ----------
    symmetry_type : str
        "translational", "rotational", or "scaling".
    encoder : SymmetryEncoder
        Trained encoder returned by identify_symmetry().
    cluster_tol : float
        For rotational: relative tolerance (fraction of weight range) used
        to decide whether two coefficient values belong to the same cluster.

    Returns
    -------
    generators : list
        translational / scaling  → list of 1-D np.ndarray of shape (n_inputs,)
        rotational               → list of 2-D np.ndarray of shape (n_inputs, n_inputs),
                                   each antisymmetric (A + A^T = 0)
    """
    W = encoder.weight_matrix  # (n_latent, n_inputs)

    if symmetry_type in ("translational", "scaling"):
        # Null space columns are the generator directions
        ns = null_space(W)          # (n_inputs, n_inputs - n_latent)
        return [ns[:, i] for i in range(ns.shape[1])]

    elif symmetry_type == "rotational":
        # For rotational symmetry the encoder has a single row (n_latent == 1
        # in the standard case) giving the quadratic coefficients w_i such
        # that z = Σ w_i · x_i².  Dimensions with equal w_i can be mixed by
        # a rotation without changing z.
        #
        # If n_latent > 1 (unusual) we use the row with the largest L2 norm
        # as the representative coefficient vector.
        if W.shape[0] == 1:
            w = W[0]
        else:
            row_norms = np.linalg.norm(W, axis=1)
            w = W[np.argmax(row_norms)]

        n = len(w)
        w_abs = np.abs(w)

        # Normalise to [0, 1] so cluster_tol is scale-independent
        w_range = float(w_abs.max() - w_abs.min())
        if w_range < 1e-12:
            # All coefficients equal → every pair is a generator
            w_norm = np.zeros(n)
        else:
            w_norm = (w_abs - w_abs.min()) / w_range

        # Sort indices by normalised weight value
        order      = np.argsort(w_norm)
        w_sorted   = w_norm[order]

        # Greedy grouping: consecutive sorted values within cluster_tol
        clusters        = []
        current_cluster = [order[0]]
        for i in range(1, n):
            if abs(w_sorted[i] - w_sorted[i - 1]) < cluster_tol:
                current_cluster.append(order[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [order[i]]
        clusters.append(current_cluster)

        # One antisymmetric matrix per pair (i, j) within a cluster of size ≥ 2
        generators = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            for a in range(len(cluster)):
                for b in range(a + 1, len(cluster)):
                    idx_i, idx_j = cluster[a], cluster[b]
                    A = np.zeros((n, n))
                    A[idx_i, idx_j] = -1.0
                    A[idx_j, idx_i] =  1.0
                    generators.append(A)

        return generators

    else:
        raise ValueError(
            f"Unknown symmetry_type '{symmetry_type}'. "
            "Must be 'translational', 'rotational', or 'scaling'."
        )


# ------------------------------------------------------------------
# Infinitesimal transformation
# ------------------------------------------------------------------

def apply_generator(
    x: np.ndarray,
    generator: np.ndarray,
    epsilon: float,
    symmetry_type: str,
) -> np.ndarray:
    """
    Apply an infinitesimal symmetry transformation to a single data point.

    Parameters
    ----------
    x : (n_inputs,)
        Input data point (numpy 1-D array).
    generator : np.ndarray
        translational / scaling : 1-D array (n_inputs,)
        rotational              : 2-D antisymmetric array (n_inputs, n_inputs)
    epsilon : float
        Step size.
    symmetry_type : str

    Returns
    -------
    x_new : (n_inputs,)
    """
    x = np.asarray(x, dtype=float)

    if symmetry_type == "translational":
        # x → x + ε·g
        return x + epsilon * generator

    elif symmetry_type == "scaling":
        # In log space: log(x) → log(x) + ε·s
        # Equivalently:  x_i  → x_i · exp(ε · s_i)
        return x * np.exp(epsilon * generator)

    elif symmetry_type == "rotational":
        # Infinitesimal rotation: x → (I + ε·A)·x
        return x + epsilon * (generator @ x)

    else:
        raise ValueError(f"Unknown symmetry_type '{symmetry_type}'.")


# ------------------------------------------------------------------
# Orbit tracing
# ------------------------------------------------------------------

def generator_orbit(
    x_start: np.ndarray,
    generator: np.ndarray,
    n_steps: int,
    epsilon: float,
    symmetry_type: str,
) -> np.ndarray:
    """
    Trace an orbit from a starting point by iteratively applying a generator.

    For rotational symmetry, uses the exact matrix exponential at each step
    to ensure numerical accuracy and orbit closure at 2π.

    Parameters
    ----------
    x_start : (n_inputs,)
    generator : np.ndarray
        Generator from extract_generators().
    n_steps : int
        Number of steps along the orbit.
    epsilon : float
        Arc-length increment per step.
    symmetry_type : str

    Returns
    -------
    orbit : (n_steps + 1, n_inputs)
        Trajectory including the starting point at index 0.
    """
    x_start = np.asarray(x_start, dtype=float)
    n_inputs = len(x_start)
    orbit = np.empty((n_steps + 1, n_inputs))
    orbit[0] = x_start

    if symmetry_type == "translational":
        for k in range(1, n_steps + 1):
            orbit[k] = x_start + k * epsilon * generator

    elif symmetry_type == "scaling":
        for k in range(1, n_steps + 1):
            orbit[k] = x_start * np.exp(k * epsilon * generator)

    elif symmetry_type == "rotational":
        # x(k) = expm(k · ε · A) @ x_start  (exact, no Euler drift)
        A = generator
        for k in range(1, n_steps + 1):
            orbit[k] = expm(k * epsilon * A) @ x_start

    else:
        raise ValueError(f"Unknown symmetry_type '{symmetry_type}'.")

    return orbit
