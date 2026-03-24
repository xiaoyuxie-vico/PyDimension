"""
Task 1 Validation Tests (1.1 – 1.7)

Run with:
    python -m data_generation.validate
or:
    python data_generation/validate.py
"""

import sys
import numpy as np

# Allow running from the repo root
sys.path.insert(0, __file__.rsplit("/data_generation", 1)[0])

from data_generation import (
    generate_translational_data,
    generate_rotational_data,
    generate_scaling_data,
)


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


def test_1_1_shape():
    print("Test 1.1 — Shape check")
    ok = True
    for label, data in [
        ("translational", generate_translational_data(n_inputs=5, m_orbits=2, n_samples=1000, seed=0)),
        ("rotational",    generate_rotational_data(n_inputs=5, n_samples=1000, seed=0)),
        ("scaling",       generate_scaling_data(n_inputs=5, m_scaling_vars=1, n_samples=1000, seed=0)),
    ]:
        ok &= check(
            f"{label} X.shape",
            data["X"].shape == (1000, 5),
            str(data["X"].shape),
        )
        ok &= check(
            f"{label} y.shape",
            data["y"].shape == (1000,),
            str(data["y"].shape),
        )
    return ok


def test_1_2_determinism():
    print("Test 1.2 — Determinism")
    ok = True
    for label, fn, kwargs in [
        ("translational", generate_translational_data, dict(n_inputs=5, m_orbits=2, n_samples=1000, seed=42)),
        ("rotational",    generate_rotational_data,    dict(n_inputs=5, n_samples=1000, seed=42)),
        ("scaling",       generate_scaling_data,       dict(n_inputs=5, m_scaling_vars=1, n_samples=1000, seed=42)),
    ]:
        d1 = fn(**kwargs)
        d2 = fn(**kwargs)
        ok &= check(f"{label} X identical",  np.allclose(d1["X"], d2["X"]))
        ok &= check(f"{label} y identical",  np.allclose(d1["y"], d2["y"]))
    return ok


def test_1_3_translational_invariance():
    print("Test 1.3 — Translational invariance")
    data = generate_translational_data(n_inputs=4, m_orbits=2, n_samples=200, noise_level=0.0, seed=7)
    X        = data["X"]               # (200, 4)
    y_orig   = data["y_clean"]
    W        = data["orbit_directions"]           # (2, 4)  measurement matrix
    orth     = data["orthogonal_directions"]      # (4, 2)  actual orbit directions

    # Shift along each orbit direction (y must be unchanged)
    eps = 0.5
    ok = True
    for col in range(orth.shape[1]):
        d = orth[:, col]
        X_shifted = X + eps * d[np.newaxis, :]
        Z_shifted = X_shifted @ W.T
        # Recompute y from ground-truth formula
        y_shifted = np.sin(Z_shifted[:, 0]) + 0.5 * Z_shifted[:, 0] ** 2
        for i in range(1, W.shape[0]):
            y_shifted = y_shifted + 0.3 * np.cos(Z_shifted[:, i])
        diff = np.max(np.abs(y_shifted - y_orig))
        ok &= check(f"orbit dir {col} max|Δy|={diff:.2e}", diff < 1e-10)
    return ok


def test_1_4_rotational_invariance():
    print("Test 1.4 — Rotational invariance")
    # coefficients=[1,1,2]: dims 0 and 1 have equal coefficients → rotation in (0,1) plane
    data  = generate_rotational_data(n_inputs=3, coefficients=[1, 1, 2], n_samples=200, noise_level=0.0, seed=3)
    X     = data["X"]
    y_orig = data["y_clean"]
    coeffs = data["coefficients"]

    theta = np.pi / 4
    R = np.eye(3)
    R[0, 0] =  np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] =  np.sin(theta)
    R[1, 1] =  np.cos(theta)

    X_rot = X @ R.T
    r_rot = (X_rot ** 2) @ coeffs
    y_rot = np.sin(r_rot) + 0.1 * r_rot

    diff = np.max(np.abs(y_rot - y_orig))
    return check(f"max|Δy|={diff:.2e} after rotation in (0,1) plane", diff < 1e-10)


def test_1_5_scaling_invariance():
    """
    Spec: exponents=[2,-1], scaling direction x1->λ^2·x1, x2->λ^{-1}·x2.

    NOTE: The spec's scaling direction [2,-1] is NOT in the null space of [[2,-1]],
    so using it directly would break invariance (as shown in stage1_tasks_review.md,
    issue C1).  We use the mathematically correct null-space direction instead.

    Null space of [[2, -1]]:  s s.t. 2s1 - s2 = 0  →  s = t·[1, 2].
    Correct scaling: x1 → λ^1·x1, x2 → λ^2·x2  (preserves x1^2·x2^{-1}).
    """
    print("Test 1.5 — Scaling invariance")
    E = np.array([[2.0, -1.0]])  # as specified
    data  = generate_scaling_data(n_inputs=2, m_scaling_vars=1,
                                   scaling_exponents=E, n_samples=200,
                                   noise_level=0.0, seed=5)
    X     = data["X"]
    y_orig = data["y_clean"]
    E_mat  = data["scaling_vectors"]      # [[2, -1]]
    orth   = data["scaling_orbit_directions"]  # null space of E, shape (2, 1)

    # Scaling direction s from null space: X_i → X_i * λ^s_i
    s = orth[:, 0]  # null-space vector
    lam = 3.0

    X_scaled = X * (lam ** s[np.newaxis, :])  # shape (200, 2)

    # Recompute y from ground-truth formula
    log_pi_scaled = np.log(X_scaled) @ E_mat.T
    pi_scaled = np.exp(log_pi_scaled)
    y_scaled = np.sin(pi_scaled[:, 0]) + 0.1 * pi_scaled[:, 0]

    diff = np.max(np.abs(y_scaled - y_orig))
    return check(f"max|Δy|={diff:.2e} (null-space direction s={s.round(4)})", diff < 1e-10)


def test_1_6_noise_sanity():
    print("Test 1.6 — Noise sanity")
    ok = True
    for label, fn, kwargs in [
        ("translational", generate_translational_data,
         dict(n_inputs=5, m_orbits=2, n_samples=10000, noise_level=0.1, seed=11)),
        ("rotational", generate_rotational_data,
         dict(n_inputs=5, n_samples=10000, noise_level=0.1, seed=11)),
        ("scaling", generate_scaling_data,
         dict(n_inputs=5, m_scaling_vars=1, n_samples=10000, noise_level=0.1, seed=11)),
    ]:
        data = fn(**kwargs)
        y, y_clean = data["y"], data["y_clean"]
        ratio = np.std(y - y_clean) / np.std(y_clean)
        ok &= check(
            f"{label} noise ratio={ratio:.4f} ∈ [0.08, 0.12]",
            0.08 <= ratio <= 0.12,
        )
    return ok


def test_1_7_scaling_positivity():
    print("Test 1.7 — Scaling positivity")
    data = generate_scaling_data(n_inputs=5, m_scaling_vars=2, n_samples=1000, seed=99)
    ok = bool(np.all(data["X"] > 0))
    return check(f"all(X > 0)", ok, f"min={data['X'].min():.4f}")


def main():
    print("=" * 60)
    print("Task 1 Validation Tests")
    print("=" * 60)
    results = {
        "1.1": test_1_1_shape(),
        "1.2": test_1_2_determinism(),
        "1.3": test_1_3_translational_invariance(),
        "1.4": test_1_4_rotational_invariance(),
        "1.5": test_1_5_scaling_invariance(),
        "1.6": test_1_6_noise_sanity(),
        "1.7": test_1_7_scaling_positivity(),
    }
    print()
    print("=" * 60)
    passed = sum(results.values())
    total  = len(results)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All Task 1 tests PASSED.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {failed}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
