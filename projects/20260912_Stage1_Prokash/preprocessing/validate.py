"""
Task 2 Validation Tests (2.1 – 2.4)

Run from repo root:
    python projects/20260912_Stage1_Prokash/preprocessing/validate.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from preprocessing import normalize_data
from data_generation import generate_translational_data

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


def test_2_1_standard_stats():
    print("Test 2.1 — Standard normalization stats")
    data = generate_translational_data(n_inputs=5, m_orbits=2, n_samples=1000, seed=42)
    result = normalize_data(data["X"], data["y"], method="standard")
    X_norm = result["X_normalized"]
    ok = True
    for col in range(X_norm.shape[1]):
        mean = X_norm[:, col].mean()
        std  = X_norm[:, col].std()
        ok &= check(f"col {col} mean≈0", abs(mean) < 1e-10, f"{mean:.2e}")
        ok &= check(f"col {col} std≈1",  abs(std - 1) < 1e-10, f"{std:.6f}")
    return ok


def test_2_2_minmax_range():
    print("Test 2.2 — MinMax range")
    data = generate_translational_data(n_inputs=5, m_orbits=2, n_samples=1000, seed=42)
    result = normalize_data(data["X"], data["y"], method="minmax")
    X_norm = result["X_normalized"]
    ok = True
    for col in range(X_norm.shape[1]):
        lo = X_norm[:, col].min()
        hi = X_norm[:, col].max()
        ok &= check(f"col {col} min≥0", lo >= 0, f"{lo:.6f}")
        ok &= check(f"col {col} max≤1", hi <= 1, f"{hi:.6f}")
    return ok


def test_2_3_round_trip():
    print("Test 2.3 — Round-trip (normalize → inverse)")
    data = generate_translational_data(n_inputs=5, m_orbits=2, n_samples=500, seed=42)
    X_orig, y_orig = data["X"], data["y"]
    ok = True
    for method in ("standard", "minmax", "robust"):
        result = normalize_data(X_orig, y_orig, method=method)
        X_rec = result["scaler_X"].inverse_transform(result["X_normalized"])
        y_rec = result["scaler_y"].inverse_transform(
            result["y_normalized"].reshape(-1, 1)
        ).ravel()
        ok &= check(f"{method} X recovered", np.allclose(X_rec, X_orig, atol=1e-10))
        ok &= check(f"{method} y recovered", np.allclose(y_rec, y_orig, atol=1e-10))
    return ok


def test_2_4_integration():
    print("Test 2.4 — Integration with data generator")
    data = generate_translational_data(n_inputs=5, m_orbits=2, n_samples=800, seed=42)
    X, y = data["X"], data["y"]
    result = normalize_data(X, y, method="standard")
    ok = True
    ok &= check("X_normalized shape preserved",
                result["X_normalized"].shape == X.shape,
                str(result["X_normalized"].shape))
    ok &= check("y_normalized shape preserved",
                result["y_normalized"].shape == y.shape,
                str(result["y_normalized"].shape))
    return ok


def main():
    print("=" * 60)
    print("Task 2 Validation Tests")
    print("=" * 60)
    results = {
        "2.1": test_2_1_standard_stats(),
        "2.2": test_2_2_minmax_range(),
        "2.3": test_2_3_round_trip(),
        "2.4": test_2_4_integration(),
    }
    print()
    print("=" * 60)
    passed = sum(results.values())
    total  = len(results)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All Task 2 tests PASSED.")
    else:
        print(f"Failed: {[k for k, v in results.items() if not v]}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
