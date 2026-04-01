"""
Task 4 Validation Tests (4.1 – 4.6)

Run from repo root:
    python projects/20260912_Stage1_Prokash/symmetry_discovery/validate.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from data_generation import (
    generate_translational_data,
    generate_rotational_data,
    generate_scaling_data,
)
from intrinsic_coordinate import discover_latent_dimension
from symmetry_discovery import identify_symmetry

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    return condition


# ---------------------------------------------------------------------------
# Cache: run full pipeline once per symmetry type
# ---------------------------------------------------------------------------
_pipeline = {}

def _run_pipeline(key, data, max_latent=4):
    if key not in _pipeline:
        X, y = data["X"], data["y"]
        print(f"  [Task 3: discover_latent_dimension for {key}...]")
        res3 = discover_latent_dimension(X, y, max_latent=max_latent,
                                         n_epochs=600, seed=0)
        n_latent = res3["optimal_n_latent"]
        decoder  = res3["best_decoder"]
        print(f"  [Task 4: identify_symmetry for {key} (n_latent={n_latent})...]")
        res4 = identify_symmetry(X, y, n_latent, decoder,
                                  n_epochs=1500, seed=0)
        _pipeline[key] = {"data": data, "res3": res3, "res4": res4}
    return _pipeline[key]


# ---------------------------------------------------------------------------
# Tests 4.1 – 4.3: type detection
# ---------------------------------------------------------------------------

def test_4_1_translational():
    print("Test 4.1 — Type detection: translational")
    data = generate_translational_data(n_inputs=5, m_orbits=2, n_samples=2000, seed=42)
    r    = _run_pipeline("trans", data)
    t    = r["res4"]["symmetry_type"]
    return check("symmetry_type == 'translational'", t == "translational", f"got '{t}'")


def test_4_2_rotational():
    print("Test 4.2 — Type detection: rotational")
    data = generate_rotational_data(n_inputs=4, coefficients=[1, 2, 1, 3],
                                    n_samples=2000, seed=42)
    r    = _run_pipeline("rot", data)
    t    = r["res4"]["symmetry_type"]
    return check("symmetry_type == 'rotational'", t == "rotational", f"got '{t}'")


def test_4_3_scaling():
    print("Test 4.3 — Type detection: scaling")
    data = generate_scaling_data(n_inputs=3, m_scaling_vars=1,
                                  scaling_exponents=[[1, -1, 0]],
                                  n_samples=2000, seed=42)
    r    = _run_pipeline("scale", data)
    t    = r["res4"]["symmetry_type"]
    return check("symmetry_type == 'scaling'", t == "scaling", f"got '{t}'")


# ---------------------------------------------------------------------------
# Test 4.4: loss gap ≥ 2×
# ---------------------------------------------------------------------------

def test_4_4_loss_gap():
    print("Test 4.4 — Loss gap ≥ 2×")
    ok = True
    for key in ("trans", "rot", "scale"):
        if key not in _pipeline:
            continue
        losses  = _pipeline[key]["res4"]["losses"]
        winner  = _pipeline[key]["res4"]["symmetry_type"]
        sorted_ = sorted(losses.items(), key=lambda kv: kv[1])
        best_loss   = sorted_[0][1]
        second_loss = sorted_[1][1]
        gap         = second_loss / (best_loss + 1e-12)
        ok &= check(
            f"{key}: winner={winner}, gap={gap:.2f}× (≥2× required)",
            gap >= 2.0,
        )
    return ok


# ---------------------------------------------------------------------------
# Test 4.5: rotational coefficient recovery [1, 2, 1, 3]
# ---------------------------------------------------------------------------

def test_4_5_rotational_coefficients():
    print("Test 4.5 — Rotational coefficient recovery (truth=[1,2,1,3])")
    key  = "rot"
    if key not in _pipeline:
        return check("pipeline not run", False)
    res4  = _pipeline[key]["res4"]
    coeff = np.asarray(res4["coefficients"], dtype=float).ravel()
    truth = np.array([1., 2., 1., 3.])

    if len(coeff) != len(truth):
        return check(f"length mismatch {len(coeff)} vs {len(truth)}", False)

    # Normalise both by their first element
    detected_norm = coeff / coeff[0]
    truth_norm    = truth / truth[0]
    rel_err = np.abs(detected_norm - truth_norm)
    max_err = float(np.max(rel_err))

    return check(
        f"max relative error < 0.1 (detected_norm={np.round(detected_norm,3)})",
        max_err < 0.1,
        f"max_err={max_err:.4f}",
    )


# ---------------------------------------------------------------------------
# Test 4.6: scaling exponent recovery [[1, -1, 0]]
# ---------------------------------------------------------------------------

def test_4_6_scaling_coefficients():
    print("Test 4.6 — Scaling exponent recovery (truth=[1,-1,0])")
    key  = "scale"
    if key not in _pipeline:
        return check("pipeline not run", False)
    res4  = _pipeline[key]["res4"]
    coeff = np.asarray(res4["coefficients"], dtype=float).ravel()
    truth = np.array([1., -1., 0.])

    if len(coeff) != len(truth):
        return check(f"length mismatch {len(coeff)} vs {len(truth)}", False)

    # Normalise both by first element (handles sign flip)
    detected_norm = coeff / coeff[0]
    truth_norm    = truth / truth[0]
    rel_err = np.abs(detected_norm - truth_norm)

    # Third component: truth is 0, so use absolute error for that element
    max_err = float(np.max(np.abs(detected_norm - truth_norm)))

    return check(
        f"max error < 0.1 (detected_norm={np.round(detected_norm,3)})",
        max_err < 0.1,
        f"max_err={max_err:.4f}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Task 4 Validation Tests")
    print("=" * 60)

    # Pre-generate + pre-run all pipelines
    data_trans = generate_translational_data(n_inputs=5, m_orbits=2,
                                              n_samples=2000, seed=42)
    data_rot   = generate_rotational_data(n_inputs=4, coefficients=[1, 2, 1, 3],
                                           n_samples=2000, seed=42)
    data_scale = generate_scaling_data(n_inputs=3, m_scaling_vars=1,
                                        scaling_exponents=[[1, -1, 0]],
                                        n_samples=2000, seed=42)

    print("Running full Task 3+4 pipeline for all symmetry types …")
    _run_pipeline("trans", data_trans)
    _run_pipeline("rot",   data_rot)
    _run_pipeline("scale", data_scale)
    print()

    results = {
        "4.1": test_4_1_translational(),
        "4.2": test_4_2_rotational(),
        "4.3": test_4_3_scaling(),
        "4.4": test_4_4_loss_gap(),
        "4.5": test_4_5_rotational_coefficients(),
        "4.6": test_4_6_scaling_coefficients(),
    }

    print()
    print("=" * 60)
    passed = sum(results.values())
    total  = len(results)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All Task 4 tests PASSED.")
    else:
        print(f"Failed: {[k for k, v in results.items() if not v]}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
