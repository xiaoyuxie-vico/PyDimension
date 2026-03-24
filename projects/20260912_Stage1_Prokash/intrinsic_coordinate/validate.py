"""
Task 3 Validation Tests (3.1 – 3.6)

Run from repo root:
    python projects/20260912_Stage1_Prokash/intrinsic_coordinate/validate.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from data_generation import (
    generate_translational_data,
    generate_rotational_data,
    generate_scaling_data,
)
from intrinsic_coordinate import discover_latent_dimension

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


# Shared discovery results (computed once per symmetry type)
_results = {}

def _get_result(key, data, max_latent=5):
    if key not in _results:
        print(f"  [training {key}...]")
        _results[key] = discover_latent_dimension(
            data["X"], data["y"],
            max_latent=max_latent,
            n_epochs=600,
            hidden_dim=64,
            seed=0,
        )
    return _results[key]


def test_3_1_translational():
    print("Test 3.1 — Known dimension: translational (n_inputs=5, m_orbits=2 → latent=2)")
    data = generate_translational_data(n_inputs=5, m_orbits=2, n_samples=2000, seed=42)
    res  = _get_result("trans", data)
    k    = res["optimal_n_latent"]
    return check(f"optimal_n_latent == 2", k == 2, f"got {k}")


def test_3_2_rotational():
    print("Test 3.2 — Known dimension: rotational (n_inputs=4 → latent=1)")
    data = generate_rotational_data(n_inputs=4, n_samples=2000, seed=42)
    res  = _get_result("rot", data)
    k    = res["optimal_n_latent"]
    return check(f"optimal_n_latent == 1", k == 1, f"got {k}")


def test_3_3_scaling():
    print("Test 3.3 — Known dimension: scaling (n_inputs=3, m_scaling_vars=1 → latent=1)")
    data = generate_scaling_data(n_inputs=3, m_scaling_vars=1, n_samples=2000, seed=42)
    res  = _get_result("scale", data)
    k    = res["optimal_n_latent"]
    return check(f"optimal_n_latent == 1", k == 1, f"got {k}")


def test_3_4_r2_quality():
    print("Test 3.4 — R² > 0.95 at optimal n_latent")
    ok = True
    for key in ("trans", "rot", "scale"):
        if key not in _results:
            continue
        res = _results[key]
        k   = res["optimal_n_latent"]
        r2  = res["metrics"][k]["R2"]
        ok &= check(f"{key} R²={r2:.4f} > 0.95", r2 > 0.95)
    return ok


def test_3_5_monotonicity():
    print("Test 3.5 — R² non-decreasing as n_latent increases")
    ok = True
    for key in ("trans", "rot", "scale"):
        if key not in _results:
            continue
        metrics = _results[key]["metrics"]
        ks = sorted(metrics.keys())
        for i in range(len(ks) - 1):
            k1, k2 = ks[i], ks[i + 1]
            r2_k1 = metrics[k1]["R2"]
            r2_k2 = metrics[k2]["R2"]
            ok &= check(
                f"{key} R²({k2}) >= R²({k1}) - 0.02",
                r2_k2 >= r2_k1 - 0.02,
                f"{r2_k2:.4f} vs {r2_k1:.4f}",
            )
    return ok


def test_3_6_decoder_shape():
    print("Test 3.6 — best_decoder accepts (batch, n_latent) → (batch, 1)")
    ok = True
    for key in ("trans", "rot", "scale"):
        if key not in _results:
            continue
        res     = _results[key]
        k       = res["optimal_n_latent"]
        decoder = res["best_decoder"]
        device  = next(decoder.parameters()).device
        z       = torch.zeros(8, k, device=device)
        with torch.no_grad():
            out = decoder(z)
        ok &= check(f"{key} decoder output shape", out.shape == (8, 1), str(tuple(out.shape)))
    return ok


def main():
    print("=" * 60)
    print("Task 3 Validation Tests")
    print("=" * 60)

    # Pre-generate all datasets so models are trained in order
    data_trans = generate_translational_data(n_inputs=5, m_orbits=2, n_samples=2000, seed=42)
    data_rot   = generate_rotational_data(n_inputs=4, n_samples=2000, seed=42)
    data_scale = generate_scaling_data(n_inputs=3, m_scaling_vars=1, n_samples=2000, seed=42)

    print("Pre-training models (this may take ~1 min)...")
    _get_result("trans", data_trans)
    _get_result("rot",   data_rot)
    _get_result("scale", data_scale)
    print()

    results = {
        "3.1": test_3_1_translational(),
        "3.2": test_3_2_rotational(),
        "3.3": test_3_3_scaling(),
        "3.4": test_3_4_r2_quality(),
        "3.5": test_3_5_monotonicity(),
        "3.6": test_3_6_decoder_shape(),
    }

    print()
    print("=" * 60)
    passed = sum(results.values())
    total  = len(results)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All Task 3 tests PASSED.")
    else:
        print(f"Failed: {[k for k, v in results.items() if not v]}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
