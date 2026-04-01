"""
Discover symmetry in permeability vs angle using Stage1.

Physics
-------
Porous media permeability may exhibit angular periodicity — at certain
angular intervals the permeability repeats. We use only:
    Input:  Angle (decomposed into cos θ, sin θ)
    Output: Permeability_X

With 2 inputs (cos θ, sin θ) and 1 latent dimension, there is 1 generator.
If rotational symmetry is detected, permeability depends on the angle through
a rotationally invariant combination (cos²θ + sin²θ), meaning it's constant.
If translational, it depends on a linear combination a·cos θ + b·sin θ,
which equals A·cos(θ - φ) — a single harmonic with amplitude A and phase φ.

Usage
-----
    python discover_symmetry.py --data permeability.csv
    python discover_symmetry.py --synthetic
"""

import sys
import os
import argparse
import traceback
import multiprocessing

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.join(_here, "..", ".."),
    os.path.join(_here, "..", "..", "projects", "20260912_Stage1_Prokash"),
    _here,
]:
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(os.path.join(_candidate, "preprocessing")):
        sys.path.insert(0, _candidate)
        break

try:
    import matplotlib
    matplotlib.use("Agg")
except (AttributeError, ImportError):
    pass
import matplotlib.pyplot as plt

try:
    from preprocessing.normalize import normalize_data
    from intrinsic_coordinate.discovery import discover_latent_dimension
    from symmetry_discovery.identification import identify_symmetry
    from symmetry_discovery.generators import extract_generators, generator_orbit
except ImportError as e:
    print(f"ERROR: Could not import Stage1 modules: {e}")
    sys.exit(1)

import torch.multiprocessing as _tmp
_tmp.cpu_count = lambda: 0

VARIABLE_NAMES = ["cos(Angle)", "sin(Angle)"]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n_samples=300, seed=42):
    """Synthetic data with angular periodicity."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 360, n_samples)
    angle_rad = np.radians(angle)
    # Permeability with cos(2θ) periodicity (repeats every 180°)
    y = 3.7 + 0.3 * np.cos(2 * angle_rad) + 0.1 * np.sin(2 * angle_rad)
    y += rng.normal(0, 0.03, n_samples)
    return angle, y


def load_csv_data(csv_path):
    """Load angle and Permeability_X from CSV or Excel."""
    ext = os.path.splitext(csv_path)[1].lower()
    if ext in (".xls", ".xlsx"):
        import pandas as pd
        df = pd.read_excel(csv_path)
    else:
        import pandas as pd
        df = pd.read_csv(csv_path)

    print(f"  Columns: {df.columns.tolist()}")

    # Find columns
    angle_col = None
    perm_col = None
    for col in df.columns:
        cl = col.strip().lower()
        if "angle" in cl:
            angle_col = col
        elif "permeability" in cl and "x" in cl:
            perm_col = col

    if not angle_col or not perm_col:
        # Fallback: positional (Index, Angle, ..., ..., Permeability_X)
        angle_col = df.columns[1]
        perm_col = df.columns[4]

    print(f"  Using: {angle_col} -> {perm_col}")
    angle = df[angle_col].values.astype(float)
    y = df[perm_col].values.astype(float)
    return angle, y


def load_data(args):
    """Load or generate data, return angle (degrees), X (cos/sin), y."""
    if args.synthetic:
        print("Generating synthetic data...")
        angle, y = generate_synthetic_data(args.n_samples, args.seed)
    else:
        data_path = args.data
        if not os.path.exists(data_path):
            data_path = os.path.join(_here, args.data)
        if os.path.exists(data_path):
            print(f"Loading from {data_path}...")
            angle, y = load_csv_data(data_path)
        else:
            print(f"'{args.data}' not found. Using synthetic.")
            angle, y = generate_synthetic_data(args.n_samples, args.seed)

    # Decompose angle into cos, sin
    angle_rad = np.radians(angle)
    X = np.column_stack([np.cos(angle_rad), np.sin(angle_rad)])

    print(f"  Samples: {len(angle)}")
    print(f"  Angle range: [{angle.min():.1f}, {angle.max():.1f}] degrees")
    print(f"  Permeability range: [{y.min():.4f}, {y.max():.4f}]")
    print()
    return angle, X, y


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(angle, X, y, args):
    """Run Stage1 with all normalizations, then extract equation."""
    methods = ["standard", "minmax", "robust"]
    all_results = {}

    # --- Sweep normalizations ---
    print("=" * 60)
    print("Normalization & Symmetry Sweep")
    print("=" * 60)

    for method in methods:
        print(f"\n--- {method} ---")
        sys.stdout.flush()
        norm = normalize_data(X, y, method=method)
        X_norm, y_norm = norm["X_normalized"], norm["y_normalized"]

        res_latent = discover_latent_dimension(
            X_norm, y_norm, max_latent=2,
            n_epochs=args.latent_epochs, n_restarts=args.n_restarts, seed=args.seed,
        )
        n_latent = res_latent["optimal_n_latent"]

        res_sym = identify_symmetry(
            X_norm, y_norm, n_latent=n_latent, decoder=res_latent["best_decoder"],
            n_epochs=args.sym_epochs, n_restarts=args.n_restarts, seed=args.seed,
        )

        for stype, loss in sorted(res_sym["losses"].items(), key=lambda kv: kv[1]):
            marker = " <--" if stype == res_sym["symmetry_type"] else ""
            print(f"    {stype:15s}: {loss:.6f}{marker}")

        all_results[method] = {
            "normalization": norm, "latent": res_latent, "symmetry": res_sym,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    best_method = None
    best_loss = np.inf
    for method in methods:
        sym = all_results[method]["symmetry"]
        loss = min(sym["losses"].values())
        losses_sorted = sorted(sym["losses"].values())
        gap = losses_sorted[1] / (losses_sorted[0] + 1e-12) if len(losses_sorted) >= 2 else 0
        print(f"  {method:<10} {sym['symmetry_type']:<16} MSE={loss:.6f}  gap={gap:.1f}x")
        if loss < best_loss:
            best_loss = loss
            best_method = method

    print(f"\n  Best: {best_method}")

    # --- Full pipeline with best ---
    print("\n" + "=" * 60)
    print(f"Full Pipeline ({best_method})")
    print("=" * 60)
    norm = all_results[best_method]["normalization"]
    res_latent = all_results[best_method]["latent"]
    res_sym = all_results[best_method]["symmetry"]
    X_norm = norm["X_normalized"]
    n_latent = res_latent["optimal_n_latent"]
    winner_type = res_sym["symmetry_type"]
    winner_encoder = res_sym["encoders"][winner_type]
    W = winner_encoder.weight_matrix
    generators = extract_generators(winner_type, winner_encoder)

    print(f"  Latent dim: {n_latent}")
    print(f"  Symmetry: {winner_type}")
    print(f"  Generators: {len(generators)}")
    print(f"  W: {W}")

    results = {
        "all_results": all_results,
        "best_method": best_method,
        "normalization": norm,
        "latent": res_latent,
        "symmetry": res_sym,
        "winner_type": winner_type,
        "winner_encoder": winner_encoder,
        "generators": generators,
        "W": W,
    }

    # --- Equation discovery ---
    print("\n" + "=" * 60)
    print("Equation Discovery")
    print("=" * 60)
    discover_equation(angle, X, y, results)

    return results


def discover_equation(angle, X, y, results):
    """
    Extract a closed-form equation: Permeability_X = f(Angle).

    For translational symmetry with (cos θ, sin θ) inputs:
        z = w1·cos(θ) + w2·sin(θ) = A·cos(θ - φ)
        where A = sqrt(w1² + w2²), φ = atan2(w2, w1)

    Then fit y = polynomial(z) to get the full equation.
    """
    norm = results["normalization"]
    W = results["W"]
    winner_type = results["winner_type"]
    scaler_X = norm["scaler_X"]
    X_norm = norm["X_normalized"]

    # Step 1: Compute z in normalized space
    z_norm = (X_norm @ W.T)  # (n, n_latent)

    if z_norm.shape[1] == 1:
        z = z_norm.ravel()

        # Step 2: Fit y = f(z) with polynomials
        print(f"\n  Fitting y = f(z) where z = W · [cos θ, sin θ]_norm:")
        best_r2 = -np.inf
        best_deg = 1
        best_coeffs = None

        for deg in range(1, 5):
            c = np.polyfit(z, y, deg)
            y_pred = np.polyval(c, z)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            marker = ""
            if r2 > best_r2 + 0.005:  # require meaningful improvement
                best_r2 = r2
                best_deg = deg
                best_coeffs = c
                marker = " <-- best"
            print(f"    degree {deg}: R² = {r2:.6f}{marker}")

        # Step 3: Express z in terms of original angle
        # z_norm = W · x_norm, and x_norm = scaler_X.transform(x)
        # For cos/sin features: get the transformation back to raw cos/sin
        w = W[0]  # (2,) weights for [cos_norm, sin_norm]

        if hasattr(scaler_X, 'mean_') and hasattr(scaler_X, 'scale_'):
            mean = scaler_X.center_ if hasattr(scaler_X, 'center_') else scaler_X.mean_
            scale = scaler_X.scale_
        elif hasattr(scaler_X, 'data_min_') and hasattr(scaler_X, 'data_range_'):
            mean = scaler_X.data_min_
            scale = scaler_X.data_range_
        else:
            mean = np.zeros(2)
            scale = np.ones(2)

        # z = w[0]*(cos θ - mean[0])/scale[0] + w[1]*(sin θ - mean[1])/scale[1]
        # z = (w[0]/scale[0])*cos θ + (w[1]/scale[1])*sin θ + offset
        a_cos = w[0] / scale[0]
        a_sin = w[1] / scale[1]
        offset = -(w[0] * mean[0] / scale[0] + w[1] * mean[1] / scale[1])

        # a_cos·cos θ + a_sin·sin θ = A·cos(θ - φ)
        A = np.sqrt(a_cos**2 + a_sin**2)
        phi = np.degrees(np.arctan2(a_sin, a_cos))

        print(f"\n  Latent variable decomposition:")
        print(f"    z = {a_cos:.6f}·cos(θ) + {a_sin:.6f}·sin(θ) + {offset:.6f}")
        print(f"    z = {A:.6f}·cos(θ - {phi:.1f}°) + {offset:.6f}")

        # Step 4: Verify by direct harmonic fit on angle
        print(f"\n  Direct harmonic fit (bypassing encoder):")
        angle_rad = np.radians(angle)
        for n_harm in range(1, 4):
            # Build Fourier basis: cos(kθ), sin(kθ) for k=1..n_harm + constant
            basis = [np.ones(len(angle))]
            basis_names = ["1"]
            for k in range(1, n_harm + 1):
                basis.append(np.cos(k * angle_rad))
                basis_names.append(f"cos({k}θ)")
                basis.append(np.sin(k * angle_rad))
                basis_names.append(f"sin({k}θ)")
            B = np.column_stack(basis)
            coeffs_harm, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
            y_pred_harm = B @ coeffs_harm
            ss_res = np.sum((y - y_pred_harm) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2_harm = 1 - ss_res / (ss_tot + 1e-12)
            print(f"    {n_harm} harmonic(s): R² = {r2_harm:.6f}")

            if n_harm == 1 or r2_harm > 0.95:
                # Print this equation
                terms = []
                for c_val, name in zip(coeffs_harm, basis_names):
                    if abs(c_val) > 1e-6:
                        terms.append(f"{c_val:+.4f}·{name}")
                if r2_harm > 0.9:
                    print(f"      {' '.join(terms)}")

        # Best harmonic fit for final equation
        # Use up to 3 harmonics, pick simplest with R² > 0.95
        final_n_harm = 1
        for n_harm in range(1, 4):
            basis = [np.ones(len(angle))]
            basis_names = ["1"]
            for k in range(1, n_harm + 1):
                basis.append(np.cos(k * angle_rad))
                basis_names.append(f"cos({k}θ)")
                basis.append(np.sin(k * angle_rad))
                basis_names.append(f"sin({k}θ)")
            B = np.column_stack(basis)
            coeffs_harm, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
            y_pred_harm = B @ coeffs_harm
            ss_res = np.sum((y - y_pred_harm) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2_harm = 1 - ss_res / (ss_tot + 1e-12)
            if r2_harm > 0.90:
                final_n_harm = n_harm
                final_coeffs = coeffs_harm
                final_names = basis_names
                final_r2 = r2_harm
                break
            final_n_harm = n_harm
            final_coeffs = coeffs_harm
            final_names = basis_names
            final_r2 = r2_harm

        # Print final equation
        print(f"\n  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  DISCOVERED EQUATION                                    │")
        print(f"  │                                                         │")
        terms = []
        for c_val, name in zip(final_coeffs, final_names):
            if abs(c_val) > 1e-6:
                if name == "1":
                    terms.append(f"{c_val:.4f}")
                else:
                    terms.append(f"{c_val:+.4f}·{name}")
        eq = " ".join(terms)
        print(f"  │  Perm_X = {eq}")
        print(f"  │                                                         │")
        print(f"  │  where θ = Angle (degrees)                              │")
        print(f"  │  R² = {final_r2:.4f}                                          │")
        if final_n_harm == 1:
            # Express as amplitude-phase form
            c0 = final_coeffs[0]
            c1 = final_coeffs[1]  # cos θ
            s1 = final_coeffs[2]  # sin θ
            amp = np.sqrt(c1**2 + s1**2)
            phase = np.degrees(np.arctan2(s1, c1))
            print(f"  │                                                         │")
            print(f"  │  = {c0:.4f} + {amp:.4f}·cos(θ - {phase:.1f}°)                │")
            print(f"  │  Period: 360°, Phase: {phase:.1f}°                           │")
        elif final_n_harm == 2:
            c0 = final_coeffs[0]
            # 2nd harmonic: cos(2θ), sin(2θ)
            if len(final_coeffs) >= 5:
                c2 = final_coeffs[3]  # cos(2θ)
                s2 = final_coeffs[4]  # sin(2θ)
                amp2 = np.sqrt(c2**2 + s2**2)
                phase2 = np.degrees(np.arctan2(s2, c2)) / 2
                print(f"  │                                                         │")
                print(f"  │  Dominant: {amp2:.4f}·cos(2θ - {2*phase2:.1f}°)              │")
                print(f"  │  Period: 180°                                           │")
        print(f"  └─────────────────────────────────────────────────────────┘")

        results["equation"] = {
            "harmonic_coeffs": final_coeffs,
            "harmonic_names": final_names,
            "R2": final_r2,
            "n_harmonics": final_n_harm,
        }

    else:
        # n_latent > 1: multi-harmonic
        print(f"  Multi-dimensional latent (n={z_norm.shape[1]})")
        print(f"  Direct harmonic fit:")
        angle_rad = np.radians(angle)
        for n_harm in range(1, 5):
            basis = [np.ones(len(angle))]
            for k in range(1, n_harm + 1):
                basis.append(np.cos(k * angle_rad))
                basis.append(np.sin(k * angle_rad))
            B = np.column_stack(basis)
            c, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
            y_pred = B @ c
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            print(f"    {n_harm} harmonic(s): R² = {r2:.6f}")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(angle, X, y, results, output_dir):
    """4-panel figure: raw data, sweep, symmetry, equation fit."""
    os.makedirs(output_dir, exist_ok=True)
    sym_res = results["symmetry"]
    all_results = results["all_results"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Permeability vs Angle — Symmetry & Equation Discovery",
                 fontsize=14, fontweight="bold")

    # Panel 1: Raw data (Permeability vs Angle)
    ax = axes[0]
    ax.scatter(angle, y, c="#4C72B0", s=25, alpha=0.7, edgecolors="none")
    # Sort for line plot
    order = np.argsort(angle)
    ax.plot(angle[order], y[order], "r-", alpha=0.3, lw=1)
    ax.set_xlabel("Angle (degrees)", fontsize=11)
    ax.set_ylabel("Permeability_X", fontsize=11)
    ax.set_title("Raw Data", fontsize=12)

    # Panel 2: Normalization sweep
    ax = axes[1]
    methods = list(all_results.keys())
    sym_types = ["translational", "rotational", "scaling"]
    x_pos = np.arange(len(methods))
    width = 0.25
    for i, stype in enumerate(sym_types):
        vals = [all_results[m]["symmetry"]["losses"].get(stype, 1.0) for m in methods]
        bars = ax.bar(x_pos + i * width, vals, width, label=stype, alpha=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Validation MSE", fontsize=10)
    ax.set_title("Normalization Sweep", fontsize=12)
    ax.legend(fontsize=8)

    # Panel 3: Symmetry losses (best method)
    ax = axes[2]
    types = list(sym_res["losses"].keys())
    losses = [sym_res["losses"][t] for t in types]
    colors = ["#55A868" if t == sym_res["symmetry_type"] else "#DD8452" for t in types]
    bars = ax.bar(types, losses, color=colors, edgecolor="black", lw=1)
    ax.set_ylabel("Validation MSE", fontsize=10)
    ax.set_title(f"Winner: {sym_res['symmetry_type']} ({results['best_method']})", fontsize=12)
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{loss:.4f}", ha="center", va="bottom", fontsize=9)

    # Panel 4: Equation fit overlay
    ax = axes[3]
    ax.scatter(angle, y, c="#4C72B0", s=25, alpha=0.7, edgecolors="none", label="data")

    if "equation" in results and results["equation"]:
        eq = results["equation"]
        angle_fit = np.linspace(0, 360, 500)
        angle_fit_rad = np.radians(angle_fit)

        coeffs = eq["harmonic_coeffs"]
        n_harm = eq["n_harmonics"]
        basis = [np.ones(len(angle_fit))]
        for k in range(1, n_harm + 1):
            basis.append(np.cos(k * angle_fit_rad))
            basis.append(np.sin(k * angle_fit_rad))
        B = np.column_stack(basis)
        y_fit = B @ coeffs

        ax.plot(angle_fit, y_fit, "r-", lw=2.5, label=f"fit (R²={eq['R2']:.4f})")
        ax.legend(fontsize=9)

    ax.set_xlabel("Angle (degrees)", fontsize=11)
    ax.set_ylabel("Permeability_X", fontsize=11)
    ax.set_title("Discovered Equation", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "permeability_symmetry_discovery.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Discover symmetry in permeability vs angle"
    )
    parser.add_argument("--data", default="permeability.csv")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-epochs", type=int, default=600)
    parser.add_argument("--sym-epochs", type=int, default=1500)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output-dir", default="output_permeability_symmetry")
    args = parser.parse_args()

    angle, X, y = load_data(args)
    results = run_pipeline(angle, X, y, args)

    print("\n" + "=" * 60)
    print("Visualization")
    print("=" * 60)
    plot_results(angle, X, y, results, args.output_dir)

    print("\nDONE.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
