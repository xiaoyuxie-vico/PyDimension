"""
Discover symmetry in porous media permeability using Stage1.

Physics
-------
Different circular porous media geometries are characterised by:
    - Angle      : orientation (degrees)
    - Porosity   : void fraction (%)
    - Surface_A  : specific surface area

For each geometry, Permeability_X repeats every 180 degrees in angle.
We encode this known periodicity by decomposing angle into:
    cos(2θ) and sin(2θ)    (period = 180°)

Input features to the pipeline (4 total):
    [cos(2θ), sin(2θ), Porosity, Surface_A]

The pipeline discovers:
    1. Symmetry type (translational / rotational / scaling)
    2. Lie-algebra generators
    3. Closed-form equation: Perm_X = f(Angle, Porosity, Surface_A)

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

# After decomposition: [cos(2θ), sin(2θ), Porosity, Surface_A]
FEATURE_NAMES = ["cos(2θ)", "sin(2θ)", "Porosity", "Surface_A"]
RAW_NAMES = ["Angle", "Porosity", "Surface_A"]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n_samples=500, seed=42):
    """Synthetic data: multiple geometries, permeability repeats every 180°."""
    rng = np.random.default_rng(seed)
    angle     = rng.uniform(0, 350, n_samples)
    porosity  = rng.uniform(60, 63, n_samples)
    surface_a = rng.uniform(5800, 6600, n_samples)

    angle_rad = np.radians(angle)
    # Permeability depends on all 3 variables with 180° periodicity in angle
    y = (3.2
         + 0.05 * (porosity - 61)
         + 0.0003 * (surface_a - 6200)
         + 0.30 * np.cos(2 * angle_rad)
         + 0.10 * np.sin(2 * angle_rad)
         + rng.normal(0, 0.04, n_samples))

    return angle, porosity, surface_a, y


def load_csv_data(csv_path):
    """Load Angle, Porosity, Surface_A, Permeability_X from CSV/Excel."""
    ext = os.path.splitext(csv_path)[1].lower()
    if ext in (".xls", ".xlsx"):
        import pandas as pd
        df = pd.read_excel(csv_path)
    else:
        import pandas as pd
        df = pd.read_csv(csv_path)

    print(f"  Columns: {df.columns.tolist()}")

    angle_col = porosity_col = surface_col = perm_col = None
    for col in df.columns:
        cl = col.strip().lower()
        if "angle" in cl:
            angle_col = col
        elif "porosity" in cl:
            porosity_col = col
        elif "surface" in cl:
            surface_col = col
        elif "permeability" in cl and "x" in cl:
            perm_col = col

    if not all([angle_col, porosity_col, surface_col, perm_col]):
        # Positional fallback: Index, Angle, Porosity, Surface_A, Permeability_X
        angle_col = df.columns[1]
        porosity_col = df.columns[2]
        surface_col = df.columns[3]
        perm_col = df.columns[4]

    print(f"  Using: {angle_col}, {porosity_col}, {surface_col} -> {perm_col}")

    angle = df[angle_col].values.astype(float)
    porosity = df[porosity_col].values.astype(float)
    surface_a = df[surface_col].values.astype(float)
    y = df[perm_col].values.astype(float)

    return angle, porosity, surface_a, y


def load_data(args):
    """Load data, decompose angle into cos(2θ), sin(2θ)."""
    if args.synthetic:
        print("Generating synthetic data...")
        angle, porosity, surface_a, y = generate_synthetic_data(args.n_samples, args.seed)
    else:
        data_path = args.data
        if not os.path.exists(data_path):
            data_path = os.path.join(_here, args.data)
        if os.path.exists(data_path):
            print(f"Loading from {data_path}...")
            angle, porosity, surface_a, y = load_csv_data(data_path)
        else:
            print(f"'{args.data}' not found. Using synthetic.")
            angle, porosity, surface_a, y = generate_synthetic_data(args.n_samples, args.seed)

    # Decompose angle with known 180° periodicity
    angle_rad = np.radians(angle)
    X = np.column_stack([
        np.cos(2 * angle_rad),
        np.sin(2 * angle_rad),
        porosity,
        surface_a,
    ])

    print(f"  Samples: {len(angle)}")
    print(f"  Angle:       [{angle.min():.1f}, {angle.max():.1f}] deg")
    print(f"  Porosity:    [{porosity.min():.4f}, {porosity.max():.4f}]")
    print(f"  Surface_A:   [{surface_a.min():.1f}, {surface_a.max():.1f}]")
    print(f"  Perm_X:      [{y.min():.4f}, {y.max():.4f}]")
    print(f"\n  Features: {FEATURE_NAMES}")
    print()

    raw = {"angle": angle, "porosity": porosity, "surface_a": surface_a}
    return raw, X, y


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(raw, X, y, args):
    """Sweep normalizations, discover symmetry, extract equation."""
    methods = ["standard", "minmax", "robust"]
    all_results = {}

    print("=" * 60)
    print("Normalization & Symmetry Sweep")
    print("=" * 60)

    for method in methods:
        print(f"\n--- {method} ---")
        sys.stdout.flush()
        norm = normalize_data(X, y, method=method)
        X_norm, y_norm = norm["X_normalized"], norm["y_normalized"]

        res_latent = discover_latent_dimension(
            X_norm, y_norm, max_latent=3,
            n_epochs=args.latent_epochs, n_restarts=args.n_restarts, seed=args.seed,
        )
        n_latent = res_latent["optimal_n_latent"]

        res_sym = identify_symmetry(
            X_norm, y_norm, n_latent=n_latent, decoder=res_latent["best_decoder"],
            n_epochs=args.sym_epochs, n_restarts=args.n_restarts, seed=args.seed,
        )

        print(f"  Latent dim: {n_latent}")
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

    # Best results
    norm = all_results[best_method]["normalization"]
    res_latent = all_results[best_method]["latent"]
    res_sym = all_results[best_method]["symmetry"]
    winner_type = res_sym["symmetry_type"]
    winner_encoder = res_sym["encoders"][winner_type]
    W = winner_encoder.weight_matrix
    generators = extract_generators(winner_type, winner_encoder)

    print(f"\n  Symmetry: {winner_type}")
    print(f"  Latent dim: {res_latent['optimal_n_latent']}")
    print(f"  Generators: {len(generators)}")
    print(f"  Encoder weights W:")
    for row_i in range(W.shape[0]):
        parts = [f"{FEATURE_NAMES[j]}:{W[row_i, j]:+.4f}" for j in range(W.shape[1])]
        print(f"    z{row_i+1}: [{', '.join(parts)}]")

    # Generator interpretation
    if generators:
        print(f"\n  Generator interpretation:")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = [f"{FEATURE_NAMES[j]}: {g[j]:+.3f}"
                         for j in range(len(g)) if abs(g[j]) > 0.05]
                print(f"    G{i+1}: [{', '.join(parts)}]")

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

    # Equation discovery
    print("\n" + "=" * 60)
    print("Equation Discovery")
    print("=" * 60)
    discover_equation(raw, X, y, results)

    return results


def discover_equation(raw, X, y, results):
    """
    Extract closed-form equation: Perm_X = f(Angle, Porosity, Surface_A).

    Uses direct harmonic + linear regression, bypassing the neural decoder,
    to get a fully interpretable equation.
    """
    angle = raw["angle"]
    porosity = raw["porosity"]
    surface_a = raw["surface_a"]
    angle_rad = np.radians(angle)

    # Build basis: constant + cos(2θ) + sin(2θ) + Porosity + Surface_A
    # Then try adding higher harmonics and cross terms
    print(f"\n  Fitting Perm_X = f(Angle, Porosity, Surface_A):")

    models = {}

    # Model 1: Linear in all features (no harmonics)
    B1 = np.column_stack([np.ones(len(y)), porosity, surface_a])
    names1 = ["1", "Porosity", "Surface_A"]
    c1, _, _, _ = np.linalg.lstsq(B1, y, rcond=None)
    r2_1 = _r2(y, B1 @ c1)
    models["linear (no angle)"] = (c1, names1, r2_1)
    print(f"    Linear (no angle):        R² = {r2_1:.6f}")

    # Model 2: + cos(2θ), sin(2θ)
    B2 = np.column_stack([np.ones(len(y)), np.cos(2 * angle_rad), np.sin(2 * angle_rad),
                          porosity, surface_a])
    names2 = ["1", "cos(2θ)", "sin(2θ)", "Porosity", "Surface_A"]
    c2, _, _, _ = np.linalg.lstsq(B2, y, rcond=None)
    r2_2 = _r2(y, B2 @ c2)
    models["+ cos(2θ), sin(2θ)"] = (c2, names2, r2_2)
    print(f"    + cos(2θ), sin(2θ):       R² = {r2_2:.6f}")

    # Model 3: + cos(4θ), sin(4θ) (2nd harmonic of 180°)
    B3 = np.column_stack([B2, np.cos(4 * angle_rad), np.sin(4 * angle_rad)])
    names3 = names2 + ["cos(4θ)", "sin(4θ)"]
    c3, _, _, _ = np.linalg.lstsq(B3, y, rcond=None)
    r2_3 = _r2(y, B3 @ c3)
    models["+ cos(4θ), sin(4θ)"] = (c3, names3, r2_3)
    print(f"    + cos(4θ), sin(4θ):       R² = {r2_3:.6f}")

    # Model 4: + cos(6θ), sin(6θ) (3rd harmonic)
    B4 = np.column_stack([B3, np.cos(6 * angle_rad), np.sin(6 * angle_rad)])
    names4 = names3 + ["cos(6θ)", "sin(6θ)"]
    c4, _, _, _ = np.linalg.lstsq(B4, y, rcond=None)
    r2_4 = _r2(y, B4 @ c4)
    models["+ cos(6θ), sin(6θ)"] = (c4, names4, r2_4)
    print(f"    + cos(6θ), sin(6θ):       R² = {r2_4:.6f}")

    # Pick simplest model with R² > 0.90, or best overall
    best_name = None
    best_r2 = -np.inf
    for name, (c, names, r2) in models.items():
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
    # Prefer simpler if R² > 0.90
    for name, (c, names, r2) in models.items():
        if r2 > 0.90:
            best_name = name
            break

    best_c, best_names, best_r2 = models[best_name]

    # Print equation
    print(f"\n  Best model: {best_name} (R² = {best_r2:.6f})")

    # Amplitude-phase for angular terms
    cos2_coeff = sin2_coeff = 0.0
    for c_val, name in zip(best_c, best_names):
        if name == "cos(2θ)":
            cos2_coeff = c_val
        elif name == "sin(2θ)":
            sin2_coeff = c_val
    if abs(cos2_coeff) > 1e-8 or abs(sin2_coeff) > 1e-8:
        amp = np.sqrt(cos2_coeff**2 + sin2_coeff**2)
        phase = np.degrees(np.arctan2(sin2_coeff, cos2_coeff))
        print(f"\n  Angular term: {cos2_coeff:.4f}·cos(2θ) + {sin2_coeff:.4f}·sin(2θ)")
        print(f"              = {amp:.4f}·cos(2θ - {phase:.1f}°)")
        print(f"              Period = 180°")

    # Full equation
    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  DISCOVERED EQUATION                                         │")
    print(f"  │                                                              │")
    terms = []
    for c_val, name in zip(best_c, best_names):
        if abs(c_val) < 1e-8:
            continue
        if name == "1":
            terms.append(f"{c_val:.4f}")
        else:
            terms.append(f"{c_val:+.4f}·{name}")
    eq = " ".join(terms)
    print(f"  │  Perm_X = {eq}")
    print(f"  │                                                              │")
    print(f"  │  where θ = Angle (degrees)                                   │")
    print(f"  │  R² = {best_r2:.4f}                                                │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    results["equation"] = {
        "coeffs": best_c,
        "names": best_names,
        "R2": best_r2,
        "model_name": best_name,
    }


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(raw, X, y, results, output_dir):
    """4-panel figure."""
    os.makedirs(output_dir, exist_ok=True)
    angle = raw["angle"]
    sym_res = results["symmetry"]
    all_results = results["all_results"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Porous Media Permeability — Symmetry & Equation Discovery",
                 fontsize=14, fontweight="bold")

    # Panel 1: Raw data (Perm vs Angle, colored by Porosity)
    ax = axes[0]
    sc = ax.scatter(angle, y, c=raw["porosity"], cmap="viridis", s=25, alpha=0.7,
                    edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Porosity", fraction=0.046, pad=0.04)
    ax.set_xlabel("Angle (degrees)", fontsize=11)
    ax.set_ylabel("Permeability_X", fontsize=11)
    ax.set_title("Raw Data (color=Porosity)", fontsize=12)

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

    # Panel 3: Symmetry losses (best)
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
        coeffs = eq["coeffs"]
        names = eq["names"]

        # Evaluate fit on fine angle grid at mean porosity/surface_a
        angle_fit = np.linspace(0, 360, 500)
        angle_fit_rad = np.radians(angle_fit)
        por_mean = raw["porosity"].mean()
        sa_mean = raw["surface_a"].mean()

        # Build basis for fit curve
        y_fit = np.zeros(len(angle_fit))
        for c_val, name in zip(coeffs, names):
            if name == "1":
                y_fit += c_val
            elif name == "cos(2θ)":
                y_fit += c_val * np.cos(2 * angle_fit_rad)
            elif name == "sin(2θ)":
                y_fit += c_val * np.sin(2 * angle_fit_rad)
            elif name == "cos(4θ)":
                y_fit += c_val * np.cos(4 * angle_fit_rad)
            elif name == "sin(4θ)":
                y_fit += c_val * np.sin(4 * angle_fit_rad)
            elif name == "cos(6θ)":
                y_fit += c_val * np.cos(6 * angle_fit_rad)
            elif name == "sin(6θ)":
                y_fit += c_val * np.sin(6 * angle_fit_rad)
            elif name == "Porosity":
                y_fit += c_val * por_mean
            elif name == "Surface_A":
                y_fit += c_val * sa_mean

        ax.plot(angle_fit, y_fit, "r-", lw=2.5,
                label=f"fit @ mean (R²={eq['R2']:.4f})")
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
        description="Discover symmetry in porous media permeability"
    )
    parser.add_argument("--data", default="permeability.csv")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-epochs", type=int, default=600)
    parser.add_argument("--sym-epochs", type=int, default=1500)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output-dir", default="output_permeability_symmetry")
    args = parser.parse_args()

    raw, X, y = load_data(args)
    results = run_pipeline(raw, X, y, args)

    print("\n" + "=" * 60)
    print("Visualization")
    print("=" * 60)
    plot_results(raw, X, y, results, args.output_dir)

    print("\nDONE.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
