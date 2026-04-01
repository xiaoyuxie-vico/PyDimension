"""
Discover symmetry in porous media permeability data using Stage1.

Physics
-------
Permeability of a porous medium depends on microstructure:
    - Angle       : orientation angle (degrees)
    - Porosity    : void fraction (%)
    - Surface_A   : specific surface area

The angle variable is decomposed into (cos theta, sin theta) so the
rotational encoder's x^2 transform can detect angular periodicity:
    cos^2(theta) + sin^2(theta) = 1  (rotationally invariant)

The script sweeps all three normalization methods (standard, minmax, robust)
and all three symmetry types to find the best combination.

Usage
-----
    # With real data (place CSV/Excel in this directory):
    python discover_symmetry.py --data permeability_data.csv

    # With synthetic data:
    python discover_symmetry.py --synthetic
"""

import sys
import os
import argparse
import traceback
import multiprocessing

import numpy as np
import torch

# Add Stage1 modules to path
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
    print(f"Copy preprocessing/, intrinsic_coordinate/, symmetry_discovery/ from")
    print(f"projects/20260912_Stage1_Prokash/ into the same directory as this script.")
    sys.exit(1)

# Prevent multiprocessing crashes on Windows
import torch.multiprocessing as _tmp
_tmp.cpu_count = lambda: 0

# After angle decomposition: [cos(angle), sin(angle), Porosity, Surface_A]
VARIABLE_NAMES = ["cos(Angle)", "sin(Angle)", "Porosity", "Surface_A"]
VARIABLE_UNITS = ["", "", "%", ""]
RAW_VARIABLE_NAMES = ["Angle", "Porosity", "Surface_A"]


# ──────────────────────────────────────────────────────────────────────────────
# Angle decomposition
# ──────────────────────────────────────────────────────────────────────────────

def decompose_angle(X_raw: np.ndarray) -> np.ndarray:
    """
    Replace Angle (col 0) with cos(Angle) and sin(Angle).

    Input:  (n, 3) = [Angle, Porosity, Surface_A]
    Output: (n, 4) = [cos(Angle), sin(Angle), Porosity, Surface_A]
    """
    angle_rad = np.radians(X_raw[:, 0])
    return np.column_stack([
        np.cos(angle_rad),
        np.sin(angle_rad),
        X_raw[:, 1],  # Porosity
        X_raw[:, 2],  # Surface_A
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 300, seed: int = 42) -> dict:
    """Generate synthetic permeability data with rotational symmetry in angle."""
    rng = np.random.default_rng(seed)

    angle     = rng.uniform(0, 360, n_samples)
    porosity  = rng.uniform(60, 63, n_samples)
    surface_a = rng.uniform(5800, 6600, n_samples)

    X_raw = np.column_stack([angle, porosity, surface_a])

    # Permeability with angular periodicity (rotational symmetry):
    # Depends on cos^2(angle) + sin^2(angle) invariant combinations
    # plus porosity and surface area
    angle_rad = np.radians(angle)
    z = (0.05 * porosity
         + 0.0003 * surface_a
         + 0.3 * np.cos(2 * angle_rad)    # 180-degree periodicity
         + 0.1 * np.sin(2 * angle_rad))
    y = z + rng.normal(0, 0.03, n_samples)

    return {"X_raw": X_raw, "y": y}


def load_csv_data(csv_path: str) -> dict:
    """Load permeability data from CSV or Excel."""
    ext = os.path.splitext(csv_path)[1].lower()

    if ext in (".xls", ".xlsx"):
        try:
            import pandas as pd
            df = pd.read_excel(csv_path)
        except ImportError:
            print("  pandas/openpyxl needed for Excel. Install: pip install pandas openpyxl")
            sys.exit(1)
    else:
        import pandas as pd
        df = pd.read_csv(csv_path)

    print(f"  Columns found: {df.columns.tolist()}")

    # Find input columns
    angle_col = None
    porosity_col = None
    surface_col = None
    perm_col = None

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
        print("  Using positional column mapping (Index, Angle, Porosity, Surface_A, Permeability_X)")
        angle_col = df.columns[1]
        porosity_col = df.columns[2]
        surface_col = df.columns[3]
        perm_col = df.columns[4]

    print(f"  Inputs: {angle_col}, {porosity_col}, {surface_col}")
    print(f"  Output: {perm_col}")

    X_raw = df[[angle_col, porosity_col, surface_col]].values.astype(float)
    y = df[perm_col].values.astype(float)

    return {"X_raw": X_raw, "y": y}


def load_data(args):
    """Load or generate data, decompose angle."""
    if args.synthetic:
        print("Generating synthetic permeability data...")
        data = generate_synthetic_data(n_samples=args.n_samples, seed=args.seed)
    elif args.data and os.path.exists(args.data):
        print(f"Loading permeability data from {args.data}...")
        data = load_csv_data(args.data)
    else:
        print(f"'{args.data}' not found. Using synthetic data.")
        data = generate_synthetic_data(n_samples=args.n_samples, seed=args.seed)

    X_raw, y = data["X_raw"], data["y"]

    print(f"  Samples: {X_raw.shape[0]}")
    for i, name in enumerate(RAW_VARIABLE_NAMES):
        print(f"    {name:12s}: [{X_raw[:, i].min():.4f}, {X_raw[:, i].max():.4f}]")
    print(f"  Permeability_X: [{y.min():.4f}, {y.max():.4f}]")

    # Decompose angle into cos/sin
    X = decompose_angle(X_raw)
    print(f"\n  After angle decomposition: {X.shape[1]} features")
    print(f"    {VARIABLE_NAMES}")
    print()

    return X_raw, X, y


# ──────────────────────────────────────────────────────────────────────────────
# Normalization sweep
# ──────────────────────────────────────────────────────────────────────────────

def sweep_normalizations(X, y, args):
    """
    Try all normalization methods and symmetry types.
    Return summary table and best combination.
    """
    methods = ["standard", "minmax", "robust"]
    all_results = {}

    print("=" * 60)
    print("Normalization & Symmetry Sweep")
    print("=" * 60)
    sys.stdout.flush()

    for method in methods:
        print(f"\n--- Normalization: {method} ---")
        sys.stdout.flush()

        norm = normalize_data(X, y, method=method)
        X_norm, y_norm = norm["X_normalized"], norm["y_normalized"]
        print(f"  X range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")

        # Quick latent dimension (fewer epochs for sweep)
        res_latent = discover_latent_dimension(
            X_norm, y_norm, max_latent=min(3, X.shape[1]),
            n_epochs=max(200, args.latent_epochs // 3),
            n_restarts=2, seed=args.seed,
        )
        n_latent = res_latent["optimal_n_latent"]
        print(f"  Latent dim: {n_latent}")

        # Identify symmetry
        res_sym = identify_symmetry(
            X_norm, y_norm, n_latent=n_latent,
            decoder=res_latent["best_decoder"],
            n_epochs=max(500, args.sym_epochs // 2),
            n_restarts=2, seed=args.seed,
        )

        for stype, loss in sorted(res_sym["losses"].items(), key=lambda kv: kv[1]):
            marker = " <--" if stype == res_sym["symmetry_type"] else ""
            print(f"    {stype:15s}: {loss:.6f}{marker}")

        all_results[method] = {
            "normalization": norm,
            "latent": res_latent,
            "symmetry": res_sym,
            "best_loss": min(res_sym["losses"].values()),
        }

    # Print summary table
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(f"  {'Method':<12} {'Winner':<16} {'Best MSE':<12} {'Gap':<8}")
    print(f"  {'-'*12} {'-'*16} {'-'*12} {'-'*8}")

    best_method = None
    best_overall_loss = np.inf

    for method in methods:
        r = all_results[method]
        sym = r["symmetry"]
        losses_sorted = sorted(sym["losses"].values())
        gap = losses_sorted[1] / (losses_sorted[0] + 1e-12)
        print(f"  {method:<12} {sym['symmetry_type']:<16} {losses_sorted[0]:<12.6f} {gap:<8.1f}x")

        if losses_sorted[0] < best_overall_loss:
            best_overall_loss = losses_sorted[0]
            best_method = method

    print(f"\n  Best: {best_method} normalization")
    print()

    return all_results, best_method


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline on best normalization
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(X, y, best_method, args):
    """Run full pipeline with the best normalization method."""
    print("=" * 60)
    print(f"Full Pipeline (normalization: {best_method})")
    print("=" * 60)
    sys.stdout.flush()

    results = {}

    # Normalize
    norm = normalize_data(X, y, method=best_method)
    X_norm, y_norm = norm["X_normalized"], norm["y_normalized"]
    results["normalization"] = norm
    results["norm_method"] = best_method
    print(f"  X range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")

    # Latent dimension (full epochs)
    print("\n  Discovering latent dimension...")
    sys.stdout.flush()
    res_latent = discover_latent_dimension(
        X_norm, y_norm, max_latent=min(3, X.shape[1]),
        n_epochs=args.latent_epochs, n_restarts=args.n_restarts, seed=args.seed,
    )
    results["latent"] = res_latent
    n_latent = res_latent["optimal_n_latent"]
    print(f"  Optimal latent dimension: {n_latent}")
    for k, m in res_latent["metrics"].items():
        print(f"    k={k}: R2={m['R2']:.4f}")

    # Symmetry identification (full epochs)
    print("\n  Identifying symmetry type...")
    sys.stdout.flush()
    res_sym = identify_symmetry(
        X_norm, y_norm, n_latent=n_latent, decoder=res_latent["best_decoder"],
        n_epochs=args.sym_epochs, n_restarts=args.n_restarts, seed=args.seed,
    )
    results["symmetry"] = res_sym
    print(f"  Detected symmetry: {res_sym['symmetry_type']}")
    for stype, loss in sorted(res_sym["losses"].items(), key=lambda kv: kv[1]):
        marker = " <--" if stype == res_sym["symmetry_type"] else ""
        print(f"    {stype:15s}: {loss:.6f}{marker}")
    sorted_losses = sorted(res_sym["losses"].values())
    if len(sorted_losses) >= 2 and sorted_losses[0] > 0:
        print(f"  Loss gap: {sorted_losses[1] / sorted_losses[0]:.1f}x")

    # Extract generators
    winner_type = res_sym["symmetry_type"]
    winner_encoder = res_sym["encoders"][winner_type]
    generators = extract_generators(winner_type, winner_encoder)
    results["generators"] = generators
    results["winner_type"] = winner_type
    results["winner_encoder"] = winner_encoder

    W = winner_encoder.weight_matrix
    results["W"] = W
    print(f"\n  Generators: {len(generators)}")
    print(f"  Encoder weights W:")
    if W.shape[0] == 1:
        for j, name in enumerate(VARIABLE_NAMES):
            print(f"    {name:12s}: {W[0, j]:+.4f}")
    else:
        for row_i in range(W.shape[0]):
            parts = [f"{VARIABLE_NAMES[j]}:{W[row_i, j]:+.4f}" for j in range(W.shape[1])]
            print(f"    z{row_i+1}: [{', '.join(parts)}]")

    # Physical interpretation
    print(f"\n  Physical interpretation of generators:")
    if winner_type == "translational" and generators:
        print(f"  Each generator: x -> x + eps*g preserves permeability.\n")
        for i, g in enumerate(generators):
            parts = [f"{VARIABLE_NAMES[j]}: {g[j]:+.3f}"
                     for j in range(len(g)) if abs(g[j]) > 0.05]
            print(f"  Generator {i+1}: [{', '.join(parts)}]")
            _interpret_translational(g)
            print()
    elif winner_type == "scaling" and generators:
        print(f"  Each generator: x -> x*exp(eps*s) preserves permeability.\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = [f"{VARIABLE_NAMES[j]} x exp({g[j]:+.3f}*eps)"
                         for j in range(len(g)) if abs(g[j]) > 0.05]
                print(f"  Generator {i+1}: [{', '.join(parts)}]")
                print()
    elif winner_type == "rotational" and generators:
        print(f"  Each generator: rotation in a 2D plane preserving permeability.\n")
        for i, g in enumerate(generators):
            if g.ndim == 2:
                print(f"  Generator {i+1}:")
                for a in range(g.shape[0]):
                    for b in range(a + 1, g.shape[1]):
                        if abs(g[a, b]) > 0.01:
                            print(f"    Rotation in ({VARIABLE_NAMES[a]}, {VARIABLE_NAMES[b]}) plane")
                            if a < 2 and b < 2:
                                print(f"    -> Angular symmetry: permeability repeats under rotation")
                print()
    print()

    return results


def _interpret_translational(g):
    """Interpret a translational generator."""
    pos = [(VARIABLE_NAMES[j], g[j]) for j in range(len(g)) if g[j] > 0.05]
    neg = [(VARIABLE_NAMES[j], g[j]) for j in range(len(g)) if g[j] < -0.05]
    if pos and neg:
        inc = ", ".join(n for n, v in sorted(pos, key=lambda x: -abs(x[1]))[:2])
        dec = ", ".join(n for n, v in sorted(neg, key=lambda x: -abs(x[1]))[:2])
        print(f"    -> Increase {inc} while decreasing {dec}")
        print(f"       to maintain the same Permeability_X")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (4 panels: sweep + 3 best results)
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(X_raw, X, y, results, sweep_results, output_dir):
    """Create a 4-panel figure: sweep summary + latent var + losses + orbits."""
    os.makedirs(output_dir, exist_ok=True)
    generators = results["generators"]
    winner_type = results["winner_type"]
    norm = results["normalization"]
    sym_res = results["symmetry"]
    W = results["W"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.suptitle("Porous Media Permeability — Symmetry Discovery (all normalizations)",
                 fontsize=14, fontweight="bold")

    # --- Panel 0: Normalization sweep summary ---
    ax = axes[0]
    methods = list(sweep_results.keys())
    sym_types_list = ["translational", "rotational", "scaling"]
    x_pos = np.arange(len(methods))
    width = 0.25
    for i, stype in enumerate(sym_types_list):
        vals = [sweep_results[m]["symmetry"]["losses"].get(stype, 1.0) for m in methods]
        bars = ax.bar(x_pos + i * width, vals, width, label=stype, alpha=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x_pos + width)
    xticklabels = methods
    ax.set_xticks(x_pos + width)
    ax_xticks = ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Validation MSE", fontsize=10)
    ax.set_title("Normalization Sweep", fontsize=12)
    ax.legend(fontsize=8)

    # --- Panel 1: Latent variable vs permeability ---
    ax = axes[1]
    X_norm = norm["X_normalized"]
    z = X_norm @ W.T
    if z.shape[1] == 1:
        z = z.ravel()
        ax.scatter(z, y, c="#4C72B0", s=20, alpha=0.6, edgecolors="none")
        coeffs = np.polyfit(z, y, 2)
        z_fit = np.linspace(z.min(), z.max(), 200)
        ax.plot(z_fit, np.polyval(coeffs, z_fit), "r-", lw=2, label="quadratic fit")
        ss_res = np.sum((y - np.polyval(coeffs, z)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        ax.set_xlabel("z = W·x (learned latent)", fontsize=10)
        ax.set_ylabel("Permeability_X", fontsize=10)
        ax.set_title(f"Latent vs Permeability (R²={r2:.3f})", fontsize=12)
        ax.legend(fontsize=8)
    else:
        sc = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="viridis", s=20, alpha=0.6)
        fig.colorbar(sc, ax=ax, label="Permeability_X")
        ax.set_xlabel("z1", fontsize=10)
        ax.set_ylabel("z2", fontsize=10)
        ax.set_title("Latent Variables", fontsize=12)

    # --- Panel 2: Symmetry type losses (best normalization) ---
    ax = axes[2]
    types = list(sym_res["losses"].keys())
    losses = [sym_res["losses"][t] for t in types]
    colors = ["#55A868" if t == sym_res["symmetry_type"] else "#DD8452" for t in types]
    bars = ax.bar(types, losses, color=colors, edgecolor="black", lw=1)
    ax.set_ylabel("Validation MSE", fontsize=10)
    ax.set_title(f"Winner: {sym_res['symmetry_type']} ({results['norm_method']})", fontsize=12)
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{loss:.4f}", ha="center", va="bottom", fontsize=9)

    # --- Panel 3: Generator orbits ---
    ax = axes[3]
    if generators:
        g = generators[0]

        if winner_type in ("translational", "scaling") and g.ndim == 1:
            abs_g = np.abs(g)
            top2 = np.argsort(abs_g)[-2:][::-1]
            d0, d1 = top2[0], top2[1]

            sc = ax.scatter(X[:, d0], X[:, d1], c=y, cmap="viridis", s=20, alpha=0.6,
                            edgecolors="none")
            fig.colorbar(sc, ax=ax, label="Perm_X", fraction=0.046, pad=0.04)

            orbit_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
            rng = np.random.default_rng(42)
            start_indices = rng.choice(len(X), min(3, len(X)), replace=False)
            for k, idx in enumerate(start_indices):
                x_start = norm["X_normalized"][idx]
                fwd = generator_orbit(x_start, g, 100, 0.03, winner_type)
                back = generator_orbit(x_start, g, 100, -0.03, winner_type)
                orb = np.vstack([back[::-1], fwd[1:]])
                orb_orig = norm["scaler_X"].inverse_transform(orb)
                ax.plot(orb_orig[:, d0], orb_orig[:, d1],
                        color=orbit_colors[k % len(orbit_colors)], lw=2, alpha=0.8,
                        label=f"orbit {k+1}")

            ax.set_xlabel(f"{VARIABLE_NAMES[d0]}", fontsize=10)
            ax.set_ylabel(f"{VARIABLE_NAMES[d1]}", fontsize=10)
            ax.set_title("Generator Orbits", fontsize=12)
            ax.legend(fontsize=8, loc="best")

        elif winner_type == "rotational" and g.ndim == 2:
            # Find the rotation plane
            pairs = []
            for a in range(g.shape[0]):
                for b in range(a + 1, g.shape[1]):
                    if abs(g[a, b]) > 0.01:
                        pairs.append((a, b))
            d0, d1 = pairs[0] if pairs else (0, 1)

            sc = ax.scatter(X[:, d0], X[:, d1], c=y, cmap="viridis", s=20, alpha=0.6,
                            edgecolors="none")
            fig.colorbar(sc, ax=ax, label="Perm_X", fraction=0.046, pad=0.04)

            orbit_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
            rng = np.random.default_rng(42)
            start_indices = rng.choice(len(X), min(3, len(X)), replace=False)
            for k, idx in enumerate(start_indices):
                x_start = norm["X_normalized"][idx]
                orb = generator_orbit(x_start, g, 200, 0.03, winner_type)
                orb_orig = norm["scaler_X"].inverse_transform(orb)
                ax.plot(orb_orig[:, d0], orb_orig[:, d1],
                        color=orbit_colors[k % len(orbit_colors)], lw=2, alpha=0.8,
                        label=f"orbit {k+1}")

            ax.set_xlabel(f"{VARIABLE_NAMES[d0]}", fontsize=10)
            ax.set_ylabel(f"{VARIABLE_NAMES[d1]}", fontsize=10)
            ax.set_title("Generator Orbits (rotation)", fontsize=12)
            ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "No generators found",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Generator Orbits")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, "permeability_symmetry_discovery.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Discover symmetry in porous media permeability data"
    )
    parser.add_argument("--data", default="permeability_data.csv")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-epochs", type=int, default=600)
    parser.add_argument("--sym-epochs", type=int, default=1500)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output-dir", default="output_permeability_symmetry")
    args = parser.parse_args()

    X_raw, X, y = load_data(args)

    # Sweep all normalizations
    sweep_results, best_method = sweep_normalizations(X, y, args)

    # Full pipeline with best normalization
    print()
    results = run_full_pipeline(X, y, best_method, args)

    print("=" * 60)
    print("Creating visualizations")
    print("=" * 60)
    plot_results(X_raw, X, y, results, sweep_results, args.output_dir)

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    sym_type = results["symmetry"]["symmetry_type"]
    print(f"  Best normalization: {best_method}")
    print(f"  Symmetry: {sym_type}")
    print(f"  Generators: {len(results['generators'])}")
    if sym_type == "rotational":
        print(f"  Angular periodicity detected: permeability repeats")
        print(f"  under rotation of the orientation angle.")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
