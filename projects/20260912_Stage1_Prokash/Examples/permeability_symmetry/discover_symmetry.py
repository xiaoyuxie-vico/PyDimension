"""
Discover symmetry in porous media permeability data using Stage1.

Physics
-------
Permeability of a porous medium depends on microstructure:
    - Angle       : orientation angle (degrees)
    - Porosity    : void fraction (%)
    - Surface_A   : specific surface area

The relationship Permeability_X = f(Angle, Porosity, Surface_A) may exhibit
translational, rotational, or scaling symmetry depending on how these
variables combine.

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

VARIABLE_NAMES = ["Angle", "Porosity", "Surface_A"]
VARIABLE_UNITS = ["deg", "%", ""]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 300, seed: int = 42) -> dict:
    """Generate synthetic permeability data with realistic ranges."""
    rng = np.random.default_rng(seed)

    angle     = rng.uniform(0, 360, n_samples)
    porosity  = rng.uniform(60, 63, n_samples)
    surface_a = rng.uniform(5800, 6600, n_samples)

    X = np.column_stack([angle, porosity, surface_a])

    # Synthetic relationship: permeability depends on a linear combination
    # with some angle periodicity
    z = 0.05 * porosity + 0.0003 * surface_a + 0.002 * np.sin(np.radians(angle))
    y = z + rng.normal(0, 0.05, n_samples)

    return {"X": X, "y": y}


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
        # Fallback: assume columns by position (skip Index)
        print("  Using positional column mapping (Index, Angle, Porosity, Surface_A, Permeability_X)")
        angle_col = df.columns[1]
        porosity_col = df.columns[2]
        surface_col = df.columns[3]
        perm_col = df.columns[4]

    print(f"  Inputs: {angle_col}, {porosity_col}, {surface_col}")
    print(f"  Output: {perm_col}")

    X = df[[angle_col, porosity_col, surface_col]].values.astype(float)
    y = df[perm_col].values.astype(float)

    return {"X": X, "y": y}


def load_data(args):
    """Load or generate data."""
    if args.synthetic:
        print("Generating synthetic permeability data...")
        data = generate_synthetic_data(n_samples=args.n_samples, seed=args.seed)
    elif args.data and os.path.exists(args.data):
        print(f"Loading permeability data from {args.data}...")
        data = load_csv_data(args.data)
    else:
        print(f"'{args.data}' not found. Using synthetic data.")
        data = generate_synthetic_data(n_samples=args.n_samples, seed=args.seed)

    X, y = data["X"], data["y"]
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    for i, (name, unit) in enumerate(zip(VARIABLE_NAMES, VARIABLE_UNITS)):
        print(f"    {name:12s}: [{X[:, i].min():.4f}, {X[:, i].max():.4f}] {unit}")
    print(f"  Permeability_X: [{y.min():.4f}, {y.max():.4f}]")
    print()
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(X, y, args):
    """Run Stage1 symmetry discovery pipeline."""
    results = {}

    # --- Normalize ---
    print("=" * 60)
    print("Step 1: Normalizing data")
    print("=" * 60)
    sys.stdout.flush()
    norm = normalize_data(X, y, method="standard")
    X_norm, y_norm = norm["X_normalized"], norm["y_normalized"]
    results["normalization"] = norm
    print(f"  X range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
    print()

    # --- Discover latent dimension ---
    print("=" * 60)
    print("Step 2: Discovering intrinsic latent dimension")
    print("=" * 60)
    sys.stdout.flush()
    res_latent = discover_latent_dimension(
        X_norm, y_norm, max_latent=min(3, X.shape[1]),
        n_epochs=args.latent_epochs, n_restarts=args.n_restarts, seed=args.seed,
    )
    results["latent"] = res_latent
    n_latent = res_latent["optimal_n_latent"]
    print(f"\n  Optimal latent dimension: {n_latent}")
    for k, m in res_latent["metrics"].items():
        print(f"    k={k}: R2={m['R2']:.4f}")
    print()

    # --- Identify symmetry type ---
    print("=" * 60)
    print("Step 3: Identifying symmetry type")
    print("=" * 60)
    sys.stdout.flush()
    res_sym = identify_symmetry(
        X_norm, y_norm, n_latent=n_latent, decoder=res_latent["best_decoder"],
        n_epochs=args.sym_epochs, n_restarts=args.n_restarts, seed=args.seed,
    )
    results["symmetry"] = res_sym
    print(f"\n  Detected symmetry: {res_sym['symmetry_type']}")
    for stype, loss in sorted(res_sym["losses"].items(), key=lambda kv: kv[1]):
        marker = " <--" if stype == res_sym["symmetry_type"] else ""
        print(f"    {stype:15s}: {loss:.6f}{marker}")
    sorted_losses = sorted(res_sym["losses"].values())
    if len(sorted_losses) >= 2 and sorted_losses[0] > 0:
        print(f"  Loss gap: {sorted_losses[1] / sorted_losses[0]:.1f}x")
    print()

    # --- Extract generators ---
    print("=" * 60)
    print("Step 4: Extracting Lie-algebra generators")
    print("=" * 60)
    winner_type = res_sym["symmetry_type"]
    winner_encoder = res_sym["encoders"][winner_type]
    generators = extract_generators(winner_type, winner_encoder)
    results["generators"] = generators
    results["winner_type"] = winner_type
    results["winner_encoder"] = winner_encoder

    # Encoder weights
    W = winner_encoder.weight_matrix
    results["W"] = W
    print(f"  Symmetry type: {winner_type}")
    print(f"  Generators: {len(generators)}")
    print(f"\n  Learned encoder weights W:")
    if W.shape[0] == 1:
        for j, name in enumerate(VARIABLE_NAMES):
            print(f"    {name:12s}: {W[0, j]:+.4f}")
    else:
        for row_i in range(W.shape[0]):
            parts = [f"{VARIABLE_NAMES[j]}:{W[row_i, j]:+.4f}" for j in range(W.shape[1])]
            print(f"    z{row_i+1}: [{', '.join(parts)}]")
    print()

    # --- Interpret generators ---
    print("=" * 60)
    print("Step 5: Physical interpretation of generators")
    print("=" * 60)
    if winner_type == "translational" and generators:
        print(f"  Each generator is a direction g where x → x + ε·g preserves permeability.\n")
        for i, g in enumerate(generators):
            parts = []
            for j, name in enumerate(VARIABLE_NAMES):
                if abs(g[j]) > 0.05:
                    parts.append(f"{name}: {g[j]:+.3f}")
            print(f"  Generator {i+1}: [{', '.join(parts)}]")
            _interpret_translational(g, i + 1)
            print()
    elif winner_type == "scaling" and generators:
        print(f"  Each generator is a log-space direction: x → x·exp(ε·s).\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = []
                for j, name in enumerate(VARIABLE_NAMES):
                    if abs(g[j]) > 0.05:
                        parts.append(f"{name} x exp({g[j]:+.3f}*eps)")
                print(f"  Generator {i+1}: [{', '.join(parts)}]")
                _interpret_scaling(g, i + 1)
                print()
    elif winner_type == "rotational" and generators:
        print(f"  Each generator is an antisymmetric matrix (rotation plane).\n")
        for i, g in enumerate(generators):
            print(f"  Generator {i+1}:")
            # Find the non-zero pair
            if g.ndim == 2:
                for a in range(g.shape[0]):
                    for b in range(a + 1, g.shape[1]):
                        if abs(g[a, b]) > 0.01:
                            print(f"    Rotation in ({VARIABLE_NAMES[a]}, {VARIABLE_NAMES[b]}) plane")
            print()
    print()

    return results


def _interpret_translational(g, idx):
    """Interpret a translational generator."""
    abs_g = np.abs(g)
    top = np.argsort(abs_g)[::-1]
    pos = [(VARIABLE_NAMES[j], g[j]) for j in top if g[j] > 0.05]
    neg = [(VARIABLE_NAMES[j], g[j]) for j in top if g[j] < -0.05]
    if pos and neg:
        inc = ", ".join(n for n, v in pos[:2])
        dec = ", ".join(n for n, v in neg[:2])
        print(f"    → Increase {inc} while decreasing {dec}")
        print(f"      to maintain the same Permeability_X")


def _interpret_scaling(g, idx):
    """Interpret a scaling generator."""
    abs_g = np.abs(g)
    dominant = np.argmax(abs_g)
    coupled = [(VARIABLE_NAMES[j], g[j]) for j in range(len(g))
               if j != dominant and abs(g[j]) > 0.05]
    if coupled:
        direction = "increase" if g[dominant] > 0 else "decrease"
        compensations = []
        for cname, cval in coupled:
            cdirection = "increase" if cval > 0 else "decrease"
            compensations.append(f"{cdirection} {cname}")
        print(f"    → {direction} {VARIABLE_NAMES[dominant]} while {', '.join(compensations)}")
        print(f"      to keep Permeability_X unchanged")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (3 panels)
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(X, y, results, output_dir):
    """Create a 3-panel figure."""
    os.makedirs(output_dir, exist_ok=True)
    generators = results["generators"]
    winner_type = results["winner_type"]
    norm = results["normalization"]
    sym_res = results["symmetry"]
    W = results["W"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Porous Media Permeability — Symmetry Discovery",
                 fontsize=15, fontweight="bold")

    # --- Panel 1: Learned latent variable z vs permeability ---
    ax = axes[0]
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
        ax.set_xlabel("z = W·x (learned latent variable)", fontsize=11)
        ax.set_ylabel("Permeability_X", fontsize=11)
        ax.set_title(f"Latent Variable vs Permeability (R²={r2:.3f})", fontsize=12)
        ax.legend(fontsize=9)
    else:
        sc = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="viridis", s=20, alpha=0.6)
        fig.colorbar(sc, ax=ax, label="Permeability_X")
        ax.set_xlabel("z₁", fontsize=11)
        ax.set_ylabel("z₂", fontsize=11)
        ax.set_title("Latent Variables (colored by permeability)", fontsize=12)

    # --- Panel 2: Symmetry type losses ---
    ax = axes[1]
    types = list(sym_res["losses"].keys())
    losses = [sym_res["losses"][t] for t in types]
    colors = ["#55A868" if t == sym_res["symmetry_type"] else "#DD8452" for t in types]
    bars = ax.bar(types, losses, color=colors, edgecolor="black", lw=1)
    ax.set_ylabel("Validation MSE", fontsize=11)
    ax.set_title(f"Symmetry Type (winner: {sym_res['symmetry_type']})", fontsize=12)
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{loss:.4f}", ha="center", va="bottom", fontsize=9)

    # --- Panel 3: Generator orbits ---
    ax = axes[2]
    if generators:
        g = generators[0]

        if winner_type in ("translational", "scaling") and g.ndim == 1:
            abs_g = np.abs(g)
            top2 = np.argsort(abs_g)[-2:][::-1]
            d0, d1 = top2[0], top2[1]

            sc = ax.scatter(X[:, d0], X[:, d1], c=y, cmap="viridis", s=20, alpha=0.6,
                            edgecolors="none")
            fig.colorbar(sc, ax=ax, label="Permeability_X", fraction=0.046, pad=0.04)

            orbit_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
            rng = np.random.default_rng(42)
            start_indices = rng.choice(len(X), min(3, len(X)), replace=False)

            for k, idx in enumerate(start_indices):
                x_start = norm["X_normalized"][idx]
                n_steps = 100
                eps = 0.03
                fwd = generator_orbit(x_start, g, n_steps, eps, winner_type)
                back = generator_orbit(x_start, g, n_steps, -eps, winner_type)
                orb = np.vstack([back[::-1], fwd[1:]])
                orb_orig = norm["scaler_X"].inverse_transform(orb)

                if winner_type == "scaling":
                    ax.plot(np.log10(np.abs(orb_orig[:, d0]) + 1e-12),
                            np.log10(np.abs(orb_orig[:, d1]) + 1e-12),
                            color=orbit_colors[k % len(orbit_colors)], lw=2, alpha=0.8,
                            label=f"orbit {k+1}")
                else:
                    ax.plot(orb_orig[:, d0], orb_orig[:, d1],
                            color=orbit_colors[k % len(orbit_colors)], lw=2, alpha=0.8,
                            label=f"orbit {k+1}")

            xlabel = f"log₁₀({VARIABLE_NAMES[d0]})" if winner_type == "scaling" else f"{VARIABLE_NAMES[d0]} ({VARIABLE_UNITS[d0]})"
            ylabel = f"log₁₀({VARIABLE_NAMES[d1]})" if winner_type == "scaling" else f"{VARIABLE_NAMES[d1]} ({VARIABLE_UNITS[d1]})"
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title("Generator Orbits (constant-permeability lines)", fontsize=12)
            ax.legend(fontsize=8, loc="best")

        elif winner_type == "rotational" and g.ndim == 2:
            # Find the rotation plane
            pairs = []
            for a in range(g.shape[0]):
                for b in range(a + 1, g.shape[1]):
                    if abs(g[a, b]) > 0.01:
                        pairs.append((a, b))
            if pairs:
                d0, d1 = pairs[0]
            else:
                d0, d1 = 0, 1

            sc = ax.scatter(X[:, d0], X[:, d1], c=y, cmap="viridis", s=20, alpha=0.6,
                            edgecolors="none")
            fig.colorbar(sc, ax=ax, label="Permeability_X", fraction=0.046, pad=0.04)

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

            ax.set_xlabel(f"{VARIABLE_NAMES[d0]} ({VARIABLE_UNITS[d0]})", fontsize=11)
            ax.set_ylabel(f"{VARIABLE_NAMES[d1]} ({VARIABLE_UNITS[d1]})", fontsize=11)
            ax.set_title("Generator Orbits (rotation plane)", fontsize=12)
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

    X, y = load_data(args)
    results = run_pipeline(X, y, args)

    print("=" * 60)
    print("Creating visualizations")
    print("=" * 60)
    plot_results(X, y, results, args.output_dir)

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    sym_type = results["symmetry"]["symmetry_type"]
    print(f"  Symmetry: {sym_type}")
    print(f"  Generators: {len(results['generators'])}")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
