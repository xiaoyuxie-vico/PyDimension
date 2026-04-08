"""
Discover symmetry in porous media permeability using Stage1.

Physics
-------
Each row belongs to one of 180 circular porous media geometries.
Within each geometry (fixed Porosity & Surface_A), permeability is
measured at angles 0, 10, 20, ..., 360 and repeats every 180 degrees.

We encode this known 180-degree periodicity by decomposing angle into:
    cos(2θ) and sin(2θ)

Input features to the pipeline (4 total):
    [cos(2θ), sin(2θ), Porosity, Surface_A]

The pipeline discovers:
    1. Symmetry type (translational / rotational / scaling)
    2. Lie-algebra generators

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

FEATURE_NAMES = ["cos(2θ)", "sin(2θ)"]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

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
    data_path = args.data
    if not os.path.exists(data_path):
        data_path = os.path.join(_here, args.data)
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {args.data}")
        print(f"Place your permeability CSV (with Angle, Porosity, Surface_A, Permeability_X)")
        print(f"in {_here}/ and run:")
        print(f"  python discover_symmetry.py --data permeability.csv")
        sys.exit(1)

    print(f"Loading from {data_path}...")
    angle, porosity, surface_a, y = load_csv_data(data_path)

    angle_rad = np.radians(angle)
    X = np.column_stack([
        np.cos(2 * angle_rad),
        np.sin(2 * angle_rad),
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
# Pipeline (minmax normalization only)
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(raw, X, y, args):
    """Run Stage1 pipeline with minmax normalization."""

    # --- Normalize ---
    print("=" * 60)
    print("Step 1: Normalizing data (minmax)")
    print("=" * 60)
    sys.stdout.flush()
    norm = normalize_data(X, y, method="minmax")
    X_norm, y_norm = norm["X_normalized"], norm["y_normalized"]
    print(f"  X range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
    print()

    # --- Discover latent dimension ---
    print("=" * 60)
    print("Step 2: Discovering intrinsic latent dimension")
    print("=" * 60)
    sys.stdout.flush()
    enc_kwargs = {}
    if getattr(args, "encoder_hidden", None):
        enc_kwargs["encoder_hidden_dims"] = args.encoder_hidden

    res_latent = discover_latent_dimension(
        X_norm, y_norm, max_latent=2,
        n_epochs=args.latent_epochs, n_restarts=args.n_restarts, seed=args.seed,
        **enc_kwargs,
    )
    n_latent = res_latent["optimal_n_latent"]
    print(f"\n  Optimal latent dimension: {n_latent}")
    for k, m in res_latent["metrics"].items():
        r2_tr = m.get("R2_train", float("nan"))
        print(f"    k={k}: R2_train={r2_tr:.4f}, R2_test={m['R2']:.4f}, MSE={m['MSE']:.6f}")
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
    W = winner_encoder.weight_matrix
    generators = extract_generators(winner_type, winner_encoder)

    print(f"  Symmetry type: {winner_type}")
    print(f"  Generators: {len(generators)}")
    print(f"\n  Encoder weights W:")
    for row_i in range(W.shape[0]):
        parts = [f"{FEATURE_NAMES[j]}:{W[row_i, j]:+.4f}" for j in range(W.shape[1])]
        print(f"    z{row_i+1}: [{', '.join(parts)}]")

    # Generator interpretation
    print(f"\n  Physical interpretation of generators:")
    if winner_type == "translational" and generators:
        print(f"  Each generator: x -> x + eps*g preserves permeability.\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = [f"{FEATURE_NAMES[j]}: {g[j]:+.3f}"
                         for j in range(len(g)) if abs(g[j]) > 0.05]
                print(f"    G{i+1}: [{', '.join(parts)}]")
    elif winner_type == "rotational" and generators:
        print(f"  Each generator: rotation in a 2D plane.\n")
        for i, g in enumerate(generators):
            if g.ndim == 2:
                print(f"    G{i+1}:")
                for a in range(g.shape[0]):
                    for b in range(a + 1, g.shape[1]):
                        if abs(g[a, b]) > 0.01:
                            print(f"      Rotation in ({FEATURE_NAMES[a]}, {FEATURE_NAMES[b]}) plane")
    elif winner_type == "scaling" and generators:
        print(f"  Each generator: x -> x*exp(eps*s) in log-space.\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = [f"{FEATURE_NAMES[j]} x exp({g[j]:+.3f}*eps)"
                         for j in range(len(g)) if abs(g[j]) > 0.05]
                print(f"    G{i+1}: [{', '.join(parts)}]")
    print()

    results = {
        "normalization": norm,
        "latent": res_latent,
        "symmetry": res_sym,
        "winner_type": winner_type,
        "winner_encoder": winner_encoder,
        "generators": generators,
        "W": W,
    }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (3 panels)
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(raw, X, y, results, output_dir):
    """3-panel figure: raw data, symmetry losses, generator orbits."""
    os.makedirs(output_dir, exist_ok=True)
    angle = raw["angle"]
    sym_res = results["symmetry"]
    generators = results["generators"]
    winner_type = results["winner_type"]
    norm = results["normalization"]
    W = results["W"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Porous Media Permeability — Symmetry Discovery",
                 fontsize=15, fontweight="bold")

    # Panel 1: Raw data (Perm vs Angle, colored by Porosity)
    ax = axes[0]
    sc = ax.scatter(angle, y, c=raw["porosity"], cmap="viridis", s=8, alpha=0.5,
                    edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Porosity", fraction=0.046, pad=0.04)
    ax.set_xlabel("Angle (degrees)", fontsize=11)
    ax.set_ylabel("Permeability_X", fontsize=11)
    ax.set_title("Raw Data (color=Porosity)", fontsize=12)

    # Panel 2: Symmetry losses
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

    # Panel 3: Generator orbits
    ax = axes[2]
    if generators:
        g = generators[0]

        if g.ndim == 1:
            # Pick two most important features
            abs_g = np.abs(g)
            top2 = np.argsort(abs_g)[-2:][::-1]
            d0, d1 = top2[0], top2[1]

            sc = ax.scatter(X[:, d0], X[:, d1], c=y, cmap="viridis", s=8, alpha=0.4,
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

            ax.set_xlabel(FEATURE_NAMES[d0], fontsize=11)
            ax.set_ylabel(FEATURE_NAMES[d1], fontsize=11)
            ax.set_title("Generator Orbits", fontsize=12)
            ax.legend(fontsize=8, loc="best")

        elif g.ndim == 2:
            # Rotational: find the rotation plane
            pairs = []
            for a in range(g.shape[0]):
                for b in range(a + 1, g.shape[1]):
                    if abs(g[a, b]) > 0.01:
                        pairs.append((a, b))
            d0, d1 = pairs[0] if pairs else (0, 1)

            sc = ax.scatter(X[:, d0], X[:, d1], c=y, cmap="viridis", s=8, alpha=0.4,
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

            ax.set_xlabel(FEATURE_NAMES[d0], fontsize=11)
            ax.set_ylabel(FEATURE_NAMES[d1], fontsize=11)
            ax.set_title("Generator Orbits", fontsize=12)
            ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "No generators found",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-epochs", type=int, default=600)
    parser.add_argument("--sym-epochs", type=int, default=1500)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output-dir", default="output_permeability_symmetry")
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=None,
                        help="Hidden layer widths for multi-layer encoder (e.g. --encoder-hidden 64 32)")
    args = parser.parse_args()

    raw, X, y = load_data(args)
    results = run_pipeline(raw, X, y, args)

    print("=" * 60)
    print("Creating visualizations")
    print("=" * 60)
    plot_results(raw, X, y, results, args.output_dir)

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Symmetry: {results['winner_type']}")
    print(f"  Generators: {len(results['generators'])}")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
