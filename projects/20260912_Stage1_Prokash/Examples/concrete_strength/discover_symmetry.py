"""
Discover translational symmetry in concrete compressive strength data.

Physics
-------
Concrete compressive strength depends on mix proportions (all in kg/m³):
    cement, blast furnace slag, fly ash, water, superplasticizer,
    coarse aggregate, fine aggregate, and curing age (days).

The key governing relationships are **additive**:
    - total binder = cement + slag + fly ash
    - water/binder ratio = water / (cement + slag + fly ash)
    - strength ≈ f(a₁·cement + a₂·slag + a₃·fly_ash + a₄·water + ...)

This additive structure means the symmetry is **translational**:
    y = f(W·x)  where W·x is a linear combination (no log or x² transform).

The generators are directions g in input space with W·g = 0, i.e.,
shifts in mix proportions that preserve strength.

Data
----
UCI ML Repository dataset #165 (I-Cheng Yeh, 1998):
    1030 samples, 8 inputs, 1 output (compressive strength in MPa).

Usage
-----
    python discover_symmetry.py --data Concrete_Data.csv
    python discover_symmetry.py --data Concrete_Data.xls
    python discover_symmetry.py --data Concrete_Data.csv --encoder-hidden 64 32
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

VARIABLE_NAMES = [
    "Cement", "Slag", "Fly Ash", "Water",
    "Superplast.", "Coarse Agg.", "Fine Agg.", "Age",
]
VARIABLE_UNITS = [
    "kg/m³", "kg/m³", "kg/m³", "kg/m³",
    "kg/m³", "kg/m³", "kg/m³", "days",
]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_data(csv_path: str) -> dict:
    """Load concrete data from CSV/XLS."""
    ext = os.path.splitext(csv_path)[1].lower()

    if ext in (".xls", ".xlsx"):
        try:
            import pandas as pd
            df = pd.read_excel(csv_path)
            X = df.iloc[:, :8].values.astype(float)
            y = df.iloc[:, 8].values.astype(float)
            print(f"  Loaded Excel: {df.columns.tolist()}")
            return {"X": X, "y": y}
        except ImportError:
            print("  pandas/openpyxl needed for Excel files. Install with:")
            print("    pip install pandas openpyxl")
            sys.exit(1)

    # CSV fallback
    import csv
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # First 8 columns = inputs, last column = strength
    X = np.array([[float(row[i]) for i in range(8)] for row in rows])
    y = np.array([float(row[8]) for row in rows])
    print(f"  Loaded CSV: {[h.strip() for h in header]}")
    print(f"  Inputs: {header[:8]}")
    print(f"  Output: {header[8]}")
    return {"X": X, "y": y}


def load_data(args):
    """Load experimental data from CSV/Excel file."""
    data_path = args.data
    if not os.path.exists(data_path):
        # Try relative to script directory
        data_path = os.path.join(_here, os.path.basename(args.data))
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {args.data}")
        print(f"Download the UCI Concrete dataset and place it in {_here}/")
        print(f"  python discover_symmetry.py --data Concrete_Data.csv")
        print(f"  python discover_symmetry.py --data Concrete_Data.xls")
        sys.exit(1)

    print(f"Loading concrete data from {data_path}...")
    data = load_csv_data(data_path)

    X, y = data["X"], data["y"]
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Strength range: [{y.min():.1f}, {y.max():.1f}] MPa")
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
    enc_kwargs = {}
    if getattr(args, "encoder_hidden", None):
        enc_kwargs["encoder_hidden_dims"] = args.encoder_hidden

    res_latent = discover_latent_dimension(
        X_norm, y_norm, max_latent=4,
        n_epochs=args.latent_epochs, n_restarts=args.n_restarts, seed=args.seed,
        **enc_kwargs,
    )
    results["latent"] = res_latent
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

    print(f"  Symmetry type: {winner_type}")
    print(f"  Generators: {len(generators)}")

    # --- Encoder weights (learned linear combination) ---
    W = winner_encoder.weight_matrix
    results["W"] = W
    print(f"\n  Learned encoder weights W:")
    if W.shape[0] == 1:
        for j, name in enumerate(VARIABLE_NAMES):
            print(f"    {name:12s}: {W[0, j]:+.4f}")
    print()

    # --- Interpret generators physically ---
    print("=" * 60)
    print("Step 5: Physical interpretation of generators")
    print("=" * 60)
    if winner_type == "translational" and generators:
        print(f"  Each generator is a direction in input space along which")
        print(f"  strength is preserved: x → x + ε·g keeps f(W·x) unchanged.\n")
        print(f"  Physically: mix substitutions that maintain the same strength.\n")
        for i, g in enumerate(generators):
            parts = []
            for j, name in enumerate(VARIABLE_NAMES):
                if abs(g[j]) > 0.05:
                    parts.append(f"{name}: {g[j]:+.3f}")
            print(f"  Generator {i+1}: [{', '.join(parts)}]")
            _interpret_translational_generator(g, i + 1)
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
                print()
    elif winner_type == "rotational" and generators:
        for i, g in enumerate(generators):
            print(f"  Generator {i+1} (antisymmetric matrix):")
            print(f"    {np.round(g, 4)}")
    print()

    return results


def _interpret_translational_generator(g, idx):
    """Give a physical interpretation of a translational generator."""
    abs_g = np.abs(g)
    # Find the two most significant components
    top = np.argsort(abs_g)[::-1]
    significant = [(VARIABLE_NAMES[j], g[j]) for j in top if abs(g[j]) > 0.05]

    if len(significant) >= 2:
        # Find pairs that go in opposite directions (substitutions)
        pos = [(n, v) for n, v in significant if v > 0]
        neg = [(n, v) for n, v in significant if v < 0]
        if pos and neg:
            inc = ", ".join(f"{n}" for n, v in pos[:2])
            dec = ", ".join(f"{n}" for n, v in neg[:2])
            print(f"    → Increase {inc} while decreasing {dec}")
            print(f"      to maintain the same compressive strength")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (3 panels)
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(X, y, results, output_dir):
    """Create a 3-panel figure: learned z vs strength, symmetry losses, generator orbits."""
    os.makedirs(output_dir, exist_ok=True)
    generators = results["generators"]
    winner_type = results["winner_type"]
    norm = results["normalization"]
    sym_res = results["symmetry"]
    W = results["W"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Concrete Compressive Strength — Symmetry Discovery",
                 fontsize=15, fontweight="bold")

    # --- Panel 1: Learned latent variable z vs strength ---
    ax = axes[0]
    X_norm = norm["X_normalized"]
    # Compute z = W·x for each sample
    z = X_norm @ W.T  # (n_samples, n_latent)
    if z.shape[1] == 1:
        z = z.ravel()
        ax.scatter(z, y, c="#4C72B0", s=12, alpha=0.5, edgecolors="none")
        # Fit line
        coeffs = np.polyfit(z, y, 2)
        z_fit = np.linspace(z.min(), z.max(), 200)
        ax.plot(z_fit, np.polyval(coeffs, z_fit), "r-", lw=2, label="quadratic fit")
        ss_res = np.sum((y - np.polyval(coeffs, z)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        ax.set_xlabel("z = W·x (learned latent variable)", fontsize=11)
        ax.set_ylabel("Compressive Strength (MPa)", fontsize=11)
        ax.set_title(f"Latent Variable vs Strength (R²={r2:.3f})", fontsize=12)
        ax.legend(fontsize=9)
    else:
        ax.scatter(z[:, 0], z[:, 1], c=y, cmap="viridis", s=12, alpha=0.5)
        ax.set_xlabel("z₁", fontsize=11)
        ax.set_ylabel("z₂", fontsize=11)
        ax.set_title("Latent Variables (colored by strength)", fontsize=12)

    # --- Panel 2: Symmetry type identification ---
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
    if generators and winner_type == "translational":
        g = generators[0]
        abs_g = np.abs(g)
        top2 = np.argsort(abs_g)[-2:][::-1]
        d0, d1 = top2[0], top2[1]

        sc = ax.scatter(X[:, d0], X[:, d1], c=y, cmap="viridis", s=12, alpha=0.5,
                        edgecolors="none")
        fig.colorbar(sc, ax=ax, label="Strength (MPa)", fraction=0.046, pad=0.04)

        # Trace orbits from different starting points
        orbit_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        rng = np.random.default_rng(42)
        start_indices = rng.choice(len(X), min(4, len(X)), replace=False)

        for k, idx in enumerate(start_indices):
            x_start = norm["X_normalized"][idx]
            n_steps = 100
            eps = 0.03
            fwd = generator_orbit(x_start, g, n_steps, eps, winner_type)
            back = generator_orbit(x_start, g, n_steps, -eps, winner_type)
            orb = np.vstack([back[::-1], fwd[1:]])
            orb_orig = norm["scaler_X"].inverse_transform(orb)

            ax.plot(orb_orig[:, d0], orb_orig[:, d1],
                    color=orbit_colors[k % len(orbit_colors)], lw=2, alpha=0.8,
                    label=f"orbit {k+1}" if k < 3 else None)

        ax.set_xlabel(f"{VARIABLE_NAMES[d0]} ({VARIABLE_UNITS[d0]})", fontsize=11)
        ax.set_ylabel(f"{VARIABLE_NAMES[d1]} ({VARIABLE_UNITS[d1]})", fontsize=11)
        ax.set_title("Generator Orbits (constant-strength lines)", fontsize=12)
        ax.legend(fontsize=8, loc="best")

    elif generators and winner_type == "scaling":
        g = generators[0]
        abs_g = np.abs(g)
        top2 = np.argsort(abs_g)[-2:][::-1]
        d0, d1 = top2[0], top2[1]

        sc = ax.scatter(np.log10(X[:, d0] + 1e-12), np.log10(X[:, d1] + 1e-12),
                        c=y, cmap="viridis", s=12, alpha=0.5, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="Strength (MPa)", fraction=0.046, pad=0.04)
        ax.set_xlabel(f"log₁₀({VARIABLE_NAMES[d0]})", fontsize=11)
        ax.set_ylabel(f"log₁₀({VARIABLE_NAMES[d1]})", fontsize=11)
        ax.set_title("Generator Orbits (log-space)", fontsize=12)
    else:
        ax.text(0.5, 0.5, f"No orbits for {winner_type}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Generator Orbits")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, "concrete_symmetry_discovery.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Discover symmetry in concrete compressive strength data"
    )
    parser.add_argument("--data", default="Concrete_Data.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-epochs", type=int, default=600)
    parser.add_argument("--sym-epochs", type=int, default=1500)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output-dir", default="output_concrete_symmetry")
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=None,
                        help="Hidden layer widths for multi-layer encoder (e.g. --encoder-hidden 64 32)")
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
    if sym_type == "translational":
        print(f"  These generators show mix substitutions (e.g., replace cement")
        print(f"  with slag) that preserve the same compressive strength.")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
