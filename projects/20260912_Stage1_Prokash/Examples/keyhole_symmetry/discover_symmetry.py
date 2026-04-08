"""
Discover the symmetry structure of the known keyhole number using Stage1.

Physics
-------
The keyhole eccentricity e* in laser welding is governed by the known
dimensionless keyhole number (Eq. 12 in the paper):

    Ke = etaP / ((Tl-T0) * pi * rho * Cp * sqrt(alpha * Vs * r0^3))

This example takes the known Ke as given and uses the Stage1 pipeline to:
  1. Confirm that the relationship e* = f(Ke) is a scaling symmetry
  2. Extract the Lie-algebra generators of the symmetry group
  3. Visualize the generators — showing which variable rescalings
     preserve the keyhole number

The generators are the novel output: they reveal the continuous family
of unit-rescaling transformations under which Ke (and thus e*) is invariant.

Usage
-----
    python discover_symmetry.py --data dataset_keyhole.csv
    python discover_symmetry.py --data dataset_keyhole.csv --encoder-hidden 64 32
"""

import sys
import os
import argparse
import traceback
import multiprocessing

import numpy as np
import torch

# Add the Stage1 project to the path — try multiple locations
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

# Prevent silent multiprocessing crashes on Windows
import torch.multiprocessing as _tmp
_tmp.cpu_count = lambda: 0

VARIABLE_NAMES = ["etaP", "Vs", "r0", "alpha", "rho", "cp", "Tl-T0"]
VARIABLE_UNITS = ["W", "m/s", "m", "m²/s", "kg/m³", "J/(kg·K)", "K"]

# Known keyhole number exponents (Eq. 12):
#   Ke = etaP^1 * Vs^(-0.5) * r0^(-1.5) * alpha^(-0.5) * rho^(-1) * cp^(-1) * (Tl-T0)^(-1)
KNOWN_KE_EXPONENTS = np.array([1.0, -0.5, -1.5, -0.5, -1.0, -1.0, -1.0])


def compute_ke(X: np.ndarray) -> np.ndarray:
    """Compute the known keyhole number Ke from 7 physical variables."""
    etaP, Vs, r0, alpha, rho, cp, Tl_T0 = [X[:, i] for i in range(7)]
    return etaP / (Tl_T0 * np.pi * rho * cp * np.sqrt(alpha * Vs * r0**3))


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_data(csv_path: str) -> dict:
    """Load keyhole data from CSV, skipping non-numeric columns."""
    import csv
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    input_cols = []
    for var in VARIABLE_NAMES:
        for i, h in enumerate(header):
            if h.strip() == var:
                input_cols.append(i)
                break

    output_col = None
    for target in ["e*", "Ke", "e"]:
        for i, h in enumerate(header):
            if h.strip() == target:
                output_col = i
                break
        if output_col is not None:
            break

    if len(input_cols) != 7:
        raise ValueError(f"Expected 7 input variables, found {len(input_cols)} in: {header}")
    if output_col is None:
        raise ValueError(f"Could not find output column (e*, Ke, or e) in: {header}")

    X = np.array([[float(rows[r][c]) for c in input_cols] for r in range(len(rows))])
    y = np.array([float(rows[r][output_col]) for r in range(len(rows))])
    print(f"  Loaded: {[header[i].strip() for i in input_cols]} -> {header[output_col].strip()}")
    return {"X": X, "y": y}


def load_data(args):
    """Load data from CSV, compute Ke, return (X, y, Ke)."""
    data_path = args.data
    if not os.path.exists(data_path):
        # Try relative to script directory
        data_path = os.path.join(_here, os.path.basename(args.data))
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {args.data}")
        print(f"Place your keyhole CSV (with columns {VARIABLE_NAMES} and e*/Ke)")
        print(f"in {_here}/ and run:")
        print(f"  python discover_symmetry.py --data <your_file.csv>")
        sys.exit(1)

    print(f"Loading keyhole data from {data_path}...")
    data = load_csv_data(data_path)

    X, y = data["X"], data["y"]
    Ke = compute_ke(X)

    print(f"  Samples: {X.shape[0]}")
    print(f"  Ke range: [{Ke.min():.4g}, {Ke.max():.4g}]")
    print(f"  e* range: [{y.min():.4f}, {y.max():.4f}]")
    print()
    return X, y, Ke


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(X, y, Ke, args):
    """Run Stage1 symmetry discovery on the physical variables."""
    results = {"Ke": Ke}

    # --- Normalize ---
    print("=" * 60)
    print("Step 1: Normalizing data")
    print("=" * 60)
    sys.stdout.flush()
    norm = normalize_data(X, y, method="minmax")
    X_norm, y_norm = norm["X_normalized"], norm["y_normalized"]
    results["normalization"] = norm
    print(f"  X range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
    print()

    # --- Discover latent dimension ---
    print("=" * 60)
    print("Step 2: Discovering intrinsic latent dimension")
    print("=" * 60)
    sys.stdout.flush()
    # Optional: multi-layer encoder and Pi group augmentation
    enc_kwargs = {}
    if getattr(args, "encoder_hidden", None):
        enc_kwargs["encoder_hidden_dims"] = args.encoder_hidden
    if getattr(args, "pi_basis", False):
        # Use known Ke exponents as Pi group basis (1 group)
        enc_kwargs["pi_basis_vectors"] = KNOWN_KE_EXPONENTS.reshape(-1, 1)

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
    print()

    # --- Interpret generators physically ---
    print("=" * 60)
    print("Step 5: Physical interpretation of generators")
    print("=" * 60)
    if winner_type == "scaling" and generators:
        print(f"  Each generator is a direction in log-space along which Ke is preserved.")
        print(f"  Physically: simultaneous rescaling of variables that keeps the physics invariant.\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = []
                for j, name in enumerate(VARIABLE_NAMES):
                    if abs(g[j]) > 0.05:
                        parts.append(f"{name} x exp({g[j]:+.3f}*eps)")
                print(f"  Generator {i+1}:")
                print(f"    {', '.join(parts)}")
                # Physical meaning
                _interpret_generator(g, i + 1)
                print()
    elif winner_type == "rotational" and generators:
        for i, g in enumerate(generators):
            print(f"  Generator {i+1} (antisymmetric matrix):")
            print(f"    {np.round(g, 4)}")
    else:
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = [f"{name}:{g[j]:+.3f}" for j, name in enumerate(VARIABLE_NAMES) if abs(g[j]) > 0.05]
                print(f"  Generator {i+1}: [{', '.join(parts)}]")
    print()

    return results


def _interpret_generator(g, idx):
    """Give a physical interpretation of a scaling generator."""
    # Find the dominant variable
    abs_g = np.abs(g)
    dominant = np.argmax(abs_g)
    name = VARIABLE_NAMES[dominant]

    # Find coupled variables (others that must change to preserve Ke)
    coupled = [(VARIABLE_NAMES[j], g[j]) for j in range(len(g))
               if j != dominant and abs(g[j]) > 0.05]

    if coupled:
        direction = "increase" if g[dominant] > 0 else "decrease"
        compensations = []
        for cname, cval in coupled:
            cdirection = "increase" if cval > 0 else "decrease"
            compensations.append(f"{cdirection} {cname}")
        print(f"    Meaning: {direction} {name} while {', '.join(compensations)}")
        print(f"             to keep Ke (and e*) unchanged")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (3 panels: Ke vs e*, symmetry losses, generator orbits)
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(X, y, results, output_dir):
    """Create a focused 3-panel figure."""
    os.makedirs(output_dir, exist_ok=True)
    Ke = results["Ke"]
    generators = results["generators"]
    winner_type = results["winner_type"]
    norm = results["normalization"]
    sym_res = results["symmetry"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Keyhole — Symmetry Discovery", fontsize=15, fontweight="bold")

    # --- Panel 1: Known Ke vs e* ---
    ax = axes[0]
    ax.scatter(Ke, y, c="#4C72B0", s=20, alpha=0.6, edgecolors="none")
    coeffs = np.polyfit(Ke, y, 2)
    Ke_fit = np.linspace(Ke.min(), Ke.max(), 200)
    ax.plot(Ke_fit, np.polyval(coeffs, Ke_fit), "r-", lw=2, label="polynomial fit")
    ss_res = np.sum((y - np.polyval(coeffs, Ke))**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    ax.set_xlabel("Ke (known keyhole number)", fontsize=11)
    ax.set_ylabel("e*", fontsize=11)
    ax.set_title(f"Known Ke vs e*   (R² = {r2:.3f})", fontsize=12)
    ax.legend(fontsize=9)

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

    # --- Panel 3: Generator orbits in log-space ---
    ax = axes[2]
    if generators and winner_type == "scaling":
        # Pick two most important variables from the first generator
        g = generators[0]
        importance = np.abs(g)
        top2 = np.argsort(importance)[-2:][::-1]
        d0, d1 = top2[0], top2[1]

        sc = ax.scatter(np.log10(X[:, d0] + 1e-12), np.log10(X[:, d1] + 1e-12),
                        c=y, cmap="plasma", s=15, alpha=0.5, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="e*", fraction=0.046, pad=0.04)

        # Trace multiple orbits
        orbit_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        rng = np.random.default_rng(42)
        start_indices = rng.choice(len(X), min(4, len(X)), replace=False)

        for k, idx in enumerate(start_indices):
            x_start = norm["X_normalized"][idx]
            n_steps = 150
            eps = 0.02
            fwd = generator_orbit(x_start, g, n_steps, eps, winner_type)
            back = generator_orbit(x_start, g, n_steps, -eps, winner_type)
            orb = np.vstack([back[::-1], fwd[1:]])
            orb_orig = norm["scaler_X"].inverse_transform(orb)

            ax.plot(np.log10(np.abs(orb_orig[:, d0]) + 1e-12),
                    np.log10(np.abs(orb_orig[:, d1]) + 1e-12),
                    color=orbit_colors[k % len(orbit_colors)], lw=2, alpha=0.8,
                    label=f"orbit {k+1}" if k < 3 else None)

        ax.set_xlabel(f"log₁₀({VARIABLE_NAMES[d0]})", fontsize=11)
        ax.set_ylabel(f"log₁₀({VARIABLE_NAMES[d1]})", fontsize=11)
        ax.set_title("Generator Orbits (scaling directions)", fontsize=12)
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, f"No scaling orbits\n(detected: {winner_type})",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Generator Orbits")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, "keyhole_symmetry_discovery.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Discover symmetry in keyhole welding data")
    parser.add_argument("--data", default="dataset_keyhole.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-epochs", type=int, default=600)
    parser.add_argument("--sym-epochs", type=int, default=1500)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output-dir", default="output_keyhole_symmetry")
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=None,
                        help="Hidden layer widths for multi-layer encoder (e.g. --encoder-hidden 64 32)")
    parser.add_argument("--pi-basis", action="store_true",
                        help="Augment encoder input with known Ke Pi group")
    args = parser.parse_args()

    X, y, Ke = load_data(args)
    results = run_pipeline(X, y, Ke, args)

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
    if sym_type == "scaling":
        print(f"  These generators show how physical variables can be")
        print(f"  simultaneously rescaled while preserving Ke and e*.")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
