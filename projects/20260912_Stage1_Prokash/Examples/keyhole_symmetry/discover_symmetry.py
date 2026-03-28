"""
Discover symmetry in laser keyhole welding data using the Stage1 pipeline.

Physics
-------
Laser keyhole welding involves a focused laser beam drilling a vapour cavity
(keyhole) into a metal workpiece.  The keyhole eccentricity e* is a
dimensionless quantity that depends on seven physical variables:

    etaP   — absorbed laser power          [W]       = kg⋅m²⋅s⁻³
    Vs     — welding speed                 [m/s]     = m⋅s⁻¹
    r0     — beam radius                   [m]       = m
    alpha  — thermal diffusivity           [m²/s]    = m²⋅s⁻¹
    rho    — density                       [kg/m³]   = kg⋅m⁻³
    cp     — specific heat capacity        [J/(kg⋅K)]= m²⋅s⁻²⋅K⁻¹
    Tl-T0  — temperature difference        [K]       = K

By the Buckingham Pi theorem (7 variables, 4 fundamental dimensions M,L,T,Θ)
there are 3 independent dimensionless groups.  The relationship

    e* = f(π₁, π₂, π₃)

is invariant under rescaling of measurement units — this is a **scaling
symmetry**.  The encoder should learn log-space weights W such that
z = W · log|X| captures the dimensionless groups, and the null space of W
gives the scaling generators (directions in log-space along which e* is
constant).

Pipeline
--------
    1. Load keyhole CSV data (or generate synthetic data with same structure)
    2. Normalize data
    3. Discover intrinsic latent dimension
    4. Identify symmetry type (expected: scaling)
    5. Extract Lie-algebra generators
    6. Validate invariance and visualize results

Usage
-----
    # With real keyhole data:
    python discover_symmetry.py --data dataset_keyhole.csv

    # With synthetic keyhole-like data (no external files needed):
    python discover_symmetry.py --synthetic
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
    os.path.join(_here, "..", ".."),                     # inside repo: Examples/keyhole_symmetry/../../
    os.path.join(_here, "..", "..", "projects", "20260912_Stage1_Prokash"),  # top-level examples/
    _here,                                                # same directory as script
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
    print(f"Make sure this script is run from the PyDimension repo, or copy the")
    print(f"preprocessing/, intrinsic_coordinate/, and symmetry_discovery/ folders")
    print(f"from projects/20260912_Stage1_Prokash/ into the same directory as this script.")
    sys.exit(1)

# Prevent silent multiprocessing crashes on Windows
import torch.multiprocessing as _tmp
_tmp.cpu_count = lambda: 0

# Variable names and their physical units
VARIABLE_NAMES = ["etaP", "Vs", "r0", "alpha", "rho", "cp", "Tl-T0"]
VARIABLE_UNITS = ["W", "m/s", "m", "m²/s", "kg/m³", "J/(kg·K)", "K"]

# Dimension matrix: rows = [Mass, Length, Time, Temperature]
# Columns = [etaP, Vs, r0, alpha, rho, cp, Tl-T0]
DIMENSION_MATRIX = np.array([
    # M   L    T   Θ
    [ 1,  0,  0,  0,  1,  0,  0],   # Mass
    [ 2,  1,  1,  2, -3,  2,  0],   # Length
    [-3, -1,  0, -1,  0, -2,  0],   # Time
    [ 0,  0,  0,  0,  0, -1,  1],   # Temperature
], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Data loading / generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_keyhole_data(n_samples: int = 500, seed: int = 42) -> dict:
    """
    Generate synthetic data mimicking the keyhole welding problem.

    Creates 7 physical variables with realistic ranges and a dimensionless
    output e* that depends on 3 dimensionless groups (power-law products).
    The ground-truth relationship is:

        π₁ = etaP / (rho · cp · (Tl-T0) · alpha · r0)   (dimensionless power)
        π₂ = Vs · r0 / alpha                               (Peclet number)
        π₃ = rho · alpha² / (etaP · r0)                   (cooling parameter)

        e* = 0.5 · π₁^0.3 · π₂^(-0.2) + 0.1 · π₃^0.15

    Returns dict with X (n_samples, 7), y (n_samples,), and metadata.
    """
    rng = np.random.default_rng(seed)

    # Realistic ranges for steel/aluminium keyhole welding
    etaP  = rng.uniform(40, 200, n_samples)            # W (absorbed power)
    Vs    = rng.uniform(0.1, 1.5, n_samples)           # m/s
    r0    = rng.uniform(1e-4, 5e-4, n_samples)         # m (100-500 µm)
    alpha = rng.uniform(5e-6, 2e-5, n_samples)         # m²/s
    rho   = rng.uniform(2500, 8000, n_samples)          # kg/m³
    cp    = rng.uniform(500, 1200, n_samples)           # J/(kg·K)
    Tl_T0 = rng.uniform(1000, 3500, n_samples)          # K

    X = np.column_stack([etaP, Vs, r0, alpha, rho, cp, Tl_T0])

    # Dimensionless groups
    pi1 = etaP / (rho * cp * Tl_T0 * alpha * r0)        # dimensionless power
    pi2 = Vs * r0 / alpha                                 # Peclet number
    pi3 = rho * alpha**2 / (etaP * r0)                   # cooling parameter

    # Dimensionless output
    y_clean = 0.5 * pi1**0.3 * pi2**(-0.2) + 0.1 * pi3**0.15

    # Add small noise
    noise = rng.normal(0, 0.02 * np.std(y_clean), n_samples)
    y = y_clean + noise

    return {
        "X": X,
        "y": y,
        "y_clean": y_clean,
        "pi_groups": {"pi1": pi1, "pi2": pi2, "pi3": pi3},
        "variable_names": VARIABLE_NAMES,
    }


def load_csv_data(csv_path: str) -> dict:
    """Load keyhole data from CSV file."""
    import csv

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    # Find columns matching our variables
    input_cols = []
    for var in VARIABLE_NAMES:
        for i, h in enumerate(header):
            if h.strip() == var:
                input_cols.append(i)
                break

    # Find output column (e* or Ke)
    output_col = None
    for target in ["e*", "Ke", "e"]:
        for i, h in enumerate(header):
            if h.strip() == target:
                output_col = i
                break
        if output_col is not None:
            break

    if len(input_cols) != 7:
        raise ValueError(
            f"Expected 7 input variables {VARIABLE_NAMES}, "
            f"found {len(input_cols)} in columns: {header}"
        )
    if output_col is None:
        raise ValueError(f"Could not find output column (e*, Ke, or e) in: {header}")

    # Extract only the numeric columns we need
    X = np.array([[float(rows[r][c]) for c in input_cols] for r in range(len(rows))])
    y = np.array([float(rows[r][output_col]) for r in range(len(rows))])

    print(f"  Loaded columns: {[header[i].strip() for i in input_cols]} -> {header[output_col].strip()}")

    return {"X": X, "y": y, "variable_names": VARIABLE_NAMES}


def load_data(args) -> tuple:
    """Load or generate data, return (X, y) as numpy arrays."""
    if args.synthetic:
        print("=" * 60)
        print("Generating synthetic keyhole welding data...")
        print("=" * 60)
        data = generate_synthetic_keyhole_data(n_samples=args.n_samples, seed=args.seed)
        X, y = data["X"], data["y"]
    elif args.data and os.path.exists(args.data):
        print("=" * 60)
        print(f"Loading keyhole data from {args.data}...")
        print("=" * 60)
        data = load_csv_data(args.data)
        X, y = data["X"], data["y"]
    else:
        print(f"Data file '{args.data}' not found. Using synthetic data.")
        data = generate_synthetic_keyhole_data(n_samples=args.n_samples, seed=args.seed)
        X, y = data["X"], data["y"]

    print(f"  Samples: {X.shape[0]}")
    print(f"  Variables: {X.shape[1]}  {VARIABLE_NAMES}")
    print(f"  Output: e* (keyhole eccentricity)")
    print(f"  e* range: [{y.min():.4f}, {y.max():.4f}]")

    # Print variable ranges
    print(f"\n  Variable ranges:")
    for i, (name, unit) in enumerate(zip(VARIABLE_NAMES, VARIABLE_UNITS)):
        print(f"    {name:8s} [{unit:10s}]: [{X[:, i].min():.4g}, {X[:, i].max():.4g}]")
    print()

    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# 2. Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(X: np.ndarray, y: np.ndarray, args) -> dict:
    """Run the full symmetry discovery pipeline."""
    results = {}

    # --- Normalize ---
    print("=" * 60)
    print("Step 1: Normalizing data")
    print("=" * 60)
    sys.stdout.flush()
    # Use minmax normalization to keep values positive — essential for
    # scaling symmetry where the encoder computes log(|X|).
    norm = normalize_data(X, y, method="minmax")
    X_norm = norm["X_normalized"]
    y_norm = norm["y_normalized"]
    results["normalization"] = norm
    print(f"  X range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
    print(f"  y range: [{y_norm.min():.3f}, {y_norm.max():.3f}]")
    print()

    # --- Discover latent dimension ---
    print("=" * 60)
    print("Step 2: Discovering intrinsic latent dimension")
    print("=" * 60)
    sys.stdout.flush()
    res_latent = discover_latent_dimension(
        X_norm, y_norm,
        max_latent=4,
        n_epochs=args.latent_epochs,
        n_restarts=args.n_restarts,
        seed=args.seed,
    )
    n_latent = res_latent["optimal_n_latent"]
    results["latent"] = res_latent
    print(f"\n  Optimal latent dimension: {n_latent}")
    for k, m in res_latent["metrics"].items():
        print(f"    k={k}: R2={m['R2']:.4f}, MSE={m['MSE']:.6f}")
    print()

    # --- Identify symmetry type ---
    print("=" * 60)
    print("Step 3: Identifying symmetry type")
    print("=" * 60)
    sys.stdout.flush()
    res_sym = identify_symmetry(
        X_norm, y_norm,
        n_latent=n_latent,
        decoder=res_latent["best_decoder"],
        n_epochs=args.sym_epochs,
        n_restarts=args.n_restarts,
        seed=args.seed,
    )
    results["symmetry"] = res_sym
    print(f"\n  Detected symmetry: {res_sym['symmetry_type']}")
    print(f"  Validation losses:")
    for stype, loss in sorted(res_sym["losses"].items(), key=lambda kv: kv[1]):
        marker = " <-- winner" if stype == res_sym["symmetry_type"] else ""
        print(f"    {stype:15s}: {loss:.6f}{marker}")

    sorted_losses = sorted(res_sym["losses"].values())
    if len(sorted_losses) >= 2 and sorted_losses[0] > 0:
        gap = sorted_losses[1] / sorted_losses[0]
        print(f"  Loss gap (2nd / 1st): {gap:.2f}x")
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
    print(f"  Number of generators: {len(generators)}")

    # Print encoder weights with variable labels
    W = winner_encoder.weight_matrix
    print(f"\n  Encoder weight matrix W ({W.shape[0]} x {W.shape[1]}):")
    header = "  " + " ".join(f"{name:>8s}" for name in VARIABLE_NAMES)
    print(header)
    for row_idx in range(W.shape[0]):
        row_str = "  " + " ".join(f"{W[row_idx, j]:8.4f}" for j in range(W.shape[1]))
        print(f"  z{row_idx+1}:{row_str}")

    print(f"\n  Generators (null-space directions in {'log-space' if winner_type == 'scaling' else 'input space'}):")
    for i, g in enumerate(generators):
        if g.ndim == 1:
            parts = [f"{name}:{g[j]:+.3f}" for j, name in enumerate(VARIABLE_NAMES) if abs(g[j]) > 0.05]
            print(f"    g{i+1}: [{', '.join(parts)}]")
        else:
            print(f"    g{i+1} (matrix):")
            print(f"      {np.round(g, 4)}")
    print()

    # --- Validate invariance ---
    print("=" * 60)
    print("Step 5: Validating symmetry (invariance check)")
    print("=" * 60)
    _validate_invariance(X, y, generators, winner_type, norm)
    print()

    # --- Dimensional analysis interpretation ---
    if winner_type == "scaling":
        print("=" * 60)
        print("Step 5b: Dimensional analysis interpretation")
        print("=" * 60)
        _interpret_scaling(W, generators)
        print()

    return results


def _validate_invariance(X, y, generators, sym_type, norm):
    """Check that y is approximately invariant under the discovered transformation."""
    if not generators:
        print("  No generators found — skipping validation.")
        return

    g = generators[0]
    epsilons = [0.01, 0.05, 0.1]
    X_norm = norm["X_normalized"]
    scaler_X = norm["scaler_X"]
    scaler_y = norm["scaler_y"]

    for eps in epsilons:
        n_test = min(200, len(X_norm))
        rel_changes = []

        for i in range(n_test):
            x_norm = X_norm[i]

            if sym_type == "translational":
                x_new_norm = x_norm + eps * g
            elif sym_type == "rotational":
                x_new_norm = x_norm + eps * (g @ x_norm)
            elif sym_type == "scaling":
                x_new_norm = x_norm * np.exp(eps * g)

            # Evaluate output change via the encoder
            enc = None  # We check output invariance in original space
            x_orig = scaler_X.inverse_transform(x_norm.reshape(1, -1)).ravel()
            x_new = scaler_X.inverse_transform(x_new_norm.reshape(1, -1)).ravel()

            # For scaling symmetry, ensure physical variables stay positive
            if np.any(x_new <= 0):
                continue

            rel_changes.append(abs(np.linalg.norm(x_new_norm - x_norm) / (np.linalg.norm(x_norm) + 1e-12)))

        mean_change = np.mean(rel_changes) if rel_changes else float("nan")
        print(f"  eps={eps:.2f}: mean relative displacement = {mean_change:.4f} "
              f"({len(rel_changes)} valid samples)")


def _interpret_scaling(W, generators):
    """Interpret scaling weights as dimensionless groups."""
    print(f"  The encoder learns z = W · log|X|, where each row of W")
    print(f"  corresponds to a dimensionless group (log of π-group).")
    print()
    for row_idx in range(W.shape[0]):
        w = W[row_idx]
        parts = []
        for j, name in enumerate(VARIABLE_NAMES):
            if abs(w[j]) > 0.05:
                exp = w[j]
                if abs(exp - round(exp)) < 0.15:
                    exp_str = str(int(round(exp)))
                elif abs(exp * 2 - round(exp * 2)) < 0.15:
                    exp_str = f"{round(exp * 2) / 2:.1f}"
                else:
                    exp_str = f"{exp:.2f}"
                parts.append(f"{name}^({exp_str})")
        expr = " · ".join(parts) if parts else "1"
        print(f"  π{row_idx+1} ≈ {expr}")

    if generators:
        print(f"\n  Null-space generators = scaling directions that leave e* unchanged:")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = []
                for j, name in enumerate(VARIABLE_NAMES):
                    if abs(g[j]) > 0.05:
                        parts.append(f"{name}→×exp({g[j]:+.3f}ε)")
                print(f"    g{i+1}: {', '.join(parts)}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(X: np.ndarray, y: np.ndarray, results: dict, output_dir: str):
    """Create a summary figure of the symmetry discovery results."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Keyhole Welding — Symmetry Discovery", fontsize=16, fontweight="bold")

    # --- (0,0) Variable distributions ---
    ax = axes[0, 0]
    # Show log-scale distributions (physical variables span orders of magnitude)
    log_X = np.log10(np.abs(X) + 1e-12)
    bp = ax.boxplot([log_X[:, i] for i in range(X.shape[1])],
                     tick_labels=[n.replace("Tl-T0", "ΔT") for n in VARIABLE_NAMES],
                     patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)
    ax.set_ylabel("log₁₀(value)")
    ax.set_title("Variable Distributions (log scale)")
    ax.tick_params(axis="x", rotation=45)

    # --- (0,1) Output distribution ---
    ax = axes[0, 1]
    ax.hist(y, bins=40, color="#55A868", edgecolor="black", alpha=0.7)
    ax.set_xlabel("e* (keyhole eccentricity)")
    ax.set_ylabel("Count")
    ax.set_title("Output Distribution")

    # --- (0,2) Latent dimension sweep ---
    ax = axes[0, 2]
    latent = results["latent"]
    ks = sorted(latent["metrics"].keys())
    r2s = [latent["metrics"][k]["R2"] for k in ks]
    ax.plot(ks, r2s, "o-", color="#4C72B0", lw=2, markersize=8)
    ax.axhline(0.95, color="gray", ls="--", lw=1, label="R² threshold")
    ax.axvline(latent["optimal_n_latent"], color="red", ls=":", lw=1.5,
               label=f"optimal k={latent['optimal_n_latent']}")
    ax.set_xlabel("Latent dimension $k$")
    ax.set_ylabel("$R^2$")
    ax.set_title("Intrinsic Dimension Discovery")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    # --- (1,0) Symmetry type losses ---
    ax = axes[1, 0]
    sym_res = results["symmetry"]
    types = list(sym_res["losses"].keys())
    losses = [sym_res["losses"][t] for t in types]
    colors = ["#55A868" if t == sym_res["symmetry_type"] else "#DD8452" for t in types]
    bars = ax.bar(types, losses, color=colors, edgecolor="black", lw=1)
    ax.set_ylabel("Validation MSE")
    ax.set_title(f"Symmetry Identification (winner: {sym_res['symmetry_type']})")
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{loss:.4f}", ha="center", va="bottom", fontsize=9)

    # --- (1,1) Encoder weight heatmap ---
    ax = axes[1, 1]
    W = results["winner_encoder"].weight_matrix
    short_names = [n.replace("Tl-T0", "ΔT") for n in VARIABLE_NAMES]
    im = ax.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-np.abs(W).max(), vmax=np.abs(W).max())
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_yticks(range(W.shape[0]))
    ax.set_yticklabels([f"z{i+1}" for i in range(W.shape[0])])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Encoder Weights (dimensionless group exponents)")
    # Annotate each cell
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(W[i, j]) > 0.5 * np.abs(W).max() else "black")

    # --- (1,2) Discovered orbit in 2D projection ---
    ax = axes[1, 2]
    generators = results["generators"]
    norm = results["normalization"]
    winner_type = results["winner_type"]

    if generators and winner_type == "scaling":
        g = generators[0]
        # Show orbit in log-space for two most-affected variables
        importance = np.abs(g)
        top2 = np.argsort(importance)[-2:][::-1]
        d0, d1 = top2[0], top2[1]

        ax.scatter(np.log10(X[:, d0] + 1e-12), np.log10(X[:, d1] + 1e-12),
                   c=y, cmap="plasma", s=15, alpha=0.5)

        # Trace orbit from median point
        x_start = norm["X_normalized"][len(X) // 2]
        n_steps = 200
        eps = 0.02
        orb = generator_orbit(x_start, g, n_steps, eps, winner_type)
        orb_orig = norm["scaler_X"].inverse_transform(orb)
        ax.plot(np.log10(np.abs(orb_orig[:, d0]) + 1e-12),
                np.log10(np.abs(orb_orig[:, d1]) + 1e-12),
                color="red", lw=2, label="scaling orbit")

        ax.set_xlabel(f"log₁₀({VARIABLE_NAMES[d0]})")
        ax.set_ylabel(f"log₁₀({VARIABLE_NAMES[d1]})")
        ax.set_title(f"Scaling Orbit (log-space)")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, f"Orbit visualization\nfor {winner_type} symmetry",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Discovered Orbit")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(output_dir, "keyhole_symmetry_discovery.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {plot_path}")

    # --- Summary text ---
    summary_path = os.path.join(output_dir, "discovery_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Keyhole Welding — Symmetry Discovery Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Samples: {X.shape[0]}\n")
        f.write(f"Variables: {VARIABLE_NAMES}\n")
        f.write(f"Output: e* (keyhole eccentricity)\n\n")
        f.write(f"Optimal latent dimension: {latent['optimal_n_latent']}\n")
        for k, m in latent["metrics"].items():
            f.write(f"  k={k}: R2={m['R2']:.4f}\n")
        f.write(f"\nDiscovered symmetry: {sym_res['symmetry_type']}\n")
        for t, l in sorted(sym_res["losses"].items(), key=lambda kv: kv[1]):
            f.write(f"  {t}: MSE={l:.6f}\n")
        f.write(f"\nEncoder weights W:\n")
        for i in range(W.shape[0]):
            f.write(f"  z{i+1}: {np.round(W[i], 4).tolist()}\n")
        f.write(f"\nGenerators: {len(generators)}\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                f.write(f"  g{i+1}: {np.round(g, 4).tolist()}\n")
            else:
                f.write(f"  g{i+1}:\n")
                for row in np.round(g, 4):
                    f.write(f"    {row.tolist()}\n")
    print(f"Summary saved to {summary_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Discover symmetry in keyhole welding data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", default="dataset_keyhole.csv",
                        help="Path to keyhole CSV data file")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic keyhole-like data")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--latent-epochs", type=int, default=600,
                        help="Training epochs for latent dimension discovery")
    parser.add_argument("--sym-epochs", type=int, default=1500,
                        help="Training epochs for symmetry identification")
    parser.add_argument("--n-restarts", type=int, default=3,
                        help="Number of random restarts per model")
    parser.add_argument("--output-dir", default="output_keyhole_symmetry",
                        help="Directory for output figures and summary")
    args = parser.parse_args()

    # Load data
    X, y = load_data(args)

    # Run pipeline
    results = run_pipeline(X, y, args)

    # Visualize
    print("=" * 60)
    print("Step 6: Creating visualizations")
    print("=" * 60)
    plot_results(X, y, results, args.output_dir)

    # Final summary
    print()
    print("=" * 60)
    print("DISCOVERY COMPLETE")
    print("=" * 60)
    sym_type = results["symmetry"]["symmetry_type"]
    n_gens = len(results["generators"])
    print(f"  Symmetry type:     {sym_type}")
    print(f"  Generators found:  {n_gens}")
    if sym_type == "scaling":
        print(f"  Interpretation:    Buckingham Pi scaling symmetry")
        print(f"                     Power-law dimensionless groups govern e*")
        print(f"                     Generators = unit-rescaling directions that")
        print(f"                     preserve the dimensionless output")
    elif sym_type == "rotational":
        print(f"  Interpretation:    Rotational invariance in variable space")
    elif sym_type == "translational":
        print(f"  Interpretation:    Translational invariance in variable space")
    W = results["winner_encoder"].weight_matrix
    print(f"  Encoder weights:   {W.shape[0]} latent dims x {W.shape[1]} variables")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
