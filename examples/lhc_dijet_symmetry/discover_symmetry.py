"""
Discover symmetry in LHC dijet events using the PyDimension Stage1 pipeline.

Physics
-------
In proton-proton collisions at the LHC, dijet events exhibit azimuthal
rotational symmetry: rotating both jets' transverse momenta by an angle phi
in the transverse plane preserves all Lorentz-invariant observables (e.g.
the dijet invariant mass).

    Input:  X = (p1x, p1y, p2x, p2y)   — leading dijet px/py components
    Output: y = m_jj_T                   — dijet transverse invariant mass

The expected symmetry is SO(2) rotation with generator:

        [ 0 -1  0  0 ]
    A = [ 1  0  0  0 ]   (simultaneous rotation of both jets)
        [ 0  0  0 -1 ]
        [ 0  0  1  0 ]

Pipeline
--------
    1. Load prepared data (or generate synthetic LHC-like data as fallback)
    2. Compute output: dijet transverse mass m_jj_T
    3. Normalize data
    4. Discover latent dimension (autoencoder sweep)
    5. Identify symmetry type (competitive encoder training)
    6. Extract Lie-algebra generators
    7. Validate: check invariance of y under discovered transformation

Usage
-----
    # With real LHC data (run prepare_data.py first):
    python discover_symmetry.py --data lhc_dijet_data.pt

    # With synthetic LHC-like data (no external files needed):
    python discover_symmetry.py --synthetic
"""

import sys
import os
import argparse
import traceback

import numpy as np
import torch
import torch.multiprocessing

# On Windows, multiprocessing workers can cause silent crashes.
# Force the safe "spawn" start method and patch cpu_count to avoid worker use.
if sys.platform == "win32":
    torch.multiprocessing.set_start_method("spawn", force=True)

# Add the Stage1 project to the path
_here = os.path.dirname(os.path.abspath(__file__))
_stage1 = os.path.join(os.path.dirname(_here), "..", "projects", "20260912_Stage1_Prokash")
sys.path.insert(0, _stage1)

try:
    import matplotlib
    matplotlib.use("Agg")
except (AttributeError, ImportError):
    pass
import matplotlib.pyplot as plt

from preprocessing.normalize import normalize_data
from intrinsic_coordinate.discovery import discover_latent_dimension
from symmetry_discovery.identification import identify_symmetry
from symmetry_discovery.generators import extract_generators, generator_orbit


# ──────────────────────────────────────────────────────────────────────────────
# 1. Data loading / generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_lhc_data(n_samples: int = 5000, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic data mimicking LHC dijet kinematics.

    Each event is two back-to-back jets in the transverse plane with realistic
    pT spectrum (exponentially falling) and approximate momentum conservation.

    Returns X with columns [p1x, p1y, p2x, p2y].
    """
    rng = np.random.default_rng(seed)

    # Realistic pT spectrum: exponential + power-law tail
    # pT ~ exp(-pT / <pT>) with <pT> ~ 100-500 GeV
    pT1 = rng.exponential(scale=200.0, size=n_samples) + 50.0
    pT2 = rng.exponential(scale=180.0, size=n_samples) + 50.0

    # Random azimuthal angle for jet 1
    phi1 = rng.uniform(-np.pi, np.pi, size=n_samples)

    # Jet 2 is approximately back-to-back (phi2 ~ phi1 + pi) with smearing
    # from ISR/FSR and underlying event
    delta_phi_smear = rng.normal(0, 0.15, size=n_samples)
    phi2 = phi1 + np.pi + delta_phi_smear

    # Convert to Cartesian
    p1x = pT1 * np.cos(phi1)
    p1y = pT1 * np.sin(phi1)
    p2x = pT2 * np.cos(phi2)
    p2y = pT2 * np.sin(phi2)

    X = np.column_stack([p1x, p1y, p2x, p2y])
    return X


def compute_dijet_mass(X: np.ndarray) -> np.ndarray:
    """
    Compute the dijet transverse invariant mass from (p1x, p1y, p2x, p2y).

    m_jj_T = sqrt(2 * pT1 * pT2 * (1 - cos(dphi)))

    This quantity is invariant under simultaneous azimuthal rotation of
    both jets — the SO(2) symmetry we aim to discover.
    """
    p1x, p1y, p2x, p2y = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    pT1 = np.sqrt(p1x**2 + p1y**2)
    pT2 = np.sqrt(p2x**2 + p2y**2)

    # cos(dphi) via dot product
    cos_dphi = (p1x * p2x + p1y * p2y) / (pT1 * pT2 + 1e-12)
    cos_dphi = np.clip(cos_dphi, -1.0, 1.0)

    m_jj_T = np.sqrt(2 * pT1 * pT2 * (1 - cos_dphi) + 1e-12)
    return m_jj_T


def load_data(args) -> tuple:
    """Load or generate data, return (X, y) as numpy arrays."""
    if args.synthetic:
        print("=" * 60)
        print("Generating synthetic LHC-like dijet data...")
        print("=" * 60)
        X = generate_synthetic_lhc_data(n_samples=args.n_samples, seed=args.seed)
    elif args.data and os.path.exists(args.data):
        print("=" * 60)
        print(f"Loading prepared LHC data from {args.data}...")
        print("=" * 60)
        X_tensor = torch.load(args.data, weights_only=True)
        X = X_tensor.cpu().numpy()
    else:
        print(f"Data file '{args.data}' not found. Using synthetic data as fallback.")
        X = generate_synthetic_lhc_data(n_samples=args.n_samples, seed=args.seed)

    print(f"  Events: {X.shape[0]}")
    print(f"  Features: {X.shape[1]} (p1x, p1y, p2x, p2y)")

    # Compute the rotationally-invariant output
    y = compute_dijet_mass(X)
    print(f"  Output: dijet transverse mass m_jj_T")
    print(f"  m_jj_T range: [{y.min():.1f}, {y.max():.1f}] GeV")
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
    norm = normalize_data(X, y, method="standard")
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

    # Loss gap
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
    print(f"  Encoder coefficients: {res_sym['coefficients']}")
    for i, g in enumerate(generators):
        if g.ndim == 1:
            print(f"  Generator {i+1} (vector): {np.round(g, 4)}")
        else:
            print(f"  Generator {i+1} (matrix):")
            print(f"    {np.round(g, 4)}")
    print()

    # --- Validate invariance ---
    print("=" * 60)
    print("Step 5: Validating symmetry (invariance check)")
    print("=" * 60)
    _validate_invariance(X, y, generators, winner_type, norm)
    print()

    return results


def _validate_invariance(X, y, generators, sym_type, norm):
    """Check that y is approximately invariant under the discovered transformation."""
    if not generators:
        print("  No generators found — skipping validation.")
        return

    g = generators[0]
    epsilons = [0.1, 0.5, 1.0]

    # Use the original (un-normalized) data for physical interpretation
    X_norm = norm["X_normalized"]
    scaler_X = norm["scaler_X"]

    for eps in epsilons:
        # Apply generator in normalized space, then map back
        n_test = min(500, len(X_norm))
        max_rel_change = 0.0

        for i in range(n_test):
            x_orig = X[i]
            x_norm = X_norm[i]

            # Transform in normalized space
            if sym_type == "translational":
                x_new_norm = x_norm + eps * g
            elif sym_type == "rotational":
                x_new_norm = x_norm + eps * (g @ x_norm)
            elif sym_type == "scaling":
                x_new_norm = x_norm * np.exp(eps * g)

            # Map back to original space
            x_new = scaler_X.inverse_transform(x_new_norm.reshape(1, -1)).ravel()

            # Compute output for transformed point
            y_orig = compute_dijet_mass(x_orig.reshape(1, -1))[0]
            y_new = compute_dijet_mass(x_new.reshape(1, -1))[0]

            rel_change = abs(y_new - y_orig) / (abs(y_orig) + 1e-12)
            max_rel_change = max(max_rel_change, rel_change)

        print(f"  eps={eps:.1f}: max |dy/y| = {max_rel_change:.4f} "
              f"({'INVARIANT' if max_rel_change < 0.1 else 'NOT invariant'})")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(X: np.ndarray, y: np.ndarray, results: dict, output_dir: str):
    """Create a summary figure of the symmetry discovery results."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("LHC Dijet Symmetry Discovery", fontsize=16, fontweight="bold")

    # --- (0,0) Input space: p1x vs p1y coloured by m_jj ---
    ax = axes[0, 0]
    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="plasma", s=4, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="$m_{jj}^T$ [GeV]")
    ax.set_xlabel("$p_{1x}$ [GeV]")
    ax.set_ylabel("$p_{1y}$ [GeV]")
    ax.set_title("Jet 1 Transverse Momenta")
    ax.set_aspect("equal")

    # --- (0,1) Input space: p2x vs p2y coloured by m_jj ---
    ax = axes[0, 1]
    sc = ax.scatter(X[:, 2], X[:, 3], c=y, cmap="plasma", s=4, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="$m_{jj}^T$ [GeV]")
    ax.set_xlabel("$p_{2x}$ [GeV]")
    ax.set_ylabel("$p_{2y}$ [GeV]")
    ax.set_title("Jet 2 Transverse Momenta")
    ax.set_aspect("equal")

    # --- (0,2) Latent dimension sweep: R2 vs k ---
    ax = axes[0, 2]
    latent = results["latent"]
    ks = sorted(latent["metrics"].keys())
    r2s = [latent["metrics"][k]["R2"] for k in ks]
    ax.plot(ks, r2s, "o-", color="#4C72B0", lw=2, markersize=8)
    ax.axhline(0.95, color="gray", ls="--", lw=1, label="R2 threshold")
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

    # --- (1,1) Discovered orbit in (p1x, p1y) plane ---
    ax = axes[1, 1]
    winner_type = results["winner_type"]
    generators = results["generators"]
    norm = results["normalization"]

    ax.scatter(X[:, 0], X[:, 1], c="lightgray", s=4, alpha=0.3)

    if generators:
        g = generators[0]
        # Trace orbit from a representative point
        x_start = norm["X_normalized"][len(X) // 2]
        scaler_X = norm["scaler_X"]

        orbit_points = []
        if winner_type == "rotational":
            n_steps = 200
            eps = 2 * np.pi / n_steps
        else:
            n_steps = 200
            eps = 0.05

        orb = generator_orbit(x_start, g, n_steps, eps, winner_type)

        # Map orbits back to original space
        orb_orig = scaler_X.inverse_transform(orb)
        orbit_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
        ax.plot(orb_orig[:, 0], orb_orig[:, 1], color=orbit_colors[0],
                lw=2, label="discovered orbit")
        ax.scatter([orb_orig[0, 0]], [orb_orig[0, 1]], color=orbit_colors[0],
                   s=60, zorder=5, marker="*")

    ax.set_xlabel("$p_{1x}$ [GeV]")
    ax.set_ylabel("$p_{1y}$ [GeV]")
    ax.set_title("Discovered Orbit (Jet 1 plane)")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)

    # --- (1,2) Discovered orbit in (p2x, p2y) plane ---
    ax = axes[1, 2]
    ax.scatter(X[:, 2], X[:, 3], c="lightgray", s=4, alpha=0.3)

    if generators and len(orb_orig[0]) >= 4:
        ax.plot(orb_orig[:, 2], orb_orig[:, 3], color=orbit_colors[1],
                lw=2, label="discovered orbit")
        ax.scatter([orb_orig[0, 2]], [orb_orig[0, 3]], color=orbit_colors[1],
                   s=60, zorder=5, marker="*")

    ax.set_xlabel("$p_{2x}$ [GeV]")
    ax.set_ylabel("$p_{2y}$ [GeV]")
    ax.set_title("Discovered Orbit (Jet 2 plane)")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(output_dir, "lhc_symmetry_discovery.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {plot_path}")

    # --- Summary text ---
    summary_path = os.path.join(output_dir, "discovery_summary.txt")
    with open(summary_path, "w") as f:
        f.write("LHC Dijet Symmetry Discovery — Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Events: {X.shape[0]}\n")
        f.write(f"Features: (p1x, p1y, p2x, p2y)\n")
        f.write(f"Output: dijet transverse mass m_jj_T\n\n")
        f.write(f"Optimal latent dimension: {latent['optimal_n_latent']}\n")
        for k, m in latent["metrics"].items():
            f.write(f"  k={k}: R2={m['R2']:.4f}\n")
        f.write(f"\nDiscovered symmetry: {sym_res['symmetry_type']}\n")
        for t, l in sorted(sym_res["losses"].items(), key=lambda kv: kv[1]):
            f.write(f"  {t}: MSE={l:.6f}\n")
        f.write(f"\nEncoder coefficients: {sym_res['coefficients']}\n")
        f.write(f"Number of generators: {len(generators)}\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                f.write(f"  Generator {i+1}: {np.round(g, 4).tolist()}\n")
            else:
                f.write(f"  Generator {i+1}:\n")
                for row in np.round(g, 4):
                    f.write(f"    {row.tolist()}\n")
    print(f"Summary saved to {summary_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Discover symmetry in LHC dijet events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", default="lhc_dijet_data.pt",
                        help="Path to prepared data tensor (.pt file)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic LHC-like data instead of real data")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Number of synthetic events to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--latent-epochs", type=int, default=600,
                        help="Training epochs for latent dimension discovery")
    parser.add_argument("--sym-epochs", type=int, default=1500,
                        help="Training epochs for symmetry identification")
    parser.add_argument("--n-restarts", type=int, default=3,
                        help="Number of random restarts per model")
    parser.add_argument("--output-dir", default="output_lhc_symmetry",
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
    if sym_type == "rotational":
        print(f"  Interpretation:    SO(2) azimuthal rotation in the transverse plane")
        print(f"                     Both jets rotate together, preserving m_jj")
    elif sym_type == "translational":
        print(f"  Interpretation:    Translational invariance in momentum space")
        print(f"                     Shift directions that leave m_jj unchanged")
    elif sym_type == "scaling":
        print(f"  Interpretation:    Scale invariance in momentum magnitudes")
    print(f"  Coefficients:      {np.round(results['symmetry']['coefficients'], 4)}")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
