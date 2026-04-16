"""
Discover the hidden scaling symmetry in Laser Powder Bed Fusion (LPBF)
porosity data using the PyDimension Stage1 pipeline.

Physics
-------
LPBF additive manufacturing defects — in particular the pore fraction `f`
left in the solidified track — are empirically controlled by a single
dimensionless "normalised enthalpy" group.  The same quantity was
extracted by hand in the accompanying notebook ``4_plot_3d-ZGAN(1).ipynb``
to collapse porosity curves from five different alloys (Al2024, Al6061,
Cu, SS304, Ti64) onto one master logistic curve:

    Pi = (Lv * rho * A * P * V) / (k^2 * (Tb - Tm)^2)

with seven physical inputs:

    | Variable                | Symbol | Units     | Dimensions        |
    |-------------------------|--------|-----------|-------------------|
    | Laser power             | P      | W         | kg·m²·s⁻³         |
    | Scan speed              | V      | m/s       | m·s⁻¹             |
    | Absorptivity            | A      | —         | dimensionless     |
    | Metal density           | rho    | kg/m³     | kg·m⁻³            |
    | Thermal conductivity    | k      | W/(m·K)   | kg·m·s⁻³·K⁻¹      |
    | Latent heat of vapour.  | Lv     | J/kg      | m²·s⁻²            |
    | Superheat (Tb-Tm)       | dT     | K         | K                 |

Four fundamental dimensions (M, L, T, K) in seven inputs give
``7 - 4 = 3`` independent dimensionless groups (Buckingham Pi).  ``Pi``
is one of them; for a single-output porosity problem one group suffices
to capture the leading-order collapse seen in the notebook.

The scaling symmetry is therefore: any continuous rescaling of
(P, V, A, rho, k, Lv, dT) that leaves ``Pi`` unchanged must also leave
the pore fraction unchanged.  The Stage1 pipeline recovers this
invariance directly from the experimental data without being told the
formula for ``Pi``.

This script uses the Stage1 pipeline to:
  1. Confirm that pore fraction ``f`` depends on a single latent (the
     normalised enthalpy ``Pi``),
  2. Confirm the symmetry type is **scaling** (competitive loss),
  3. Extract the Lie-algebra generators of the scaling group, which span
     the null-space of the linear log-space encoder — the simultaneous
     unit rescalings that preserve ``Pi`` and hence porosity,
  4. Produce a focussed 3-panel figure (Pi-collapse, symmetry-type bars,
     generator orbits).

Usage
-----
    python discover_symmetry.py --data dataset_lpbf.csv
    python discover_symmetry.py --data dataset_lpbf.csv --encoder-hidden 64 32
    python discover_symmetry.py --data dataset_lpbf.csv --pi-basis
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

VARIABLE_NAMES = ["P", "V", "A", "rho", "k", "Lv", "dT"]
VARIABLE_UNITS = ["W", "m/s", "-", "kg/m³", "W/(m·K)", "J/kg", "K"]

# Pi = Lv^1 * rho^1 * A^1 * P^1 * V^1 * k^-2 * dT^-2
KNOWN_PI_EXPONENTS = np.array([1.0, 1.0, 1.0, 1.0, -2.0, 1.0, -2.0])


def compute_pi(X: np.ndarray) -> np.ndarray:
    """Compute the notebook's normalised-enthalpy Pi from 7 physical variables."""
    P, V, A, rho, k, Lv, dT = [X[:, i] for i in range(7)]
    return (Lv * rho * A * P * V) / (k ** 2 * dT ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_data(csv_path: str) -> dict:
    """Load LPBF porosity data from CSV (columns P, V, A, rho, k, Lv, dT, Pore)."""
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
    for target in ["Pore", "pore", "pore_fraction", "f", "porosity"]:
        for i, h in enumerate(header):
            if h.strip() == target:
                output_col = i
                break
        if output_col is not None:
            break

    if len(input_cols) != 7:
        raise ValueError(
            f"Expected 7 input variables {VARIABLE_NAMES}, "
            f"found {len(input_cols)} in: {header}"
        )
    if output_col is None:
        raise ValueError(f"Could not find output column (Pore/porosity) in: {header}")

    X_list, y_list = [], []
    for r in rows:
        try:
            X_list.append([float(r[c]) for c in input_cols])
            y_list.append(float(r[output_col]))
        except (ValueError, IndexError):
            continue
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Loaded: {[header[i].strip() for i in input_cols]} -> {header[output_col].strip()}")
    return {"X": X, "y": y}


def load_data(args):
    """Load data from CSV, compute Pi, return (X, y, Pi)."""
    data_path = args.data
    if not os.path.exists(data_path):
        # Try relative to script directory
        data_path = os.path.join(_here, os.path.basename(args.data))
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {args.data}")
        print(f"Place your LPBF CSV (with columns {VARIABLE_NAMES} + Pore)")
        print(f"in {_here}/ and run:")
        print(f"  python discover_symmetry.py --data <your_file.csv>")
        sys.exit(1)

    print(f"Loading LPBF porosity data from {data_path}...")
    data = load_csv_data(data_path)

    X, y = data["X"], data["y"]
    Pi = compute_pi(X)

    # Remove rows with non-positive / non-finite Pi (log10 will be used later)
    mask = np.isfinite(Pi) & (Pi > 0)
    dropped = (~mask).sum()
    if dropped:
        print(f"  Dropping {dropped} rows with non-positive Pi")
        X, y, Pi = X[mask], y[mask], Pi[mask]

    # Clamp Pore fraction to [0, 1]
    y = np.clip(y, 0.0, 1.0)

    print(f"  Samples: {X.shape[0]}")
    print(f"  Pi range: [{Pi.min():.4g}, {Pi.max():.4g}]")
    print(f"  Pore range: [{y.min():.4f}, {y.max():.4f}]")
    print()
    return X, y, Pi


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(X, y, Pi, args):
    """Run Stage1 symmetry discovery on the LPBF physical variables."""
    results = {"Pi": Pi}

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
    enc_kwargs = {}
    if getattr(args, "encoder_hidden", None):
        enc_kwargs["encoder_hidden_dims"] = args.encoder_hidden
    if getattr(args, "pi_basis", False):
        # NB: we cannot use ``pi_basis_vectors`` here because the library
        # applies it to the *normalised* X, which has exact zeros after
        # min-max scaling (LPBF material-property columns only take 5
        # discrete values, so entire material groups map to 0).  Taking
        # log(0) and raising to negative exponents would blow up to NaN.
        #
        # Instead: compute the Pi feature from the *raw* positive X, take
        # log10, min-max-scale it to [0, 1] so it lives on the same scale
        # as the other encoder inputs, and inject it through the
        # ``pi_features`` argument (which just concatenates it untouched).
        pi_raw = compute_pi(X)  # uses raw physical X — always > 0
        log_pi = np.log10(np.maximum(pi_raw, 1e-30))
        log_pi_n = (log_pi - log_pi.min()) / (log_pi.max() - log_pi.min() + 1e-12)
        enc_kwargs["pi_features"] = log_pi_n.reshape(-1, 1)
        print(f"  Injecting precomputed log10(Pi) feature "
              f"(raw Pi range: {pi_raw.min():.3g} .. {pi_raw.max():.3g})")

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

    # --- Report the winning encoder's weight vector ---
    # The scaling encoder computes z = W · log(X.clamp(min=0.1)) with X the
    # normalised (not augmented) input, so W has shape (n_latent, 7) and its
    # columns correspond 1:1 to VARIABLE_NAMES.  Printing both the raw weights
    # and their L2-normalised version lets the reader compare to the known
    # Pi exponents [+1, +1, +1, +1, −2, +1, −2].
    W = winner_encoder.weight_matrix  # (n_latent, n_inputs)
    print("=" * 60)
    print("  Winning encoder weight vector(s)")
    print("=" * 60)
    for i in range(W.shape[0]):
        row = W[i]
        denom = np.linalg.norm(row) + 1e-12
        row_n = row / denom
        print(f"  Row {i+1} (scaling):")
        header = "    " + "  ".join(f"{n:>7s}" for n in VARIABLE_NAMES)
        raw    = "    " + "  ".join(f"{v:+7.4f}" for v in row)
        normed = "    " + "  ".join(f"{v:+7.4f}" for v in row_n)
        print(header)
        print(f"  raw :{raw}")
        print(f"  L2-n:{normed}")
        # Compare direction against known Pi exponents
        ref = np.array([+1.0, +1.0, +1.0, +1.0, -2.0, +1.0, -2.0])
        ref_n = ref / np.linalg.norm(ref)
        cos = float(np.dot(row_n, ref_n))
        print(f"  cos<row, known-Pi-exponents> = {cos:+.4f}  "
              f"(±1 means perfect alignment)")
    print()

    # --- Interpret generators physically ---
    print("=" * 60)
    print("Step 5: Physical interpretation of generators")
    print("=" * 60)
    if winner_type == "scaling" and generators:
        print(f"  Each generator is a direction in log-space along which Pi is preserved.")
        print(f"  Physically: simultaneous rescaling of variables that keeps the")
        print(f"  normalised enthalpy Pi (and therefore the pore fraction) invariant.\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = []
                for j, name in enumerate(VARIABLE_NAMES):
                    if abs(g[j]) > 0.05:
                        parts.append(f"{name} x exp({g[j]:+.3f}*eps)")
                print(f"  Generator {i+1}:")
                print(f"    {', '.join(parts)}")
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
    abs_g = np.abs(g)
    dominant = np.argmax(abs_g)
    name = VARIABLE_NAMES[dominant]

    coupled = [(VARIABLE_NAMES[j], g[j]) for j in range(len(g))
               if j != dominant and abs(g[j]) > 0.05]

    if coupled:
        direction = "increase" if g[dominant] > 0 else "decrease"
        compensations = []
        for cname, cval in coupled:
            cdirection = "increase" if cval > 0 else "decrease"
            compensations.append(f"{cdirection} {cname}")
        print(f"    Meaning: {direction} {name} while {', '.join(compensations)}")
        print(f"             to keep Pi (and pore fraction) unchanged")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (3 panels: Pi vs Pore, symmetry losses, generator orbits)
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(X, y, results, output_dir):
    """Create a focused 3-panel figure."""
    os.makedirs(output_dir, exist_ok=True)
    Pi = results["Pi"]
    generators = results["generators"]
    winner_type = results["winner_type"]
    norm = results["normalization"]
    sym_res = results["symmetry"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("LPBF Porosity — Symmetry Discovery", fontsize=15, fontweight="bold")

    # --- Panel 1: Known Pi vs Pore fraction ---
    ax = axes[0]
    logPi = np.log10(Pi)
    ax.scatter(logPi, y, c="#4C72B0", s=20, alpha=0.6, edgecolors="none")
    # Fit a simple logistic to show collapse
    try:
        from scipy.optimize import curve_fit
        def _logistic(x, k, x0):
            return 1.0 / (1.0 + np.exp(-k * (x - x0)))
        order = np.argsort(logPi)
        x0_init = logPi[order][np.argmin(np.abs(y[order] - 0.5))]
        popt, _ = curve_fit(_logistic, logPi, y,
                            p0=[4.0, x0_init],
                            bounds=([0.1, logPi.min() - 2], [50.0, logPi.max() + 5]),
                            maxfev=10000)
        x_fit = np.linspace(logPi.min(), logPi.max(), 200)
        y_fit = _logistic(x_fit, *popt)
        ss_res = np.sum((y - _logistic(logPi, *popt)) ** 2)
    except Exception:
        coeffs = np.polyfit(logPi, y, 2)
        x_fit = np.linspace(logPi.min(), logPi.max(), 200)
        y_fit = np.polyval(coeffs, x_fit)
        ss_res = np.sum((y - np.polyval(coeffs, logPi)) ** 2)
    ax.plot(x_fit, y_fit, "r-", lw=2, label="logistic fit")
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    ax.set_xlabel(r"$\log_{10}(\Pi)$ — normalised enthalpy", fontsize=11)
    ax.set_ylabel("Pore fraction", fontsize=11)
    ax.set_title(f"Pi-collapse   (R² = {r2:.3f})", fontsize=12)
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
    #
    # A scaling symmetry acts multiplicatively on X, equivalently additively
    # on log(X).  Its orbits are therefore STRAIGHT LINES in log-log space,
    # with slope s = g[d1] / g[d0] set by the null-space generator g.
    #
    # The earlier implementation traced the orbit in minmax-*normalised*
    # coordinates and inverse-transformed back — because minmax is affine
    # and not multiplicative, an exp(ε g) step in normalised space bends
    # into a curve in log(X_raw).  Here we instead draw the iso-invariant
    # lines directly in log10(X_raw), rooted at data points.  That is the
    # honest geometric picture of the discovered scaling generator.
    ax = axes[2]
    if generators and winner_type == "scaling":
        g = generators[0]
        importance = np.abs(g)
        top2 = np.argsort(importance)[-2:][::-1]
        d0, d1 = top2[0], top2[1]

        # Scatter the data (log10 of raw physical X) coloured by pore fraction
        logX0 = np.log10(np.maximum(X[:, d0], 1e-30))
        logX1 = np.log10(np.maximum(X[:, d1], 1e-30))
        sc = ax.scatter(logX0, logX1, c=y, cmap="plasma",
                        s=18, alpha=0.75, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="Pore fraction", fraction=0.046, pad=0.04)

        # Slope in (log10 X[d0], log10 X[d1]) space from the generator:
        # log(X_new) = log(X) + ε · g, so in log10 space the direction is g/ln(10).
        # The line slope only needs the ratio g[d1] / g[d0].
        x_lo, x_hi = logX0.min(), logX0.max()
        x_pad = 0.1 * (x_hi - x_lo + 1e-9)
        x_line = np.linspace(x_lo - x_pad, x_hi + x_pad, 2)

        orbit_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        rng = np.random.default_rng(42)
        # Pick starting points spread across the data, avoiding duplicates in
        # the (d0, d1) plane (LPBF material-property columns take discrete values).
        pts2d = np.column_stack([logX0, logX1])
        uniq, uniq_idx = np.unique(np.round(pts2d, 4), axis=0, return_index=True)
        start_indices = rng.choice(uniq_idx, min(4, len(uniq_idx)), replace=False)

        if abs(g[d0]) < 1e-12:
            # Vertical iso-line: constant log(X[d0])
            for k, idx in enumerate(start_indices):
                ax.axvline(logX0[idx],
                           color=orbit_colors[k % len(orbit_colors)],
                           lw=2, alpha=0.85,
                           label=f"orbit {k+1}" if k < 3 else None)
        else:
            slope = g[d1] / g[d0]
            for k, idx in enumerate(start_indices):
                y_line = logX1[idx] + slope * (x_line - logX0[idx])
                ax.plot(x_line, y_line,
                        color=orbit_colors[k % len(orbit_colors)],
                        lw=2, alpha=0.85,
                        label=f"orbit {k+1}" if k < 3 else None)
            # Clip y-axis to the data range + a small pad so the lines don't
            # wander off and hide the scatter.
            y_lo, y_hi = logX1.min(), logX1.max()
            y_pad = 0.15 * (y_hi - y_lo + 1e-9)
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
            ax.set_xlim(x_lo - x_pad, x_hi + x_pad)

        # Annotate the generator direction in the panel title
        slope_str = (f"slope = {g[d1] / g[d0]:+.2f}"
                     if abs(g[d0]) > 1e-12 else "vertical")
        ax.set_xlabel(
            f"log₁₀({VARIABLE_NAMES[d0]})  [{VARIABLE_UNITS[d0]}]",
            fontsize=11,
        )
        ax.set_ylabel(
            f"log₁₀({VARIABLE_NAMES[d1]})  [{VARIABLE_UNITS[d1]}]",
            fontsize=11,
        )
        ax.set_title(f"Iso-invariant lines  ({slope_str})", fontsize=12)
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, f"No scaling orbits\n(detected: {winner_type})",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Generator Orbits")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, "lpbf_porosity_symmetry_discovery.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Discover symmetry in LPBF porosity data")
    parser.add_argument("--data", default="dataset_lpbf.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-epochs", type=int, default=600)
    parser.add_argument("--sym-epochs", type=int, default=1500)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output-dir", default="output_lpbf_porosity_symmetry")
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=None,
                        help="Hidden layer widths for multi-layer encoder (e.g. --encoder-hidden 64 32)")
    parser.add_argument("--pi-basis", action="store_true",
                        help="Augment encoder input with known Pi group (LPBF normalised enthalpy)")
    args = parser.parse_args()

    X, y, Pi = load_data(args)
    results = run_pipeline(X, y, Pi, args)

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
        print(f"  These generators show how P, V, A, rho, k, Lv, dT can be")
        print(f"  simultaneously rescaled while preserving the normalised enthalpy Pi")
        print(f"  — and therefore the LPBF pore fraction.")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
