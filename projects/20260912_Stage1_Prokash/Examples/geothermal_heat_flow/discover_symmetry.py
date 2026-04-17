"""
Discover the hidden scaling symmetry in Antarctic geothermal heat-flow
data using the PyDimension Stage1 pipeline.

Physics
-------
Steady-state conductive heat flow through rock or sediment obeys
Fourier's law:

    Q = k · G

where Q is the surface heat flow (mW/m²), k is the thermal conductivity
(W/(m·K)), and G is the vertical temperature gradient (°C/km, converted
to K/m internally).  The dataset also records the probe penetration
depth z (m) and water depth d (m), which do NOT enter Fourier's law.

With four input variables {k, G, z, d} having three independent
dimensions (W = kg·m²·s⁻³, K, m) the Buckingham Pi theorem gives
4 − 3 = 1 independent dimensionless group controlling Q:

    Pi_1 = Q / (k · G)   ≡ 1   (Fourier's law)

plus a purely geometric ratio Pi_2 = z / d that is irrelevant to Q.

The scaling symmetry is: any simultaneous rescaling of k and G that
preserves k·G also preserves Q; z and d are free (they don't appear
in Fourier's law at all).

This script uses the Stage1 pipeline to:
  1. Confirm that Q depends on a single latent coordinate (k·G),
  2. Confirm the symmetry type is **scaling** (competitive loss),
  3. Extract Lie-algebra generators — null-space of the log-space
     encoder W — revealing z and d as free directions,
  4. Produce a 3-panel figure (Fourier collapse, symmetry-type bars,
     generator orbits with known-Fourier reference).

Dataset
-------
Antarctic Geothermal Heat Flow Database (Dziadek et al., 2021).
210 measurements with columns: station, lat, lon, k, G, z, d, Q,
quality, method.

Usage
-----
    python discover_symmetry.py --data dataset_ghf.csv
    python discover_symmetry.py --data dataset_ghf.csv --encoder-hidden 64 32
    python discover_symmetry.py --data dataset_ghf.csv --log-normalize
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
    print(f"Copy preprocessing/, intrinsic_coordinate/, symmetry_discovery/ from")
    print(f"projects/20260912_Stage1_Prokash/ into the same directory as this script.")
    sys.exit(1)

import torch.multiprocessing as _tmp
_tmp.cpu_count = lambda: 0

VARIABLE_NAMES = ["k", "G", "z", "d"]
VARIABLE_UNITS = ["W/(m·K)", "°C/km", "m", "m"]

# Fourier's law: Q = k^1 · G^1 · z^0 · d^0
KNOWN_Q_EXPONENTS = np.array([1.0, 1.0, 0.0, 0.0])


def compute_fourier(X: np.ndarray) -> np.ndarray:
    """Compute Fourier heat flow Q = k * G from raw physical variables."""
    k, G = X[:, 0], X[:, 1]
    return k * G


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_data(csv_path: str) -> dict:
    """Load geothermal heat-flow data from CSV."""
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
    for target in ["Q", "q", "heat_flow", "HF"]:
        for i, h in enumerate(header):
            if h.strip() == target:
                output_col = i
                break
        if output_col is not None:
            break

    if len(input_cols) != len(VARIABLE_NAMES):
        raise ValueError(
            f"Expected {len(VARIABLE_NAMES)} input variables {VARIABLE_NAMES}, "
            f"found {len(input_cols)} in: {header}"
        )
    if output_col is None:
        raise ValueError(f"Could not find output column (Q/heat_flow) in: {header}")

    X_list, y_list = [], []
    for r in rows:
        try:
            vals = [float(r[c]) for c in input_cols]
            out = float(r[output_col])
            if all(v > 0 for v in vals) and out > 0:
                X_list.append(vals)
                y_list.append(out)
        except (ValueError, IndexError):
            continue
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Loaded: {[header[i].strip() for i in input_cols]} -> {header[output_col].strip()}")
    return {"X": X, "y": y}


def load_data(args):
    """Load data from CSV, compute Fourier Q, return (X, y, Q_pred)."""
    data_path = args.data
    if not os.path.exists(data_path):
        data_path = os.path.join(_here, os.path.basename(args.data))
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {args.data}")
        print(f"Place your geothermal CSV (with columns {VARIABLE_NAMES} + Q)")
        print(f"in {_here}/ and run:")
        print(f"  python discover_symmetry.py --data <your_file.csv>")
        sys.exit(1)

    print(f"Loading geothermal heat-flow data from {data_path}...")
    data = load_csv_data(data_path)

    X, y = data["X"], data["y"]
    Q_pred = compute_fourier(X)

    mask = np.isfinite(Q_pred) & (Q_pred > 0)
    dropped = (~mask).sum()
    if dropped:
        print(f"  Dropping {dropped} rows with non-positive k*G")
        X, y, Q_pred = X[mask], y[mask], Q_pred[mask]

    print(f"  Samples: {X.shape[0]}")
    print(f"  k·G range: [{Q_pred.min():.4g}, {Q_pred.max():.4g}]")
    print(f"  Q range: [{y.min():.4g}, {y.max():.4g}]")
    rel_err = np.abs(y - Q_pred) / (y + 1e-12)
    print(f"  Fourier verification: median |Q - k·G|/Q = {np.median(rel_err):.4f}")
    print()
    return X, y, Q_pred


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(X, y, Q_pred, args):
    """Run Stage1 symmetry discovery on the geothermal physical variables."""
    results = {"Q_pred": Q_pred, "X_raw": X}

    # --- Normalize ---
    print("=" * 60)
    print("Step 1: Normalizing data")
    print("=" * 60)
    sys.stdout.flush()

    if getattr(args, "log_normalize", False):
        log10_X = np.log10(np.maximum(X, 1e-30))
        gmean_exp = log10_X.mean(axis=0)
        X_prescaled = 10 ** (log10_X - gmean_exp)
        norm = normalize_data(X_prescaled, y, method="minmax")
        norm["log_prescaled"] = True
        norm["gmean_exp"] = gmean_exp
        print(f"  Log-prenormalisation enabled (geometric-mean centring)")
        print(f"  X_prescaled range: [{X_prescaled.min():.3g}, {X_prescaled.max():.3g}]")
    else:
        norm = normalize_data(X, y, method="minmax")
        norm["log_prescaled"] = False

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
    if getattr(args, "fourier_basis", False):
        q_raw = compute_fourier(X)
        log_q = np.log10(np.maximum(q_raw, 1e-30))
        log_q_n = (log_q - log_q.min()) / (log_q.max() - log_q.min() + 1e-12)
        enc_kwargs["pi_features"] = log_q_n.reshape(-1, 1)
        print(f"  Injecting precomputed log10(k·G) feature "
              f"(raw k·G range: {q_raw.min():.3g} .. {q_raw.max():.3g})")

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
    W = winner_encoder.weight_matrix
    print("=" * 60)
    print("  Winning encoder weight vector(s)")
    print("=" * 60)
    for i in range(W.shape[0]):
        row = W[i]
        denom = np.linalg.norm(row) + 1e-12
        row_n = row / denom
        print(f"  Row {i+1} (scaling):")
        header_str = "    " + "  ".join(f"{n:>7s}" for n in VARIABLE_NAMES)
        raw_str    = "    " + "  ".join(f"{v:+7.4f}" for v in row)
        normed_str = "    " + "  ".join(f"{v:+7.4f}" for v in row_n)
        print(header_str)
        print(f"  raw :{raw_str}")
        print(f"  L2-n:{normed_str}")
        ref = KNOWN_Q_EXPONENTS.copy()
        ref_n = ref / (np.linalg.norm(ref) + 1e-12)
        cos = float(np.dot(row_n, ref_n))
        print(f"  cos<row, known-Fourier-exponents> = {cos:+.4f}  "
              f"(±1 means perfect alignment)")
    print()

    # --- Interpret generators physically ---
    print("=" * 60)
    print("Step 5: Physical interpretation of generators")
    print("=" * 60)
    if winner_type == "scaling" and generators:
        print(f"  Each generator is a direction in log-space along which k·G is preserved.")
        print(f"  Physically: simultaneous rescaling of variables that keeps the")
        print(f"  heat flow Q = k·G invariant.\n")
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
        print(f"             to keep Q = k·G (heat flow) unchanged")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (3 panels: Fourier collapse, symmetry losses, generator orbits)
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(X, y, results, output_dir):
    """Create a focused 3-panel figure."""
    os.makedirs(output_dir, exist_ok=True)
    Q_pred = results["Q_pred"]
    generators = results["generators"]
    winner_type = results["winner_type"]
    sym_res = results["symmetry"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Geothermal Heat Flow — Scaling Symmetry Discovery",
                 fontsize=15, fontweight="bold")

    # --- Panel 1: Fourier collapse Q vs k·G ---
    ax = axes[0]
    log_kG = np.log10(Q_pred)
    log_Q = np.log10(y)
    ax.scatter(log_kG, log_Q, c="#4C72B0", s=20, alpha=0.6, edgecolors="none")
    # Perfect Fourier line: log Q = log(k·G)
    fit_range = np.linspace(min(log_kG.min(), log_Q.min()),
                            max(log_kG.max(), log_Q.max()), 200)
    ax.plot(fit_range, fit_range, "r-", lw=2, label="Q = k·G (Fourier)")
    # R² of the identity collapse
    ss_res = np.sum((log_Q - log_kG) ** 2)
    ss_tot = np.sum((log_Q - log_Q.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    ax.set_xlabel(r"$\log_{10}(k \cdot G)$", fontsize=11)
    ax.set_ylabel(r"$\log_{10}(Q)$  [mW/m²]", fontsize=11)
    ax.set_title(f"Fourier collapse   (R² = {r2:.3f})", fontsize=12)
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
        g = generators[0]
        importance = np.abs(g)
        top2 = np.argsort(importance)[-2:][::-1]
        d0, d1 = top2[0], top2[1]

        logX0 = np.log10(np.maximum(X[:, d0], 1e-30))
        logX1 = np.log10(np.maximum(X[:, d1], 1e-30))
        sc = ax.scatter(logX0, logX1, c=y, cmap="plasma",
                        s=22, alpha=0.9, edgecolors="black", linewidth=0.3)
        fig.colorbar(sc, ax=ax, label="Q [mW/m²]", fraction=0.046, pad=0.04)

        x_lo, x_hi = logX0.min(), logX0.max()
        y_lo, y_hi = logX1.min(), logX1.max()
        x_pad = 0.1 * (x_hi - x_lo + 1e-9)
        y_pad = 0.15 * (y_hi - y_lo + 1e-9)
        x_line = np.linspace(x_lo - x_pad, x_hi + x_pad, 2)

        # ---- Reference: iso-Q contours from KNOWN Fourier's law ----------
        q_exp = KNOWN_Q_EXPONENTS
        if abs(q_exp[d1]) > 1e-12:
            log10_Q = np.log10(np.maximum(y, 1e-30))
            levels = np.linspace(log10_Q.min(), log10_Q.max(), 6)
            log10_X_all = np.log10(np.maximum(X, 1e-30))
            gmean_log = log10_X_all.mean(axis=0)
            other_mask = np.ones(len(q_exp), dtype=bool)
            other_mask[d0] = other_mask[d1] = False
            const = float(np.dot(q_exp[other_mask], gmean_log[other_mask]))
            ref_slope = -q_exp[d0] / q_exp[d1]
            first = True
            for lev in levels:
                y_ref = (lev - const - q_exp[d0] * x_line) / q_exp[d1]
                ax.plot(x_line, y_ref,
                        color="grey", ls="--", lw=1.0, alpha=0.55,
                        label="known-Fourier iso-contour" if first else None,
                        zorder=1)
                first = False
        else:
            ref_slope = float("nan")

        # ---- Discovered iso-invariant lines from scaling generator -------
        orbit_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        rng = np.random.default_rng(42)
        pts2d = np.column_stack([logX0, logX1])
        uniq, uniq_idx = np.unique(np.round(pts2d, 4), axis=0, return_index=True)
        start_indices = rng.choice(uniq_idx, min(4, len(uniq_idx)), replace=False)

        if abs(g[d0]) < 1e-12:
            slope = float("inf")
            for ki, idx in enumerate(start_indices):
                ax.axvline(logX0[idx],
                           color=orbit_colors[ki % len(orbit_colors)],
                           lw=2.2, alpha=0.9,
                           label=f"discovered orbit {ki+1}" if ki < 3 else None,
                           zorder=3)
        else:
            slope = g[d1] / g[d0]
            for ki, idx in enumerate(start_indices):
                y_line = logX1[idx] + slope * (x_line - logX0[idx])
                ax.plot(x_line, y_line,
                        color=orbit_colors[ki % len(orbit_colors)],
                        lw=2.2, alpha=0.95,
                        label=f"discovered orbit {ki+1}" if ki < 3 else None,
                        zorder=3)
                mid_x = logX0[idx]
                mid_y = logX1[idx]
                tvec = np.array([1.0, slope])
                tvec /= np.linalg.norm(tvec) + 1e-12
                arr_len = 0.15 * (x_hi - x_lo + 1e-9)
                ax.annotate(
                    "",
                    xy=(mid_x + arr_len * tvec[0], mid_y + arr_len * tvec[1]),
                    xytext=(mid_x, mid_y),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=orbit_colors[ki % len(orbit_colors)],
                        lw=2.0, shrinkA=0, shrinkB=0,
                    ),
                    zorder=4,
                )

        # ---- Restricted encoder orbit ----
        W = results["winner_encoder"].weight_matrix
        w = W[0] if W.ndim == 2 else W
        if abs(w[d1]) > 1e-12:
            restricted_slope = -w[d0] / w[d1]
            cx = float(logX0.mean())
            cy = float(logX1.mean())
            y_restr = cy + restricted_slope * (x_line - cx)
            ax.plot(x_line, y_restr,
                    color="black", ls="-.", lw=2.0, alpha=0.75,
                    label=f"encoder restricted (slope {restricted_slope:+.2f})",
                    zorder=2)
        else:
            restricted_slope = float("nan")

        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)

        slope_str = (f"{slope:+.2f}" if np.isfinite(slope) else "vertical")
        ref_str   = (f"{ref_slope:+.2f}" if np.isfinite(ref_slope) else "—")
        ax.set_xlabel(
            f"log₁₀({VARIABLE_NAMES[d0]})  [{VARIABLE_UNITS[d0]}]",
            fontsize=11,
        )
        ax.set_ylabel(
            f"log₁₀({VARIABLE_NAMES[d1]})  [{VARIABLE_UNITS[d1]}]",
            fontsize=11,
        )
        ax.set_title(
            f"Iso-invariant lines   discovered slope = {slope_str}   "
            f"(known Fourier: {ref_str})",
            fontsize=11,
        )
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
    else:
        ax.text(0.5, 0.5, f"No scaling orbits\n(detected: {winner_type})",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Generator Orbits")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, "geothermal_symmetry_discovery.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Discover scaling symmetry in geothermal heat-flow data"
    )
    parser.add_argument("--data", default="dataset_ghf.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-epochs", type=int, default=600)
    parser.add_argument("--sym-epochs", type=int, default=1500)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output-dir", default="output_geothermal_symmetry")
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=None,
                        help="Hidden layer widths for multi-layer encoder")
    parser.add_argument("--fourier-basis", action="store_true",
                        help="Augment encoder input with known k·G product")
    parser.add_argument("--log-normalize", action="store_true",
                        help="Geometric-mean centre each column before scaling")
    args = parser.parse_args()

    X, y, Q_pred = load_data(args)
    results = run_pipeline(X, y, Q_pred, args)

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
        print(f"  These generators show how k, G, z, d can be")
        print(f"  simultaneously rescaled while preserving Q = k·G")
        print(f"  — Fourier's law of conductive heat flow.")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
