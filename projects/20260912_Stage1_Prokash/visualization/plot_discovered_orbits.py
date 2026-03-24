"""
Post-discovery orbit visualization.

Runs the full Tasks 3→4→5 pipeline on 2-D datasets for each symmetry type,
then produces a 3-column × 3-row figure comparing discovered orbits against
the ground-truth orbits in input space.

Rows
----
  0 – Input space scatter (coloured by y) + ground-truth orbit (dashed)
        + discovered orbit (solid, from generator_orbit)
  1 – Discovered latent z vs y  (should collapse data to a 1-D curve)
  2 – Multiple discovered orbits from 4 starting points (coloured by start)

Usage
-----
    python projects/20260912_Stage1_Prokash/visualization/plot_discovered_orbits.py
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_proj = os.path.join(_here, "..")
sys.path.insert(0, _proj)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import expm

from data_generation.translational import generate_translational_data
from data_generation.rotational     import generate_rotational_data
from data_generation.scaling        import generate_scaling_data
from intrinsic_coordinate.discovery import discover_latent_dimension
from symmetry_discovery.identification import identify_symmetry
from symmetry_discovery.generators     import extract_generators, generator_orbit


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Generate 2-D datasets  (n_inputs=2 for clean 2-D orbit plots)
# ──────────────────────────────────────────────────────────────────────────────
N = 1000
print("Generating datasets …")

# 2-D for translational & scaling (clean scatter plots).
# Rotational uses n_inputs=4 with coefficients=[1,2,1,3] — same as the
# validated Task-5 test — because 2-D rotational data causes the encoder
# to produce coefficients that don't cluster reliably.  We then project
# onto dims (0, 2) — the equal-coefficient pair — for all 2-D scatter plots.
trans_data = generate_translational_data(n_inputs=2, m_orbits=1, n_samples=N, seed=0)
rot_data   = generate_rotational_data(n_inputs=4, coefficients=[1.0, 2.0, 1.0, 3.0],
                                      n_samples=N, seed=0)
scale_data = generate_scaling_data(n_inputs=2, m_scaling_vars=1, n_samples=N, seed=0)

# Dims used for 2-D scatter plots (all others projected away).
# For rotational we show the equal-coefficient subspace (dims 0 & 2, coeff=1).
PLOT_DIMS = {
    "translational": (0, 1),
    "rotational":    (0, 2),   # dims where coefficient == 1
    "scaling":       (0, 1),
}

datasets   = [trans_data, rot_data, scale_data]
sym_types  = ["translational", "rotational", "scaling"]
col_titles = ["Translational Symmetry", "Rotational Symmetry\n(proj. equal-coeff. plane)",
              "Scaling Symmetry"]
colors     = ["#4C72B0", "#DD8452", "#55A868"]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Run pipeline for each symmetry type
# ──────────────────────────────────────────────────────────────────────────────
print("Running Tasks 3→4→5 pipeline …")

pipelines = {}
for sym, data in zip(sym_types, datasets):
    print(f"  [{sym}]", flush=True)
    X, y = data["X"], data["y"]

    t3 = discover_latent_dimension(X, y, seed=0, n_epochs=500, n_restarts=3)
    t4 = identify_symmetry(
        X, y,
        n_latent=t3["optimal_n_latent"],
        decoder=t3["best_decoder"],
        n_epochs=1500, n_restarts=3, seed=0,
    )
    enc  = t4["encoders"][sym]
    gens = extract_generators(sym, enc)

    pipelines[sym] = dict(t3=t3, t4=t4, enc=enc, gens=gens)
    print(f"    detected={t4['symmetry_type']}, n_generators={len(gens)}")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Helper: compute discovered latent for a dataset
# ──────────────────────────────────────────────────────────────────────────────
def discovered_latent(sym, X):
    enc = pipelines[sym]["enc"]
    dev = next(enc.parameters()).device
    enc.eval()
    with torch.no_grad():
        z = enc(torch.tensor(X, dtype=torch.float32).to(dev)).cpu().numpy()
    # Return first latent dim for plotting
    return z[:, 0] if z.ndim == 2 else z.ravel()


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Ground-truth orbit helpers  (same as plot_symmetries.py)
# ──────────────────────────────────────────────────────────────────────────────
def _true_orbit_trans(data):
    """Line through the median orbit in x1-x2 space."""
    W   = data["orbit_directions"]          # (1, 2)
    orth = data["orthogonal_directions"]    # (2, 1)
    z_vals = (data["X"] @ W.T).ravel()
    z0 = np.median(z_vals)
    anchor = W.ravel() * z0 / np.dot(W.ravel(), W.ravel())
    t = np.linspace(-4, 4, 400)
    return anchor[0] + t * orth[0, 0], anchor[1] + t * orth[1, 0]


def _true_orbit_rot(data, d0=0, d1=2):
    """Circle in the equal-coefficient subspace (dims d0, d1) at median radius.

    The orbit lives in the subspace where coeff[d0] == coeff[d1], so fixing
    r = c[d0]*x_{d0}^2 + c[d1]*x_{d1}^2 = const traces a circle (c[d0]==c[d1]).
    Other dimensions are held at their median values (contribute a constant to r).
    """
    c   = np.array(data["coefficients"])   # shape (n_inputs,)
    X   = data["X"]
    # Total radius value at median point
    r_total = np.median((X ** 2) @ c)
    # Contribution from the non-plotted dims (held at median)
    other_dims = [i for i in range(X.shape[1]) if i not in (d0, d1)]
    r_other = float(np.median(
        np.sum(c[other_dims] * X[:, other_dims] ** 2, axis=1)
    )) if other_dims else 0.0
    r_sub = max(r_total - r_other, 1e-6)   # radius in the 2-D subspace
    # c[d0] == c[d1]  →  circle
    theta = np.linspace(0, 2 * np.pi, 400)
    return (np.sqrt(r_sub / c[d0]) * np.cos(theta),
            np.sqrt(r_sub / c[d1]) * np.sin(theta))


def _true_orbit_scale(data):
    """Line in log-log space at median log-pi."""
    E    = data["scaling_vectors"]           # (1, 2)
    orth = data["scaling_orbit_directions"]  # (2, 1)
    lp_vals = (np.log(data["X"]) @ E.T).ravel()
    lp0 = np.median(lp_vals)
    anchor_log = E.ravel() * lp0 / np.dot(E.ravel(), E.ravel())
    t = np.linspace(-2, 2, 400)
    return anchor_log[0] + t * orth[0, 0], anchor_log[1] + t * orth[1, 0]


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Discovered orbit helpers
# ──────────────────────────────────────────────────────────────────────────────
def _disc_orbit_single(sym, data, x_start=None, n_steps=400):
    """Trace one discovered orbit from x_start.
    Returns (coord_a, coord_b) projected onto PLOT_DIMS[sym].
    For scaling, coords are in log-space.
    """
    gens = pipelines[sym]["gens"]
    if not gens:
        return np.array([]), np.array([])

    g = gens[0]
    d0, d1 = PLOT_DIMS[sym]

    if x_start is None:
        if sym == "rotational":
            # X is standard-normal → coordinate-wise median ≈ 0, which makes
            # every rotation expm(k·ε·A) @ x_start ≈ 0 (collapsed orbit).
            # Instead pick the sample whose r = Σ cᵢ·xᵢ² is closest to the
            # median r — that point has a typical non-zero radius.
            c = np.array(data["coefficients"])
            r_vals = (data["X"] ** 2) @ c
            x_start = data["X"][np.argmin(np.abs(r_vals - np.median(r_vals)))]
        else:
            x_start = np.median(data["X"], axis=0)

    if sym == "translational":
        eps = 8.0 / n_steps
        fwd  = generator_orbit(x_start, g, n_steps // 2,  eps, sym)
        back = generator_orbit(x_start, g, n_steps // 2, -eps, sym)
        orb  = np.vstack([back[::-1], fwd[1:]])
        return orb[:, d0], orb[:, d1]

    elif sym == "rotational":
        eps = 2 * np.pi / n_steps
        orb = generator_orbit(x_start, g, n_steps, eps, sym)
        return orb[:, d0], orb[:, d1]

    else:  # scaling – orbit in log-space
        eps = 4.0 / n_steps
        back = generator_orbit(x_start, g, n_steps // 2, -eps, sym)
        fwd  = generator_orbit(x_start, g, n_steps // 2,  eps, sym)
        orb  = np.vstack([back[::-1], fwd[1:]])
        return np.log(orb[:, d0]), np.log(orb[:, d1])


def _multi_orbits(sym, data, n_starts=4):
    """Return list of (coord_a, coord_b) for n_starts orbit traces."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(data["X"]), n_starts, replace=False)
    result = []
    for i in idx:
        x0 = data["X"][i]
        result.append(_disc_orbit_single(sym, data, x_start=x0))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Build figure  (3 rows × 3 cols)
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 13))
fig.suptitle("Post-Discovery Orbit Visualization\n"
             "(solid = discovered orbit, dashed = ground truth)",
             fontsize=14, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

row_labels = [
    "Input space + orbits",
    "Discovered latent vs output",
    "Multiple discovered orbits",
]

start_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

for col, (sym, data, title, base_color) in enumerate(
        zip(sym_types, datasets, col_titles, colors)):

    X   = data["X"]
    y   = data["y"]
    lat = discovered_latent(sym, X)
    is_scale = (sym == "scaling")
    d0, d1 = PLOT_DIMS[sym]
    X_plot = np.log(X[:, [d0, d1]]) if is_scale else X[:, [d0, d1]]

    # ── Row 0: scatter + true orbit (dashed) + discovered orbit (solid) ──
    ax0 = fig.add_subplot(gs[0, col])
    sc = ax0.scatter(X_plot[:, 0], X_plot[:, 1],
                     c=y, cmap="RdYlBu_r", s=10, alpha=0.5, linewidths=0, zorder=1)
    fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.04, label="$y$")

    # True orbit
    if sym == "translational":
        tx, ty = _true_orbit_trans(data)
    elif sym == "rotational":
        tx, ty = _true_orbit_rot(data, d0=d0, d1=d1)
    else:
        tx, ty = _true_orbit_scale(data)
    ax0.plot(tx, ty, "k--", lw=1.8, label="true orbit", zorder=3)

    # Discovered orbit (single, from median)
    dx, dy_ = _disc_orbit_single(sym, data, n_steps=600)
    if len(dx):
        ax0.plot(dx, dy_, color=base_color, lw=2.0,
                 label="discovered orbit", zorder=4)

    ax0.set_title(title, fontsize=11, fontweight="bold")
    xl = (rf"$\log x_{{{d0+1}}}$" if is_scale else f"$x_{{{d0+1}}}$")
    yl = (rf"$\log x_{{{d1+1}}}$" if is_scale else f"$x_{{{d1+1}}}$")
    ax0.set_xlabel(xl, fontsize=10)
    ax0.set_ylabel(yl, fontsize=10)
    ax0.legend(fontsize=7, loc="upper right")
    if col == 0:
        ax0.annotate(row_labels[0], xy=(-0.38, 0.5), xycoords="axes fraction",
                     rotation=90, va="center", fontsize=9, color="gray")

    # ── Row 1: discovered latent vs y ────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, col])
    sort_idx = np.argsort(lat)
    ax1.scatter(lat, y, c=base_color, s=8, alpha=0.4, linewidths=0)
    ax1.plot(lat[sort_idx], data["y_clean"][sort_idx],
             color="black", lw=1.5, label="$y_{clean}$", zorder=5)
    ax1.set_xlabel("discovered latent $z$", fontsize=10)
    ax1.set_ylabel("$y$", fontsize=10)
    ax1.set_title("Discovered Latent → Output", fontsize=10)
    ax1.legend(fontsize=8)
    if col == 0:
        ax1.annotate(row_labels[1], xy=(-0.38, 0.5), xycoords="axes fraction",
                     rotation=90, va="center", fontsize=9, color="gray")

    # ── Row 2: multiple discovered orbits ────────────────────────────────
    ax2 = fig.add_subplot(gs[2, col])
    ax2.scatter(X_plot[:, 0], X_plot[:, 1],
                c="lightgray", s=8, alpha=0.35, linewidths=0, zorder=1)

    orbits = _multi_orbits(sym, data, n_starts=4)
    for k, (ox, oy_) in enumerate(orbits):
        if len(ox):
            ax2.plot(ox, oy_, color=start_colors[k], lw=2.0,
                     label=f"orbit {k+1}", zorder=4)
            ax2.scatter([ox[0]], [oy_[0]], color=start_colors[k],
                        s=50, zorder=5, marker="o")

    ax2.set_xlabel(xl, fontsize=10)
    ax2.set_ylabel(yl, fontsize=10)
    ax2.set_title("Discovered Orbits (4 starts)", fontsize=10)
    ax2.legend(fontsize=7, loc="upper right")
    if col == 0:
        ax2.annotate(row_labels[2], xy=(-0.38, 0.5), xycoords="axes fraction",
                     rotation=90, va="center", fontsize=9, color="gray")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Save
# ──────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(_here, "discovered_orbits.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")
plt.close(fig)
