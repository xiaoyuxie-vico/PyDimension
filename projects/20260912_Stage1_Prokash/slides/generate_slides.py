"""Generate PDF slides for the Stage1 Symmetry Discovery Pipeline."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.image as mpimg

# Colors
C_TITLE = "#2C3E50"
C_BOX = "#3498DB"
C_GREEN = "#27AE60"
C_ORANGE = "#E67E22"
C_RED = "#E74C3C"
C_LIGHT = "#ECF0F1"
C_DARK = "#34495E"
C_BG = "#FAFAFA"

_here = os.path.dirname(os.path.abspath(__file__))
_proj = os.path.dirname(_here)
_root = os.path.dirname(os.path.dirname(_proj))

# Possible figure locations
FIG_DIRS = [
    os.path.join(_root, "output_{}_symmetry"),
    os.path.join(_proj, "Examples", "{}_symmetry", "output_{}_symmetry"),
]

def _find_fig(name, key):
    for pat in FIG_DIRS:
        p = os.path.join(pat.format(key, key) if "{}" in pat else pat.format(key), name)
        if os.path.exists(p):
            return p
    return None

def new_slide(pdf, title=None, num=[0]):
    num[0] += 1
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    # Top bar
    ax.fill_between([0, 16], [9, 9], [8.85, 8.85], color=C_BOX)
    # Page number
    ax.text(15.5, 0.3, str(num[0]), fontsize=10, ha="right", color="#999")
    if title:
        ax.text(8, 8.2, title, fontsize=28, ha="center", va="center",
                fontweight="bold", color=C_TITLE)
    return fig, ax

def draw_box(ax, x, y, w, h, text, color=C_BOX, fontsize=13, textcolor="white"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor="none", alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, fontsize=fontsize, ha="center",
            va="center", color=textcolor, fontweight="bold", wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color=C_DARK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                                color=color, lw=2))

def add_bullet(ax, x, y, text, fontsize=16, color=C_DARK):
    ax.text(x, y, text, fontsize=fontsize, va="center", color=color)

def main():
    out = os.path.join(_here, "Stage1_Pipeline_Slides.pdf")
    with PdfPages(out) as pdf:

        # ── Slide 1: Title ──
        fig, ax = new_slide(pdf)
        ax.text(8, 5.5, "Symmetry Discovery in Physics Data",
                fontsize=36, ha="center", va="center", fontweight="bold", color=C_TITLE)
        ax.text(8, 4.2, "Stage1 Pipeline: From Raw Data to Lie-Algebra Generators",
                fontsize=20, ha="center", va="center", color=C_DARK)
        ax.plot([4, 12], [3.6, 3.6], color=C_BOX, lw=3)
        ax.text(8, 2.5, "PyDimension Project", fontsize=16, ha="center",
                va="center", color="#888", style="italic")
        pdf.savefig(fig); plt.close(fig)

        # ── Slide 2: Pipeline Overview ──
        fig, ax = new_slide(pdf, "Pipeline Overview")
        steps = [
            ("Step 1\nNormalize", "MinMax or\nStandard"),
            ("Step 2\nLatent Dim", "Autoencoder\nsweep k=1..K"),
            ("Step 3\nSymmetry Type", "3 competing\nencoders"),
            ("Step 4\nGenerators", "Null-space /\nantisymmetric"),
        ]
        colors = [C_GREEN, C_BOX, C_ORANGE, C_RED]
        for i, (label, desc) in enumerate(steps):
            bx = 1.0 + i * 3.8
            draw_box(ax, bx, 4.5, 3.0, 2.0, label, color=colors[i], fontsize=16)
            ax.text(bx + 1.5, 3.8, desc, fontsize=12, ha="center", va="center", color=C_DARK)
            if i < 3:
                draw_arrow(ax, bx + 3.0, 5.5, bx + 3.8, 5.5)
        ax.text(8, 2.2, "Input: X (features), y (output)    |    Output: symmetry type + generators",
                fontsize=15, ha="center", va="center", color=C_DARK)
        pdf.savefig(fig); plt.close(fig)

        # ── Slide 3: Normalization ──
        fig, ax = new_slide(pdf, "Step 1: Data Normalization")
        draw_box(ax, 1.5, 5.0, 5.5, 2.3, "", color=C_LIGHT, textcolor=C_DARK)
        ax.text(4.25, 6.8, "MinMax Normalization", fontsize=16, ha="center",
                fontweight="bold", color=C_GREEN)
        ax.text(4.25, 6.0, r"$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$",
                fontsize=18, ha="center", color=C_DARK)
        ax.text(4.25, 5.3, "Scales to [0, 1] — good for scaling symmetry",
                fontsize=12, ha="center", color="#666")

        draw_box(ax, 9.0, 5.0, 5.5, 2.3, "", color=C_LIGHT, textcolor=C_DARK)
        ax.text(11.75, 6.8, "Standard Normalization", fontsize=16, ha="center",
                fontweight="bold", color=C_BOX)
        ax.text(11.75, 6.0, r"$x_{norm} = \frac{x - \mu}{\sigma}$",
                fontsize=18, ha="center", color=C_DARK)
        ax.text(11.75, 5.3, "Zero mean, unit variance — good for translational",
                fontsize=12, ha="center", color="#666")

        ax.text(8, 3.5, "Choice of normalization affects which symmetry is easiest to detect",
                fontsize=14, ha="center", color=C_DARK, style="italic")
        ax.text(8, 2.5, "Positive data (e.g. physical quantities) works well with MinMax\n"
                "Centered data (e.g. momenta) works well with Standard",
                fontsize=13, ha="center", color="#555")
        pdf.savefig(fig); plt.close(fig)

        # ── Slide 4: Autoencoder Architecture ──
        fig, ax = new_slide(pdf, "Step 2: Latent Dimension Discovery")
        # Input
        draw_box(ax, 0.5, 4.0, 2.0, 1.5, "Input X\n(n features)", color=C_DARK, fontsize=13)
        draw_arrow(ax, 2.5, 4.75, 3.3, 4.75)
        # Augmentation
        draw_box(ax, 3.3, 3.5, 2.8, 2.5, "Augment\n[X, X², log|X|]\n(3n features)",
                 color=C_GREEN, fontsize=12)
        draw_arrow(ax, 6.1, 4.75, 6.9, 4.75)
        # Encoder
        draw_box(ax, 6.9, 4.0, 2.2, 1.5, "Encoder\nLinear/MLP", color=C_BOX, fontsize=13)
        draw_arrow(ax, 9.1, 4.75, 9.7, 4.75)
        # Bottleneck
        draw_box(ax, 9.7, 4.2, 1.2, 1.1, "z\n(k dims)", color=C_ORANGE, fontsize=13)
        draw_arrow(ax, 10.9, 4.75, 11.5, 4.75)
        # Decoder
        draw_box(ax, 11.5, 4.0, 2.2, 1.5, "Decoder\n3-layer MLP\nTanh", color=C_RED, fontsize=12)
        draw_arrow(ax, 13.7, 4.75, 14.3, 4.75)
        # Output
        draw_box(ax, 14.3, 4.2, 1.2, 1.1, "y_hat\n(1)", color=C_DARK, fontsize=13)

        ax.text(8, 2.5, "Sweep k = 1, 2, ..., K.  Pick smallest k where  "
                r"$R^2 > $ threshold",
                fontsize=15, ha="center", color=C_DARK, fontweight="bold")
        ax.text(8, 1.7, "Encoder: single linear layer (default) or multi-layer MLP (optional)\n"
                "Decoder: always nonlinear 3-layer MLP with Tanh activations",
                fontsize=12, ha="center", color="#555")
        pdf.savefig(fig); plt.close(fig)

        # ── Slide 5: Augmented Features ──
        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor("white")
        fig.suptitle("Augmented Feature Space: Why [X, X², log|X|]?",
                     fontsize=26, fontweight="bold", color=C_TITLE, y=0.94)
        # Top bar
        ax_bar = fig.add_axes([0, 0.95, 1, 0.015])
        ax_bar.set_xlim(0, 1); ax_bar.set_ylim(0, 1)
        ax_bar.fill_between([0, 1], [1, 1], [0, 0], color=C_BOX)
        ax_bar.axis("off")

        titles = ["X features\nTranslational", "X² features\nRotational",
                  "log|X| features\nScaling"]
        colors_s = [C_GREEN, C_BOX, C_ORANGE]
        for i, (t, c) in enumerate(zip(titles, colors_s)):
            ax_s = fig.add_axes([0.07 + i*0.31, 0.25, 0.26, 0.45])
            x = np.linspace(0.1, 3, 100)
            if i == 0:
                ax_s.plot(x, 2*x + 1, color=c, lw=3)
                ax_s.set_ylabel("y = ax + b", fontsize=11)
                ax_s.fill_between(x, 2*x+0.5, 2*x+1.5, alpha=0.15, color=c)
                ax_s.annotate("shift invariant", xy=(1.5, 4), fontsize=10, color=c)
            elif i == 1:
                t_ = np.linspace(0, 2*np.pi, 100)
                ax_s.plot(np.cos(t_), np.sin(t_), color=c, lw=3)
                ax_s.plot(1.5*np.cos(t_), 1.5*np.sin(t_), color=c, lw=2, ls="--")
                ax_s.set_aspect("equal")
                ax_s.set_ylabel(r"$x^2 + y^2 = r^2$", fontsize=11)
                ax_s.annotate("rotation invariant", xy=(-0.5, 1.6), fontsize=10, color=c)
            else:
                ax_s.plot(x, np.log(x), color=c, lw=3)
                ax_s.set_ylabel("log(x)", fontsize=11)
                ax_s.annotate("scale invariant", xy=(1.0, 0.5), fontsize=10, color=c)
            ax_s.set_title(t, fontsize=14, fontweight="bold", color=c)
            ax_s.grid(True, alpha=0.3)

        fig.text(0.5, 0.1, "Each augmentation lets the linear encoder capture\n"
                 "a different symmetry type without prior knowledge",
                 fontsize=14, ha="center", color=C_DARK, style="italic")
        # Page number
        fig.text(0.97, 0.03, "5", fontsize=10, ha="right", color="#999")
        pdf.savefig(fig); plt.close(fig)

        # ── Slide 6: R² Sweep ──
        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor("white")
        fig.suptitle("Latent Dimension Selection", fontsize=26,
                     fontweight="bold", color=C_TITLE, y=0.94)
        ax_bar2 = fig.add_axes([0, 0.95, 1, 0.015])
        ax_bar2.set_xlim(0, 1); ax_bar2.set_ylim(0, 1)
        ax_bar2.fill_between([0, 1], [1, 1], [0, 0], color=C_BOX)
        ax_bar2.axis("off")

        ax_r2 = fig.add_axes([0.08, 0.2, 0.4, 0.6])
        ks = [1, 2, 3, 4]
        r2s = [0.45, 0.92, 0.96, 0.97]
        bars = ax_r2.bar(ks, r2s, color=[C_RED, C_ORANGE, C_GREEN, C_GREEN],
                         edgecolor="black", lw=1)
        ax_r2.axhline(0.95, color="red", ls="--", lw=2, label=r"$R^2$ threshold")
        ax_r2.set_xlabel("Latent dimension k", fontsize=14)
        ax_r2.set_ylabel(r"$R^2$ (validation)", fontsize=14)
        ax_r2.set_ylim(0, 1.05)
        ax_r2.set_xticks(ks)
        ax_r2.legend(fontsize=12)
        ax_r2.annotate("elbow", xy=(3, 0.96), xytext=(3.5, 0.75),
                       fontsize=14, fontweight="bold", color=C_RED,
                       arrowprops=dict(arrowstyle="->", color=C_RED, lw=2))

        ax_txt = fig.add_axes([0.55, 0.2, 0.4, 0.6])
        ax_txt.axis("off")
        txt = ("Algorithm:\n\n"
               "1. Train autoencoder for each k\n\n"
               "2. Compute R² on validation set\n\n"
               "3. Select smallest k where\n"
               r"    R² > threshold (0.95)" + "\n\n"
               "4. Report both R²_train and R²_test\n"
               "   to check for overfitting")
        ax_txt.text(0.05, 0.9, txt, fontsize=16, va="top", color=C_DARK,
                    transform=ax_txt.transAxes, linespacing=1.3)
        fig.text(0.97, 0.03, "6", fontsize=10, ha="right", color="#999")
        pdf.savefig(fig); plt.close(fig)

        # ── Slide 7: Symmetry Identification ──
        fig, ax = new_slide(pdf, "Step 3: Identify Symmetry Type")
        types = [
            ("Translational", r"$\phi(x) = x$", r"$z = W \cdot x$", C_GREEN),
            ("Rotational", r"$\phi(x) = x^2$", r"$z = W \cdot x^2$", C_BOX),
            ("Scaling", r"$\phi(x) = \log|x|$", r"$z = W \cdot \log|x|$", C_ORANGE),
        ]
        for i, (name, phi, enc, col) in enumerate(types):
            bx = 1.0 + i * 4.8
            draw_box(ax, bx, 5.0, 4.0, 2.2, "", color=col, textcolor="white")
            ax.text(bx + 2.0, 6.8, name, fontsize=16, ha="center",
                    fontweight="bold", color="white")
            ax.text(bx + 2.0, 6.1, phi, fontsize=16, ha="center", color="white")
            ax.text(bx + 2.0, 5.4, enc, fontsize=14, ha="center", color="#eee")

        ax.text(8, 3.8, "All three share the same frozen decoder from Step 2",
                fontsize=14, ha="center", color=C_DARK, style="italic")
        ax.text(8, 2.8, "Winner = encoder with lowest validation loss",
                fontsize=16, ha="center", color=C_DARK, fontweight="bold")
        ax.text(8, 1.8, "The linear encoder W stays interpretable for generator extraction in Step 4",
                fontsize=13, ha="center", color="#666")
        pdf.savefig(fig); plt.close(fig)

        # ── Slide 8: Generator Extraction ──
        fig, ax = new_slide(pdf, "Step 4: Extract Lie-Algebra Generators")
        items = [
            ("Translational", "Generators = null(W)", r"$x(\epsilon) = x_0 + \epsilon \cdot g$",
             "Shift along g preserves y", C_GREEN),
            ("Scaling", "Generators = null(W) in log-space",
             r"$x(\epsilon) = x_0 \cdot e^{\epsilon \cdot g}$",
             "Rescaling along g preserves y", C_ORANGE),
            ("Rotational", "Generators = antisymmetric from W",
             r"$x(\epsilon) = e^{\epsilon G} \cdot x_0$",
             "Rotation by G preserves y", C_BOX),
        ]
        for i, (name, method, formula, meaning, col) in enumerate(items):
            by = 6.5 - i * 2.1
            draw_box(ax, 0.5, by, 2.8, 1.5, name, color=col, fontsize=15)
            ax.text(4.0, by + 1.1, method, fontsize=14, va="center", color=C_DARK,
                    fontweight="bold")
            ax.text(4.0, by + 0.5, formula, fontsize=15, va="center", color=C_DARK)
            ax.text(10.5, by + 0.75, meaning, fontsize=13, va="center",
                    color="#555", style="italic")
        ax.text(8, 0.8, "Generators span the Lie algebra of the discovered symmetry group",
                fontsize=14, ha="center", color=C_DARK, fontweight="bold")
        pdf.savefig(fig); plt.close(fig)

        # ── Slide 9: Generator Orbits ──
        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor("white")
        fig.suptitle("Generator Orbits: Visualizing Symmetry", fontsize=26,
                     fontweight="bold", color=C_TITLE, y=0.94)
        ax_bar3 = fig.add_axes([0, 0.95, 1, 0.015])
        ax_bar3.set_xlim(0, 1); ax_bar3.set_ylim(0, 1)
        ax_bar3.fill_between([0, 1], [1, 1], [0, 0], color=C_BOX); ax_bar3.axis("off")

        ax_orb = fig.add_axes([0.08, 0.15, 0.45, 0.7])
        xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
        zz = xx**2 + yy**2
        ax_orb.contour(xx, yy, zz, levels=8, colors="#ccc", linewidths=1)
        t_ = np.linspace(0, 2*np.pi, 100)
        for r in [1.0, 2.0]:
            ax_orb.plot(r*np.cos(t_), r*np.sin(t_), color=C_RED, lw=2.5)
            ax_orb.annotate("", xy=(r*np.cos(0.3), r*np.sin(0.3)),
                           xytext=(r*np.cos(0), r*np.sin(0)),
                           arrowprops=dict(arrowstyle="->", color=C_RED, lw=2))
        ax_orb.set_xlabel(r"$x_1$", fontsize=14)
        ax_orb.set_ylabel(r"$x_2$", fontsize=14)
        ax_orb.set_title("Orbits trace level curves of y", fontsize=14, color=C_DARK)
        ax_orb.set_aspect("equal")
        ax_orb.grid(True, alpha=0.2)

        ax_txt2 = fig.add_axes([0.58, 0.15, 0.38, 0.7])
        ax_txt2.axis("off")
        txt2 = ("Each orbit starts from a data point\n"
                "and traces a path along which\n"
                "the output y stays constant.\n\n"
                "Multiple orbits from different\n"
                "starting points trace parallel\n"
                "paths = global symmetry.\n\n"
                "Red curves: orbits of a\n"
                "rotational generator\n\n"
                "Gray contours: level sets of\n"
                r"$y = x_1^2 + x_2^2$")
        ax_txt2.text(0.05, 0.95, txt2, fontsize=15, va="top", color=C_DARK,
                     transform=ax_txt2.transAxes, linespacing=1.5)
        fig.text(0.97, 0.03, "9", fontsize=10, ha="right", color="#999")
        pdf.savefig(fig); plt.close(fig)

        # ── Slides 10-13: Examples ──
        examples = [
            ("Keyhole Welding — Scaling Symmetry", "keyhole",
             "keyhole_symmetry_discovery.png",
             "7 inputs: absorbed power, scan speed,\n"
             "beam radius, diffusivity, density, cp, temp diff\n\n"
             r"$Ke = \frac{\eta P}{\Delta T \cdot \pi \rho c_p \sqrt{\alpha V_s r_0^3}}$"
             "\n\nDetected: Scaling symmetry\nGenerators: 6 (log-space scaling directions)"),
            ("Concrete Strength — Translational Symmetry", "concrete",
             "concrete_symmetry_discovery.png",
             "8 inputs: cement, slag, fly ash, water,\n"
             "superplasticizer, coarse/fine aggregate, age\n\n"
             "Output: compressive strength (MPa)\n\n"
             "Detected: Translational symmetry\n"
             "Generators: 7 substitution directions\n"
             'e.g. "increase fly ash, decrease cement"'),
            ("Permeability — Angular Periodicity", "permeability",
             "permeability_symmetry_discovery.png",
             "180 geometries x 37 angles (0-360 deg)\n"
             "= 6660 rows\n\n"
             "Permeability repeats every 180 degrees\n\n"
             r"Inputs: $\cos(2\theta),\ \sin(2\theta)$"
             "\n\nDetected: Rotational symmetry\n"
             "Encodes the 180-degree periodicity"),
            ("LHC Dijet — Rotational Symmetry", "lhc",
             "lhc_symmetry_discovery.png",
             "4 inputs: p1x, p1y, p2x, p2y\n"
             "(anti-kT jet clustering, R=1.0)\n\n"
             r"$m_{jj}^T = \sqrt{2 p_{T1} p_{T2} (1-\cos\Delta\phi)}$"
             "\n\nExpected: SO(2) azimuthal rotation\n\n"
             "Note: y is deterministic from inputs,\n"
             "so R2 is artificially high"),
        ]
        for idx, (title, key, figname, desc) in enumerate(examples):
            fig = plt.figure(figsize=(16, 9))
            fig.patch.set_facecolor("white")
            fig.suptitle(f"Example: {title}", fontsize=24,
                         fontweight="bold", color=C_TITLE, y=0.94)
            ax_bar4 = fig.add_axes([0, 0.95, 1, 0.015])
            ax_bar4.set_xlim(0, 1); ax_bar4.set_ylim(0, 1)
            ax_bar4.fill_between([0, 1], [1, 1], [0, 0], color=C_BOX); ax_bar4.axis("off")

            # Description on the left
            ax_desc = fig.add_axes([0.03, 0.08, 0.35, 0.78])
            ax_desc.axis("off")
            ax_desc.text(0.05, 0.95, desc, fontsize=14, va="top",
                         transform=ax_desc.transAxes, color=C_DARK, linespacing=1.5)

            # Figure on the right
            figpath = _find_fig(figname, key)
            ax_img = fig.add_axes([0.4, 0.05, 0.58, 0.82])
            if figpath:
                img = mpimg.imread(figpath)
                ax_img.imshow(img)
                ax_img.axis("off")
            else:
                ax_img.axis("off")
                ax_img.text(0.5, 0.5, f"[Run discover_symmetry.py to\ngenerate {figname}]",
                           fontsize=16, ha="center", va="center",
                           transform=ax_img.transAxes, color="#999", style="italic")

            fig.text(0.97, 0.03, str(10 + idx), fontsize=10, ha="right", color="#999")
            pdf.savefig(fig); plt.close(fig)

        # ── Slide 14: Summary ──
        fig, ax = new_slide(pdf, "Summary")
        bullets = [
            "Stage1 discovers continuous symmetries from data — no prior physics needed",
            "4-step pipeline: Normalize  ->  Latent Dimension  ->  Symmetry Type  ->  Generators",
            "Three symmetry types: Translational (shifts), Rotational (rotations), Scaling (rescaling)",
            "Augmented features [X, X², log|X|] let the encoder capture all three types",
            "Generators reveal the Lie algebra — continuous transformation families preserving physics",
            "Optional: multi-layer encoder for Step 2, Pi group augmentation for scaling problems",
            "Validated on: keyhole welding, concrete strength, permeability, LHC dijet data",
        ]
        for i, b in enumerate(bullets):
            y = 7.0 - i * 0.85
            ax.plot(1.5, y, "o", color=C_BOX, markersize=8)
            ax.text(2.0, y, b, fontsize=15, va="center", color=C_DARK)
        pdf.savefig(fig); plt.close(fig)

    print(f"Slides saved to: {out}")
    print(f"Total: 14 slides")

if __name__ == "__main__":
    main()
