# Porous Media Permeability — Symmetry Discovery

Discover symmetry in porous media permeability data using the
PyDimension Stage1 pipeline.

## Physics Background

Permeability of a porous medium characterizes how easily fluid flows through
its microstructure. The directional permeability `Permeability_X` depends on:

| Variable | Description | Units |
|----------|-------------|-------|
| Angle | Orientation angle | degrees |
| Porosity | Void fraction | % |
| Surface_A | Specific surface area | — |

The relationship `Permeability_X = f(Angle, Porosity, Surface_A)` may exhibit:
- **Translational symmetry**: if permeability depends on a linear combination of inputs
- **Rotational symmetry**: if angle enters quadratically (e.g., cos²θ + sin²θ invariance)
- **Scaling symmetry**: if variables combine as a power law (e.g., Kozeny-Carman: k ∝ φ³/S²)

The Stage1 pipeline discovers which symmetry type best describes the data.

## What are the generators?

With 3 input variables and 1 latent dimension, there are **2 generators**.
Each generator describes a transformation of the input variables that
leaves the permeability unchanged — revealing the hidden invariances
of the pore structure.

## Pipeline

1. **Normalize** — Standard scaling
2. **Latent dimension** — Autoencoder sweep (k = 1, 2, 3)
3. **Symmetry identification** — Competitive training across translational, rotational, scaling
4. **Generator extraction** — Null-space (translational/scaling) or antisymmetric matrices (rotational)
5. **Visualization** — 3-panel figure: latent variable vs permeability, symmetry losses, generator orbits

## Dataset

Place your permeability data file (CSV or Excel) in this directory.
Expected columns: `Index`, `Angle`, `Porosity`, `Surface_A`, `Permeability_X`.

## Usage

```bash
# With real data:
python discover_symmetry.py --data permeability_data.xlsx

# With synthetic data:
python discover_symmetry.py --synthetic
```

## References

1. Kozeny, J. "Uber kapillare Leitung des Wassers im Boden."
   *Sitzungsber Akad. Wiss.* 136 (1927): 271-306.
2. Carman, P. C. "Fluid flow through granular beds."
   *Trans. Inst. Chem. Eng.* 15 (1937): 150-166.
