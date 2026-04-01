# Concrete Compressive Strength — Translational Symmetry Discovery

Discover **translational symmetry** in concrete compressive strength data
using the PyDimension Stage1 pipeline.

## Physics Background

Concrete compressive strength depends on mix proportions and curing age.
Unlike dimensional analysis problems (where variables combine as products
of powers), concrete strength is governed by **additive** relationships:

- Total binder = cement + slag + fly ash
- Water-to-binder ratio = water / (cement + slag + fly ash)
- Strength ≈ f(a₁·cement + a₂·slag + a₃·fly_ash + a₄·water + ...)

This additive structure means the symmetry is **translational**:
the output depends on a linear combination `z = W·x`, and any shift
`x → x + ε·g` in the null space of W leaves the strength unchanged.

| Variable | Units | Range |
|----------|-------|-------|
| Cement | kg/m³ | 102–540 |
| Blast Furnace Slag | kg/m³ | 0–359 |
| Fly Ash | kg/m³ | 0–200 |
| Water | kg/m³ | 122–247 |
| Superplasticizer | kg/m³ | 0–32 |
| Coarse Aggregate | kg/m³ | 801–1145 |
| Fine Aggregate | kg/m³ | 594–993 |
| Age | days | 1–365 |

**Output**: Compressive strength (MPa), 1030 samples.

## What are the generators?

With 8 input variables and 1 latent dimension, there are **7 generators**.
Each generator is a direction in input space along which strength is preserved:
`x → x + ε·g` keeps `f(W·x)` unchanged.

Physically, these are **mix substitutions** — e.g., "replace cement with fly ash
while keeping the same compressive strength." Each generator tells you exactly
how much of one ingredient compensates for another.

## Pipeline

1. **Normalize** — Standard scaling (zero mean, unit variance)
2. **Latent dimension** — Autoencoder sweep (expected: k = 1)
3. **Symmetry identification** — Competitive training across translational, rotational, scaling
4. **Generator extraction** — Null-space of encoder weight matrix W
5. **Visualization** — 3-panel figure: latent variable vs strength, symmetry losses, generator orbits

## Dataset

- **Source**: UCI Machine Learning Repository, Dataset #165
- **Samples**: 1030
- **Format**: Excel (.xls/.xlsx) or CSV
- **Download**: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

## Usage

```bash
# With real UCI data (place file in this directory):
python discover_symmetry.py --data Concrete_Data.xls

# With synthetic data (no external files needed):
python discover_symmetry.py --synthetic
```

## Expected Results

- **Symmetry type**: translational (clear winner, ~35x loss gap over scaling/rotational)
- **Latent dimension**: 1
- **Generators**: 7 independent substitution directions
- Example generator: "increase fly ash while decreasing cement" → same strength

## References

1. Yeh, I-C. "Modeling of strength of high-performance concrete using artificial
   neural networks." *Cement and Concrete Research* 28.12 (1998): 1797-1808.
2. UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/165
