# Keyhole Welding Symmetry Discovery

Discover hidden scaling symmetry in laser keyhole welding data using the
PyDimension Stage1 pipeline.

## Physics Background

Laser keyhole welding involves a focused laser beam drilling a vapour cavity
(keyhole) into a metal workpiece.  The keyhole eccentricity `e*` depends on
seven physical variables with four fundamental dimensions (M, L, T, K):

| Variable | Symbol | Units | Dimensions |
|----------|--------|-------|------------|
| Absorbed power | `etaP` | W | kg·m²·s⁻³ |
| Laser speed | `Vs` | m/s | m·s⁻¹ |
| Beam radius | `r0` | m | m |
| Thermal diffusivity | `alpha` | m²/s | m²·s⁻¹ |
| Density | `rho` | kg/m³ | kg·m⁻³ |
| Specific heat | `cp` | J/(kg·K) | m²·s⁻²·K⁻¹ |
| Temperature diff. | `Tl-T0` | K | K |

By the **Buckingham Pi theorem** (7 variables - 4 dimensions = 3 groups),
there exist 3 independent dimensionless groups π₁, π₂, π₃ such that:

```
e* = f(π₁, π₂, π₃)
```

This is a **scaling symmetry**: rescaling measurement units (e.g. metres to
centimetres) changes the numerical values of the variables but not the
dimensionless output.  The Stage1 encoder discovers these power-law groups
via log-space linear weights.

## Pipeline

1. **Data loading** — Read keyhole CSV or generate synthetic data
2. **Normalization** — Standardize features
3. **Latent dimension** — Autoencoder sweep (expected: k ≈ 2-3)
4. **Symmetry identification** — Competitive training (expected: scaling)
5. **Generator extraction** — Null-space of encoder weights = scaling directions
6. **Visualization** — 6-panel summary figure

## Usage

### Synthetic data (no files needed)

```bash
cd projects/20260912_Stage1_Prokash/Examples/keyhole_symmetry
python discover_symmetry.py --synthetic
```

### Real keyhole data

```bash
python discover_symmetry.py --data path/to/dataset_keyhole.csv
```

## Expected Results

- **Symmetry type**: scaling
- **Latent dimension**: 2-3 (matching 3 dimensionless groups)
- **Encoder weights**: log-space exponents forming dimensionless groups
  (e.g. Peclet number Pe = Vs·r0/alpha)
- **Generators**: null-space directions corresponding to unit rescalings
  that preserve e*
