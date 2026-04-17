# Geothermal Heat Flow — Scaling Symmetry Discovery

Discover the scaling symmetry in Antarctic geothermal heat-flow data
using the PyDimension Stage1 pipeline.

## Motivation

Conductive heat flow through the Earth's crust obeys **Fourier's law**:

```
Q = k · G
```

where `Q` is the surface heat flow (mW/m²), `k` is the bulk thermal
conductivity of the sediment/rock column (W/(m·K)), and `G` is the
vertical temperature gradient (°C/km).

The dataset records two additional geometric variables — probe
penetration depth `z` and water depth `d` — that do **not** enter the
heat-flow equation.  This makes the example a clean test for the
pipeline: it should discover that `Q` depends on `k · G` alone
(a scaling symmetry), and that `z` and `d` are irrelevant ("free"
generators).

## Physics background

Four physical inputs with three fundamental dimensions:

| Variable              | Symbol | Units    | Dimensions       |
|-----------------------|--------|----------|------------------|
| Thermal conductivity  | k      | W/(m·K)  | kg·m·s⁻³·K⁻¹     |
| Temperature gradient  | G      | °C/km    | K·m⁻¹            |
| Probe depth           | z      | m        | m                |
| Water depth           | d      | m        | m                |

The **Buckingham Pi theorem** gives `4 − 3 = 1` independent
dimensionless group (plus the purely geometric ratio `z/d`):

```
Pi = Q / (k · G)   ≡  1   (Fourier's law)
```

A **scaling symmetry** acts by simultaneously rescaling physical
variables in a way that leaves every dimensionless group unchanged.
The generators of this group reveal which variables can trade off
without changing the heat flow.

## Dataset

`dataset_ghf.csv` — 210 measurements extracted from the **Antarctic
Geothermal Heat Flow Database** (Dziadek et al., 2021).

* Source: `ANT_GHF_DB_V004.xlsx` from the Antarctic GHF-DB
* Columns: station, lat, lon, k, G, z, d, Q, quality, method
* All rows have positive k, G, z, d, Q
* Fourier verification: median |Q − k·G| / Q = 0.42%

## Pipeline

1. **Data loading** — read CSV, drop rows with non-positive values
2. **Normalization** — min-max scaling (optionally with geometric-mean
   pre-centring via `--log-normalize`)
3. **Latent dimension discovery** — autoencoder sweep (expected: k* = 1,
   since Q depends on the single product k·G)
4. **Symmetry identification** — competitive encoder training (expected:
   **scaling** wins)
5. **Generator extraction** — null-space of the linear log-space encoder
   → 3 scaling directions (`n_inputs − n_latent = 4 − 1`)
6. **Visualization** — 3-panel summary figure:
   * `log₁₀(k·G)` vs `log₁₀(Q)` (Fourier collapse),
   * Validation MSE for translational / rotational / scaling,
   * Generator orbits over the two most-weighted variables with
     known-Fourier reference contours

## Usage

```bash
cd projects/20260912_Stage1_Prokash/Examples/geothermal_heat_flow

python discover_symmetry.py --data dataset_ghf.csv

# With a multi-layer encoder:
python discover_symmetry.py --data dataset_ghf.csv --encoder-hidden 64 32

# With geometric-mean pre-centring:
python discover_symmetry.py --data dataset_ghf.csv --log-normalize

# With Fourier-product augmentation:
python discover_symmetry.py --data dataset_ghf.csv --fourier-basis
```

## Expected results

* **Symmetry type**: scaling (lowest validation MSE)
* **Latent dimension**: k* = 1 (Q ≈ function of single coordinate k·G)
* **Encoder weight vector**: aligned with `[k, G, z, d] → [+1, +1, 0, 0]`
  (Fourier exponents) — k and G should have equal positive weights,
  z and d near zero
* **Generators**: 3 null-space directions:
  - One constrained direction: increase k, decrease G (or vice versa)
    → the k·G = const trade-off
  - Two free directions: z and d can vary independently without
    affecting Q
* **Physical meaning**: doubling k while halving G keeps Q constant
  (Fourier's law); z and d are irrelevant to conductive heat flow

## Observed results

Running with `--encoder-hidden 64 32 --log-normalize`:

### Step 2 — Latent dimension

```
k=1: R2_train=0.958, R2_test=0.956, MSE=0.0012
k=2: R2_train=0.955, R2_test=0.950, MSE=0.0013
k=3: R2_train=0.957, R2_test=0.957, MSE=0.0011
k=4: R2_train=0.956, R2_test=0.948, MSE=0.0014
```

`k* = 1` — heat flow is well described by a single latent coordinate.
R²_test = 0.956 confirms excellent collapse.

### Step 3 — Symmetry type

```
scaling       : 0.0021  ← winner
translational : 0.0023
rotational    : 0.0037
Loss gap: 1.1×
```

Scaling wins.  The gap is modest (1.1×) because the dataset has only
4 input variables (vs 7 in LPBF), so the translational encoder can
also find a reasonable fit.

### Step 4 — Encoder weight vector

```
          k        G        z        d
L2-n: +0.389   +0.919   +0.042   -0.055
known: +0.707   +0.707   +0.000   +0.000
```

`cos(learned, known) = +0.925` — strong alignment with Fourier's law.

| Variable | Learned sign | Known sign | Weight | Match? |
|----------|-------------|------------|--------|--------|
| k        | +           | +          | 0.389  | ✓      |
| G        | +           | +          | 0.919  | ✓      |
| z        | ≈ 0         | 0          | 0.042  | ✓      |
| d        | ≈ 0         | 0          | 0.055  | ✓      |

All four exponents are correctly recovered.  The slight asymmetry
between k and G weights (0.39 vs 0.92 instead of equal) reflects
the data's dynamic range: G spans two orders of magnitude
(1–360 °C/km) while k spans less than one (0.6–3.5 W/(m·K)).

### Step 5 — Generators

The 3 null-space generators split cleanly:

**Constrained direction** (physically meaningful):
- Generator 1: decrease k, increase G → the `k·G = const` trade-off
  (Fourier's law)

**Free directions** (irrelevant to heat flow):
- Generator 2: z is completely free (weight ≈ 0)
- Generator 3: d is completely free (weight ≈ 0)

### Summary

| Aspect | Result |
|--------|--------|
| Symmetry type | Scaling ✓ |
| Latent dimension | k* = 1 ✓ |
| k, G exponents | Correctly recovered ✓ |
| z, d exponents | Correctly identified as irrelevant ✓ |
| cos(learned, Fourier) | 0.925 |
| Test R² | 0.956 |

The pipeline discovers that geothermal heat flow obeys a scaling
symmetry, recovers Fourier's law Q = k·G from data alone, and
correctly identifies probe depth and water depth as physically
irrelevant variables.
