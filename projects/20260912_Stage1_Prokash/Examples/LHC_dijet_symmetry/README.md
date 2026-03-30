# LHC Dijet Symmetry Discovery

Discover hidden symmetries in Large Hadron Collider dijet events using the
PyDimension Stage1 pipeline.

## Physics Background

In proton-proton collisions at the LHC, the two leading jets in dijet events
carry the dominant kinematic information. Their transverse momentum components
`(p1x, p1y, p2x, p2y)` encode a fundamental symmetry of the underlying
physics: **azimuthal rotational invariance**.

Rotating both jets by the same angle in the transverse plane does not change
any Lorentz-invariant observable. In particular, the dijet transverse invariant
mass is preserved:

```
m_jj_T = sqrt(2 * pT1 * pT2 * (1 - cos(dphi)))
```

This is an **SO(2) rotational symmetry** with generator:

```
    [ 0 -1  0  0 ]
A = [ 1  0  0  0 ]
    [ 0  0  0 -1 ]
    [ 0  0  1  0 ]
```

## Pipeline

The example runs the full Stage1 symmetry discovery pipeline:

1. **Data preparation** — Load the LHCO R&D dataset, cluster particles into
   jets with FastJet (anti-kT, R=1.0), extract leading dijet momenta
2. **Output construction** — Compute dijet transverse mass `m_jj_T`
3. **Normalization** — Standardize features
4. **Latent dimension discovery** — Autoencoder sweep to find intrinsic
   dimensionality
5. **Symmetry identification** — Competitive encoder training across
   translational, rotational, and scaling candidates
6. **Generator extraction** — Recover the Lie-algebra generators of the
   discovered symmetry group
7. **Validation** — Verify that `m_jj_T` is invariant under the discovered
   transformation

## Quick Start

### With synthetic LHC-like data (no external files needed)

```bash
cd examples/lhc_dijet_symmetry
python discover_symmetry.py --synthetic
```

### With real LHC data

1. Download the dataset from https://zenodo.org/record/4536377
   (file: `events_anomalydetection_v2.h5`)

2. Prepare the data:
```bash
python prepare_data.py --input events_anomalydetection_v2.h5
```

3. Run symmetry discovery:
```bash
python discover_symmetry.py --data lhc_dijet_data.pt
```

## Output

Results are saved to `output_lhc_symmetry/`:
- `lhc_symmetry_discovery.png` — 6-panel summary figure
- `discovery_summary.txt` — text summary of all results

## Expected Results

The pipeline should discover **rotational symmetry** with:
- Encoder coefficients approximately proportional to `[1, 1, 1, 1]`
  (equal weighting on all four momentum components in the quadratic form)
- Antisymmetric generator matrices coupling `(p1x, p1y)` and `(p2x, p2y)`
- Near-zero validation loss for the rotational encoder compared to
  translational and scaling alternatives
