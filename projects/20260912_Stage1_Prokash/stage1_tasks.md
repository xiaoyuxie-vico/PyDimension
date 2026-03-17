# OpenSymmetry Stage 1: Task Breakdown

## How to Use This Document

Each task is self-contained and builds on the previous one. Complete them in order.
Each task ends with **Validation Tests** — a checklist of concrete, runnable checks.
Do not move to the next task until all validations pass.

---

## Task 1: Data Generation

**Goal:** Build the three synthetic data generators and confirm they produce correct data.

**Files to create:**
- `data_generation/__init__.py`
- `data_generation/translational.py` → `generate_translational_data()`
- `data_generation/rotational.py` → `generate_rotational_data()`
- `data_generation/scaling.py` → `generate_scaling_data()`

**Scope:**
- Each generator returns a dict with `X`, `y`, and ground-truth metadata.
- Translational: `y = f(W·x)` where `W` is the orthogonal complement of orbit directions.
- Rotational: `y = f(a·x₁² + b·x₂² + ...)`.
- Scaling: `y = f(x₁^a · x₂^b · ...)`, inputs must be positive.
- Use a nonlinear `f` (e.g., `sin`, `exp`, polynomial) so the problem is non-trivial.
- Support a `noise_level` parameter for additive Gaussian noise on `y`.

**Validation Tests:**

| # | Test | Pass Criteria |
|---|------|---------------|
| 1.1 | **Shape check.** For each generator with `n_inputs=5, n_samples=1000`: `X.shape == (1000, 5)` and `y.shape == (1000,)`. | Shapes match exactly. |
| 1.2 | **Determinism.** Call each generator twice with the same `np.random.seed`. Outputs must be identical. | `np.allclose(X1, X2)` and `np.allclose(y1, y2)`. |
| 1.3 | **Translational invariance.** Generate translational data with known orbit direction `d`. Shift all samples by `ε·d`. Recompute `y` from the ground-truth formula. Verify `|y_shifted − y_original| < 1e-10`. | Max absolute difference < 1e-10 (noise_level=0). |
| 1.4 | **Rotational invariance.** Generate rotational data with `coefficients=[1, 1, 2]`. Apply a rotation mixing dimensions 0 and 1 (where coefficients are equal). Recompute `y`. Verify invariance. | Max absolute difference < 1e-10. |
| 1.5 | **Scaling invariance.** Generate scaling data with exponents `[2, -1]`. Scale inputs as `x₁ → λ²·x₁, x₂ → λ⁻¹·x₂`. Verify `y` unchanged. | Max absolute difference < 1e-10. |
| 1.6 | **Noise sanity.** Generate data with `noise_level=0.1`. Verify `std(y − y_clean) ≈ 0.1·std(y_clean)` within 20%. | Ratio is in [0.08, 0.12] · std(y_clean). |
| 1.7 | **Scaling positivity.** All entries of `X` from scaling generator are > 0. | `np.all(X > 0)`. |

---

## Task 2: Preprocessing

**Goal:** Build a normalization utility and confirm it works correctly with the data generators.

**Files to create:**
- `preprocessing/__init__.py`
- `preprocessing/normalize.py` → `normalize_data()`

**Scope:**
- Support `method` ∈ {"standard", "minmax", "robust"}.
- Return normalized arrays plus fitted scaler objects for inverse transform.

**Validation Tests:**

| # | Test | Pass Criteria |
|---|------|---------------|
| 2.1 | **Standard normalization stats.** After normalizing with `method="standard"`: mean ≈ 0, std ≈ 1 for each column of `X_normalized`. | `abs(mean) < 1e-10` and `abs(std - 1) < 1e-10` per column. |
| 2.2 | **Minmax range.** After `method="minmax"`: all values in [0, 1]. | `min >= 0` and `max <= 1` per column. |
| 2.3 | **Round-trip.** Normalize, then inverse-transform. Recover original `X` and `y`. | `np.allclose(X_recovered, X_original, atol=1e-10)`. |
| 2.4 | **Integration.** Generate translational data → normalize → check shapes preserved. | Shapes identical to input shapes. |

---

## Task 3: Intrinsic Coordinate Discovery (Process 1)

**Goal:** Build the autoencoder that discovers the minimal latent dimension, without identifying symmetry type.

**Files to create:**
- `intrinsic_coordinate/__init__.py`
- `intrinsic_coordinate/autoencoder.py` → `IntrinsicCoordinateAutoencoder`
- `intrinsic_coordinate/discovery.py` → `discover_latent_dimension()`

**Scope:**
- Autoencoder: `Input(n) → Encoder → Latent(k) → Decoder → Output(1)`.
- Train for `n_latent = 1, 2, ..., max_latent`. Record R² and MSE on validation set.
- Select minimal `n_latent` via elbow method (R² > 0.95 threshold as fallback).
- Return the best encoder, decoder, and all metrics.

**Validation Tests:**

| # | Test | Pass Criteria |
|---|------|---------------|
| 3.1 | **Known dimension — translational.** Generate translational data with `n_inputs=5, m_orbits=2` (true latent dim = 2). Run `discover_latent_dimension()`. | `optimal_n_latent == 2`. |
| 3.2 | **Known dimension — rotational.** Generate rotational data with `n_inputs=4` (true latent dim = 1). | `optimal_n_latent == 1`. |
| 3.3 | **Known dimension — scaling.** Generate scaling data with `n_inputs=3, m_scaling_vars=1` (true latent dim = 1). | `optimal_n_latent == 1`. |
| 3.4 | **R² quality.** At the optimal `n_latent`, validation R² > 0.95. | `metrics[optimal_n_latent]["R2"] > 0.95`. |
| 3.5 | **Monotonicity.** R² is non-decreasing (within noise) as `n_latent` increases. | Each R²(k+1) ≥ R²(k) − 0.02. |
| 3.6 | **Decoder reuse.** The returned `best_decoder` accepts a tensor of shape `(batch, n_latent)` and returns shape `(batch, 1)`. | Shape check passes. |

---

## Task 4: Symmetry Identification (Process 2)

**Goal:** Given the latent dimension from Task 3, identify which symmetry type best explains the data.

**Files to create:**
- `symmetry_discovery/__init__.py`
- `symmetry_discovery/encoders.py` → `SymmetryEncoder`
- `symmetry_discovery/identification.py` → `identify_symmetry()`

**Scope:**
- `SymmetryEncoder`: single linear layer, no hidden layers. Input transform depends on symmetry type (identity / square / log).
- Train each of the three encoder types using the frozen decoder from Task 3.
- Select the type with lowest reconstruction loss.
- Extract coefficients from the winning encoder's weight matrix.

**Validation Tests:**

| # | Test | Pass Criteria |
|---|------|---------------|
| 4.1 | **Type detection — translational.** Generate translational data → Task 3 → `identify_symmetry()`. | `symmetry_type == "translational"`. |
| 4.2 | **Type detection — rotational.** Same pipeline with rotational data. | `symmetry_type == "rotational"`. |
| 4.3 | **Type detection — scaling.** Same pipeline with scaling data. | `symmetry_type == "scaling"`. |
| 4.4 | **Loss gap.** The winning type's loss is at least 2× lower than the runner-up. | `loss_winner * 2 < loss_runner_up`. |
| 4.5 | **Coefficient recovery — rotational.** Ground truth `[1, 2, 1, 3]`. Detected coefficients (after normalization) match up to relative error < 0.1. | `max(abs(detected/detected[0] - truth/truth[0])) < 0.1`. |
| 4.6 | **Coefficient recovery — scaling.** Ground truth exponents `[[1, -1, 0]]`. Detected exponents (row of weight matrix, normalized) match up to relative error < 0.1. | Same criterion as 4.5 applied to exponent ratios. |

---

## Task 5: Generator Extraction (Process 3)

**Goal:** Extract the Lie-algebra generators from the trained encoder weights.

**Files to create:**
- `symmetry_discovery/generators.py` → `extract_generators()`, `apply_generator()`, `generator_orbit()`

**Scope:**
- Translational & Scaling: compute null space of `W` via SVD.
- Rotational: cluster columns of `W` by coefficient similarity, build antisymmetric matrices for each pair within a cluster.
- `apply_generator()`: apply infinitesimal transformation to data.
- `generator_orbit()`: trace an orbit from a starting point.

**Validation Tests:**

| # | Test | Pass Criteria |
|---|------|---------------|
| 5.1 | **Generator count — translational.** `n_inputs=5, n_latent=2` → 3 generators. | `n_generators == 3`. |
| 5.2 | **Generator count — rotational.** Coefficients `[1, 2, 1, 3]` → clusters {0,2} and {1},{3} → 1 generator (C(2,2)=1). | `n_generators == 1`. |
| 5.3 | **Generator count — scaling.** `n_inputs=3, n_latent=1` → 2 generators. | `n_generators == 2`. |
| 5.4 | **Algebraic check — translational.** For each generator `g`: `‖W·g‖ < 1e-6`. | All residuals below threshold. |
| 5.5 | **Algebraic check — rotational.** For each generator `A`: `‖A + Aᵀ‖ < 1e-6` (antisymmetry). | Residual below threshold. |
| 5.6 | **Algebraic check — scaling.** For each generator `s`: `‖W·s‖ < 1e-6`. | All residuals below threshold. |
| 5.7 | **Functional invariance.** Apply each generator to 100 test points with `ε=0.01`. Pass through encoder→decoder. Verify `|Δy| < 1e-3` for all points. | Max `|Δy|` across all generators < 1e-3. |
| 5.8 | **Orbit closure.** For a rotational generator, trace an orbit for a full loop (ε·n_steps ≈ 2π). Verify the endpoint returns near the start. | `‖x_end − x_start‖ < 0.1`. |

---

## Task 6: Verification, Visualization & End-to-End Demo

**Goal:** Build the verification suite, visualization functions, and a single end-to-end demo script.

**Files to create:**
- `verification/__init__.py`
- `verification/tests.py` → `verify_detection()`, `run_verification_suite()`
- `verification/test_generators.py` → `verify_generator_algebraic()`, `verify_generator_functional()`, `verify_generator_completeness()`, `verify_generators_against_ground_truth()`, `run_generator_verification_suite()`
- `verification/visualization.py` → `plot_latent_dimension_selection()`, `plot_symmetry_comparison()`, `plot_coefficient_comparison()`
- `verification/visualization_generators.py` → `plot_generator_field_2d()`, `plot_orbit()`, `plot_invariance_check()`, `plot_coefficient_clusters()`, `plot_generator_summary()`
- `examples/stage1_demo.py`

**Scope:**
- Verification functions compare detected results against ground truth.
- Visualization functions produce matplotlib figures for diagnostics.
- `stage1_demo.py` runs the full pipeline for all three symmetry types and prints a summary.

**Validation Tests:**

| # | Test | Pass Criteria |
|---|------|---------------|
| 6.1 | **Full pipeline — translational.** `stage1_demo.py` runs end-to-end for translational data. All verifications pass. | `all_passed == True`. |
| 6.2 | **Full pipeline — rotational.** Same for rotational. | `all_passed == True`. |
| 6.3 | **Full pipeline — scaling.** Same for scaling. | `all_passed == True`. |
| 6.4 | **Plots generate without error.** Each `plot_*` function returns a `matplotlib.figure.Figure` and does not raise. | No exceptions; returned object is `Figure`. |
| 6.5 | **Summary figure.** `plot_generator_summary()` produces a 2×2 subplot figure. | Figure has 4 axes. |
| 6.6 | **Noise robustness.** Run full pipeline with `noise_level=0.05`. Symmetry type still detected correctly for all three types. | `type_correct == True` for all three. |
| 6.7 | **Verification suite.** `run_verification_suite()` returns `overall_accuracy == 1.0`. | All three symmetry types pass. |

---

## Summary: Task Dependency Graph

```
Task 1: Data Generation
  └──► Task 2: Preprocessing
         └──► Task 3: Intrinsic Coordinate Discovery
                └──► Task 4: Symmetry Identification
                       └──► Task 5: Generator Extraction
                              └──► Task 6: Verification & Demo
```

Each task adds one module. The full pipeline connects them all in Task 6.

---

## Quick Reference: File → Task Mapping

| File | Task |
|------|------|
| `data_generation/translational.py` | 1 |
| `data_generation/rotational.py` | 1 |
| `data_generation/scaling.py` | 1 |
| `preprocessing/normalize.py` | 2 |
| `intrinsic_coordinate/autoencoder.py` | 3 |
| `intrinsic_coordinate/discovery.py` | 3 |
| `symmetry_discovery/encoders.py` | 4 |
| `symmetry_discovery/identification.py` | 4 |
| `symmetry_discovery/generators.py` | 5 |
| `verification/tests.py` | 6 |
| `verification/test_generators.py` | 6 |
| `verification/visualization.py` | 6 |
| `verification/visualization_generators.py` | 6 |
| `examples/stage1_demo.py` | 6 |
