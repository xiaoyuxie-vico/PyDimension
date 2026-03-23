# Stage 1 Tasks: Review & Potential Problems

**Reviewer:** Claude (AI-assisted review)
**Date:** 2026-03-23
**Documents reviewed:**
- `stage1_tasks.md`
- `stage1_using_synthetic_data.md`

---

## How to Read This Document

Each issue is tagged with a severity level and the task(s) it affects. Fix **Critical** issues before writing any code — they are mathematical or architectural bugs that will propagate through the entire pipeline. **High** issues should be resolved during design. **Medium** and **Low** issues can be addressed during implementation.

---

## Critical Issues

### C1. Test 1.5 — Scaling Invariance Test Uses Wrong Symmetry Direction

**Affects:** Task 1 (Test 1.5)
**Type:** Mathematical error in the test specification

**What the test says:**
> Generate scaling data with exponents `[2, -1]`. Scale inputs as `x₁ → λ²·x₁, x₂ → λ⁻¹·x₂`. Verify `y` unchanged.

**Why it's wrong:**

The scaling variable is `z = x₁² · x₂⁻¹`. The test applies the transformation `xᵢ → λ^(aᵢ)·xᵢ` where the scaling direction `[a₁, a₂] = [2, -1]` equals the exponents themselves. Check:

```
z' = (λ²·x₁)² · (λ⁻¹·x₂)⁻¹
   = λ⁴·x₁² · λ¹·x₂⁻¹
   = λ⁵ · z
```

The output picks up a factor of `λ⁵`. This is **not invariant** — the test will always fail.

**The correct scaling direction** is the null space of the exponent vector `W = [2, -1]`:

```
Null space: s such that W·s = 0  →  2·s₁ + (-1)·s₂ = 0  →  s = t·[1, 2]
```

The correct test should scale as `x₁ → λ¹·x₁, x₂ → λ²·x₂`:

```
z' = (λ·x₁)² · (λ²·x₂)⁻¹
   = λ²·x₁² · λ⁻²·x₂⁻¹
   = z  ✓
```

**Fix:** Replace the scaling direction in test 1.5 with the null-space vector `[1, 2]`:
> Scale inputs as `x₁ → λ·x₁, x₂ → λ²·x₂`. Verify `y` unchanged.

---

### C2. Contradictory Translational Symmetry Definitions Between Documents

**Affects:** Task 1 (data generation), Task 3 (latent dimension), Task 4 (symmetry identification)
**Type:** Conceptual inconsistency

**Document 1 — `stage1_tasks.md` (line 23):**
> Translational: `y = f(W·x)` where `W` is the orthogonal complement of orbit directions.

This means: W contains the directions that y **depends on** (perpendicular to orbits). The orbit directions are the null space of W — directions along which y is invariant. This is mathematically correct.

**Document 2 — `stage1_using_synthetic_data.md` (lines 64–65):**
```
latent_i = dot(x, orbit_direction_i)
y = f(latent_1, latent_2, ..., latent_m)
```

This means: y depends **directly on the orbit directions**. If y depends on the orbit direction, then shifting along it changes y — it is not an invariant direction. This **contradicts** the symmetry.

**Additionally, the parameter `m_orbits` is ambiguous:**

Test 3.1 says: `n_inputs=5, m_orbits=2 → true latent dim = 2`.

- If `m_orbits=2` means 2 invariant orbit directions in 5D, the orthogonal complement is 3D, so latent dim = 3 (not 2).
- If `m_orbits=2` means 2 measurement directions (rank of W), then latent dim = 2 is correct, but the name "m_orbits" is misleading.

**Fix (two options):**

**Option A — Fix `stage1_using_synthetic_data.md`:**
Rename `orbit_directions` to `measurement_directions` (or `weight_directions`). Change the math to:
```
latent_i = dot(x, measurement_direction_i)
y = f(latent_1, ..., latent_k)     # k = number of measurement directions
orbit directions = null space of W  # n - k invariant directions
```

**Option B — Fix `stage1_tasks.md`:**
Change the description to:
> `y = f(W·x)` where rows of `W` are the measurement directions (NOT the orbit directions).

In either case, rename `m_orbits` to `n_latent` or `m_measurements` to avoid confusion.

---

### C3. Standard Normalization Destroys Scaling Data Positivity

**Affects:** Task 2 → Task 4 pipeline
**Type:** Silent pipeline failure

The data flow is:
```
Task 1: generate_scaling_data()           → X > 0 guaranteed (test 1.7)
Task 2: normalize_data(X, "standard")     → X can be negative (centered to mean 0)
Task 4: SymmetryEncoder("scaling")        → torch.log(X_norm) → NaN or -inf
```

Standard normalization (zero mean, unit variance) makes approximately half the values negative. `log(negative)` is undefined. The pipeline will crash or silently produce `NaN` values that propagate through training.

**This also affects minmax normalization** if the range is [0, 1], since `log(0) = -inf`.

**Fix:** Specify normalization method per symmetry type, or add a constraint:
- Scaling data: use `method="log_standard"` (log-transform first, then standardize), or skip normalization for X entirely and only normalize y.
- Alternative: in the SymmetryEncoder, use `log(abs(x) + epsilon)` as a safe log transform. But this changes the mathematical meaning.
- Best approach: add a `positivity_preserving` option to `normalize_data()` that shifts the data to be strictly positive before normalization.

---

## High Severity Issues

### H1. Frozen Decoder May Be Incompatible with Linear Encoder

**Affects:** Task 3 → Task 4 transition
**Type:** Architectural risk

Task 3 trains a nonlinear encoder-decoder:
```
x → [nonlinear encoder with hidden layers] → z → [decoder] → y
```

Task 4 replaces the encoder with a single linear layer and freezes the decoder:
```
transform(x) → [linear layer, no hidden layers] → z → [frozen decoder] → y
```

The decoder was trained to interpret latent vectors `z` produced by a specific nonlinear encoder. These vectors live on a specific manifold in latent space. The new linear encoder must produce vectors on the **same manifold** for the decoder to work correctly.

**When this fails:** If Task 3's encoder learns a highly nonlinear mapping (e.g., `z = tanh(W₂·relu(W₁·x))`), the resulting latent manifold may not be reachable by any linear function of `transform(x)`. All three symmetry encoders will have poor loss, making discrimination impossible.

**When this works:** If the decoder is "flexible enough" that many different latent representations can produce good y-predictions, the linear encoder can find a compatible mapping through training.

**Fix options:**
1. **Constrain Task 3's encoder** to be as linear as possible (shallow, wide, with soft activation like tanh instead of ReLU), so the latent space is approximately linear.
2. **Retrain the decoder** jointly with each SymmetryEncoder in Task 4. This loses the "controlled comparison" aspect but is more robust.
3. **Add a small adapter layer** between the SymmetryEncoder and the frozen decoder (e.g., a single hidden layer) that can bridge the gap. This is a compromise.
4. **Use the decoder only for scoring, not frozen:** Train `SymmetryEncoder + new decoder` for each type, compare final losses.

---

### H2. Tests 3.1–3.3 Require Exact Integer Outcomes from Stochastic Training

**Affects:** Task 3 (Tests 3.1, 3.2, 3.3)
**Type:** Flaky test design

```
Test 3.1: optimal_n_latent == 2  (must be exactly 2)
Test 3.2: optimal_n_latent == 1  (must be exactly 1)
Test 3.3: optimal_n_latent == 1  (must be exactly 1)
```

Neural network training involves random weight initialization, random batch ordering, and random validation splitting. The elbow/threshold selection can easily choose a different integer depending on the random seed. These tests will pass ~80–90% of the time and fail ~10–20%.

**Fix:**
1. Fix random seeds (`np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) for all validation tests.
2. Alternatively, run the selection 5 times and take the majority vote.
3. At minimum, allow `optimal_n_latent` to be within ±1 of the expected value and test that R² at the true dimension is above threshold.

---

### H3. The Elbow Method Is Algorithmically Undefined

**Affects:** Task 3
**Type:** Ambiguous specification

The spec says:
> "Select minimal n_latent via elbow method (R² > 0.95 threshold as fallback)."

Undefined aspects:
- **Which elbow algorithm?** Kneedle algorithm? Maximum curvature? Second derivative? Visual inspection?
- **When does the elbow method apply vs. the R² > 0.95 fallback?** Always try elbow first? Use elbow only if R² never reaches 0.95?
- **What if R² reaches 0.95 at n_latent=1 but the true dimension is 2?** The threshold alone would select 1.
- **What if R² never reaches 0.95?** Select the elbow regardless?

Two developers implementing "elbow method" independently will produce different selection logic and different results on the same data.

**Fix:** Define the algorithm precisely. Recommended approach:
```python
# For each k from 1 to max_latent:
#   if R2[k] > 0.95 and (k == 1 or R2[k] - R2[k-1] < 0.05):
#       return k
# Fallback: return argmax(R2)
```
Or specify the Kneedle algorithm with specific parameters (sensitivity, curve direction, etc.).

---

### H4. Test 4.4 — 2× Loss Gap Has No Theoretical Basis

**Affects:** Task 4 (Test 4.4)
**Type:** Arbitrary threshold

```
loss_winner * 2 < loss_runner_up
```

**Why it can fail on valid data:**

For small positive inputs near 1, `log(x) ≈ x - 1` (Taylor expansion). This means the scaling encoder (`W·log(x)`) and translational encoder (`W·x`) will produce similar latent representations. For translational data with inputs uniformly drawn near 1.0, the loss gap between translational and scaling can easily be less than 2×.

Similarly, for large inputs where `x²` dominates, the rotational and scaling encoders can have similar performance.

**Fix:** Replace the 2× gap with a statistical test (e.g., the winning type's loss is significantly lower by a paired t-test on per-sample losses), or simply remove this test and rely on tests 4.1–4.3 (correct type detection).

---

### H5. Test 4.6 — Coefficient Comparison Can Hit Division by Zero

**Affects:** Task 4 (Test 4.6)
**Type:** Undefined edge case

Ground truth exponents: `[[1, -1, 0]]`. The comparison formula from test 4.5 is:
```
max(abs(detected/detected[0] - truth/truth[0])) < 0.1
```

For the ground truth `[1, -1, 0]`, normalizing by the first element gives `[1, -1, 0]`. But if the normalization is element-wise (detected[i]/truth[i]), then `detected[2] / truth[2]` = `anything / 0` — **division by zero**.

Even with the ratio-to-first-element approach, if `detected[0] ≈ 0` (due to training noise), you get division by zero on the detected side.

**Fix:** Use a robust comparison metric:
```python
# Normalize both to unit vectors, then compare
detected_norm = detected / np.linalg.norm(detected)
truth_norm = truth / np.linalg.norm(truth)
error = np.linalg.norm(detected_norm - truth_norm)  # or use subspace angle
```

---

## Medium Severity Issues

### M1. The "Autoencoder" Is Actually a Bottleneck Regression Network

**Affects:** Task 3
**Type:** Misleading terminology

The architecture is:
```
Input(n_inputs) → Encoder → Latent(k) → Decoder → Output(1)
```

A standard autoencoder reconstructs its **input**: `Input(n) → Latent(k) → Output(n)`. This architecture predicts a **scalar output y**, not the input. It is a regression network with a bottleneck.

**Why it matters:**
- Standard autoencoder theory (e.g., relationship between latent dim and PCA) does not apply.
- The R² metric measures y-prediction quality, not reconstruction quality. A high R² at `n_latent=k` means "k dimensions are sufficient to predict y," not "k dimensions capture the input variance."
- Literature references to autoencoders will mislead developers.

**Fix:** Rename to `BottleneckRegressor` or `IntrinsicCoordinateRegressor`. Update documentation to clarify that this is a **regression** task, not a **reconstruction** task.

---

### M2. Test 5.8 — Euler Integration Will Drift Beyond Orbit Closure Tolerance

**Affects:** Task 5 (Test 5.8)
**Type:** Numerical method limitation

The orbit is traced by repeated application of `x → x + ε·A·x` (forward Euler). For a rotation generator A with `ε=0.05`, a full loop requires ~`2π/0.05 ≈ 126` steps.

Euler's method does **not preserve the norm** for rotational systems. Each step increases the radius slightly:
```
‖x_{k+1}‖² = ‖x_k + ε·A·x_k‖² = ‖x_k‖² + 2ε·x_k^T·A·x_k + ε²·‖A·x_k‖²
```
Since A is antisymmetric, `x^T·A·x = 0`, so `‖x_{k+1}‖² = ‖x_k‖²·(1 + ε²·‖A‖²)`. After 126 steps:
```
‖x_end‖/‖x_start‖ ≈ (1 + 0.0025)^63 ≈ 1.17
```
The orbit spirals outward by ~17%, easily exceeding the `‖x_end − x_start‖ < 0.1` tolerance.

**Fix:** Use the matrix exponential for exact integration:
```python
from scipy.linalg import expm
x_next = expm(epsilon * A) @ x_current
```
This preserves the norm exactly and the orbit will close properly.

---

### M3. Module Structure Conflicts with Existing Codebase

**Affects:** All tasks
**Type:** Integration conflict

The tasks create new top-level modules:
```
data_generation/          ← new
preprocessing/            ← new
intrinsic_coordinate/     ← new
symmetry_discovery/       ← new
verification/             ← new
```

The existing repository already contains:
```
pydimension/data_preprocessing/     (normalizer.py, transforms.py, pipeline.py, etc.)
pydimension/intrinsic_coordinate/   (autoencoder.py, pca.py, sir.py, engine.py)
pydimension/symmetry_discovery/     (engine.py, scoring.py)
```

**Problems:**
- Import path confusion: `from intrinsic_coordinate.autoencoder import ...` vs `from pydimension.intrinsic_coordinate.autoencoder import ...`
- Duplicate file names (both old and new have `autoencoder.py`)
- Unclear whether to extend existing modules or replace them

**Fix:** Decide explicitly:
- **Option A:** Build inside `pydimension/` namespace, extending existing modules.
- **Option B:** Create a new top-level package (e.g., `opensymmetry/`) completely separate from `pydimension/`.
- **Option C:** Replace the existing modules entirely (document what is being removed and why).

---

### M4. No Overfitting Prevention

**Affects:** Tasks 3 and 4
**Type:** Missing training safeguard

Task 3 specifies `n_epochs=1000` with `n_samples=1000` and encoder hidden layers `[64, 32]`. After a 20% validation split, that's 800 training samples for ~3,000+ parameters. Overfitting is highly likely.

Missing from the spec:
- Early stopping (monitor validation loss, stop when it increases)
- Weight decay / L2 regularization
- Dropout
- Learning rate scheduling
- Batch size

**Fix:** At minimum, add early stopping with patience ~50 epochs. Specify `weight_decay=1e-4` for the optimizer.

---

### M5. Rotational Case Is Structurally Harder for the Bottleneck Network

**Affects:** Task 3 (Test 3.2)
**Type:** Architecture limitation

For rotational data, `y = f(a·x₁² + b·x₂² + ...)`. The true latent variable is a **quadratic** function of inputs. The encoder must learn this quadratic mapping using nonlinear hidden layers.

Whether a `[64, 32]` encoder with (unspecified) activation can learn an accurate quadratic mapping with 1000 samples depends heavily on:
- Activation function: ReLU approximates piecewise linear, which needs many pieces for a quadratic. Tanh is smoother and may be better.
- Number of inputs: with `n_inputs=4`, the quadratic has 4 terms. With `n_inputs=10`, it has 10 terms. Difficulty scales with input dimension.
- Sample count: 1000 samples may be insufficient for higher dimensions.

Test 3.2 may be the most fragile of all latent dimension tests.

**Fix:** Recommend Tanh or GELU activation for the encoder. Consider increasing samples to 2000+ for rotational data. Add a note that rotational detection may need more training or wider networks.

---

### M6. Generator Count Depends on Learned (Not Ground-Truth) Coefficients

**Affects:** Task 5 (Test 5.2)
**Type:** Cascading error from Task 4

Test 5.2 expects 1 generator for rotational data with coefficients `[1, 2, 1, 3]`. This relies on the clustering step finding the cluster `{0, 2}` (both have coefficient 1). But:
- The coefficients come from Task 4's trained encoder, not ground truth.
- If Task 4 recovers `[1.0, 2.1, 1.05, 2.9]`, clustering with tight tolerance may fail to group indices 0 and 2.
- If Task 4 recovers `[1.0, 2.0, 1.0, 2.0]` (merging 2 and 3), clustering may find too many generators.

The tolerance parameter `tol` in `extract_generators()` controls this, but its optimal value depends on the noise level and training quality — neither of which are known at extraction time.

**Fix:** Make the tolerance adaptive (e.g., based on the spread of detected coefficients), or provide a clear guideline: `tol = 0.1 * max(abs(coefficients))`.

---

## Low Severity Issues

### L1. No Fixed Random Seeds Anywhere

**Affects:** All tasks
**Fix:** Add `seed` parameter to all functions and fix it in all validation tests.

### L2. No Device Specification (CPU vs GPU)

**Affects:** Tasks 3, 4, 5
**Fix:** Add `device` parameter to PyTorch-using functions. Default to `"cpu"` for reproducibility.

### L3. Noise Model Is Simplistic

**Affects:** Task 1
**Details:** Additive Gaussian noise on y only. Real data has input noise, heteroscedastic noise, and correlated noise. Results may not transfer to real data.
**Fix:** Consider adding `input_noise_level` parameter for future robustness testing.

### L4. No "No Symmetry" Detection Option

**Affects:** Task 4
**Details:** The pipeline always picks the best of three symmetry types. If the data has no symmetry (or a different type), the pipeline will confidently return a wrong answer with no way to flag uncertainty.
**Fix:** Add a baseline "no symmetry" option (e.g., generic nonlinear encoder). If the baseline loss is comparable to the best symmetry encoder, flag "no clear symmetry detected."

### L5. Test 2.1 Tolerance Is Inconsistent with Other Tests

**Affects:** Task 2 (Test 2.1)
**Details:** `abs(std - 1) < 1e-10` is extremely tight, while Tasks 5–6 use `1e-3` to `1e-6` tolerances. The tight tolerance is achievable (sklearn uses ddof=0 on training data) but is inconsistent with the rest of the spec.
**Fix:** Align tolerances across all tests or document why different levels are used.

### L6. Validation Split Not Specified

**Affects:** Task 3
**Details:** `validation_split=0.2` is specified but not whether the split is random, stratified, or fixed. Different splits lead to different R² values and potentially different latent dimension selection.
**Fix:** Use a fixed split (e.g., last 20% of samples) or fix the random seed for splitting.

---

## Recommended Action Plan

### Before Writing Code (fix the spec)

1. **Fix Test 1.5 math** — replace `[2, -1]` scaling direction with null-space vector `[1, 2]` (Issue C1)
2. **Resolve translational symmetry contradiction** — unify terminology across both documents, rename `m_orbits` to `n_latent` or `m_measurements` (Issue C2)
3. **Specify normalization per symmetry type** — scaling data must not be standard-normalized before log transform (Issue C3)
4. **Define the elbow algorithm precisely** — pick a specific method with specific parameters (Issue H3)
5. **Fix coefficient comparison** — use unit-vector normalization instead of ratio-based (Issue H5)
6. **Decide module placement** — inside `pydimension/` or separate top-level package (Issue M3)

### During Implementation

7. Add fixed random seeds to all validation tests (L1)
8. Use `scipy.linalg.expm` for orbit integration instead of Euler (M2)
9. Add early stopping with patience ~50 epochs (M4)
10. Use Tanh or GELU activation for encoder (M5)
11. Make clustering tolerance adaptive (M6)

### Future Considerations

12. Add a "no symmetry" baseline to Task 4 (L4)
13. Consider mixed-symmetry detection for real data
14. Add input noise parameters for robustness testing (L3)
15. Evaluate whether retraining the decoder in Task 4 (instead of freezing) improves results (H1)
