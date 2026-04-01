# OpenSymmetry Stage 1: Method Development Using Synthetic Data

## Principles

1. **First-principles**: Build methods from fundamental mathematical concepts
2. **Occam's Razor**: Prefer the simplest explanation that fits the data
3. **Code quality**: Keep code simple, consistent, and stable
4. **Language**: All code and documentation in English

---

## Overview

Develop and verify symmetry detection methods using synthetic data for three symmetry types:

| Symmetry Type | Mathematical Form | Latent Variable |
|---------------|-------------------|-----------------|
| Translational | Linear combinations | `z = w · x` |
| Rotational | Quadratic form | `z = a·x₁² + b·x₂² + ...` |
| Scaling | Power law products | `z = x₁^a · x₂^b · ...` |

---

## Step 1: Repository Setup

```bash
git clone -b opensymmetry https://github.com/<org>/pydimension.git
cd pydimension
```

---

## Step 2: Data Generation Module

**Module:** `data_generation/`

### 2.1 Translational Symmetry

**File:** `data_generation/translational.py`

```python
def generate_translational_data(
    n_inputs: int,
    m_orbits: int,
    orbit_directions: np.ndarray,  # shape: (m_orbits, n_inputs)
    n_samples: int = 1000,
    noise_level: float = 0.0
) -> dict:
    """
    Generate synthetic data with translational symmetry.

    Returns:
        {
            "X": np.ndarray,                    # shape: (n_samples, n_inputs)
            "y": np.ndarray,                    # shape: (n_samples,)
            "orbit_directions": np.ndarray,    # ground truth for verification
            "orthogonal_directions": np.ndarray # orthogonal complement
        }
    """
```

**Math:**
```
latent_i = dot(x, orbit_direction_i)
y = f(latent_1, latent_2, ..., latent_m)  # nonlinear function
```

### 2.2 Rotational Symmetry

**File:** `data_generation/rotational.py`

```python
def generate_rotational_data(
    n_inputs: int,
    coefficients: np.ndarray,  # shape: (n_inputs,), values [a, b, c, ...]
    n_samples: int = 1000,
    noise_level: float = 0.0
) -> dict:
    """
    Generate synthetic data with rotational symmetry.

    Returns:
        {
            "X": np.ndarray,           # shape: (n_samples, n_inputs)
            "y": np.ndarray,           # shape: (n_samples,)
            "coefficients": np.ndarray # ground truth [a, b, c, ...] for verification
        }
    """
```

**Math:**
```
latent = a*x₁² + b*x₂² + c*x₃² + ...
y = f(latent)  # nonlinear function
```

### 2.3 Scaling Symmetry

**File:** `data_generation/scaling.py`

```python
def generate_scaling_data(
    n_inputs: int,
    m_scaling_vars: int,
    scaling_exponents: np.ndarray,  # shape: (m_scaling_vars, n_inputs)
    n_samples: int = 1000,
    noise_level: float = 0.0
) -> dict:
    """
    Generate synthetic data with scaling symmetry.

    Returns:
        {
            "X": np.ndarray,               # shape: (n_samples, n_inputs), positive values
            "y": np.ndarray,               # shape: (n_samples,)
            "scaling_vectors": np.ndarray  # ground truth exponents for verification
        }
    """
```

**Math:**
```
scaling_var_i = x₁^(a_i) * x₂^(b_i) * x₃^(c_i) * ...
y = f(scaling_var_1, scaling_var_2, ..., scaling_var_m)  # nonlinear function
```

---

## Step 3: Data Preprocessing Module

**Module:** `preprocessing/`

**File:** `preprocessing/normalize.py`

```python
def normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "standard"  # "standard", "minmax", "robust"
) -> dict:
    """
    Normalize input and output data.

    Returns:
        {
            "X_normalized": np.ndarray,
            "y_normalized": np.ndarray,
            "scaler_X": object,  # fitted scaler for inverse transform
            "scaler_y": object
        }
    """
```

---

## Step 4: Method Development

### Process 1: Intrinsic Coordinate Module

**Module:** `intrinsic_coordinate/`

**Objective:** Discover the minimal number of latent variables (without identifying symmetry type).

**Reference:** Champion et al., "Data-driven discovery of coordinates and governing equations"

**File:** `intrinsic_coordinate/autoencoder.py`

```python
class IntrinsicCoordinateAutoencoder(nn.Module):
    """
    Encoder-decoder architecture to discover intrinsic coordinates.

    Architecture:
        Input (n_inputs) → Encoder → Latent (n_latent) → Decoder → Output (1)
    """

    def __init__(
        self,
        n_inputs: int,
        n_latent: int,
        encoder_hidden: list[int] = [64, 32],
        decoder_hidden: list[int] = [32, 64]
    ):
        pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: (output, latent)"""
        pass
```

**File:** `intrinsic_coordinate/discovery.py`

```python
def discover_latent_dimension(
    X: np.ndarray,
    y: np.ndarray,
    max_latent: int = 4,
    n_epochs: int = 1000,
    validation_split: float = 0.2
) -> dict:
    """
    Train autoencoders with n_latent = 1, 2, 3, ..., max_latent.
    Select minimal latent dimension using elbow method.

    Returns:
        {
            "optimal_n_latent": int,
            "metrics": {
                1: {"R2": float, "MSE": float},
                2: {"R2": float, "MSE": float},
                ...
            },
            "best_encoder": nn.Module,  # trained encoder for Process 2
            "best_decoder": nn.Module
        }
    """
```

**Selection Criteria:**
- Compute R² and MSE for each `n_latent`
- Find elbow point where adding more latent dimensions yields diminishing returns
- Select minimal `n_latent` with acceptable R² (e.g., > 0.95)

---

### Process 2: Symmetry Discovery Module

**Module:** `symmetry_discovery/`

**Objective:** Identify the specific symmetry type from the trained encoder.

**File:** `symmetry_discovery/encoders.py`

```python
class SymmetryEncoder(nn.Module):
    """
    Single-layer linear encoder with input transformation.
    No hidden layers, no activation functions.

    Architecture:
        transform(Input) → Linear → Latent
    """

    def __init__(
        self,
        n_inputs: int,
        n_latent: int,
        symmetry_type: str  # "translational", "rotational", "scaling"
    ):
        self.symmetry_type = symmetry_type
        self.linear = nn.Linear(n_inputs, n_latent, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.symmetry_type == "translational":
            transformed = x                  # no transform
        elif self.symmetry_type == "rotational":
            transformed = x ** 2             # square inputs
        elif self.symmetry_type == "scaling":
            transformed = torch.log(x)       # log inputs
        return self.linear(transformed)
```

**File:** `symmetry_discovery/identification.py`

```python
def identify_symmetry(
    X: np.ndarray,
    y: np.ndarray,
    n_latent: int,
    decoder: nn.Module,  # from Process 1
    n_epochs: int = 500
) -> dict:
    """
    Train each symmetry encoder type and select the best fit.

    Returns:
        {
            "symmetry_type": str,           # "translational", "rotational", or "scaling"
            "coefficients": np.ndarray,     # extracted from encoder weights
            "all_losses": {
                "translational": float,
                "rotational": float,
                "scaling": float
            },
            "trained_encoder": nn.Module
        }
    """
```

**Encoder Summary:**

| Symmetry | Input Transform | Encoder | Extracted Coefficients |
|----------|-----------------|---------|------------------------|
| Translational | `x` | `Linear(n, k)` | Orbit directions (weight matrix rows) |
| Rotational | `x²` | `Linear(n, k)` | Quadratic coefficients [a, b, c, ...] |
| Scaling | `log(x)` | `Linear(n, k)` | Scaling exponents [a, b, c, ...] |

---

### Research Question: Unified Architecture

> Can we use a single NN structure with customized loss functions to distinguish symmetry types?

**File:** `symmetry_discovery/unified.py` (experimental)

```python
class UnifiedSymmetryEncoder(nn.Module):
    """
    Single encoder with learnable transform selection.
    """

    def __init__(self, n_inputs: int, n_latent: int):
        self.linear = nn.Linear(n_inputs, n_latent, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        transform: str = "none"
    ) -> torch.Tensor:
        if transform == "none":
            z = x
        elif transform == "square":
            z = x ** 2
        elif transform == "log":
            z = torch.log(x)
        return self.linear(z)


def symmetry_regularization_loss(
    weights: torch.Tensor,
    symmetry_type: str
) -> torch.Tensor:
    """
    Custom regularization to encourage symmetry-specific weight patterns.
    """
    if symmetry_type == "translational":
        # Encourage orthogonal weight vectors
        return orthogonality_loss(weights)
    elif symmetry_type == "rotational":
        # Encourage positive weights (for x² terms)
        return negative_penalty(weights)
    elif symmetry_type == "scaling":
        # Encourage sparse integer-like exponents
        return sparsity_loss(weights)
```

---

### Process 3: Generator Extraction Module

**Module:** `symmetry_discovery/`

**Objective:** Extract the infinitesimal generators of the detected symmetry group from the trained encoder weights. Generators describe the continuous transformations that leave the output y invariant.

**Theory:**

The encoder learns weight matrix `W ∈ ℝ^(k×n)` mapping (transformed) inputs to latent space. The symmetry generators are transformations in input space that leave the latent representation unchanged:

| Symmetry | Generator Type | Defining Relation | Group Action |
|----------|---------------|-------------------|--------------|
| Translational | Vectors `g ∈ ℝⁿ` | `W·g = 0` (null space of W) | `x → x + ε·g` |
| Rotational | Antisymmetric matrices `A ∈ ℝⁿˣⁿ` | `A·Q + Q·Aᵀ = 0`, where `Q = diag(W)` | `x → exp(ε·A)·x` |
| Scaling | Vectors `s ∈ ℝⁿ` | `W·s = 0` (null space of W in log-space) | `xᵢ → λ^(sᵢ)·xᵢ` |

**Generator count (Lie algebra dimension):**
- Translational: `n - k` (dimension of null space)
- Rotational: `Σ C(cᵢ, 2)` where `cᵢ` is the size of the i-th coefficient cluster
- Scaling: `n - k` (dimension of null space in log-space)

**File:** `symmetry_discovery/generators.py`

```python
def extract_generators(
    trained_encoder: nn.Module,
    symmetry_type: str,
    n_inputs: int,
    n_latent: int,
    tol: float = 1e-2
) -> dict:
    """
    Extract symmetry generators from trained encoder weights.

    Args:
        trained_encoder: Trained SymmetryEncoder with a .linear attribute.
        symmetry_type: One of "translational", "rotational", "scaling".
        n_inputs: Dimension of input space.
        n_latent: Dimension of latent space (rank of W).
        tol: Tolerance for coefficient clustering (rotational case).

    Returns:
        {
            "generators": list[np.ndarray],
                Translational: list of vectors in R^n (translation directions).
                Rotational: list of antisymmetric matrices in R^(n x n).
                Scaling: list of vectors in R^n (log-space directions).
            "generator_type": str,
                "translation_vector", "rotation_matrix", or "scaling_vector".
            "n_generators": int,
                Dimension of the continuous symmetry group.
            "lie_algebra_dimension": int,
                Expected dimension from theory.
        }
    """
```

**Extraction methods:**

**Translational / Scaling:** SVD of the weight matrix W. The right singular vectors beyond the numerical rank span the null space. Each null vector is a generator.

```python
U, S, Vt = np.linalg.svd(W, full_matrices=True)
rank = np.sum(S > S[0] * 1e-6)
generators = [Vt[i] for i in range(rank, Vt.shape[0])]
```

**Rotational:** Cluster the columns of W by coefficient similarity. Each pair of columns within a cluster yields an antisymmetric generator matrix (a rotation generator in that 2D plane).

```python
# For each pair (i, j) where W[:, i] ≈ W[:, j]:
A = np.zeros((n_inputs, n_inputs))
A[i, j] = 1.0
A[j, i] = -1.0
```

**Finite group action helpers:**

```python
def apply_generator(
    X: np.ndarray,
    generator: np.ndarray,
    symmetry_type: str,
    epsilon: float = 0.01
) -> np.ndarray:
    """
    Apply an infinitesimal generator to data points.

    Translational: x → x + ε·g
    Rotational:    x → exp(ε·A)·x  (matrix exponential)
    Scaling:       xᵢ → exp(ε·sᵢ)·xᵢ

    Returns:
        X_transformed: shape (n_samples, n_inputs)
    """


def generator_orbit(
    x0: np.ndarray,
    generator: np.ndarray,
    symmetry_type: str,
    n_steps: int = 100,
    epsilon: float = 0.05
) -> np.ndarray:
    """
    Trace an orbit by repeatedly applying an infinitesimal generator.

    Returns:
        orbit: shape (n_steps + 1, n_inputs), including starting point.
    """
```

**Design note — tolerance in the rotational case:**

The coefficient clustering tolerance `tol` controls whether near-equal coefficients (e.g., 1.98 vs 2.02) are treated as identical. On synthetic data this is straightforward, but on real data this becomes the key decision point for approximate symmetries. Sensitivity analysis of `tol` on synthetic data with controlled noise is recommended before moving to real data.

---

### Process 4: Verification and Visualization Module

**Module:** `verification/`

**File:** `verification/tests.py`

```python
def verify_detection(
    detected_type: str,
    detected_coefficients: np.ndarray,
    ground_truth_type: str,
    ground_truth_coefficients: np.ndarray
) -> dict:
    """
    Verify symmetry detection against ground truth.

    Returns:
        {
            "type_correct": bool,
            "coefficient_error": float,  # relative error norm
            "coefficient_correlation": float
        }
    """


def run_verification_suite() -> dict:
    """
    Run full verification on all symmetry types.

    Test cases:
        - Translational: n_inputs=5, m_orbits=2
        - Rotational: n_inputs=4, coefficients=[1, 2, 1, 3]
        - Scaling: n_inputs=3, exponents=[[1, -1, 0], [0, 1, -1]]

    Returns:
        {
            "translational": {"passed": bool, "details": dict},
            "rotational": {"passed": bool, "details": dict},
            "scaling": {"passed": bool, "details": dict},
            "overall_accuracy": float
        }
    """
```

**File:** `verification/test_generators.py`

```python
def verify_generator_algebraic(
    generator: np.ndarray,
    W: np.ndarray,
    symmetry_type: str,
    tol: float = 1e-3
) -> dict:
    """
    Check that the generator satisfies the algebraic defining relation.

    Translational: ||W·g|| ≈ 0
    Rotational:    ||A + Aᵀ|| ≈ 0 and ||A·Q + Q·Aᵀ|| ≈ 0
    Scaling:       ||W·s|| ≈ 0

    Returns:
        {
            "passed": bool,
            "residual": float,
            "details": str
        }
    """


def verify_generator_functional(
    generator: np.ndarray,
    X: np.ndarray,
    symmetry_type: str,
    encoder: nn.Module,
    decoder: nn.Module,
    epsilon: float = 0.01,
    atol: float = 1e-3
) -> dict:
    """
    Apply the generator as a finite transformation and check y-invariance.
    This is the ground-truth test: moving along a true generator should not
    change the model's prediction.

    Returns:
        {
            "passed": bool,
            "max_y_change": float,
            "mean_y_change": float,
            "relative_change": float
        }
    """


def verify_generator_completeness(
    generators: list[np.ndarray],
    symmetry_type: str,
    n_inputs: int,
    n_latent: int,
    W: np.ndarray,
    tol: float = 1e-2
) -> dict:
    """
    Check that the correct number of independent generators were found.

    Translational/Scaling: n_generators = n_inputs - n_latent
    Rotational: n_generators = Σ C(cluster_size, 2)

    Returns:
        {
            "passed": bool,
            "n_found": int,
            "n_expected": int,
            "independent": bool,
            "details": str
        }
    """


def verify_generators_against_ground_truth(
    generators: list[np.ndarray],
    symmetry_type: str,
    ground_truth: dict,
    tol: float = 1e-2
) -> dict:
    """
    Compare extracted generators against known ground truth.

    Translational/Scaling: subspace angle between detected and true null spaces.
    Rotational: check that detected rotation planes match ground truth
                coefficient equalities.

    Returns:
        {
            "passed": bool,
            "subspace_angle": float,  # translational/scaling
            "planes_correct": bool,   # rotational
            "details": str
        }
    """


def run_generator_verification_suite(
    extract_result: dict,
    X: np.ndarray,
    symmetry_type: str,
    encoder: nn.Module,
    decoder: nn.Module,
    ground_truth: dict,
    n_inputs: int,
    n_latent: int
) -> dict:
    """
    Run the full generator verification pipeline.

    Returns:
        {
            "algebraic": list[dict],
            "functional": list[dict],
            "completeness": dict,
            "ground_truth": dict,
            "all_passed": bool
        }
    """
```

**File:** `verification/visualization.py`

```python
def plot_latent_dimension_selection(metrics: dict) -> plt.Figure:
    """Plot R² and MSE vs n_latent with elbow point marked."""


def plot_symmetry_comparison(all_losses: dict) -> plt.Figure:
    """Bar chart comparing reconstruction loss for each symmetry type."""


def plot_coefficient_comparison(
    detected: np.ndarray,
    ground_truth: np.ndarray,
    symmetry_type: str
) -> plt.Figure:
    """Scatter plot of detected vs ground truth coefficients."""
```

**File:** `verification/visualization_generators.py`

```python
def plot_generator_field_2d(
    generator: np.ndarray,
    symmetry_type: str,
    xlim: tuple[float, float] = (-3.0, 3.0),
    ylim: tuple[float, float] = (-3.0, 3.0),
    n_grid: int = 15,
    dims: tuple[int, int] = (0, 1),
    ax: plt.Axes = None,
    title: str = None
) -> plt.Figure:
    """
    Plot the generator as a 2D vector field projected onto two input dimensions.

    Translational: constant parallel arrows.
    Rotational: circular rotation pattern.
    Scaling: radial outward/inward pattern.
    """


def plot_orbit(
    orbit: np.ndarray,
    dims: tuple[int, int] = (0, 1),
    y_values: np.ndarray = None,
    ax: plt.Axes = None,
    title: str = None
) -> plt.Figure:
    """
    Plot an orbit trajectory in 2D, optionally colored by y-value.
    If the generator is correct, y-value should be constant along the orbit.
    """


def plot_invariance_check(
    functional_results: list[dict],
    generator_labels: list[str] = None,
    ax: plt.Axes = None
) -> plt.Figure:
    """
    Bar chart showing max and mean |Δy| for each generator (log scale).
    A correct generator should have near-zero bars.
    """


def plot_coefficient_clusters(
    W: np.ndarray,
    clusters: list[list[int]],
    ax: plt.Axes = None
) -> plt.Figure:
    """
    Visualize coefficient clustering for rotational symmetry.
    Shows encoder weights colored by cluster membership.
    Clusters with 2+ members admit rotation generators.
    """


def plot_generator_summary(
    extract_result: dict,
    verification_result: dict,
    symmetry_type: str,
    W: np.ndarray
) -> plt.Figure:
    """
    Combined summary figure with all generator diagnostics.

    Layout:
        Top-left:     Vector field of first generator (2D projection)
        Top-right:    Invariance bar chart
        Bottom-left:  Coefficient clusters (rotational) or SVD spectrum
        Bottom-right: Verification summary table
    """
```

---

## Project Structure

```
pydimension/
├── data_generation/
│   ├── __init__.py
│   ├── translational.py
│   ├── rotational.py
│   └── scaling.py
├── preprocessing/
│   ├── __init__.py
│   └── normalize.py
├── intrinsic_coordinate/
│   ├── __init__.py
│   ├── autoencoder.py
│   └── discovery.py
├── symmetry_discovery/
│   ├── __init__.py
│   ├── encoders.py
│   ├── identification.py
│   ├── generators.py       # generator extraction (Process 3)
│   └── unified.py          # experimental
├── verification/
│   ├── __init__.py
│   ├── tests.py
│   ├── test_generators.py          # generator verification
│   └── visualization.py
│   └── visualization_generators.py # generator visualization
└── examples/
    └── stage1_demo.py      # end-to-end demonstration
```

---

## Dependencies

```
numpy>=1.21
torch>=2.0
scipy>=1.7
scikit-learn>=1.0
matplotlib>=3.5
pytest>=7.0
```

---

## Usage Example

```python
from data_generation import generate_rotational_data
from preprocessing import normalize_data
from intrinsic_coordinate import discover_latent_dimension
from symmetry_discovery import identify_symmetry
from symmetry_discovery.generators import extract_generators
from verification import verify_detection
from verification.test_generators import run_generator_verification_suite
from verification.visualization_generators import plot_generator_summary

# Step 1: Generate data
data = generate_rotational_data(
    n_inputs=4,
    coefficients=np.array([1.0, 2.0, 1.0, 3.0]),
    n_samples=2000
)

# Step 2: Normalize
normalized = normalize_data(data["X"], data["y"])

# Step 3: Discover latent dimension
result = discover_latent_dimension(
    normalized["X_normalized"],
    normalized["y_normalized"],
    max_latent=4
)
print(f"Optimal latent dimension: {result['optimal_n_latent']}")

# Step 4: Identify symmetry type
symmetry = identify_symmetry(
    normalized["X_normalized"],
    normalized["y_normalized"],
    n_latent=result["optimal_n_latent"],
    decoder=result["best_decoder"]
)
print(f"Detected symmetry: {symmetry['symmetry_type']}")
print(f"Coefficients: {symmetry['coefficients']}")

# Step 5: Extract generators
gen_result = extract_generators(
    trained_encoder=symmetry["trained_encoder"],
    symmetry_type=symmetry["symmetry_type"],
    n_inputs=4,
    n_latent=result["optimal_n_latent"]
)
print(f"Found {gen_result['n_generators']} generators")
print(f"Generator type: {gen_result['generator_type']}")

# Step 6: Verify symmetry detection
verification = verify_detection(
    symmetry["symmetry_type"],
    symmetry["coefficients"],
    "rotational",
    data["coefficients"]
)
print(f"Verification passed: {verification['type_correct']}")

# Step 7: Verify generators
gen_verification = run_generator_verification_suite(
    extract_result=gen_result,
    X=normalized["X_normalized"],
    symmetry_type=symmetry["symmetry_type"],
    encoder=symmetry["trained_encoder"],
    decoder=result["best_decoder"],
    ground_truth={"coefficients": data["coefficients"]},
    n_inputs=4,
    n_latent=result["optimal_n_latent"]
)
print(f"Generator verification: {'ALL PASSED' if gen_verification['all_passed'] else 'SOME FAILED'}")

# Step 8: Visualize generator diagnostics
fig = plot_generator_summary(
    gen_result, gen_verification,
    symmetry["symmetry_type"],
    symmetry["trained_encoder"].linear.weight.detach().numpy()
)
fig.savefig("generator_summary.png", dpi=150, bbox_inches="tight")
```
