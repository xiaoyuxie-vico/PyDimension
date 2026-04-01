# PyDimension 3.0 Code Structure Task

## Goal

Design the `PyDimension 3.0` architecture so that it can evolve naturally into `OpenSymmetry`.

The design should move the codebase from a dimensionless-learning-only pipeline to a general symmetry-aware scientific discovery framework, while keeping the implementation concise, readable, and internally consistent.

## General Principles

### First Principles

PyDimension 3.0 should be designed from first principles:

- Start from the scientific object we want to discover: hidden symmetry in data.
- Separate shared infrastructure from symmetry-specific methods.
- Make every module name reflect its role clearly.
- Avoid duplicated logic across symmetry classes.
- Prefer a small number of strong abstractions over many ad hoc scripts.

### Simplicity

The code and documentation should stay simple:

- short module names,
- clear interfaces,
- minimal hidden coupling,
- minimal special cases,
- consistent naming across package, configs, scripts, and docs.

### Consistency

The same concepts should use the same names everywhere:

- data generation,
- data preprocessing,
- intrinsic coordinates,
- symmetry encoders,
- discovery,
- results,
- benchmarks.

### Extensibility

The new structure should make it easy to add:

- translational symmetry,
- rotational symmetry,
- scaling symmetry,
- future symmetry classes,
- future intrinsic-coordinate methods,
- future downstream discovery methods.

## Main Structural Changes

## 1. Keep `data_generation`, but generalize it

`data_generation` should be kept as an independent module.

However, the current data generation logic is effectively tailored to the current translational/log-transformed style benchmark workflow and should be expanded to support multiple symmetry-aware synthetic generators.

### Required change

`data_generation` should provide multiple generators, for example:

- translational generator,
- rotational generator,
- scaling generator,
- mixed or composite symmetry generator later.

### Intent

The module should become the benchmark and synthetic-data engine for all symmetry classes, not just the current dimensionless learning examples.

## 2. Merge `DATA_PREPROCESSING` and `dimensional_analysis` into `data_preprocessing`

The current split between:

- `data_preprocessing`
- `dimensional_analysis`

should be removed in PyDimension 3.0.

They should be merged into one broader `data_preprocessing` module.

### Why

`dimensional_analysis` is really one preprocessing method for constructing physically meaningful coordinates before learning.

In PyDimension 3.0, preprocessing should become the stage that:

- loads data,
- validates metadata,
- normalizes or transforms data,
- parses units and dimensions,
- optionally performs dimensional-analysis-based coordinate construction,
- prepares inputs for intrinsic-coordinate discovery and symmetry discovery.

### Important idea

`dimensional_analysis` should remain as a method inside preprocessing, not as a separate top-level module.

## 3. Rename `constraint_filtering` to `intrinsic_coordinate`

The current module `constraint_filtering` should be renamed to `intrinsic_coordinate`.

### Why

The current name is too narrow and tied to the old workflow.

What the module is really doing is discovering or estimating a small set of latent or intrinsic coordinates that explain the output.

### Existing methods to keep

- PCA
- SIR

### New method to add

Add an autoencoder-decoder style method:

- multiple inputs,
- compress to a small number of hidden variables,
- decode or map to one output.

This should be treated as another intrinsic-coordinate method, not as a separate unrelated package.

## 4. Rewrite `optimization_discovery`

`optimization_discovery` should be comprehensively rewritten.

Its new role is broader than neural-network-based gamma optimization.

In PyDimension 3.0, this module should become the main symmetry-discovery stage after preprocessing and intrinsic-coordinate estimation.

### New role

Before symbolic or compact relation discovery, this module should provide symmetry-specific encoders such as:

- translational encoder,
- rotational encoder,
- scaling encoder.

These encoders should help determine:

- whether a symmetry is hidden in the data,
- which symmetry class is most plausible,
- how that symmetry should be parameterized,
- how to construct symmetry-aware reduced features for downstream discovery.

### Important change in logic

The module should not only fit a relation after a coordinate choice.

It should participate in identifying the symmetry structure itself.

## Proposed Logic Structure

PyDimension 3.0 should follow this high-level logic:

1. Generate or load data.
2. Preprocess data and metadata.
3. Construct candidate physically meaningful coordinates.
4. Estimate intrinsic coordinate dimension.
5. Use symmetry encoders to test and characterize hidden symmetries.
6. Build reduced symmetry-aware representations.
7. Discover compact relations, scaling laws, or governing structures.
8. Save interpretable outputs and benchmark reports.

## Proposed Pipeline

### Stage 1: Data Generation

Purpose:

- create synthetic benchmarks for known symmetry classes.

Outputs:

- dataset,
- metadata,
- ground-truth symmetry information,
- benchmark labels.

### Stage 2: Data Preprocessing

Purpose:

- unify loading, cleaning, normalization, units, dimensional metadata, and optional dimensional-analysis-based coordinate construction.

Possible methods inside this stage:

- raw normalization,
- unit parsing,
- dimension-matrix construction,
- dimensional-analysis basis construction,
- log transform,
- coordinate conversion.

### Stage 3: Intrinsic Coordinate

Purpose:

- estimate how many hidden variables are needed,
- generate candidate low-dimensional coordinates.

Methods:

- PCA,
- SIR,
- autoencoder-decoder,
- future manifold-learning methods.

### Stage 4: Optimization Discovery

Purpose:

- discover whether and how symmetries are hidden in the data.

Methods:

- translational encoder,
- rotational encoder,
- scaling encoder,
- future symmetry encoders.

Outputs:

- symmetry scores,
- discovered symmetry parameters,
- transformed latent representation,
- symmetry-aware candidate formulas or features.

### Stage 5: Interpretation and Export

Purpose:

- convert learned representations into compact, interpretable outputs.

Outputs:

- equations,
- symmetry descriptions,
- latent variables,
- plots,
- benchmark metrics,
- reproducibility artifacts.

## Proposed File Structure

Below is a proposed file structure for PyDimension 3.0.

```text
PyDimension/
├── pydimension/
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── base.py
│   │   ├── generator.py
│   │   ├── translational.py
│   │   ├── rotational.py
│   │   ├── scaling.py
│   │   └── README.md
│   ├── data_preprocessing/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── base.py
│   │   ├── loader.py
│   │   ├── normalizer.py
│   │   ├── unit_parser.py
│   │   ├── dimension_matrix.py
│   │   ├── dimensional_analysis.py
│   │   ├── transforms.py
│   │   └── README.md
│   ├── intrinsic_coordinate/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── base.py
│   │   ├── pca.py
│   │   ├── sir.py
│   │   ├── autoencoder.py
│   │   ├── decoder.py
│   │   └── README.md
│   ├── optimization_discovery/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── base.py
│   │   ├── engine.py
│   │   ├── encoders/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── translational_encoder.py
│   │   │   ├── rotational_encoder.py
│   │   │   └── scaling_encoder.py
│   │   ├── scoring.py
│   │   ├── relation_heads.py
│   │   └── README.md
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── synthetic_translation.py
│   │   ├── synthetic_rotation.py
│   │   ├── synthetic_scaling.py
│   │   └── keyhole.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── paths.py
│   │   ├── plotting.py
│   │   ├── validation.py
│   │   ├── artifacts.py
│   │   └── types.py
│   └── configs/
│       ├── __init__.py
│       ├── config_translation.json
│       ├── config_rotation.json
│       ├── config_scaling.json
│       └── README.md
├── generate_data.py
├── preprocess_data.py
├── intrinsic_coordinate.py
├── discover_symmetry.py
├── run_pipeline.py
├── streamlit_app.py
├── README.md
└── projects/
```

## Proposed Logic Inside Each Package

## `data_generation`

Responsibilities:

- generate synthetic datasets,
- emit ground-truth symmetry metadata,
- standardize benchmark output format.

Design rule:

- each symmetry-specific generator should expose the same interface.

## `data_preprocessing`

Responsibilities:

- load raw data,
- validate schema,
- normalize and transform,
- process units,
- optionally apply dimensional analysis,
- write a standardized preprocessed artifact.

Design rule:

- preprocessing methods should be composable and method-oriented.

## `intrinsic_coordinate`

Responsibilities:

- estimate latent dimension,
- build compact hidden coordinates,
- compare methods consistently.

Design rule:

- all intrinsic-coordinate methods should return a shared result object:
  - latent coordinates,
  - suggested dimension,
  - method metadata,
  - confidence or score fields.

## `optimization_discovery`

Responsibilities:

- symmetry-aware encoding,
- symmetry scoring,
- representation discovery,
- compact relation construction support.

Design rule:

- each symmetry encoder should have the same contract:
  - fit,
  - transform,
  - score symmetry,
  - export interpretable parameters.

## `benchmarks`

Responsibilities:

- define reproducible tasks,
- package synthetic and real examples,
- provide expected outputs and validation targets.

Design rule:

- every benchmark should be runnable through the same pipeline.

## `common`

Responsibilities:

- remove duplicated utility code,
- centralize artifacts, paths, plotting, validation, and shared types.

Design rule:

- do not duplicate helper logic across high-level modules.

## Suggested Config Logic

The config structure should also become simpler and more consistent.

### Current issue

The current config structure mirrors the old pipeline exactly and carries some duplicated ideas.

### Proposed config sections

Use a structure like:

```json
{
  "DATA_GENERATION": {},
  "DATA_PREPROCESSING": {},
  "INTRINSIC_COORDINATE": {},
  "OPTIMIZATION_DISCOVERY": {},
  "OUTPUT": {},
  "BENCHMARK": {}
}
```

### Naming rule

The section names should match package names and script names as closely as possible.

## Suggested Script Layer

For top-level scripts, use concise and consistent names:

- `generate_data.py`
- `preprocess_data.py`
- `intrinsic_coordinate.py`
- `discover_symmetry.py`
- `run_pipeline.py`

This is clearer than keeping names tied only to the old 2.0 pipeline terminology.

## Required README Changes

`README.md` should be updated together with the architecture redesign task.

### It should include

1. The intro figure:
   - `projects/20260307_LEARNING_STAGE/Picture1.png`
2. A concise statement that PyDimension is evolving toward OpenSymmetry.
3. A cleaner and more consistent package structure section.
4. Terminology aligned with the new module names:
   - `data_generation`
   - `data_preprocessing`
   - `intrinsic_coordinate`
   - `optimization_discovery`

### README writing principle

The README should be concise and consistent with the code:

- if a term appears in the README, it should exist in the code,
- if a module name changes, the README should change too,
- avoid long, duplicated explanations.

## Recommended Implementation Order

To reduce risk, implement the restructuring in this order:

1. Update architecture documents and README.
2. Merge `dimensional_analysis` into `data_preprocessing`.
3. Rename `constraint_filtering` to `intrinsic_coordinate`.
4. Create the new `optimization_discovery` skeleton with encoder interfaces.
5. Split current and future synthetic generators under `data_generation`.
6. Add the first non-current symmetry benchmark.
7. Gradually port old scripts to the new structure.

## Deliverable for This Task

The deliverable is a PyDimension 3.0 code-structure plan that:

- adapts to `OpenSymmetry`,
- keeps the design simple,
- keeps naming consistent,
- supports multiple symmetry classes,
- and is suitable for incremental implementation rather than a risky full rewrite at once.

## Final Direction

PyDimension 3.0 should no longer be viewed only as a package for discovering dimensionless relations.

It should become a concise, modular, symmetry-aware framework where:

- data generation supports multiple symmetry types,
- preprocessing unifies dimensional and non-dimensional preparation,
- intrinsic-coordinate discovery becomes a first-class stage,
- optimization discovery becomes symmetry-aware encoding and discovery,
- and the whole system remains readable, minimal, and extensible.
