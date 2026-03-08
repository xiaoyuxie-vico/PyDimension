# Learning Stage Notes

## Date

2026-03-07

## Purpose

This note summarizes what I learned from:

1. The current `PyDimension` codebase.
2. The two core papers:
   - `A Tutorial on Dimensionless Learning: Geometric Interpretation and the Effect of Noise`
   - `Data-driven discovery of dimensionless numbers and governing laws from scarce measurements`
3. The project vision materials in `projects/20260307_LEARNING_STAGE`, including:
   - `Project Description.pdf`
   - `References Cited.pdf`
   - `Notes for symmetry.pptx`
   - `figures.pptx`

The goal of this note is to provide an AI-friendly, implementation-oriented understanding of the current system and to propose concrete directions for extending `PyDimension` into a symmetry-based framework, tentatively called `OpenSymmetry`.

## Executive Summary

`PyDimension` already implements a strong and fairly clean workflow for one important symmetry class: scaling symmetry. In practice, the code discovers dimensionless groups by solving a null-space problem from a dimension matrix, reduces the basis to a smaller number of dominant directions with PCA/SIR, and then learns the best linear combinations of basis vectors through a neural network with optional quantization regularization.

The deeper conceptual shift in the project materials is this:

- Dimensionless learning should no longer be treated as an isolated method.
- It should be treated as one instance of symmetry discovery.
- The next step is not only "better scaling-law discovery," but a general ecosystem for symmetry-aware scientific discovery.

In other words:

- Current `PyDimension` = production implementation of dimensionless learning.
- Proposed `OpenSymmetry` = a larger framework where scaling symmetry is one module among several symmetry classes, alongside translational, rotational, boost, and potentially other invariances.

The most important practical conclusion is that the next branch should likely focus first on architecture, interfaces, benchmarks, and canonical workflow design, not on jumping directly into a large number of new algorithms.

## What the Current Codebase Does

## High-Level Pipeline

The current package is organized as a 5-stage pipeline:

1. `data_generation`
2. `data_preprocessing`
3. `dimensional_analysis`
4. `constraint_filtering`
5. `optimization_discovery`

The top-level script `run_pipeline.py` orchestrates these stages in sequence and passes artifacts between them through files in `output/data`, `output/results`, and `output/figures`.

## Main Architectural Pattern

Each stage has:

- a config class,
- a core implementation class,
- a CLI entrypoint,
- documentation,
- saved artifacts for downstream stages.

This is already close to a reusable workflow architecture and is a good base for future symmetry modules.

## What Each Module Does

### 1. Data Generation

Implemented mainly in `pydimension/data_generation/generator.py`.

This module creates synthetic benchmark datasets with known hidden dimensionless structure.

Key behavior:

- Generates `N` input variables with `M` samples.
- Builds a random dimension matrix with rank 4.
- Computes the null space of that matrix.
- Simplifies null-space basis vectors using SymPy rational arithmetic.
- Optionally normalizes basis vectors to unit vectors.
- Combines basis vectors with user-defined or auto-generated `gamma` vectors.
- Constructs one or more true dimensionless groups `pi`.
- Defines the output as:
  - a polynomial of one `pi`, or
  - a nonlinear function of multiple `pi` groups.
- Can inject noise and discrete sampling effects.

This module is important because it functions as an internal benchmark generator and a controlled testbed for theory, robustness, and future symmetry extensions.

### 2. Data Preprocessing

Implemented mainly in `pydimension/data_preprocessing/preprocessor.py`.

This stage:

- loads CSV data,
- detects inputs and outputs,
- loads or infers a dimension matrix,
- normalizes each selected variable by its maximum,
- writes normalized data and dimension metadata.

Important observation:

This module is currently tailored to dimensional analysis workflows, not general symmetry workflows. It assumes a "variables + dimension matrix + output" model. For `OpenSymmetry`, preprocessing will need to become more abstract so it can also support:

- coordinate transforms,
- group actions,
- symmetry metadata,
- paired transformed/original samples,
- benchmark metadata and provenance.

### 3. Dimensional Analysis

Implemented mainly in `pydimension/dimensional_analysis/analyzer.py`.

This is the mathematical core of dimensionless learning.

It:

- loads normalized data and the dimension matrix,
- computes the null space `Dw = 0`,
- simplifies basis vectors to primitive rational/integer-like structure,
- optionally normalizes them,
- constructs basis dimensionless variables `pi`,
- saves transformed data,
- additionally saves `normalized_lg_afterDA_data.csv`.

The log transform is especially important. Products of powers become linear combinations in log space:

- original space: `pi = product(p_i ^ w_i)`
- log space: `log(pi) = w^T log(p)`

This is the bridge between dimensional analysis and the learning model.

### 4. Constraint Filtering

Implemented mainly in `pydimension/constraint_filtering/filterer.py`.

This stage estimates how many dimensionless groups are likely needed.

Methods:

- PCA on standardized data including inputs and output.
- SIR on input basis-groups conditioned on output.

Output:

- explained variance information,
- suggested dominant count,
- a JSON artifact consumed by optimization discovery.

This module matters because it avoids blindly optimizing in the full null-space basis dimension. It acts as an intrinsic-dimension estimator for the physically relevant subspace.

### 5. Optimization Discovery

Implemented mainly in `pydimension/optimization_discovery/optimizer.py`.

This stage trains a neural network with a special first layer:

- Inputs: `lg(pi_basis)` values.
- First linear layer: learns `gamma`.
- Hidden layers: learn the nonlinear relation from discovered groups to output.

This is essentially the implementation of the paper's second optimization level, though in the current package it is packaged as a single neural architecture rather than the original grid-search / pattern-search emphasis of the Nature paper.

Important details:

- multiple ensemble models are trained,
- best model is selected by `R^2`,
- learned `gamma` is mapped back to original parameter exponents using saved basis vectors,
- optional gamma regularization encourages integers / half-integers / quarter-integers.

This regularization is a major interpretability mechanism and one of the most important features of the modern codebase.

## What Dimensionless Learning Is, Conceptually

## Core Problem

Given:

- dimensional input variables,
- an output quantity,
- scarce and possibly noisy data,

we want to discover:

1. which combinations of inputs define the important dimensionless groups,
2. how many such groups matter,
3. the scaling law or governing relation between them and the output.

## Mathematical Core

### Step 1: Build the Dimension Matrix

Each input variable is represented by exponents over fundamental dimensions such as:

- mass,
- length,
- time,
- temperature.

Stacking these vectors gives the dimension matrix `D`.

### Step 2: Null Space = All Valid Dimensionless Combinations

Dimensionless groups correspond to exponent vectors `w` satisfying:

`D w = 0`

So the null space of `D` is the space of all dimensionally valid combinations.

This is the key geometric idea:

- Buckingham Pi theorem gives the existence and count of groups.
- Linear algebra gives the actual structure.
- The null space is not a single answer but a whole subspace.

### Step 3: Basis Vectors Are Not the Final Physics

Any basis of the null space is mathematically valid, but the basis itself is not usually the physically dominant set of groups.

Instead, the real task is to discover a smaller set of useful combinations inside that null-space basis.

That is why the code and papers introduce `gamma`:

- basis vectors = spanning directions,
- gamma = coordinates that choose meaningful combinations.

### Step 4: Use Log Space

Power-law monomials become linear combinations in log space.

This is why the pipeline naturally moves from:

- physical variables
- to dimensionless basis groups
- to log-transformed basis groups
- to learned linear combinations via a neural first layer.

### Step 5: Learn the Scaling Law

Once the correct dimensionless groups are formed, the output should lie on a low-dimensional manifold:

- one curve if there is one dominant group,
- one surface if there are two dominant groups,
- higher-dimensional manifolds otherwise.

Dimensionless learning therefore acts as a physics-based dimension reduction method.

## Main Insights from the Nature Communications Paper

The 2022 paper establishes the original breakthrough and application value.

## Main Contributions

1. It embeds dimensional invariance into a two-level learning framework.
2. It shows discovery of dominant dimensionless numbers from scarce experimental data.
3. It discovers scaling laws, not only candidate groups.
4. It extends to governing-equation discovery by combining dimensionless learning with SINDy.
5. It already introduces a symmetry-aware idea beyond dimensional analysis through symmetric invariant SINDy.

## Important Engineering Examples

The paper demonstrates discovery in:

- Rayleigh-Benard convection,
- keyhole dynamics in laser-metal interaction,
- porosity formation in additive manufacturing,
- parameterized governing equations like the vorticity form of Navier-Stokes.

This matters because the project is not only theoretical. It is already validated on difficult, noisy, real engineering datasets.

## Important Lessons from the Paper

### 1. Dimensional Analysis Alone Is Not Enough

Buckingham Pi gives a valid subspace, but not the dominant coordinates in that subspace.

### 2. The Search Space Must Be Physically Constrained

The paper emphasizes small rational coefficients and physically interpretable powers. This is exactly why `gamma` regularization is valuable.

### 3. Symmetry Is Already Present in the 2022 Work

The paper is officially about dimensionless learning, but the deeper theme is invariance:

- dimensional invariance,
- scale invariance,
- and, in the governing-equation workflow, symmetric invariance.

This is one of the strongest arguments for evolving toward `OpenSymmetry`.

### 4. Generalization Comes from Physical Structure

The method generalizes better than generic black-box ML because it embeds invariance and reduces effective dimensionality.

## Main Insights from the Tutorial Paper

The tutorial paper is especially useful because it explains the geometric and practical meaning of the method.

## Key Conceptual Gains

### 1. A Geometric Interpretation

The tutorial makes it explicit that dimensionless learning should be viewed as navigation inside the null-space manifold.

This is a crucial conceptual upgrade. It suggests that future symmetry-based methods should also be framed geometrically:

- identify the admissible invariant subspace,
- estimate intrinsic dimensionality,
- search for simple, interpretable coordinates inside that subspace.

### 2. Filtering Before Optimization Matters

PCA and SIR are used to estimate the number of dominant groups before final discovery.

SIR is especially important for nonlinear relationships and is more robust than PCA in several noisy cases.

### 3. Quantization Regularization Is Not Cosmetic

The quantization regularizer:

- improves interpretability,
- promotes sparsity,
- and often improves robustness to noise and discrete sampling.

This is a major design principle that should likely be reused in future symmetry modules.

### 4. Multiple Dominant Groups Create a Subspace, Not a Unique Point

When there are multiple relevant dimensionless groups, the learned solutions often form a subspace of equivalent representations rather than one isolated coefficient vector.

This is a major idea for `OpenSymmetry`:

- future symmetry discovery may frequently recover equivalence classes or invariant subspaces,
- not unique symbolic forms.

So the framework should be designed around:

- subspace discovery,
- canonicalization,
- sparsification,
- and representative selection.

### 5. The Real Challenges Are Now Clear

The tutorial highlights several unsolved problems:

- more than two or three dominant groups is hard,
- uncertainty effects are not fully understood,
- input selection is unresolved,
- user accessibility is still limited.

These are good targets for future work and ecosystem design.

## What the Project Description Adds

The project description changes the framing from "a package for dimensionless learning" to "an ecosystem for symmetry discovery."

## Core Shift

The project explicitly argues that:

- dimensionless learning is one symmetry-discovery workflow,
- scaling symmetry can be interpreted as translational symmetry in log-parameter space,
- PyDimension should evolve into `OpenSymmetry`,
- the immediate challenge is organizational and infrastructural, not only algorithmic.

## Important Strategic Conclusions

### 1. OpenSymmetry Is About Generalization of Symmetry Classes

The target classes mentioned include:

- scaling symmetry,
- translational symmetry,
- rotational symmetry,
- boost symmetry,
- other symmetries.

### 2. The Project Is Not Only About New Algorithms

The proposal strongly emphasizes:

- canonical workflow,
- shared interfaces,
- benchmarks,
- reproducibility,
- governance,
- contribution pathways,
- containerized execution environments.

This is extremely important. The next branch should likely prioritize framework design over research-only code.

### 3. There Is Already a Reusable Foundation

The current `PyDimension` package already has:

- modular stages,
- configuration-driven execution,
- CLI and API access,
- documentation,
- benchmark-like synthetic generation,
- a web interface.

So `OpenSymmetry` should build on that structure rather than replace it.

## What the PPTX Materials Add

The slides add both scientific motivation and conceptual vocabulary.

## Symmetry Concepts Highlighted

- invariance,
- covariance,
- group theory,
- Lie groups,
- Lie algebras,
- generators,
- Noether's theorem,
- representation theory,
- ridge-function viewpoints,
- symmetry in ML as enforce / discover / promote.

These materials suggest that `OpenSymmetry` should not only be a software package for a few heuristics. It should become a framework that can express:

- a transformation group,
- its action on data or coordinates,
- invariants,
- equivariants,
- candidate generators,
- discovered reduced coordinates,
- and validation tests for those structures.

## Practical Interpretation for the Branch

The slides imply that future work should connect:

- scientific ML,
- physically meaningful transformation groups,
- interpretable reduced variables,
- equation discovery,
- and symmetry-aware AI.

This means the branch should probably create abstractions that are broad enough to support:

- data-level symmetry discovery,
- model-level symmetry enforcement,
- equation-level invariant discovery.

## Most Important Synthesis

Here is the single most important takeaway from all materials:

Dimensionless learning is best viewed as a special case of symmetry-aware scientific discovery where:

- the group action is scaling in physical parameter space,
- or translation in log-parameter space,
- the invariant coordinates are dimensionless combinations,
- and the final goal is a low-dimensional, interpretable, reusable law.

If that is true, then the natural extension is:

- define a common symmetry-discovery workflow,
- treat scaling as one plug-in module,
- add other symmetry modules that follow the same workflow contract.

## Current Gaps Between PyDimension and OpenSymmetry

## Gap 1: The Architecture Is Pipeline-Oriented, Not Symmetry-Oriented

Current modules are named by processing stage, not by symmetry abstraction.

That is good for the current method, but limiting for future extension.

Needed future concept:

- `SymmetrySpec`
- `TransformationGroup`
- `InvariantBuilder`
- `DominantSubspaceEstimator`
- `SymmetryDiscoveryModel`
- `Canonicalizer`

## Gap 2: Scaling Symmetry Is Implicit, Not Explicit

The code uses dimension matrices, basis vectors, and `gamma`, but it does not expose "scaling symmetry" as an explicit first-class object.

Needed future concept:

- make the transformation and invariance assumptions explicit in code and config.

## Gap 3: Inputs and Metadata Are Too Narrow

Current preprocessing assumes mostly tabular variables and dimension matrices.

Future symmetry workflows may also need:

- coordinate systems,
- transformation parameter ranges,
- transformed sample pairs,
- benchmark tags,
- provenance metadata,
- assumptions about invariance or equivariance.

## Gap 4: Benchmarking Is Informal

The code supports synthetic data generation and example configs, but there is not yet a formal benchmark suite for:

- reproducibility,
- noise stress tests,
- sparsity tests,
- missing-variable tests,
- symmetry recovery accuracy,
- cross-environment parity.

## Gap 5: Equation Discovery Is Not Yet a First-Class Integrated Module

The papers point toward SINDy integration and symmetry-aware equation discovery, but the current packaged workflow is centered on scaling-law discovery.

## Gap 6: Canonicalization of Equivalent Solutions Is Not Yet Systematic

The tutorial makes clear that many equivalent representations exist.

Future framework work should explicitly support:

- equivalence classes,
- canonical representatives,
- sparse representatives,
- confidence intervals over discovered subspaces.

## Proposed Design Direction

## Principle

Do not start by hard-coding many new symmetry algorithms.

Start by creating a general symmetry workflow interface, then port dimensionless learning into that interface as the first reference implementation.

## Proposed Canonical Workflow

### Stage A: Data and Metadata Ingestion

Inputs:

- dataset,
- physical variables,
- units / dimensions,
- coordinate or transformation metadata,
- output definition,
- benchmark metadata.

### Stage B: Symmetry Specification

Inputs:

- symmetry class,
- group action,
- invariance / equivariance assumptions,
- optional priors or constraints.

Examples:

- scaling,
- translation,
- rotation,
- boost.

### Stage C: Basis / Generator Construction

For scaling:

- null-space basis vectors.

For other symmetry classes:

- generators,
- invariant features,
- transformed basis coordinates,
- or learned latent candidates.

### Stage D: Dominant Subspace Estimation

Generalized version of current PCA/SIR stage.

Goal:

- estimate how many invariant coordinates matter.

### Stage E: Symmetry-Constrained Discovery

Generalized version of optimization discovery.

Goal:

- discover invariant coordinates and the low-dimensional relation to the output.

### Stage F: Canonicalization and Interpretation

Goal:

- map discovered structures back to original variables,
- simplify,
- quantify equivalence,
- choose interpretable representatives.

### Stage G: Validation

Goal:

- test invariance,
- test benchmark recovery,
- test stability under noise and subsampling,
- test reproducibility across environments.

## Concrete Code Architecture Proposal

## 1. Keep Existing Modules Working

Do not break the current pipeline immediately.

Instead:

- preserve the current dimensionless learning workflow,
- add a new internal abstraction layer,
- gradually refactor existing modules to implement it.

## 2. Introduce New Core Abstractions

Suggested new package area:

- `pydimension/symmetry/`

Suggested initial interfaces:

- `base.py`
  - `SymmetryProblem`
  - `SymmetrySpec`
  - `SymmetryResult`
- `groups.py`
  - `ScalingSymmetry`
  - `TranslationSymmetry`
  - placeholder classes for `RotationSymmetry`, `BoostSymmetry`
- `workflow.py`
  - canonical orchestration API
- `canonicalization.py`
  - sparse/simple representative selection
- `validation.py`
  - invariance and reproducibility checks

## 3. Port Current Dimensionless Learning into a Symmetry Module

Suggested wrapper:

- `pydimension/symmetry/scaling/`

It would internally reuse:

- preprocessing,
- dimensional analysis,
- filtering,
- optimization discovery.

But the external interface would describe the workflow in symmetry terms rather than only DA terms.

## 4. Add Translation Symmetry as a Closely Related Second Module

Because the materials explicitly connect scaling symmetry to translational symmetry in log space, this is likely the safest next symmetry class.

This provides:

- conceptual continuity,
- mathematical closeness,
- a good test of whether the new architecture is genuinely general.

## Proposed Actions and Projects

## Short-Term Actions

### Action 1: Write a Canonical Workflow Spec

Create a design doc that defines:

- stage names,
- stage inputs and outputs,
- common artifact schema,
- naming conventions,
- validation contracts.

This should happen before large refactoring.

### Action 2: Make Scaling Symmetry Explicit

Refactor the current pipeline so that dimensionless learning is labeled internally as a `ScalingSymmetry` workflow.

This is the bridge between present code and future `OpenSymmetry`.

### Action 3: Add Formal Benchmark Definitions

Convert current examples into benchmark objects:

- synthetic 1D scaling benchmark,
- synthetic 2D scaling benchmark,
- noisy scaling benchmark,
- keyhole benchmark,
- Rayleigh benchmark if data is available.

Each benchmark should define:

- inputs,
- expected invariants,
- expected dominant count,
- expected quality thresholds.

### Action 4: Add Validation Reports

Automate checks for:

- dimensional consistency,
- invariance preservation,
- reproduction of expected benchmark outputs,
- robustness under noise and subsampling.

### Action 5: Separate "Discovery" from "Canonicalization"

Currently, learning and simplification are mixed conceptually.

Future framework should explicitly separate:

- finding a valid invariant subspace,
- selecting a human-interpretable representative.

## Medium-Term Projects

### Project 1: `ScalingSymmetry` Module Refactor

Goal:

- wrap current dimensionless learning in a symmetry-centered API.

Deliverables:

- symmetry config schema,
- reusable result schema,
- backward-compatible adapter to the current pipeline.

### Project 2: `TranslationSymmetry` Reference Module

Goal:

- show that the architecture supports another symmetry class.

Minimum version:

- implement translational symmetry detection on transformed/log coordinates,
- reuse filtering and discovery ideas where possible.

### Project 3: Subspace Canonicalization Toolkit

Goal:

- handle non-uniqueness explicitly.

Methods may include:

- sparsification,
- integer/half-integer projection,
- basis alignment,
- representative selection by complexity score.

### Project 4: Symmetry-Aware Equation Discovery Bridge

Goal:

- create a clean integration path from invariant-coordinate discovery to equation discovery tools such as SINDy.

This is already motivated by the Nature paper and the project description.

### Project 5: Benchmark and Reproducibility Hub

Goal:

- package benchmarks, containerized environments, reports, and CI validation together.

This is essential if the project is truly moving toward an ecosystem model.

## Longer-Term Projects

### Project 6: Rotation and Boost Symmetry Interfaces

Goal:

- define common APIs and toy benchmarks first,
- postpone full large-scale implementations until the architecture is stable.

### Project 7: AI-Assisted Symmetry Discovery Layer

Goal:

- use LLM/agent workflows to help with:
  - variable selection,
  - unit validation,
  - symmetry hypothesis generation,
  - benchmark authoring,
  - documentation and interpretation.

This aligns strongly with the tutorial's open challenge on input selection and user accessibility.

### Project 8: Symmetry-Aware Generative and Scientific ML Interfaces

Goal:

- expose discovered symmetries to downstream ML systems,
- connect with the "enforce, discover, promote" framework from the slides.

## Recommended Branch Priorities for `opensymmetry`

If I were planning the branch in a practical order, I would do this:

1. Preserve and stabilize the current PyDimension workflow.
2. Define a symmetry-centered architecture and artifact schema.
3. Recast dimensionless learning as the first `ScalingSymmetry` module.
4. Build a benchmark suite and validation layer.
5. Add one closely related second symmetry class, probably translation in log space.
6. Only after that, explore more ambitious symmetry modules such as rotation or boost.

This order reduces risk and produces useful infrastructure early.

## Practical Development Recommendations

## Recommendation 1

Treat `OpenSymmetry` first as a software architecture and benchmarking project, not only as a research prototype.

## Recommendation 2

Use current `PyDimension` as the reference implementation and baseline benchmark suite.

## Recommendation 3

Keep mathematical language explicit in the code:

- group action,
- invariance,
- equivariance,
- generator,
- subspace,
- canonical representative.

This will make future extension much easier.

## Recommendation 4

Make non-uniqueness a first-class concept in data structures and outputs.

Do not assume one "correct" expression always exists.

## Recommendation 5

Invest early in AI-friendly documentation and machine-readable metadata.

This will help both human contributors and coding agents work on the framework safely.

## AI-Friendly Summary

### What PyDimension currently is

`PyDimension` is a modular implementation of dimensionless learning. It discovers scale-invariant low-dimensional structure from dimensional tabular data using:

- null-space basis construction,
- PCA/SIR dimensional filtering,
- neural learning of basis combinations,
- optional quantized gamma regularization.

### What dimensionless learning really is

Dimensionless learning is a symmetry-aware discovery method for scaling invariance. It searches inside the null space of the dimension matrix for a small set of dominant invariant coordinates and a compact law relating them to the output.

### What OpenSymmetry should become

`OpenSymmetry` should be a general framework for data-driven symmetry discovery where:

- dimensionless learning is one symmetry module,
- symmetry assumptions are explicit,
- benchmarked workflows are canonical,
- reproducibility and contribution pathways are built in,
- and additional symmetry classes can be added without rewriting core infrastructure.

### Best immediate next step

The best immediate next step is to create a canonical symmetry-workflow abstraction and port the existing dimensionless learning pipeline into it as a `ScalingSymmetry` reference module.

## Suggested Next Documents to Write

1. `opensymmetry_architecture.md`
2. `canonical_workflow_spec.md`
3. `benchmark_suite_plan.md`
4. `scaling_symmetry_module_design.md`
5. `translation_symmetry_module_design.md`

## Final Takeaway

The project is not starting from zero. The current codebase already contains a strong, validated implementation of one symmetry-discovery workflow. The opportunity is to elevate that workflow into a general, reproducible, extensible symmetry ecosystem.

That means the next major success is not merely discovering one more dimensionless number. It is designing the software and benchmark structure that will let many symmetry-discovery methods live in one coherent framework.
