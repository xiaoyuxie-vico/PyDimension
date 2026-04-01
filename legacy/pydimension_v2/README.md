# PyDimension 2.0 Benchmark Package

This package preserves the runnable PyDimension 2.0 benchmark surface while the main `pydimension/` package is migrated toward PyDimension 3.0 and OpenSymmetry.

## Purpose

- provide a stable benchmark import path: `legacy.pydimension_v2`
- keep the original translational dimensionless-learning workflow runnable
- support parity checks against the new 3.0 translational pipeline

## Entry Point

```bash
python legacy/run_pipeline_v2.py --config legacy/pydimension_v2/configs/config_synthetic_v2.json
```
