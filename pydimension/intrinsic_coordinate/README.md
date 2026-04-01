# Intrinsic Coordinate Module

This module is the PyDimension 3.0 replacement for `constraint_filtering`.

## Current Scope

- `PCAAndSIRIntrinsicCoordinate` reuses the validated 2.0 PCA/SIR logic
- `AutoencoderIntrinsicCoordinate` is an initial scaffold for future nonlinear latent-coordinate discovery
- `IntrinsicCoordinateFinder` dispatches between the available methods

## Quick Start

```bash
python intrinsic_coordinate.py --config pydimension/configs/config_translation_v3.json
```

```python
from pydimension.intrinsic_coordinate import IntrinsicCoordinateConfig, IntrinsicCoordinateFinder

config = IntrinsicCoordinateConfig.from_json("pydimension/configs/config_translation_v3.json")
artifacts = IntrinsicCoordinateFinder(config).process(verbose=True)
print(artifacts.suggested_count)
```

## Config

The v3 config uses the `INTRINSIC_COORDINATE` section:

```json
{
  "INTRINSIC_COORDINATE": {
    "enabled": true,
    "method": "pca_sir",
    "input_file": null,
    "run_pca": true,
    "run_sir": true,
    "latent_dim": 1
  }
}
```
