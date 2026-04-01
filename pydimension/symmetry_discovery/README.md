# Symmetry Discovery Module

This module is the PyDimension 3.0 replacement for the old 2.0 `optimization_discovery` stage.

## Current Scope

- `SymmetryDiscoveryEngine` is the 3.0 dispatcher
- `TranslationalSymmetryEncoder` is implemented and reuses the validated 2.0 optimizer backend
- `RotationalSymmetryEncoder` and `ScalingSymmetryEncoder` are scaffolded for later phases

## Quick Start

```bash
python discover_symmetry.py --config pydimension/configs/config_translation_v3.json
```

```python
from pydimension.symmetry_discovery import SymmetryDiscoveryConfig, SymmetryDiscoveryEngine

config = SymmetryDiscoveryConfig.from_json("pydimension/configs/config_translation_v3.json")
artifacts = SymmetryDiscoveryEngine(config).process(verbose=True)
print(artifacts.results_file)
```
