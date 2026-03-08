# Configuration Files

**All configuration files for PyDimension should be maintained in this directory** (`pydimension/configs/`).

## Config Families

PyDimension carries three config families that correspond to different symmetry types and pipeline versions:

| Config file | Pipeline | Symmetry type | Status |
|---|---|---|---|
| `config_synthetic.json` | Legacy 2.0 | translational | production |
| `config_synthetic_with_noise.json` | Legacy 2.0 | translational (5% noise) | production |
| `config_keyhole.json` | Legacy 2.0 | translational (real data) | production |
| `config_translation.json` | 3.0 | translational | active |
| `config_translation_v3.json` | 3.0 | translational (alias) | kept for backward compat |
| `config_rotation.json` | 3.0 | rotational | scaffold |
| `config_scaling.json` | 3.0 | scaling | scaffold |

The 2.0 benchmark also has a frozen copy at `legacy/pydimension_v2/configs/config_synthetic_v2.json`.

## Config Section Naming (3.0)

The 3.0 config uses section names that match the new module names:

```json
{
  "DATA_GENERATION": { },
  "DATA_PREPROCESSING": { "preprocessing_method": "dimensional_analysis" },
  "INTRINSIC_COORDINATE": { "method": "pca_sir" },
  "SYMMETRY_DISCOVERY": { "symmetry_type": "translational", "encoder_name": "translational" },
  "OUTPUT": { "output_dir": "output_v3_translation" },
  "DATA_GENERATION_OUTPUT": { },
  "DATA_PREPROCESSING_OUTPUT": { },
  "DIMENSIONAL_ANALYSIS_OUTPUT": { },
  "INTRINSIC_COORDINATE_OUTPUT": { },
  "SYMMETRY_DISCOVERY_OUTPUT": { }
}
```

Legacy 2.0 sections (`CONSTRAINT_FILTERING`, `OPTIMIZATION_DISCOVERY`) are still accepted as fallbacks by the config loaders, so existing user configs continue to work.

## Usage

```bash
# 3.0 translational pipeline
python run_pipeline.py --pipeline-version v3 --config pydimension/configs/config_translation.json

# Legacy 2.0 benchmark
python run_pipeline.py --pipeline-version v2 --config legacy/pydimension_v2/configs/config_synthetic_v2.json

# Individual modules
python generate_data.py --config pydimension/configs/config_synthetic.json --plot
python preprocess_data.py --config pydimension/configs/config_synthetic.json --plot
python discover_symmetry.py --config pydimension/configs/config_translation.json
```

## Extending the Config

When adding a new symmetry type or module:

1. Copy an existing config file (e.g., `config_translation.json`).
2. Update `SYMMETRY_DISCOVERY.symmetry_type` and `encoder_name`.
3. Update `OUTPUT.output_dir` to a new directory name.
4. Add any module-specific parameters.
5. Register the corresponding benchmark in `pydimension/benchmarks/`.

## Config File Priority

When loading configs, the system checks in this order:
1. Unified `OUTPUT` section for shared output paths.
2. Module-specific output sections (e.g., `SYMMETRY_DISCOVERY_OUTPUT`) for filenames.
3. Legacy fallback sections (e.g., `OPTIMIZATION_DISCOVERY`) for backward compatibility.
