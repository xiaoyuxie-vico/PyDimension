# Test Commands

**Note**: Optimization Discovery automatically uses the suggested dominant count from Dimensional Filtering. Run Dimensional Filtering first.

## Complete Pipeline

Activate the environment:
`micromamba activate pydimension`

### Automated (Recommended)

```bash
# Run all modules with visualization
python run_pipeline.py --config pydimension/configs/config_synthetic.json --plot

# Without visualization
python run_pipeline.py --config pydimension/configs/config_synthetic.json

# Custom output directory
python run_pipeline.py --config pydimension/configs/config_synthetic.json --output_dir my_output --plot

# Continue on error
python run_pipeline.py --config pydimension/configs/config_synthetic.json --continue-on-error

# Skip specific steps
python run_pipeline.py --config pydimension/configs/config_synthetic.json --skip data_generation --skip data_preprocessing

# Stop after a specific step
python run_pipeline.py --config pydimension/configs/config_synthetic.json --stop-after dimensional_analysis
```

### Manual Step-by-Step

Syntheic data example:
```bash
# Step 1: Generate data
python generate_data.py --config pydimension/configs/config_synthetic.json --plot

# Step 2: Preprocess data
python preprocess_data.py --config pydimension/configs/config_synthetic.json --plot

# Step 3: Dimensional analysis (with normalized lg data)
python analyze_dimensions.py --config pydimension/configs/config_synthetic.json --save-normalized-lg --plot

# Step 4: Dimensional filtering (outputs suggested_dominant_count.json)
python filter_constraints.py --config pydimension/configs/config_synthetic.json --plot

# Step 5: Optimization discovery (automatically uses suggested count from step 4)
python optimize_discovery.py --config pydimension/configs/config_synthetic.json --plot
```

Keyhole problem:
```bash
# Step 1: Generate data
python generate_data.py --config pydimension/configs/config_keyhole.json --plot

# Step 2: Preprocess data
python preprocess_data.py --config pydimension/configs/config_keyhole.json --plot

# Step 3: Dimensional analysis (with normalized lg data)
python analyze_dimensions.py --config pydimension/configs/config_keyhole.json --save-normalized-lg --plot

# Step 4: Dimensional filtering (outputs suggested_dominant_count.json)
python filter_constraints.py --config pydimension/configs/config_keyhole.json --plot

# Step 5: Optimization discovery (automatically uses suggested count from step 4)
python optimize_discovery.py --config pydimension/configs/config_keyhole.json --plot
```
