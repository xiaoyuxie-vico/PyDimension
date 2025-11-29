# Dataset Directory

This directory contains datasets and dimension matrices for different problems. Each problem should have its own subfolder.

## Structure

```
dataset/
├── keyhole/                    # Keyhole welding problem
│   ├── dataset_keyhole.csv     # Keyhole dataset
│   └── dimension_matrix.csv     # Dimension matrix for keyhole problem
├── problem2/                   # Another problem (example)
│   ├── dataset_problem2.csv
│   └── dimension_matrix.csv
└── README.md                   # This file
```

## Adding a New Problem

1. Create a new subfolder: `dataset/your_problem_name/`
2. Place your dataset CSV file in that folder
3. Place your dimension matrix CSV file in that folder
4. Update your config file to point to the new paths:
   ```json
   {
     "DATA_PREPROCESSING": {
       "input_file": "dataset/your_problem_name/your_dataset.csv",
       "dimension_matrix_file": "dataset/your_problem_name/dimension_matrix.csv"
     }
   }
   ```

## Current Problems

### keyhole
- **Dataset**: `dataset/keyhole/dataset_keyhole.csv`
- **Dimension Matrix**: `dataset/keyhole/dimension_matrix.csv`
- **Config**: `pydimension/configs/config_keyhole.json`
- **Description**: Keyhole welding problem with 90 samples, 12 input variables, and 1 output variable (e*)

