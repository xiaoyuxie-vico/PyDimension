"""
Core optimization and discovery functionality using neural networks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import os
import json

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch not available. Neural network training will not work.")

# Try to import sklearn for scaling
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ scikit-learn not available. Some features may not work.")

from .config import OptimizationDiscoveryConfig


class OptimizationDiscoverer:
    """Optimization and discovery class for neural network training."""
    
    def __init__(self, config: OptimizationDiscoveryConfig):
        """Initialize optimizer with configuration."""
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.input_columns: List[str] = []
        self.output_column: Optional[str] = None
        self.basis_vectors: Optional[np.ndarray] = None  # Basis vectors matrix (num_original_params, num_pi)
        self.original_parameter_names: List[str] = []  # Original parameter names from basis vectors
        
        # Data splits
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        
        # Scalers
        self.X_scaler: Optional[Any] = None
        self.y_scaler: Optional[Any] = None
        
        # Models and results
        self.ensemble_models: List[Any] = []
        self.ensemble_histories: List[Dict[str, List[float]]] = []
        self.learned_coeffs: Optional[np.ndarray] = None
        self.model_r2_scores: Optional[np.ndarray] = None
        self.all_gamma_weights_raw: Optional[np.ndarray] = None
        
        # Predictions
        self.predictions: Optional[np.ndarray] = None
        self.prediction_std: Optional[np.ndarray] = None
        
    def load_data(self) -> pd.DataFrame:
        """Load normalized lg afterDA data from CSV file."""
        if self.config.input_file is None:
            raise ValueError("input_file must be specified in config")
        
        input_path = Path(self.config.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_file}")
        
        df = pd.read_csv(input_path)
        print(f"✅ Loaded data from: {input_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Force all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN
        if df.isnull().any().any():
            rows_before = len(df)
            df = df.dropna()
            rows_after = len(df)
            if rows_before != rows_after:
                print(f"   Dropped {rows_before - rows_after} rows with NaN values")
        
        # Assume last column is output
        cols = list(df.columns)
        if len(cols) < 2:
            raise ValueError("Not enough columns in input file (need at least 2)")
        
        self.output_column = cols[-1]
        self.input_columns = cols[:-1]
        self.data = df
        
        print(f"   Input columns: {self.input_columns}")
        print(f"   Output column: {self.output_column}")
        
        # Try to load basis vectors file to map back to original parameters
        self._load_basis_vectors()
        
        return self.data
    
    def _load_basis_vectors(self):
        """Load basis vectors file to map discovered pi back to original parameters."""
        basis_vectors_file = self.config.basis_vectors_file
        if basis_vectors_file is None:
            # Try to find default basis_vectors.csv
            base_path = Path(self.config.output_dir).resolve()
            default_basis_paths = [
                base_path / self.config.data_dir / 'basis_vectors.csv',
                Path('output') / self.config.data_dir / 'basis_vectors.csv',
                Path('basis_vectors.csv')
            ]
            cwd = Path.cwd()
            default_basis_paths.extend([
                cwd / self.config.output_dir / self.config.data_dir / 'basis_vectors.csv',
                cwd / 'output' / self.config.data_dir / 'basis_vectors.csv',
                cwd / 'basis_vectors.csv'
            ])
            for path in default_basis_paths:
                if path.exists():
                    basis_vectors_file = str(path)
                    break
        
        if basis_vectors_file and Path(basis_vectors_file).exists():
            try:
                bv_df = pd.read_csv(basis_vectors_file)
                # Find columns that start with 'w' (w1, w2, w3, ...)
                w_cols = [c for c in bv_df.columns if c.startswith('w')]
                if len(w_cols) > 0:
                    # Get variable names (first column or index)
                    if 'Variable' in bv_df.columns:
                        self.original_parameter_names = bv_df['Variable'].tolist()
                    else:
                        self.original_parameter_names = bv_df.index.tolist()
                    
                    # Extract basis vectors matrix (num_original_params, num_pi)
                    self.basis_vectors = bv_df[w_cols].values
                    print(f"✅ Loaded basis vectors from: {basis_vectors_file}")
                    print(f"   Original parameters: {self.original_parameter_names}")
                    print(f"   Basis vectors shape: {self.basis_vectors.shape}")
            except Exception as e:
                print(f"⚠️  Could not load basis vectors: {e}")
                self.basis_vectors = None
                self.original_parameter_names = []
        else:
            self.basis_vectors = None
            self.original_parameter_names = []
    
    def _prepare_data_splits(self):
        """Prepare train/test splits."""
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        X = self.data[self.input_columns].values.astype(np.float32)
        y = self.data[self.output_column].values.astype(np.float32).reshape(-1, 1)
        
        # Train/test split
        n = len(X)
        np.random.seed(self.config.random_seed)
        idx = np.arange(n)
        np.random.shuffle(idx)
        cut = int(self.config.train_percent * n)
        tr_idx, te_idx = idx[:cut], idx[cut:]
        
        self.X_train = X[tr_idx]
        self.y_train = y[tr_idx]
        self.X_test = X[te_idx]
        self.y_test = y[te_idx]
        
        print(f"\n=== Data Splits ===")
        print(f"   Training: {self.X_train.shape[0]} samples")
        print(f"   Testing: {self.X_test.shape[0]} samples")
        
        # Apply standardization if sklearn is available
        if HAS_SKLEARN:
            self.X_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            
            self.X_train = self.X_scaler.fit_transform(self.X_train)
            self.y_train = self.y_scaler.fit_transform(self.y_train)
            self.X_test = self.X_scaler.transform(self.X_test)
            self.y_test = self.y_scaler.transform(self.y_test)
            
            print(f"   Applied StandardScaler to inputs and outputs")
    
    def _create_model(self, input_dim: int, fixed_gamma: Optional[np.ndarray] = None) -> Any:
        """Create a neural network model."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for neural network training")
        
        class EnsembleNeuralNetwork(nn.Module):
            def __init__(self, input_dim, linear_nodes, num_hidden_layers, hidden_nodes, fixed_gamma=None):
                super(EnsembleNeuralNetwork, self).__init__()
                # Linear combination layer (no activation, no bias)
                self.linear_combination = nn.Linear(input_dim, linear_nodes, bias=False)
                
                # If fixed gamma values are provided, set them and freeze
                if fixed_gamma is not None:
                    with torch.no_grad():
                        self.linear_combination.weight.copy_(torch.FloatTensor(fixed_gamma))
                    self.linear_combination.weight.requires_grad = False
                
                # Build hidden layers
                self.hidden_layers = nn.ModuleList()
                prev_dim = linear_nodes
                for _ in range(num_hidden_layers):
                    self.hidden_layers.append(nn.Linear(prev_dim, hidden_nodes))
                    prev_dim = hidden_nodes
                
                # Output layer
                self.output = nn.Linear(prev_dim, 1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                # First layer: linear combination without activation
                x = self.linear_combination(x)
                
                # Hidden layers with ReLU
                for layer in self.hidden_layers:
                    x = self.relu(layer(x))
                
                # Output layer
                x = self.output(x)
                return x
            
            def get_linear_combination_weights(self):
                """Get the weights from the linear combination layer"""
                return self.linear_combination.weight.data.cpu().numpy()
        
        model = EnsembleNeuralNetwork(
            input_dim,
            self.config.num_linear,
            self.config.num_hidden_layers,
            self.config.num_hidden_nodes,
            fixed_gamma=fixed_gamma
        )
        
        return model
    
    def _compute_gamma_regularization_loss(self, model: Any) -> Any:
        """Compute soft quantization regularization loss for gamma values."""
        gamma_weights = model.linear_combination.weight  # Shape: (num_gamma, input_dim)
        
        # Define target simple values based on resolution
        resolution = self.config.gamma_reg_resolution
        
        if resolution == "integers":
            simple_values = torch.tensor([
                -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
            ], device=gamma_weights.device)
        elif resolution == "half-integers":
            simple_values = torch.tensor([
                -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
            ], device=gamma_weights.device)
        else:  # quarter-integers
            simple_values = torch.tensor([
                -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0,
                -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                1.75, 2.0, 2.25, 2.5, 2.75, 3.0
            ], device=gamma_weights.device)
        
        penalty = 0.0
        for gamma_value in gamma_weights.flatten():
            distances = torch.abs(gamma_value - simple_values)
            min_distance = torch.min(distances)
            penalty += min_distance
        
        return penalty
    
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """Train ensemble of neural networks."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for neural network training")
        
        if self.data is None:
            self.load_data()
        
        self._prepare_data_splits()
        
        print(f"\n=== Training Neural Networks ===")
        print(f"   Architecture: {self.config.num_linear} linear nodes, "
              f"{self.config.num_hidden_layers} hidden layers, "
              f"{self.config.num_hidden_nodes} nodes per layer")
        print(f"   Ensemble size: {self.config.num_ensembles}")
        print(f"   Epochs: {self.config.epochs}")
        print(f"   Learning rate: {self.config.learning_rate}")
        
        if self.config.use_fixed_gamma:
            print(f"   Using FIXED gamma values (not learning)")
        elif self.config.use_gamma_regularization:
            print(f"   Gamma regularization: {self.config.gamma_reg_strength} ({self.config.gamma_reg_resolution})")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(self.X_train)
        y_train_tensor = torch.FloatTensor(self.y_train)
        X_test_tensor = torch.FloatTensor(self.X_test)
        y_test_tensor = torch.FloatTensor(self.y_test)
        
        # Create data loaders
        batch_size = min(32, len(self.X_train))
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Train ensemble
        self.ensemble_models = []
        self.ensemble_histories = []
        
        criterion = nn.MSELoss()
        
        for ensemble_idx in range(self.config.num_ensembles):
            if verbose:
                print(f"\n--- Training Model {ensemble_idx + 1}/{self.config.num_ensembles} ---")
            
            # Set seed for this model
            if self.config.num_ensembles > 1:
                seed = self.config.random_seed + ensemble_idx * 12345
            else:
                seed = self.config.random_seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create model
            fixed_gamma = None
            if self.config.use_fixed_gamma and self.config.fixed_gamma_values is not None:
                fixed_gamma = self.config.fixed_gamma_values
            
            model = self._create_model(self.X_train.shape[1], fixed_gamma=fixed_gamma)
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            
            # Training history
            model_history = {"train": [], "test": []}
            
            # Training loop
            for epoch in range(self.config.epochs):
                # Training
                model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    # Prediction loss
                    prediction_loss = criterion(outputs, batch_y)
                    
                    # Add gamma regularization if enabled
                    if self.config.use_gamma_regularization and not self.config.use_fixed_gamma:
                        reg_strength = self.config.gamma_reg_strength
                        gamma_reg_loss = self._compute_gamma_regularization_loss(model)
                        total_loss = prediction_loss + reg_strength * gamma_reg_loss
                    else:
                        total_loss = prediction_loss
                    
                    total_loss.backward()
                    optimizer.step()
                    train_loss += total_loss.item()
                
                # Testing
                model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        test_loss += loss.item()
                
                train_loss = train_loss / len(train_loader)
                test_loss = test_loss / len(test_loader)
                
                model_history["train"].append(train_loss)
                model_history["test"].append(test_loss)
                
                # Print progress
                if verbose and (epoch + 1) % max(100, self.config.epochs // 10) == 0:
                    print(f"  Epoch [{epoch+1}/{self.config.epochs}], Train: {train_loss:.6f}, Test: {test_loss:.6f}")
            
            self.ensemble_models.append(model)
            self.ensemble_histories.append(model_history)
            
            if verbose:
                print(f"  Final test loss: {test_loss:.6f}")
        
        # Evaluate ensemble
        self._evaluate_ensemble(X_test_tensor, verbose=verbose)
        
        results = {
            'num_ensembles': self.config.num_ensembles,
            'num_linear': self.config.num_linear,
            'final_train_loss': [h['train'][-1] for h in self.ensemble_histories],
            'final_test_loss': [h['test'][-1] for h in self.ensemble_histories],
            'model_r2_scores': self.model_r2_scores.tolist() if self.model_r2_scores is not None else None
        }
        
        return results
    
    def _evaluate_ensemble(self, X_test_tensor: Any, verbose: bool = True):
        """Evaluate ensemble models and compute predictions."""
        if not HAS_TORCH or not HAS_SKLEARN:
            return
        
        num_ensembles = len(self.ensemble_models)
        
        # Get predictions from all ensemble members
        all_predictions = []
        all_gamma_weights = []
        model_r2_scores = []
        
        for model_idx, model in enumerate(self.ensemble_models):
            model.eval()
            with torch.no_grad():
                pred_scaled = model(X_test_tensor).cpu().numpy()
                all_predictions.append(pred_scaled)
                
                # Get gamma weights
                weights = model.get_linear_combination_weights()
                all_gamma_weights.append(weights)
                
                # Calculate R² for this model
                pred_orig = self.y_scaler.inverse_transform(pred_scaled)
                y_test_orig = self.y_scaler.inverse_transform(self.y_test)
                r2 = r2_score(y_test_orig, pred_orig)
                model_r2_scores.append(r2)
        
        all_predictions = np.array(all_predictions)  # shape: (n_ensembles, batch_size, 1)
        self.all_gamma_weights_raw = np.array(all_gamma_weights)  # shape: (n_ensembles, linear_nodes, input_dim)
        self.model_r2_scores = np.array(model_r2_scores)
        
        # Compute mean and std across ensemble
        mean_pred_scaled = all_predictions.mean(axis=0)
        std_pred_scaled = all_predictions.std(axis=0)
        
        # Transform back to original scale
        self.predictions = self.y_scaler.inverse_transform(mean_pred_scaled)
        self.prediction_std = std_pred_scaled * self.y_scaler.scale_
        
        # Use best R² model for learned coefficients
        best_model_idx = np.argmax(self.model_r2_scores)
        self.learned_coeffs = self.all_gamma_weights_raw[best_model_idx]
        
        if verbose:
            print(f"\n=== Ensemble Evaluation ===")
            print(f"   Best model: {best_model_idx + 1} (R² = {self.model_r2_scores[best_model_idx]:.6f})")
            if num_ensembles > 1:
                print(f"   Mean prediction uncertainty (std): {self.prediction_std.mean():.6f}")
    
    def _construct_discovered_equation(self) -> Dict[str, Any]:
        """Construct the discovered equation from learned gamma vectors."""
        if self.learned_coeffs is None:
            return {}
        
        # Input columns are log-transformed dimensionless variables (lgπ₁, lgπ₂, ...)
        # Gamma vectors combine these: π_discovered = exp(gamma @ lgπ) = π₁^gamma[0] * π₂^gamma[1] * ...
        
        discovered_equations = []
        dimensionless_groups = []
        dimensionless_groups_original = []  # Expressions in terms of original parameters
        
        for i in range(self.learned_coeffs.shape[0]):
            gamma = self.learned_coeffs[i, :]
            
            # Construct dimensionless group expression in terms of π
            expr_parts = []
            for j, var in enumerate(self.input_columns):
                coeff = gamma[j]
                if abs(coeff) > 1e-6:  # Only include significant coefficients
                    # Remove 'lg' prefix if present to get original variable name
                    var_name = var.replace('lg', '').replace('π', 'π')
                    if abs(coeff - 1.0) < 1e-6:
                        expr_parts.append(f"{var_name}")
                    elif abs(coeff + 1.0) < 1e-6:
                        expr_parts.append(f"{var_name}^(-1)")
                    else:
                        # Round to reasonable precision
                        coeff_rounded = round(coeff, 4)
                        expr_parts.append(f"{var_name}^({coeff_rounded})")
            
            if expr_parts:
                pi_expr = " × ".join(expr_parts)
                dimensionless_groups.append(f"π_discovered_{i+1} = {pi_expr}")
            else:
                pi_expr = "1"
                dimensionless_groups.append(f"π_discovered_{i+1} = 1")
            
            # Construct expression in terms of original parameters if basis vectors are available
            original_expr = None
            if self.basis_vectors is not None and len(self.original_parameter_names) > 0:
                # Map gamma vector to original parameters: original_coeffs = gamma @ basis_vectors.T
                # gamma shape: (num_pi,), basis_vectors shape: (num_original_params, num_pi)
                # Result: (num_original_params,)
                original_coeffs = gamma @ self.basis_vectors.T
                
                original_expr_parts = []
                for param_idx, param_name in enumerate(self.original_parameter_names):
                    coeff = original_coeffs[param_idx]
                    if abs(coeff) > 1e-6:  # Only include significant coefficients
                        if abs(coeff - 1.0) < 1e-6:
                            original_expr_parts.append(f"{param_name}")
                        elif abs(coeff + 1.0) < 1e-6:
                            original_expr_parts.append(f"{param_name}^(-1)")
                        elif abs(coeff - round(coeff)) < 1e-6:
                            # Integer coefficient
                            coeff_int = int(round(coeff))
                            if coeff_int == 0:
                                continue
                            elif abs(coeff_int) == 1:
                                original_expr_parts.append(f"{param_name}" if coeff_int > 0 else f"{param_name}^(-1)")
                            else:
                                original_expr_parts.append(f"{param_name}^{coeff_int}")
                        else:
                            # Non-integer coefficient
                            coeff_rounded = round(coeff, 4)
                            original_expr_parts.append(f"{param_name}^({coeff_rounded})")
                
                if original_expr_parts:
                    original_expr = " × ".join(original_expr_parts)
                else:
                    original_expr = "1"
            
            discovered_equations.append({
                'index': i + 1,
                'gamma_vector': gamma.tolist(),
                'expression': f"π_discovered_{i+1} = {pi_expr}",
                'expression_original_params': original_expr
            })
            
            # Add original parameter expression to dimensionless_groups_original
            if original_expr:
                dimensionless_groups_original.append(f"π_discovered_{i+1} = {original_expr}")
        
        # Construct overall equation description
        if self.config.num_linear == 1:
            equation_desc = f"Discovered: {dimensionless_groups[0]}"
        else:
            equation_desc = f"Discovered {self.config.num_linear} dimensionless groups:\n" + "\n".join(dimensionless_groups)
        
        return {
            'discovered_equation': equation_desc,
            'dimensionless_groups': dimensionless_groups,
            'dimensionless_groups_original_params': dimensionless_groups_original if dimensionless_groups_original else None,
            'detailed_equations': discovered_equations,
            'note': 'The neural network learns nonlinear relationships between these dimensionless groups and the output'
        }
    
    def _get_target_equation(self) -> Optional[Dict[str, Any]]:
        """Try to extract target equation from data generation config if available.
        
        Only returns target equation if:
        1. DATA_GENERATION is enabled in the config
        2. The input data file appears to be synthetic (contains 'synthetic' in path)
        """
        # Check if input file is from synthetic data generation
        input_file = self.config.input_file
        if input_file:
            input_path = Path(input_file)
            # Check if file path suggests synthetic data
            is_synthetic = 'synthetic' in str(input_path).lower() or 'dataset_synthetic' in str(input_path)
            if not is_synthetic:
                # Real experimental data - no target equation
                return None
        
        # Try to find data generation config file
        config_paths = [
            Path(self.config.output_dir) / '..' / 'pydimension' / 'configs' / 'config_synthetic.json',
            Path('pydimension/configs/config_synthetic.json'),
            Path('config_synthetic.json'),
            Path.cwd() / 'pydimension' / 'configs' / 'config_synthetic.json',
            Path.cwd() / 'config_synthetic.json'
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        full_config = json.load(f)
                    
                    data_gen = full_config.get('DATA_GENERATION', {})
                    # Check if data generation is enabled and has synthetic data parameters
                    if (data_gen and 
                        data_gen.get('enabled') is not False and  # enabled is True or not specified (default True)
                        data_gen.get('N') is not None):  # Indicates synthetic data
                        
                        ndim = data_gen.get('ndim', 1)
                        coefficients = data_gen.get('coefficients', [])
                        poly_order = data_gen.get('poly_order', 1)
                        
                        # Construct target equation
                        if ndim == 1:
                            # Polynomial: p* = c0 + c1*π1 + c2*π1^2 + ...
                            terms = []
                            for i, coeff in enumerate(coefficients):
                                if i == 0:
                                    terms.append(f"{coeff:.4g}")
                                elif i == 1:
                                    terms.append(f"{coeff:.4g}×π₁")
                                else:
                                    terms.append(f"{coeff:.4g}×π₁^{i}")
                            
                            target_eq = "p* = " + " + ".join(terms)
                        else:
                            # Nonlinear: p* = exp(A×π₁) + π₂^B + log(1+C×π₃)
                            terms = []
                            if len(coefficients) > 0:
                                terms.append(f"exp({coefficients[0]:.4g}×π₁)")
                            if ndim >= 2 and len(coefficients) > 1:
                                terms.append(f"π₂^{coefficients[1]:.4g}")
                            if ndim >= 3 and len(coefficients) > 2:
                                terms.append(f"log(1+{coefficients[2]:.4g}×π₃)")
                            
                            target_eq = "p* = " + " + ".join(terms) if terms else "p* = (nonlinear function)"
                        
                        return {
                            'target_equation': target_eq,
                            'ndim': ndim,
                            'coefficients': coefficients,
                            'poly_order': poly_order if ndim == 1 else None,
                            'data_type': 'synthetic',
                            'source_config': str(config_path)
                        }
                except Exception:
                    pass
        
        return None
    
    def save_results(self) -> Tuple[str, str]:
        """Save training results to JSON file."""
        if self.learned_coeffs is None:
            raise ValueError("No results to save. Train models first.")
        
        # Create results directory
        results_dir = Path(self.config.output_dir) / self.config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct discovered equation
        discovered_eq_info = self._construct_discovered_equation()
        
        # Try to get target equation (for synthetic data)
        target_eq_info = self._get_target_equation()
        
        # Prepare results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_linear': self.config.num_linear,
                'num_hidden_layers': self.config.num_hidden_layers,
                'num_hidden_nodes': self.config.num_hidden_nodes,
                'num_ensembles': self.config.num_ensembles,
                'epochs': self.config.epochs,
                'learning_rate': self.config.learning_rate,
                'train_percent': self.config.train_percent,
                'random_seed': self.config.random_seed,
                'use_fixed_gamma': self.config.use_fixed_gamma,
                'use_gamma_regularization': self.config.use_gamma_regularization,
                'gamma_reg_strength': self.config.gamma_reg_strength,
                'gamma_reg_resolution': self.config.gamma_reg_resolution
            },
            'training_history': {
                'train_loss': [h['train'] for h in self.ensemble_histories],
                'test_loss': [h['test'] for h in self.ensemble_histories]
            },
            'model_performance': {
                'model_r2_scores': self.model_r2_scores.tolist() if self.model_r2_scores is not None else None,
                'final_train_loss': [h['train'][-1] for h in self.ensemble_histories],
                'final_test_loss': [h['test'][-1] for h in self.ensemble_histories]
            },
            'learned_gamma_vectors': {
                'best_model_index': int(np.argmax(self.model_r2_scores)) if self.model_r2_scores is not None else 0,
                'gamma_vectors': self.learned_coeffs.tolist(),
                'all_models_gamma_vectors': self.all_gamma_weights_raw.tolist() if self.all_gamma_weights_raw is not None else None
            },
            'discovered_equation': discovered_eq_info,
            'discovered_pi': discovered_eq_info.get('dimensionless_groups', []) if discovered_eq_info else [],
            'input_columns': self.input_columns,
            'output_column': self.output_column
        }
        
        # Add target equation if available (for synthetic data)
        if target_eq_info:
            results['target_equation'] = target_eq_info
            print(f"\n📊 Target equation (synthetic data): {target_eq_info['target_equation']}")
        
        # Print discovered equation with gamma coefficients
        if discovered_eq_info:
            print(f"\n🔍 Discovered equation:")
            print(f"   {discovered_eq_info.get('discovered_equation', 'N/A')}")
            
            # Print final identified pi expressions
            dimensionless_groups = discovered_eq_info.get('dimensionless_groups', [])
            if dimensionless_groups:
                print(f"\n📐 Final identified dimensionless groups (π):")
                for pi_expr in dimensionless_groups:
                    print(f"   {pi_expr}")
            
            # Print expressions in terms of original parameters if available
            dimensionless_groups_original = discovered_eq_info.get('dimensionless_groups_original_params')
            if dimensionless_groups_original:
                print(f"\n📐 Final identified dimensionless groups (in terms of original parameters):")
                for pi_expr in dimensionless_groups_original:
                    print(f"   {pi_expr}")
            
            # Print gamma coefficients for each discovered dimensionless group
            detailed_eqs = discovered_eq_info.get('detailed_equations', [])
            if detailed_eqs:
                print(f"\n📊 Gamma coefficients (γ vectors):")
                for eq in detailed_eqs:
                    idx = eq.get('index', 0)
                    gamma = eq.get('gamma_vector', [])
                    if gamma:
                        # Format gamma vector nicely
                        gamma_str = ", ".join([f"{g:.6f}" for g in gamma])
                        print(f"   γ_{idx}: [{gamma_str}]")
                        print(f"      → {eq.get('expression', 'N/A')}")
        
        # Create original scale π vs output plot
        try:
            original_scale_pi_plot_path = self.create_original_scale_pi_plot()
            results['original_scale_pi_plot_path'] = original_scale_pi_plot_path
        except Exception as e:
            print(f"⚠️  Could not create original scale π plot: {e}")
        
        # Create gamma distribution plot if length of gamma vectors (num_input_pi) == 3
        try:
            gamma_dist_plot_path = self.create_gamma_distribution_plot()
            if gamma_dist_plot_path:
                results['gamma_distribution_plot_path'] = gamma_dist_plot_path
        except Exception as e:
            print(f"⚠️  Could not create gamma distribution plot: {e}")
        
        # Save gamma vectors for each model to CSV
        try:
            gamma_vectors_csv_path = self._save_gamma_vectors_csv(results_dir)
            if gamma_vectors_csv_path:
                results['gamma_vectors_csv_path'] = gamma_vectors_csv_path
        except Exception as e:
            print(f"⚠️  Could not save gamma vectors CSV: {e}")
        
        # Save to file
        results_path = results_dir / self.config.model_results_filename
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Saved results to: {results_path}")
        
        return str(results_path), str(results_dir)
    
    def create_gamma_distribution_plot(self, filename: Optional[str] = None) -> Optional[str]:
        """Create and save a 3D plot of all identified gamma vectors when length equals 3.
        
        This plot shows all gamma vectors from all ensemble models in 3D space.
        Each axis represents one of the 3 input π components.
        Each point represents one gamma vector from one model.
        
        Args:
            filename: Filename for saved plot (default: 'gamma_distribution.png')
            
        Returns:
            Path to saved plot file, or None if length of gamma vectors != 3 or data not available
        """
        if self.all_gamma_weights_raw is None:
            return None
        
        # all_gamma_weights_raw shape: (n_ensembles, num_gamma_vectors, num_input_pi)
        # Check if length of gamma vectors (num_input_pi) equals 3
        num_input_pi = self.all_gamma_weights_raw.shape[2]
        if num_input_pi != 3:
            return None
        
        if filename is None:
            filename = 'gamma_distribution.png'
        
        # Create figures directory
        figures_dir = Path(self.config.output_dir) / self.config.figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = figures_dir / filename
        
        # Flatten all gamma vectors from all models
        # Shape: (n_ensembles * num_gamma_vectors, 3)
        n_ensembles = self.all_gamma_weights_raw.shape[0]
        num_gamma_vectors = self.all_gamma_weights_raw.shape[1]
        
        # Reshape to get all gamma vectors: (n_ensembles * num_gamma_vectors, 3)
        all_gamma_flat = self.all_gamma_weights_raw.reshape(-1, 3)
        
        # Extract components for 3D plot
        x_values = all_gamma_flat[:, 0]  # First input π component
        y_values = all_gamma_flat[:, 1]  # Second input π component
        z_values = all_gamma_flat[:, 2]  # Third input π component
        
        # Create labels for axes (use input column names if available)
        if self.input_columns and len(self.input_columns) >= 3:
            x_label = f'γ·{self.input_columns[0]}'
            y_label = f'γ·{self.input_columns[1]}'
            z_label = f'γ·{self.input_columns[2]}'
        else:
            x_label = 'γ·π₁'
            y_label = 'γ·π₂'
            z_label = 'γ·π₃'
        
        # Create 3D plot
        plt.close('all')
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot with single blue color for all gamma vectors
        scatter = ax.scatter(
            x_values,
            y_values,
            z_values,
            c='blue',
            s=60,
            alpha=0.7,
            edgecolors='black',
            linewidth=1.0,
        )
        
        # Add labels
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_zlabel(z_label, fontsize=12, fontweight='bold')
        ax.set_title(f'Distribution of All Identified Gamma Vectors\n({n_ensembles} models × {num_gamma_vectors} gamma vectors = {len(all_gamma_flat)} points)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved gamma distribution plot to: {plot_path}")
        print(f"   Plotted {len(all_gamma_flat)} gamma vectors from {n_ensembles} ensemble models")
        print(f"   Each gamma vector has 3 components (3 input π)")
        
        return str(plot_path)
    
    def _save_gamma_vectors_csv(self, results_dir: Path) -> Optional[str]:
        """Save gamma vectors for each ensemble model to a CSV file.
        
        Args:
            results_dir: Directory to save the CSV file
            
        Returns:
            Path to saved CSV file, or None if data not available
        """
        if self.all_gamma_weights_raw is None:
            return None
        
        # all_gamma_weights_raw shape: (n_ensembles, num_gamma_vectors, num_input_pi)
        n_ensembles = self.all_gamma_weights_raw.shape[0]
        num_gamma_vectors = self.all_gamma_weights_raw.shape[1]
        num_input_pi = self.all_gamma_weights_raw.shape[2]
        
        # Create DataFrame with gamma vectors for each model
        # Each row represents one ensemble model
        # Columns: Model, R²_Score, γ₁_component1, γ₁_component2, ..., γ₂_component1, ...
        data_rows = []
        
        for model_idx in range(n_ensembles):
            row = {'Model': f'Model_{model_idx + 1}'}
            
            # Add R² score if available
            if self.model_r2_scores is not None:
                row['R²_Score'] = self.model_r2_scores[model_idx]
            else:
                row['R²_Score'] = None
            
            # Add each gamma vector's components
            for gamma_idx in range(num_gamma_vectors):
                gamma_vector = self.all_gamma_weights_raw[model_idx, gamma_idx, :]
                for pi_idx in range(num_input_pi):
                    # Column name: γ{gamma_idx+1}_{input_column} or γ{gamma_idx+1}_π{pi_idx+1}
                    if self.input_columns and pi_idx < len(self.input_columns):
                        col_name = f'γ{gamma_idx+1}_{self.input_columns[pi_idx]}'
                    else:
                        col_name = f'γ{gamma_idx+1}_π{pi_idx+1}'
                    row[col_name] = gamma_vector[pi_idx]
            
            data_rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        # Save to CSV
        csv_path = results_dir / 'gamma_vectors_all_models.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Saved gamma vectors for all models to: {csv_path}")
        print(f"   {n_ensembles} models, {num_gamma_vectors} gamma vectors per model, {num_input_pi} components per gamma vector")
        
        return str(csv_path)
    
    def create_architecture_plot(self, filename: Optional[str] = None) -> str:
        """Create and save a schematic plot of the neural network architecture.
        
        The plot reflects the current Optimization Discovery hyperparameters:
        - Number of input π-groups (from data)
        - Number of linear (γ) nodes
        - Number of hidden layers and nodes per layer
        - Single output node
        """
        if self.X_train is None:
            raise ValueError("Training data must be prepared before plotting architecture. Run train() or process() first.")
        
        # Determine layer sizes from configuration and data
        input_dim = self.X_train.shape[1]
        num_linear = self.config.num_linear
        num_hidden_layers = self.config.num_hidden_layers
        num_hidden_nodes = self.config.num_hidden_nodes
        
        layers = [
            ("Input layer", f"{input_dim} input π-groups"),
            ("Input π groups", f"{num_linear} nodes"),
        ]
        
        for i in range(num_hidden_layers):
            layers.append((f"Hidden layer {i+1}", f"{num_hidden_nodes} nodes"))
        
        layers.append(("Output layer", "1 node"))
        
        # Choose filename and directory
        if filename is None:
            filename = "optimization_discovery_architecture.png"
        
        figures_dir = Path(self.config.output_dir) / self.config.figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_path = figures_dir / filename
        
        # Build node-level schematic with circles and connections
        plt.close('all')
        fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(layers)), 6))
        
        # Define layer sizes and names
        layer_sizes = [input_dim, num_linear] + [num_hidden_nodes] * num_hidden_layers + [1]
        layer_names = [name for name, _ in layers]
        
        # Horizontal positions for each layer
        x_positions = np.linspace(0.5, len(layer_sizes) - 0.5, len(layer_sizes))
        
        # Colors per layer type
        layer_colors = []
        for idx, name in enumerate(layer_names):
            # First layer: raw inputs
            if idx == 0:
                layer_colors.append("#1f77b4")  # blue
            # Second layer: input π groups (linear combination layer)
            elif idx == 1:
                layer_colors.append("#ff7f0e")  # orange
            # Hidden layers
            elif "Hidden" in name:
                layer_colors.append("#2ca02c")  # green
            # Output layer
            else:
                layer_colors.append("#d62728")  # red
        
        # Store node coordinates to draw connections
        node_coords = {}
        
        # Draw neurons (circles) for each layer
        for layer_idx, (n_nodes, x, color, name) in enumerate(
            zip(layer_sizes, x_positions, layer_colors, layer_names)
        ):
            # Vertically spread nodes, leaving extra room at the top for labels
            if n_nodes == 1:
                y_positions = np.array([0.45])
            else:
                y_positions = np.linspace(0.1, 0.8, n_nodes)
            
            node_coords[layer_idx] = []
            for node_idx, y in enumerate(y_positions):
                ax.scatter(
                    x,
                    y,
                    s=300,
                    color=color,
                    edgecolors="black",
                    zorder=3,
                )
                node_coords[layer_idx].append((x, y))
            
            # Add layer label above the neurons (in a non-overlapping band)
            ax.text(
                x,
                0.98,
                name,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )
            ax.text(
                x,
                0.92,
                f"{n_nodes} node{'s' if n_nodes != 1 else ''}",
                ha="center",
                va="top",
                fontsize=9,
            )
        
        # Draw connections between layers
        for layer_idx in range(len(layer_sizes) - 1):
            for (x1, y1) in node_coords[layer_idx]:
                for (x2, y2) in node_coords[layer_idx + 1]:
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color="#bbbbbb",
                        linewidth=1.0,
                        zorder=1,
                    )
        
        # Title summarizing architecture
        arch_desc = (
            f"{input_dim} input features, "
            f"{num_linear} input π groups, "
            f"{num_hidden_layers} hidden layer(s) × {num_hidden_nodes} nodes, "
            "1 output"
        )
        ax.set_title(
            f"Neural Network Architecture for Optimization Discovery\n{arch_desc}",
            fontsize=12,
            fontweight="bold",
            pad=20,
        )
        
        ax.set_xlim(0, len(layer_sizes) + 0.5)
        ax.set_ylim(0, 1.05)
        ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"✅ Saved architecture plot to: {plot_path}")
        
        return str(plot_path)
    
    def create_visualization(self, filename: Optional[str] = None) -> str:
        """Create and save visualization plots with individual loss history for each model."""
        if self.learned_coeffs is None:
            raise ValueError("No results to visualize. Train models first.")
        
        if filename is None:
            filename = self.config.plot_filename
        
        # Create figures directory
        figures_dir = Path(self.config.output_dir) / self.config.figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = figures_dir / filename
        
        num_ensembles = len(self.ensemble_models)
        num_gamma = self.learned_coeffs.shape[0]
        
        # Calculate layout: 2 columns (loss history + predictions) for each model
        n_cols = 2
        n_rows = num_ensembles + 1  # +1 for ensemble summary plot
        
        # Create figure with subplots - each model gets its own row
        fig = plt.figure(figsize=(16, 5 * n_rows))
        
        # Plot 0: Summary - All models training history (overview)
        ax_summary = plt.subplot(n_rows, n_cols, 1)
        for idx, history in enumerate(self.ensemble_histories):
            ax_summary.plot(history['train'], label=f'Model {idx+1} Train', alpha=0.7, linewidth=1.5)
            ax_summary.plot(history['test'], label=f'Model {idx+1} Test', linestyle='--', alpha=0.7, linewidth=1.5)
        ax_summary.set_xlabel('Epoch', fontsize=10)
        ax_summary.set_ylabel('Loss', fontsize=10)
        ax_summary.set_title(f'Training History - All Models Overview ({num_ensembles} Models)', fontsize=12, fontweight='bold')
        ax_summary.legend(ncol=2, fontsize=8, loc='upper right')
        ax_summary.grid(True, alpha=0.3)
        
        # Plot 1: Ensemble predictions vs actual
        ax_ensemble = plt.subplot(n_rows, n_cols, 2)
        if self.predictions is not None and HAS_SKLEARN:
            y_test_orig = self.y_scaler.inverse_transform(self.y_test)
            ax_ensemble.scatter(y_test_orig.flatten(), self.predictions.flatten(), alpha=0.6, s=20, c='blue')
            min_val = min(y_test_orig.min(), self.predictions.min())
            max_val = max(y_test_orig.max(), self.predictions.max())
            ax_ensemble.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Calculate R²
            r2 = r2_score(y_test_orig, self.predictions)
            ax_ensemble.set_title(f'Ensemble Predictions vs Actual (R² = {r2:.4f})', fontsize=12, fontweight='bold')
            ax_ensemble.legend(fontsize=9)
        else:
            ax_ensemble.text(0.5, 0.5, 'Predictions not available', ha='center', va='center', transform=ax_ensemble.transAxes)
        ax_ensemble.set_xlabel('Actual', fontsize=10)
        ax_ensemble.set_ylabel('Predicted', fontsize=10)
        ax_ensemble.grid(True, alpha=0.3)
        
        # Individual plots for each model
        for model_idx in range(num_ensembles):
            row = model_idx + 1  # +1 because row 0 is summary
            
            # Get model history
            history = self.ensemble_histories[model_idx]
            
            # Plot: Individual model loss history
            ax_loss = plt.subplot(n_rows, n_cols, row * n_cols + 1)
            epochs = range(1, len(history['train']) + 1)
            ax_loss.plot(epochs, history['train'], label='Train Loss', color='blue', linewidth=2, alpha=0.8)
            ax_loss.plot(epochs, history['test'], label='Test Loss', color='red', linewidth=2, alpha=0.8, linestyle='--')
            
            # Add final loss values as text
            final_train = history['train'][-1]
            final_test = history['test'][-1]
            ax_loss.text(0.02, 0.98, f'Final Train: {final_train:.6f}\nFinal Test: {final_test:.6f}',
                        transform=ax_loss.transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Add R² score if available
            if self.model_r2_scores is not None:
                r2_score_model = self.model_r2_scores[model_idx]
                is_best = (model_idx == np.argmax(self.model_r2_scores))
                title_suffix = ' * (Best)' if is_best else ''
                ax_loss.set_title(f'Model {model_idx+1} Loss History (R² = {r2_score_model:.4f}){title_suffix}',
                                fontsize=11, fontweight='bold')
            else:
                ax_loss.set_title(f'Model {model_idx+1} Loss History', fontsize=11, fontweight='bold')
            
            ax_loss.set_xlabel('Epoch', fontsize=10)
            ax_loss.set_ylabel('Loss', fontsize=10)
            ax_loss.legend(fontsize=9, loc='upper right')
            ax_loss.grid(True, alpha=0.3)
            
            # Plot: Individual model predictions vs actual
            ax_pred = plt.subplot(n_rows, n_cols, row * n_cols + 2)
            if HAS_TORCH and HAS_SKLEARN and self.X_scaler is not None and self.y_scaler is not None:
                # Get predictions for this specific model
                # Note: self.X_test is already scaled from _prepare_data_splits()
                model = self.ensemble_models[model_idx]
                X_test_tensor = torch.FloatTensor(self.X_test)
                
                model.eval()
                with torch.no_grad():
                    pred_scaled = model(X_test_tensor).cpu().numpy()
                    pred_orig = self.y_scaler.inverse_transform(pred_scaled).flatten()
                
                y_test_orig = self.y_scaler.inverse_transform(self.y_test).flatten()
                
                ax_pred.scatter(y_test_orig, pred_orig, alpha=0.6, s=20, c='green')
                min_val = min(y_test_orig.min(), pred_orig.min())
                max_val = max(y_test_orig.max(), pred_orig.max())
                ax_pred.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                
                # Calculate R² for this model
                r2_model = r2_score(y_test_orig, pred_orig)
                is_best = (model_idx == np.argmax(self.model_r2_scores)) if self.model_r2_scores is not None else False
                title_suffix = ' * (Best)' if is_best else ''
                ax_pred.set_title(f'Model {model_idx+1} Predictions (R² = {r2_model:.4f}){title_suffix}',
                                fontsize=11, fontweight='bold')
                ax_pred.legend(fontsize=9)
            else:
                ax_pred.text(0.5, 0.5, 'Predictions not available', ha='center', va='center',
                           transform=ax_pred.transAxes)
                ax_pred.set_title(f'Model {model_idx+1} Predictions', fontsize=11, fontweight='bold')
            
            ax_pred.set_xlabel('Actual', fontsize=10)
            ax_pred.set_ylabel('Predicted', fontsize=10)
            ax_pred.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved plot to: {plot_path}")
        
        # Also create a schematic of the neural network architecture
        try:
            self.create_architecture_plot()
        except Exception as e:
            print(f"⚠️  Could not create architecture plot: {e}")
        
        return str(plot_path)
    
    def _load_original_data(self) -> Optional[pd.DataFrame]:
        """Try to load the original dataset file from data preprocessing.
        
        First tries to load original_data.csv saved by data preprocessing module,
        then falls back to original dataset files.
        
        Returns:
            DataFrame with original data, or None if not found
        """
        base_path = Path(self.config.output_dir).resolve()
        cwd = Path.cwd()
        
        # First, try to load original_data.csv saved by data preprocessing
        possible_paths = [
            base_path / self.config.data_dir / 'original_data.csv',
            Path('output') / self.config.data_dir / 'original_data.csv',
            Path('original_data.csv'),
            cwd / 'output' / self.config.data_dir / 'original_data.csv',
            cwd / 'original_data.csv',
        ]
        
        # Also try to infer from the input file path
        if self.config.input_file:
            input_path = Path(self.config.input_file)
            if 'normalized_lg_afterDA_data' in str(input_path):
                possible_paths.insert(0, input_path.parent / 'original_data.csv')
                possible_paths.insert(0, input_path.parent.parent / 'original_data.csv')
        
        for path in possible_paths:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    print(f"✅ Loaded original data from: {path}")
                    return df
                except Exception as e:
                    print(f"⚠️  Could not load {path}: {e}")
                    continue
        
        # Fall back to original dataset files
        fallback_paths = [
            base_path / self.config.data_dir / 'dataset_synthetic.csv',
            base_path / self.config.data_dir / 'dataset_keyhole.csv',
            Path('output') / self.config.data_dir / 'dataset_synthetic.csv',
            Path('output') / self.config.data_dir / 'dataset_keyhole.csv',
            Path('dataset_synthetic.csv'),
            Path('dataset_keyhole.csv'),
            cwd / 'output' / self.config.data_dir / 'dataset_synthetic.csv',
            cwd / 'output' / self.config.data_dir / 'dataset_keyhole.csv',
            cwd / 'dataset_synthetic.csv',
            cwd / 'dataset_keyhole.csv',
        ]
        
        if self.config.input_file:
            input_path = Path(self.config.input_file)
            if 'normalized_lg_afterDA_data' in str(input_path):
                fallback_paths.insert(0, input_path.parent / 'dataset_synthetic.csv')
                fallback_paths.insert(0, input_path.parent.parent / 'dataset_synthetic.csv')
        
        for path in fallback_paths:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    print(f"✅ Loaded original dataset from: {path}")
                    return df
                except Exception as e:
                    print(f"⚠️  Could not load {path}: {e}")
                    continue
        
        return None
    
    def create_discovered_pi_plot(self, filename: Optional[str] = None) -> str:
        """Create and save the discovered input π vs output π plot.
        
        This plot shows the discovered input dimensionless number (π_discovered) on the x-axis
        and the output π (or output variable) on the y-axis, calculated from original data scale.
        
        The discovered π is calculated by:
        1. Using basis vectors and gamma vectors to get power indices for each original parameter
        2. Applying these power indices to the original data: π_discovered = p1^power[0] * p2^power[1] * ...
        
        Args:
            filename: Filename for saved plot (default: 'discovered_pi_vs_output.png')
            
        Returns:
            Path to saved plot file
        """
        if self.learned_coeffs is None:
            raise ValueError("No results to visualize. Train models first.")
        
        if self.basis_vectors is None or len(self.original_parameter_names) == 0:
            raise ValueError("Basis vectors and original parameter names are required. "
                           "Cannot calculate discovered π from original parameters.")
        
        if filename is None:
            filename = 'discovered_pi_vs_output.png'
        
        # Create figures directory
        figures_dir = Path(self.config.output_dir) / self.config.figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = figures_dir / filename
        
        # Load original data
        original_data = self._load_original_data()
        if original_data is None:
            raise ValueError("Could not find original dataset file. "
                           "Please ensure the original dataset file (e.g., dataset_synthetic.csv) exists.")
        
        # Get the first gamma vector (for the primary discovered dimensionless group)
        gamma = self.learned_coeffs[0, :]  # Shape: (num_input_pi,)
        
        # Calculate power indices for each original parameter
        # basis_vectors shape: (num_original_params, num_input_pi)
        # gamma shape: (num_input_pi,)
        # power_indices shape: (num_original_params,)
        # power_indices = basis_vectors @ gamma
        power_indices = np.dot(self.basis_vectors, gamma)  # Shape: (num_original_params,)
        
        # Get original parameter columns from the original data
        # Match original_parameter_names with columns in original_data
        original_params_data = []
        matched_param_names = []
        for param_name in self.original_parameter_names:
            if param_name in original_data.columns:
                original_params_data.append(original_data[param_name].values)
                matched_param_names.append(param_name)
            else:
                print(f"⚠️  Warning: Parameter {param_name} not found in original data")
        
        if len(original_params_data) == 0:
            raise ValueError("No matching parameters found between basis vectors and original data")
        
        # Filter power indices to match found parameters
        # Find indices of matched parameters in original_parameter_names
        matched_indices = [i for i, name in enumerate(self.original_parameter_names) 
                          if name in matched_param_names]
        matched_power_indices = power_indices[matched_indices]
        
        # Calculate discovered π from original parameters
        # π_discovered = p1^power[0] * p2^power[1] * ... * pN^power[N-1]
        # Use log to avoid overflow: log(π) = sum(power[i] * log(p_i))
        n_samples = len(original_params_data[0])
        log_discovered_pi = np.zeros(n_samples)
        
        for i, (param_data, power) in enumerate(zip(original_params_data, matched_power_indices)):
            # Avoid log(0) by clipping small values
            param_data_clipped = np.maximum(param_data, 1e-10)
            log_discovered_pi += power * np.log(param_data_clipped)
        
        discovered_pi = np.exp(log_discovered_pi)  # Shape: (n_samples,)
        
        # Get output values from training data (same scale as model training)
        # self.data contains normalized_lg_afterDA_data.csv which is what the model was trained on
        if self.data is None:
            raise ValueError("Training data must be loaded first. Call load_data() before creating plot.")
        
        # Use the output column from training data (same as used for model training)
        # This output is in the same scale as the training data (normalized but not standardized)
        output_pi = self.data[self.output_column].values  # Shape: (n_samples,)
        
        print(f"✅ Using output from training data: {self.output_column}")
        
        # Ensure we have the same number of samples
        # Both should have the same number since they come from the same dataset
        if len(discovered_pi) != len(output_pi):
            min_len = min(len(discovered_pi), len(output_pi))
            print(f"⚠️  Warning: Sample count mismatch. Using first {min_len} samples.")
            discovered_pi = discovered_pi[:min_len]
            output_pi = output_pi[:min_len]
        
        # Recreate the same train/test split used during training
        # This ensures we use the same data points for training and testing
        n_samples = len(discovered_pi)
        np.random.seed(self.config.random_seed)
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        cut = int(self.config.train_percent * n_samples)
        tr_idx, te_idx = idx[:cut], idx[cut:]
        
        # Split data into training and test sets
        discovered_pi_train = discovered_pi[tr_idx]
        discovered_pi_test = discovered_pi[te_idx]
        output_pi_train = output_pi[tr_idx]
        output_pi_test = output_pi[te_idx]
        
        # Create the plot
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Scatter plot: discovered input π vs output π with different colors for train/test
        ax.scatter(discovered_pi_train, output_pi_train, alpha=0.6, s=30, 
                  c='blue', edgecolors='navy', label='Training data', zorder=2)
        ax.scatter(discovered_pi_test, output_pi_test, alpha=0.6, s=30, 
                  c='red', edgecolors='darkred', label='Test data', zorder=2)
        
        # Set labels
        ax.set_xlabel('Discovered Input π', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Output π ({self.output_column})', fontsize=12, fontweight='bold')
        ax.set_title('Discovered Input π vs Output π', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved discovered π plot to: {plot_path}")
        print(f"   Calculated from normalized parameters: {matched_param_names}")
        print(f"   Power indices: {matched_power_indices}")
        print(f"   Output from training data scale: {self.output_column}")
        print(f"   Training samples: {len(discovered_pi_train)}, Test samples: {len(discovered_pi_test)}")
        
        return str(plot_path)
    
    def create_original_scale_pi_plot(self, filename: Optional[str] = None) -> str:
        """Create and save the input π vs output π plot in original data scale.
        
        This plot shows the input dimensionless number (π_input) calculated from original
        (unnormalized) parameters using the identified power indices, plotted against the
        output π in original scale.
        
        The input π is calculated by:
        1. Using the identified power indices (from basis_vectors @ gamma)
        2. Applying these power indices to the original (unnormalized) data:
           π_input = p1_orig^power[0] * p2_orig^power[1] * ...
        
        Args:
            filename: Filename for saved plot (default: 'original_scale_pi_vs_output.png')
            
        Returns:
            Path to saved plot file
        """
        if self.learned_coeffs is None:
            raise ValueError("No results to visualize. Train models first.")
        
        if self.basis_vectors is None or len(self.original_parameter_names) == 0:
            raise ValueError("Basis vectors and original parameter names are required. "
                           "Cannot calculate input π from original parameters.")
        
        if filename is None:
            filename = 'original_scale_pi_vs_output.png'
        
        # Create figures directory
        figures_dir = Path(self.config.output_dir) / self.config.figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = figures_dir / filename
        
        # Load original data (saved by data preprocessing)
        original_data = self._load_original_data()
        if original_data is None:
            raise ValueError("Could not find original data file. "
                           "Please ensure original_data.csv exists from data preprocessing step.")
        
        # Get the first gamma vector (for the primary discovered dimensionless group)
        gamma = self.learned_coeffs[0, :]  # Shape: (num_input_pi,)
        
        # Calculate power indices for each original parameter
        # basis_vectors shape: (num_original_params, num_input_pi)
        # gamma shape: (num_input_pi,)
        # power_indices shape: (num_original_params,)
        power_indices = np.dot(self.basis_vectors, gamma)  # Shape: (num_original_params,)
        
        # Get original parameter columns from the original data
        # Match original_parameter_names with columns in original_data
        original_params_data = []
        matched_param_names = []
        for param_name in self.original_parameter_names:
            if param_name in original_data.columns:
                original_params_data.append(original_data[param_name].values)
                matched_param_names.append(param_name)
            else:
                print(f"⚠️  Warning: Parameter {param_name} not found in original data")
        
        if len(original_params_data) == 0:
            raise ValueError("No matching parameters found between basis vectors and original data")
        
        # Filter power indices to match found parameters
        matched_indices = [i for i, name in enumerate(self.original_parameter_names) 
                          if name in matched_param_names]
        matched_power_indices = power_indices[matched_indices]
        
        # Calculate input π from original (unnormalized) parameters
        # π_input = p1_orig^power[0] * p2_orig^power[1] * ... * pN_orig^power[N-1]
        # Use log to avoid overflow: log(π) = sum(power[i] * log(p_i_orig))
        n_samples = len(original_params_data[0])
        log_input_pi = np.zeros(n_samples)
        
        for i, (param_data, power) in enumerate(zip(original_params_data, matched_power_indices)):
            # Avoid log(0) by clipping small values
            param_data_clipped = np.maximum(param_data, 1e-10)
            log_input_pi += power * np.log(param_data_clipped)
        
        input_pi = np.exp(log_input_pi)  # Shape: (n_samples,)
        
        # Get output values from original data (in original scale)
        output_col = None
        if self.output_column:
            # Try to find output in original data
            output_name = self.output_column.replace('lg', '').replace('π', '').strip()
            
            if output_name in original_data.columns:
                output_col = output_name
            elif self.output_column in original_data.columns:
                output_col = self.output_column
            else:
                # Try to find columns ending with * (common output pattern)
                for col in original_data.columns:
                    if col.endswith('*') or col == 'p*' or col == 'e*':
                        output_col = col
                        break
        
        if output_col is None:
            # Use last column as fallback
            output_col = original_data.columns[-1]
            print(f"⚠️  Using last column as output: {output_col}")
        else:
            print(f"✅ Using output column: {output_col}")
        
        if output_col not in original_data.columns:
            raise ValueError(f"Output column '{output_col}' not found in original data. "
                           f"Available columns: {list(original_data.columns)}")
        
        output_pi = original_data[output_col].values
        
        # Ensure we have the same number of samples
        if len(input_pi) != len(output_pi):
            min_len = min(len(input_pi), len(output_pi))
            print(f"⚠️  Warning: Sample count mismatch. Using first {min_len} samples.")
            input_pi = input_pi[:min_len]
            output_pi = output_pi[:min_len]
        
        # Recreate the same train/test split used during training
        n_samples = len(input_pi)
        np.random.seed(self.config.random_seed)
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        cut = int(self.config.train_percent * n_samples)
        tr_idx, te_idx = idx[:cut], idx[cut:]
        
        # Split data into training and test sets
        input_pi_train = input_pi[tr_idx]
        input_pi_test = input_pi[te_idx]
        output_pi_train = output_pi[tr_idx]
        output_pi_test = output_pi[te_idx]
        
        # Create the plot
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Scatter plot: input π vs output π with different colors for train/test
        ax.scatter(input_pi_train, output_pi_train, alpha=0.6, s=30, 
                  c='blue', edgecolors='navy', label='Training data', zorder=2)
        ax.scatter(input_pi_test, output_pi_test, alpha=0.6, s=30, 
                  c='red', edgecolors='darkred', label='Test data', zorder=2)
        
        # Set labels
        ax.set_xlabel('Input π (Original Scale)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Output π ({output_col})', fontsize=12, fontweight='bold')
        ax.set_title('Input π vs Output π (Original Data Scale)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved original scale π plot to: {plot_path}")
        print(f"   Calculated from original (unnormalized) parameters: {matched_param_names}")
        print(f"   Power indices: {matched_power_indices}")
        print(f"   Input π range: [{input_pi.min():.6f}, {input_pi.max():.6f}]")
        print(f"   Output π range: [{output_pi.min():.6f}, {output_pi.max():.6f}]")
        print(f"   Training samples: {len(input_pi_train)}, Test samples: {len(input_pi_test)}")
        
        return str(plot_path)
    
    def process(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the complete optimization and discovery process."""
        # Load data
        self.load_data()
        
        # Train models
        results = self.train(verbose=verbose)
        
        return results

