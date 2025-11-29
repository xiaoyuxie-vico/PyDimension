"""
Core data generation logic for synthetic datasets with known dimensionless relationships.
"""

import numpy as np
import pandas as pd
from numpy.linalg import matrix_rank
from scipy.linalg import null_space
import os
from typing import List, Optional, Tuple
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from .config import DataGenerationConfig


class DataGenerator:
    """
    Generate synthetic datasets with known dimensionless relationships.
    
    This class implements the core data generation logic extracted from the GUI,
    making it usable in a config-based workflow.
    """
    
    def __init__(self, config: DataGenerationConfig):
        """
        Initialize the data generator with a configuration.
        
        Args:
            config: DataGenerationConfig object with all parameters
        """
        self.config = config
        
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        # Data storage (will be populated during generation)
        self.input_data: Optional[np.ndarray] = None
        self.dimension_matrix: Optional[np.ndarray] = None
        self.dimension_names: List[str] = ['Mass', 'Length', 'Time', 'Temperature', 'Current', 'Amount', 'Luminous']
        self.basis_vectors: Optional[np.ndarray] = None
        self.final_vectors: List[np.ndarray] = []
        self.gamma_vectors: List[np.ndarray] = []
        self.pi_values: List[np.ndarray] = []
        self.pi_expressions: List[str] = []
        self.output_values: Optional[np.ndarray] = None
        self.variable_names: List[str] = []
        
        # For backward compatibility
        self.final_vector: Optional[np.ndarray] = None
        self.pi1_values: Optional[np.ndarray] = None
        self.pi1_expression: str = ""
    
    def generate(self, max_trials: int = 10, verbose: bool = True) -> dict:
        """
        Generate synthetic data according to the configuration.
        
        Args:
            max_trials: Maximum number of trials to find valid vectors (components within [-5, 5])
            verbose: Whether to print progress messages
        
        Returns:
            Dictionary with generation results and metadata
        """
        if verbose:
            print(f"\n=== Generating Synthetic Data ===")
            print(f"Configuration:")
            print(f"  N (variables): {self.config.N}")
            print(f"  M (datapoints): {self.config.M}")
            print(f"  ndim (dimensionless groups): {self.config.ndim}")
            print(f"  poly_order: {self.config.poly_order}")
            print(f"  random_seed: {self.config.random_seed}")
            print(f"  noise_level: {self.config.noise_level}%")
            print(f"  n_discrete: {self.config.n_discrete}")
            print(f"  n_fix: {self.config.n_fix}")
        
        # Retry loop for w1 component constraint
        current_seed = self.config.random_seed
        w1_valid = False
        max_overall = None
        
        for trial in range(1, max_trials + 1):
            # Set seed and generate
            np.random.seed(current_seed)
            if verbose:
                print(f"\n--- Trial {trial}/{max_trials} (Seed: {current_seed}) ---")
            
            # Step 1: Generate input variables
            if verbose:
                print(f"Step 1: Generating {self.config.N} input variables with {self.config.M} datapoints...")
            self._generate_input_variables(verbose=verbose)
            
            # Step 2: Generate dimension matrix with rank=4
            if verbose:
                print(f"Step 2: Generating dimension matrix with rank=4...")
            self._generate_dimension_matrix(verbose=verbose)
            
            # Step 3: Get basis vectors
            if verbose:
                print(f"Step 3: Computing basis vectors...")
            self._compute_basis_vectors(verbose=verbose)
            
            # Step 4: Generate final vectors (w1, w2, ..., w_ndim)
            if verbose:
                print(f"Step 4: Generating {self.config.ndim} final vector(s)...")
            self._generate_final_vectors(verbose=verbose)
            
            # Check all final vector component ranges
            max_components = [np.max(np.abs(fv)) for fv in self.final_vectors]
            max_overall = max(max_components)
            
            if verbose:
                print(f"   max|wi| across all vectors = {max_overall:.3f}")
            
            if max_overall <= 5.0:
                if verbose:
                    print(f"   [OK] All vector components within [-5, 5]")
                w1_valid = True
                break
            else:
                if verbose:
                    print(f"   ✗ Some components outside [-5, 5], retrying with new seed...")
                current_seed += 1
        
        # Update config with final seed used
        self.config.random_seed = current_seed
        
        if not w1_valid and verbose:
            print(f"\n⚠️  Warning: After {max_trials} trials, could not generate vectors within [-5, 5]")
            print(f"   Proceeding with seed {current_seed}, max|wi| = {max_overall:.3f}")
        
        # Step 5: Calculate dimensionless variables
        if verbose:
            print(f"\nStep 5: Calculating {self.config.ndim} dimensionless variable(s)...")
        self._calculate_pi_values(verbose=verbose)
        
        # Step 6: Calculate output p*
        if verbose:
            print(f"\nStep 6: Calculating output p*...")
        self._calculate_output(verbose=verbose)
        
        # Prepare results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'random_seed_used': current_seed,
            'max_vector_component': max_overall,
            'trials_needed': trial if w1_valid else max_trials,
            'variable_names': self.variable_names,
            'dimension_matrix': self.dimension_matrix.tolist() if self.dimension_matrix is not None else None,
            'dimension_names': self.dimension_names,
            'basis_vectors': self.basis_vectors.tolist() if self.basis_vectors is not None else None,
            'gamma_vectors': [g.tolist() for g in self.gamma_vectors],
            'final_vectors': [v.tolist() for v in self.final_vectors],
            'pi_expressions': self.pi_expressions,
            'pi_ranges': {f'π{i+1}': [float(np.min(pi)), float(np.max(pi))] 
                         for i, pi in enumerate(self.pi_values)},
            'output_range': [float(np.min(self.output_values)), float(np.max(self.output_values))],
            'output_mean': float(np.mean(self.output_values))
        }
        
        if verbose:
            print(f"\n=== Generation Complete ===")
            print(f"Variables: {self.config.N}")
            print(f"Datapoints: {self.config.M}")
            print(f"Dimensionless groups: {self.config.ndim}")
            print(f"Matrix rank: 4")
            print(f"Basis vectors: {self.basis_vectors.shape[1] if self.basis_vectors is not None else 0}")
            print(f"max|wi|: {max_overall:.3f}")
        
        return results
    
    def _generate_input_variables(self, verbose: bool = True):
        """Step 1: Generate N input variables with M datapoints"""
        self.input_data = np.zeros((self.config.M, self.config.N))
        self.variable_names = [f"p{i+1}" for i in range(self.config.N)]
        
        # Full available range is [0.5, 1]
        full_range_min = 0.5
        full_range_max = 1.0
        
        # Randomly select which variables are discrete (not always the first ones)
        n_discrete = min(self.config.n_discrete, self.config.N)
        if n_discrete > 0:
            # Randomly choose n_discrete indices from [0, N-1]
            discrete_indices = np.random.choice(self.config.N, size=n_discrete, replace=False)
            discrete_indices_set = set(discrete_indices)
            if verbose:
                print(f"   Randomly selected discrete variables: {[f'p{i+1}' for i in sorted(discrete_indices)]}")
        else:
            discrete_indices_set = set()
        
        for i in range(self.config.N):
            if i in discrete_indices_set:
                # Discrete variable: randomly select n_fix values and repeat them
                fixed_values = np.random.uniform(full_range_min, full_range_max, self.config.n_fix)
                self.input_data[:, i] = np.random.choice(fixed_values, size=self.config.M, replace=True)
                if verbose:
                    print(f"      p{i+1}: discrete with {self.config.n_fix} fixed values: {sorted(fixed_values)}")
            else:
                # Continuous variable: uniform distribution
                self.input_data[:, i] = np.random.uniform(full_range_min, full_range_max, self.config.M)
        
        if verbose:
            print(f"   Generated {self.config.N} variables with {self.config.M} datapoints")
            if n_discrete > 0:
                print(f"   {n_discrete} discrete variables (randomly selected)")
                print(f"   {self.config.N - n_discrete} continuous variables")
            else:
                print(f"   All variables are continuous")
    
    def _generate_dimension_matrix(self, verbose: bool = True):
        """Step 2: Generate random dimension matrix with rank=4"""
        max_attempts = 1000
        possible_values = np.arange(-2, 3, 1)  # -2, -1, 0, 1, 2
        
        for attempt in range(max_attempts):
            # Initialize matrix: 7 dimensions × N variables
            dim_matrix = np.zeros((7, self.config.N))
            
            # First 4 dimensions (M, L, T, Temp): random values from -2 to 2
            for i in range(4):
                dim_matrix[i, :] = np.random.choice(possible_values, self.config.N)
            
            # Last 3 dimensions (Current, Amount, Luminous): all zeros (already zero)
            
            # Check rank
            rank = matrix_rank(dim_matrix)
            
            if rank == 4:
                self.dimension_matrix = dim_matrix
                if verbose:
                    print(f"   [OK] Generated dimension matrix with rank=4 (attempt {attempt+1})")
                    print(f"   Matrix shape: {dim_matrix.shape}")
                return
        
        raise ValueError(f"Failed to generate rank=4 matrix after {max_attempts} attempts")
    
    def _compute_basis_vectors(self, verbose: bool = True):
        """Step 3: Compute basis vectors using same method as dimensional_analysis_gui.py"""
        # Get null space
        null_sp = null_space(self.dimension_matrix)
        if verbose:
            print(f"   Null space shape: {null_sp.shape}")
        
        if null_sp.shape[1] == 0:
            raise ValueError("No null space found!")
        
        # Simplify to sparse, simple integer basis vectors
        self.basis_vectors = self._simplify_basis_vectors(null_sp)
        if verbose:
            print(f"   [OK] Computed {self.basis_vectors.shape[1]} basis vectors")
        
        # Normalize basis vectors to unit vectors (magnitude = 1)
        self.basis_vectors = self._normalize_to_unit_vectors(self.basis_vectors)
        if verbose:
            print(f"   [OK] Normalized basis vectors to unit vectors (||v|| = 1)")
        
        # Verify
        verification = self.dimension_matrix @ self.basis_vectors
        max_error = np.max(np.abs(verification))
        if verbose:
            print(f"   Verification error: {max_error:.2e}")
    
    def _simplify_basis_vectors(self, null_sp: np.ndarray) -> np.ndarray:
        """Create sparse basis vectors using SymPy exact rational arithmetic"""
        try:
            from sympy import Matrix, ilcm, igcd
        except ImportError:
            print("  Warning: SymPy not available, using original vectors")
            return null_sp
        
        # Convert numpy array to SymPy Matrix and get exact null space
        M = Matrix(self.dimension_matrix)
        nullspace_vectors = M.nullspace()
        
        if not nullspace_vectors:
            return null_sp
        
        # Convert to primitive integer vectors efficiently
        primitive_vectors = []
        for v in nullspace_vectors:
            # Clear denominators: multiply by LCM of denominators
            denominators = [x.as_numer_denom()[1] for x in v if x != 0]
            L = denominators[0] if denominators else 1
            for d in denominators[1:]:
                L = ilcm(L, d)
            
            w = (v * L)
            
            # Make primitive: divide by GCD of all elements
            elements = [abs(int(x)) for x in w if x != 0]
            g = elements[0] if elements else 1
            for elem in elements[1:]:
                g = igcd(g, elem)
            
            if g > 1:
                w = w // g
            
            # Normalize sign (make first non-zero element positive)
            for x in w:
                if x != 0:
                    if x < 0:
                        w = -w
                    break
            
            primitive_vectors.append(np.array([float(x) for x in w]))
        
        return np.column_stack(primitive_vectors)
    
    def _normalize_to_unit_vectors(self, basis_vectors: np.ndarray) -> np.ndarray:
        """Normalize each basis vector to unit vector (magnitude = 1)"""
        normalized_vectors = []
        
        for j in range(basis_vectors.shape[1]):
            vec = basis_vectors[:, j].copy()
            
            # Compute Euclidean norm (magnitude)
            norm = np.linalg.norm(vec)
            
            if norm > 1e-10:
                # Normalize to unit vector
                vec = vec / norm
            
            normalized_vectors.append(vec)
        
        result = np.column_stack(normalized_vectors)
        
        return result
    
    def _generate_final_vectors(self, verbose: bool = True):
        """Step 4: Generate final vectors (w1, w2, ..., w_ndim) by combining basis vectors"""
        n_basis = self.basis_vectors.shape[1]
        
        # Generate multiple final vectors for multi-dimensional case
        self.final_vectors = []
        self.gamma_vectors = []
        
        # Generate default gamma vectors if not provided
        if self.config.gamma_vectors is None:
            default_gammas = self._generate_default_gammas(n_basis, self.config.ndim)
        else:
            default_gammas = self.config.gamma_vectors
        
        for i in range(self.config.ndim):
            # Get gamma vector
            if i < len(default_gammas):
                gamma = np.array(default_gammas[i], dtype=float)
            else:
                # Fallback: use default pattern
                gamma = np.zeros(n_basis)
                if i < n_basis:
                    gamma[i] = 1
                else:
                    gamma = np.ones(n_basis)
            
            # Ensure correct length
            if len(gamma) != n_basis:
                if verbose:
                    print(f"   Warning: Gamma{i+1} length mismatch. Expected {n_basis}, got {len(gamma)}. Padding/truncating.")
                if len(gamma) < n_basis:
                    gamma = np.pad(gamma, (0, n_basis - len(gamma)), constant_values=0)
                else:
                    gamma = gamma[:n_basis]
            
            # w_i = basis_vectors * gamma_i
            final_vector = self.basis_vectors @ gamma
            self.final_vectors.append(final_vector)
            self.gamma_vectors.append(gamma)
            
            if verbose:
                print(f"   [OK] Generated final vector w{i+1}")
                print(f"     Gamma{i+1}: {gamma}")
                print(f"     w{i+1}: {final_vector}")
        
        # Set backward compatible variables
        self.final_vector = self.final_vectors[0] if self.final_vectors else None
    
    def _generate_default_gammas(self, gamma_dim: int, ndim: int) -> List[List[float]]:
        """Generate default linearly independent gamma vectors"""
        default_gammas = []
        
        # Create linearly independent vectors
        for i in range(min(ndim, 3)):
            gamma = []
            for j in range(gamma_dim):
                if j == i % gamma_dim:
                    gamma.append(1)
                elif j == (i + 1) % gamma_dim:
                    gamma.append(1)
                elif i == 1 and j == (i + 2) % gamma_dim:
                    gamma.append(1)
                else:
                    gamma.append(0)
            
            # For gamma2, gamma3, add some variety
            if i == 1 and gamma_dim > 1:
                gamma[0] = 2
                gamma[1] = 0
            elif i == 2 and gamma_dim > 2:
                gamma[0] = 0
                gamma[1] = 2
            
            default_gammas.append(gamma)
        
        return default_gammas
    
    def _calculate_pi_values(self, verbose: bool = True):
        """Step 5: Calculate dimensionless variables (π1, π2, ..., π_ndim) using element-wise power law"""
        self.pi_values = []
        self.pi_expressions = []
        
        for i in range(self.config.ndim):
            final_vector = self.final_vectors[i]
            
            # Calculate πi using log to avoid overflow
            log_pi = np.zeros(self.config.M)
            for j in range(self.config.N):
                if abs(final_vector[j]) > 1e-10:
                    data_col = np.maximum(self.input_data[:, j], 1e-10)
                    log_pi += final_vector[j] * np.log(data_col)
            
            pi_values = np.exp(log_pi)
            self.pi_values.append(pi_values)
            
            # Generate expression string
            expr_parts = []
            for j in range(self.config.N):
                w = final_vector[j]
                if abs(w) > 1e-10:
                    var = self.variable_names[j]
                    if abs(w - 1.0) < 1e-10:
                        expr_parts.append(f"{var}")
                    elif abs(w + 1.0) < 1e-10:
                        expr_parts.append(f"{var}^(-1)")
                    elif abs(w - round(w)) < 1e-10:
                        expr_parts.append(f"{var}^{int(round(w))}")
                    else:
                        expr_parts.append(f"{var}^{w:.3f}")
            
            pi_expression = " × ".join(expr_parts) if expr_parts else "1"
            self.pi_expressions.append(pi_expression)
            
            if verbose:
                print(f"   [OK] Calculated π{i+1}")
                print(f"     Expression: π{i+1} = {pi_expression}")
                print(f"     Range: [{np.min(pi_values):.6f}, {np.max(pi_values):.6f}]")
        
        # Set backward compatible variables
        self.pi1_values = self.pi_values[0] if self.pi_values else None
        self.pi1_expression = self.pi_expressions[0] if self.pi_expressions else ""
    
    def _calculate_output(self, verbose: bool = True):
        """Step 6: Calculate output p* = polynomial(π1) or nonlinear(π1,π2,...) + noise"""
        if self.config.ndim == 1:
            # Single dimensionless variable: p* = c0 + c1*π1 + c2*π1^2 + ... + cn*π1^n
            self.output_values = np.zeros(self.config.M)
            term_values = []
            term_names = []
            
            for i, coeff in enumerate(self.config.coefficients):
                term = coeff * (self.pi1_values ** i)
                self.output_values += term
                term_values.append(term)
                coeff_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
                if i == 0:
                    term_names.append(f"{coeff_names[i] if i < len(coeff_names) else 'C'} (intercept)")
                else:
                    term_names.append(f"{coeff_names[i] if i < len(coeff_names) else 'C'}×π1^{i}")
        else:
            # Multiple dimensionless variables: nonlinear functions
            # p* = exp(A×π1) + π2^B + log(1+C×π3) (for ndim up to 3)
            self.output_values = np.zeros(self.config.M)
            term_values = []
            term_names = []
            
            # exp(A×π1) - always present for ndim >= 2
            if len(self.config.coefficients) > 0:
                A = self.config.coefficients[0]
                exp_term = np.clip(A * self.pi_values[0], -10, 10)
                term = np.exp(exp_term)
                self.output_values += term
                term_values.append(term)
                term_names.append(f"exp(A×π1) with A={A:.3g}")
            
            # π2^B if ndim >= 2
            if self.config.ndim >= 2 and len(self.config.coefficients) > 1:
                B = self.config.coefficients[1]
                B_clipped = np.clip(B, -10, 10)
                term = np.power(np.abs(self.pi_values[1]) + 1e-10, B_clipped)
                self.output_values += term
                term_values.append(term)
                term_names.append(f"π2^B with B={B:.3g}")
            
            # log(1+C×π3) if ndim >= 3
            if self.config.ndim >= 3 and len(self.config.coefficients) > 2:
                C = self.config.coefficients[2]
                log_arg = 1.0 + C * self.pi_values[2]
                log_arg = np.maximum(log_arg, 1e-10)
                term = np.log(log_arg)
                self.output_values += term
                term_values.append(term)
                term_names.append(f"log(1+C×π3) with C={C:.3g}")
        
        # Build expression string for printing
        if self.config.ndim == 1:
            coeff_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
            expr_parts = []
            for i, coeff in enumerate(self.config.coefficients):
                if i == 0:
                    expr_parts.append(f"{coeff:.3g}")
                elif i == 1:
                    expr_parts.append(f"{coeff:.3g}×π1")
                else:
                    expr_parts.append(f"{coeff:.3g}×π1^{i}")
            expr = " + ".join(expr_parts)
        else:
            # Nonlinear formula: exp(A×π1) + π2^B + log(1+C×π3) (for ndim up to 3)
            expr = f"exp({self.config.coefficients[0]:.3g}×π1)"
            if self.config.ndim >= 2:
                expr += f" + π2^{self.config.coefficients[1]:.3g}"
            if self.config.ndim >= 3:
                expr += f" + log(1+{self.config.coefficients[2]:.3g}×π3)"
        
        # Add noise if noise_level > 0
        if self.config.noise_level > 0:
            output_range = np.max(self.output_values) - np.min(self.output_values)
            noise_std = (self.config.noise_level / 100.0) * output_range
            noise = np.random.normal(0, noise_std, self.config.M)
            self.output_values = self.output_values + noise
            if verbose:
                print(f"   [OK] Calculated output p* with {self.config.noise_level}% noise")
                print(f"   p* = {expr} + noise")
                print(f"   Noise std dev: {noise_std:.6f}")
        else:
            if verbose:
                print(f"   [OK] Calculated output p* (no noise)")
                print(f"   p* = {expr}")
        
        if verbose:
            print(f"   p* range: [{np.min(self.output_values):.6f}, {np.max(self.output_values):.6f}]")
            print(f"   p* mean: {np.mean(self.output_values):.6f}")
    
    def _calculate_predicted_output(self) -> np.ndarray:
        """Calculate predicted output from formula (for validation)"""
        if self.config.ndim == 1:
            # Single dimensionless variable: p* = c0 + c1*π1 + c2*π1^2 + ...
            predicted = np.zeros(self.config.M)
            for i, coeff in enumerate(self.config.coefficients):
                predicted += coeff * (self.pi1_values ** i)
        else:
            # Multiple dimensionless variables: nonlinear functions
            predicted = np.zeros(self.config.M)
            
            # exp(A×π1)
            if len(self.config.coefficients) > 0:
                A = self.config.coefficients[0]
                exp_term = np.clip(A * self.pi_values[0], -10, 10)
                predicted += np.exp(exp_term)
            
            # π2^B
            if self.config.ndim >= 2 and len(self.config.coefficients) > 1:
                B = self.config.coefficients[1]
                B_clipped = np.clip(B, -10, 10)
                predicted += np.power(np.abs(self.pi_values[1]) + 1e-10, B_clipped)
            
            # log(1+C×π3)
            if self.config.ndim >= 3 and len(self.config.coefficients) > 2:
                C = self.config.coefficients[2]
                log_arg = 1.0 + C * self.pi_values[2]
                log_arg = np.maximum(log_arg, 1e-10)
                predicted += np.log(log_arg)
        
        return predicted
    
    def create_visualization(self, output_dir: Optional[str] = None, 
                            filename: Optional[str] = None,
                            show: bool = False) -> str:
        """
        Create visualization plot for the generated data.
        
        Creates a single plot showing p* vs π1 (first dimensionless variable).
        
        Args:
            output_dir: Base directory to save plots (defaults to config.output_dir)
            filename: Filename for saved plot (default: 'data_generation_plots.png')
            show: Whether to display plots (default: False, saves to file)
        
        Returns:
            Path to saved plot file
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        # Create figures subdirectory
        figures_dir = os.path.join(output_dir, self.config.figures_dir)
        os.makedirs(figures_dir, exist_ok=True)
        
        if filename is None:
            filename = 'data_generation_plots.png'
        
        # Close any existing figures
        plt.close('all')
        
        # Create a single figure for π1 vs p*
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        # Get π1 values (first dimensionless variable)
        pi1_values = self.pi_values[0] if len(self.pi_values) > 0 else None
        
        if pi1_values is None:
            raise ValueError("No dimensionless variables available for plotting")
        
        # Scatter plot
        ax.scatter(pi1_values, self.output_values, alpha=0.6, s=30, 
                  c='blue', edgecolors='navy', label='Generated data')
        
        # Theoretical line
        pi_sorted = np.sort(pi1_values)
        p_star_theory = np.zeros(len(pi_sorted))
        for j, coeff in enumerate(self.config.coefficients):
            p_star_theory += coeff * (pi_sorted ** j)
        
        # Create theory label based on polynomial order
        if self.config.poly_order == 1:
            theory_label = f'Theory: p* = {self.config.coefficients[0]:.3g} + {self.config.coefficients[1]:.3g}×π1'
        elif self.config.poly_order == 2:
            theory_label = f'Theory: p* = {self.config.coefficients[0]:.3g} + {self.config.coefficients[1]:.3g}×π1 + {self.config.coefficients[2]:.3g}×π1²'
        elif self.config.poly_order == 3:
            theory_label = f'Theory: p* = {self.config.coefficients[0]:.3g} + {self.config.coefficients[1]:.3g}×π1 + {self.config.coefficients[2]:.3g}×π1² + {self.config.coefficients[3]:.3g}×π1³'
        else:
            theory_label = f'Theory: p* = polynomial(π1) [order {self.config.poly_order}]'
        
        ax.plot(pi_sorted, p_star_theory, 'r-', linewidth=2, label=theory_label)
        
        # Use plain-text "pi1" to avoid mathtext parsing issues
        ax.set_xlabel('First Dimensionless Group (pi1)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Dimensionless Output (p*)', fontsize=11, fontweight='bold')
        ax.set_title('Dimensionless Output vs First Dimensionless Group', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Save figure to figures subdirectory
        plot_path = os.path.join(figures_dir, filename)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close('all')
        
        return plot_path
    
    def save_datasets(self, output_dir: Optional[str] = None) -> Tuple[str, str]:
        """
        Save generated datasets to CSV files.
        
        Args:
            output_dir: Base directory to save files (defaults to config.output_dir)
        
        Returns:
            Tuple of (dataset_path, dimension_matrix_path)
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        # Create data subdirectory
        data_dir = os.path.join(output_dir, self.config.data_dir)
        os.makedirs(data_dir, exist_ok=True)
        
        # Build column list: [p1, p2, ..., pN, π1, π2, ..., π_ndim, p*]
        pi_columns = [f'π{i+1}' for i in range(self.config.ndim)]
        all_columns = self.variable_names + pi_columns + ['p*']
        
        # Stack data: inputs + all π values + output
        pi_data = np.column_stack(self.pi_values) if len(self.pi_values) > 0 else self.pi1_values.reshape(-1, 1)
        full_data = np.column_stack([self.input_data, pi_data, self.output_values])
        
        dataset_df = pd.DataFrame(full_data, columns=all_columns)
        dataset_path = os.path.join(data_dir, self.config.dataset_filename)
        dataset_df.to_csv(dataset_path, index=False, float_format='%.10f')
        
        # Save dimension matrix
        dim_df = pd.DataFrame(self.dimension_matrix, columns=self.variable_names)
        dim_df.insert(0, 'Dimension', self.dimension_names)
        dim_matrix_path = os.path.join(data_dir, self.config.dimension_matrix_filename)
        dim_df.to_csv(dim_matrix_path, index=False)
        
        return dataset_path, dim_matrix_path

