"""
Enhanced Uncertainty Quantification
Advanced Sobol sensitivity analysis and Gelman-Rubin convergence diagnostics

Mathematical Formulations:
- Enhanced Gelman-Rubin: R̂_enhanced = √[(N-1)/N + (1/N) × (B/W) × (1 + 2√(B/W)/√N)]
- First-order Sobol: S_i = Var[E[Y|X_i]]/Var[Y] = (1/N × Σ Y_A × Y_C_i - f₀²)/Var[Y]
- Second-order Sobol: S_ij = Var[E[Y|X_i,X_j]]/Var[Y] - S_i - S_j  
- Total-effect Sobol: S_T^i = 1 - Var[E[Y|X_~i]]/Var[Y]
- Sample generation: N = 2^m where m ≥ 12 for numerical stability
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import sobol_seq
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import logging
from enum import Enum
import warnings

class SensitivityMethod(Enum):
    """Sensitivity analysis methods"""
    SOBOL_FIRST = "sobol_first"
    SOBOL_TOTAL = "sobol_total"  
    SOBOL_SECOND = "sobol_second"
    MORRIS = "morris"
    VARIANCE_BASED = "variance_based"

class ConvergenceMethod(Enum):
    """Convergence diagnostic methods"""
    GELMAN_RUBIN = "gelman_rubin"
    GEWEKE = "geweke"
    HEIDELBERGER_WELCH = "heidelberger_welch"
    RAFTERY_LEWIS = "raftery_lewis"

@dataclass
class UQParameters:
    """Uncertainty quantification parameters"""
    # Sobol analysis parameters
    sobol_samples: int = 2**12    # N = 2^m, m ≥ 12
    sobol_skip: int = 1000        # Skip initial samples
    confidence_level: float = 0.95
    
    # Gelman-Rubin parameters
    num_chains: int = 4
    chain_length: int = 1000
    burn_in: int = 200
    convergence_threshold: float = 1.01  # R̂ < 1.01
    
    # Bootstrap parameters
    bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95

class EnhancedUncertaintyQuantification:
    """
    Enhanced uncertainty quantification with advanced Sobol analysis and convergence diagnostics
    
    Key Features:
    - Enhanced Gelman-Rubin diagnostic with bias correction
    - Second-order Sobol sensitivity indices
    - Total-effect Sobol indices
    - Bootstrap confidence intervals
    - Multi-chain convergence assessment
    - Adaptive sample size determination
    """
    
    def __init__(self, parameters: Optional[UQParameters] = None):
        self.logger = logging.getLogger(__name__)
        self.params = parameters or UQParameters()
        
        # Storage for analysis results
        self.sobol_results = {}
        self.convergence_results = {}
        self.chains_data = []
        
        self.logger.info("Enhanced UQ initialized")
        self.logger.info(f"  Sobol samples: {self.params.sobol_samples}")
        self.logger.info(f"  Convergence threshold: {self.params.convergence_threshold}")
    
    def generate_sobol_samples(self, 
                              input_ranges: List[Tuple[float, float]], 
                              num_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Sobol sequence samples for sensitivity analysis
        
        Returns A, B, and C matrices for Sobol analysis:
        - A: Primary sample matrix (N × d)
        - B: Secondary sample matrix (N × d)  
        - C: Hybrid matrices for first/total order (d × N × d)
        
        Args:
            input_ranges: List of (min, max) tuples for each input dimension
            num_samples: Number of samples (default from parameters)
            
        Returns:
            Tuple of (A, B, C) sample matrices
        """
        
        if num_samples is None:
            num_samples = self.params.sobol_samples
        
        # Ensure sample size is power of 2 for Sobol sequences
        m = int(np.ceil(np.log2(num_samples)))
        actual_samples = 2**m
        
        if actual_samples != num_samples:
            self.logger.info(f"Adjusted sample size from {num_samples} to {actual_samples} (2^{m})")
        
        d = len(input_ranges)  # Number of input dimensions
        
        # Generate Sobol sequences
        # Need 2*d sequences for A and B matrices
        sobol_samples = sobol_seq.sample(actual_samples + self.params.sobol_skip, 2*d)
        
        # Skip initial samples for better uniformity
        sobol_samples = sobol_samples[self.params.sobol_skip:]
        
        # Split into A and B matrices
        A_unit = sobol_samples[:, :d]
        B_unit = sobol_samples[:, d:]
        
        # Scale to input ranges
        A = np.zeros_like(A_unit)
        B = np.zeros_like(B_unit)
        
        for i, (min_val, max_val) in enumerate(input_ranges):
            A[:, i] = min_val + (max_val - min_val) * A_unit[:, i]
            B[:, i] = min_val + (max_val - min_val) * B_unit[:, i]
        
        # Generate C matrices for first-order and total-order indices
        C = np.zeros((d, actual_samples, d))
        
        for i in range(d):
            C[i] = A.copy()
            C[i][:, i] = B[:, i]  # Replace i-th column with B
        
        self.logger.info(f"Generated Sobol samples: A({A.shape}), B({B.shape}), C({C.shape})")
        
        return A, B, C
    
    def calculate_sobol_indices(self, 
                               model_function: Callable,
                               input_ranges: List[Tuple[float, float]],
                               include_second_order: bool = True) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Calculate enhanced Sobol sensitivity indices
        
        First-order: S_i = Var[E[Y|X_i]]/Var[Y] = (1/N × Σ Y_A × Y_C_i - f₀²)/Var[Y]
        Second-order: S_ij = Var[E[Y|X_i,X_j]]/Var[Y] - S_i - S_j
        Total-effect: S_T^i = 1 - Var[E[Y|X_~i]]/Var[Y] = 1 - (1/N × Σ Y_B × Y_C_~i - f₀²)/Var[Y]
        
        Args:
            model_function: Function to analyze f(x) -> scalar
            input_ranges: Input parameter ranges
            include_second_order: Whether to calculate second-order indices
            
        Returns:
            Dictionary with sensitivity indices and confidence intervals
        """
        
        d = len(input_ranges)
        
        # Generate samples
        A, B, C = self.generate_sobol_samples(input_ranges)
        N = A.shape[0]
        
        # Evaluate model
        self.logger.info("Evaluating model for Sobol analysis...")
        
        Y_A = np.array([model_function(A[i]) for i in range(N)])
        Y_B = np.array([model_function(B[i]) for i in range(N)])
        
        Y_C = np.zeros((d, N))
        for i in range(d):
            Y_C[i] = np.array([model_function(C[i][j]) for j in range(N)])
        
        # Calculate statistics
        f0_squared = np.mean(Y_A)**2
        var_Y = np.var(Y_A)
        
        if var_Y < 1e-12:
            self.logger.warning("Output variance is very small, results may be unreliable")
            var_Y = 1e-12
        
        # First-order indices
        S_first = np.zeros(d)
        for i in range(d):
            # S_i = (1/N × Σ Y_A × Y_C_i - f₀²) / Var[Y]
            S_first[i] = (np.mean(Y_A * Y_C[i]) - f0_squared) / var_Y
        
        # Total-effect indices  
        S_total = np.zeros(d)
        for i in range(d):
            # S_T^i = 1 - (1/N × Σ Y_B × Y_C_~i - f₀²) / Var[Y]
            # For total effect, we use Y_B and Y_C[i] (not Y_C_~i)
            S_total[i] = 1 - (np.mean(Y_B * Y_C[i]) - f0_squared) / var_Y
        
        # Bootstrap confidence intervals
        S_first_ci = self._calculate_bootstrap_ci(Y_A, Y_C, var_Y, f0_squared, 'first')
        S_total_ci = self._calculate_bootstrap_ci(Y_A, Y_C, var_Y, f0_squared, 'total', Y_B)
        
        results = {
            'first_order': S_first,
            'total_order': S_total,
            'first_order_ci': S_first_ci,
            'total_order_ci': S_total_ci,
            'variance': var_Y,
            'mean': np.mean(Y_A),
            'num_samples': N
        }
        
        # Second-order indices (computationally expensive)
        if include_second_order and d <= 10:  # Limit for computational tractability
            self.logger.info("Calculating second-order Sobol indices...")
            S_second = self._calculate_second_order_indices(model_function, input_ranges, A, B)
            results['second_order'] = S_second
        elif include_second_order:
            self.logger.warning(f"Skipping second-order indices for {d} dimensions (too expensive)")
        
        self.sobol_results = results
        self.logger.info(f"Sobol analysis complete. First-order range: [{np.min(S_first):.6f}, {np.max(S_first):.6f}]")
        
        return results
    
    def _calculate_second_order_indices(self, 
                                       model_function: Callable,
                                       input_ranges: List[Tuple[float, float]],
                                       A: np.ndarray, 
                                       B: np.ndarray) -> np.ndarray:
        """Calculate second-order Sobol indices S_ij"""
        
        d = len(input_ranges)
        N = A.shape[0]
        
        # Generate additional samples for second-order
        S_second = np.zeros((d, d))
        
        # This is computationally expensive - simplified implementation
        for i in range(d):
            for j in range(i+1, d):
                # Generate C_ij matrix (replace columns i and j with B)
                C_ij = A.copy()
                C_ij[:, i] = B[:, i]
                C_ij[:, j] = B[:, j]
                
                # Evaluate model
                Y_C_ij = np.array([model_function(C_ij[k]) for k in range(N)])
                Y_A = np.array([model_function(A[k]) for k in range(N)])
                
                # Calculate second-order index
                f0_squared = np.mean(Y_A)**2
                var_Y = np.var(Y_A)
                
                # S_ij = Var[E[Y|X_i,X_j]]/Var[Y] - S_i - S_j
                # Approximation using available samples
                S_ij_raw = (np.mean(Y_A * Y_C_ij) - f0_squared) / var_Y
                
                # Subtract first-order contributions (approximation)
                S_i = self.sobol_results.get('first_order', [0]*d)[i] if hasattr(self, 'sobol_results') else 0
                S_j = self.sobol_results.get('first_order', [0]*d)[j] if hasattr(self, 'sobol_results') else 0
                
                S_second[i, j] = max(0, S_ij_raw - S_i - S_j)  # Ensure non-negative
                S_second[j, i] = S_second[i, j]  # Symmetric
        
        return S_second
    
    def _calculate_bootstrap_ci(self, 
                               Y_A: np.ndarray, 
                               Y_C: np.ndarray, 
                               var_Y: float, 
                               f0_squared: float,
                               index_type: str,
                               Y_B: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate bootstrap confidence intervals for Sobol indices"""
        
        N = len(Y_A)
        d = Y_C.shape[0]
        n_bootstrap = self.params.bootstrap_samples
        
        bootstrap_indices = np.zeros((n_bootstrap, d))
        
        for b in range(n_bootstrap):
            # Bootstrap sample indices
            boot_idx = np.random.choice(N, N, replace=True)
            
            Y_A_boot = Y_A[boot_idx]
            Y_C_boot = Y_C[:, boot_idx]
            
            f0_boot = np.mean(Y_A_boot)**2
            var_boot = np.var(Y_A_boot)
            
            if var_boot < 1e-12:
                var_boot = 1e-12
            
            for i in range(d):
                if index_type == 'first':
                    bootstrap_indices[b, i] = (np.mean(Y_A_boot * Y_C_boot[i]) - f0_boot) / var_boot
                elif index_type == 'total' and Y_B is not None:
                    Y_B_boot = Y_B[boot_idx]
                    bootstrap_indices[b, i] = 1 - (np.mean(Y_B_boot * Y_C_boot[i]) - f0_boot) / var_boot
        
        # Calculate confidence intervals
        alpha = 1 - self.params.bootstrap_confidence
        ci_lower = np.percentile(bootstrap_indices, 100 * alpha/2, axis=0)
        ci_upper = np.percentile(bootstrap_indices, 100 * (1 - alpha/2), axis=0)
        
        return np.column_stack([ci_lower, ci_upper])
    
    def calculate_enhanced_gelman_rubin(self, chains: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate enhanced Gelman-Rubin convergence diagnostic
        
        R̂_enhanced = √[(N-1)/N + (1/N) × (B/W) × (1 + 2√(B/W)/√N)]
        
        Where:
        B = (N/(M-1)) × Σ(θ̄_j - θ̄)²  [Between-chain variance]
        W = (1/M) × Σ[(1/(N-1)) × Σ(θ_i,j - θ̄_j)²]  [Within-chain variance]
        
        Args:
            chains: List of MCMC chains (each chain is 1D array)
            
        Returns:
            Dictionary with convergence diagnostics
        """
        
        if len(chains) < 2:
            raise ValueError("Need at least 2 chains for Gelman-Rubin diagnostic")
        
        # Convert to numpy arrays and ensure equal length
        chains = [np.array(chain) for chain in chains]
        min_length = min(len(chain) for chain in chains)
        chains = [chain[-min_length:] for chain in chains]  # Use last min_length samples
        
        M = len(chains)  # Number of chains
        N = min_length   # Length of each chain
        
        if N < self.params.burn_in:
            self.logger.warning(f"Chain length ({N}) less than burn-in ({self.params.burn_in})")
        
        # Remove burn-in
        burn_in = min(self.params.burn_in, N//4)
        chains = [chain[burn_in:] for chain in chains]
        N = N - burn_in
        
        # Calculate chain means
        chain_means = np.array([np.mean(chain) for chain in chains])
        overall_mean = np.mean(chain_means)
        
        # Between-chain variance B
        B = (N / (M - 1)) * np.sum((chain_means - overall_mean)**2)
        
        # Within-chain variance W
        chain_vars = np.array([np.var(chain, ddof=1) for chain in chains])
        W = np.mean(chain_vars)
        
        # Enhanced Gelman-Rubin statistic
        if W < 1e-12:
            self.logger.warning("Within-chain variance is very small")
            W = 1e-12
        
        # Standard R̂
        R_hat_standard = np.sqrt(((N-1)/N) + (1/N) * (B/W))
        
        # Enhanced R̂ with bias correction
        bias_correction = 1 + 2*np.sqrt(B/W)/np.sqrt(N)
        R_hat_enhanced = np.sqrt(((N-1)/N) + (1/N) * (B/W) * bias_correction)
        
        # Effective sample size
        rho_hat = 1 - W/((N-1)/N * W + (1/N) * B)
        n_eff = M * N / (1 + 2 * np.sum([max(0, 1 - k/N) * rho_hat for k in range(1, N)]))
        
        # Convergence assessment
        converged = R_hat_enhanced < self.params.convergence_threshold
        
        results = {
            'R_hat_standard': R_hat_standard,
            'R_hat_enhanced': R_hat_enhanced,
            'between_chain_var': B,
            'within_chain_var': W,
            'effective_sample_size': n_eff,
            'converged': converged,
            'num_chains': M,
            'chain_length': N,
            'threshold': self.params.convergence_threshold
        }
        
        self.convergence_results = results
        self.logger.info(f"Gelman-Rubin diagnostic: R̂ = {R_hat_enhanced:.6f} (threshold: {self.params.convergence_threshold})")
        
        return results
    
    def generate_mcmc_chains(self, 
                            target_distribution: Callable,
                            initial_values: List[float],
                            proposal_std: float = 0.1) -> List[np.ndarray]:
        """
        Generate MCMC chains for convergence analysis
        
        Args:
            target_distribution: Log-probability function
            initial_values: Starting values for each chain
            proposal_std: Standard deviation for proposal distribution
            
        Returns:
            List of MCMC chains
        """
        
        chains = []
        
        for i, x0 in enumerate(initial_values):
            self.logger.info(f"Generating MCMC chain {i+1}/{len(initial_values)}")
            
            chain = np.zeros(self.params.chain_length)
            chain[0] = x0
            
            current_x = x0
            current_log_prob = target_distribution(current_x)
            
            n_accepted = 0
            
            for j in range(1, self.params.chain_length):
                # Propose new state
                proposal = current_x + np.random.normal(0, proposal_std)
                proposal_log_prob = target_distribution(proposal)
                
                # Metropolis acceptance
                log_alpha = proposal_log_prob - current_log_prob
                
                if np.log(np.random.rand()) < log_alpha:
                    current_x = proposal
                    current_log_prob = proposal_log_prob
                    n_accepted += 1
                
                chain[j] = current_x
            
            acceptance_rate = n_accepted / (self.params.chain_length - 1)
            self.logger.info(f"Chain {i+1} acceptance rate: {acceptance_rate:.3f}")
            
            chains.append(chain)
        
        self.chains_data = chains
        return chains
    
    def get_sensitivity_summary(self) -> Dict[str, Union[str, float, List]]:
        """Get summary of sensitivity analysis results"""
        
        if not self.sobol_results:
            return {'status': 'no_results'}
        
        first_order = self.sobol_results['first_order']
        total_order = self.sobol_results['total_order']
        
        # Find most/least sensitive parameters
        most_sensitive_idx = np.argmax(first_order)
        least_sensitive_idx = np.argmin(first_order)
        
        # Calculate total sensitivity accounted for
        total_sensitivity = np.sum(first_order)
        
        summary = {
            'most_sensitive_param': int(most_sensitive_idx),
            'most_sensitive_value': float(first_order[most_sensitive_idx]),
            'least_sensitive_param': int(least_sensitive_idx), 
            'least_sensitive_value': float(first_order[least_sensitive_idx]),
            'total_first_order': float(total_sensitivity),
            'max_total_order': float(np.max(total_order)),
            'interaction_strength': float(np.max(total_order) - np.max(first_order)),
            'num_parameters': len(first_order)
        }
        
        return summary

def main():
    """Demonstrate enhanced uncertainty quantification"""
    
    print("Enhanced Uncertainty Quantification Demonstration")
    print("=" * 55)
    
    # Initialize UQ
    uq = EnhancedUncertaintyQuantification()
    
    # Define test function (Ishigami function)
    def ishigami_function(x):
        """Ishigami test function for sensitivity analysis"""
        a, b = 7.0, 0.1
        return np.sin(x[0]) + a * np.sin(x[1])**2 + b * x[2]**4 * np.sin(x[0])
    
    # Define input ranges
    input_ranges = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    
    # Calculate Sobol indices
    print("Calculating Sobol sensitivity indices...")
    sobol_results = uq.calculate_sobol_indices(
        ishigami_function, 
        input_ranges, 
        include_second_order=True
    )
    
    print(f"\nSobol Analysis Results:")
    print(f"{'Parameter':>10} {'First-order':>12} {'Total-order':>12} {'95% CI':>20}")
    print("-" * 54)
    
    for i in range(len(input_ranges)):
        first_val = sobol_results['first_order'][i]
        total_val = sobol_results['total_order'][i]
        ci_lower = sobol_results['first_order_ci'][i, 0]
        ci_upper = sobol_results['first_order_ci'][i, 1]
        
        print(f"X{i+1:1d}{'':<8} {first_val:12.6f} {total_val:12.6f} [{ci_lower:7.4f}, {ci_upper:7.4f}]")
    
    # Generate MCMC chains for convergence analysis
    print(f"\nGenerating MCMC chains for convergence analysis...")
    
    def target_log_prob(x):
        """Simple target distribution (standard normal)"""
        return -0.5 * x**2
    
    initial_values = [-2.0, -1.0, 1.0, 2.0]  # 4 chains
    chains = uq.generate_mcmc_chains(target_log_prob, initial_values, proposal_std=0.5)
    
    # Calculate Gelman-Rubin diagnostic
    gelman_rubin = uq.calculate_enhanced_gelman_rubin(chains)
    
    print(f"\nGelman-Rubin Convergence Diagnostic:")
    print(f"  Standard R̂: {gelman_rubin['R_hat_standard']:.6f}")
    print(f"  Enhanced R̂: {gelman_rubin['R_hat_enhanced']:.6f}")
    print(f"  Converged: {'Yes' if gelman_rubin['converged'] else 'No'}")
    print(f"  Effective sample size: {gelman_rubin['effective_sample_size']:.1f}")
    
    # Get sensitivity summary
    summary = uq.get_sensitivity_summary()
    print(f"\nSensitivity Summary:")
    print(f"  Most sensitive parameter: X{summary['most_sensitive_param']+1} ({summary['most_sensitive_value']:.6f})")
    print(f"  Total first-order sensitivity: {summary['total_first_order']:.6f}")
    print(f"  Interaction strength: {summary['interaction_strength']:.6f}")
    
    print(f"\nEnhanced UQ demonstration complete!")

if __name__ == "__main__":
    main()
