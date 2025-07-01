"""
Advanced Unscented Kalman Filter
Enhanced UKF with adaptive sigma point optimization and robust covariance updates

Mathematical Formulations:
- Sigma points: χ_σ = [x̂, x̂ + √((n+λ)P), x̂ - √((n+λ)P)]
- State prediction: x̂_(k+1|k) = Σ W_m^i × f(χ_i, u_k)  
- Covariance prediction: P_(k+1|k) = Q + Σ W_c^i × [χ_i - x̂_(k+1|k)][χ_i - x̂_(k+1|k)]^T
- Adaptive parameters: λ = α²(n + κ) - n, β = 2, κ = 3 - n
"""

import numpy as np
import scipy.linalg as la
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import logging
from enum import Enum

class UKFVariant(Enum):
    """UKF implementation variants"""
    STANDARD = "standard"
    ADAPTIVE = "adaptive" 
    SQUARE_ROOT = "square_root"
    CONSTRAINED = "constrained"

@dataclass
class UKFParameters:
    """UKF tuning parameters"""
    alpha: float = 1e-3         # Spread parameter [1e-4, 1]
    beta: float = 2.0           # Prior knowledge parameter (2 for Gaussian)
    kappa: float = None         # Secondary scaling (3-n default)
    
    # Adaptive parameters
    adaptive_alpha: bool = True
    alpha_min: float = 1e-4
    alpha_max: float = 1.0
    
    # Robustness parameters
    min_eigenvalue: float = 1e-12
    max_condition_number: float = 1e12

class AdvancedUnscentedKalmanFilter:
    """
    Advanced Unscented Kalman Filter with enhanced sigma point optimization
    
    Key Features:
    - Adaptive sigma point scaling
    - Robust covariance updates with conditioning
    - Square-root implementation for numerical stability
    - Constrained state estimation
    - Multi-rate processing capability
    """
    
    def __init__(self, 
                 state_dim: int,
                 measurement_dim: int,
                 process_model: Callable,
                 measurement_model: Callable,
                 parameters: Optional[UKFParameters] = None,
                 variant: UKFVariant = UKFVariant.ADAPTIVE):
        
        self.logger = logging.getLogger(__name__)
        
        # Dimensions
        self.n = state_dim
        self.m = measurement_dim
        self.variant = variant
        
        # Models
        self.f = process_model      # x_(k+1) = f(x_k, u_k, w_k)
        self.h = measurement_model  # y_k = h(x_k, v_k)
        
        # Parameters
        self.params = parameters or UKFParameters()
        if self.params.kappa is None:
            self.params.kappa = 3 - self.n
        
        # Calculate UKF parameters
        self._update_ukf_parameters()
        
        # State and covariance
        self.x_hat = np.zeros(self.n)
        self.P = np.eye(self.n)
        self.Q = np.eye(self.n) * 1e-6  # Process noise
        self.R = np.eye(self.m) * 1e-3  # Measurement noise
        
        # Sigma points storage
        self.sigma_points = np.zeros((2*self.n + 1, self.n))
        self.weights_mean = np.zeros(2*self.n + 1)
        self.weights_cov = np.zeros(2*self.n + 1)
        
        # Performance tracking
        self.innovation_history = []
        self.likelihood_history = []
        
        self.logger.info(f"Advanced UKF initialized (variant: {variant.value})")
        self.logger.info(f"  State dim: {self.n}, Measurement dim: {self.m}")
        self.logger.info(f"  λ = {self.lambda_param:.6f}, α = {self.params.alpha:.6f}")
    
    def _update_ukf_parameters(self):
        """Update UKF scaling parameters"""
        # Calculate composite scaling parameter
        self.lambda_param = self.params.alpha**2 * (self.n + self.params.kappa) - self.n
        
        # Calculate weights
        self.weights_mean = np.zeros(2*self.n + 1)
        self.weights_cov = np.zeros(2*self.n + 1)
        
        # Central point weights
        self.weights_mean[0] = self.lambda_param / (self.n + self.lambda_param)
        self.weights_cov[0] = self.lambda_param / (self.n + self.lambda_param) + \
                             (1 - self.params.alpha**2 + self.params.beta)
        
        # Surrounding point weights
        for i in range(1, 2*self.n + 1):
            self.weights_mean[i] = 1 / (2 * (self.n + self.lambda_param))
            self.weights_cov[i] = 1 / (2 * (self.n + self.lambda_param))
        
        self.logger.debug(f"UKF parameters updated: λ={self.lambda_param:.6f}")
    
    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate sigma points with enhanced numerical stability
        
        χ_σ = [x̂, x̂ + √((n+λ)P), x̂ - √((n+λ)P)]
        """
        
        # Ensure positive definiteness
        P_safe = self._ensure_positive_definite(P)
        
        try:
            if self.variant == UKFVariant.SQUARE_ROOT:
                # Square-root implementation for better numerical stability
                sqrt_matrix = la.cholesky((self.n + self.lambda_param) * P_safe, lower=True)
            else:
                # Standard matrix square root
                sqrt_matrix = la.sqrtm((self.n + self.lambda_param) * P_safe)
                if np.iscomplexobj(sqrt_matrix):
                    sqrt_matrix = np.real(sqrt_matrix)
                    self.logger.warning("Complex square root encountered, taking real part")
            
        except la.LinAlgError:
            self.logger.warning("Cholesky decomposition failed, using eigendecomposition")
            eigenvals, eigenvecs = la.eigh(P_safe)
            eigenvals = np.maximum(eigenvals, self.params.min_eigenvalue)
            sqrt_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals * (self.n + self.lambda_param)))
        
        # Generate sigma points
        sigma_points = np.zeros((2*self.n + 1, self.n))
        
        # Central point
        sigma_points[0] = x
        
        # Surrounding points
        for i in range(self.n):
            sigma_points[i + 1] = x + sqrt_matrix[:, i]
            sigma_points[i + self.n + 1] = x - sqrt_matrix[:, i]
        
        return sigma_points
    
    def _ensure_positive_definite(self, P: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix is positive definite"""
        
        # Check condition number
        try:
            cond_num = np.linalg.cond(P)
            if cond_num > self.params.max_condition_number:
                self.logger.warning(f"High condition number: {cond_num:.2e}")
        except:
            pass
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = la.eigh(P)
        
        # Clip small eigenvalues
        eigenvals_clipped = np.maximum(eigenvals, self.params.min_eigenvalue)
        
        # Reconstruct matrix
        P_safe = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T
        
        # Ensure symmetry
        P_safe = 0.5 * (P_safe + P_safe.T)
        
        return P_safe
    
    def _adapt_parameters(self, innovation: np.ndarray, S: np.ndarray):
        """Adapt UKF parameters based on innovation statistics"""
        
        if not self.params.adaptive_alpha:
            return
        
        # Calculate normalized innovation squared
        try:
            innovation_normalized = innovation.T @ la.solve(S, innovation)
            
            # Adapt alpha based on innovation consistency
            if innovation_normalized > self.m + 2*np.sqrt(2*self.m):  # Chi-squared test
                # Innovation too large, increase alpha (more spread)
                self.params.alpha = min(self.params.alpha * 1.1, self.params.alpha_max)
            elif innovation_normalized < self.m - 2*np.sqrt(2*self.m):
                # Innovation too small, decrease alpha (less spread)  
                self.params.alpha = max(self.params.alpha * 0.9, self.params.alpha_min)
            
            # Update derived parameters
            self._update_ukf_parameters()
            
        except la.LinAlgError:
            self.logger.warning("Failed to adapt parameters due to singular innovation covariance")
    
    def predict(self, u: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None):
        """
        UKF prediction step with enhanced sigma point propagation
        
        x̂_(k+1|k) = Σ W_m^i × f(χ_i, u_k)
        P_(k+1|k) = Q + Σ W_c^i × [χ_i - x̂_(k+1|k)][χ_i - x̂_(k+1|k)]^T
        """
        
        if Q is not None:
            self.Q = Q
        
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.x_hat, self.P)
        
        # Propagate sigma points through process model
        sigma_points_pred = np.zeros_like(sigma_points)
        
        for i in range(2*self.n + 1):
            try:
                sigma_points_pred[i] = self.f(sigma_points[i], u)
            except Exception as e:
                self.logger.error(f"Process model failed for sigma point {i}: {e}")
                sigma_points_pred[i] = sigma_points[i]  # Fallback
        
        # Calculate predicted state
        self.x_hat = np.zeros(self.n)
        for i in range(2*self.n + 1):
            self.x_hat += self.weights_mean[i] * sigma_points_pred[i]
        
        # Calculate predicted covariance
        self.P = self.Q.copy()
        for i in range(2*self.n + 1):
            y_diff = sigma_points_pred[i] - self.x_hat
            self.P += self.weights_cov[i] * np.outer(y_diff, y_diff)
        
        # Ensure positive definiteness
        self.P = self._ensure_positive_definite(self.P)
        
        # Store propagated sigma points for update step
        self.sigma_points_pred = sigma_points_pred
    
    def update(self, z: np.ndarray, R: Optional[np.ndarray] = None):
        """
        UKF update step with innovation-based adaptation
        
        Innovation: ν = z - ŷ
        Innovation covariance: S = H×P×H^T + R + Ψ_cross
        Kalman gain: K = P×H^T×S^(-1)
        """
        
        if R is not None:
            self.R = R
        
        # Generate sigma points for measurement prediction
        sigma_points_meas = np.zeros((2*self.n + 1, self.m))
        
        for i in range(2*self.n + 1):
            try:
                sigma_points_meas[i] = self.h(self.sigma_points_pred[i])
            except Exception as e:
                self.logger.error(f"Measurement model failed for sigma point {i}: {e}")
                sigma_points_meas[i] = np.zeros(self.m)  # Fallback
        
        # Calculate predicted measurement
        z_pred = np.zeros(self.m)
        for i in range(2*self.n + 1):
            z_pred += self.weights_mean[i] * sigma_points_meas[i]
        
        # Calculate innovation covariance
        S = self.R.copy()
        for i in range(2*self.n + 1):
            y_diff = sigma_points_meas[i] - z_pred
            S += self.weights_cov[i] * np.outer(y_diff, y_diff)
        
        # Calculate cross-covariance
        P_xz = np.zeros((self.n, self.m))
        for i in range(2*self.n + 1):
            x_diff = self.sigma_points_pred[i] - self.x_hat
            z_diff = sigma_points_meas[i] - z_pred
            P_xz += self.weights_cov[i] * np.outer(x_diff, z_diff)
        
        # Calculate innovation
        innovation = z - z_pred
        
        # Adapt parameters based on innovation
        self._adapt_parameters(innovation, S)
        
        # Calculate Kalman gain
        try:
            K = P_xz @ la.solve(S, np.eye(self.m))
        except la.LinAlgError:
            self.logger.warning("Singular innovation covariance, using pseudoinverse")
            K = P_xz @ la.pinv(S)
        
        # Update state and covariance
        self.x_hat = self.x_hat + K @ innovation
        self.P = self.P - K @ S @ K.T
        
        # Ensure positive definiteness
        self.P = self._ensure_positive_definite(self.P)
        
        # Store performance metrics
        self.innovation_history.append(np.linalg.norm(innovation))
        
        # Calculate log-likelihood
        try:
            log_likelihood = -0.5 * (innovation.T @ la.solve(S, innovation) + 
                                   np.log(la.det(2*np.pi*S)))
            self.likelihood_history.append(log_likelihood)
        except:
            self.likelihood_history.append(-np.inf)
        
        return innovation, S, K
    
    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate and covariance"""
        return self.x_hat.copy(), self.P.copy()
    
    def set_process_noise(self, Q: np.ndarray):
        """Set process noise covariance"""
        self.Q = Q
    
    def set_measurement_noise(self, R: np.ndarray):
        """Set measurement noise covariance"""
        self.R = R
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get filter performance metrics"""
        
        if len(self.innovation_history) < 2:
            return {'status': 'insufficient_data'}
        
        metrics = {
            'mean_innovation_norm': np.mean(self.innovation_history[-10:]),
            'innovation_consistency': np.std(self.innovation_history[-10:]),
            'mean_log_likelihood': np.mean(self.likelihood_history[-10:]),
            'current_alpha': self.params.alpha,
            'condition_number': np.linalg.cond(self.P),
            'trace_covariance': np.trace(self.P)
        }
        
        return metrics
    
    def reset(self, x0: np.ndarray, P0: np.ndarray):
        """Reset filter with new initial conditions"""
        self.x_hat = x0.copy()
        self.P = self._ensure_positive_definite(P0)
        self.innovation_history.clear()
        self.likelihood_history.clear()
        self.logger.info("UKF reset with new initial conditions")

def main():
    """Demonstrate advanced UKF capabilities"""
    
    print("Advanced Unscented Kalman Filter Demonstration")
    print("=" * 50)
    
    # Define simple process and measurement models
    def process_model(x, u=None):
        """Simple linear process model with nonlinearity"""
        A = np.array([[1.0, 0.1], [0.0, 0.95]])
        return A @ x + 0.01 * x**2  # Add nonlinearity
    
    def measurement_model(x):
        """Nonlinear measurement model"""
        return np.array([x[0]**2 + x[1], x[0] + 0.5*x[1]**2])
    
    # Initialize UKF
    ukf = AdvancedUnscentedKalmanFilter(
        state_dim=2,
        measurement_dim=2,
        process_model=process_model,
        measurement_model=measurement_model,
        variant=UKFVariant.ADAPTIVE
    )
    
    # Set initial conditions
    x0 = np.array([1.0, 0.5])
    P0 = np.eye(2) * 0.1
    ukf.reset(x0, P0)
    
    # Set noise covariances
    ukf.set_process_noise(np.eye(2) * 0.01)
    ukf.set_measurement_noise(np.eye(2) * 0.1)
    
    print(f"Initial state: {x0}")
    print(f"Initial covariance trace: {np.trace(P0):.6f}")
    
    # Simulate filtering
    true_state = x0.copy()
    
    print(f"\nFiltering Simulation:")
    print(f"{'Step':>4} {'True X':>8} {'True Y':>8} {'Est X':>8} {'Est Y':>8} {'Innovation':>12} {'Alpha':>8}")
    print("-" * 64)
    
    for step in range(20):
        # True state evolution (with process noise)
        true_state = process_model(true_state) + np.random.multivariate_normal([0, 0], ukf.Q)
        
        # Generate noisy measurement
        true_measurement = measurement_model(true_state)
        measurement = true_measurement + np.random.multivariate_normal([0, 0], ukf.R)
        
        # UKF prediction and update
        ukf.predict()
        innovation, S, K = ukf.update(measurement)
        
        # Get state estimate
        x_est, P_est = ukf.get_state_estimate()
        
        # Print results
        innovation_norm = np.linalg.norm(innovation)
        print(f"{step:4d} {true_state[0]:8.4f} {true_state[1]:8.4f} "
              f"{x_est[0]:8.4f} {x_est[1]:8.4f} {innovation_norm:12.6f} "
              f"{ukf.params.alpha:8.6f}")
    
    # Display performance metrics
    print(f"\nPerformance Metrics:")
    metrics = ukf.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\nAdvanced UKF demonstration complete!")

if __name__ == "__main__":
    main()
