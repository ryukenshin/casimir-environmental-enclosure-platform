"""
Enhanced Multi-Physics Coupling Matrix
Advanced formulations for thermal-mechanical-electromagnetic-quantum coupling

Mathematical Formulations:
- Enhanced coupling matrix: C_enhanced with physics-based cross-coupling terms
- Thermal-mechanical: θ_tm = 2.3×10⁻⁵ × E_young × ΔT  
- Electromagnetic-thermal: ε_me = q_density × v × B / (ρ × c_p)
- Quantum-classical: γ_qt = ℏω_backaction / (k_B × T_classical)
- Multi-physics state evolution with cross-domain interactions
"""

import numpy as np
import scipy.constants as const
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from enum import Enum

class PhysicsDomain(Enum):
    """Physics domains for multi-physics coupling"""
    MECHANICAL = 0
    THERMAL = 1
    ELECTROMAGNETIC = 2
    QUANTUM = 3

@dataclass
class CouplingParameters:
    """Physics-based coupling parameters"""
    # Material properties
    young_modulus: float = 200e9        # Pa (steel)
    density: float = 7850               # kg/m³
    specific_heat: float = 460          # J/(kg·K)
    thermal_expansion: float = 12e-6    # K⁻¹
    
    # Electromagnetic properties
    conductivity: float = 5.8e7         # S/m
    permeability: float = const.mu_0    # H/m
    permittivity: float = const.epsilon_0 # F/m
    
    # Quantum properties
    decoherence_rate: float = 1e6       # Hz
    coupling_strength: float = 1e-20    # J
    
    # Environmental conditions
    temperature: float = 293.15         # K
    magnetic_field: float = 1e-4        # T
    electric_field: float = 1e3         # V/m

class EnhancedMultiPhysicsCoupling:
    """
    Enhanced multi-physics coupling with physics-based cross-coupling terms
    
    Implements advanced coupling matrix formulations:
    C_enhanced = [[1.0, θ_tm×α_te, ε_me×β_mt, γ_qt×δ_qm],
                  [α_tm×θ_te, 1.0, σ_em×ρ_et, φ_qe×ψ_qm],
                  [β_em×ε_mt, ρ_me×σ_et, 1.0, ω_qem×ξ_qet],
                  [δ_qm×γ_qt, ψ_qe×φ_qt, ξ_qem×ω_qet, 1.0]]
    """
    
    def __init__(self, parameters: Optional[CouplingParameters] = None):
        self.logger = logging.getLogger(__name__)
        self.params = parameters or CouplingParameters()
        
        # Calculate fundamental coupling coefficients
        self._calculate_coupling_coefficients()
        
        # Initialize enhanced coupling matrix
        self.coupling_matrix = self._build_enhanced_coupling_matrix()
        
        self.logger.info("Enhanced multi-physics coupling initialized")
    
    def _calculate_coupling_coefficients(self):
        """Calculate physics-based coupling coefficients"""
        
        # Thermal-mechanical coupling coefficients
        # θ_tm = 2.3×10⁻⁵ × E_young × ΔT (thermal stress coefficient)
        delta_T = 1.0  # Reference temperature change (K)
        self.theta_tm = 2.3e-5 * self.params.young_modulus * delta_T
        self.alpha_te = self.params.thermal_expansion
        
        # Electromagnetic-thermal coupling coefficients  
        # ε_me = q_density × v × B / (ρ × c_p) (Joule heating)
        charge_density = 1e6  # C/m³ (typical for conductors)
        velocity = 1e-3       # m/s (typical drift velocity)
        self.epsilon_me = (charge_density * velocity * self.params.magnetic_field) / \
                         (self.params.density * self.params.specific_heat)
        self.beta_mt = self.params.conductivity / self.params.density
        
        # Quantum-classical coupling coefficients
        # γ_qt = ℏω_backaction / (k_B × T_classical) (quantum backaction)
        omega_backaction = 2 * np.pi * self.params.decoherence_rate  # rad/s
        self.gamma_qt = (const.hbar * omega_backaction) / \
                       (const.k * self.params.temperature)
        self.delta_qm = self.params.coupling_strength / \
                       (const.k * self.params.temperature)
        
        # Additional cross-coupling terms
        self.sigma_em = self.params.conductivity * self.params.electric_field
        self.rho_et = self.params.thermal_expansion * self.params.temperature
        self.phi_qe = self.params.coupling_strength / (const.e * self.params.electric_field)
        self.psi_qm = self.gamma_qt * 0.5  # Reduced quantum-mechanical coupling
        
        # Electromagnetic-quantum coupling
        self.omega_qem = (const.hbar * const.c) / \
                        (const.e * self.params.electric_field * 1e-6)  # m (Compton scale)
        self.xi_qet = self.phi_qe * self.rho_et
        
        self.logger.info(f"Coupling coefficients calculated:")
        self.logger.info(f"  θ_tm = {self.theta_tm:.3e}, α_te = {self.alpha_te:.3e}")
        self.logger.info(f"  ε_me = {self.epsilon_me:.3e}, β_mt = {self.beta_mt:.3e}")
        self.logger.info(f"  γ_qt = {self.gamma_qt:.3e}, δ_qm = {self.delta_qm:.3e}")
    
    def _build_enhanced_coupling_matrix(self) -> np.ndarray:
        """Build enhanced 4×4 coupling matrix with physics-based terms"""
        
        # Enhanced coupling matrix with physics-based cross-coupling
        C = np.array([
            # Mechanical row
            [1.0, 
             self.theta_tm * self.alpha_te,
             self.epsilon_me * self.beta_mt,
             self.gamma_qt * self.delta_qm],
            
            # Thermal row  
            [self.alpha_te * self.theta_tm,
             1.0,
             self.sigma_em * self.rho_et,
             self.phi_qe * self.psi_qm],
            
            # Electromagnetic row
            [self.beta_mt * self.epsilon_me,
             self.rho_et * self.sigma_em,
             1.0,
             self.omega_qem * self.xi_qet],
            
            # Quantum row
            [self.delta_qm * self.gamma_qt,
             self.psi_qm * self.phi_qe,
             self.xi_qet * self.omega_qem,
             1.0]
        ])
        
        # Enhanced numerical stability measures
        # 1. Check condition number
        condition_number = np.linalg.cond(C)
        if condition_number > 100:  # UQ CRITICAL: Condition number too high
            self.logger.warning(f"High condition number detected: {condition_number:.2e}")
            
            # Apply eigenvalue regularization
            eigenvals, eigenvecs = np.linalg.eigh(C)
            eigenvals = np.clip(eigenvals, 0.01, 10.0)  # Bound eigenvalues
            C = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            self.logger.info(f"Applied eigenvalue regularization, new condition number: {np.linalg.cond(C):.2e}")
        
        # 2. Normalize off-diagonal terms to ensure stability  
        max_off_diag = np.max(np.abs(C - np.eye(4)))
        if max_off_diag > 0.05:  # UQ CRITICAL: Tighter limit for stability
            scaling_factor = 0.05 / max_off_diag  # More conservative scaling
            C = np.eye(4) + scaling_factor * (C - np.eye(4))
            self.logger.warning(f"Coupling matrix scaled by {scaling_factor:.3f} for enhanced stability")
        
        # 3. Ensure symmetry for physical consistency
        C = 0.5 * (C + C.T)
        
        # 4. Final validation
        final_condition = np.linalg.cond(C)
        if final_condition > 50:  # UQ CRITICAL: Final check
            self.logger.error(f"Critical UQ concern: Final condition number {final_condition:.2e} exceeds safety threshold")
            # Emergency fallback to nearly diagonal matrix
            C = np.eye(4) + 0.01 * (C - np.eye(4))
        
        return C
    
    def calculate_multi_physics_state_evolution(self, 
                                              state: np.ndarray,
                                              control: np.ndarray,
                                              time_step: float) -> np.ndarray:
        """
        Calculate multi-physics state evolution with cross-domain coupling
        
        Enhanced state evolution:
        dx/dt = v_mech + C_tm × dT/dt + C_em × E_field + C_qm × ψ_quantum
        dv/dt = (F_total - c×v - k×x)/m + ξ_thermal + ξ_em + ξ_quantum  
        dT/dt = (Q_gen - h×A×(T - T_amb))/(ρ×c_p×V) + coupling_mechanical + coupling_em
        dE/dt = -(E/(μ₀×εᵣ×ε₀)) + coupling_mechanical + coupling_thermal
        
        Args:
            state: Current state [x, v, T, E] (position, velocity, temperature, E-field)
            control: Control inputs [F, Q, B, ψ] (force, heat, B-field, quantum control)
            time_step: Integration time step (s)
            
        Returns:
            New state after time_step
        """
        
        x, v, T, E = state
        F_control, Q_control, B_control, psi_control = control
        
        # System parameters
        mass = 1.0          # kg
        spring_k = 1000.0   # N/m  
        damping_c = 10.0    # N·s/m
        heat_capacity = self.params.density * self.params.specific_heat * 1e-3  # J/K
        heat_transfer_coeff = 100.0  # W/(m²·K)
        surface_area = 0.01  # m²
        T_ambient = 293.15   # K
        
        # Calculate coupling terms from enhanced matrix
        coupling_tm = self.coupling_matrix[0, 1] * (T - T_ambient)
        coupling_em = self.coupling_matrix[0, 2] * E
        coupling_qm = self.coupling_matrix[0, 3] * psi_control
        
        coupling_mt = self.coupling_matrix[1, 0] * abs(v)
        coupling_et = self.coupling_matrix[1, 2] * E**2 / (2 * self.params.permittivity)
        coupling_qt = self.coupling_matrix[1, 3] * psi_control**2
        
        coupling_me = self.coupling_matrix[2, 0] * v * self.params.magnetic_field
        coupling_te = self.coupling_matrix[2, 1] * (T - T_ambient)
        coupling_qe = self.coupling_matrix[2, 3] * psi_control
        
        # Enhanced state evolution equations
        # Mechanical dynamics
        dx_dt = v + coupling_tm + coupling_em + coupling_qm
        
        dv_dt = (F_control - damping_c * v - spring_k * x) / mass + \
                coupling_mt + coupling_et + coupling_qt
        
        # Thermal dynamics  
        Q_generated = Q_control + coupling_me + coupling_qe
        Q_loss = heat_transfer_coeff * surface_area * (T - T_ambient)
        dT_dt = (Q_generated - Q_loss) / heat_capacity + coupling_mt + coupling_et
        
        # Electromagnetic dynamics
        E_decay = E / (self.params.permeability * self.params.permittivity * 1e-6)
        dE_dt = -E_decay + coupling_me + coupling_te + coupling_qe
        
        # Euler integration
        new_state = state + time_step * np.array([dx_dt, dv_dt, dT_dt, dE_dt])
        
        return new_state
    
    def get_coupling_strength(self, domain1: PhysicsDomain, domain2: PhysicsDomain) -> float:
        """Get coupling strength between two physics domains"""
        return self.coupling_matrix[domain1.value, domain2.value]
    
    def update_parameters(self, new_params: CouplingParameters):
        """Update coupling parameters and recalculate matrix"""
        self.params = new_params
        self._calculate_coupling_coefficients()
        self.coupling_matrix = self._build_enhanced_coupling_matrix()
        self.logger.info("Coupling parameters updated")
    
    def get_coupling_matrix(self) -> np.ndarray:
        """Get the enhanced coupling matrix"""
        return self.coupling_matrix.copy()
    
    def validate_coupling_stability(self) -> Dict[str, float]:
        """Validate coupling matrix stability and conditioning"""
        
        # Check matrix conditioning
        condition_number = np.linalg.cond(self.coupling_matrix)
        
        # Check eigenvalues for stability
        eigenvalues = np.linalg.eigvals(self.coupling_matrix)
        max_real_part = np.max(np.real(eigenvalues))
        
        # Check symmetry (should be approximately symmetric for physical systems)
        symmetry_error = np.max(np.abs(self.coupling_matrix - self.coupling_matrix.T))
        
        stability_metrics = {
            'condition_number': condition_number,
            'max_eigenvalue_real': max_real_part,
            'symmetry_error': symmetry_error,
            'is_stable': condition_number < 100 and max_real_part < 10.0
        }
        
        self.logger.info(f"Coupling stability: {stability_metrics}")
        return stability_metrics

def main():
    """Demonstrate enhanced multi-physics coupling"""
    
    print("Enhanced Multi-Physics Coupling Demonstration")
    print("=" * 50)
    
    # Initialize coupling system
    coupling = EnhancedMultiPhysicsCoupling()
    
    # Display coupling matrix
    print("\nEnhanced Coupling Matrix:")
    print(f"{'':>12} {'Mech':>10} {'Thermal':>10} {'EM':>10} {'Quantum':>10}")
    domains = ['Mechanical', 'Thermal', 'EM', 'Quantum']
    
    for i, domain in enumerate(domains):
        row = f"{domain:>12}"
        for j in range(4):
            row += f" {coupling.coupling_matrix[i,j]:>9.2e}"
        print(row)
    
    # Validate stability
    print(f"\nCoupling Stability Validation:")
    stability = coupling.validate_coupling_stability()
    for key, value in stability.items():
        print(f"  {key}: {value}")
    
    # Demonstrate state evolution
    print(f"\nMulti-Physics State Evolution:")
    initial_state = np.array([0.0, 0.0, 293.15, 1000.0])  # [x, v, T, E]
    control_input = np.array([10.0, 100.0, 1e-4, 1.0])    # [F, Q, B, ψ]
    
    print(f"Initial state: {initial_state}")
    print(f"Control input: {control_input}")
    
    # Evolve system for 10 time steps
    state = initial_state.copy()
    dt = 0.01  # 10 ms time step
    
    for step in range(10):
        state = coupling.calculate_multi_physics_state_evolution(state, control_input, dt)
        if step % 3 == 0:  # Print every 3rd step
            print(f"Step {step:2d}: x={state[0]:7.4f}, v={state[1]:7.4f}, T={state[2]:7.2f}, E={state[3]:7.1f}")
    
    print(f"\nFinal state: {state}")
    print(f"Coupling demonstration complete!")

if __name__ == "__main__":
    main()
