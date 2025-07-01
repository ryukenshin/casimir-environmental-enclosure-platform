"""
Multi-Physics Digital Twin Framework
Coupled thermal dynamics with electromagnetic and quantum field interactions

Mathematical Formulations:
- C_s dT_s/dt = Q_in - Q_out - Q_conduction  [Thermal dynamics]
- σ_thermal = E × α × (T - T_ref)  [Thermal stress]
- dx_digital/dt = f_coupled(x_mech, x_thermal, x_EM, x_quantum, u, w, t)  [Multi-physics coupling]
"""

import numpy as np
import scipy.integrate
import scipy.linalg
import scipy.constants as const
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import logging
from enum import Enum
import time

class PhysicsDomain(Enum):
    """Physics domain types"""
    MECHANICAL = "mechanical"
    THERMAL = "thermal"
    ELECTROMAGNETIC = "electromagnetic"
    QUANTUM = "quantum"
    FLUIDIC = "fluidic"

@dataclass
class PhysicsState:
    """State vector for a physics domain"""
    domain: PhysicsDomain
    state_vector: np.ndarray
    state_names: List[str]
    units: List[str]
    constraints: Optional[Dict] = None

@dataclass
class CouplingMatrix:
    """Coupling matrix between physics domains"""
    source_domain: PhysicsDomain
    target_domain: PhysicsDomain
    coupling_strength: float
    coupling_matrix: np.ndarray
    coupling_function: Optional[Callable] = None

@dataclass
class MultiPhysicsParameters:
    """Multi-physics system parameters"""
    # Thermal parameters
    thermal_capacity: float = 1000.0        # J/(kg·K)
    thermal_conductivity: float = 50.0      # W/(m·K)
    density: float = 2700.0                 # kg/m³
    convection_coefficient: float = 10.0    # W/(m²·K)
    
    # Mechanical parameters
    elastic_modulus: float = 70e9           # Pa
    poisson_ratio: float = 0.33
    thermal_expansion_coeff: float = 23e-6  # K⁻¹
    
    # Electromagnetic parameters
    permittivity: float = 8.85e-12         # F/m
    permeability: float = 4*np.pi*1e-7     # H/m
    conductivity: float = 3.5e7            # S/m
    
    # Quantum parameters
    casimir_constant: float = const.hbar * const.c * np.pi**2 / 240  # J·m³
    decoherence_time: float = 1e-12        # s
    coupling_strength: float = 1e-15       # J

class MultiPhysicsDigitalTwin:
    """
    Multi-physics digital twin with coupled dynamics
    
    Implements:
    - Thermal-mechanical coupling
    - Electromagnetic field interactions
    - Quantum field effects
    - Real-time state estimation
    - Predictive simulation
    """
    
    def __init__(self, parameters: MultiPhysicsParameters):
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters
        self.physics_states = {}
        self.coupling_matrices = []
        self.time_history = []
        self.state_history = []
        
        # Initialize physics domains
        self._initialize_physics_domains()
        self._setup_coupling_matrices()
        
    def _initialize_physics_domains(self):
        """Initialize individual physics domain states"""
        
        # Thermal domain: [T, dT/dt, heat_flux]
        thermal_state = PhysicsState(
            domain=PhysicsDomain.THERMAL,
            state_vector=np.array([293.15, 0.0, 0.0]),  # K, K/s, W/m²
            state_names=['temperature', 'temperature_rate', 'heat_flux'],
            units=['K', 'K/s', 'W/m²']
        )
        
        # Mechanical domain: [displacement, velocity, stress]
        mechanical_state = PhysicsState(
            domain=PhysicsDomain.MECHANICAL,
            state_vector=np.array([0.0, 0.0, 0.0]),  # m, m/s, Pa
            state_names=['displacement', 'velocity', 'stress'],
            units=['m', 'm/s', 'Pa']
        )
        
        # Electromagnetic domain: [E_field, B_field, current_density]
        em_state = PhysicsState(
            domain=PhysicsDomain.ELECTROMAGNETIC,
            state_vector=np.array([0.0, 0.0, 0.0]),  # V/m, T, A/m²
            state_names=['electric_field', 'magnetic_field', 'current_density'],
            units=['V/m', 'T', 'A/m²']
        )
        
        # Quantum domain: [field_amplitude, phase, coherence]
        quantum_state = PhysicsState(
            domain=PhysicsDomain.QUANTUM,
            state_vector=np.array([1.0, 0.0, 1.0]),  # normalized
            state_names=['field_amplitude', 'phase', 'coherence'],
            units=['1', 'rad', '1']
        )
        
        self.physics_states = {
            PhysicsDomain.THERMAL: thermal_state,
            PhysicsDomain.MECHANICAL: mechanical_state,
            PhysicsDomain.ELECTROMAGNETIC: em_state,
            PhysicsDomain.QUANTUM: quantum_state
        }
        
        self.logger.info("Physics domains initialized")
    
    def _setup_coupling_matrices(self):
        """Setup coupling matrices between physics domains"""
        
        # Thermal-Mechanical coupling
        thermal_mechanical = CouplingMatrix(
            source_domain=PhysicsDomain.THERMAL,
            target_domain=PhysicsDomain.MECHANICAL,
            coupling_strength=self.parameters.thermal_expansion_coeff,
            coupling_matrix=np.array([[1.0, 0.0, 0.0],    # T -> displacement
                                     [0.0, 0.0, 0.0],     # dT/dt -> velocity
                                     [1.0, 0.0, 0.0]])    # T -> stress
        )
        
        # Mechanical-Thermal coupling (stress-induced heating)
        mechanical_thermal = CouplingMatrix(
            source_domain=PhysicsDomain.MECHANICAL,
            target_domain=PhysicsDomain.THERMAL,
            coupling_strength=1e-9,  # Pa⁻¹·K
            coupling_matrix=np.array([[0.0, 0.0, 1.0],    # stress -> T
                                     [0.0, 0.0, 0.0],     # 
                                     [0.0, 0.0, 1.0]])    # stress -> heat_flux
        )
        
        # Electromagnetic-Thermal coupling (Joule heating)
        em_thermal = CouplingMatrix(
            source_domain=PhysicsDomain.ELECTROMAGNETIC,
            target_domain=PhysicsDomain.THERMAL,
            coupling_strength=1.0 / self.parameters.conductivity,
            coupling_matrix=np.array([[0.0, 0.0, 1.0],    # J -> T
                                     [0.0, 0.0, 0.0],     #
                                     [0.0, 0.0, 1.0]])    # J -> heat_flux
        )
        
        # Quantum-Electromagnetic coupling
        quantum_em = CouplingMatrix(
            source_domain=PhysicsDomain.QUANTUM,
            target_domain=PhysicsDomain.ELECTROMAGNETIC,
            coupling_strength=self.parameters.coupling_strength,
            coupling_matrix=np.array([[1.0, 1.0, 0.0],    # quantum -> E_field
                                     [0.0, 1.0, 0.0],     # phase -> B_field
                                     [0.0, 0.0, 0.0]])    #
        )
        
        self.coupling_matrices = [
            thermal_mechanical,
            mechanical_thermal,
            em_thermal,
            quantum_em
        ]
        
        self.logger.info(f"Coupling matrices setup: {len(self.coupling_matrices)} couplings")
    
    def thermal_dynamics(self, thermal_state: np.ndarray, 
                        external_inputs: Dict, t: float) -> np.ndarray:
        """
        Thermal dynamics differential equation
        
        Mathematical formulation:
        C_s dT_s/dt = Q_in - Q_out - Q_conduction
        
        Args:
            thermal_state: [T, dT/dt, heat_flux]
            external_inputs: External thermal inputs
            t: Time
            
        Returns:
            State derivatives
        """
        T, dT_dt, q = thermal_state
        
        # External inputs
        Q_in = external_inputs.get('heat_input', 0.0)      # W
        Q_ambient = external_inputs.get('ambient_temp', 293.15)  # K
        surface_area = external_inputs.get('surface_area', 0.01)  # m²
        volume = external_inputs.get('volume', 1e-6)       # m³
        
        # Heat capacity
        C_s = self.parameters.thermal_capacity * self.parameters.density * volume
        
        # Heat transfer mechanisms
        Q_conduction = self.parameters.thermal_conductivity * surface_area * (T - Q_ambient)
        Q_convection = self.parameters.convection_coefficient * surface_area * (T - Q_ambient)
        Q_out = Q_conduction + Q_convection
        
        # Thermal dynamics: C_s dT/dt = Q_in - Q_out
        d2T_dt2 = (Q_in - Q_out) / C_s
        
        # Heat flux dynamics (simplified)
        dq_dt = -q / 1.0 + (T - Q_ambient) * self.parameters.thermal_conductivity
        
        return np.array([dT_dt, d2T_dt2, dq_dt])
    
    def mechanical_dynamics(self, mechanical_state: np.ndarray,
                          external_inputs: Dict, t: float) -> np.ndarray:
        """
        Mechanical dynamics with thermal stress
        
        Mathematical formulation:
        σ_thermal = E × α × (T - T_ref)
        
        Args:
            mechanical_state: [displacement, velocity, stress]
            external_inputs: External mechanical inputs
            t: Time
            
        Returns:
            State derivatives
        """
        x, v, sigma = mechanical_state
        
        # External inputs
        F_external = external_inputs.get('external_force', 0.0)  # N
        area = external_inputs.get('cross_section_area', 1e-4)   # m²
        length = external_inputs.get('length', 0.1)              # m
        mass = external_inputs.get('mass', 0.1)                  # kg
        
        # Thermal coupling
        T_current = self.physics_states[PhysicsDomain.THERMAL].state_vector[0]
        T_ref = external_inputs.get('reference_temp', 293.15)
        
        # Thermal stress: σ_thermal = E × α × (T - T_ref)
        sigma_thermal = (self.parameters.elastic_modulus * 
                        self.parameters.thermal_expansion_coeff * 
                        (T_current - T_ref))
        
        # Total stress
        sigma_mechanical = F_external / area
        sigma_total = sigma_mechanical + sigma_thermal
        
        # Mechanical equation of motion
        # F = ma = (σ × A)
        acceleration = (sigma_total * area) / mass
        
        # Stress evolution (simplified)
        dsigma_dt = self.parameters.elastic_modulus * v / length
        
        return np.array([v, acceleration, dsigma_dt])
    
    def electromagnetic_dynamics(self, em_state: np.ndarray,
                               external_inputs: Dict, t: float) -> np.ndarray:
        """
        Electromagnetic field dynamics
        
        Args:
            em_state: [E_field, B_field, current_density]
            external_inputs: External EM inputs
            t: Time
            
        Returns:
            State derivatives
        """
        E, B, J = em_state
        
        # External inputs
        charge_density = external_inputs.get('charge_density', 0.0)  # C/m³
        
        # Maxwell's equations (simplified 1D)
        # ∇×E = -∂B/∂t
        # ∇×B = μ₀J + μ₀ε₀∂E/∂t
        
        mu_0 = self.parameters.permeability
        epsilon_0 = self.parameters.permittivity
        
        dE_dt = (B / mu_0 - J) / epsilon_0
        dB_dt = -mu_0 * E
        
        # Current density evolution (Ohm's law + drift)
        dJ_dt = self.parameters.conductivity * E - J / 1e-9  # Relaxation
        
        return np.array([dE_dt, dB_dt, dJ_dt])
    
    def quantum_dynamics(self, quantum_state: np.ndarray,
                        external_inputs: Dict, t: float) -> np.ndarray:
        """
        Quantum field dynamics with decoherence
        
        Args:
            quantum_state: [field_amplitude, phase, coherence]
            external_inputs: External quantum inputs
            t: Time
            
        Returns:
            State derivatives
        """
        psi, phi, coherence = quantum_state
        
        # Quantum field oscillation
        omega_q = external_inputs.get('quantum_frequency', 1e12)  # rad/s
        
        # Schrödinger-like evolution
        dpsi_dt = -omega_q * np.sin(phi) * coherence
        dphi_dt = omega_q
        
        # Decoherence
        gamma = 1.0 / self.parameters.decoherence_time
        dcoherence_dt = -gamma * coherence
        
        return np.array([dpsi_dt, dphi_dt, dcoherence_dt])
    
    def calculate_coupling_effects(self) -> Dict[PhysicsDomain, np.ndarray]:
        """
        Calculate coupling effects between physics domains
        
        Returns:
            Coupling contributions for each domain
        """
        coupling_effects = {domain: np.zeros_like(state.state_vector) 
                           for domain, state in self.physics_states.items()}
        
        for coupling in self.coupling_matrices:
            source_state = self.physics_states[coupling.source_domain].state_vector
            
            # Linear coupling: effect = strength × matrix × source_state
            coupling_contribution = (coupling.coupling_strength * 
                                   coupling.coupling_matrix @ source_state)
            
            # Add to target domain
            if coupling.target_domain in coupling_effects:
                coupling_effects[coupling.target_domain] += coupling_contribution
        
        return coupling_effects
    
    def coupled_system_dynamics(self, combined_state: np.ndarray, 
                              t: float, external_inputs: Dict) -> np.ndarray:
        """
        Complete multi-physics coupled system dynamics
        
        Mathematical formulation:
        dx_digital/dt = f_coupled(x_mech, x_thermal, x_EM, x_quantum, u, w, t)
        
        Args:
            combined_state: Combined state vector from all domains
            t: Time
            external_inputs: External inputs for all domains
            
        Returns:
            Combined state derivatives
        """
        # Extract individual domain states
        state_sizes = {
            PhysicsDomain.THERMAL: 3,
            PhysicsDomain.MECHANICAL: 3,
            PhysicsDomain.ELECTROMAGNETIC: 3,
            PhysicsDomain.QUANTUM: 3
        }
        
        idx = 0
        for domain, size in state_sizes.items():
            self.physics_states[domain].state_vector = combined_state[idx:idx+size]
            idx += size
        
        # Calculate individual domain dynamics
        thermal_derivs = self.thermal_dynamics(
            self.physics_states[PhysicsDomain.THERMAL].state_vector,
            external_inputs.get('thermal', {}), t
        )
        
        mechanical_derivs = self.mechanical_dynamics(
            self.physics_states[PhysicsDomain.MECHANICAL].state_vector,
            external_inputs.get('mechanical', {}), t
        )
        
        em_derivs = self.electromagnetic_dynamics(
            self.physics_states[PhysicsDomain.ELECTROMAGNETIC].state_vector,
            external_inputs.get('electromagnetic', {}), t
        )
        
        quantum_derivs = self.quantum_dynamics(
            self.physics_states[PhysicsDomain.QUANTUM].state_vector,
            external_inputs.get('quantum', {}), t
        )
        
        # Calculate coupling effects
        coupling_effects = self.calculate_coupling_effects()
        
        # Apply coupling effects
        thermal_derivs += coupling_effects[PhysicsDomain.THERMAL]
        mechanical_derivs += coupling_effects[PhysicsDomain.MECHANICAL]
        em_derivs += coupling_effects[PhysicsDomain.ELECTROMAGNETIC]
        quantum_derivs += coupling_effects[PhysicsDomain.QUANTUM]
        
        # Combine all derivatives
        combined_derivatives = np.concatenate([
            thermal_derivs, mechanical_derivs, em_derivs, quantum_derivs
        ])
        
        return combined_derivatives
    
    def simulate_multi_physics_system(self,
                                    simulation_time: float,
                                    external_inputs: Dict,
                                    initial_conditions: Optional[Dict] = None) -> Dict:
        """
        Simulate complete multi-physics system
        
        Args:
            simulation_time: Total simulation time (s)
            external_inputs: Time-dependent external inputs
            initial_conditions: Initial conditions for all domains
            
        Returns:
            Simulation results
        """
        # Set initial conditions
        if initial_conditions:
            for domain, ic in initial_conditions.items():
                if domain in self.physics_states:
                    self.physics_states[domain].state_vector = np.array(ic)
        
        # Combine initial state
        initial_state = np.concatenate([
            state.state_vector for state in self.physics_states.values()
        ])
        
        # Time points
        t_span = (0, simulation_time)
        t_eval = np.linspace(0, simulation_time, int(simulation_time * 100))
        
        # Define ODE system
        def ode_system(t, y):
            return self.coupled_system_dynamics(y, t, external_inputs)
        
        # Solve ODE system
        try:
            solution = scipy.integrate.solve_ivp(
                ode_system,
                t_span,
                initial_state,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-6,
                atol=1e-9
            )
            
            if solution.success:
                # Parse results
                results = {
                    'time': solution.t,
                    'success': True,
                    'domains': {}
                }
                
                # Extract domain results
                state_sizes = [3, 3, 3, 3]  # thermal, mechanical, EM, quantum
                domain_names = list(self.physics_states.keys())
                
                idx = 0
                for i, (domain, size) in enumerate(zip(domain_names, state_sizes)):
                    results['domains'][domain] = {
                        'states': solution.y[idx:idx+size, :],
                        'state_names': self.physics_states[domain].state_names,
                        'units': self.physics_states[domain].units
                    }
                    idx += size
                
                self.logger.info(f"Multi-physics simulation completed successfully")
                
            else:
                results = {'success': False, 'message': solution.message}
                self.logger.error(f"Simulation failed: {solution.message}")
                
        except Exception as e:
            results = {'success': False, 'error': str(e)}
            self.logger.error(f"Simulation error: {e}")
        
        return results
    
    def analyze_coupling_strength(self) -> Dict:
        """
        Analyze coupling strength between domains
        
        Returns:
            Coupling analysis results
        """
        coupling_analysis = {}
        
        for coupling in self.coupling_matrices:
            # Calculate coupling magnitude
            coupling_magnitude = np.linalg.norm(coupling.coupling_matrix) * coupling.coupling_strength
            
            # Eigenvalue analysis
            eigenvalues = np.linalg.eigvals(coupling.coupling_matrix)
            max_eigenvalue = np.max(np.real(eigenvalues))
            
            coupling_info = {
                'source': coupling.source_domain.value,
                'target': coupling.target_domain.value,
                'strength': coupling.coupling_strength,
                'magnitude': coupling_magnitude,
                'max_eigenvalue': max_eigenvalue,
                'matrix_condition': np.linalg.cond(coupling.coupling_matrix)
            }
            
            coupling_key = f"{coupling.source_domain.value}_to_{coupling.target_domain.value}"
            coupling_analysis[coupling_key] = coupling_info
        
        return coupling_analysis

def main():
    """Demonstration of multi-physics digital twin capabilities"""
    
    # Initialize parameters
    parameters = MultiPhysicsParameters(
        thermal_capacity=800.0,      # J/(kg·K)
        thermal_conductivity=200.0,  # W/(m·K)
        elastic_modulus=70e9,        # Pa
        thermal_expansion_coeff=23e-6 # K⁻¹
    )
    
    # Create digital twin
    digital_twin = MultiPhysicsDigitalTwin(parameters)
    
    print("Multi-Physics Digital Twin Analysis")
    print("="*50)
    
    # Analyze coupling strength
    coupling_analysis = digital_twin.analyze_coupling_strength()
    print(f"\nCoupling Analysis:")
    for coupling_name, info in coupling_analysis.items():
        print(f"{coupling_name}: strength={info['strength']:.2e}, "
              f"magnitude={info['magnitude']:.2e}")
    
    # Define external inputs
    external_inputs = {
        'thermal': {
            'heat_input': 1.0,          # W
            'ambient_temp': 293.15,     # K
            'surface_area': 0.01,       # m²
            'volume': 1e-6              # m³
        },
        'mechanical': {
            'external_force': 0.1,      # N
            'cross_section_area': 1e-4, # m²
            'length': 0.1,              # m
            'mass': 0.1,                # kg
            'reference_temp': 293.15    # K
        },
        'electromagnetic': {
            'charge_density': 0.0       # C/m³
        },
        'quantum': {
            'quantum_frequency': 1e12   # rad/s
        }
    }
    
    # Initial conditions
    initial_conditions = {
        PhysicsDomain.THERMAL: [293.15, 0.0, 0.0],      # T, dT/dt, q
        PhysicsDomain.MECHANICAL: [0.0, 0.0, 0.0],      # x, v, σ
        PhysicsDomain.ELECTROMAGNETIC: [0.0, 0.0, 0.0], # E, B, J
        PhysicsDomain.QUANTUM: [1.0, 0.0, 1.0]          # ψ, φ, coherence
    }
    
    # Run simulation
    print(f"\nRunning multi-physics simulation...")
    results = digital_twin.simulate_multi_physics_system(
        simulation_time=10.0,
        external_inputs=external_inputs,
        initial_conditions=initial_conditions
    )
    
    if results['success']:
        print(f"Simulation completed successfully")
        
        # Extract key results
        thermal_results = results['domains'][PhysicsDomain.THERMAL]
        mechanical_results = results['domains'][PhysicsDomain.MECHANICAL]
        
        # Temperature evolution
        final_temp = thermal_results['states'][0, -1]
        temp_change = final_temp - thermal_results['states'][0, 0]
        
        # Mechanical response
        final_displacement = mechanical_results['states'][0, -1]
        final_stress = mechanical_results['states'][2, -1]
        
        print(f"\nSimulation Results:")
        print(f"Final temperature: {final_temp:.3f} K (Δ{temp_change:.3f} K)")
        print(f"Final displacement: {final_displacement*1e9:.2f} nm")
        print(f"Final thermal stress: {final_stress/1e6:.2f} MPa")
        
        # Multi-physics coupling verification
        thermal_expansion = parameters.thermal_expansion_coeff * temp_change * 0.1  # 10cm length
        print(f"Expected thermal expansion: {thermal_expansion*1e9:.2f} nm")
        print(f"Coupling verification: {abs(final_displacement - thermal_expansion)*1e9:.2f} nm difference")
        
    else:
        print(f"Simulation failed: {results.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()
