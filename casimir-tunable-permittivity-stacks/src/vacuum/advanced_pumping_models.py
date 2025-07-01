"""
Advanced Pumping Models
Dynamic outgassing and pumping speed calculations for UHV systems

Mathematical Formulations:
- Q = S × (P₁ - P₂)  [Basic pumping equation]
- S_eff = 1/(1/S_pump + 1/S_conductance)  [Effective pumping speed]
- Outgassing Rate < 10⁻¹⁰ Pa⋅m³/s
"""

import numpy as np
import scipy.constants as const
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from enum import Enum

class PumpType(Enum):
    """Types of vacuum pumps"""
    TURBOMOLECULAR = "turbomolecular"
    ION = "ion"
    TITANIUM_SUBLIMATION = "titanium_sublimation"
    NEG = "non_evaporable_getter"
    CRYOPUMP = "cryopump"

@dataclass
class PumpCharacteristics:
    """Vacuum pump characteristics"""
    pump_type: PumpType
    nominal_speed: float        # m³/s
    base_pressure: float        # Pa
    compression_ratio: float
    molecular_weight_dep: Dict[str, float]  # Speed dependence on molecular weight
    temperature_coefficient: float
    power_consumption: float    # W

@dataclass
class ConductanceElement:
    """Vacuum conductance element"""
    geometry: str              # 'circular', 'rectangular', 'elbow'
    diameter: float           # m (for circular)
    length: float            # m
    conductance: float       # m³/s

class AdvancedPumpingSystem:
    """
    Advanced pumping system with multi-stage analysis
    
    Implements:
    - Multi-stage pumping calculations
    - Conductance-limited analysis
    - Dynamic outgassing compensation
    - Real-time pressure optimization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pump_database = self._initialize_pump_database()
        self.target_pressure = 1e-6  # Pa
        
    def _initialize_pump_database(self) -> Dict[str, PumpCharacteristics]:
        """Initialize pump characteristics database"""
        return {
            'turbo_high_speed': PumpCharacteristics(
                pump_type=PumpType.TURBOMOLECULAR,
                nominal_speed=2.0,  # m³/s
                base_pressure=1e-9,  # Pa
                compression_ratio=1e8,
                molecular_weight_dep={'N2': 1.0, 'H2O': 0.8, 'H2': 1.2},
                temperature_coefficient=0.02,  # /K
                power_consumption=500  # W
            ),
            'ion_pump_large': PumpCharacteristics(
                pump_type=PumpType.ION,
                nominal_speed=0.5,  # m³/s
                base_pressure=1e-10,  # Pa
                compression_ratio=1e6,
                molecular_weight_dep={'N2': 1.0, 'H2O': 1.2, 'H2': 0.3},
                temperature_coefficient=-0.01,  # /K
                power_consumption=300  # W
            ),
            'tsp_pump': PumpCharacteristics(
                pump_type=PumpType.TITANIUM_SUBLIMATION,
                nominal_speed=5.0,  # m³/s (for active gases)
                base_pressure=1e-11,  # Pa
                compression_ratio=1e9,
                molecular_weight_dep={'N2': 2.0, 'H2O': 3.0, 'H2': 0.1},
                temperature_coefficient=0.05,  # /K
                power_consumption=100  # W
            ),
            'neg_pump': PumpCharacteristics(
                pump_type=PumpType.NEG,
                nominal_speed=1.0,  # m³/s
                base_pressure=1e-12,  # Pa
                compression_ratio=1e10,
                molecular_weight_dep={'N2': 1.5, 'H2O': 2.0, 'H2': 0.05},
                temperature_coefficient=0.03,  # /K
                power_consumption=50  # W
            )
        }
    
    def calculate_conductance(self, element: ConductanceElement, 
                            gas_type: str = 'N2',
                            temperature: float = 300.0) -> float:
        """
        Calculate vacuum conductance for various geometries
        
        Mathematical formulations:
        - Circular: C = 11.6 × A × √(T/M) / (1 + 3.8×L/D)
        - Molecular flow regime
        
        Args:
            element: Conductance element description
            gas_type: Gas type for molecular weight
            temperature: Temperature in K
            
        Returns:
            Conductance in m³/s
        """
        # Molecular weights (kg/mol)
        molecular_weights = {
            'N2': 0.028, 'O2': 0.032, 'H2O': 0.018, 
            'H2': 0.002, 'He': 0.004, 'Ar': 0.040
        }
        
        if gas_type not in molecular_weights:
            raise ValueError(f"Gas type {gas_type} not supported")
            
        M = molecular_weights[gas_type]  # kg/mol
        
        if element.geometry == 'circular':
            # Circular conductance: C = 11.6 × A × √(T/M) / (1 + 3.8×L/D)
            area = np.pi * (element.diameter/2)**2  # m²
            aspect_ratio_factor = 1 + 3.8 * element.length / element.diameter
            
            conductance = 11.6 * area * np.sqrt(temperature / M) / aspect_ratio_factor
            
        elif element.geometry == 'rectangular':
            # Simplified rectangular conductance
            area = element.diameter * element.length  # Using diameter as width
            conductance = 9.0 * area * np.sqrt(temperature / M)
            
        elif element.geometry == 'elbow':
            # Elbow reduction factor
            straight_conductance = self.calculate_conductance(
                ConductanceElement('circular', element.diameter, element.length, 0),
                gas_type, temperature
            )
            elbow_factor = 0.7  # Typical reduction
            conductance = straight_conductance * elbow_factor
            
        else:
            raise ValueError(f"Geometry {element.geometry} not supported")
        
        self.logger.debug(f"Conductance ({element.geometry}): {conductance:.2e} m³/s")
        return conductance
    
    def calculate_effective_pumping_speed(self,
                                        pump_speeds: List[float],
                                        conductances: List[float]) -> float:
        """
        Calculate effective pumping speed with conductance limitations
        
        Mathematical formulation:
        S_eff = 1 / (1/S_pump + 1/S_conductance)
        For multiple pumps: 1/S_total = Σ(1/S_i)
        
        Args:
            pump_speeds: List of pump speeds in m³/s
            conductances: List of conductance values in m³/s
            
        Returns:
            Effective pumping speed in m³/s
        """
        if len(pump_speeds) != len(conductances):
            raise ValueError("Pump speeds and conductances lists must have same length")
        
        # Calculate effective speed for each pump-conductance pair
        effective_speeds = []
        for S_pump, S_cond in zip(pump_speeds, conductances):
            # S_eff = 1 / (1/S_pump + 1/S_conductance)
            S_eff = 1.0 / (1.0/S_pump + 1.0/S_cond)
            effective_speeds.append(S_eff)
        
        # Total effective speed (pumps in parallel)
        S_total = sum(effective_speeds)
        
        self.logger.info(f"Effective pumping speed: {S_total:.2e} m³/s "
                        f"(individual: {[f'{s:.2e}' for s in effective_speeds]})")
        
        return S_total
    
    def optimize_pump_configuration(self,
                                  chamber_volume: float,
                                  total_outgassing: float,
                                  available_pumps: List[str],
                                  conductance_elements: List[ConductanceElement],
                                  target_pressure: Optional[float] = None) -> Dict:
        """
        Optimize pump configuration for target pressure
        
        Args:
            chamber_volume: Chamber volume in m³
            total_outgassing: Total outgassing rate in Pa⋅m³/s
            available_pumps: List of available pump names
            conductance_elements: List of conductance elements
            target_pressure: Target pressure in Pa
            
        Returns:
            Optimization results
        """
        if target_pressure is None:
            target_pressure = self.target_pressure
        
        # Calculate conductances
        conductances = [self.calculate_conductance(elem) for elem in conductance_elements]
        
        best_config = None
        best_performance = 0
        
        # Test different pump combinations
        for pump_name in available_pumps:
            if pump_name not in self.pump_database:
                continue
                
            pump = self.pump_database[pump_name]
            
            # Single pump configuration
            pump_speeds = [pump.nominal_speed]
            S_eff = self.calculate_effective_pumping_speed(pump_speeds, [min(conductances)])
            
            # Achievable pressure: P = Q / S_eff
            achievable_pressure = total_outgassing / S_eff
            
            # Performance metric (lower pressure is better)
            performance = target_pressure / achievable_pressure if achievable_pressure > 0 else 0
            
            config = {
                'pump_name': pump_name,
                'pump_type': pump.pump_type.value,
                'effective_speed': S_eff,
                'achievable_pressure': achievable_pressure,
                'performance_ratio': performance,
                'meets_target': achievable_pressure <= target_pressure,
                'power_consumption': pump.power_consumption,
                'base_pressure_limit': pump.base_pressure
            }
            
            if performance > best_performance:
                best_performance = performance
                best_config = config
        
        # Multi-pump optimization (simplified)
        if len(available_pumps) > 1:
            # Try combination of turbo + ion pump
            turbo_pumps = [name for name in available_pumps 
                          if name in self.pump_database 
                          and self.pump_database[name].pump_type == PumpType.TURBOMOLECULAR]
            ion_pumps = [name for name in available_pumps 
                        if name in self.pump_database 
                        and self.pump_database[name].pump_type == PumpType.ION]
            
            if turbo_pumps and ion_pumps:
                turbo_pump = self.pump_database[turbo_pumps[0]]
                ion_pump = self.pump_database[ion_pumps[0]]
                
                combined_speeds = [turbo_pump.nominal_speed, ion_pump.nominal_speed]
                combined_conductances = [min(conductances), min(conductances)]
                
                S_eff_combined = self.calculate_effective_pumping_speed(
                    combined_speeds, combined_conductances
                )
                
                achievable_combined = total_outgassing / S_eff_combined
                performance_combined = target_pressure / achievable_combined if achievable_combined > 0 else 0
                
                if performance_combined > best_performance:
                    best_config = {
                        'pump_name': f"{turbo_pumps[0]} + {ion_pumps[0]}",
                        'pump_type': 'combined',
                        'effective_speed': S_eff_combined,
                        'achievable_pressure': achievable_combined,
                        'performance_ratio': performance_combined,
                        'meets_target': achievable_combined <= target_pressure,
                        'power_consumption': turbo_pump.power_consumption + ion_pump.power_consumption,
                        'base_pressure_limit': min(turbo_pump.base_pressure, ion_pump.base_pressure)
                    }
        
        optimization_results = {
            'best_configuration': best_config,
            'target_pressure': target_pressure,
            'total_outgassing': total_outgassing,
            'conductance_analysis': {
                'elements': [elem.geometry for elem in conductance_elements],
                'conductances': conductances,
                'limiting_conductance': min(conductances)
            }
        }
        
        self.logger.info(f"Optimal pump configuration: {best_config['pump_name']} "
                        f"(Performance ratio: {best_config['performance_ratio']:.1f})")
        
        return optimization_results
    
    def calculate_pumpdown_curve(self,
                               chamber_volume: float,
                               effective_pumping_speed: float,
                               outgassing_rate: float,
                               initial_pressure: float = 1e5,
                               time_points: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate pressure vs time during pumpdown
        
        Mathematical formulation:
        P(t) = (P_0 - P_ultimate) × exp(-S×t/V) + P_ultimate
        where P_ultimate = Q/S
        
        Args:
            chamber_volume: Chamber volume in m³
            effective_pumping_speed: Effective pumping speed in m³/s
            outgassing_rate: Outgassing rate in Pa⋅m³/s
            initial_pressure: Initial pressure in Pa
            time_points: Time points for calculation
            
        Returns:
            Pumpdown curve data
        """
        if time_points is None:
            # Default time points: 0 to 24 hours
            time_points = np.logspace(-2, 5, 100)  # seconds
        
        # Ultimate pressure limited by outgassing
        P_ultimate = outgassing_rate / effective_pumping_speed
        
        # Time constant
        tau = chamber_volume / effective_pumping_speed
        
        # Pressure vs time: P(t) = (P_0 - P_ultimate) × exp(-t/τ) + P_ultimate
        pressures = (initial_pressure - P_ultimate) * np.exp(-time_points / tau) + P_ultimate
        
        # Time to reach specific pressures
        target_pressures = [1e2, 1e0, 1e-3, 1e-6, 1e-9]  # Pa
        times_to_pressure = {}
        
        for P_target in target_pressures:
            if P_target > P_ultimate:
                # t = τ × ln((P_0 - P_ultimate)/(P_target - P_ultimate))
                t_target = tau * np.log((initial_pressure - P_ultimate) / (P_target - P_ultimate))
                times_to_pressure[f"{P_target:.0e}_Pa"] = t_target
            else:
                times_to_pressure[f"{P_target:.0e}_Pa"] = np.inf  # Cannot reach
        
        pumpdown_data = {
            'time_points': time_points,
            'pressures': pressures,
            'ultimate_pressure': P_ultimate,
            'time_constant': tau,
            'times_to_pressure': times_to_pressure,
            'initial_pressure': initial_pressure
        }
        
        self.logger.info(f"Pumpdown analysis: Ultimate pressure {P_ultimate:.2e} Pa, "
                        f"Time constant {tau:.1f} s")
        
        return pumpdown_data

def main():
    """Demonstration of advanced pumping system capabilities"""
    
    # Initialize pumping system
    pump_system = AdvancedPumpingSystem()
    
    # Define system geometry
    conductance_elements = [
        ConductanceElement('circular', 0.1, 0.5, 0),    # Main port: 10cm diameter, 50cm long
        ConductanceElement('elbow', 0.05, 0.2, 0),      # Elbow: 5cm diameter
        ConductanceElement('circular', 0.02, 1.0, 0)    # Pump port: 2cm diameter, 1m long
    ]
    
    # System parameters
    chamber_volume = 0.1  # m³
    total_outgassing = 1e-10  # Pa⋅m³/s
    available_pumps = ['turbo_high_speed', 'ion_pump_large', 'tsp_pump']
    
    # Optimize pump configuration
    optimization = pump_system.optimize_pump_configuration(
        chamber_volume, total_outgassing, available_pumps, conductance_elements
    )
    
    print("Advanced Pumping System Analysis:")
    print(f"Target pressure: ≤ {pump_system.target_pressure:.2e} Pa")
    
    best_config = optimization['best_configuration']
    print(f"\nOptimal Configuration:")
    print(f"Pump: {best_config['pump_name']}")
    print(f"Effective speed: {best_config['effective_speed']:.2e} m³/s")
    print(f"Achievable pressure: {best_config['achievable_pressure']:.2e} Pa")
    print(f"Meets target: {best_config['meets_target']}")
    print(f"Power consumption: {best_config['power_consumption']} W")
    
    # Calculate pumpdown curve
    pumpdown = pump_system.calculate_pumpdown_curve(
        chamber_volume, 
        best_config['effective_speed'],
        total_outgassing
    )
    
    print(f"\nPumpdown Analysis:")
    print(f"Ultimate pressure: {pumpdown['ultimate_pressure']:.2e} Pa")
    print(f"Time constant: {pumpdown['time_constant']:.1f} s")
    print(f"Time to 1e-6 Pa: {pumpdown['times_to_pressure'].get('1e-06_Pa', 'N/A')} s")

if __name__ == "__main__":
    main()
