"""
Multi-Material Thermal Compensation System
Advanced thermal expansion control with material-specific coefficients for ±0.01 K stability

Mathematical Formulations:
- L(T) = L₀ × [1 + α₁ΔT + α₂(ΔT)² + α₃(ΔT)³]  [Nonlinear thermal expansion]
- f_thermal(T, material) = 1 + α₁ × ΔT + α₂ × (ΔT)²  [Material-specific function]
- Precision Material Database with validated coefficients
"""

import numpy as np
import scipy.constants as const
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from enum import Enum

class MaterialType(Enum):
    """Types of precision materials"""
    ULTRA_LOW_EXPANSION = "ultra_low_expansion"
    STRUCTURAL = "structural"
    OPTICAL = "optical"
    ELECTRONIC = "electronic"

@dataclass
class ThermalMaterialProperties:
    """Comprehensive thermal material properties"""
    name: str
    material_type: MaterialType
    alpha_1: float              # Linear expansion coefficient (K⁻¹)
    alpha_1_uncertainty: float  # Uncertainty in α₁ (K⁻¹)
    alpha_2: float              # Quadratic coefficient (K⁻²)
    alpha_3: float              # Cubic coefficient (K⁻³)
    thermal_conductivity: float # W/(m·K)
    specific_heat: float        # J/(kg·K)
    density: float              # kg/m³
    elastic_modulus: float      # Pa
    temperature_range: Tuple[float, float]  # Valid temperature range (K)
    quality_factor: float       # Surface/bulk quality (0-1)

@dataclass
class ThermalCompensationElement:
    """Individual thermal compensation element"""
    element_id: str
    material: str
    length_nominal: float       # m
    cross_section: float        # m²
    orientation: np.ndarray     # Unit vector
    constraint_stiffness: float # N/m
    current_temperature: float  # K
    reference_temperature: float # K

class MultiMaterialThermalCompensation:
    """
    Advanced multi-material thermal compensation system
    
    Implements:
    - Nonlinear thermal expansion models
    - Material-specific coefficient database
    - Real-time compensation calculations
    - Multi-dimensional thermal coupling
    """
    
    def __init__(self, reference_temperature: float = 293.15):
        self.logger = logging.getLogger(__name__)
        self.reference_temperature = reference_temperature  # K
        self.target_stability = 0.01  # K
        self.material_database = self._initialize_material_database()
        
    def _initialize_material_database(self) -> Dict[str, ThermalMaterialProperties]:
        """Initialize precision material database with validated coefficients"""
        return {
            'zerodur': ThermalMaterialProperties(
                name='Zerodur',
                material_type=MaterialType.ULTRA_LOW_EXPANSION,
                alpha_1=5e-9,           # K⁻¹ (ultra-low expansion)
                alpha_1_uncertainty=0.5e-9,  # K⁻¹
                alpha_2=1e-12,          # K⁻²
                alpha_3=0.0,            # K⁻³
                thermal_conductivity=1.46,    # W/(m·K)
                specific_heat=808,            # J/(kg·K)
                density=2530,                 # kg/m³
                elastic_modulus=91e9,         # Pa
                temperature_range=(4.0, 573.0),  # K
                quality_factor=0.99
            ),
            'invar': ThermalMaterialProperties(
                name='Invar',
                material_type=MaterialType.STRUCTURAL,
                alpha_1=1.2e-6,         # K⁻¹
                alpha_1_uncertainty=0.1e-6,  # K⁻¹
                alpha_2=2e-9,           # K⁻²
                alpha_3=0.0,            # K⁻³
                thermal_conductivity=13.8,    # W/(m·K)
                specific_heat=515,            # J/(kg·K)
                density=8100,                 # kg/m³
                elastic_modulus=141e9,        # Pa
                temperature_range=(4.0, 773.0),  # K
                quality_factor=0.95
            ),
            'silicon': ThermalMaterialProperties(
                name='Silicon',
                material_type=MaterialType.OPTICAL,
                alpha_1=2.6e-6,         # K⁻¹
                alpha_1_uncertainty=0.1e-6,  # K⁻¹
                alpha_2=3.7e-9,         # K⁻²
                alpha_3=-2.0e-12,       # K⁻³
                thermal_conductivity=148,     # W/(m·K)
                specific_heat=705,            # J/(kg·K)
                density=2329,                 # kg/m³
                elastic_modulus=130e9,        # Pa
                temperature_range=(4.0, 1273.0), # K
                quality_factor=0.98
            ),
            'ule_glass': ThermalMaterialProperties(
                name='ULE Glass',
                material_type=MaterialType.OPTICAL,
                alpha_1=3e-8,           # K⁻¹
                alpha_1_uncertainty=5e-9,    # K⁻¹
                alpha_2=1e-11,          # K⁻²
                alpha_3=0.0,            # K⁻³
                thermal_conductivity=1.31,    # W/(m·K)
                specific_heat=772,            # J/(kg·K)
                density=2210,                 # kg/m³
                elastic_modulus=67.6e9,       # Pa
                temperature_range=(4.0, 773.0),  # K
                quality_factor=0.97
            ),
            'titanium_6al4v': ThermalMaterialProperties(
                name='Ti-6Al-4V',
                material_type=MaterialType.STRUCTURAL,
                alpha_1=8.6e-6,         # K⁻¹
                alpha_1_uncertainty=0.2e-6,  # K⁻¹
                alpha_2=1.5e-9,         # K⁻²
                alpha_3=0.0,            # K⁻³
                thermal_conductivity=6.7,     # W/(m·K)
                specific_heat=553,            # J/(kg·K)
                density=4430,                 # kg/m³
                elastic_modulus=113.8e9,      # Pa
                temperature_range=(4.0, 873.0),  # K
                quality_factor=0.92
            ),
            'aluminum_6061': ThermalMaterialProperties(
                name='Al-6061',
                material_type=MaterialType.STRUCTURAL,
                alpha_1=23.6e-6,        # K⁻¹
                alpha_1_uncertainty=0.5e-6,  # K⁻¹
                alpha_2=5e-9,           # K⁻²
                alpha_3=0.0,            # K⁻³
                thermal_conductivity=167,     # W/(m·K)
                specific_heat=896,            # J/(kg·K)
                density=2700,                 # kg/m³
                elastic_modulus=68.9e9,       # Pa
                temperature_range=(4.0, 773.0),  # K
                quality_factor=0.88
            )
        }
    
    def calculate_thermal_expansion(self,
                                  material_name: str,
                                  length_nominal: float,
                                  temperature: float,
                                  reference_temp: Optional[float] = None) -> Dict:
        """
        Calculate thermal expansion using nonlinear model
        
        Mathematical formulation:
        L(T) = L₀ × [1 + α₁ΔT + α₂(ΔT)² + α₃(ΔT)³]
        
        Args:
            material_name: Material name from database
            length_nominal: Nominal length at reference temperature (m)
            temperature: Current temperature (K)
            reference_temp: Reference temperature (K)
            
        Returns:
            Thermal expansion analysis
        """
        if material_name not in self.material_database:
            raise ValueError(f"Material {material_name} not in database")
        
        if reference_temp is None:
            reference_temp = self.reference_temperature
            
        material = self.material_database[material_name]
        
        # Check temperature range
        if not (material.temperature_range[0] <= temperature <= material.temperature_range[1]):
            self.logger.warning(f"Temperature {temperature:.1f} K outside valid range "
                               f"{material.temperature_range} for {material_name}")
        
        # Temperature difference
        delta_T = temperature - reference_temp
        
        # Nonlinear thermal expansion:
        # L(T) = L₀ × [1 + α₁ΔT + α₂(ΔT)² + α₃(ΔT)³]
        expansion_factor = (1.0 + 
                           material.alpha_1 * delta_T + 
                           material.alpha_2 * delta_T**2 + 
                           material.alpha_3 * delta_T**3)
        
        length_expanded = length_nominal * expansion_factor
        absolute_expansion = length_expanded - length_nominal
        
        # Uncertainty analysis
        # δL/L ≈ δα₁ × ΔT (dominant term for small ΔT)
        relative_uncertainty = material.alpha_1_uncertainty * abs(delta_T)
        absolute_uncertainty = length_nominal * relative_uncertainty
        
        # Thermal stress (if constrained)
        thermal_strain = expansion_factor - 1.0
        thermal_stress = material.elastic_modulus * thermal_strain
        
        expansion_analysis = {
            'material': material_name,
            'length_nominal': length_nominal,
            'length_expanded': length_expanded,
            'absolute_expansion': absolute_expansion,
            'relative_expansion': absolute_expansion / length_nominal,
            'expansion_factor': expansion_factor,
            'temperature': temperature,
            'delta_temperature': delta_T,
            'thermal_strain': thermal_strain,
            'thermal_stress': thermal_stress,
            'absolute_uncertainty': absolute_uncertainty,
            'relative_uncertainty': relative_uncertainty,
            'coefficients_used': {
                'alpha_1': material.alpha_1,
                'alpha_2': material.alpha_2,
                'alpha_3': material.alpha_3
            }
        }
        
        self.logger.debug(f"Thermal expansion ({material_name}): "
                         f"{absolute_expansion*1e9:.2f} nm @ ΔT={delta_T:.2f} K")
        
        return expansion_analysis
    
    def calculate_material_thermal_function(self,
                                          material_name: str,
                                          temperature: float,
                                          reference_temp: Optional[float] = None) -> float:
        """
        Calculate material-specific thermal function
        
        Mathematical formulation:
        f_thermal(T, material) = 1 + α₁ × ΔT + α₂ × (ΔT)²
        
        Args:
            material_name: Material name
            temperature: Current temperature (K)
            reference_temp: Reference temperature (K)
            
        Returns:
            Thermal function value
        """
        if reference_temp is None:
            reference_temp = self.reference_temperature
            
        if material_name not in self.material_database:
            raise ValueError(f"Material {material_name} not in database")
            
        material = self.material_database[material_name]
        delta_T = temperature - reference_temp
        
        # f_thermal(T, material) = 1 + α₁ × ΔT + α₂ × (ΔT)²
        f_thermal = 1.0 + material.alpha_1 * delta_T + material.alpha_2 * delta_T**2
        
        return f_thermal
    
    def design_thermal_compensation_system(self,
                                         target_displacement: float,
                                         temperature_range: Tuple[float, float],
                                         available_materials: List[str],
                                         length_constraint: float) -> Dict:
        """
        Design optimal thermal compensation system
        
        Minimizes thermal displacement over temperature range using
        multiple materials with different expansion coefficients.
        
        Args:
            target_displacement: Maximum allowed displacement (m)
            temperature_range: Operating temperature range (K)
            available_materials: List of available materials
            length_constraint: Total length constraint (m)
            
        Returns:
            Optimal compensation design
        """
        T_min, T_max = temperature_range
        T_mid = (T_min + T_max) / 2.0
        
        best_design = None
        best_performance = float('inf')
        
        # Test different material combinations
        for primary_material in available_materials:
            if primary_material not in self.material_database:
                continue
                
            for compensator_material in available_materials:
                if compensator_material == primary_material:
                    continue
                if compensator_material not in self.material_database:
                    continue
                
                # Try different length ratios
                for primary_fraction in np.linspace(0.5, 0.95, 10):
                    L_primary = length_constraint * primary_fraction
                    L_compensator = length_constraint * (1 - primary_fraction)
                    
                    # Calculate displacement over temperature range
                    displacements = []
                    temperatures = np.linspace(T_min, T_max, 20)
                    
                    for T in temperatures:
                        # Primary element expansion
                        primary_expansion = self.calculate_thermal_expansion(
                            primary_material, L_primary, T, self.reference_temperature
                        )
                        
                        # Compensator element expansion (opposite direction)
                        compensator_expansion = self.calculate_thermal_expansion(
                            compensator_material, L_compensator, T, self.reference_temperature
                        )
                        
                        # Net displacement
                        net_displacement = (primary_expansion['absolute_expansion'] - 
                                          compensator_expansion['absolute_expansion'])
                        displacements.append(abs(net_displacement))
                    
                    # Performance metric: maximum displacement
                    max_displacement = max(displacements)
                    
                    if max_displacement < best_performance:
                        best_performance = max_displacement
                        best_design = {
                            'primary_material': primary_material,
                            'compensator_material': compensator_material,
                            'primary_length': L_primary,
                            'compensator_length': L_compensator,
                            'primary_fraction': primary_fraction,
                            'max_displacement': max_displacement,
                            'meets_target': max_displacement <= target_displacement,
                            'displacement_profile': displacements,
                            'temperature_points': temperatures.tolist(),
                            'performance_margin': target_displacement / max_displacement if max_displacement > 0 else float('inf')
                        }
        
        if best_design is None:
            raise ValueError("No valid compensation design found")
        
        self.logger.info(f"Optimal thermal compensation: "
                        f"{best_design['primary_material']} + {best_design['compensator_material']} "
                        f"(Max displacement: {best_design['max_displacement']*1e9:.1f} nm)")
        
        return best_design
    
    def calculate_multi_element_system(self,
                                     elements: List[ThermalCompensationElement],
                                     temperature_field: Dict[str, float]) -> Dict:
        """
        Calculate multi-element thermal compensation system
        
        Args:
            elements: List of compensation elements
            temperature_field: Temperature for each element {element_id: temperature}
            
        Returns:
            System-level analysis
        """
        element_analyses = {}
        total_displacement = np.zeros(3)  # 3D displacement vector
        total_stress_energy = 0.0
        
        for element in elements:
            if element.element_id not in temperature_field:
                self.logger.warning(f"No temperature data for element {element.element_id}")
                continue
                
            temperature = temperature_field[element.element_id]
            
            # Individual element analysis
            expansion = self.calculate_thermal_expansion(
                element.material,
                element.length_nominal,
                temperature,
                element.reference_temperature
            )
            
            # 3D displacement contribution
            displacement_vector = (expansion['absolute_expansion'] * 
                                 element.orientation)
            total_displacement += displacement_vector
            
            # Stress energy (if constrained)
            if element.constraint_stiffness > 0:
                constraint_force = element.constraint_stiffness * expansion['absolute_expansion']
                stress_energy = 0.5 * constraint_force * expansion['absolute_expansion']
                total_stress_energy += stress_energy
            else:
                constraint_force = 0.0
                stress_energy = 0.0
            
            element_analyses[element.element_id] = {
                'expansion_analysis': expansion,
                'displacement_vector': displacement_vector,
                'constraint_force': constraint_force,
                'stress_energy': stress_energy,
                'temperature': temperature
            }
        
        # System-level metrics
        total_displacement_magnitude = np.linalg.norm(total_displacement)
        
        system_analysis = {
            'element_analyses': element_analyses,
            'total_displacement_vector': total_displacement,
            'total_displacement_magnitude': total_displacement_magnitude,
            'total_stress_energy': total_stress_energy,
            'stability_achieved': total_displacement_magnitude <= (self.target_stability * 1e-6),  # Convert K to m equivalent
            'number_of_elements': len(elements),
            'temperature_range': {
                'min': min(temperature_field.values()),
                'max': max(temperature_field.values()),
                'span': max(temperature_field.values()) - min(temperature_field.values())
            }
        }
        
        self.logger.info(f"Multi-element system: Total displacement {total_displacement_magnitude*1e9:.2f} nm "
                        f"(Stability: {'ACHIEVED' if system_analysis['stability_achieved'] else 'NOT ACHIEVED'})")
        
        return system_analysis

def main():
    """Demonstration of multi-material thermal compensation capabilities"""
    
    # Initialize thermal compensation system
    thermal_system = MultiMaterialThermalCompensation()
    
    print("Multi-Material Thermal Compensation Analysis")
    print("="*50)
    
    # Example 1: Single material thermal expansion
    print("\n1. Single Material Analysis:")
    expansion = thermal_system.calculate_thermal_expansion(
        'zerodur', 0.1, 293.25, 293.15  # 10cm Zerodur, 0.1K temperature change
    )
    print(f"Material: {expansion['material']}")
    print(f"Temperature change: {expansion['delta_temperature']:.3f} K")
    print(f"Absolute expansion: {expansion['absolute_expansion']*1e9:.2f} nm")
    print(f"Relative expansion: {expansion['relative_expansion']:.2e}")
    
    # Example 2: Material comparison
    print("\n2. Material Comparison (ΔT = 0.01 K):")
    materials = ['zerodur', 'invar', 'silicon', 'ule_glass']
    length = 0.1  # m
    delta_T = 0.01  # K
    
    for mat in materials:
        expansion = thermal_system.calculate_thermal_expansion(
            mat, length, thermal_system.reference_temperature + delta_T
        )
        print(f"{mat:12}: {expansion['absolute_expansion']*1e9:6.2f} nm")
    
    # Example 3: Optimal compensation design
    print("\n3. Thermal Compensation Design:")
    design = thermal_system.design_thermal_compensation_system(
        target_displacement=1e-9,  # 1 nm
        temperature_range=(293.0, 294.0),  # 1 K range
        available_materials=['zerodur', 'invar', 'silicon'],
        length_constraint=0.2  # 20 cm total
    )
    
    print(f"Primary: {design['primary_material']} ({design['primary_length']*100:.1f} cm)")
    print(f"Compensator: {design['compensator_material']} ({design['compensator_length']*100:.1f} cm)")
    print(f"Max displacement: {design['max_displacement']*1e9:.2f} nm")
    print(f"Meets target: {design['meets_target']}")
    print(f"Performance margin: {design['performance_margin']:.1f}x")
    
    # Example 4: Multi-element system
    print("\n4. Multi-Element System Analysis:")
    elements = [
        ThermalCompensationElement(
            'element_1', 'zerodur', 0.05, 1e-4, 
            np.array([1, 0, 0]), 1e6, 293.16, 293.15
        ),
        ThermalCompensationElement(
            'element_2', 'invar', 0.03, 5e-5,
            np.array([0, 1, 0]), 5e5, 293.14, 293.15
        )
    ]
    
    temperature_field = {
        'element_1': 293.16,  # +0.01 K
        'element_2': 293.14   # -0.01 K
    }
    
    system = thermal_system.calculate_multi_element_system(elements, temperature_field)
    print(f"Total displacement: {system['total_displacement_magnitude']*1e9:.2f} nm")
    print(f"Stability achieved: {system['stability_achieved']}")
    print(f"Temperature span: {system['temperature_range']['span']:.3f} K")

if __name__ == "__main__":
    main()
