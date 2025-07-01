"""
Precision Material Database
Validated material coefficients for environmental enclosure systems

Mathematical Formulations:
- α_Zerodur = 5×10⁻⁹ ± 0.5×10⁻⁹ K⁻¹  [Ultra-low expansion]
- α_Invar = 1.2×10⁻⁶ ± 0.1×10⁻⁶ K⁻¹  [Low expansion alloy]
- α_Silicon = 2.6×10⁻⁶ ± 0.1×10⁻⁶ K⁻¹  [Semiconductor grade]
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from enum import Enum

class MaterialCategory(Enum):
    """Material categories for environmental enclosure applications"""
    ULTRA_LOW_EXPANSION = "ultra_low_expansion"
    STRUCTURAL_METAL = "structural_metal"
    OPTICAL_MATERIAL = "optical_material"
    ELECTRONIC_MATERIAL = "electronic_material"
    THERMAL_INTERFACE = "thermal_interface"

@dataclass
class PrecisionMaterialData:
    """Comprehensive precision material properties"""
    name: str
    category: MaterialCategory
    
    # Thermal properties
    alpha_linear: float         # Linear thermal expansion coefficient (K⁻¹)
    alpha_uncertainty: float    # Uncertainty in α (K⁻¹)
    alpha_quadratic: float      # Quadratic coefficient (K⁻²)
    alpha_cubic: float          # Cubic coefficient (K⁻³)
    thermal_conductivity: float # W/(m·K)
    specific_heat: float        # J/(kg·K)
    
    # Mechanical properties
    density: float              # kg/m³
    elastic_modulus: float      # Pa
    poisson_ratio: float        # dimensionless
    yield_strength: float       # Pa
    
    # Environmental properties
    temperature_range: Tuple[float, float]  # Operating range (K)
    vacuum_compatibility: float             # Outgassing rate (Pa·m³/s·m²)
    surface_quality: float                  # Surface roughness factor (0-1)
    
    # Cost and availability
    relative_cost: float        # Relative cost factor (1.0 = baseline)
    availability: str           # "excellent", "good", "limited"
    
    # References and validation
    data_source: str            # Reference source
    validation_status: str      # "validated", "estimated", "literature"

class PrecisionMaterialDatabase:
    """
    Comprehensive database of precision materials for environmental enclosure systems
    
    Provides validated coefficients and properties for:
    - Ultra-low expansion materials
    - Structural metals and alloys
    - Optical materials
    - Electronic materials
    - Thermal interface materials
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.materials = self._initialize_database()
        self.logger.info(f"Precision material database initialized with {len(self.materials)} materials")
    
    def _initialize_database(self) -> Dict[str, PrecisionMaterialData]:
        """Initialize comprehensive material database"""
        
        materials = {}
        
        # Ultra-low expansion materials
        materials['zerodur'] = PrecisionMaterialData(
            name='Zerodur',
            category=MaterialCategory.ULTRA_LOW_EXPANSION,
            alpha_linear=5e-9,           # K⁻¹ (±0.5×10⁻⁹)
            alpha_uncertainty=0.5e-9,
            alpha_quadratic=1e-12,       # K⁻²
            alpha_cubic=0.0,
            thermal_conductivity=1.46,   # W/(m·K)
            specific_heat=808,           # J/(kg·K)
            density=2530,                # kg/m³
            elastic_modulus=91e9,        # Pa
            poisson_ratio=0.24,
            yield_strength=50e6,         # Pa (compression)
            temperature_range=(4.0, 573.0),  # K
            vacuum_compatibility=1e-12,  # Pa·m³/s·m²
            surface_quality=0.99,
            relative_cost=10.0,          # Expensive
            availability="good",
            data_source="Schott Technical Data",
            validation_status="validated"
        )
        
        materials['ule_glass'] = PrecisionMaterialData(
            name='ULE Glass',
            category=MaterialCategory.ULTRA_LOW_EXPANSION,
            alpha_linear=3e-8,           # K⁻¹ (±5×10⁻⁹)
            alpha_uncertainty=5e-9,
            alpha_quadratic=1e-11,       # K⁻²
            alpha_cubic=0.0,
            thermal_conductivity=1.31,   # W/(m·K)
            specific_heat=772,           # J/(kg·K)
            density=2210,                # kg/m³
            elastic_modulus=67.6e9,      # Pa
            poisson_ratio=0.17,
            yield_strength=48e6,         # Pa
            temperature_range=(4.0, 773.0),
            vacuum_compatibility=5e-13,
            surface_quality=0.98,
            relative_cost=8.0,
            availability="good",
            data_source="Corning Technical Data",
            validation_status="validated"
        )
        
        # Structural metals
        materials['invar'] = PrecisionMaterialData(
            name='Invar (Fe-36Ni)',
            category=MaterialCategory.STRUCTURAL_METAL,
            alpha_linear=1.2e-6,         # K⁻¹ (±0.1×10⁻⁶)
            alpha_uncertainty=0.1e-6,
            alpha_quadratic=2e-9,        # K⁻²
            alpha_cubic=0.0,
            thermal_conductivity=13.8,   # W/(m·K)
            specific_heat=515,           # J/(kg·K)
            density=8100,                # kg/m³
            elastic_modulus=141e9,       # Pa
            poisson_ratio=0.26,
            yield_strength=276e6,        # Pa
            temperature_range=(4.0, 773.0),
            vacuum_compatibility=1e-10,
            surface_quality=0.95,
            relative_cost=3.0,
            availability="excellent",
            data_source="ASM Handbook",
            validation_status="validated"
        )
        
        materials['super_invar'] = PrecisionMaterialData(
            name='Super Invar (Fe-32Ni-5Co)',
            category=MaterialCategory.STRUCTURAL_METAL,
            alpha_linear=0.5e-6,         # K⁻¹ (±0.1×10⁻⁶)
            alpha_uncertainty=0.1e-6,
            alpha_quadratic=1e-9,        # K⁻²
            alpha_cubic=0.0,
            thermal_conductivity=14.2,   # W/(m·K)
            specific_heat=460,           # J/(kg·K)
            density=8200,                # kg/m³
            elastic_modulus=144e9,       # Pa
            poisson_ratio=0.26,
            yield_strength=310e6,        # Pa
            temperature_range=(4.0, 673.0),
            vacuum_compatibility=5e-11,
            surface_quality=0.96,
            relative_cost=5.0,
            availability="good",
            data_source="Carpenter Technology",
            validation_status="validated"
        )
        
        # Optical materials
        materials['silicon'] = PrecisionMaterialData(
            name='Silicon (100)',
            category=MaterialCategory.OPTICAL_MATERIAL,
            alpha_linear=2.6e-6,         # K⁻¹ (±0.1×10⁻⁶)
            alpha_uncertainty=0.1e-6,
            alpha_quadratic=3.7e-9,      # K⁻²
            alpha_cubic=-2.0e-12,        # K⁻³
            thermal_conductivity=148,    # W/(m·K) at 300K
            specific_heat=705,           # J/(kg·K)
            density=2329,                # kg/m³
            elastic_modulus=130e9,       # Pa
            poisson_ratio=0.28,
            yield_strength=7e9,          # Pa (theoretical)
            temperature_range=(4.0, 1273.0),
            vacuum_compatibility=1e-12,
            surface_quality=0.99,
            relative_cost=2.0,
            availability="excellent",
            data_source="CRC Handbook",
            validation_status="validated"
        )
        
        materials['fused_silica'] = PrecisionMaterialData(
            name='Fused Silica',
            category=MaterialCategory.OPTICAL_MATERIAL,
            alpha_linear=5.5e-7,         # K⁻¹ (±0.1×10⁻⁷)
            alpha_uncertainty=0.1e-7,
            alpha_quadratic=8e-10,       # K⁻²
            alpha_cubic=0.0,
            thermal_conductivity=1.38,   # W/(m·K)
            specific_heat=703,           # J/(kg·K)
            density=2203,                # kg/m³
            elastic_modulus=73e9,        # Pa
            poisson_ratio=0.17,
            yield_strength=48e6,         # Pa
            temperature_range=(4.0, 1473.0),
            vacuum_compatibility=1e-13,
            surface_quality=0.98,
            relative_cost=1.5,
            availability="excellent",
            data_source="Corning Technical Data",
            validation_status="validated"
        )
        
        # Electronic materials
        materials['kovar'] = PrecisionMaterialData(
            name='Kovar (Fe-29Ni-17Co)',
            category=MaterialCategory.ELECTRONIC_MATERIAL,
            alpha_linear=5.2e-6,         # K⁻¹ (±0.2×10⁻⁶)
            alpha_uncertainty=0.2e-6,
            alpha_quadratic=3e-9,        # K⁻²
            alpha_cubic=0.0,
            thermal_conductivity=17.3,   # W/(m·K)
            specific_heat=460,           # J/(kg·K)
            density=8350,                # kg/m³
            elastic_modulus=138e9,       # Pa
            poisson_ratio=0.31,
            yield_strength=345e6,        # Pa
            temperature_range=(4.0, 723.0),
            vacuum_compatibility=5e-11,
            surface_quality=0.94,
            relative_cost=4.0,
            availability="good",
            data_source="CRS Holdings",
            validation_status="validated"
        )
        
        # High-performance structural materials
        materials['titanium_6al4v'] = PrecisionMaterialData(
            name='Ti-6Al-4V',
            category=MaterialCategory.STRUCTURAL_METAL,
            alpha_linear=8.6e-6,         # K⁻¹ (±0.2×10⁻⁶)
            alpha_uncertainty=0.2e-6,
            alpha_quadratic=1.5e-9,      # K⁻²
            alpha_cubic=0.0,
            thermal_conductivity=6.7,    # W/(m·K)
            specific_heat=553,           # J/(kg·K)
            density=4430,                # kg/m³
            elastic_modulus=113.8e9,     # Pa
            poisson_ratio=0.32,
            yield_strength=880e6,        # Pa
            temperature_range=(4.0, 873.0),
            vacuum_compatibility=1e-11,
            surface_quality=0.93,
            relative_cost=6.0,
            availability="excellent",
            data_source="TIMET Technical Data",
            validation_status="validated"
        )
        
        materials['aluminum_6061'] = PrecisionMaterialData(
            name='Al-6061-T6',
            category=MaterialCategory.STRUCTURAL_METAL,
            alpha_linear=23.6e-6,        # K⁻¹ (±0.5×10⁻⁶)
            alpha_uncertainty=0.5e-6,
            alpha_quadratic=5e-9,        # K⁻²
            alpha_cubic=0.0,
            thermal_conductivity=167,    # W/(m·K)
            specific_heat=896,           # J/(kg·K)
            density=2700,                # kg/m³
            elastic_modulus=68.9e9,      # Pa
            poisson_ratio=0.33,
            yield_strength=276e6,        # Pa
            temperature_range=(4.0, 773.0),
            vacuum_compatibility=5e-10,
            surface_quality=0.90,
            relative_cost=1.0,           # Baseline cost
            availability="excellent",
            data_source="Alcoa Technical Data",
            validation_status="validated"
        )
        
        return materials
    
    def get_material(self, material_name: str) -> Optional[PrecisionMaterialData]:
        """Get material data by name"""
        return self.materials.get(material_name.lower())
    
    def list_materials_by_category(self, category: MaterialCategory) -> List[str]:
        """List all materials in a specific category"""
        return [name for name, data in self.materials.items() 
                if data.category == category]
    
    def find_materials_by_expansion_range(self, 
                                        min_alpha: float, 
                                        max_alpha: float) -> List[str]:
        """Find materials within specified thermal expansion range"""
        matching_materials = []
        for name, data in self.materials.items():
            if min_alpha <= data.alpha_linear <= max_alpha:
                matching_materials.append(name)
        return matching_materials
    
    def compare_materials(self, material_names: List[str], 
                         properties: List[str]) -> Dict:
        """Compare specified properties across multiple materials"""
        comparison = {}
        
        for prop in properties:
            comparison[prop] = {}
            for name in material_names:
                material = self.get_material(name)
                if material and hasattr(material, prop):
                    comparison[prop][name] = getattr(material, prop)
        
        return comparison
    
    def get_thermal_expansion_uncertainty(self, 
                                        material_name: str,
                                        temperature_change: float) -> Tuple[float, float]:
        """
        Calculate thermal expansion uncertainty for given temperature change
        
        Args:
            material_name: Material name
            temperature_change: Temperature change (K)
            
        Returns:
            Tuple of (nominal_expansion, uncertainty)
        """
        material = self.get_material(material_name)
        if not material:
            raise ValueError(f"Material {material_name} not found")
        
        # Nominal expansion: ΔL/L = α × ΔT
        nominal_expansion = material.alpha_linear * temperature_change
        
        # Uncertainty: δ(ΔL/L) = δα × ΔT
        uncertainty = material.alpha_uncertainty * abs(temperature_change)
        
        return nominal_expansion, uncertainty

def main():
    """Demonstration of precision material database capabilities"""
    
    # Initialize database
    db = PrecisionMaterialDatabase()
    
    print("Precision Material Database")
    print("="*50)
    
    # List ultra-low expansion materials
    ule_materials = db.list_materials_by_category(MaterialCategory.ULTRA_LOW_EXPANSION)
    print(f"\nUltra-Low Expansion Materials:")
    for material in ule_materials:
        data = db.get_material(material)
        print(f"  {data.name}: α = {data.alpha_linear:.2e} ± {data.alpha_uncertainty:.2e} K⁻¹")
    
    # Compare thermal expansion coefficients
    materials_to_compare = ['zerodur', 'invar', 'silicon', 'aluminum_6061']
    comparison = db.compare_materials(
        materials_to_compare, 
        ['alpha_linear', 'alpha_uncertainty', 'thermal_conductivity']
    )
    
    print(f"\nMaterial Comparison:")
    for prop, values in comparison.items():
        print(f"{prop}:")
        for material, value in values.items():
            if prop == 'alpha_linear' or prop == 'alpha_uncertainty':
                print(f"  {material}: {value:.2e} K⁻¹")
            else:
                print(f"  {material}: {value:.1f} W/(m·K)")
    
    # Find materials for ±0.01 K stability (< 1 nm expansion for 10 cm length)
    target_strain = 1e-8  # 1 nm / 10 cm
    temp_change = 0.01    # K
    max_alpha = target_strain / temp_change  # 1e-6 K⁻¹
    
    suitable_materials = db.find_materials_by_expansion_range(0, max_alpha)
    print(f"\nMaterials suitable for ±0.01 K stability (α < {max_alpha:.2e} K⁻¹):")
    for material in suitable_materials:
        data = db.get_material(material)
        print(f"  {data.name}: α = {data.alpha_linear:.2e} K⁻¹")
    
    # Uncertainty analysis for Zerodur
    material_name = 'zerodur'
    length = 0.1  # m (10 cm)
    temp_change = 0.01  # K
    
    nominal_strain, strain_uncertainty = db.get_thermal_expansion_uncertainty(
        material_name, temp_change
    )
    
    nominal_expansion = nominal_strain * length
    expansion_uncertainty = strain_uncertainty * length
    
    print(f"\nZerodur Uncertainty Analysis (10 cm, ΔT = 0.01 K):")
    print(f"Nominal expansion: {nominal_expansion*1e9:.2f} nm")
    print(f"Uncertainty: ±{expansion_uncertainty*1e9:.2f} nm")
    print(f"Relative uncertainty: ±{strain_uncertainty/nominal_strain*100:.1f}%")

if __name__ == "__main__":
    main()
