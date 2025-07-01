"""
Casimir Environmental Enclosure Platform
Advanced mathematical formulations for UHV, temperature, and vibration control

Key Modules:
- vacuum: Enhanced Casimir pressure calculations and UHV systems
- thermal: Multi-material thermal compensation and PID control
- vibration: Multi-rate control architecture for nanometer stability
- materials: Precision material database with validated coefficients
"""

__version__ = "1.0.0"
__author__ = "Environmental Enclosure Research Team"

# Import key classes for convenient access
from .vacuum.vacuum_engineering import EnhancedCasimirPressure, AdvancedVacuumSystem
from .vacuum.advanced_pumping_models import AdvancedPumpingSystem
from .thermal.multi_material_thermal_compensation import ThermalCompensationSystem
from .thermal.enhanced_translational_drift_control import EnhancedThermalDriftController
from .vibration.enhanced_angular_parallelism_control import EnhancedAngularParallelismController
from .materials.precision_material_database import PrecisionMaterialDatabase

__all__ = [
    'EnhancedCasimirPressure',
    'AdvancedVacuumSystem',
    'AdvancedPumpingSystem',
    'ThermalCompensationSystem',
    'EnhancedThermalDriftController',
    'EnhancedAngularParallelismController',
    'PrecisionMaterialDatabase'
]
