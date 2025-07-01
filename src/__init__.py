"""
Casimir Environmental Enclosure Platform
Advanced mathematical formulations for UHV, temperature, and vibration control

Key Modules:
- vacuum: Enhanced Casimir pressure calculations and UHV systems
- thermal: Multi-material thermal compensation and PID control
- vibration: Multi-rate control architecture for nanometer stability
- materials: Precision material database with validated coefficients
- digital_twin: Advanced multi-physics coupling and state estimation

Enhanced Digital Twin Framework v2.0:
- Multi-physics coupling with physics-based cross-domain interactions
- Advanced Kalman filtering with adaptive sigma point optimization
- Enhanced uncertainty quantification with second-order Sobol analysis
- Advanced H∞ robust control with quantified stability margins >50%
- Enhanced model predictive control with probabilistic constraints
- Digital twin fidelity metrics with multi-domain assessment (R² ≥ 0.995)
"""

__version__ = "2.0.0"
__author__ = "Environmental Enclosure Research Team"

# Import key classes for convenient access
from .vacuum.vacuum_engineering import EnhancedCasimirPressure, AdvancedVacuumSystem
from .vacuum.advanced_pumping_models import AdvancedPumpingSystem
from .thermal.multi_material_thermal_compensation import MultiMaterialThermalCompensation
from .thermal.enhanced_translational_drift_control import EnhancedThermalDriftController
from .vibration.enhanced_angular_parallelism_control import MultiRateVibrationController
from .materials.precision_material_database import PrecisionMaterialDatabase

# Digital Twin Framework
from .digital_twin.digital_twin_core import DigitalTwinCore, DigitalTwinConfiguration
from .digital_twin.multi_physics_coupling import EnhancedMultiPhysicsCoupling
from .digital_twin.advanced_kalman_filter import AdvancedUnscentedKalmanFilter
from .digital_twin.uncertainty_quantification import EnhancedUncertaintyQuantification
from .digital_twin.robust_control import AdvancedHInfinityController
from .digital_twin.predictive_control import EnhancedModelPredictiveController

__all__ = [
    # Environmental Control Systems
    'EnhancedCasimirPressure',
    'AdvancedVacuumSystem', 
    'AdvancedPumpingSystem',
    'MultiMaterialThermalCompensation',
    'EnhancedThermalDriftController',
    'MultiRateVibrationController',
    'PrecisionMaterialDatabase',
    
    # Digital Twin Framework
    'DigitalTwinCore',
    'DigitalTwinConfiguration',
    'EnhancedMultiPhysicsCoupling',
    'AdvancedUnscentedKalmanFilter',
    'EnhancedUncertaintyQuantification', 
    'AdvancedHInfinityController',
    'EnhancedModelPredictiveController'
]
