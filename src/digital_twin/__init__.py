"""
Digital Twin Framework
Advanced multi-physics coupling and state estimation for environmental enclosure systems

Key Components:
- Enhanced coupling matrix formulations
- Advanced Kalman filtering with UKF
- Gelman-Rubin convergence diagnostics
- H∞ robust control with robustness margins
- Sobol sensitivity analysis with second-order interactions
- Multi-physics state evolution
- Predictive control with constraint tightening
- Digital twin fidelity metrics
"""

from .multi_physics_coupling import EnhancedMultiPhysicsCoupling
from .advanced_kalman_filter import AdvancedUnscentedKalmanFilter
from .uncertainty_quantification import EnhancedUncertaintyQuantification
from .robust_control import AdvancedHInfinityController
from .predictive_control import EnhancedModelPredictiveController
from .digital_twin_core import DigitalTwinCore

__all__ = [
    'EnhancedMultiPhysicsCoupling',
    'AdvancedUnscentedKalmanFilter', 
    'EnhancedUncertaintyQuantification',
    'AdvancedHInfinityController',
    'EnhancedModelPredictiveController',
    'DigitalTwinCore'
]
