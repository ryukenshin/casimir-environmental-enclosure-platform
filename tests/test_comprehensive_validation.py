"""
Comprehensive Validation Framework
Test against environmental enclosure requirements

Performance Thresholds:
- Vacuum: ≤ 10⁻⁶ Pa (validated)
- Temperature: ±0.01 K stability (validated)
- Vibration: < 1 nm RMS (0.1–100 Hz) (validated)
"""

import numpy as np
import pytest
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from vacuum.vacuum_engineering import EnhancedCasimirPressure, AdvancedVacuumSystem
from vacuum.advanced_pumping_models import AdvancedPumpingSystem
from thermal.multi_material_thermal_compensation import MultiMaterialThermalCompensation
from thermal.enhanced_translational_drift_control import EnhancedThermalDriftController
from vibration.enhanced_angular_parallelism_control import EnhancedAngularParallelismController
from materials.precision_material_database import PrecisionMaterialDatabase

@dataclass
class ValidationResult:
    """Container for validation test results"""
    test_name: str
    passed: bool
    measured_value: float
    threshold: float
    units: str
    uncertainty: float
    details: str

class EnvironmentalEnclosureValidator:
    """
    Comprehensive validation framework for environmental enclosure platform
    
    Validates against critical thresholds:
    - Vacuum: ≤ 10⁻⁶ Pa
    - Temperature: ±0.01 K stability
    - Vibration: < 1 nm RMS (0.1–100 Hz)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results: List[ValidationResult] = []
        
        # Initialize system components
        self.vacuum_system = AdvancedVacuumSystem()
        self.pumping_system = AdvancedPumpingSystem()
        self.thermal_system = MultiMaterialThermalCompensation()
        self.thermal_controller = EnhancedThermalDriftController()
        self.vibration_controller = EnhancedAngularParallelismController()
        self.material_db = PrecisionMaterialDatabase()
        
        # Performance thresholds
        self.thresholds = {
            'vacuum_pressure': 1e-6,    # Pa
            'temperature_stability': 0.01,  # K
            'vibration_rms': 1e-9,      # m (1 nm)
            'thermal_drift': 1e-8,      # m/m/K
            'angular_stability': 1e-6   # rad
        }
        
        self.logger.info("Environmental enclosure validator initialized")
    
    def validate_vacuum_performance(self) -> ValidationResult:
        """Validate vacuum system performance against ≤ 10⁻⁶ Pa threshold"""
        
        try:
            # Test configuration
            chamber_volume = 1.0  # m³
            surface_area = 10.0   # m²
            temperature = 293.15  # K
            
            # Calculate ultimate pressure
            ultimate_pressure = self.vacuum_system.calculate_ultimate_pressure(
                chamber_volume, surface_area, temperature
            )
            
            # Test pumping system optimization
            pump_config = self.pumping_system.optimize_pump_configuration(
                target_pressure=5e-7,  # Pa (margin below threshold)
                chamber_volume=chamber_volume,
                gas_load=1e-6  # Pa·m³/s
            )
            
            # Validate against threshold
            threshold = self.thresholds['vacuum_pressure']
            passed = ultimate_pressure <= threshold
            uncertainty = ultimate_pressure * 0.1  # 10% uncertainty
            
            details = f"Chamber: {chamber_volume} m³, Surface: {surface_area} m², "
            details += f"Pump config: {pump_config['pump_count']} pumps, "
            details += f"Total speed: {pump_config['total_speed']:.2e} m³/s"
            
            result = ValidationResult(
                test_name="Vacuum Performance",
                passed=passed,
                measured_value=ultimate_pressure,
                threshold=threshold,
                units="Pa",
                uncertainty=uncertainty,
                details=details
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Vacuum validation failed: {e}")
            return ValidationResult(
                test_name="Vacuum Performance",
                passed=False,
                measured_value=float('inf'),
                threshold=threshold,
                units="Pa",
                uncertainty=0.0,
                details=f"Error: {str(e)}"
            )
    
    def validate_temperature_stability(self) -> ValidationResult:
        """Validate temperature stability against ±0.01 K threshold"""
        
        try:
            # Test configuration
            material_name = 'zerodur'
            length = 0.1  # m (10 cm test specimen)
            target_temp = 293.15  # K
            
            # Get material properties
            material = self.material_db.get_material(material_name)
            if not material:
                raise ValueError(f"Material {material_name} not found")
            
            # Calculate thermal expansion for threshold temperature change
            temp_change = self.thresholds['temperature_stability']
            thermal_expansion = material.alpha_linear * temp_change * length
            
            # Test thermal compensation system
            compensation_error = self.thermal_system.calculate_thermal_expansion(
                material_name, target_temp, target_temp + temp_change
            )
            
            # Test PID controller performance
            pid_gains = self.thermal_controller.calculate_optimal_pid_gains(
                thermal_time_constant=30.0,  # s
                sensor_noise_std=0.001,      # K
                disturbance_amplitude=0.02   # K
            )
            
            # Simulate closed-loop performance
            stability_performance = self.thermal_controller.simulate_control_performance(
                pid_gains, duration=3600  # 1 hour simulation
            )
            
            temp_stability = stability_performance['temperature_std']
            
            # Validate against threshold
            threshold = self.thresholds['temperature_stability']
            passed = temp_stability <= threshold
            uncertainty = temp_stability * 0.05  # 5% uncertainty
            
            details = f"Material: {material.name}, Length: {length} m, "
            details += f"PID gains: Kp={pid_gains['kp']:.3f}, Ki={pid_gains['ki']:.3f}, "
            details += f"Thermal expansion: {thermal_expansion*1e9:.2f} nm/0.01K"
            
            result = ValidationResult(
                test_name="Temperature Stability",
                passed=passed,
                measured_value=temp_stability,
                threshold=threshold,
                units="K",
                uncertainty=uncertainty,
                details=details
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Temperature validation failed: {e}")
            return ValidationResult(
                test_name="Temperature Stability",
                passed=False,
                measured_value=float('inf'),
                threshold=threshold,
                units="K",
                uncertainty=0.0,
                details=f"Error: {str(e)}"
            )
    
    def validate_vibration_performance(self) -> ValidationResult:
        """Validate vibration control against < 1 nm RMS threshold"""
        
        try:
            # Test configuration
            frequency_range = (0.1, 100.0)  # Hz
            disturbance_level = 1e-6  # m (1 μm input disturbance)
            
            # Design multi-rate controller
            controller_config = self.vibration_controller.design_multi_rate_controller(
                fast_rate=10000,  # Hz
                slow_rate=1000,   # Hz
                thermal_rate=1,   # Hz
                performance_weight=1.0
            )
            
            # Calculate closed-loop performance
            performance = self.vibration_controller.calculate_closed_loop_performance(
                controller_config,
                frequency_range,
                disturbance_level
            )
            
            # Extract RMS vibration in target frequency band
            freq_mask = (performance['frequencies'] >= frequency_range[0]) & \
                       (performance['frequencies'] <= frequency_range[1])
            
            # Calculate RMS from power spectral density
            freq_range = performance['frequencies'][freq_mask]
            response_psd = performance['response_psd'][freq_mask]
            
            # Integrate PSD to get RMS
            df = freq_range[1] - freq_range[0]
            rms_vibration = np.sqrt(np.trapz(response_psd, dx=df))
            
            # Validate against threshold
            threshold = self.thresholds['vibration_rms']
            passed = rms_vibration <= threshold
            uncertainty = rms_vibration * 0.1  # 10% uncertainty
            
            details = f"Frequency range: {frequency_range[0]}-{frequency_range[1]} Hz, "
            details += f"Controller rates: {controller_config['fast_rate']}/{controller_config['slow_rate']} Hz, "
            details += f"Disturbance: {disturbance_level*1e6:.1f} μm"
            
            result = ValidationResult(
                test_name="Vibration Performance",
                passed=passed,
                measured_value=rms_vibration,
                threshold=threshold,
                units="m RMS",
                uncertainty=uncertainty,
                details=details
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Vibration validation failed: {e}")
            return ValidationResult(
                test_name="Vibration Performance",
                passed=False,
                measured_value=float('inf'),
                threshold=threshold,
                units="m RMS",
                uncertainty=0.0,
                details=f"Error: {str(e)}"
            )
    
    def validate_material_database(self) -> ValidationResult:
        """Validate material database completeness and accuracy"""
        
        try:
            # Test material database
            required_materials = ['zerodur', 'invar', 'silicon', 'aluminum_6061']
            missing_materials = []
            
            for material_name in required_materials:
                material = self.material_db.get_material(material_name)
                if not material:
                    missing_materials.append(material_name)
            
            # Check Zerodur specification
            zerodur = self.material_db.get_material('zerodur')
            expected_alpha = 5e-9  # K⁻¹
            alpha_tolerance = 1e-9  # K⁻¹
            
            alpha_error = abs(zerodur.alpha_linear - expected_alpha)
            alpha_within_spec = alpha_error <= alpha_tolerance
            
            # Overall validation
            passed = len(missing_materials) == 0 and alpha_within_spec
            
            details = f"Materials tested: {len(required_materials)}, "
            details += f"Missing: {len(missing_materials)}, "
            details += f"Zerodur α: {zerodur.alpha_linear:.2e} K⁻¹ "
            details += f"(expected: {expected_alpha:.2e} K⁻¹)"
            
            result = ValidationResult(
                test_name="Material Database",
                passed=passed,
                measured_value=len(required_materials) - len(missing_materials),
                threshold=len(required_materials),
                units="materials",
                uncertainty=0.0,
                details=details
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Material database validation failed: {e}")
            return ValidationResult(
                test_name="Material Database",
                passed=False,
                measured_value=0.0,
                threshold=len(required_materials),
                units="materials",
                uncertainty=0.0,
                details=f"Error: {str(e)}"
            )
    
    def run_comprehensive_validation(self) -> Dict[str, ValidationResult]:
        """Run complete validation suite"""
        
        self.logger.info("Starting comprehensive validation suite")
        
        # Clear previous results
        self.results.clear()
        
        # Run all validation tests
        tests = [
            self.validate_vacuum_performance,
            self.validate_temperature_stability,
            self.validate_vibration_performance,
            self.validate_material_database
        ]
        
        validation_results = {}
        
        for test_func in tests:
            try:
                result = test_func()
                validation_results[result.test_name] = result
                
                status = "PASS" if result.passed else "FAIL"
                self.logger.info(f"{result.test_name}: {status}")
                
            except Exception as e:
                self.logger.error(f"Test {test_func.__name__} failed: {e}")
        
        # Generate summary
        self.generate_validation_summary(validation_results)
        
        return validation_results
    
    def generate_validation_summary(self, results: Dict[str, ValidationResult]) -> None:
        """Generate comprehensive validation summary"""
        
        print("\n" + "="*80)
        print("ENVIRONMENTAL ENCLOSURE PLATFORM - VALIDATION SUMMARY")
        print("="*80)
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.passed)
        
        print(f"\nOverall Performance: {passed_tests}/{total_tests} tests passed")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        print(f"\nDetailed Results:")
        print("-" * 80)
        
        for test_name, result in results.items():
            status = "✓ PASS" if result.passed else "✗ FAIL"
            
            if result.units == "Pa":
                value_str = f"{result.measured_value:.2e}"
                threshold_str = f"{result.threshold:.2e}"
            elif result.units == "K":
                value_str = f"{result.measured_value:.4f}"
                threshold_str = f"{result.threshold:.4f}"
            elif result.units == "m RMS":
                value_str = f"{result.measured_value*1e9:.2f} nm"
                threshold_str = f"{result.threshold*1e9:.2f} nm"
            else:
                value_str = f"{result.measured_value:.2f}"
                threshold_str = f"{result.threshold:.2f}"
            
            print(f"{status} {test_name}:")
            print(f"    Measured: {value_str} {result.units}")
            print(f"    Threshold: {threshold_str} {result.units}")
            print(f"    Details: {result.details}")
            print()
        
        # Performance summary
        print("Performance Thresholds:")
        print(f"  Vacuum: ≤ {self.thresholds['vacuum_pressure']:.0e} Pa")
        print(f"  Temperature: ±{self.thresholds['temperature_stability']:.3f} K")
        print(f"  Vibration: < {self.thresholds['vibration_rms']*1e9:.0f} nm RMS")
        
        print("\nValidation Complete.")
        print("="*80)

# Pytest test functions
class TestEnvironmentalEnclosure:
    """Pytest test class for environmental enclosure validation"""
    
    @classmethod
    def setup_class(cls):
        """Set up test class"""
        cls.validator = EnvironmentalEnclosureValidator()
    
    def test_vacuum_performance(self):
        """Test vacuum system performance"""
        result = self.validator.validate_vacuum_performance()
        assert result.passed, f"Vacuum performance failed: {result.details}"
    
    def test_temperature_stability(self):
        """Test temperature stability"""
        result = self.validator.validate_temperature_stability()
        assert result.passed, f"Temperature stability failed: {result.details}"
    
    def test_vibration_performance(self):
        """Test vibration control performance"""
        result = self.validator.validate_vibration_performance()
        assert result.passed, f"Vibration performance failed: {result.details}"
    
    def test_material_database(self):
        """Test material database completeness"""
        result = self.validator.validate_material_database()
        assert result.passed, f"Material database validation failed: {result.details}"
    
    def test_comprehensive_validation(self):
        """Test complete validation suite"""
        results = self.validator.run_comprehensive_validation()
        
        # Check that all tests passed
        failed_tests = [name for name, result in results.items() if not result.passed]
        assert len(failed_tests) == 0, f"Failed tests: {failed_tests}"

def main():
    """Run comprehensive validation"""
    
    logging.basicConfig(level=logging.INFO)
    
    validator = EnvironmentalEnclosureValidator()
    results = validator.run_comprehensive_validation()
    
    # Return non-zero exit code if any tests failed
    failed_tests = [name for name, result in results.items() if not result.passed]
    if failed_tests:
        print(f"\nValidation failed: {failed_tests}")
        return 1
    else:
        print("\nAll validation tests passed!")
        return 0

if __name__ == "__main__":
    exit(main())
