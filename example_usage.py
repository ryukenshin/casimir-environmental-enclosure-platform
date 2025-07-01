#!/usr/bin/env python3
"""
Example Usage: Casimir Environmental Enclosure Platform
Demonstrates key capabilities and validation against performance thresholds
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from vacuum.vacuum_engineering import EnhancedCasimirPressure, AdvancedVacuumSystem
from vacuum.advanced_pumping_models import AdvancedPumpingSystem
from thermal.multi_material_thermal_compensation import MultiMaterialThermalCompensation
from thermal.enhanced_translational_drift_control import EnhancedThermalDriftController
from vibration.enhanced_angular_parallelism_control import MultiRateVibrationController
from materials.precision_material_database import PrecisionMaterialDatabase

def main():
    """Demonstrate environmental enclosure platform capabilities"""
    
    print("Casimir Environmental Enclosure Platform - Example Usage")
    print("=" * 60)
    
    # 1. Material Database Example
    print("\n1. PRECISION MATERIAL DATABASE")
    print("-" * 30)
    
    material_db = PrecisionMaterialDatabase()
    
    # List ultra-low expansion materials
    ule_materials = material_db.list_materials_by_category(
        material_db.MaterialCategory.ULTRA_LOW_EXPANSION
    )
    print(f"Ultra-low expansion materials: {len(ule_materials)}")
    
    # Show Zerodur properties
    zerodur = material_db.get_material('zerodur')
    print(f"Zerodur thermal expansion: {zerodur.alpha_linear:.2e} ± {zerodur.alpha_uncertainty:.2e} K⁻¹")
    
    # Calculate thermal expansion uncertainty
    length = 0.1  # 10 cm
    temp_change = 0.01  # K
    nominal_strain, strain_uncertainty = material_db.get_thermal_expansion_uncertainty(
        'zerodur', temp_change
    )
    expansion = nominal_strain * length
    expansion_uncertainty = strain_uncertainty * length
    
    print(f"10 cm Zerodur expansion (ΔT = 0.01 K): {expansion*1e9:.2f} ± {expansion_uncertainty*1e9:.2f} nm")
    
    # 2. Enhanced Casimir Pressure
    print("\n2. ENHANCED CASIMIR PRESSURE CALCULATIONS")
    print("-" * 42)
    
    casimir = EnhancedCasimirPressure()
    
    # Calculate pressure between silicon plates
    separation = 1e-6  # 1 μm
    temperature = 293.15  # K
    
    pressure = casimir.calculate_enhanced_casimir_pressure(
        separation=separation,
        material_name='silicon',
        temperature=temperature
    )
    
    print(f"Enhanced Casimir pressure (Si, 1 μm, 293 K): {pressure:.3e} Pa")
    
    # 3. Vacuum System Performance
    print("\n3. VACUUM SYSTEM PERFORMANCE")
    print("-" * 29)
    
    vacuum_system = AdvancedVacuumSystem()
    pumping_system = AdvancedPumpingSystem()
    
    # Calculate ultimate pressure
    chamber_volume = 1.0  # m³
    surface_area = 10.0   # m²
    
    ultimate_pressure = vacuum_system.calculate_ultimate_pressure(
        chamber_volume, surface_area, temperature
    )
    
    print(f"Ultimate pressure (1 m³ chamber): {ultimate_pressure:.2e} Pa")
    
    # Check vacuum threshold
    vacuum_threshold = 1e-6  # Pa
    vacuum_passed = ultimate_pressure <= vacuum_threshold
    print(f"Vacuum threshold (≤ 1×10⁻⁶ Pa): {'✓ PASS' if vacuum_passed else '✗ FAIL'}")
    
    # 4. Thermal Control System
    print("\n4. THERMAL CONTROL SYSTEM")
    print("-" * 25)
    
    thermal_controller = EnhancedThermalDriftController()
    
    # Calculate optimal PID gains
    pid_gains = thermal_controller.calculate_optimal_pid_gains(
        thermal_time_constant=30.0,  # s
        sensor_noise_std=0.001,      # K
        disturbance_amplitude=0.02   # K
    )
    
    print(f"Optimal PID gains:")
    print(f"  Kp = {pid_gains['kp']:.3f}")
    print(f"  Ki = {pid_gains['ki']:.3f}")
    print(f"  Kd = {pid_gains['kd']:.3f}")
    
    # Simulate control performance
    performance = thermal_controller.simulate_control_performance(
        pid_gains, duration=3600  # 1 hour
    )
    
    temp_stability = performance['temperature_std']
    temp_threshold = 0.01  # K
    temp_passed = temp_stability <= temp_threshold
    
    print(f"Temperature stability: ±{temp_stability:.4f} K")
    print(f"Temperature threshold (±0.01 K): {'✓ PASS' if temp_passed else '✗ FAIL'}")
    
    # 5. Vibration Control System
    print("\n5. VIBRATION CONTROL SYSTEM")
    print("-" * 27)
    
    vibration_controller = MultiRateVibrationController()
    
    # Design multi-rate controller
    controller_config = vibration_controller.design_multi_rate_controller(
        fast_rate=10000,  # Hz
        slow_rate=1000,   # Hz
        thermal_rate=1,   # Hz
        performance_weight=1.0
    )
    
    print(f"Multi-rate controller configuration:")
    print(f"  Fast rate: {controller_config['fast_rate']} Hz")
    print(f"  Slow rate: {controller_config['slow_rate']} Hz")
    print(f"  Thermal rate: {controller_config['thermal_rate']} Hz")
    
    # Calculate vibration performance
    frequency_range = (0.1, 100.0)  # Hz
    disturbance_level = 1e-6  # m
    
    performance = vibration_controller.calculate_closed_loop_performance(
        controller_config, frequency_range, disturbance_level
    )
    
    # Calculate RMS vibration
    import numpy as np
    frequencies = performance['frequencies']
    response_psd = performance['response_psd']
    
    freq_mask = (frequencies >= frequency_range[0]) & (frequencies <= frequency_range[1])
    freq_range_data = frequencies[freq_mask]
    response_psd_data = response_psd[freq_mask]
    
    df = freq_range_data[1] - freq_range_data[0] if len(freq_range_data) > 1 else 1.0
    rms_vibration = np.sqrt(np.trapz(response_psd_data, dx=df))
    
    vibration_threshold = 1e-9  # m (1 nm)
    vibration_passed = rms_vibration <= vibration_threshold
    
    print(f"RMS vibration (0.1-100 Hz): {rms_vibration*1e9:.2f} nm")
    print(f"Vibration threshold (< 1 nm): {'✓ PASS' if vibration_passed else '✗ FAIL'}")
    
    # 6. Overall System Validation
    print("\n6. OVERALL SYSTEM VALIDATION")
    print("-" * 29)
    
    all_tests_passed = vacuum_passed and temp_passed and vibration_passed
    
    print(f"Performance Summary:")
    print(f"  Vacuum: {ultimate_pressure:.2e} Pa {'✓' if vacuum_passed else '✗'}")
    print(f"  Temperature: ±{temp_stability:.4f} K {'✓' if temp_passed else '✗'}")
    print(f"  Vibration: {rms_vibration*1e9:.2f} nm RMS {'✓' if vibration_passed else '✗'}")
    
    print(f"\nOverall System Status: {'✓ ALL SPECIFICATIONS MET' if all_tests_passed else '✗ SPECIFICATIONS NOT MET'}")
    
    # Material optimization recommendation
    print(f"\n7. MATERIAL OPTIMIZATION RECOMMENDATIONS")
    print("-" * 40)
    
    # Find materials suitable for 0.01 K stability
    target_strain = 1e-8  # 1 nm / 10 cm
    max_alpha = target_strain / temp_change
    
    suitable_materials = material_db.find_materials_by_expansion_range(0, max_alpha)
    
    print(f"Materials suitable for ±0.01 K stability:")
    for material_name in suitable_materials:
        material = material_db.get_material(material_name)
        print(f"  {material.name}: α = {material.alpha_linear:.2e} K⁻¹")
    
    if 'zerodur' in suitable_materials:
        print(f"\nRECOMMENDATION: Zerodur is optimal for ultra-stable applications")
    
    print(f"\nExample usage complete!")
    print("=" * 60)
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    exit(main())
