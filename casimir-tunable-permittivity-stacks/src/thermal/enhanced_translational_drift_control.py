"""
Enhanced Translational Drift Control System
Advanced PID thermal control with optimized gains for ±0.01 K stability

Mathematical Formulations:
- f_compensation(t) = 1 + [K_p × e(t) + K_i × ∫e(t)dt + K_d × de(t)/dt]
- K_p = (2ζω_n τ - 1) / K_thermal
- K_i = ω_n² τ / K_thermal  
- K_d = τ / K_thermal
"""

import numpy as np
import scipy.signal
import scipy.optimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
import time
from collections import deque

@dataclass
class PIDControllerConfig:
    """PID controller configuration"""
    K_p: float              # Proportional gain
    K_i: float              # Integral gain
    K_d: float              # Derivative gain
    setpoint: float         # Target temperature (K)
    output_limits: Tuple[float, float]  # Output limits (min, max)
    integral_limit: float   # Integral windup limit
    derivative_filter_time: float  # Derivative filter time constant
    sample_time: float      # Control loop sample time (s)

@dataclass
class ThermalSystemModel:
    """Thermal system model parameters"""
    K_thermal: float        # Thermal gain (K/W)
    tau_thermal: float      # Thermal time constant (s)
    tau_sensor: float       # Sensor time constant (s)
    deadtime: float         # System dead time (s)
    disturbance_amplitude: float  # Disturbance amplitude (K)
    disturbance_frequency: float  # Disturbance frequency (Hz)

class EnhancedThermalDriftController:
    """
    Enhanced PID thermal drift controller with optimized gains
    
    Implements:
    - Optimized PID gain calculation
    - Adaptive control with system identification
    - Real-time performance monitoring
    - Disturbance rejection optimization
    """
    
    def __init__(self, system_model: ThermalSystemModel):
        self.logger = logging.getLogger(__name__)
        self.system_model = system_model
        self.target_stability = 0.01  # K
        
        # Controller state
        self.last_error = 0.0
        self.integral_sum = 0.0
        self.last_time = time.time()
        self.error_history = deque(maxlen=1000)
        self.output_history = deque(maxlen=1000)
        
        # Performance metrics
        self.performance_metrics = {
            'settling_time': 0.0,
            'overshoot': 0.0,
            'steady_state_error': 0.0,
            'control_effort': 0.0
        }
        
    def calculate_optimal_pid_gains(self,
                                  desired_damping: float = 0.707,
                                  desired_bandwidth: float = 0.1) -> PIDControllerConfig:
        """
        Calculate optimal PID gains using pole placement method
        
        Mathematical formulations:
        - K_p = (2ζω_n τ - 1) / K_thermal
        - K_i = ω_n² τ / K_thermal
        - K_d = τ / K_thermal
        
        Args:
            desired_damping: Desired damping ratio (ζ)
            desired_bandwidth: Desired bandwidth (Hz)
            
        Returns:
            Optimized PID configuration
        """
        zeta = desired_damping
        omega_n = 2 * np.pi * desired_bandwidth  # rad/s
        
        K_thermal = self.system_model.K_thermal
        tau = self.system_model.tau_thermal
        
        # Optimal PID gains:
        # K_p = (2ζω_n τ - 1) / K_thermal
        K_p = (2 * zeta * omega_n * tau - 1) / K_thermal
        
        # K_i = ω_n² τ / K_thermal
        K_i = (omega_n**2 * tau) / K_thermal
        
        # K_d = τ / K_thermal
        K_d = tau / K_thermal
        
        # Ensure gains are physically realizable
        K_p = max(0.1, K_p)  # Minimum proportional gain
        K_i = max(0.01, K_i)  # Minimum integral gain
        K_d = max(0.001, K_d)  # Minimum derivative gain
        
        # Calculate derivative filter time constant
        # τ_filter = τ_d / N, where N is typically 8-20
        N_filter = 10
        derivative_filter_time = K_d / (K_p * N_filter)
        
        # Sample time based on system dynamics
        sample_time = min(tau / 10, 0.1)  # 10x faster than system time constant
        
        pid_config = PIDControllerConfig(
            K_p=K_p,
            K_i=K_i,
            K_d=K_d,
            setpoint=self.system_model.K_thermal,  # Default setpoint
            output_limits=(-10.0, 10.0),  # W (heating/cooling power)
            integral_limit=5.0,  # Prevent windup
            derivative_filter_time=derivative_filter_time,
            sample_time=sample_time
        )
        
        self.logger.info(f"Optimal PID gains calculated: "
                        f"Kp={K_p:.3f}, Ki={K_i:.3f}, Kd={K_d:.3f}")
        
        return pid_config
    
    def calculate_compensation_function(self,
                                     error: float,
                                     integral: float,
                                     derivative: float,
                                     pid_config: PIDControllerConfig) -> float:
        """
        Calculate thermal compensation function
        
        Mathematical formulation:
        f_compensation(t) = 1 + [K_p × e(t) + K_i × ∫e(t)dt + K_d × de(t)/dt]
        
        Args:
            error: Current temperature error (K)
            integral: Integral of error (K·s)
            derivative: Derivative of error (K/s)
            pid_config: PID configuration
            
        Returns:
            Compensation function value
        """
        # PID control law
        pid_output = (pid_config.K_p * error + 
                     pid_config.K_i * integral + 
                     pid_config.K_d * derivative)
        
        # Compensation function: f_compensation(t) = 1 + PID_output
        f_compensation = 1.0 + pid_output
        
        # Apply output limits
        min_output, max_output = pid_config.output_limits
        f_compensation = np.clip(f_compensation, 1.0 + min_output, 1.0 + max_output)
        
        return f_compensation
    
    def update_pid_controller(self,
                            current_temperature: float,
                            setpoint: float,
                            pid_config: PIDControllerConfig) -> Tuple[float, Dict]:
        """
        Update PID controller with current measurement
        
        Args:
            current_temperature: Current temperature measurement (K)
            setpoint: Target temperature (K)
            pid_config: PID configuration
            
        Returns:
            Tuple of (control_output, controller_state)
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Use configured sample time if available
        if dt < pid_config.sample_time:
            return self.last_output if hasattr(self, 'last_output') else 0.0, {}
        
        # Calculate error
        error = setpoint - current_temperature
        
        # Update integral with windup protection
        self.integral_sum += error * dt
        if abs(self.integral_sum) > pid_config.integral_limit:
            self.integral_sum = np.sign(self.integral_sum) * pid_config.integral_limit
        
        # Calculate derivative with filtering
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        
        # Apply derivative filter
        if hasattr(self, 'filtered_derivative'):
            alpha = dt / (pid_config.derivative_filter_time + dt)
            self.filtered_derivative = alpha * derivative + (1 - alpha) * self.filtered_derivative
        else:
            self.filtered_derivative = derivative
        
        # Calculate compensation function
        control_output = self.calculate_compensation_function(
            error, self.integral_sum, self.filtered_derivative, pid_config
        )
        
        # Update controller state
        self.last_error = error
        self.last_time = current_time
        self.last_output = control_output
        
        # Store history for analysis
        self.error_history.append(error)
        self.output_history.append(control_output)
        
        controller_state = {
            'error': error,
            'integral': self.integral_sum,
            'derivative': self.filtered_derivative,
            'control_output': control_output,
            'timestamp': current_time
        }
        
        return control_output, controller_state
    
    def analyze_closed_loop_performance(self,
                                      pid_config: PIDControllerConfig,
                                      simulation_time: float = 100.0) -> Dict:
        """
        Analyze closed-loop system performance
        
        Args:
            pid_config: PID configuration to analyze
            simulation_time: Simulation time (s)
            
        Returns:
            Performance analysis results
        """
        # Create transfer function model
        # Plant: G(s) = K_thermal / (τs + 1) * exp(-Td*s)
        # Approximation of dead time: Pade approximation
        K = self.system_model.K_thermal
        tau = self.system_model.tau_thermal
        Td = self.system_model.deadtime
        
        # Plant transfer function (without dead time)
        num_plant = [K]
        den_plant = [tau, 1]
        
        # PID controller transfer function
        # C(s) = Kp + Ki/s + Kd*s
        num_pid = [pid_config.K_d, pid_config.K_p, pid_config.K_i]
        den_pid = [1, 0]
        
        # Closed-loop transfer function
        # T(s) = C(s)*G(s) / (1 + C(s)*G(s))
        num_ol = np.convolve(num_pid, num_plant)
        den_ol = np.convolve(den_pid, den_plant)
        
        # Add integrator pole from PID
        den_ol = np.append(den_ol, 0)
        
        # Closed-loop transfer function
        num_cl = num_ol
        den_cl = np.polyadd(den_ol, num_ol)
        
        # Step response analysis
        t_step = np.linspace(0, simulation_time, int(simulation_time * 10))
        
        try:
            system_cl = scipy.signal.TransferFunction(num_cl, den_cl)
            t_out, y_out = scipy.signal.step(system_cl, T=t_step)
            
            # Performance metrics
            steady_state_value = y_out[-1]
            settling_time = self._calculate_settling_time(t_out, y_out, steady_state_value)
            overshoot = self._calculate_overshoot(y_out, steady_state_value)
            rise_time = self._calculate_rise_time(t_out, y_out, steady_state_value)
            
            # Frequency response
            w = np.logspace(-3, 2, 1000)  # rad/s
            w_out, mag, phase = scipy.signal.bode(system_cl, w)
            
            # Stability margins
            gm, pm, wg, wp = scipy.signal.margin(scipy.signal.TransferFunction(num_ol, den_ol))
            
            performance_analysis = {
                'time_response': {
                    'time': t_out,
                    'output': y_out,
                    'settling_time': settling_time,
                    'overshoot': overshoot,
                    'rise_time': rise_time,
                    'steady_state_value': steady_state_value
                },
                'frequency_response': {
                    'frequency': w_out,
                    'magnitude': mag,
                    'phase': phase
                },
                'stability_margins': {
                    'gain_margin': gm,
                    'phase_margin': pm,
                    'gain_crossover_freq': wg,
                    'phase_crossover_freq': wp
                },
                'meets_stability_target': overshoot < 0.1 and settling_time < 10.0
            }
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            performance_analysis = {'error': str(e)}
        
        return performance_analysis
    
    def _calculate_settling_time(self, t: np.ndarray, y: np.ndarray, 
                               steady_state: float, tolerance: float = 0.02) -> float:
        """Calculate 2% settling time"""
        target_band = steady_state * tolerance
        
        for i in reversed(range(len(y))):
            if abs(y[i] - steady_state) > target_band:
                return t[i] if i < len(t) - 1 else t[-1]
        
        return t[0]
    
    def _calculate_overshoot(self, y: np.ndarray, steady_state: float) -> float:
        """Calculate percentage overshoot"""
        max_value = np.max(y)
        if steady_state > 0:
            overshoot = (max_value - steady_state) / steady_state * 100
        else:
            overshoot = 0.0
        return max(0.0, overshoot)
    
    def _calculate_rise_time(self, t: np.ndarray, y: np.ndarray, 
                           steady_state: float) -> float:
        """Calculate 10% to 90% rise time"""
        target_10 = steady_state * 0.1
        target_90 = steady_state * 0.9
        
        t_10 = None
        t_90 = None
        
        for i, val in enumerate(y):
            if t_10 is None and val >= target_10:
                t_10 = t[i]
            if t_90 is None and val >= target_90:
                t_90 = t[i]
                break
        
        if t_10 is not None and t_90 is not None:
            return t_90 - t_10
        else:
            return 0.0
    
    def optimize_pid_parameters(self,
                              optimization_objective: str = 'settling_time',
                              constraints: Optional[Dict] = None) -> PIDControllerConfig:
        """
        Optimize PID parameters using optimization algorithms
        
        Args:
            optimization_objective: 'settling_time', 'overshoot', 'ise', 'iae'
            constraints: Optimization constraints
            
        Returns:
            Optimized PID configuration
        """
        if constraints is None:
            constraints = {
                'max_overshoot': 5.0,  # %
                'max_settling_time': 20.0,  # s
                'min_phase_margin': 45.0,  # degrees
                'min_gain_margin': 6.0  # dB
            }
        
        def objective_function(params):
            """Objective function for optimization"""
            K_p, K_i, K_d = params
            
            if K_p <= 0 or K_i <= 0 or K_d <= 0:
                return 1e6  # Invalid parameters
            
            # Create temporary PID config
            temp_config = PIDControllerConfig(
                K_p=K_p, K_i=K_i, K_d=K_d,
                setpoint=self.system_model.K_thermal,
                output_limits=(-10.0, 10.0),
                integral_limit=5.0,
                derivative_filter_time=0.1,
                sample_time=0.01
            )
            
            try:
                # Analyze performance
                performance = self.analyze_closed_loop_performance(temp_config, 50.0)
                
                if 'error' in performance:
                    return 1e6
                
                time_resp = performance['time_response']
                stability = performance['stability_margins']
                
                # Check constraints
                if (time_resp['overshoot'] > constraints['max_overshoot'] or
                    time_resp['settling_time'] > constraints['max_settling_time'] or
                    stability['phase_margin'] < constraints['min_phase_margin'] or
                    20 * np.log10(stability['gain_margin']) < constraints['min_gain_margin']):
                    return 1e6
                
                # Objective function based on optimization goal
                if optimization_objective == 'settling_time':
                    return time_resp['settling_time']
                elif optimization_objective == 'overshoot':
                    return time_resp['overshoot']
                elif optimization_objective == 'ise':
                    # Integral squared error (approximation)
                    return time_resp['settling_time'] + time_resp['overshoot'] * 0.1
                else:
                    return time_resp['settling_time']
                    
            except Exception:
                return 1e6
        
        # Initial guess from analytical calculation
        initial_config = self.calculate_optimal_pid_gains()
        initial_guess = [initial_config.K_p, initial_config.K_i, initial_config.K_d]
        
        # Optimization bounds
        bounds = [(0.1, 100), (0.01, 10), (0.001, 1)]
        
        # Optimize using scipy
        try:
            result = scipy.optimize.minimize(
                objective_function,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            if result.success:
                K_p_opt, K_i_opt, K_d_opt = result.x
                
                optimized_config = PIDControllerConfig(
                    K_p=K_p_opt,
                    K_i=K_i_opt,
                    K_d=K_d_opt,
                    setpoint=self.system_model.K_thermal,
                    output_limits=(-10.0, 10.0),
                    integral_limit=5.0,
                    derivative_filter_time=K_d_opt / (K_p_opt * 10),
                    sample_time=0.01
                )
                
                self.logger.info(f"PID optimization successful: "
                                f"Kp={K_p_opt:.3f}, Ki={K_i_opt:.3f}, Kd={K_d_opt:.3f}")
                
                return optimized_config
            else:
                self.logger.warning("PID optimization failed, using analytical solution")
                return initial_config
                
        except Exception as e:
            self.logger.error(f"PID optimization error: {e}")
            return initial_config

def main():
    """Demonstration of enhanced thermal drift control capabilities"""
    
    # Define thermal system model
    system_model = ThermalSystemModel(
        K_thermal=2.0,          # K/W
        tau_thermal=10.0,       # s
        tau_sensor=1.0,         # s
        deadtime=0.5,           # s
        disturbance_amplitude=0.005,  # K
        disturbance_frequency=0.1     # Hz
    )
    
    # Initialize controller
    controller = EnhancedThermalDriftController(system_model)
    
    print("Enhanced Thermal Drift Control Analysis")
    print("="*50)
    
    # Calculate optimal PID gains
    pid_config = controller.calculate_optimal_pid_gains(
        desired_damping=0.707,
        desired_bandwidth=0.05  # Hz
    )
    
    print(f"Optimal PID Gains:")
    print(f"K_p = {pid_config.K_p:.3f}")
    print(f"K_i = {pid_config.K_i:.3f}")
    print(f"K_d = {pid_config.K_d:.3f}")
    print(f"Sample time = {pid_config.sample_time:.3f} s")
    
    # Performance analysis
    performance = controller.analyze_closed_loop_performance(pid_config)
    
    if 'error' not in performance:
        print(f"\nClosed-Loop Performance:")
        print(f"Settling time: {performance['time_response']['settling_time']:.2f} s")
        print(f"Overshoot: {performance['time_response']['overshoot']:.2f} %")
        print(f"Rise time: {performance['time_response']['rise_time']:.2f} s")
        print(f"Gain margin: {20*np.log10(performance['stability_margins']['gain_margin']):.1f} dB")
        print(f"Phase margin: {performance['stability_margins']['phase_margin']:.1f} degrees")
        print(f"Meets stability target: {performance['meets_stability_target']}")
    
    # Optimize PID parameters
    print(f"\nOptimizing PID parameters...")
    optimized_config = controller.optimize_pid_parameters('settling_time')
    
    print(f"Optimized PID Gains:")
    print(f"K_p = {optimized_config.K_p:.3f}")
    print(f"K_i = {optimized_config.K_i:.3f}")
    print(f"K_d = {optimized_config.K_d:.3f}")
    
    # Simulate controller operation
    print(f"\n5. Controller Simulation:")
    setpoint = 293.15  # K
    current_temp = 293.10  # K (initial error)
    
    for i in range(5):
        control_output, state = controller.update_pid_controller(
            current_temp, setpoint, optimized_config
        )
        
        print(f"Step {i+1}: Temp={current_temp:.3f} K, "
              f"Error={state['error']:.3f} K, "
              f"Output={control_output:.3f}")
        
        # Simulate temperature response (simplified)
        current_temp += (control_output - 1.0) * system_model.K_thermal * 0.1

if __name__ == "__main__":
    main()
