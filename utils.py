import math


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class PIDController:
    """Simple PID controller implementation."""
    
    def __init__(self, kp: float, ki: float, kd: float, output_limits: tuple = (None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min, self.output_max = output_limits
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_call = True
    
    def reset(self):
        """Reset the controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_call = True
    
    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        """Compute PID output given setpoint and measurement."""
        error = setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        if self.first_call:
            d_term = 0.0
            self.first_call = False
        else:
            d_term = self.kd * (error - self.prev_error) / dt
        
        self.prev_error = error
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.output_min is not None:
            output = max(self.output_min, output)
        if self.output_max is not None:
            output = min(self.output_max, output)
        
        return output

