import numpy as np
import math
from utils import wrap_angle


class PlanarQuadrotorDynamics:
    """
    2D Planar Quadrotor Dynamics (Plant).
    
    Manages the physical state and integration of the drone.
    
    Model: Vertical 2D plane (x=horizontal, y=vertical/altitude)
    States: [x, y, vx, vy, theta, omega]
    Inputs: [T (thrust), tau (torque)]
    
    Physics:
    - theta = 0 means drone is level, thrust points UP (+y)
    - theta > 0 means pitched forward (nose down), thrust has -x component
    - Gravity acts in -y direction
    """
    
    GRAVITY = 9.81
    
    def __init__(self, 
                 initial_state: list,
                 mass: float,
                 inertia: float,
                 drag_coefficient: list):
        """
        Args:
            initial_state: [x, y, vx, vy, theta, omega]
            mass: Mass in kg
            inertia: Moment of inertia
            drag_coefficient: [linear_drag, angular_drag]
        """
        self.state = np.array(initial_state, dtype=float)
        self.mass = mass
        self.inertia = inertia
        self.drag_coefficient = drag_coefficient
        
        # Store last calculated accelerations for logging
        self.ax = 0.0
        self.ay = 0.0
    
    def _compute_derivatives(self, thrust: float, torque: float) -> list:
        """
        Compute state derivatives using 2D quadrotor dynamics.
        
        Args:
            thrust: Thrust force (N)
            torque: Torque (Nm)
            
        Returns:
            [x_dot, y_dot, vx_dot, vy_dot, theta_dot, omega_dot]
        """
        x, y, vx, vy, theta, omega = self.state
        theta = wrap_angle(theta)
        
        inertial_drag, rotational_drag = self.drag_coefficient
        
        speed = math.sqrt(vx**2 + vy**2)
        
        # Drag forces (Quadratic drag assumption)
        f_drag_x = -inertial_drag * speed * vx
        f_drag_y = -inertial_drag * speed * vy
        
        # 2D Quadrotor Dynamics (Vertical Plane)
        # Fx = -thrust * sin(theta)
        # Fy = thrust * cos(theta) - m*g
        x_dot = vx
        y_dot = vy
        vx_dot = (-thrust * math.sin(theta) + f_drag_x) / self.mass
        vy_dot = (thrust * math.cos(theta) - self.mass * self.GRAVITY + f_drag_y) / self.mass
        theta_dot = omega
        omega_dot = (torque - rotational_drag * omega) / self.inertia
        
        return [x_dot, y_dot, vx_dot, vy_dot, theta_dot, omega_dot]
        
    def step(self, thrust: float, torque: float, dt: float):
        """
        Advance the simulation by one time step.
        
        Args:
            thrust: Thrust force (N)
            torque: Torque (Nm)
            dt: Time step (s)
            
        Returns:
            Updated state vector
        """
        # Calculate derivatives
        state_dots = self._compute_derivatives(thrust, torque)
        
        # Store accelerations (vx_dot, vy_dot)
        self.ax = state_dots[2]
        self.ay = state_dots[3]
        
        # Euler integration
        self.state = self.state + np.array(state_dots) * dt
        self.state[4] = wrap_angle(self.state[4])  # Wrap theta
        
        return self.state
    
    def get_state(self):
        """Return current state vector."""
        return self.state
