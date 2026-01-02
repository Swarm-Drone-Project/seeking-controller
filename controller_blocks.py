"""
Controller Blocks for Hierarchical Quadrotor Control

This module contains modular PID controller blocks for cascaded control:
- PositionController: Outer loop, generates velocity setpoints
- VelocityController: Middle loop, generates thrust and attitude setpoints
- AttitudeController: Inner loop, generates torque commands
"""

import math
import numpy as np
from utils import wrap_angle, PIDController


class PositionController:
    """
    Level 1: Position Controller
    
    Generates velocity setpoints from position error.
    """
    
    def __init__(self, 
                 kp: tuple = (1.0, 1.0),
                 ki: tuple = (0.0, 0.0),
                 kd: tuple = (0.0, 0.0),
                 max_velocity: tuple = (5.0, 5.0)):
        """
        Args:
            kp: (kp_x, kp_y) - Proportional gains
            ki: (ki_x, ki_y) - Integral gains
            kd: (kd_x, kd_y) - Derivative gains
            max_velocity: (max_vx, max_vy) - Velocity limits
        """
        self.max_vx, self.max_vy = max_velocity
        
        self.pid_x = PIDController(
            kp=kp[0], ki=ki[0], kd=kd[0],
            output_limits=(-self.max_vx, self.max_vx)
        )
        self.pid_y = PIDController(
            kp=kp[1], ki=ki[1], kd=kd[1],
            output_limits=(-self.max_vy, self.max_vy)
        )
    
    def reset(self):
        """Reset controller states."""
        self.pid_x.reset()
        self.pid_y.reset()
    
    def compute(self, x_des: float, y_des: float, 
                x: float, y: float, dt: float) -> tuple:
        """
        Compute velocity setpoints from position error.
        
        Args:
            x_des, y_des: Desired position
            x, y: Current position
            dt: Time step
            
        Returns:
            (vx_des, vy_des): Desired velocities
        """
        vx_des = self.pid_x.compute(x_des, x, dt)
        vy_des = self.pid_y.compute(y_des, y, dt)
        
        return vx_des, vy_des


class VelocityController:
    """
    Level 2: Velocity Controller
    
    Generates attitude setpoint and thrust from velocity error.
    Converts desired accelerations to thrust magnitude and pitch angle.
    """
    
    GRAVITY = 9.81
    
    def __init__(self,
                 mass: float = 1.0,
                 kp: tuple = (2.0, 2.0),
                 ki: tuple = (0.1, 0.1),
                 kd: tuple = (0.0, 0.0),
                 max_acceleration: tuple = (5.0, 5.0),
                 max_tilt: float = math.pi / 4,
                 max_thrust: float = 30.0):
        """
        Args:
            mass: Drone mass (kg)
            kp: (kp_vx, kp_vy) - Proportional gains
            ki: (ki_vx, ki_vy) - Integral gains  
            kd: (kd_vx, kd_vy) - Derivative gains
            max_acceleration: (max_ax, max_ay) - Acceleration limits
            max_tilt: Maximum pitch angle (rad)
            max_thrust: Maximum thrust (N)
        """
        self.mass = mass
        self.max_tilt = max_tilt
        self.max_thrust = max_thrust
        self.max_ax, self.max_ay = max_acceleration
        
        self.pid_vx = PIDController(
            kp=kp[0], ki=ki[0], kd=kd[0],
            output_limits=(-self.max_ax, self.max_ax)
        )
        self.pid_vy = PIDController(
            kp=kp[1], ki=ki[1], kd=kd[1],
            output_limits=(-self.max_ay, self.max_ay)
        )
        
        # For logging
        self._ax_des = 0.0
        self._ay_des = 0.0
    
    def reset(self):
        """Reset controller states."""
        self.pid_vx.reset()
        self.pid_vy.reset()
    
    def compute(self, vx_des: float, vy_des: float,
                vx: float, vy: float, dt: float) -> tuple:
        """
        Compute thrust and attitude setpoint from velocity error.
        
        Args:
            vx_des, vy_des: Desired velocities
            vx, vy: Current velocities
            dt: Time step
            
        Returns:
            (thrust, theta_des): Thrust magnitude and desired pitch angle
        """
        # Compute desired accelerations
        ax_des = self.pid_vx.compute(vx_des, vx, dt)
        ay_des = self.pid_vy.compute(vy_des, vy, dt)
        
        # ===== Thrust Allocation =====
        # We need: ax = -(T/m)*sin(theta), ay = (T/m)*cos(theta) - g
        # Rearranging: (T/m)*sin(theta) = -ax_des, (T/m)*cos(theta) = ay_des + g
        
        ay_total = ay_des + self.GRAVITY  # Total vertical thrust component
        
        # Thrust magnitude: T = m * sqrt(ax_des^2 + ay_total^2)
        thrust = self.mass * math.sqrt(ax_des**2 + ay_total**2)
        thrust = np.clip(thrust, 0.0, self.max_thrust)
        
        # Desired pitch angle: theta_des = atan2(-ax_des, ay_total)
        # When ax_des > 0 (want to accelerate right), theta_des < 0 (tilt left... wait)
        # Actually: ax = -(T/m)*sin(theta), so for ax > 0, need sin(theta) < 0, so theta < 0
        # atan2(-ax_des, ay_total): for ax_des > 0, gives negative theta
        theta_des = math.atan2(-ax_des, ay_total)
        theta_des = np.clip(theta_des, -self.max_tilt, self.max_tilt)
        
        # Store for logging
        self._ax_des = ax_des
        self._ay_des = ay_des
        
        return thrust, theta_des


class AttitudeController:
    """
    Level 3: Attitude Controller (Innermost Loop)
    
    Generates torque command from attitude error.
    Runs at the highest rate for tight tracking.
    """
    
    def __init__(self,
                 kp: float = 20.0,
                 ki: float = 0.0,
                 kd: float = 5.0,
                 max_torque: float = 5.0):
        """
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            max_torque: Maximum torque (Nm)
        """
        self.max_torque = max_torque
        
        self.pid_theta = PIDController(
            kp=kp, ki=ki, kd=kd,
            output_limits=(-max_torque, max_torque)
        )
        
        # For logging
        self._theta_error = 0.0
    
    def reset(self):
        """Reset controller state."""
        self.pid_theta.reset()
    
    def compute(self, theta_des: float, theta: float, 
                omega: float, dt: float) -> float:
        """
        Compute torque command from attitude error.
        
        Args:
            theta_des: Desired pitch angle (rad)
            theta: Current pitch angle (rad)
            omega: Current angular velocity (rad/s)
            dt: Time step
            
        Returns:
            torque: Torque command (Nm)
        """
        # Wrap the angle error to [-pi, pi]
        theta_error = wrap_angle(theta_des - theta)
        
        # Use the error directly with a modified PID approach
        # P term on angle error, D term on angular rate (negative feedback)
        p_term = self.pid_theta.kp * theta_error
        
        # Integral term
        self.pid_theta.integral += theta_error * dt
        i_term = self.pid_theta.ki * self.pid_theta.integral
        
        # Derivative term: use angular rate directly for damping
        d_term = -self.pid_theta.kd * omega
        
        torque = p_term + i_term + d_term
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        
        # Store for logging
        self._theta_error = theta_error
        
        return torque
