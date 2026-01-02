"""
2D Planar Quadrotor Hierarchical Controller (Logic Only)

This module contains the pure control logic for the hierarchical controller.
It does not handle physics simulation or visualization.
"""

import math
import numpy as np
from utils import wrap_angle
from controller_blocks import PositionController, VelocityController, AttitudeController


class HierarchicalController:
    """
    2D Planar Quadrotor Hierarchical Controller.
    
    Control hierarchy:
    Position PID -> Velocity PID -> Attitude PID
    """
    
    GRAVITY = 9.81
    
    def __init__(self,
                 mass: float,
                 # Position controller gains
                 pos_kp: tuple,
                 pos_ki: tuple,
                 pos_kd: tuple,
                 max_velocity: tuple,
                 # Velocity controller gains
                 vel_kp: tuple,
                 vel_ki: tuple,
                 vel_kd: tuple,
                 max_acceleration: tuple,
                 # Attitude controller gains
                 att_kp: float,
                 att_ki: float,
                 att_kd: float,
                 # Limits
                 max_tilt: float,
                 max_thrust: float,
                 max_torque: float,
                 # Control mode
                 control_mode: str):
        
        self.mass = mass
        self.control_mode = control_mode
        
        # Initialize controllers
        self.position_ctrl = PositionController(
            kp=pos_kp, ki=pos_ki, kd=pos_kd,
            max_velocity=max_velocity
        )
        
        self.velocity_ctrl = VelocityController(
            mass=mass,
            kp=vel_kp, ki=vel_ki, kd=vel_kd,
            max_acceleration=max_acceleration,
            max_tilt=max_tilt,
            max_thrust=max_thrust
        )
        
        self.attitude_ctrl = AttitudeController(
            kp=att_kp, ki=att_ki, kd=att_kd,
            max_torque=max_torque
        )
        
        # Setpoints
        self.x_des = 0.0
        self.y_des = 0.0
        self.vx_setpoint = 0.0
        self.vy_setpoint = 0.0
        
        # Intermediate values (for logging)
        self.vx_des = 0.0
        self.vy_des = 0.0
        self.theta_des = 0.0
        self.ax_des = 0.0
        self.ay_des = 0.0
    
    def set_position_setpoint(self, x_des: float, y_des: float):
        """Set desired position (for position control mode)."""
        self.x_des = x_des
        self.y_des = y_des
    
    def set_velocity_setpoint(self, vx_des: float, vy_des: float):
        """Set desired velocity (for velocity control mode)."""
        self.vx_setpoint = vx_des
        self.vy_setpoint = vy_des
    
    def reset(self):
        """Reset all controller states."""
        self.position_ctrl.reset()
        self.velocity_ctrl.reset()
        self.attitude_ctrl.reset()
    
    def compute_control(self, state: np.ndarray, dt: float) -> tuple:
        """
        Compute control outputs based on current state.
        
        Args:
            state: [x, y, vx, vy, theta, omega]
            dt: Time step
            
        Returns:
            (thrust, torque)
        """
        x, y, vx, vy, theta, omega = state
        
        if self.control_mode == 'position':
            # ===== Level 1: Position Controller =====
            vx_des, vy_des = self.position_ctrl.compute(
                self.x_des, self.y_des, x, y, dt
            )
        else:  # velocity mode
            # Bypass position controller
            vx_des = self.vx_setpoint
            vy_des = self.vy_setpoint
        
        # ===== Level 2: Velocity Controller =====
        thrust, theta_des = self.velocity_ctrl.compute(
            vx_des, vy_des, vx, vy, dt
        )
        
        # ===== Level 3: Attitude Controller =====
        torque = self.attitude_ctrl.compute(
            theta_des, theta, omega, dt
        )
        
        # Store intermediate values for logging
        self.vx_des = vx_des
        self.vy_des = vy_des
        self.theta_des = theta_des
        self.ax_des = self.velocity_ctrl._ax_des
        self.ay_des = self.velocity_ctrl._ay_des
        
        return thrust, torque
