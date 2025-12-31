import numpy as np
import math
from utils import wrap_angle

def state_space(states, inputs, drag_coefficient, m, inertia_tensor):
    inertial_drag, rotational_drag = drag_coefficient
    x, y, vx, vy, theta, omega = states
    theta = wrap_angle(theta)  # Wrap theta to [-pi, pi]
    u1, u2 = inputs # u1=Thrust (N), u2=Torque (Nm)

    g = 9.81 # Gravity

    speed = math.sqrt(vx**2 + vy**2)
    
    # Drag forces (Quadratic drag assumption)
    # Drag opposes velocity
    f_drag_x = -inertial_drag * speed * vx
    f_drag_y = -inertial_drag * speed * vy

    # 2D Quadrotor Dynamics (Vertical Plane)
    # theta = 0 is horizontal (thrust up)
    # Fx = -u1 * sin(theta)
    # Fy = u1 * cos(theta) - mg
    
    x_dot = vx
    y_dot = vy
    vx_dot = (-u1 * math.sin(theta) + f_drag_x) / m
    vy_dot = (u1 * math.cos(theta) - m * g + f_drag_y) / m
    theta_dot = omega
    omega_dot = (u2 - rotational_drag * omega) / inertia_tensor

    return [x_dot, y_dot, vx_dot, vy_dot, theta_dot, omega_dot]

