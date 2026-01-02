"""
2D Planar Quadrotor Hierarchical Controller

Model: Vertical 2D plane (x=horizontal, y=vertical/altitude)
States: [x, y, vx, vy, theta, omega]
Inputs: [T (thrust), tau (torque)]

Hierarchical Control Architecture (3-Level Cascade):
=====================================================

Level 1 - Position Controller (Outermost Loop):
    - Input: Position setpoint (x_des, y_des) and current position (x, y)
    - Output: Velocity setpoint (vx_des, vy_des)
    - PID on position error

Level 2 - Velocity Controller (Middle Loop):
    - Input: Velocity setpoint (vx_des, vy_des) and current velocity (vx, vy)
    - Output: Desired acceleration (ax_des, ay_des) -> converted to attitude setpoint (theta_des, thrust)
    - PID on velocity error
    - Thrust allocation: T = m * sqrt(ax_des^2 + (ay_des + g)^2)
    - Attitude setpoint: theta_des = atan2(-ax_des, ay_des + g)

Level 3 - Attitude Controller (Innermost Loop):
    - Input: Attitude setpoint (theta_des) and current attitude (theta, omega)
    - Output: Torque command (tau)
    - PID on attitude error

Physics:
- theta = 0 means drone is level, thrust points UP (+y)
- theta > 0 means pitched forward (nose down), thrust has -x component
- Gravity acts in -y direction

Usage:
------
Position control mode (default):
    python hierarchical_controller.py --mode position --setpoint 10.0 3.0

Velocity control mode:
    python hierarchical_controller.py --mode velocity --setpoint 2.0 1.5

Options:
    --duration SECONDS    Simulation duration (default: 12.0)
    --no-live            Skip live animation, only show plots
"""

import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simple_model import state_space
from utils import wrap_angle
from controller_blocks import PositionController, VelocityController, AttitudeController


class HierarchicalQuadrotor:
    """
    2D Planar Quadrotor with Hierarchical Cascaded Control.
    
    Supports two modes:
    - Position control: Position PID -> Velocity PID -> Attitude PID
    - Velocity control: Velocity PID -> Attitude PID (bypasses position loop)
    
    Coordinate system:
    - x: horizontal (positive = right)
    - y: vertical (positive = up)
    - theta: pitch angle (positive = nose down / tilted right)
    """
    
    GRAVITY = 9.81
    
    def __init__(self,
                 initial_states: list,
                 mass: float = 1.0,
                 inertia: float = 0.1,
                 drag_coefficient: list = [0.1, 0.1],
                 dt: float = 0.01,
                 # Position controller gains
                 pos_kp: tuple = (1.0, 1.0),
                 pos_ki: tuple = (0.0, 0.0),
                 pos_kd: tuple = (0.0, 0.0),
                 max_velocity: tuple = (5.0, 5.0),
                 # Velocity controller gains
                 vel_kp: tuple = (2.0, 2.0),
                 vel_ki: tuple = (0.1, 0.1),
                 vel_kd: tuple = (0.0, 0.0),
                 max_acceleration: tuple = (5.0, 5.0),
                 # Attitude controller gains
                 att_kp: float = 25.0,
                 att_ki: float = 0.0,
                 att_kd: float = 6.0,
                 # Limits
                 max_tilt: float = math.pi / 4,
                 max_thrust: float = 30.0,
                 max_torque: float = 5.0,
                 # Control mode
                 control_mode: str = 'position'):
        
        self.mass = mass
        self.inertia = inertia
        self.drag_coefficient = drag_coefficient
        self.dt = dt
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
        
        # Limits (stored for reference)
        self.max_tilt = max_tilt
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        
        # States
        self.states = np.array(initial_states, dtype=float)
        
        # Setpoints
        # Position setpoint (used in position mode)
        self.x_des = initial_states[0]
        self.y_des = initial_states[1]
        # Velocity setpoint (used in velocity mode, or computed in position mode)
        self.vx_setpoint = 0.0
        self.vy_setpoint = 0.0
        
        # Intermediate setpoints (for logging)
        self._vx_des = 0.0
        self._vy_des = 0.0
        self._theta_des = 0.0
        self._ax_des = 0.0
        self._ay_des = 0.0
        
        # Logging
        self.history = {
            'time': [],
            # States
            'x': [], 'y': [], 'vx': [], 'vy': [], 'theta': [], 'omega': [],
            # Position setpoints
            'x_des': [], 'y_des': [],
            # Velocity setpoints (from position controller or direct input)
            'vx_des': [], 'vy_des': [],
            # Attitude setpoint (from velocity controller)
            'theta_des': [],
            # Acceleration commands
            'ax_des': [], 'ay_des': [],
            # Control outputs
            'thrust': [], 'torque': [],
            # Errors at each level
            'x_error': [], 'y_error': [],
            'vx_error': [], 'vy_error': [],
            'theta_error': []
        }
    
    def set_position_setpoint(self, x_des: float, y_des: float):
        """Set desired position (for position control mode)."""
        self.x_des = x_des
        self.y_des = y_des
    
    def set_velocity_setpoint(self, vx_des: float, vy_des: float):
        """Set desired velocity (for velocity control mode)."""
        self.vx_setpoint = vx_des
        self.vy_setpoint = vy_des
    
    def reset_controllers(self):
        """Reset all controller states."""
        self.position_ctrl.reset()
        self.velocity_ctrl.reset()
        self.attitude_ctrl.reset()
    
    def compute_control(self) -> tuple:
        """
        Hierarchical cascaded control.
        
        Position mode:
            Level 1: Position -> Velocity setpoint
            Level 2: Velocity -> Thrust + Attitude setpoint
            Level 3: Attitude -> Torque
            
        Velocity mode:
            Level 2: Velocity -> Thrust + Attitude setpoint
            Level 3: Attitude -> Torque
        """
        x, y, vx, vy, theta, omega = self.states
        
        if self.control_mode == 'position':
            # ===== Level 1: Position Controller =====
            vx_des, vy_des = self.position_ctrl.compute(
                self.x_des, self.y_des, x, y, self.dt
            )
        else:  # velocity mode
            # Bypass position controller, use direct velocity setpoint
            vx_des = self.vx_setpoint
            vy_des = self.vy_setpoint
        
        # ===== Level 2: Velocity Controller =====
        thrust, theta_des = self.velocity_ctrl.compute(
            vx_des, vy_des, vx, vy, self.dt
        )
        
        # ===== Level 3: Attitude Controller =====
        torque = self.attitude_ctrl.compute(
            theta_des, theta, omega, self.dt
        )
        
        # Store intermediate values for logging
        self._vx_des = vx_des
        self._vy_des = vy_des
        self._theta_des = theta_des
        self._ax_des = self.velocity_ctrl._ax_des
        self._ay_des = self.velocity_ctrl._ay_des
        
        return thrust, torque
    
    def step(self):
        """Simulate one timestep."""
        thrust, torque = self.compute_control()
        
        # Dynamics
        state_dots = state_space(
            self.states.tolist(),
            [thrust, torque],
            self.drag_coefficient,
            self.mass,
            self.inertia
        )
        
        # Euler integration
        self.states = self.states + np.array(state_dots) * self.dt
        self.states[4] = wrap_angle(self.states[4])  # Wrap theta
        
        return thrust, torque
    
    def log_state(self, time: float, thrust: float, torque: float):
        """Log state for plotting."""
        x, y, vx, vy, theta, omega = self.states
        
        self.history['time'].append(time)
        # States
        self.history['x'].append(x)
        self.history['y'].append(y)
        self.history['vx'].append(vx)
        self.history['vy'].append(vy)
        self.history['theta'].append(theta)
        self.history['omega'].append(omega)
        # Position setpoints
        self.history['x_des'].append(self.x_des)
        self.history['y_des'].append(self.y_des)
        # Velocity setpoints
        self.history['vx_des'].append(self._vx_des)
        self.history['vy_des'].append(self._vy_des)
        # Attitude setpoint
        self.history['theta_des'].append(self._theta_des)
        # Acceleration commands
        self.history['ax_des'].append(self._ax_des)
        self.history['ay_des'].append(self._ay_des)
        # Control outputs
        self.history['thrust'].append(thrust)
        self.history['torque'].append(torque)
        # Errors
        self.history['x_error'].append(self.x_des - x)
        self.history['y_error'].append(self.y_des - y)
        self.history['vx_error'].append(self._vx_des - vx)
        self.history['vy_error'].append(self._vy_des - vy)
        self.history['theta_error'].append(wrap_angle(self._theta_des - theta))
    
    def run_simulation(self, duration: float):
        """Run simulation for given duration."""
        num_steps = int(duration / self.dt)
        for i in range(num_steps):
            time = i * self.dt
            thrust, torque = self.step()
            self.log_state(time, thrust, torque)
    
    def run_live(self, duration: float):
        """Run with live animation."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        mode_str = "Position Control" if self.control_mode == 'position' else "Velocity Control"
        ax.set_title(f'2D Planar Quadrotor - Hierarchical Controller ({mode_str})')
        
        # Trajectory
        traj_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)
        
        # Drone body
        drone_body, = ax.plot([], [], 'k-', linewidth=3)
        
        # Thrust vector
        thrust_arrow = ax.arrow(0, 0, 0, 0.5, head_width=0.1, fc='red', ec='red')
        
        # Setpoint marker (only for position mode)
        setpoint_marker, = ax.plot([], [], 'g*', markersize=15)
        
        # Velocity vector (desired)
        vel_des_arrow = ax.arrow(0, 0, 0, 0, head_width=0.05, fc='blue', ec='blue', alpha=0.5)
        
        # Info text
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=8,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        x_hist, y_hist = [], []
        num_steps = int(duration / self.dt)
        steps_per_frame = max(1, int(0.03 / self.dt))
        
        def init():
            ax.set_xlim(-2, 12)
            ax.set_ylim(-2, 8)
            return traj_line, drone_body, setpoint_marker, info_text
        
        def update(frame):
            nonlocal thrust_arrow, vel_des_arrow
            
            for _ in range(steps_per_frame):
                if len(self.history['time']) >= num_steps:
                    break
                time = len(self.history['time']) * self.dt
                thrust, torque = self.step()
                self.log_state(time, thrust, torque)
            
            x, y, vx, vy, theta, omega = self.states
            
            x_hist.append(x)
            y_hist.append(y)
            traj_line.set_data(x_hist, y_hist)
            
            # Draw drone body
            arm_len = 0.3
            dx = arm_len * math.cos(theta)
            dy = arm_len * math.sin(theta)
            drone_body.set_data([x - dx, x + dx], [y - dy, y + dy])
            
            # Draw thrust vector
            thrust_arrow.remove()
            thrust_mag = self.history['thrust'][-1] if self.history['thrust'] else 0
            thrust_scale = 0.05
            thrust_dx = -thrust_scale * thrust_mag * math.sin(theta)
            thrust_dy = thrust_scale * thrust_mag * math.cos(theta)
            thrust_arrow = ax.arrow(x, y, thrust_dx, thrust_dy,
                                   head_width=0.08, fc='red', ec='red')
            
            # Draw desired velocity vector
            vel_des_arrow.remove()
            vel_scale = 0.3
            vel_des_arrow = ax.arrow(x, y, self._vx_des * vel_scale, self._vy_des * vel_scale,
                                    head_width=0.05, fc='cyan', ec='cyan', alpha=0.7)
            
            # Setpoint marker (only show for position mode)
            if self.control_mode == 'position':
                setpoint_marker.set_data([self.x_des], [self.y_des])
            else:
                setpoint_marker.set_data([], [])
            
            # Info
            if self.control_mode == 'position':
                info = (f'Time: {len(self.history["time"]) * self.dt:.2f}s\n'
                       f'Mode: POSITION CONTROL\n'
                       f'─── Position ───\n'
                       f'Pos: ({x:.2f}, {y:.2f}) → ({self.x_des:.1f}, {self.y_des:.1f})\n'
                       f'─── Velocity ───\n'
                       f'Vel: ({vx:.2f}, {vy:.2f})\n'
                       f'Vel_des: ({self._vx_des:.2f}, {self._vy_des:.2f})\n'
                       f'─── Attitude ───\n'
                       f'θ: {math.degrees(theta):.1f}° (des: {math.degrees(self._theta_des):.1f}°)\n'
                       f'─── Commands ───\n'
                       f'Thrust: {thrust_mag:.2f} N | Torque: {self.history["torque"][-1]:.2f} Nm')
            else:
                info = (f'Time: {len(self.history["time"]) * self.dt:.2f}s\n'
                       f'Mode: VELOCITY CONTROL\n'
                       f'─── Position ───\n'
                       f'Pos: ({x:.2f}, {y:.2f})\n'
                       f'─── Velocity ───\n'
                       f'Vel: ({vx:.2f}, {vy:.2f})\n'
                       f'Vel_des: ({self._vx_des:.2f}, {self._vy_des:.2f}) [SETPOINT]\n'
                       f'─── Attitude ───\n'
                       f'θ: {math.degrees(theta):.1f}° (des: {math.degrees(self._theta_des):.1f}°)\n'
                       f'─── Commands ───\n'
                       f'Thrust: {thrust_mag:.2f} N | Torque: {self.history["torque"][-1]:.2f} Nm')
            info_text.set_text(info)
            
            # Auto-adjust view
            margin = 2
            x_min = min(min(x_hist), 0) - margin
            x_max = max(max(x_hist), self.x_des if self.control_mode == 'position' else max(x_hist)) + margin
            y_min = min(min(y_hist), 0) - margin
            y_max = max(max(y_hist), self.y_des if self.control_mode == 'position' else max(y_hist)) + margin
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            return traj_line, drone_body, setpoint_marker, info_text
        
        num_frames = num_steps // steps_per_frame + 1
        ani = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                           blit=False, interval=30, repeat=False)
        plt.show()
    
    def plot_results(self):
        """Plot comprehensive simulation results."""
        fig, axes = plt.subplots(5, 2, figsize=(14, 16))
        mode_str = "Position Control" if self.control_mode == 'position' else "Velocity Control"
        fig.suptitle(f'Hierarchical Controller Results ({mode_str})', fontsize=14, fontweight='bold')
        
        # Trajectory (X-Y)
        axes[0, 0].plot(self.history['x'], self.history['y'], 'b-', linewidth=1.5)
        axes[0, 0].plot(self.history['x'][0], self.history['y'][0], 'go', markersize=10, label='Start')
        axes[0, 0].plot(self.history['x'][-1], self.history['y'][-1], 'ro', markersize=10, label='End')
        if self.control_mode == 'position':
            axes[0, 0].plot(self.x_des, self.y_des, 'g*', markersize=15, label='Setpoint')
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Y (m)')
        axes[0, 0].set_title('Trajectory')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].axis('equal')
        
        # Position tracking
        axes[0, 1].plot(self.history['time'], self.history['x'], 'b-', label='x')
        axes[0, 1].plot(self.history['time'], self.history['y'], 'r-', label='y')
        if self.control_mode == 'position':
            axes[0, 1].plot(self.history['time'], self.history['x_des'], 'b--', alpha=0.5, label='x_des')
            axes[0, 1].plot(self.history['time'], self.history['y_des'], 'r--', alpha=0.5, label='y_des')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Position (m)')
        axes[0, 1].set_title('Level 1: Position Tracking' if self.control_mode == 'position' else 'Position')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Position errors (only meaningful in position mode)
        if self.control_mode == 'position':
            axes[1, 0].plot(self.history['time'], self.history['x_error'], 'b-', label='x_error')
            axes[1, 0].plot(self.history['time'], self.history['y_error'], 'r-', label='y_error')
            axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            axes[1, 0].set_title('Position Errors')
        else:
            axes[1, 0].text(0.5, 0.5, 'N/A in Velocity Mode', ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=12, color='gray')
            axes[1, 0].set_title('Position Errors (N/A)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Error (m)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Velocity tracking
        axes[1, 1].plot(self.history['time'], self.history['vx'], 'b-', label='vx')
        axes[1, 1].plot(self.history['time'], self.history['vy'], 'r-', label='vy')
        axes[1, 1].plot(self.history['time'], self.history['vx_des'], 'b--', alpha=0.5, label='vx_des')
        axes[1, 1].plot(self.history['time'], self.history['vy_des'], 'r--', alpha=0.5, label='vy_des')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Velocity (m/s)')
        axes[1, 1].set_title('Level 2: Velocity Tracking')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Velocity errors
        axes[2, 0].plot(self.history['time'], self.history['vx_error'], 'b-', label='vx_error')
        axes[2, 0].plot(self.history['time'], self.history['vy_error'], 'r-', label='vy_error')
        axes[2, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Error (m/s)')
        axes[2, 0].set_title('Velocity Errors')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Attitude tracking
        axes[2, 1].plot(self.history['time'], np.rad2deg(self.history['theta']), 'b-', label='θ')
        axes[2, 1].plot(self.history['time'], np.rad2deg(self.history['theta_des']), 'r--', label='θ_des')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Angle (deg)')
        axes[2, 1].set_title('Level 3: Attitude Tracking')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # Attitude error
        axes[3, 0].plot(self.history['time'], np.rad2deg(self.history['theta_error']), 'b-')
        axes[3, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        axes[3, 0].set_xlabel('Time (s)')
        axes[3, 0].set_ylabel('Error (deg)')
        axes[3, 0].set_title('Attitude Error')
        axes[3, 0].grid(True)
        
        # Angular velocity
        axes[3, 1].plot(self.history['time'], self.history['omega'], 'b-')
        axes[3, 1].set_xlabel('Time (s)')
        axes[3, 1].set_ylabel('ω (rad/s)')
        axes[3, 1].set_title('Angular Velocity')
        axes[3, 1].grid(True)
        
        # Thrust
        axes[4, 0].plot(self.history['time'], self.history['thrust'], 'b-')
        axes[4, 0].axhline(y=self.mass * self.GRAVITY, color='r', linestyle='--', 
                          label=f'Hover ({self.mass * self.GRAVITY:.1f} N)')
        axes[4, 0].set_xlabel('Time (s)')
        axes[4, 0].set_ylabel('Thrust (N)')
        axes[4, 0].set_title('Thrust Command')
        axes[4, 0].legend()
        axes[4, 0].grid(True)
        
        # Torque
        axes[4, 1].plot(self.history['time'], self.history['torque'], 'b-')
        axes[4, 1].set_xlabel('Time (s)')
        axes[4, 1].set_ylabel('Torque (Nm)')
        axes[4, 1].set_title('Torque Command')
        axes[4, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_control_cascade(self):
        """Plot the cascade control signals to visualize hierarchy."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        mode_str = "Position Control" if self.control_mode == 'position' else "Velocity Control"
        fig.suptitle(f'Control Cascade Visualization ({mode_str})', fontsize=14, fontweight='bold')
        
        # Level 1: Position -> Velocity setpoint
        ax1 = axes[0]
        if self.control_mode == 'position':
            ax1.set_title('Level 1: Position Controller Output (Velocity Setpoints)')
        else:
            ax1.set_title('Level 1: Direct Velocity Setpoints (Position Controller Bypassed)')
        ax1.plot(self.history['time'], self.history['vx_des'], 'b-', label='vx_des', linewidth=2)
        ax1.plot(self.history['time'], self.history['vy_des'], 'r-', label='vy_des', linewidth=2)
        ax1.plot(self.history['time'], self.history['vx'], 'b--', alpha=0.5, label='vx (actual)')
        ax1.plot(self.history['time'], self.history['vy'], 'r--', alpha=0.5, label='vy (actual)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.legend()
        ax1.grid(True)
        
        # Level 2: Velocity -> Attitude + Thrust
        ax2 = axes[1]
        ax2.set_title('Level 2: Velocity Controller Output (Thrust & Attitude Setpoint)')
        ax2_twin = ax2.twinx()
        l1, = ax2.plot(self.history['time'], self.history['thrust'], 'g-', label='Thrust', linewidth=2)
        l2, = ax2_twin.plot(self.history['time'], np.rad2deg(self.history['theta_des']), 
                           'm-', label='θ_des', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Thrust (N)', color='g')
        ax2_twin.set_ylabel('θ_des (deg)', color='m')
        ax2.legend(handles=[l1, l2], loc='upper right')
        ax2.grid(True)
        
        # Level 3: Attitude -> Torque
        ax3 = axes[2]
        ax3.set_title('Level 3: Attitude Controller Output (Torque)')
        ax3.plot(self.history['time'], self.history['torque'], 'b-', label='Torque', linewidth=2)
        ax3.plot(self.history['time'], np.rad2deg(self.history['theta_error']), 
                'r--', alpha=0.5, label='θ_error (deg)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Torque (Nm) / Error (deg)')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()


def create_drone(control_mode: str = 'position') -> HierarchicalQuadrotor:
    """Create a drone with default tuned parameters."""
    initial_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return HierarchicalQuadrotor(
        initial_states=initial_states,
        mass=1.0,
        inertia=0.1,
        drag_coefficient=[0.1, 0.1],
        dt=0.01,
        # Position controller (outer loop) - slower response
        pos_kp=(1.2, 1.2),
        pos_ki=(0.0, 0.0),
        pos_kd=(0.0, 0.0),
        max_velocity=(4.0, 4.0),
        # Velocity controller (middle loop) - medium response
        vel_kp=(3.0, 3.0),
        vel_ki=(0.2, 0.2),
        vel_kd=(0.1, 0.1),
        max_acceleration=(6.0, 6.0),
        # Attitude controller (inner loop) - fastest response
        att_kp=30.0,
        att_ki=0.5,
        att_kd=8.0,
        # Limits
        max_tilt=math.pi / 4,
        max_thrust=25.0,
        max_torque=5.0,
        # Mode
        control_mode=control_mode
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='2D Planar Quadrotor Hierarchical Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Position control to (10, 3):
    python hierarchical_controller.py --mode position --setpoint 10.0 3.0
    
  Velocity control at (2, 1) m/s:
    python hierarchical_controller.py --mode velocity --setpoint 2.0 1.0
    
  Run for 20 seconds without live animation:
    python hierarchical_controller.py --mode position --setpoint 5 5 --duration 20 --no-live
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['position', 'velocity'],
        default='position',
        help='Control mode: "position" for position control, "velocity" for velocity control (default: position)'
    )
    
    parser.add_argument(
        '--setpoint', '-s',
        type=float,
        nargs=2,
        default=None,
        metavar=('X', 'Y'),
        help='Setpoint values. For position mode: (x, y) in meters. For velocity mode: (vx, vy) in m/s'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=12.0,
        help='Simulation duration in seconds (default: 12.0)'
    )
    
    parser.add_argument(
        '--no-live',
        action='store_true',
        help='Skip live animation, only show result plots'
    )
    
    return parser.parse_args()


def main():
    """Main entry point with CLI support."""
    args = parse_args()
    
    # Create drone
    drone = create_drone(control_mode=args.mode)
    
    # Set setpoint based on mode
    if args.mode == 'position':
        if args.setpoint is None:
            x_des, y_des = 10.0, 3.0  # Default position setpoint
        else:
            x_des, y_des = args.setpoint
        drone.set_position_setpoint(x_des, y_des)
        setpoint_str = f"Position: ({x_des}, {y_des}) m"
    else:  # velocity mode
        if args.setpoint is None:
            vx_des, vy_des = 2.0, 1.0  # Default velocity setpoint
        else:
            vx_des, vy_des = args.setpoint
        drone.set_velocity_setpoint(vx_des, vy_des)
        setpoint_str = f"Velocity: ({vx_des}, {vy_des}) m/s"
    
    # Print info
    print("=" * 60)
    print("2D Planar Quadrotor - Hierarchical Cascaded Controller")
    print("=" * 60)
    print(f"\nControl Mode: {args.mode.upper()}")
    print(f"Setpoint: {setpoint_str}")
    print(f"Duration: {args.duration}s")
    print("\nControl Architecture:")
    if args.mode == 'position':
        print("  Level 1: Position PID -> Velocity Setpoint")
    else:
        print("  Level 1: [BYPASSED] Direct Velocity Setpoint")
    print("  Level 2: Velocity PID -> Thrust + Attitude Setpoint")
    print("  Level 3: Attitude PID -> Torque Command")
    print("\nParameters:")
    print(f"  Mass: {drone.mass} kg")
    print(f"  Hover thrust: {drone.mass * drone.GRAVITY:.2f} N")
    print("=" * 60)
    
    # Run simulation
    if args.no_live:
        print("\nRunning simulation...")
        drone.run_simulation(duration=args.duration)
    else:
        print("\nStarting live animation...")
        drone.run_live(duration=args.duration)
    
    # Final state
    print(f"\nFinal position: ({drone.states[0]:.3f}, {drone.states[1]:.3f})")
    print(f"Final velocity: ({drone.states[2]:.3f}, {drone.states[3]:.3f})")
    
    if args.mode == 'position':
        print(f"Position error: ({drone.x_des - drone.states[0]:.4f}, {drone.y_des - drone.states[1]:.4f})")
    else:
        print(f"Velocity error: ({drone.vx_setpoint - drone.states[2]:.4f}, {drone.vy_setpoint - drone.states[3]:.4f})")
    
    # Plot results
    drone.plot_results()
    drone.plot_control_cascade()


if __name__ == "__main__":
    main()
