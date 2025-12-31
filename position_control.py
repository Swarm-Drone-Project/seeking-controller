"""
2D Planar Quadrotor Controller

Model: Vertical 2D plane (x=horizontal, y=vertical/altitude)
States: [x, y, vx, vy, theta, omega]
Inputs: [T (thrust), tau (torque)]

Physics:
- theta = 0 means drone is level, thrust points UP (+y)
- theta > 0 means pitched forward (nose down), thrust has -x component
- Gravity acts in -y direction

Control Architecture (Cascaded):
1. Position PD -> desired acceleration (ax_des, ay_des)
2. Thrust Allocation: 
   - T = m * sqrt(ax_des^2 + (ay_des + g)^2)
   - theta_des = atan2(-ax_des, ay_des + g)
3. Attitude PD -> torque tau
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simple_model import state_space
from utils import wrap_angle


class PIDController:
    """Simple PID controller."""
    
    def __init__(self, kp: float, ki: float, kd: float, output_limits: tuple = (None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min, self.output_max = output_limits
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_call = True
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_call = True
    
    def compute(self, error: float, dt: float) -> float:
        # Proportional
        p_term = self.kp * error
        
        # Integral
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative
        if self.first_call:
            d_term = 0.0
            self.first_call = False
        else:
            d_term = self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        
        output = p_term + i_term + d_term
        
        # Clamp
        if self.output_min is not None:
            output = max(self.output_min, output)
        if self.output_max is not None:
            output = min(self.output_max, output)
        
        return output


class PlanarQuadrotor:
    """
    2D Planar Quadrotor with position control.
    
    Coordinate system:
    - x: horizontal (positive = right)
    - y: vertical (positive = up)
    - theta: pitch angle (positive = nose down / tilted right)
    """
    
    # Physical parameters
    GRAVITY = 9.81
    
    def __init__(self,
                 initial_states: list,
                 mass: float = 1.0,
                 inertia: float = 0.1,
                 drag_coefficient: list = [0.1, 0.1],
                 dt: float = 0.01,
                 # Position control gains
                 kp_pos: tuple = (2.0, 2.0),      # (kp_x, kp_y)
                 kd_pos: tuple = (2.0, 2.0),      # (kd_x, kd_y)
                 # Attitude control gains
                 kp_att: float = 20.0,
                 kd_att: float = 5.0,
                 # Limits
                 max_tilt: float = math.pi / 4,   # 45 degrees
                 max_thrust: float = 30.0,        # N
                 max_torque: float = 5.0):        # Nm
        
        self.mass = mass
        self.inertia = inertia
        self.drag_coefficient = drag_coefficient
        self.dt = dt
        
        # Gains
        self.kp_x, self.kp_y = kp_pos
        self.kd_x, self.kd_y = kd_pos
        self.kp_att = kp_att
        self.kd_att = kd_att
        
        # Limits
        self.max_tilt = max_tilt
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        
        # States
        self.states = np.array(initial_states, dtype=float)
        
        # Setpoint (position)
        self.x_des = initial_states[0]
        self.y_des = initial_states[1]
        
        # Logging
        self.history = {
            'time': [], 'x': [], 'y': [], 'vx': [], 'vy': [],
            'theta': [], 'omega': [], 'theta_des': [],
            'x_des': [], 'y_des': [],
            'thrust': [], 'torque': [],
            'x_error': [], 'y_error': [], 'theta_error': []
        }
    
    def set_position_setpoint(self, x_des: float, y_des: float):
        """Set desired position."""
        self.x_des = x_des
        self.y_des = y_des
    
    def compute_control(self) -> tuple:
        """
        Cascaded control: Position -> Desired Accel -> Thrust + Desired Theta -> Torque
        """
        x, y, vx, vy, theta, omega = self.states
        
        # ===== Outer Loop: Position Control =====
        # Position errors
        ex = self.x_des - x
        ey = self.y_des - y
        
        # Desired accelerations (PD control)
        ax_des = self.kp_x * ex - self.kd_x * vx
        ay_des = self.kp_y * ey - self.kd_y * vy
        
        # ===== Thrust Allocation =====
        # Total desired acceleration (including gravity compensation)
        # We need: ax = -(T/m)*sin(theta), ay = (T/m)*cos(theta) - g
        # Rearranging: (T/m)*sin(theta) = -ax_des, (T/m)*cos(theta) = ay_des + g
        
        ay_total = ay_des + self.GRAVITY  # Total vertical thrust component
        
        # Thrust magnitude
        thrust = self.mass * math.sqrt(ax_des**2 + ay_total**2)
        thrust = np.clip(thrust, 0.0, self.max_thrust)
        
        # Desired pitch angle
        # theta_des = atan2(-ax_des, ay_total)
        theta_des = math.atan2(-ax_des, ay_total)
        theta_des = np.clip(theta_des, -self.max_tilt, self.max_tilt)
        
        # ===== Inner Loop: Attitude Control =====
        theta_error = wrap_angle(theta_des - theta)
        
        # PD control for attitude
        torque = self.kp_att * theta_error - self.kd_att * omega
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        
        # Store for logging
        self._theta_des = theta_des
        self._errors = (ex, ey, theta_error)
        
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
        self.history['time'].append(time)
        self.history['x'].append(self.states[0])
        self.history['y'].append(self.states[1])
        self.history['vx'].append(self.states[2])
        self.history['vy'].append(self.states[3])
        self.history['theta'].append(self.states[4])
        self.history['omega'].append(self.states[5])
        self.history['theta_des'].append(self._theta_des)
        self.history['x_des'].append(self.x_des)
        self.history['y_des'].append(self.y_des)
        self.history['thrust'].append(thrust)
        self.history['torque'].append(torque)
        self.history['x_error'].append(self._errors[0])
        self.history['y_error'].append(self._errors[1])
        self.history['theta_error'].append(self._errors[2])
    
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
        ax.set_title('2D Planar Quadrotor')
        
        # Trajectory
        traj_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)
        
        # Drone body
        drone_body, = ax.plot([], [], 'k-', linewidth=3)
        
        # Thrust vector
        thrust_arrow = ax.arrow(0, 0, 0, 0.5, head_width=0.1, fc='red', ec='red')
        
        # Setpoint marker
        setpoint_marker, = ax.plot([], [], 'g*', markersize=15)
        
        # Info text
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        x_hist, y_hist = [], []
        num_steps = int(duration / self.dt)
        steps_per_frame = max(1, int(0.03 / self.dt))
        
        def init():
            ax.set_xlim(-2, 10)
            ax.set_ylim(-2, 10)
            return traj_line, drone_body, setpoint_marker, info_text
        
        def update(frame):
            nonlocal thrust_arrow
            
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
            
            # Draw drone body (a line representing the quadrotor arms)
            arm_len = 0.3
            dx = arm_len * math.cos(theta)
            dy = arm_len * math.sin(theta)
            drone_body.set_data([x - dx, x + dx], [y - dy, y + dy])
            
            # Draw thrust vector
            thrust_arrow.remove()
            thrust_mag = self.history['thrust'][-1] if self.history['thrust'] else 0
            thrust_scale = 0.05  # Scale factor for visualization
            thrust_dx = -thrust_scale * thrust_mag * math.sin(theta)
            thrust_dy = thrust_scale * thrust_mag * math.cos(theta)
            thrust_arrow = ax.arrow(x, y, thrust_dx, thrust_dy, 
                                   head_width=0.08, fc='red', ec='red')
            
            # Setpoint
            setpoint_marker.set_data([self.x_des], [self.y_des])
            
            # Info
            info = (f'Time: {len(self.history["time"]) * self.dt:.2f}s\n'
                   f'Pos: ({x:.2f}, {y:.2f})\n'
                   f'Vel: ({vx:.2f}, {vy:.2f})\n'
                   f'θ: {math.degrees(theta):.1f}° (des: {math.degrees(self._theta_des):.1f}°)\n'
                   f'Thrust: {thrust_mag:.2f} N\n'
                   f'Setpoint: ({self.x_des:.1f}, {self.y_des:.1f})')
            info_text.set_text(info)
            
            # Auto-adjust view
            margin = 2
            ax.set_xlim(min(x_hist) - margin, max(max(x_hist), self.x_des) + margin)
            ax.set_ylim(min(min(y_hist), 0) - margin, max(max(y_hist), self.y_des) + margin)
            
            return traj_line, drone_body, setpoint_marker, info_text
        
        num_frames = num_steps // steps_per_frame + 1
        ani = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                           blit=False, interval=30, repeat=False)
        plt.show()
    
    def plot_results(self):
        """Plot simulation results."""
        fig, axes = plt.subplots(4, 2, figsize=(12, 12))
        
        # Trajectory (X-Y)
        axes[0, 0].plot(self.history['x'], self.history['y'], 'b-', linewidth=1.5)
        axes[0, 0].plot(self.history['x'][0], self.history['y'][0], 'go', markersize=10, label='Start')
        axes[0, 0].plot(self.history['x'][-1], self.history['y'][-1], 'ro', markersize=10, label='End')
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
        axes[0, 1].plot(self.history['time'], self.history['x_des'], 'b--', alpha=0.5, label='x_des')
        axes[0, 1].plot(self.history['time'], self.history['y_des'], 'r--', alpha=0.5, label='y_des')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Position (m)')
        axes[0, 1].set_title('Position Tracking')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Position errors
        axes[1, 0].plot(self.history['time'], self.history['x_error'], 'b-', label='x_error')
        axes[1, 0].plot(self.history['time'], self.history['y_error'], 'r-', label='y_error')
        axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Error (m)')
        axes[1, 0].set_title('Position Errors')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Attitude
        axes[1, 1].plot(self.history['time'], np.rad2deg(self.history['theta']), 'b-', label='θ')
        axes[1, 1].plot(self.history['time'], np.rad2deg(self.history['theta_des']), 'r--', label='θ_des')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Angle (deg)')
        axes[1, 1].set_title('Pitch Angle')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Angular velocity
        axes[2, 0].plot(self.history['time'], self.history['omega'], 'b-')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('ω (rad/s)')
        axes[2, 0].set_title('Angular Velocity')
        axes[2, 0].grid(True)
        
        # Velocities
        axes[2, 1].plot(self.history['time'], self.history['vx'], 'b-', label='vx')
        axes[2, 1].plot(self.history['time'], self.history['vy'], 'r-', label='vy')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Velocity (m/s)')
        axes[2, 1].set_title('Velocities')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # Thrust
        axes[3, 0].plot(self.history['time'], self.history['thrust'], 'b-')
        axes[3, 0].axhline(y=self.mass * self.GRAVITY, color='r', linestyle='--', label='Hover thrust')
        axes[3, 0].set_xlabel('Time (s)')
        axes[3, 0].set_ylabel('Thrust (N)')
        axes[3, 0].set_title('Thrust')
        axes[3, 0].legend()
        axes[3, 0].grid(True)
        
        # Torque
        axes[3, 1].plot(self.history['time'], self.history['torque'], 'b-')
        axes[3, 1].set_xlabel('Time (s)')
        axes[3, 1].set_ylabel('Torque (Nm)')
        axes[3, 1].set_title('Torque')
        axes[3, 1].grid(True)
        
        plt.tight_layout()
        plt.show()


def run_demo():
    """Demo: Hover at origin, then fly to a waypoint."""
    
    # Initial state: [x, y, vx, vy, theta, omega]
    # Start at origin, hovering
    initial_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    drone = PlanarQuadrotor(
        initial_states=initial_states,
        mass=1.0,
        inertia=0.1,
        drag_coefficient=[0.1, 0.1],
        dt=0.01,
        kp_pos=(1.5, 1.5),
        kd_pos=(2.0, 2.0),
        kp_att=30.0,
        kd_att=8.0,
        max_tilt=math.pi / 4,
        max_thrust=25.0,
        max_torque=5.0
    )
    
    # Set waypoint
    drone.set_position_setpoint(x_des=10, y_des=3.0)
    
    print("2D Planar Quadrotor Simulation")
    print("=" * 40)
    print(f"Mass: {drone.mass} kg")
    print(f"Hover thrust: {drone.mass * drone.GRAVITY:.2f} N")
    print(f"Initial position: (0, 0)")
    print(f"Target position: ({drone.x_des}, {drone.y_des})")
    print("=" * 40)
    
    # Run live visualization
    drone.run_live(duration=10.0)
    
    # Final state
    print(f"\nFinal position: ({drone.states[0]:.3f}, {drone.states[1]:.3f})")
    print(f"Final velocity: ({drone.states[2]:.3f}, {drone.states[3]:.3f})")
    
    # Plot results
    drone.plot_results()


if __name__ == "__main__":
    run_demo()
