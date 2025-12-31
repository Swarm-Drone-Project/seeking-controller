import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from simple_model import state_space
from utils import wrap_angle


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


class DroneSetpointController:
    """
    Controller for a 2D drone model using speed and phi setpoints.
    
    States: [x, y, vx, vy, theta, omega]
    Inputs: [u1 (thrust), u2 (torque)]
    
    Controlled variables:
    - speed: magnitude of velocity sqrt(vx^2 + vy^2)
    - phi: direction of velocity atan2(vy, vx)
    """
    
    def __init__(self, 
                 initial_states: list,
                 speed_pid_gains: tuple,
                 phi_pid_gains: tuple,
                 drag_coefficient: list,
                 mass: float,
                 inertia: float,
                 dt: float):
        """
        Initialize the drone controller.
        
        Args:
            initial_states: [x, y, vx, vy, theta, omega] initial state
            speed_pid_gains: (Kp, Ki, Kd) for speed control -> u1
            phi_pid_gains: (Kp, Ki, Kd) for phi control -> u2
            drag_coefficient: [inertial_drag, rotational_drag]
            mass: drone mass
            inertia: moment of inertia
            dt: time step
        """
        self.drag_coefficient = drag_coefficient
        self.mass = mass
        self.inertia = inertia
        self.dt = dt
        
        # Initialize PID controllers
        self.speed_pid = PIDController(*speed_pid_gains, output_limits=(0.0, 20.0))
        self.phi_pid = PIDController(*phi_pid_gains, output_limits=(-10.0, 10.0))
        
        # Initial states: [x, y, vx, vy, theta, omega]
        if initial_states is not None:
            self.states = np.array(initial_states)
        else:
            self.states = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])  # Small initial vx to avoid atan2(0,0)
        
        # Setpoints
        self.speed_setpoint = 0.0
        self.phi_setpoint = 0.0
        
        # Data logging
        self.history = {
            'time': [],
            'x': [],
            'y': [],
            'vx': [],
            'vy': [],
            'theta': [],
            'omega': [],
            'speed': [],
            'phi': [],
            'speed_setpoint': [],
            'phi_setpoint': [],
            'speed_error': [],
            'phi_error': [],
            'u1': [],
            'u2': []
        }
    
    def get_speed_and_phi(self) -> tuple:
        """Calculate speed and phi from current states."""
        vx, vy = self.states[2], self.states[3]
        speed = math.sqrt(vx**2 + vy**2)
        phi = math.atan2(vy, vx)
        return speed, phi
    
    def set_setpoints(self, speed_setpoint: float, phi_setpoint: float):
        """Set the desired speed and phi setpoints."""
        self.speed_setpoint = speed_setpoint
        self.phi_setpoint = phi_setpoint
    
    def compute_control(self) -> tuple:
        """
        Compute control inputs u1 and u2 based on setpoints.
        
        Returns:
            (u1, u2): thrust and torque inputs
        """
        speed, phi = self.get_speed_and_phi()
        
        # PID on speed to get u1 (thrust)
        u1 = self.speed_pid.compute(self.speed_setpoint, speed, self.dt)
        
        # For phi control, we need to control the heading theta
        # The idea: to change phi, we need to orient thrust (theta) appropriately
        # A simple approach: use phi error to generate a desired omega via PID -> u2
        phi_error = wrap_angle(self.phi_setpoint - phi)
        
        # We want theta to track phi_setpoint for the velocity to align
        # PID on phi error to get desired angular rate, then u2 to achieve it
        u2 = self.phi_pid.compute(0, -phi_error, self.dt)
        
        return u1, u2
    
    def step(self):
        """Perform one simulation step."""
        # Compute control inputs
        u1, u2 = self.compute_control()
        inputs = [u1, u2]
        
        # Get state derivatives from the model
        state_dots = state_space(
            self.states.tolist(), 
            inputs, 
            self.drag_coefficient, 
            self.mass, 
            self.inertia
        )
        
        # Euler integration
        self.states = self.states + np.array(state_dots) * self.dt
        
        # Wrap theta to [-pi, pi]
        self.states[4] = wrap_angle(self.states[4])
        
        return u1, u2
    
    def log_state(self, time: float, u1: float, u2: float):
        """Log current state for plotting."""
        speed, phi = self.get_speed_and_phi()
        speed_error = self.speed_setpoint - speed
        phi_error = wrap_angle(self.phi_setpoint - phi)
        
        self.history['time'].append(time)
        self.history['x'].append(self.states[0])
        self.history['y'].append(self.states[1])
        self.history['vx'].append(self.states[2])
        self.history['vy'].append(self.states[3])
        self.history['theta'].append(self.states[4])
        self.history['omega'].append(self.states[5])
        self.history['speed'].append(speed)
        self.history['phi'].append(phi)
        self.history['speed_setpoint'].append(self.speed_setpoint)
        self.history['phi_setpoint'].append(self.phi_setpoint)
        self.history['speed_error'].append(speed_error)
        self.history['phi_error'].append(phi_error)
        self.history['u1'].append(u1)
        self.history['u2'].append(u2)
    
    def run_simulation(self, duration: float):
        """
        Run the simulation for a given duration.
        
        Args:
            duration: total simulation time in seconds
        """
        num_steps = int(duration / self.dt)
        
        for i in range(num_steps):
            time = i * self.dt
            
            # Compute control and step
            u1, u2 = self.step()
            
            # Log state
            self.log_state(time, u1, u2)
    
    def run_live(self, duration: float):
        """
        Run the simulation with live visualization.
        
        Args:
            duration: total simulation time in seconds
        """
        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Drone Setpoint Control - Live')
        
        # Initialize trajectory line
        trajectory_line, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5, label='Trajectory')
        
        # Initialize drone position marker
        drone_marker, = ax.plot([], [], 'ko', markersize=10)
        
        # Initialize theta arrow (heading direction)
        arrow_length = 1.0
        theta_arrow = ax.arrow(0, 0, arrow_length, 0, head_width=0.15, head_length=0.1, 
                               fc='red', ec='red', label='Theta (heading)')
        
        # Initialize phi marker (velocity direction) as a dashed line
        phi_line, = ax.plot([], [], 'g--', linewidth=2, label='Phi (velocity dir)')
        
        # Text annotations
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Legend
        ax.legend(loc='upper right')
        
        # Storage for trajectory
        x_history = []
        y_history = []
        
        # Number of steps
        num_steps = int(duration / self.dt)
        steps_per_frame = max(1, int(0.03 / self.dt))  # ~30 FPS
        
        def init():
            ax.set_xlim(-5, 30)
            ax.set_ylim(-5, 30)
            return trajectory_line, drone_marker, phi_line, info_text
        
        def update(frame):
            nonlocal theta_arrow
            
            # Run multiple simulation steps per frame
            for _ in range(steps_per_frame):
                if len(self.history['time']) >= num_steps:
                    break
                time = len(self.history['time']) * self.dt
                u1, u2 = self.step()
                self.log_state(time, u1, u2)
            
            # Get current state
            x, y = self.states[0], self.states[1]
            theta = self.states[4]
            speed, phi = self.get_speed_and_phi()
            
            # Update trajectory
            x_history.append(x)
            y_history.append(y)
            trajectory_line.set_data(x_history, y_history)
            
            # Update drone marker
            drone_marker.set_data([x], [y])
            
            # Update theta arrow (remove old, add new)
            theta_arrow.remove()
            dx = arrow_length * math.cos(theta)
            dy = arrow_length * math.sin(theta)
            theta_arrow = ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.1,
                                  fc='red', ec='red')
            
            # Update phi line (velocity direction marker)
            phi_length = arrow_length * 1.5
            phi_dx = phi_length * math.cos(phi)
            phi_dy = phi_length * math.sin(phi)
            phi_line.set_data([x, x + phi_dx], [y, y + phi_dy])
            
            # Update info text
            info_str = (f'Time: {len(self.history["time"]) * self.dt:.2f}s\n'
                       f'Position: ({x:.2f}, {y:.2f})\n'
                       f'Speed: {speed:.2f} (sp: {self.speed_setpoint:.2f})\n'
                       f'Phi: {math.degrees(phi):.1f}° (sp: {math.degrees(self.phi_setpoint):.1f}°)\n'
                       f'Theta: {math.degrees(theta):.1f}°')
            info_text.set_text(info_str)
            
            # Auto-adjust view to follow drone
            margin = 5
            ax.set_xlim(min(x_history) - margin, max(max(x_history), x + margin) + margin)
            ax.set_ylim(min(y_history) - margin, max(max(y_history), y + margin) + margin)
            
            return trajectory_line, drone_marker, phi_line, info_text
        
        # Calculate number of frames
        num_frames = num_steps // steps_per_frame + 1
        
        ani = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                           blit=False, interval=30, repeat=False)
        
        plt.show()
    
    def plot_results(self):
        """Plot simulation results."""
        fig, axes = plt.subplots(5, 2, figsize=(12, 14))
        
        # Position trajectory
        axes[0, 0].plot(self.history['x'], self.history['y'], 'b-', linewidth=1.5)
        axes[0, 0].plot(self.history['x'][0], self.history['y'][0], 'go', markersize=10, label='Start')
        axes[0, 0].plot(self.history['x'][-1], self.history['y'][-1], 'ro', markersize=10, label='End')
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Y Position')
        axes[0, 0].set_title('Trajectory')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].axis('equal')
        
        # Speed tracking
        axes[0, 1].plot(self.history['time'], self.history['speed'], 'b-', label='Speed', linewidth=1.5)
        axes[0, 1].plot(self.history['time'], self.history['speed_setpoint'], 'r--', label='Setpoint', linewidth=1.5)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Speed')
        axes[0, 1].set_title('Speed Tracking')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Phi tracking
        axes[1, 0].plot(self.history['time'], np.rad2deg(self.history['phi']), 'b-', label='Phi', linewidth=1.5)
        axes[1, 0].plot(self.history['time'], np.rad2deg(self.history['phi_setpoint']), 'r--', label='Setpoint', linewidth=1.5)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Phi (degrees)')
        axes[1, 0].set_title('Phi (Velocity Direction) Tracking')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Theta (heading)
        axes[1, 1].plot(self.history['time'], np.rad2deg(self.history['theta']), 'b-', linewidth=1.5)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Theta (degrees)')
        axes[1, 1].set_title('Heading Angle')
        axes[1, 1].grid(True)
        
        # Speed Error (Magnitude Error)
        axes[2, 0].plot(self.history['time'], self.history['speed_error'], 'b-', linewidth=1.5)
        axes[2, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Speed Error')
        axes[2, 0].set_title('Speed Error (Magnitude Error)')
        axes[2, 0].grid(True)
        
        # Phi Error
        axes[2, 1].plot(self.history['time'], np.rad2deg(self.history['phi_error']), 'b-', linewidth=1.5)
        axes[2, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Phi Error (degrees)')
        axes[2, 1].set_title('Phi Error')
        axes[2, 1].grid(True)
        
        # Omega (angular velocity)
        axes[3, 0].plot(self.history['time'], self.history['omega'], 'b-', linewidth=1.5)
        axes[3, 0].set_xlabel('Time (s)')
        axes[3, 0].set_ylabel('Omega (rad/s)')
        axes[3, 0].set_title('Angular Velocity')
        axes[3, 0].grid(True)
        
        # Control input u1 (Thrust)
        axes[3, 1].plot(self.history['time'], self.history['u1'], 'b-', linewidth=1.5)
        axes[3, 1].set_xlabel('Time (s)')
        axes[3, 1].set_ylabel('u1 (Thrust)')
        axes[3, 1].set_title('Thrust Input')
        axes[3, 1].grid(True)
        
        # Control input u2 (Torque)
        axes[4, 0].plot(self.history['time'], self.history['u2'], 'b-', linewidth=1.5)
        axes[4, 0].set_xlabel('Time (s)')
        axes[4, 0].set_ylabel('u2 (Torque)')
        axes[4, 0].set_title('Torque Input')
        axes[4, 0].grid(True)
        
        # Hide unused subplot
        axes[4, 1].axis('off')
        
        plt.tight_layout()
        plt.show()


def run_demo():
    """Run a demonstration of the setpoint controller."""
    # Create controller with arbitrary dt and initial state
    dt = 0.05  # 50ms time step
    controller = DroneSetpointController(
        initial_states=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # [x, y, vx, vy, theta, omega]
        speed_pid_gains=(0.0, 0.0, 0.0),
        phi_pid_gains=(1.0, 0.0, 0.0),
        drag_coefficient=[0.0, 0.0],
        mass=1.0,
        inertia=1.0,
        dt=dt
    )
    
    # Set single setpoint: speed and phi
    speed_setpoint = 3.0
    phi_setpoint = math.pi / 4  # 45 degrees
    controller.set_setpoints(speed_setpoint, phi_setpoint)
    
    # Run simulation with live visualization
    print("Running setpoint control simulation with live visualization...")
    print(f"Time step (dt): {dt}s")
    print(f"Simulation duration: 10s")
    print(f"\nSetpoint: speed={speed_setpoint:.1f}, phi={math.degrees(phi_setpoint):.1f}°")
    print("\nRed arrow = Theta (heading direction)")
    print("Green dashed = Phi (velocity direction)")
    
    controller.run_live(duration=20.0)
    
    # Print final state
    speed, phi = controller.get_speed_and_phi()
    print(f"\nFinal state:")
    print(f"  Position: ({controller.states[0]:.2f}, {controller.states[1]:.2f})")
    print(f"  Speed: {speed:.2f} (setpoint: {controller.speed_setpoint:.2f})")
    print(f"  Phi: {math.degrees(phi):.2f}° (setpoint: {math.degrees(controller.phi_setpoint):.2f}°)")
    controller.plot_results()

if __name__ == "__main__":
    run_demo()
