import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import math

class PlanarVisualizer:
    """
    Visualization tools for 2D Planar Quadrotor.
    """
    
    def __init__(self, history: dict, control_mode: str, mass: float, gravity: float = 9.81):
        self.history = history
        self.control_mode = control_mode
        self.mass = mass
        self.gravity = gravity
        
    def plot_results(self):
        """Plot comprehensive simulation results."""
        fig, axes = plt.subplots(5, 2, figsize=(14, 16))
        mode_str = "Position Control" if self.control_mode == 'position' else "Velocity Control"
        fig.suptitle(f'Hierarchical Controller Results ({mode_str})', fontsize=14, fontweight='bold')
        
        # Trajectory (X-Y)
        axes[0, 0].plot(self.history['x'], self.history['y'], 'b-', linewidth=1.5)
        axes[0, 0].plot(self.history['x'][0], self.history['y'][0], 'go', markersize=10, label='Start')
        axes[0, 0].plot(self.history['x'][-1], self.history['y'][-1], 'ro', markersize=10, label='End')
        
        # Plot setpoint if available in history
        if self.control_mode == 'position' and self.history['x_des']:
            axes[0, 0].plot(self.history['x_des'][0], self.history['y_des'][0], 'g*', markersize=15, label='Setpoint')
            
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
        
        # Position errors
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
        axes[4, 0].axhline(y=self.mass * self.gravity, color='r', linestyle='--', 
                          label=f'Hover ({self.mass * self.gravity:.1f} N)')
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
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
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
        
        # Acceleration: Desired vs Actual
        ax2 = axes[1]
        ax2.set_title('Acceleration: Desired vs Actual')
        ax2.plot(self.history['time'], self.history['ax_des'], 'b-', label='ax_des', linewidth=2)
        ax2.plot(self.history['time'], self.history['ay_des'], 'r-', label='ay_des', linewidth=2)
        ax2.plot(self.history['time'], self.history['ax'], 'b--', alpha=0.5, label='ax (actual)')
        ax2.plot(self.history['time'], self.history['ay'], 'r--', alpha=0.5, label='ay (actual)')
        ax2.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Acceleration (m/s²)')
        ax2.legend()
        ax2.grid(True)
        
        # Level 2: Velocity -> Attitude + Thrust
        ax3 = axes[2]
        ax3.set_title('Level 2: Velocity Controller Output (Thrust & Attitude Setpoint)')
        ax3_twin = ax3.twinx()
        l1, = ax3.plot(self.history['time'], self.history['thrust'], 'g-', label='Thrust', linewidth=2)
        l2, = ax3_twin.plot(self.history['time'], np.rad2deg(self.history['theta_des']), 
                           'm-', label='θ_des', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Thrust (N)', color='g')
        ax3_twin.set_ylabel('θ_des (deg)', color='m')
        ax3.legend(handles=[l1, l2], loc='upper right')
        ax3.grid(True)
        
        # Level 3: Attitude -> Torque
        ax4 = axes[3]
        ax4.set_title('Level 3: Attitude Controller Output (Torque)')
        ax4.plot(self.history['time'], self.history['torque'], 'b-', label='Torque', linewidth=2)
        ax4.plot(self.history['time'], np.rad2deg(self.history['theta_error']), 
                'r--', alpha=0.5, label='θ_error (deg)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Torque (Nm) / Error (deg)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

    def animate(self, dt: float, setpoint: tuple = None):
        """Run animation from history."""
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
        
        # Setpoint marker
        setpoint_marker, = ax.plot([], [], 'g*', markersize=15)
        
        # Velocity vector (desired)
        vel_des_arrow = ax.arrow(0, 0, 0, 0, head_width=0.05, fc='blue', ec='blue', alpha=0.5)
        
        # Info text
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=8,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        num_steps = len(self.history['time'])
        steps_per_frame = max(1, int(0.03 / dt))
        
        def init():
            ax.set_xlim(-2, 12)
            ax.set_ylim(-2, 8)
            return traj_line, drone_body, setpoint_marker, info_text
        
        def update(frame):
            nonlocal thrust_arrow, vel_des_arrow
            
            idx = min(frame * steps_per_frame, num_steps - 1)
            
            x = self.history['x'][idx]
            y = self.history['y'][idx]
            vx = self.history['vx'][idx]
            vy = self.history['vy'][idx]
            theta = self.history['theta'][idx]
            
            # Update trajectory
            traj_line.set_data(self.history['x'][:idx+1], self.history['y'][:idx+1])
            
            # Draw drone body
            arm_len = 0.3
            dx = arm_len * math.cos(theta)
            dy = arm_len * math.sin(theta)
            drone_body.set_data([x - dx, x + dx], [y - dy, y + dy])
            
            # Draw thrust vector
            thrust_arrow.remove()
            thrust_mag = self.history['thrust'][idx]
            thrust_scale = 0.05
            thrust_dx = -thrust_scale * thrust_mag * math.sin(theta)
            thrust_dy = thrust_scale * thrust_mag * math.cos(theta)
            thrust_arrow = ax.arrow(x, y, thrust_dx, thrust_dy,
                                   head_width=0.08, fc='red', ec='red')
            
            # Draw desired velocity vector
            vel_des_arrow.remove()
            vel_scale = 0.3
            vx_des = self.history['vx_des'][idx]
            vy_des = self.history['vy_des'][idx]
            vel_des_arrow = ax.arrow(x, y, vx_des * vel_scale, vy_des * vel_scale,
                                    head_width=0.05, fc='cyan', ec='cyan', alpha=0.7)
            
            # Setpoint marker
            if self.control_mode == 'position' and setpoint:
                setpoint_marker.set_data([setpoint[0]], [setpoint[1]])
            else:
                setpoint_marker.set_data([], [])
            
            # Info
            theta_des = self.history['theta_des'][idx]
            torque = self.history['torque'][idx]
            
            if self.control_mode == 'position':
                info = (f'Time: {self.history["time"][idx]:.2f}s\n'
                       f'Mode: POSITION CONTROL\n'
                       f'─── Position ───\n'
                       f'Pos: ({x:.2f}, {y:.2f})\n'
                       f'─── Velocity ───\n'
                       f'Vel: ({vx:.2f}, {vy:.2f})\n'
                       f'Vel_des: ({vx_des:.2f}, {vy_des:.2f})\n'
                       f'─── Attitude ───\n'
                       f'θ: {math.degrees(theta):.1f}° (des: {math.degrees(theta_des):.1f}°)\n'
                       f'─── Commands ───\n'
                       f'Thrust: {thrust_mag:.2f} N | Torque: {torque:.2f} Nm')
            else:
                info = (f'Time: {self.history["time"][idx]:.2f}s\n'
                       f'Mode: VELOCITY CONTROL\n'
                       f'─── Position ───\n'
                       f'Pos: ({x:.2f}, {y:.2f})\n'
                       f'─── Velocity ───\n'
                       f'Vel: ({vx:.2f}, {vy:.2f})\n'
                       f'Vel_des: ({vx_des:.2f}, {vy_des:.2f}) [SETPOINT]\n'
                       f'─── Attitude ───\n'
                       f'θ: {math.degrees(theta):.1f}° (des: {math.degrees(theta_des):.1f}°)\n'
                       f'─── Commands ───\n'
                       f'Thrust: {thrust_mag:.2f} N | Torque: {torque:.2f} Nm')
            info_text.set_text(info)
            
            # Auto-adjust view
            margin = 2
            x_hist = self.history['x'][:idx+1]
            y_hist = self.history['y'][:idx+1]
            
            x_min = min(min(x_hist), 0) - margin
            x_max = max(max(x_hist), setpoint[0] if setpoint and self.control_mode == 'position' else max(x_hist)) + margin
            y_min = min(min(y_hist), 0) - margin
            y_max = max(max(y_hist), setpoint[1] if setpoint and self.control_mode == 'position' else max(y_hist)) + margin
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            return traj_line, drone_body, setpoint_marker, info_text
        
        num_frames = num_steps // steps_per_frame
        ani = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                           blit=False, interval=30, repeat=False)
        plt.show()
