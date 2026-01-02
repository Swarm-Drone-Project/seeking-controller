"""
Main entry point for 2D Planar Quadrotor Simulation.
"""

import argparse
import os
import yaml
import numpy as np
from plant import PlanarQuadrotorDynamics
from hierarchical_controller import HierarchicalController
from visualization import PlanarVisualizer
from utils import wrap_angle

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    # If path is not absolute, make it relative to script directory
    if not os.path.isabs(config_path):
        config_path = os.path.join(SCRIPT_DIR, config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_plant(config: dict) -> PlanarQuadrotorDynamics:
    """Create plant from config."""
    return PlanarQuadrotorDynamics(
        initial_state=config['simulation']['initial_state'],
        mass=config['plant']['mass'],
        inertia=config['plant']['inertia'],
        drag_coefficient=config['plant']['drag_coefficient']
    )


def create_controller(config: dict, control_mode: str) -> HierarchicalController:
    """Create controller from config."""
    ctrl = config['controller']
    return HierarchicalController(
        mass=config['plant']['mass'],
        # Position controller
        pos_kp=tuple(ctrl['position']['kp']),
        pos_ki=tuple(ctrl['position']['ki']),
        pos_kd=tuple(ctrl['position']['kd']),
        max_velocity=tuple(ctrl['position']['max_velocity']),
        # Velocity controller
        vel_kp=tuple(ctrl['velocity']['kp']),
        vel_ki=tuple(ctrl['velocity']['ki']),
        vel_kd=tuple(ctrl['velocity']['kd']),
        max_acceleration=tuple(ctrl['velocity']['max_acceleration']),
        max_tilt=ctrl['velocity']['max_tilt'],
        max_thrust=ctrl['velocity']['max_thrust'],
        # Attitude controller
        att_kp=ctrl['attitude']['kp'],
        att_ki=ctrl['attitude']['ki'],
        att_kd=ctrl['attitude']['kd'],
        max_torque=ctrl['attitude']['max_torque'],
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
    python main.py --mode position --setpoint 10.0 3.0
    
  Velocity control at (2, 1) m/s:
    python main.py --mode velocity --setpoint 2.0 1.0
    
  Run for 20 seconds without live animation:
    python main.py --mode position --setpoint 5 5 --duration 20 --no-live
    
  Use custom config file:
    python main.py --config my_config.yaml --mode position --setpoint 5 5
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
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
        default=None,
        help='Simulation duration in seconds (overrides config file)'
    )
    
    parser.add_argument(
        '--no-live',
        action='store_true',
        help='Skip live animation, only show result plots'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override duration if specified on command line
    duration = args.duration if args.duration else config['simulation']['duration']
    dt = config['simulation']['dt']
    
    # Initialize components
    plant = create_plant(config)
    controller = create_controller(config, args.mode)
    
    # Set setpoints
    if args.mode == 'position':
        if args.setpoint is None:
            x_des, y_des = 10.0, 3.0
        else:
            x_des, y_des = args.setpoint
        controller.set_position_setpoint(x_des, y_des)
        setpoint_str = f"Position: ({x_des}, {y_des}) m"
        setpoint_tuple = (x_des, y_des)
    else:
        if args.setpoint is None:
            vx_des, vy_des = 2.0, 1.0
        else:
            vx_des, vy_des = args.setpoint
        controller.set_velocity_setpoint(vx_des, vy_des)
        setpoint_str = f"Velocity: ({vx_des}, {vy_des}) m/s"
        setpoint_tuple = None
        
    # Logging history
    history = {
        'time': [],
        'x': [], 'y': [], 'vx': [], 'vy': [], 'theta': [], 'omega': [],
        'x_des': [], 'y_des': [],
        'vx_des': [], 'vy_des': [],
        'theta_des': [],
        'ax_des': [], 'ay_des': [],
        'ax': [], 'ay': [],
        'thrust': [], 'torque': [],
        'x_error': [], 'y_error': [],
        'vx_error': [], 'vy_error': [],
        'theta_error': []
    }
    
    print("=" * 60)
    print("2D Planar Quadrotor - Hierarchical Cascaded Controller")
    print("=" * 60)
    print(f"\nConfig: {args.config}")
    print(f"Control Mode: {args.mode.upper()}")
    print(f"Setpoint: {setpoint_str}")
    print(f"Duration: {duration}s")
    print("\nRunning simulation...")
    
    # Simulation Loop
    num_steps = int(duration / dt)
    for i in range(num_steps):
        time = i * dt
        
        # Get current state
        state = plant.get_state()
        x, y, vx, vy, theta, omega = state
        
        # Compute control
        thrust, torque = controller.compute_control(state, dt)
        
        # Step physics
        plant.step(thrust, torque, dt)
        
        # Log data
        history['time'].append(time)
        history['x'].append(x)
        history['y'].append(y)
        history['vx'].append(vx)
        history['vy'].append(vy)
        history['theta'].append(theta)
        history['omega'].append(omega)
        
        history['x_des'].append(controller.x_des)
        history['y_des'].append(controller.y_des)
        history['vx_des'].append(controller.vx_des)
        history['vy_des'].append(controller.vy_des)
        history['theta_des'].append(controller.theta_des)
        
        history['ax_des'].append(controller.ax_des)
        history['ay_des'].append(controller.ay_des)
        history['ax'].append(plant.ax)
        history['ay'].append(plant.ay)
        
        history['thrust'].append(thrust)
        history['torque'].append(torque)
        
        history['x_error'].append(controller.x_des - x)
        history['y_error'].append(controller.y_des - y)
        history['vx_error'].append(controller.vx_des - vx)
        history['vy_error'].append(controller.vy_des - vy)
        history['theta_error'].append(wrap_angle(controller.theta_des - theta))
        
    # Final state
    final_state = plant.get_state()
    print(f"\nFinal position: ({final_state[0]:.3f}, {final_state[1]:.3f})")
    print(f"Final velocity: ({final_state[2]:.3f}, {final_state[3]:.3f})")
    
    # Visualization
    viz = PlanarVisualizer(history, args.mode, plant.mass)
    
    if not args.no_live:
        print("\nStarting live animation...")
        viz.animate(dt, setpoint_tuple)
        
    viz.plot_results()
    viz.plot_control_cascade()


if __name__ == "__main__":
    main()
