import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PysoroGym.World import World
from PysoroGym.Body import Body
from PysoroGym.physics import Material
from PysoroGym.visulisation import OpenGLRenderer
from PysoroGym.shapes import Box
from PysoroGym.soft_robot_body import SoftRobotBody
from PysoroGym.soft_robot_controller import SoftRobotController


def main():
    """Example of a single body with two segments"""
    print("Multi-Segment Soft Robot Body Example")
    print("=" * 40)
    
    # Create renderer and world
    renderer = OpenGLRenderer(width=1024, height=768)
    
    # Use smaller timestep for better stability (same as validation script)
    world = World(gravity=(0, -9.81, 0), renderer=renderer, dt=1.0/240.0)
    
    # Create ground
    ground = Body(body_type=Body.STATIC)
    ground_material = Material(friction=1.0, elasticity=0.0)  # No bounce
    ground.add_shape(Box([100, 0.1, 100], color=(0.204, 0.275, 0.329)), material=ground_material)
    ground.position = np.array([0, -0.05, 0])
    world.add(ground)

    box = Body(mass=1, position=(0.25, 0.0, 0.25))  # Start higher to avoid initial penetration
    box_material = Material(friction=0.8, elasticity=0.1)  # Low bounce
    box.linear_damping = 0.1   # Lower damping (same as validation)
    box.angular_damping = 0.2   # Lower angular damping (same as validation)
    box_size = [0.3, 0.3, 0.3]
    box_shape = Box(box_size, color=(0.8, 0.3, 0.3))
    box.add_shape(box_shape, material=box_material)
    world.add(box)

    # Create a soft robot with two segments of different sizes
    segment_configs = [
        {
            'L': 0.05,  # 50mm length
            'rad_robot': 0.008,  # 8mm radius
            'pressures': [0.0, 0.0, 0.0],
            'tip_force': [0.0, 0.0, 0.0]  # 20mN downward at junction
        },
        {
            'L': 0.05,  # 30mm length
            'rad_robot': 0.008,  # 8mm radius
            'pressures': [0.0, 0.0, 0.0],
            'tip_force': [0.0, 0.0, 0.0]  # No external force at tip
        }
    ]
    
    # Create rotation matrix to point the robot downward
    # The kinematics solver outputs the robot pointing along +z
    # We want it pointing along -y (downward)
    # This requires a +90 degree rotation around the x-axis
    angle = np.pi/2  # +90 degrees
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    
    # Create the multi-segment robot as a single body
    multi_robot = SoftRobotBody(
        position=(0, 0.5, 0),
        orientation=rotation_matrix,
        n_segments_per_robot=15,
        n_sides=12,
        material=Material(friction=0.8, elasticity=0.2),
        scale=10.0,
        fixed_base=True,
        segment_configs=segment_configs
    )
    
    world.add(multi_robot)
    
    # Create controller for the multi-segment robot
    # This will control all segments
    controller = SoftRobotController(multi_robot, interpolation_time=2.0)

    # --- Data Collection Setup ---
    tip_positions = []
    # --- End Data Collection Setup ---

    
    # Define a sequence of pressure commands for both segments
    # Each command is (pressures_for_all_segments, duration)
    pressure_sequence = [
        # Start at zero
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 0.5),
        
        # Activate segment 1, chamber 1 - should bend in one direction
        ([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 3.0),
        
        # Activate segment 1, chamber 2 - should bend 120 degrees around
        ([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], 3.0),
        
        # Activate segment 1, chamber 3 - should bend another 120 degrees
        ([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], 3.0),
        
        # Activate segment 2, chamber 1 - should bend in the opposite direction
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 3.0),
        
        # Activate segment 2, chamber 2 - should bend 120 degrees around
        ([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 3.0),
        
        # Activate segment 2, chamber 3 - should bend another 120 degrees
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], 3.0),
        
        # Both segments together, chamber 1
        ([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], 3.0),
        
        # Both segments together, chamber 2
        ([[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]], 3.0),
        
        # Both segments together, chamber 3
        ([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]], 3.0),
        
        # Complex pattern
        ([[0.3, 0.3, 0.0], [0.0, 0.3, 0.3]], 3.0),
        
        # Different pressures
        ([[0.6, 0.0, 0.2], [0.2, 0.0, 0.6]], 3.0),
        
        # Return to zero
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 3.0),
    ]
    
    # Set the sequence
    controller.set_pressure_sequence(pressure_sequence)
    
    # Update callback with reduced debug output
    last_print_time = 0
    def physics_update(dt):
        nonlocal last_print_time
        controller.update(dt, world.simulation_time)
        
        # Only print debug info and collect data occasionally (every 0.1 seconds)
        if world.simulation_time - last_print_time > 0.1:
            # Get tip position and store it
            tip_pos = multi_robot.get_tip_position()
            tip_positions.append(tip_pos)
            
            collision_pairs = world.physics_engine.collision_detector.detect_collisions(world.physics_engine.bodies)
            
            # Check for robot collisions
            robot_collisions = []
            for pair in collision_pairs:
                if pair.body_a is multi_robot or pair.body_b is multi_robot:
                    robot_collisions.append(pair)
            
            if robot_collisions:
                print(f"\nTime: {world.simulation_time:.2f}s - Robot collisions detected: {len(robot_collisions)}")
            
            # Check proximity
            box_center = box.position
            distance = np.linalg.norm(tip_pos - box_center)
            
            
            last_print_time = world.simulation_time
    
    world.physics_update_callback = physics_update
    
    # Run simulation
    world.run()

    # --- Plotting ---
    # Plot the collected tip trajectory after the simulation is complete
    print("Simulation finished. Plotting trajectory...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if tip_positions:
        traj_x = [p[0] for p in tip_positions]
        traj_y = [p[1] for p in tip_positions]
        traj_z = [p[2] for p in tip_positions]
        ax.plot(traj_x, traj_y, traj_z, 'b-') # Line for trajectory
        ax.plot([traj_x[0]], [traj_y[0]], [traj_z[0]], 'go', label='Start')   # Marker for start position
        ax.plot([traj_x[-1]], [traj_y[-1]], [traj_z[-1]], 'ro', label='End')   # Marker for end position

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Soft Robot Tip Trajectory')
    ax.legend()
    # Auto-scale axes to fit the data
    ax.set_aspect('equal') # auto-scaling
    plt.show()


if __name__ == "__main__":
    main()
