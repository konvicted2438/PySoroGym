import numpy as np
import pygame
from PysoroGym.World import World
from PysoroGym.Body import Body
from PysoroGym.physics import Material
from PysoroGym.visulisation import OpenGLRenderer
from PysoroGym.shapes import Box, Sphere
from PysoroGym.soft_robot_body import SoftRobotBody
from PysoroGym.soft_robot_controller import SoftRobotController


def main():
    """Example of soft robot with pressure control"""
    print("Soft Robot Pressure Control Example")
    print("===================================")
    print("The robot will follow a sequence of pressure commands")
    print("Press ESC to exit")
    
    # Create renderer and world
    renderer = OpenGLRenderer(width=1024, height=768)
    
    # Position camera for better view of the robot
    renderer.camera_pos = np.array([0.0, 2.0, 5.0])
    renderer.camera_pitch = -15.0
    
    world = World(gravity=(0, -9.81, 0), renderer=renderer, dt=1.0/60.0)
    
    # Create ground
    ground = Body(body_type=Body.STATIC)
    ground_material = Material(friction=1.0, elasticity=0.0)
    ground_shape = Box([10, 0.1, 10])
    ground.position = np.array([0, -0.05, 0])
    ground.add_shape(ground_shape, material=ground_material)
    world.add(ground)
    
    # Create soft robot
    robot_scale = 10.0  # Scale up by 10x
    soft_robot = SoftRobotBody(
        position=(0, 0.1, 0),  # Position slightly above ground
        n_segments=15,         # More segments for smoother bending
        n_sides=12,            # More sides for smoother cross-section
        material=Material(friction=0.8, elasticity=0.2),
        scale=robot_scale
    )
    world.add(soft_robot)
    
    # Create controller with 1 second default interpolation time
    controller = SoftRobotController(soft_robot, interpolation_time=1.0)
    
    # Define a sequence of pressure commands
    # Each command is (pressures [p1, p2, p3], duration to reach)
    pressure_sequence = [
        ([0.0, 0.0, 0.0], 1.0),    # Start at zero
        ([1.0, 0.0, 0.0], 2.0),    # Activate chamber 1
        ([0.0, 1.0, 0.0], 2.0),    # Switch to chamber 2
        ([0.0, 0.0, 1.0], 2.0),    # Switch to chamber 3
        ([0.5, 0.5, 0.0], 1.5),    # Combine chambers 1 & 2
        ([0.0, 0.5, 0.5], 1.5),    # Combine chambers 2 & 3
        ([0.5, 0.0, 0.5], 1.5),    # Combine chambers 1 & 3
        ([0.3, 0.3, 0.3], 2.0),    # All chambers equal
    ]
    
    # Set the pressure sequence
    controller.set_pressure_sequence(pressure_sequence)
    
    # Simple update callback - just update the controller
    def physics_update(dt):
        controller.update(dt, world.simulation_time)
    
    world.physics_update_callback = physics_update
    
    # Run simulation
    world.run()


if __name__ == "__main__":
    main()