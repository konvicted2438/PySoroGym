import numpy as np
import pygame
from PysoroGym.World import World
from PysoroGym.Body import Body
from PysoroGym.physics import Material
from PysoroGym.visulisation import OpenGLRenderer
from PysoroGym.shapes import Box, Sphere, Cylinder, Capsule, ConvexHullVertices, MeshGraph


def main():
    """Validation script for PysoroGym physics engine"""
    print("PysoroGym Physics Engine Validation")
    print("====================================")
    print("Controls:")
    print("  WASD/Mouse: Camera movement")
    print("  P: Pause/Resume simulation")
    print("  SPACE: Drop a random object")
    print("  R: Reset")
    print("  N: Toggle contact normal visualization")
    print("  +/-: Increase/decrease normal arrow length")
    print("  ESC: Exit")
    
    # Create renderer
    renderer = OpenGLRenderer(width=1024, height=768)
    
    # Create physics world with renderer - smaller timestep for stability
    world = World(gravity=(0, -9.81, 0), renderer=renderer, dt=1.0/240.0)
    
    # Create ground plane (static)
    ground = Body(body_type=Body.STATIC)
    ground_material = Material(friction=1.0, elasticity=0.0)  # No bounce, high friction
    
    # Use a large box as ground
    ground_shape = Box([20, 0.5, 20])
    ground.position = np.array([0, -0.25, 0])  # Position it so top surface is at y=0
    ground.add_shape(ground_shape, material=ground_material)
    world.add(ground)
    
    # # Add some initial objects (a box)
    box = Body(mass=1, position=(0.1, 3, 0))
    box_material = Material(friction=0.8, elasticity=0.1)  # Low bounce
    box.linear_damping = 0.1   # Add damping
    box.angular_damping = 0.2   # Higher angular damping
    box_size = [0.5, 0.5, 0.5]
    box_shape = Box(box_size)
    box.add_shape(box_shape, material=box_material)
    world.add(box)
    
    # # Add a sphere too
    sphere = Body(mass=1, position=(0, 1, 0))
    sphere_material = Material(friction=0.6, elasticity=0.2)
    sphere.linear_damping = 0.1
    sphere.angular_damping = 0.2
    sphere_shape = Sphere(radius=0.3)
    sphere.add_shape(sphere_shape, material=sphere_material)
    world.add(sphere)
    
    # Create a T-shaped object
    def add_t_shape(world, position=(0, 5, 0)):
        # Define the dimensions of the T (original upright definition)
        width = 2.0      # Width of the top bar
        height = 2.0     # Total height
        thickness = 0.3  # Thickness in z-direction
        stem_width = 0.4 # Width of the vertical part
        stem_height = 1.2 # Height of the stem part
        
        # Calculate key dimensions
        bar_height = height - stem_height  # Height of the horizontal bar
        
        # Define vertices of the T-shape (16 vertices total) - UNCHANGED
        vertices = np.array([
            # Top bar vertices (8 vertices: 0-7)
            [-width/2, height/2, thickness/2], [width/2, height/2, thickness/2],
            [width/2, height/2 - bar_height, thickness/2], [-width/2, height/2 - bar_height, thickness/2],
            [-width/2, height/2, -thickness/2], [width/2, height/2, -thickness/2],
            [width/2, height/2 - bar_height, -thickness/2], [-width/2, height/2 - bar_height, -thickness/2],
            # Stem vertices (8 vertices: 8-15)
            [-stem_width/2, height/2 - bar_height, thickness/2], [stem_width/2, height/2 - bar_height, thickness/2],
            [stem_width/2, -height/2, thickness/2], [-stem_width/2, -height/2, thickness/2],
            [-stem_width/2, height/2 - bar_height, -thickness/2], [stem_width/2, height/2 - bar_height, -thickness/2],
            [stem_width/2, -height/2, -thickness/2], [-stem_width/2, -height/2, -thickness/2],
        ], dtype=np.float64)
        
        # Define triangles (counter-clockwise winding) - UNCHANGED
        triangles = np.array([
            [0, 5, 1], [0, 4, 5], [0, 1, 2], [0, 2, 3], [5, 4, 7], [5, 7, 6],
            [4, 0, 3], [4, 3, 7], [1, 5, 6], [1, 6, 2], [8, 9, 10], [8, 10, 11],
            [13, 12, 15], [13, 15, 14], [12, 8, 11], [12, 11, 15], [9, 13, 14],
            [9, 14, 10], [11, 10, 14], [11, 14, 15], [3, 8, 12], [3, 12, 7],
            [2, 6, 13], [2, 13, 9], [3, 2, 9], [3, 9, 8], [7, 12, 13], [7, 13, 6],
        ], dtype=np.int64)
        
        # Debug info
        print(f"T-shape vertices: {len(vertices)}")
        print(f"T-shape triangles: {len(triangles)}")
        print(f"Vertex bounds: min={np.min(vertices, axis=0)}, max={np.max(vertices, axis=0)}")
        
        # Create the T-shape using MeshGraph. This will calculate the COM correctly
        # for the upright shape.
        t_shape = MeshGraph(vertices, triangles)
        
        # --- KEY CHANGES START HERE ---

        # 1. Define the rotation to make the T lie flat.
        # We rotate it 90 degrees (pi/2 radians) around the X-axis.
        angle = np.pi / 2.0
        axis = np.array([1.0, 0.0, 0.0])
        s = np.sin(angle / 2.0)
        c = np.cos(angle / 2.0)
        initial_orientation = np.array([c, s * axis[0], s * axis[1], s * axis[2]])

        # 2. Determine the correct height.
        # The lowest point of the original mesh is at z = -thickness/2.
        # After rotating 90 degrees around X, this becomes the new lowest y-point.
        # So, the body needs to be positioned at a height of `thickness/2` to sit on the ground.
        initial_y_position = thickness / 2.0
        
        # Create the body with the calculated initial orientation and position.
        t_body = Body(
            mass=1, 
            position=(position[0], initial_y_position, position[2]),
            orientation=initial_orientation
        )
        
        # --- KEY CHANGES END HERE ---

        t_material = Material(friction=0.8, elasticity=0.1)
        t_body.linear_damping = 0.1
        t_body.angular_damping = 0.2
        
        # Set angular velocity to zero so it starts perfectly still.
        t_body.angular_velocity = (0, 0, 0)
        
        # Add the T-shape to the body
        t_body.add_shape(t_shape, material=t_material)
        
        # Add to world
        world.add(t_body)
        return t_body
    
    # Add the T-shape to the world, starting flat at the origin
    add_t_shape(world, position=(0, 0, 0))
    
    # Run the simulation
    world.run()


if __name__ == "__main__":
    main()