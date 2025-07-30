import numpy as np
import pygame
from PysoroGym.world import World
from PysoroGym.Body import Body
from PysoroGym.Shape import Sphere, Box, Cylinder, Plane, Polyhedron
from PysoroGym.physics import Material
from PysoroGym.visulisation import OpenGLRenderer
from PysoroGym.math_utils import q_from_axis_angle

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
    
    # Create physics world with renderer
    world = World(gravity=(0, -9.81, 0), renderer=renderer, dt=1.0/120.0)
    
    # Create ground plane (static)
    ground = Body(body_type=Body.STATIC)
    ground_material = Material(friction=0.8, elasticity=0.3)
    ground_shape = Plane([20, 20], divisions=20)
    ground.add_shape(ground_shape, material=ground_material)
    world.add(ground)
    
    # Add some walls to contain objects
    wall_height = 5.0
    wall_thickness = 0.5
    wall_length = 10.0
    
    # # Back wall
    # back_wall = Body(body_type=Body.STATIC, position=(0, wall_height/2, -wall_length/2))
    # back_wall.add_shape(Box([wall_length, wall_height, wall_thickness]))
    # world.add(back_wall)
    
    # # Left wall
    # left_wall = Body(body_type=Body.STATIC, position=(-wall_length/2, wall_height/2, 0))
    # left_wall.add_shape(Box([wall_thickness, wall_height, wall_length]))
    # world.add(left_wall)
    
    # # Right wall
    # right_wall = Body(body_type=Body.STATIC, position=(wall_length/2, wall_height/2, 0))
    # right_wall.add_shape(Box([wall_thickness, wall_height, wall_length]))
    # world.add(right_wall)
    
    
    #Add a few spheres
    # for i in range(1):
    #     sphere = Body(mass=1.0, position=(i*0.1, 2 + i*2.5, 0))
    #     sphere_shape = Sphere(radius=0.5)
    #     sphere.add_shape(sphere_shape)
    #     world.add(sphere)
    
    # Add a cylinder
    # cylinder = Body(mass=2.0, position=(3, 7, 2))
    # cylinder_shape = Cylinder(radius=0.7, height=1.5)
    # cylinder.add_shape(cylinder_shape)
    # world.add(cylinder)
    for i in range(3):
        
        box = Body(mass=1, position=(0, 1+i*2.5, 0), angular_velocity=(1,1,1))
        box_size = [0.5, 0.5, 0.5]
        box_shape = Box(box_size) # Pass the list directly
        box.add_shape(box_shape)
        world.add(box)
    
    # # Add a compound shape (snowman-like)
    # snowman = Body(mass=3.0, position=(0, 8, 3))
    # # Base sphere
    # snowman.add_shape(Sphere(radius=0.8), offset=(0, -0.8, 0))
    # # Middle sphere
    # snowman.add_shape(Sphere(radius=0.6), offset=(0, 0.3, 0))
    # # Head sphere
    # snowman.add_shape(Sphere(radius=0.4), offset=(0, 1.1, 0))
    # world.add(snowman)
    
    # Add event handler for spawning objects
    
    # Verify that all shapes have body references
    for body in world.bodies:
        for shape in body.shapes:
            if shape.body is None or shape.body is not body:
                print(f"ERROR: Shape {type(shape).__name__} has incorrect body reference")
                # Fix it
                shape.body = body
                



                
    world.run()

if __name__ == "__main__":
    main()