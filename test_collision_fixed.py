import numpy as np
import sys
import os

# Add the parent directory to the path so we can import PysoroGym modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the new collision modules directly
from PysoroGym.collision.minkowski import Simplex, support_function
from PysoroGym.collision.gjk_distance import gjk_distance
from PysoroGym.collision.epa import epa
from PysoroGym.collision.collision import detect_collision

# Import shapes and bodies
from PysoroGym.Body import Body
from PysoroGym.Shape import Box, Sphere

print("=== Testing Fixed Collision Detection System ===\n")

# Test Case 1: Two overlapping spheres (obvious case)
print("Test 1: Two overlapping spheres")
body1 = Body(position=(0, 0, 0))
sphere1 = Sphere(radius=1.0)
body1.add_shape(sphere1)

body2 = Body(position=(1.5, 0, 0))  # Centers 1.5 apart, radii sum to 2.0, so overlap by 0.5
sphere2 = Sphere(radius=1.0)
body2.add_shape(sphere2)

shape_a = body1.shapes[0]
shape_b = body2.shapes[0]
shape_a.body = body1
shape_b.body = body2

result = detect_collision(shape_a, shape_b)
if result:
    print(f"  Collision detected! ✓")
    print(f"  Normal: {result['normal']}")
    print(f"  Depth: {result['depth']}")
    print(f"  Contact A: {result['contact_a']}")
    print(f"  Contact B: {result['contact_b']}")
else:
    print(f"  No collision ✗ (INCORRECT - should detect collision)")

print("\n=== Test complete ===")