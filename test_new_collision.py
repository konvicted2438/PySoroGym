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

print("=== Testing New Collision Detection System ===\n")

# Test Case 1: Box-Sphere collision (same as debug test)
print("Test 1: Box-Sphere Collision")
body1 = Body(position=(0, 0, 0))
box1 = Box([2, 2, 2])
body1.add_shape(box1)

body2 = Body(position=(1, 0, 0))
sphere2 = Sphere(radius=1.0)
body2.add_shape(sphere2)

shape_a = body1.shapes[0]
shape_b = body2.shapes[0]
shape_a.body = body1
shape_b.body = body2

# Test collision detection
result = detect_collision(shape_a, shape_b)
if result:
    print(f"  Collision detected!")
    print(f"  Normal: {result['normal']}")
    print(f"  Depth: {result['depth']}")
    print(f"  Contact A: {result['contact_a']}")
    print(f"  Contact B: {result['contact_b']}")
else:
    print(f"  No collision")

# Test Case 2: Separated spheres
print("\nTest 2: Separated Spheres")
body3 = Body(position=(0, 0, 0))
sphere3 = Sphere(radius=1.0)
body3.add_shape(sphere3)

body4 = Body(position=(3, 0, 0))
sphere4 = Sphere(radius=1.0)
body4.add_shape(sphere4)

shape_c = body3.shapes[0]
shape_d = body4.shapes[0]
shape_c.body = body3
shape_d.body = body4

result = detect_collision(shape_c, shape_d)
if result:
    print(f"  Collision detected!")
else:
    print(f"  No collision (expected)")

# Test Case 3: Touching spheres
print("\nTest 3: Touching Spheres")
body5 = Body(position=(0, 0, 0))
sphere5 = Sphere(radius=1.0)
body5.add_shape(sphere5)

body6 = Body(position=(2, 0, 0))
sphere6 = Sphere(radius=1.0)
body6.add_shape(sphere6)

shape_e = body5.shapes[0]
shape_f = body6.shapes[0]
shape_e.body = body5
shape_f.body = body6

result = detect_collision(shape_e, shape_f)
if result:
    print(f"  Collision detected!")
    print(f"  Normal: {result['normal']}")
    print(f"  Depth: {result['depth']}")
else:
    print(f"  No collision")

# Test Case 4: Deep penetration
print("\nTest 4: Deep Penetration (Box-Box)")
body7 = Body(position=(0, 0, 0))
box7 = Box([2, 2, 2])
body7.add_shape(box7)

body8 = Body(position=(0.5, 0, 0))
box8 = Box([2, 2, 2])
body8.add_shape(box8)

shape_g = body7.shapes[0]
shape_h = body8.shapes[0]
shape_g.body = body7
shape_h.body = body8

result = detect_collision(shape_g, shape_h)
if result:
    print(f"  Collision detected!")
    print(f"  Normal: {result['normal']}")
    print(f"  Depth: {result['depth']}")
    print(f"  Contact A: {result['contact_a']}")
    print(f"  Contact B: {result['contact_b']}")
else:
    print(f"  No collision")

print("\n=== All tests completed ===")