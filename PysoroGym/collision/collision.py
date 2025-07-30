"""
Collision detection interface that uses GJK/EPA algorithms.
"""
import numpy as np
from .gjk_distance import gjk_distance
from .epa import epa
from ..collision_resolution import Contact
from ..Shape import Plane
from ..math_utils import q_to_mat3 # Import the quaternion to matrix conversion utility


def detect_collision(collider_a, collider_b):
    """
    Detects collision between two ShapeColliders.
    Dispatches to specialized functions if available (e.g., for Planes).
    """
    # Get the actual shape objects from their colliders
    shape_a = collider_a.shape
    shape_b = collider_b.shape

    # --- Dispatcher ---
    # Check if either shape is a Plane and call the specialized function
    if isinstance(shape_a, Plane):
        # The first object is the plane, call the function directly
        return detect_plane_collision(collider_a, collider_b)
    
    if isinstance(shape_b, Plane):
        # The second object is the plane. We need to call the function with
        # the plane as the *first* argument, then fix the resulting contact.
        contact = detect_plane_collision(collider_b, collider_a)
        if contact:
            # The contact was generated from B to A. We need to flip it
            # so it's consistently from A to B.
            contact.normal *= -1.0
            contact.shape_a, contact.shape_b = contact.shape_b, contact.shape_a
            contact.contact_a, contact.contact_b = contact.contact_b, contact.contact_a
        return contact

    # --- Fallback to Generic GJK/EPA ---
    #print(f"\n=== GJK/EPA COLLISION DETECTION ===")
    # Use GJK for distance query
    distance, simplex = gjk_distance(collider_a, collider_b)
    
    # If shapes are separated, no collision
    if distance > 1e-6:
        return None
    
    # Shapes are colliding, use EPA to get contact information
    normal, depth, contact_a, contact_b = epa(collider_a, collider_b, simplex)
    
    # Create contact using the standardized constructor
    contact = Contact(
        collider_a=collider_a,
        collider_b=collider_b,
        normal=normal,
        depth=depth,
        contact_a=contact_a,
        contact_b=contact_b
    )
    
    return contact


def detect_plane_collision(plane_collider, other_collider):
    """
    Special case collision detection for infinite plane.
    This function now expects ShapeCollider objects.
    """
    plane_shape = plane_collider.shape
    other_shape = other_collider.shape

    # print(f"\n=== PLANE COLLISION DETECTION ===")
    # print(f"Plane shape: {plane_shape}")
    # print(f"Other shape: {other_shape}")
    # print(f"Other shape type: {type(other_shape.shape).__name__ if hasattr(other_shape, 'shape') else type(other_shape).__name__}")
    
    # Convert the body's quaternion to a rotation matrix
    rotation_matrix = q_to_mat3(plane_collider.body.q)
    
    # Plane normal in world space 
    plane_normal = rotation_matrix @ plane_shape.normal
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    plane_point = plane_collider.body.pos
    
    # print(f"Plane normal (local): {plane_shape.normal}")
    # print(f"Plane normal (world): {plane_normal}")
    # print(f"Plane point: {plane_point}")
    
    # Find the lowest point on the other shape in the direction of -plane_normal
    # The world_support method belongs to the collider, not the raw shape.
    lowest_point = other_collider.world_support(-plane_normal)
    # print(f"Search direction: {-plane_normal}")
    # print(f"Lowest point found: {lowest_point}")
    # print(f"Other shape position: {other_collider.body.pos}")
    
    # Distance from point to plane
    distance = np.dot(lowest_point - plane_point, plane_normal)
    # print(f"Distance to plane: {distance}")
    
    if distance > 1e-6:
        #print("No collision detected")
        return None
    
    # Contact point on plane
    contact_on_plane = lowest_point - distance * plane_normal
    #print(f"Contact on plane: {contact_on_plane}")
    
    # Create contact using the standardized constructor
    contact = Contact(
        collider_a=plane_collider,
        collider_b=other_collider,
        normal=plane_normal,
        depth=-distance,  # Penetration depth
        contact_a=contact_on_plane,
        contact_b=lowest_point
    )
    
    #print(f"Created contact with depth: {contact.depth}")
    # print("=== END PLANE COLLISION DETECTION ===\n")
    
    return contact


def gjk(shape_a, shape_b):
    """
    Wrapper function for backward compatibility.
    Detects collision and returns contact information.
    """
    return detect_collision(shape_a, shape_b)