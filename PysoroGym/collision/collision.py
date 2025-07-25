"""
Collision detection interface that uses GJK/EPA algorithms.
"""
import numpy as np
from .gjk_distance import gjk_distance
from .epa import epa
from ..collision_resolution import Contact
from ..Shape import Plane


def detect_collision(shape_a, shape_b):
    """
    Detect collision between two shapes using GJK/EPA.
    
    Parameters
    ----------
    shape_a : Shape
        First shape
    shape_b : Shape
        Second shape
        
    Returns
    -------
    contact : Contact or None
        Contact information if collision detected, None otherwise
    """
    # Special case for plane collision
    if isinstance(shape_a, Plane):
        return detect_plane_collision(shape_a, shape_b)
    elif isinstance(shape_b, Plane):
        contact = detect_plane_collision(shape_b, shape_a)
        if contact:
            # Flip the contact normal and swap contact points
            contact.normal = -contact.normal
            contact.contact_a, contact.contact_b = contact.contact_b, contact.contact_a
            contact.shape_a, contact.shape_b = shape_a, shape_b
            contact.body_a, contact.body_b = shape_a.body, shape_b.body
        return contact
    
    # Use GJK for distance query
    distance, simplex = gjk_distance(shape_a, shape_b)
    
    # If shapes are separated, no collision
    if distance > 1e-6:
        return None
    
    # Shapes are colliding, use EPA to get contact information
    normal, depth, contact_a, contact_b = epa(shape_a, shape_b, simplex)
    
    # Create contact
    contact = Contact(
        shape_a=shape_a,
        shape_b=shape_b,
        normal=normal,
        depth=depth,
        contact_a=contact_a,
        contact_b=contact_b
    )
    
    return contact


def detect_plane_collision(plane_shape, other_shape):
    """
    Special case collision detection for infinite plane.
    
    Parameters
    ----------
    plane_shape : Plane
        The plane shape
    other_shape : Shape
        The other shape to test against the plane
        
    Returns
    -------
    contact : Contact or None
        Contact information if collision detected
    """
    # Plane normal in world space (assuming plane normal is [0, 1, 0] in local space)
    plane_normal = plane_shape.body.Q @ np.array([0, 1, 0])
    plane_point = plane_shape.body.pos
    
    # Find the lowest point on the other shape in the direction of -plane_normal
    lowest_point = other_shape.world_support(-plane_normal)
    
    # Distance from point to plane
    distance = np.dot(lowest_point - plane_point, plane_normal)
    
    if distance > 1e-6:
        # No collision
        return None
    
    # Contact point on plane
    contact_on_plane = lowest_point - distance * plane_normal
    
    # Create contact
    contact = Contact(
        shape_a=plane_shape,
        shape_b=other_shape,
        normal=plane_normal,
        depth=-distance,  # Penetration depth
        contact_a=contact_on_plane,
        contact_b=lowest_point
    )
    
    return contact


def gjk(shape_a, shape_b):
    """
    Wrapper function for backward compatibility.
    Detects collision and returns contact information.
    """
    return detect_collision(shape_a, shape_b)