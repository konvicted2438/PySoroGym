import numpy as np
from .minkowski import Simplex, support_function


def gjk_distance(shape_a, shape_b, max_iterations=64):
    """GJK algorithm for computing distance between two convex shapes.
    
    Parameters
    ----------
    shape_a : Shape
        First convex shape
    shape_b : Shape
        Second convex shape
    max_iterations : int
        Maximum number of iterations
        
    Returns
    -------
    distance : float
        Minimum distance between shapes (0 if intersecting)
    simplex : Simplex
        Final simplex containing closest features
    """
    simplex = Simplex()
    
    # Initial direction - from shape B to shape A
    direction = shape_a.body.pos - shape_b.body.pos if hasattr(shape_a, 'body') and hasattr(shape_b, 'body') else np.array([1, 0, 0])
    
    if np.linalg.norm(direction) < 1e-10:
        direction = np.array([1, 0, 0])
    else:
        direction = direction / np.linalg.norm(direction)
    
    # Get first support point
    v, v1, v2 = support_function(shape_a, shape_b, direction)
    simplex.add_point(v, v1, v2)
    
    # New search direction is toward origin
    direction = -v
    
    for iteration in range(max_iterations):
        # Normalize direction
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-10:
            direction = direction / direction_norm
        
        # Get new support point
        v, v1, v2 = support_function(shape_a, shape_b, direction)
        
        # Check if we're still moving toward the origin
        if np.dot(v, direction) < 0:
            # Can't get closer to origin - compute distance
            closest_point = simplex.v[0]  # Best guess of closest point
            return np.linalg.norm(closest_point), simplex
        
        # Add new point to simplex
        simplex.add_point(v, v1, v2)
        
        # Update simplex and check if origin is contained
        contains_origin, new_direction = do_simplex(simplex)
        
        if contains_origin:
            # For EPA, we need a tetrahedron containing the origin
            # If we found a collision with a smaller simplex, build it up to a tetrahedron
            if len(simplex) < 4:
                return build_tetrahedron(shape_a, shape_b, simplex)
            
            # Shapes are intersecting
            return 0.0, simplex
        
        # Update search direction
        direction = new_direction
    
    # If we reach here, algorithm didn't converge
    # Return current best distance estimate
    return np.linalg.norm(direction), simplex


def build_tetrahedron(shape_a, shape_b, simplex):
    """
    Build up a simplex to a tetrahedron for EPA.
    """
    # We already know the origin is inside the current simplex
    if len(simplex) == 3:
        # We have a triangle, find a direction perpendicular to it
        a = simplex.get_point(0)
        b = simplex.get_point(1)
        c = simplex.get_point(2)
        
        # Get the normal to the triangle
        normal = np.cross(b - a, c - a)
        normal_length = np.linalg.norm(normal)
        
        if normal_length > 1e-10:
            normal = normal / normal_length
            # Try both directions perpendicular to the triangle
            v1, _, _ = support_function(shape_a, shape_b, normal)
            v2, _, _ = support_function(shape_a, shape_b, -normal)
            
            # Use the point furthest in its direction
            if np.dot(v1, normal) > np.dot(v2, -normal):
                simplex.add_point(*support_function(shape_a, shape_b, normal))
            else:
                simplex.add_point(*support_function(shape_a, shape_b, -normal))
    
    elif len(simplex) == 2:
        # We have a line segment
        a = simplex.get_point(0)
        b = simplex.get_point(1)
        
        # Find two directions perpendicular to the line
        ab = b - a
        ab_norm = np.linalg.norm(ab)
        
        if ab_norm > 1e-10:
            ab = ab / ab_norm
            
            # Find a perpendicular direction
            perp = np.array([1, 0, 0])
            if abs(np.dot(ab, perp)) > 0.9:
                perp = np.array([0, 1, 0])
            
            # First perpendicular direction
            perp1 = np.cross(ab, perp)
            perp1_norm = np.linalg.norm(perp1)
            if perp1_norm > 1e-10:
                perp1 = perp1 / perp1_norm
                simplex.add_point(*support_function(shape_a, shape_b, perp1))
            
            # Second perpendicular direction (to form a tetrahedron)
            perp2 = np.cross(ab, perp1)
            perp2_norm = np.linalg.norm(perp2)
            if perp2_norm > 1e-10:
                perp2 = perp2 / perp2_norm
                simplex.add_point(*support_function(shape_a, shape_b, perp2))
    
    # No matter what, return collision
    return 0.0, simplex


def do_simplex(simplex):
    """Update simplex to contain closest feature to origin.
    
    Parameters
    ----------
    simplex : Simplex
        Current simplex
        
    Returns
    -------
    contains_origin : bool
        True if simplex contains origin
    direction : array, shape (3,)
        New search direction
    """
    if len(simplex) == 2:
        return do_line(simplex)
    elif len(simplex) == 3:
        return do_triangle(simplex)
    elif len(simplex) == 4:
        return do_tetrahedron(simplex)
    else:
        raise ValueError(f"Invalid simplex size: {len(simplex)}")


def do_line(simplex):
    """Process line simplex."""
    a = simplex.get_point(1)  # Most recent point
    b = simplex.get_point(0)  # Previous point
    
    ab = b - a
    ao = -a
    
    if np.dot(ab, ao) > 0:
        # Origin is in direction of line
        direction = np.cross(np.cross(ab, ao), ab)
        if np.linalg.norm(direction) < 1e-10:
            # Origin is on the line - collision detected
            return True, np.zeros(3)
    else:
        # Origin is closest to A
        simplex.set_points([1])
        direction = ao
        
    return False, direction


def do_triangle(simplex):
    """Process triangle simplex."""
    a = simplex.get_point(2)  # Most recent point
    b = simplex.get_point(1)
    c = simplex.get_point(0)
    
    ab = b - a
    ac = c - a
    ao = -a
    
    # Triangle normal
    abc = np.cross(ab, ac)
    
    # Check if origin is in direction of triangle face
    if np.dot(np.cross(ab, abc), ao) > 0:
        # Origin is outside edge AB
        if np.dot(ab, ao) > 0:
            # Origin is in direction of AB
            simplex.set_points([2, 1])  # Keep A and B
            direction = np.cross(np.cross(ab, ao), ab)
            return False, direction
        else:
            # Origin is closest to A
            simplex.set_points([2])  # Keep only A
            direction = ao
            return False, direction
            
    if np.dot(np.cross(abc, ac), ao) > 0:
        # Origin is outside edge AC
        if np.dot(ac, ao) > 0:
            # Origin is in direction of AC
            simplex.set_points([2, 0])  # Keep A and C
            direction = np.cross(np.cross(ac, ao), ac)
            return False, direction
        else:
            # Origin is closest to A
            simplex.set_points([2])  # Keep only A
            direction = ao
            return False, direction
            
    # Origin is above/below triangle face
    if np.dot(abc, ao) > 0:
        # Origin is above triangle
        direction = abc
        return False, direction
    else:
        # Origin is below triangle - likely contains origin
        # Check if origin is actually contained in the triangle
        if is_origin_in_triangle(a, b, c):
            return True, np.zeros(3)
        else:
            direction = -abc
            return False, direction


def is_origin_in_triangle(a, b, c):
    """Check if origin is contained in the triangle."""
    # Using barycentric coordinates
    v0 = b - a
    v1 = c - a
    v2 = -a
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return False
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    # If u, v, w are all between 0 and 1, origin is in triangle
    return (u >= -1e-10) and (v >= -1e-10) and (w >= -1e-10) and (u + v + w <= 1 + 1e-10)


def do_tetrahedron(simplex):
    """Process tetrahedron simplex."""
    a = simplex.get_point(3)  # Most recent point
    b = simplex.get_point(2)
    c = simplex.get_point(1)
    d = simplex.get_point(0)
    
    ab = b - a
    ac = c - a
    ad = d - a
    ao = -a
    
    # Face normals pointing outward
    abc = np.cross(ab, ac)
    acd = np.cross(ac, ad)
    adb = np.cross(ad, ab)
    
    # Ensure normals point outward
    if np.dot(abc, ad) > 0:
        abc = -abc
    if np.dot(acd, ab) > 0:
        acd = -acd
    if np.dot(adb, ac) > 0:
        adb = -adb
    
    # Check if origin is behind any of the faces
    if np.dot(abc, ao) > 0:
        simplex.set_points([3, 2, 1])  # Keep A, B, C
        return do_triangle(simplex)
        
    if np.dot(acd, ao) > 0:
        simplex.set_points([3, 1, 0])  # Keep A, C, D
        return do_triangle(simplex)
        
    if np.dot(adb, ao) > 0:
        simplex.set_points([3, 0, 2])  # Keep A, D, B
        return do_triangle(simplex)
        
    # If we get here, the origin is inside the tetrahedron
    return True, np.zeros(3)