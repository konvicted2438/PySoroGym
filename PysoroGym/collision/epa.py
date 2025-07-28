import numpy as np
from .minkowski import support_function

class Face:
    """Triangle face for EPA algorithm."""
    def __init__(self, a, b, c, vertices):
        self.vertices = vertices
        self.indices = [a, b, c]
        
        # Calculate normal and distance
        self.normal, self.distance = self.calculate_normal_and_distance()
    
    def calculate_normal_and_distance(self):
        """Calculate face normal and distance to origin."""
        a = self.vertices[self.indices[0]]
        b = self.vertices[self.indices[1]]
        c = self.vertices[self.indices[2]]
        
        # Calculate normal
        normal = np.cross(b - a, c - a)
        normal_length = np.linalg.norm(normal)
        
        if normal_length < 1e-10:
            return np.array([0, 0, 1]), 0
        
        normal = normal / normal_length
        
        # Ensure normal points away from the origin
        distance = np.dot(normal, a)
        if distance < 0:
            normal = -normal
            distance = -distance
            
        return normal, distance


def epa(shape_a, shape_b, simplex, max_iterations=64, epsilon=1e-6):
    """Expanding Polytope Algorithm for penetration depth and contact normal.
    
    Following distance3d's approach without special case handling.
    
    Parameters
    ----------
    shape_a : Shape
        First shape
    shape_b : Shape
        Second shape
    simplex : Simplex
        Initial simplex from GJK (must contain origin)
    max_iterations : int
        Maximum iterations
    epsilon : float
        Convergence tolerance
        
    Returns
    -------
    normal : array, shape (3,)
        Contact normal (from A to B)
    depth : float
        Penetration depth
    contact_a : array, shape (3,)
        Contact point on shape A
    contact_b : array, shape (3,)
        Contact point on shape B
    """
    # Create lists to store the polytope vertices and support points
    vertices = []  # Minkowski difference vertices
    support_a = []  # Support points on shape A
    support_b = []  # Support points on shape B
    
    # Copy points from simplex to our local lists
    for i in range(simplex.n_points):
        vertices.append(simplex.v[i].copy())
        support_a.append(simplex.v1[i].copy())
        support_b.append(simplex.v2[i].copy())
    
    # Build initial polytope faces
    faces = []
    
    if len(vertices) == 4:
        # For tetrahedron, we need 4 triangular faces
        faces = [
            Face(0, 1, 2, vertices),
            Face(0, 3, 1, vertices),
            Face(0, 2, 3, vertices),
            Face(1, 3, 2, vertices)
        ]
    else:
        # Build tetrahedron from whatever we have
        if len(vertices) == 3:
            # We have a triangle, add a point above/below it
            faces = [Face(0, 1, 2, vertices)]
            
            direction = faces[0].normal
            v, v1, v2 = support_function(shape_a, shape_b, direction)
            vertices.append(v)
            support_a.append(v1)
            support_b.append(v2)
            
            faces.append(Face(0, 3, 1, vertices))
            faces.append(Face(0, 2, 3, vertices))
            faces.append(Face(1, 3, 2, vertices))
            
        elif len(vertices) == 2:
            # We have a line segment, build a tetrahedron around it
            a = vertices[0]
            b = vertices[1]
            
            # Find perpendicular directions
            ab = b - a
            ab_length = np.linalg.norm(ab)
            if ab_length < 1e-10:
                # Degenerate case
                direction1 = np.array([1, 0, 0])
                direction2 = np.array([0, 1, 0])
            else:
                ab = ab / ab_length
                
                # Find a perpendicular direction
                perp = np.array([1, 0, 0])
                if abs(np.dot(ab, perp)) > 0.9:
                    perp = np.array([0, 1, 0])
                
                direction1 = np.cross(ab, perp)
                direction1_length = np.linalg.norm(direction1)
                if direction1_length > 1e-10:
                    direction1 = direction1 / direction1_length
                
                direction2 = np.cross(ab, direction1)
                direction2_length = np.linalg.norm(direction2)
                if direction2_length > 1e-10:
                    direction2 = direction2 / direction2_length
            
            # Add two more points
            v1, v1_a, v1_b = support_function(shape_a, shape_b, direction1)
            vertices.append(v1)
            support_a.append(v1_a)
            support_b.append(v1_b)
            
            v2, v2_a, v2_b = support_function(shape_a, shape_b, direction2)
            vertices.append(v2)
            support_a.append(v2_a)
            support_b.append(v2_b)
            
            # Create faces
            faces = [
                Face(0, 1, 2, vertices),
                Face(0, 3, 1, vertices),
                Face(0, 2, 3, vertices),
                Face(1, 3, 2, vertices)
            ]
    
    # Main EPA loop - following distance3d's approach
    for iteration in range(max_iterations):
        # Find face closest to the origin
        closest_face = min(faces, key=lambda face: face.distance)
        
        # Get new support point in the direction of the closest face normal
        new_minkowski, new_support_a, new_support_b = support_function(
            shape_a, shape_b, closest_face.normal
        )
        
        # Check convergence - if new point isn't significantly further than closest face
        dist_to_new_point = np.dot(new_minkowski, closest_face.normal)
        
        if dist_to_new_point - closest_face.distance < epsilon:
            # We've converged!
            normal = closest_face.normal
            depth = closest_face.distance
            
            # Calculate contact points following distance3d approach
            # The contact points are derived from the support points that form
            # the closest face, weighted by their contribution to the closest point
            
            # Get the closest point on the face to the origin
            closest_point_on_face = normal * depth
            
            # Get face vertices
            face_indices = closest_face.indices
            a = vertices[face_indices[0]]
            b = vertices[face_indices[1]]
            c = vertices[face_indices[2]]
            
            # Calculate barycentric coordinates
            weights = calculate_barycentric_coordinates(a, b, c, closest_point_on_face)
            
            # Contact points are weighted combinations of the support points
            contact_a = (
                weights[0] * support_a[face_indices[0]] +
                weights[1] * support_a[face_indices[1]] +
                weights[2] * support_a[face_indices[2]]
            )
            
            contact_b = (
                weights[0] * support_b[face_indices[0]] +
                weights[1] * support_b[face_indices[1]] +
                weights[2] * support_b[face_indices[2]]
            )
            
            return normal, depth, contact_a, contact_b
        
        # Add new point to polytope
        new_idx = len(vertices)
        vertices.append(new_minkowski)
        support_a.append(new_support_a)
        support_b.append(new_support_b)
        
        # Find all faces visible from the new point
        visible_faces = []
        visible_face_indices = []
        
        for i, face in enumerate(faces):
            # A face is visible if the new point is on the positive side of its plane
            if np.dot(face.normal, new_minkowski - vertices[face.indices[0]]) > epsilon:
                visible_faces.append(face)
                visible_face_indices.append(i)
        
        # Find the boundary edges of visible faces
        edges = {}
        for face in visible_faces:
            for i in range(3):
                edge = (face.indices[i], face.indices[(i + 1) % 3])
                reversed_edge = (edge[1], edge[0])
                
                if reversed_edge in edges:
                    # Internal edge - remove it
                    del edges[reversed_edge]
                else:
                    # Boundary edge - keep it
                    edges[edge] = True
        
        # Remove visible faces
        new_faces = [face for i, face in enumerate(faces) if i not in visible_face_indices]
        
        # Create new faces from boundary edges to new point
        for edge in edges:
            new_faces.append(Face(edge[0], edge[1], new_idx, vertices))
        
        faces = new_faces
        
        # Safety check: if no faces remain, return a fallback result
        if not faces:
            # This can happen with degenerate cases or numerical precision issues
            # Return a minimal penetration along the y-axis (common for ground contact)
            return np.array([0.0, 1.0, 0.0]), 0.001, vertices[0], vertices[0] + np.array([0.0, 0.001, 0.0])
    
    # Max iterations reached - return best estimate
    closest_face = min(faces, key=lambda face: face.distance)
    normal = closest_face.normal
    depth = closest_face.distance
    
    # Get approximate contact points
    closest_point = normal * depth
    face_indices = closest_face.indices
    weights = calculate_barycentric_coordinates(
        vertices[face_indices[0]], 
        vertices[face_indices[1]], 
        vertices[face_indices[2]], 
        closest_point
    )
    
    contact_a = (
        weights[0] * support_a[face_indices[0]] +
        weights[1] * support_a[face_indices[1]] +
        weights[2] * support_a[face_indices[2]]
    )
    
    contact_b = (
        weights[0] * support_b[face_indices[0]] +
        weights[1] * support_b[face_indices[1]] +
        weights[2] * support_b[face_indices[2]]
    )
    
    return normal, depth, contact_a, contact_b


def calculate_barycentric_coordinates(a, b, c, p):
    """Calculate barycentric coordinates of point p with respect to triangle (a,b,c).
    
    This follows the standard approach used in distance3d.
    """
    # Vectors from a to other vertices
    v0 = b - a
    v1 = c - a
    v2 = p - a
    
    # Calculate dot products
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    # Calculate barycentric coordinates
    denom = d00 * d11 - d01 * d01
    
    if abs(denom) < 1e-10:
        # Degenerate triangle - return equal weights
        return np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    # Clamp to valid range [0, 1] and normalize
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    w = max(0.0, min(1.0, w))
    
    # Normalize to ensure they sum to 1
    total = u + v + w
    if total > 1e-10:
        return np.array([u/total, v/total, w/total])
    
    return np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])


