import numpy as np
from math import pi, cos, sin, sqrt

class ConvexShape:
    """Base class for all convex shapes with mesh representation."""
    def __init__(self):
        # Mesh representation
        self.vertices = np.array([], dtype=float)  # Vertex positions
        self.indices = np.array([], dtype=int)     # Triangle indices 
        self.normals = np.array([], dtype=float)   # Vertex normals
        
    def support(self, d: np.ndarray) -> np.ndarray:
        """Return furthest point in direction d for GJK."""
        raise NotImplementedError

    def inertia(self, mass: float):
        """Return principal moments Ixx, Iyy, Izz in local space."""
        raise NotImplementedError
    
    def get_mesh(self):
        """Return the mesh data for rendering."""
        return self.vertices, self.indices, self.normals


class Sphere(ConvexShape):
    def __init__(self, radius: float, resolution: int = 16):
        super().__init__()
        self.r = float(radius)
        
        # Generate sphere mesh
        self._generate_mesh(resolution)
        
    def _generate_mesh(self, resolution):
        """Generate vertices, indices and normals for a sphere."""
        vertices = []
        indices = []
        normals = []
        
        # Generate vertices
        for i in range(resolution + 1):
            phi = pi * i / resolution
            sin_phi = sin(phi)
            cos_phi = cos(phi)
            
            for j in range(resolution * 2):
                theta = 2 * pi * j / (resolution * 2)
                sin_theta = sin(theta)
                cos_theta = cos(theta)
                
                x = self.r * sin_phi * cos_theta
                y = self.r * cos_phi
                z = self.r * sin_phi * sin_theta
                
                vertices.append([x, y, z])
                normals.append([x/self.r, y/self.r, z/self.r])
        
        # Generate indices
        for i in range(resolution):
            for j in range(resolution * 2):
                next_j = (j + 1) % (resolution * 2)
                
                p1 = i * (resolution * 2) + j
                p2 = i * (resolution * 2) + next_j
                p3 = (i + 1) * (resolution * 2) + j
                p4 = (i + 1) * (resolution * 2) + next_j
                
                indices.append([p1, p2, p3])
                indices.append([p2, p4, p3])
        
        self.vertices = np.array(vertices, dtype=float)
        self.indices = np.array(indices, dtype=int)
        self.normals = np.array(normals, dtype=float)

    def support(self, d):
        n = np.linalg.norm(d)
        if n < 1e-10:
            return np.zeros(3)
        return (d / n) * self.r

    def inertia(self, m):
        I = 0.4 * m * self.r**2  # 2/5 m r²
        return np.array([I, I, I])


class Box(ConvexShape):
    def __init__(self, half_extents):
        super().__init__()
        self.h = np.asarray(half_extents, float)
        
        # Generate box mesh
        self._generate_mesh()
        
    def _generate_mesh(self):
        """Generate vertices, indices and normals for a box."""
        hx, hy, hz = self.h
        
        # 8 vertices for cube
        vertices = np.array([
            [-hx, -hy, -hz],  # 0: left-bottom-back
            [hx, -hy, -hz],   # 1: right-bottom-back
            [hx, hy, -hz],    # 2: right-top-back
            [-hx, hy, -hz],   # 3: left-top-back
            [-hx, -hy, hz],   # 4: left-bottom-front
            [hx, -hy, hz],    # 5: right-bottom-front
            [hx, hy, hz],     # 6: right-top-front
            [-hx, hy, hz]     # 7: left-top-front
        ], dtype=float)
        
        # 12 triangles (6 faces × 2 triangles)
        indices = np.array([
            # front (z+)
            [4, 5, 7], [5, 6, 7],
            # back (z-)
            [0, 3, 1], [1, 3, 2],
            # right (x+)
            [1, 2, 5], [2, 6, 5],
            # left (x-)
            [0, 4, 3], [3, 4, 7],
            # top (y+)
            [3, 7, 2], [2, 7, 6],
            # bottom (y-)
            [0, 1, 4], [1, 5, 4]
        ], dtype=int)
        
        # Normals for each vertex (one per face)
        normals = np.array([
            [0, 0, -1],   # back
            [0, 0, -1],   # back
            [0, 0, -1],   # back
            [0, 0, -1],   # back
            [0, 0, 1],    # front
            [0, 0, 1],    # front
            [0, 0, 1],    # front
            [0, 0, 1]     # front
        ], dtype=float)
        
        self.vertices = vertices
        self.indices = indices
        self.normals = normals

    def support(self, d):
        """Return furthest point in direction d by checking all vertices."""
        dots = np.dot(self.vertices, d)
        return self.vertices[np.argmax(dots)]

    def inertia(self, m):
        x, y, z = self.h * 2  # Full lengths
        return m / 12.0 * np.array([
            y*y + z*z,
            x*x + z*z,
            x*x + y*y
        ])


class Cylinder(ConvexShape):
    def __init__(self, radius, height, resolution=16):
        super().__init__()
        self.r = float(radius)
        self.h = float(height)
        self.half_height = self.h / 2
        
        # Generate cylinder mesh
        self._generate_mesh(resolution)
        
    def _generate_mesh(self, resolution):
        """Generate vertices, indices and normals for a cylinder."""
        #half_height = self.h / 2
        vertices = []
        indices = []
        normals = []
        
        # Top and bottom centers
        vertices.append([0, self.half_height, 0])    # top center
        vertices.append([0, -self.half_height, 0])   # bottom center
        normals.append([0, 1, 0])               # top normal
        normals.append([0, -1, 0])              # bottom normal
        
        # Generate circle vertices for top and bottom
        for i in range(resolution):
            angle = 2 * pi * i / resolution
            x = self.r * cos(angle)
            z = self.r * sin(angle)
            
            # Top circle vertex
            vertices.append([x, self.half_height, z])
            normals.append([0, 1, 0])
            
            # Bottom circle vertex
            vertices.append([x, -self.half_height, z])
            normals.append([0, -1, 0])
            
            # Side vertices (same positions as top/bottom but with different normals)
            vertices.append([x, self.half_height, z])
            vertices.append([x, -self.half_height, z])
            
            # Side normals
            nx, nz = x/self.r, z/self.r
            normals.append([nx, 0, nz])
            normals.append([nx, 0, nz])
        
        # Generate indices
        for i in range(resolution):
            next_i = (i + 1) % resolution
            
            # Top face triangles
            indices.append([0, 2 + i*4, 2 + next_i*4])
            
            # Bottom face triangles
            indices.append([1, 3 + next_i*4, 3 + i*4])
            
            # Side triangles (two per quad)
            indices.append([4 + i*4, 4 + next_i*4, 5 + i*4])
            indices.append([5 + i*4, 4 + next_i*4, 5 + next_i*4])
        
        self.vertices = np.array(vertices, dtype=float)
        self.indices = np.array(indices, dtype=int)
        self.normals = np.array(normals, dtype=float)

    def support(self, d):
        """
        Returns the point on the cylinder furthest in a given direction.
        This is found by comparing the support of the cylindrical wall and the support of the top/bottom caps.
        """
        # Support point on the cylindrical wall (ignoring caps)
        d_xz = np.array([d[0], 0, d[2]])
        n_xz = np.linalg.norm(d_xz)
        
        if n_xz > 1e-9:
            # Point on the side of the cylinder
            p_side = (d_xz / n_xz) * self.r
            p_side[1] = self.half_height if d[1] > 0 else -self.half_height
        else:
            # Direction is purely along Y, support is on the cap center
            p_side = np.array([0, self.half_height if d[1] > 0 else -self.half_height, 0])

        # Support point on the top/bottom caps (a point on the rim)
        p_caps = np.array([0.0, self.half_height if d[1] > 0 else -self.half_height, 0.0])
        if n_xz > 1e-9:
            p_caps[0] = (d[0] / n_xz) * self.r
            p_caps[2] = (d[2] / n_xz) * self.r

        # Return the point that is further in direction d
        if np.dot(p_side, d) > np.dot(p_caps, d):
            return p_side
        else:
            return p_caps

    def inertia(self, m):
        # Cylinder formulas
        h = self.h
        r = self.r
        Ixx = Izz = m/12 * (3*r*r + h*h)
        Iyy = m/2 * r*r
        return np.array([Ixx, Iyy, Izz])


class Plane(ConvexShape):
    def __init__(self, size=[10, 10], divisions=10, normal=[0, 1, 0]):
        super().__init__()
        self.size = np.asarray(size, float)
        self.normal = np.asarray(normal, float)
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.draw_grid = False  # Flag for grid rendering
        
        # Generate plane mesh
        self._generate_mesh(divisions)
        
    def _generate_mesh(self, divisions):
        """Generate vertices, indices and normals for a plane."""
        w, d = self.size[0]/2, self.size[1]/2
        
        step_x = self.size[0] / divisions if divisions > 0 else self.size[0]
        step_z = self.size[1] / divisions if divisions > 0 else self.size[1]
        
        vertices = []
        indices = []
        normals = []
        
        # Generate grid of vertices
        for i in range(divisions + 1):
            for j in range(divisions + 1):
                x = -w + i * step_x
                z = -d + j * step_z
                vertices.append([x, 0, z])
                normals.append(self.normal)
        
        # Generate indices for triangles
        for i in range(divisions):
            for j in range(divisions):
                p1 = i * (divisions + 1) + j
                p2 = p1 + 1
                p3 = p1 + (divisions + 1)
                p4 = p3 + 1
                
                indices.append([p1, p2, p3])
                indices.append([p2, p4, p3])
        
        self.vertices = np.array(vertices, dtype=float)
        self.indices = np.array(indices, dtype=int)
        self.normals = np.array(normals, dtype=float)

    def support(self, d):
        # Planes are special since they're infinite, so we'll need
        # to handle this specially in the collision detection
        # But for now, just return the center of the plane
        return np.zeros(3)

    def inertia(self, m):
        # For an infinite plane, inertia is infinite
        return np.array([float('inf'), float('inf'), float('inf')])


class Polyhedron(ConvexShape):
    """Arbitrary convex shape represented by vertices and faces."""
    def __init__(self, vertices, indices=None):
        super().__init__()
        self.vertices = np.asarray(vertices, dtype=float)
        
        if indices is None:
            # If no indices are provided, we can't triangulate properly
            # This is just a placeholder - real implementation would use
            # convex hull or similar algorithm
            self.indices = np.array([], dtype=int)
        else:
            self.indices = np.asarray(indices, dtype=int)
            
        # Calculate vertex normals based on face normals
        self._calculate_normals()

    def _calculate_normals(self):
        """Calculate vertex normals based on face normals."""
        if len(self.indices) == 0:
            self.normals = np.zeros_like(self.vertices)
            return
            
        # Initialize vertex normals
        self.normals = np.zeros_like(self.vertices)
        
        # Calculate face normals and add to vertex normals
        for triangle in self.indices:
            v0, v1, v2 = [self.vertices[i] for i in triangle]
            face_normal = np.cross(v1 - v0, v2 - v0)
            n = np.linalg.norm(face_normal)
            if n > 1e-10:
                face_normal /= n
                
            # Add to each vertex's normal
            for idx in triangle:
                self.normals[idx] += face_normal
                
        # Normalize all vertex normals
        for i in range(len(self.normals)):
            n = np.linalg.norm(self.normals[i])
            if n > 1e-10:
                self.normals[i] /= n

    def support(self, d):
        """Return furthest point in direction d."""
        if len(self.vertices) == 0:
            return np.zeros(3)
            
        dots = np.dot(self.vertices, d)
        return self.vertices[np.argmax(dots)]

    def inertia(self, m):
        """Approximate inertia tensor based on vertex distribution."""
        if len(self.vertices) == 0:
            return np.array([1.0, 1.0, 1.0])
            
        # Calculate inertia based on point cloud
        v = self.vertices
        c = v.mean(axis=0)  # Center of mass
        q = v - c           # Centered vertices
        Ix = np.mean(q[:,1]**2 + q[:,2]**2)
        Iy = np.mean(q[:,0]**2 + q[:,2]**2)
        Iz = np.mean(q[:,0]**2 + q[:,1]**2)
        return m * np.array([Ix, Iy, Iz])