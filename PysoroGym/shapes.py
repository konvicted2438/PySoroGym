"""
Shape classes that inherit from distance3d colliders.
"""
import numpy as np
from distance3d import colliders
import pytransform3d.transformations as pt


class Box(colliders.Box):
    """Box shape that inherits from distance3d Box collider."""
    
    def __init__(self, size, pose=None, color=None):
        """Initialize box shape.
        
        Parameters
        ----------
        size : array_like, shape (3,)
            Size of the box along x, y, z axes.
        pose : array_like, shape (4, 4), optional
            Initial pose of the box. Identity if not provided.
        color : tuple, optional
            RGB color values (0-1). If None, uses default coloring.
        """
        self.size = np.array(size)
        self.color = color
        if pose is None:
            pose = np.eye(4)
        super().__init__(pose, self.size)
        

class Sphere(colliders.Sphere):
    """Sphere shape that inherits from distance3d Sphere collider."""
    
    def __init__(self, radius, center=None, color=None):
        """Initialize sphere shape.
        
        Parameters
        ----------
        radius : float
            Radius of the sphere.
        center : array_like, shape (3,), optional
            Center of the sphere. Origin if not provided.
        color : tuple, optional
            RGB color values (0-1). If None, uses default coloring.
        """
        self.radius = radius
        self.color = color
        if center is None:
            center = np.zeros(3)
        super().__init__(center, radius)
        

class Cylinder(colliders.Cylinder):
    """Cylinder shape that inherits from distance3d Cylinder collider."""
    
    def __init__(self, radius, length, pose=None, color=None):
        """Initialize cylinder shape.
        
        Parameters
        ----------
        radius : float
            Radius of the cylinder.
        length : float
            Length of the cylinder.
        pose : array_like, shape (4, 4), optional
            Initial pose of the cylinder. Identity if not provided.
        color : tuple, optional
            RGB color values (0-1). If None, uses default coloring.
        """
        self.radius = radius
        self.length = length
        self.color = color
        if pose is None:
            pose = np.eye(4)
        super().__init__(pose, radius, length)
        

class Capsule(colliders.Capsule):
    """Capsule shape that inherits from distance3d Capsule collider."""
    
    def __init__(self, radius, height, pose=None, color=None):
        """Initialize capsule shape.
        
        Parameters
        ----------
        radius : float
            Radius of the capsule.
        height : float
            Height of the capsule (cylinder part).
        pose : array_like, shape (4, 4), optional
            Initial pose of the capsule. Identity if not provided.
        color : tuple, optional
            RGB color values (0-1). If None, uses default coloring.
        """
        self.radius = radius
        self.height = height
        self.color = color
        if pose is None:
            pose = np.eye(4)
        super().__init__(pose, radius, height)


class Ellipsoid(colliders.Ellipsoid):
    """Ellipsoid shape that inherits from distance3d Ellipsoid collider."""
    
    def __init__(self, radii, pose=None, color=None):
        """Initialize ellipsoid shape.
        
        Parameters
        ----------
        radii : array_like, shape (3,)
            Radii of the ellipsoid along x, y, z axes.
        pose : array_like, shape (4, 4), optional
            Initial pose of the ellipsoid. Identity if not provided.
        color : tuple, optional
            RGB color values (0-1). If None, uses default coloring.
        """
        self.radii = np.array(radii)
        self.color = color
        if pose is None:
            pose = np.eye(4)
        super().__init__(pose, self.radii)


class Cone(colliders.Cone):
    """Cone shape that inherits from distance3d Cone collider."""
    
    def __init__(self, radius, height, pose=None, color=None):
        """Initialize cone shape.
        
        Parameters
        ----------
        radius : float
            Radius of the cone base.
        height : float
            Height of the cone.
        pose : array_like, shape (4, 4), optional
            Initial pose of the cone. Identity if not provided.
        color : tuple, optional
            RGB color values (0-1). If None, uses default coloring.
        """
        self.radius = radius
        self.height = height
        self.color = color
        if pose is None:
            pose = np.eye(4)
        super().__init__(pose, radius, height)


class Disk(colliders.Disk):
    """Disk shape that inherits from distance3d Disk collider."""
    
    def __init__(self, radius, center=None, normal=None, color=None):
        """Initialize disk shape.
        
        Parameters
        ----------
        radius : float
            Radius of the disk.
        center : array_like, shape (3,), optional
            Center of the disk. Origin if not provided.
        normal : array_like, shape (3,), optional
            Normal vector to the disk plane. [0, 0, 1] if not provided.
        color : tuple, optional
            RGB color values (0-1). If None, uses default coloring.
        """
        self.radius = radius
        self.color = color
        if center is None:
            center = np.zeros(3)
        if normal is None:
            normal = np.array([0, 0, 1])
        super().__init__(center, radius, normal)


class Ellipse(colliders.Ellipse):
    """Ellipse shape that inherits from distance3d Ellipse collider."""
    
    def __init__(self, radii, center=None, axes=None, color=None):
        """Initialize ellipse shape.
        
        Parameters
        ----------
        radii : array_like, shape (2,)
            Radii of the ellipse along its two axes.
        center : array_like, shape (3,), optional
            Center of the ellipse. Origin if not provided.
        axes : array_like, shape (2, 3), optional
            Two axes of the ellipse. Default axes along x and y if not provided.
        color : tuple, optional
            RGB color values (0-1). If None, uses default coloring.
        """
        self.radii = np.array(radii)
        self.color = color
        if center is None:
            center = np.zeros(3)
        if axes is None:
            axes = np.array([[1, 0, 0], [0, 1, 0]])
        super().__init__(center, axes, self.radii)


class MeshGraph(colliders.MeshGraph):
    """Mesh shape that inherits from distance3d MeshGraph collider."""
    
    def __init__(self, vertices, triangles, pose=None, color=None):
        """Initialize mesh shape.
        
        Parameters
        ----------
        vertices : array_like, shape (n_vertices, 3)
            Vertices of the mesh.
        triangles : array_like, shape (n_triangles, 3)
            Indices of vertices that form triangles.
        pose : array_like, shape (4, 4), optional
            Initial pose of the mesh. Identity if not provided.
        color : tuple, optional
            RGB color values (0-1). If None, uses default coloring.
        """
        self.original_vertices = np.array(vertices, dtype=np.float64)
        self.triangles = np.array(triangles, dtype=np.int64)
        self.color = color
        
        # Calculate center of mass
        self._center_of_mass = self.calculate_center_of_mass()
        print(f"Calculated center of mass: {self._center_of_mass}")
        
        # Shift vertices so center of mass is at origin
        self.vertices = self.original_vertices - self._center_of_mass
        
        if pose is None:
            pose = np.eye(4)
        super().__init__(pose, self.vertices, self.triangles)
        
    def calculate_center_of_mass(self):
        """Calculate center of mass for the mesh using signed volume method."""
        if len(self.triangles) == 0:
            return np.mean(self.original_vertices, axis=0)
        
        total_volume = 0.0
        weighted_center = np.zeros(3)
        
        # Use origin as reference point for tetrahedron decomposition
        origin = np.zeros(3)
        
        for triangle in self.triangles:
            v0, v1, v2 = self.original_vertices[triangle]
            
            # Calculate signed volume of tetrahedron (origin, v0, v1, v2)
            # Volume = (1/6) * dot(v0, cross(v1, v2))
            volume = np.dot(v0, np.cross(v1, v2)) / 6.0
            
            if abs(volume) > 1e-10:  # Avoid degenerate triangles
                # Centroid of tetrahedron is average of 4 vertices
                tetrahedron_center = (origin + v0 + v1 + v2) / 4.0
                
                total_volume += volume
                weighted_center += volume * tetrahedron_center
        
        if abs(total_volume) > 1e-10:
            return weighted_center / total_volume
        else:
            # Fallback to geometric center
            return np.mean(self.original_vertices, axis=0)
    
    def get_center_of_mass(self):
        """Get the center of mass offset from geometric center."""
        return self._center_of_mass.copy()
    
    def get_volume(self):
        """Calculate volume of the mesh."""
        total_volume = 0.0
        origin = np.zeros(3)
        
        for triangle in self.triangles:
            v0, v1, v2 = self.original_vertices[triangle]
            volume = np.dot(v0, np.cross(v1, v2)) / 6.0
            total_volume += volume
        
        return abs(total_volume)


class ConvexHullVertices(colliders.ConvexHullVertices):
    """Convex hull defined by vertices."""
    
    def __init__(self, vertices, color=None):
        """Initialize convex hull shape.
        
        Parameters
        ----------
        vertices : array_like, shape (n_vertices, 3)
            Vertices of the convex shape.
        color : tuple, optional
            RGB color values (0-1). If None, uses default coloring.
        """
        self.vertices = np.array(vertices)
        self.color