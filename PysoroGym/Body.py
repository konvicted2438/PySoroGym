"""
Rigid body class for physics simulation.
"""
import numpy as np
from typing import List, Optional, Tuple, Dict
from PysoroGym.materials import Material


class Body:
    """Rigid body for physics simulation"""
    
    # Body types
    STATIC = 'static'
    DYNAMIC = 'dynamic'
    KINEMATIC = 'kinematic'
    
    def __init__(self, body_type='dynamic', mass=1.0, position=(0, 0, 0), 
                 orientation=None, linear_velocity=(0, 0, 0), angular_velocity=(0, 0, 0),
                 default_material=None):
        # Body type
        self.body_type = body_type
        
        # Mass properties
        self.mass = float(mass) if body_type == 'dynamic' else float('inf')
        self.inertia_tensor = None  # Will be computed from shapes
        
        # State - ensure all arrays are float64
        # Note: position is now the CENTER OF MASS position
        self.position = np.array(position, dtype=np.float64)
        self.orientation = np.array([1, 0, 0, 0], dtype=np.float64) if orientation is None else np.array(orientation, dtype=np.float64)
        self.linear_velocity = np.array(linear_velocity, dtype=np.float64)
        self.angular_velocity = np.array(angular_velocity, dtype=np.float64)
        
        # Forces and torques
        self.force = np.zeros(3, dtype=np.float64)
        self.torque = np.zeros(3, dtype=np.float64)
        
        # Damping
        self.linear_damping = 0.01
        self.angular_damping = 0.01

        # Material for this body
        self.material = default_material if default_material is not None else Material()
        
        # Shapes attached to this body (shape -> material mapping)
        self.shapes: Dict = {}  # {shape: material}
        
        # Center of mass offset from geometric center (in local space)
        self.center_of_mass_local = np.zeros(3, dtype=np.float64)
        
        # Reference to physics engine
        self.physics_engine = None
        
        # User data
        self.user_data = {}
        
        # Sleeping/deactivation
        self.is_sleeping = False
        self.sleep_time = 0.0
        self.sleep_linear_threshold = 0.01
        self.sleep_angular_threshold = 0.01
        self.sleep_time_threshold = 0.5  # Time before sleeping
        
    def add_shape(self, shape, material=None):
        """Add a shape to this body
        
        Parameters
        ----------
        shape : Shape
            The shape to add (already has its own pose)
        material : Material, optional
            Material properties for this shape. If None, uses the body's default material.
        """
        # Use the provided material or the body's default material
        shape_material = material if material is not None else self.material
            
        self.shapes[shape] = shape_material
        
        # Update mass properties and center of mass
        self._update_mass_properties()
        
        # Update the shape's pose to match body transform
        self._update_shape_pose(shape)
        
        return shape
        
    def _update_mass_properties(self):
        """Update mass, center of mass, and inertia from shapes"""
        if self.body_type != 'dynamic':
            return
            
        if not self.shapes:
            self.inertia_tensor = np.eye(3, dtype=np.float64) * 0.1
            self.center_of_mass_local = np.zeros(3, dtype=np.float64)
            return
            
        # Calculate total mass and center of mass
        total_mass = 0.0
        weighted_center = np.zeros(3, dtype=np.float64)
        
        for shape, material in self.shapes.items():
            shape_mass = self._calculate_shape_mass(shape, material)
            total_mass += shape_mass
            
            # Get shape center of mass in local coordinates
            shape_com = self._get_shape_center_of_mass(shape)
            weighted_center += shape_mass * shape_com
            
        if total_mass > 0:
            self.center_of_mass_local = weighted_center / total_mass
            self.mass = total_mass
        else:
            self.center_of_mass_local = np.zeros(3, dtype=np.float64)
            
        # Calculate inertia tensor about center of mass
        inertia = np.zeros((3, 3), dtype=np.float64)
        
        for shape, material in self.shapes.items():
            shape_inertia = self._calculate_shape_inertia(shape, material)
            shape_mass = self._calculate_shape_mass(shape, material)
            
            # Get shape center of mass
            shape_com = self._get_shape_center_of_mass(shape)
            
            # Vector from body center of mass to shape center of mass
            r = shape_com - self.center_of_mass_local
            
            # Parallel axis theorem to transfer inertia to body center of mass
            parallel_axis_correction = shape_mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
            inertia += shape_inertia + parallel_axis_correction
            
        self.inertia_tensor = inertia
        
    def _get_shape_center_of_mass(self, shape):
        """Get center of mass of a shape in local coordinates"""
        # If shape has a get_center_of_mass method, use it
        if hasattr(shape, 'get_center_of_mass'):
            return shape.get_center_of_mass()
        else:
            # Use geometric center as approximation
            return shape.center()
            
    def get_world_center_of_mass(self):
        """Get center of mass position in world coordinates"""
        return self.position  # Position IS the center of mass
        
    def get_world_geometric_center(self):
        """Get geometric center in world coordinates"""
        # Transform center of mass offset to world space
        rotation_matrix = self.quaternion_to_matrix()
        world_offset = rotation_matrix @ self.center_of_mass_local
        return self.position - world_offset  # Subtract because COM offset was added to position
    
    def update_shapes(self):
        """Update all shape poses to match the body's transform"""
        for shape in self.shapes.keys():
            self._update_shape_pose(shape)
            
    def _update_shape_pose(self, shape):
        """Update a single shape's pose to match body transform"""
        # Get the world transform for the geometric center
        geometric_center = self.get_world_geometric_center()
        
        # Create transform matrix
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.quaternion_to_matrix()
        transform[:3, 3] = geometric_center
        
        shape.update_pose(transform)
    
    def update_sleep_state(self, dt):
        """Update sleeping state based on velocity"""
        if self.body_type != 'dynamic':
            return
            
        linear_speed = np.linalg.norm(self.linear_velocity)
        angular_speed = np.linalg.norm(self.angular_velocity)
        
        if linear_speed < self.sleep_linear_threshold and angular_speed < self.sleep_angular_threshold:
            self.sleep_time += dt
            if self.sleep_time > self.sleep_time_threshold:
                self.is_sleeping = True
                # Zero out velocities when sleeping
                self.linear_velocity = np.zeros(3, dtype=np.float64)
                self.angular_velocity = np.zeros(3, dtype=np.float64)
        else:
            self.wake_up()
            
    def wake_up(self):
        """Wake up this body"""
        self.is_sleeping = False
        self.sleep_time = 0.0
        
    def apply_force(self, force, world_point=None):
        """Apply a force to the body"""
        if self.body_type != 'dynamic':
            return
            
        # Wake up if sleeping
        if self.is_sleeping:
            self.wake_up()
            
        self.force += np.array(force, dtype=np.float64)
        
        # If world point specified, calculate torque about center of mass
        if world_point is not None:
            # Vector from center of mass to application point
            r = np.array(world_point, dtype=np.float64) - self.get_world_center_of_mass()
            self.torque += np.cross(r, force)
            
    def apply_torque(self, torque):
        """Apply a torque to the body"""
        if self.body_type != 'dynamic':
            return
            
        self.torque += np.array(torque, dtype=np.float64)
        
    def apply_impulse(self, impulse, world_point=None):
        """Apply an impulse to the body"""
        if self.body_type != 'dynamic':
            return
            
        self.linear_velocity += np.array(impulse, dtype=np.float64) / self.mass
        
        # If world point specified, also apply angular impulse about center of mass
        if world_point is not None:
            r = np.array(world_point, dtype=np.float64) - self.get_world_center_of_mass()
            if self.inertia_tensor is not None:
                angular_impulse = np.cross(r, impulse)
                self.angular_velocity += np.linalg.inv(self.inertia_tensor) @ angular_impulse
                
    def get_velocity_at_point(self, world_point):
        """Get velocity at a world point on the body"""
        # Vector from center of mass to point
        r = np.array(world_point, dtype=np.float64) - self.get_world_center_of_mass()
        return self.linear_velocity + np.cross(self.angular_velocity, r)
        
    def get_world_transform(self):
        """Get 4x4 transformation matrix for this body"""
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.quaternion_to_matrix()
        transform[:3, 3] = self.position
        return transform
        
    def quaternion_to_matrix(self):
        """Convert quaternion to 3x3 rotation matrix"""
        w, x, y, z = self.orientation
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        
    def _calculate_shape_mass(self, shape, material):
        """Calculate mass of a shape"""
        volume = self._calculate_shape_volume(shape)
        return material.density * volume
        
    def _calculate_shape_volume(self, shape):
        """Calculate volume of a shape"""
        from PysoroGym import shapes
        
        if isinstance(shape, shapes.Sphere):
            return (4/3) * np.pi * shape.radius**3
        elif isinstance(shape, shapes.Box):
            return np.prod(shape.size)
        elif isinstance(shape, shapes.Cylinder):
            return np.pi * shape.radius**2 * shape.length
        elif isinstance(shape, shapes.Capsule):
            cylinder_volume = np.pi * shape.radius**2 * shape.height
            sphere_volume = (4/3) * np.pi * shape.radius**3
            return cylinder_volume + sphere_volume
        elif isinstance(shape, shapes.Cone):
            return (1/3) * np.pi * shape.radius**2 * shape.height
        elif isinstance(shape, shapes.Ellipsoid):
            return (4/3) * np.pi * np.prod(shape.radii)
        elif isinstance(shape, shapes.Disk):
            return 0.01  # Thin disk, minimal volume
        else:
            # Default volume
            return 1.0
            
    def _calculate_shape_inertia(self, shape, material):
        """Calculate inertia tensor of a shape"""
        from PysoroGym import shapes
        
        mass = self._calculate_shape_mass(shape, material)
        
        if isinstance(shape, shapes.Sphere):
            I = (2/5) * mass * shape.radius**2
            return np.eye(3, dtype=np.float64) * I
            
        elif isinstance(shape, shapes.Box):
            sx, sy, sz = shape.size
            Ixx = mass * (sy**2 + sz**2) / 12
            Iyy = mass * (sx**2 + sz**2) / 12
            Izz = mass * (sx**2 + sy**2) / 12
            return np.diag([Ixx, Iyy, Izz]).astype(np.float64)
            
        elif isinstance(shape, shapes.Cylinder):
            r = shape.radius
            h = shape.length
            Ixx = Iyy = mass * (3*r**2 + h**2) / 12
            Izz = mass * r**2 / 2
            return np.diag([Ixx, Iyy, Izz]).astype(np.float64)
            
        elif isinstance(shape, shapes.Capsule):
            # Approximate as cylinder + 2 hemispheres
            r = shape.radius
            h = shape.height
            # Cylinder part
            m_cyl = mass * h / (h + (4/3)*r)
            Ixx_cyl = m_cyl * (3*r**2 + h**2) / 12
            Izz_cyl = m_cyl * r**2 / 2
            # Sphere part
            m_sph = mass - m_cyl
            I_sph = (2/5) * m_sph * r**2
            # Combined
            Ixx = Iyy = Ixx_cyl + I_sph + m_sph * (h/2)**2
            Izz = Izz_cyl + I_sph
            return np.diag([Ixx, Iyy, Izz]).astype(np.float64)
            
        elif isinstance(shape, shapes.Cone):
            r = shape.radius
            h = shape.height
            Ixx = Iyy = mass * (3*r**2 + 2*h**2) / 20
            Izz = (3/10) * mass * r**2
            return np.diag([Ixx, Iyy, Izz]).astype(np.float64)
            
        elif isinstance(shape, shapes.Ellipsoid):
            a, b, c = shape.radii
            Ixx = mass * (b**2 + c**2) / 5
            Iyy = mass * (a**2 + c**2) / 5
            Izz = mass * (a**2 + b**2) / 5
            return np.diag([Ixx, Iyy, Izz]).astype(np.float64)
            
        else:
            # Default inertia (sphere-like)
            return np.eye(3, dtype=np.float64) * mass * 0.1
    
    def get_shapes_for_collision(self):
        """Get all shapes with their current world transforms for collision detection"""
        # Make sure all shapes have updated poses
        self.update_shapes()
        return list(self.shapes.keys())


class ShapeInstance:
    """Instance of a shape attached to a body"""
    
    def __init__(self, shape, body, offset=(0, 0, 0), orientation=None, material=None):
        self.shape = shape
        self.body = body
        self.offset = np.array(offset, dtype=np.float64)
        self.orientation = np.array([1, 0, 0, 0], dtype=np.float64) if orientation is None else np.array(orientation, dtype=np.float64)
        self.material = material or Material()
        
    def get_world_transform(self):
        """Get world transform for this shape"""
        # Body transform
        body_transform = self.body.get_world_transform()
        
        # Local transform
        local_transform = np.eye(4, dtype=np.float64)
        local_transform[:3, :3] = self._quaternion_to_matrix(self.orientation)
        local_transform[:3, 3] = self.offset
        
        # Combined transform
        return body_transform @ local_transform
        
    def _quaternion_to_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ], dtype=np.float64)