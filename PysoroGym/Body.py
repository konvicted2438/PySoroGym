import numpy as np
from .math_utils import q_identity, q_normalize, q_mul, q_from_axis_angle, q_to_mat3, q_to_euler
from .physics import ShapeCollider, Material

# ─────────────────────────────────────────────────────────────────────────────
# Rigid body
# ─────────────────────────────────────────────────────────────────────────────
class Body:
    """
    Rigid body with mass, inertia, and poses.
    Can have multiple shapes attached.
    """
    
    # Body types (like pymunk)
    STATIC = 0    # zero mass, doesn't move
    KINEMATIC = 1 # infinite mass but can be moved programmatically
    DYNAMIC = 2   # regular physics-driven body
    _next_id = 0 # Class variable to generate unique IDs
    
    def __init__(self, mass=1.0, position=(0,0,0), velocity=(0,0,0),
                 angular_velocity=(0,0,0), orientation=(1,0,0,0),
                 body_type=DYNAMIC, inertia_tensor=None):
        
        self.id = Body._next_id
        Body._next_id += 1

        self.mass = float(mass)
        if self.mass <= 0:
            body_type = Body.STATIC   # zero or negative mass → static body
            
        self.body_type = body_type
        
        # pose
        self.pos = np.asarray(position, dtype=float)
        self.q = q_identity() if orientation is None else q_normalize(np.asarray(orientation, float))
        
        # Linear and angular motion
        self.vel = np.asarray(velocity, dtype=float)
        self.ang_vel = np.asarray(angular_velocity, dtype=float)
        
        # Forces and torques
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        
        # mass & inertia
        self.mass = 0.0 if body_type == Body.STATIC else float(mass)
        self.inv_mass = 0.0 if self.mass == 0 else 1.0/self.mass
        
        # Local inertia tensor (diagonal)
        self.J_local = np.array([1.0, 1.0, 1.0], float)  # default unit inertia
        self.J_local_inv = 1.0 / self.J_local            # cached inverse
        
        # Add world-space inverse inertia tensor attribute
        self._inv_inertia_world = None  # Cached world-space inverse inertia
        
        # accumulators
        self.force_acc = np.zeros(3, float)
        self.torque_acc = np.zeros(3, float)
        
        # damping factors (0-1, where 1 means no damping)
        self.linear_damping = 0.99
        self.angular_damping = 0.95
        
        # Sleeping parameters (optimization)
        self.sleep_threshold = 0.05
        self.is_sleeping = False
        self.sleeping_timer = 0

        # NEW: Sleep state tracking
        self.is_sleeping = False
        self.sleep_time = 0.0
        self.sleep_linear_threshold = 0.1   # m/s
        self.sleep_angular_threshold = 0.2  # rad/s

        # Attached shapes
        self.shapes = []
    
    # ––––– Shape management –––––
    def add_shape(self, shape, offset=(0,0,0), material=None):
        """Add collision shape to this body."""
        # If shape is already a ShapeCollider, attach it to this body
        if isinstance(shape, ShapeCollider):
            shape.set_body(self)
            self.shapes.append(shape)
            return shape
        
        # Otherwise create a new ShapeCollider
        collider = ShapeCollider(shape, self, offset, material)
        self.shapes.append(collider)
        
        # Recalculate inertia if dynamic
        if self.body_type == Body.DYNAMIC and hasattr(shape, "inertia"):
            self._update_inertia()
            
        return collider
    
    def _update_inertia(self):
        """Update inertia tensor based on all shapes."""
        if not self.shapes or self.mass == 0:
            return
            
        # For now, just use the first shape's inertia if available
        for collider in self.shapes:
            if hasattr(collider.shape, "inertia"):
                self.J_local = collider.shape.inertia(self.mass)
                if np.isscalar(self.J_local):
                    self.J_local = np.array([self.J_local, self.J_local, self.J_local], float)
                self.J_local_inv = 1.0 / self.J_local
                break

    # ––––– Properties –––––
    @property
    def rotation(self):
        """Get rotation as Euler angles (degrees) for visualization."""
        return q_to_euler(self.q)
    
    @property
    def position(self):
        """Return position as array."""
        return self.pos
    
    @property
    def is_static(self):
        return self.body_type == Body.STATIC
    
    @property
    def is_kinematic(self):
        return self.body_type == Body.KINEMATIC

    # ––––– Force application –––––
    def apply_force(self, force, point=None):
        """Apply force (N). If point (world) given adds torque."""
        if self.is_static:
            return
            
        self.force_acc += force
        if point is not None:
            r = point - self.pos          # lever arm
            self.torque_acc += np.cross(r, force)
    
    def apply_impulse(self, impulse, point=None):
        """Apply an instantaneous impulse (kg*m/s)."""
        if self.is_static:
            return
            
        self.vel += impulse * self.inv_mass
        if point is not None:
            r = point - self.pos
            self.ang_vel += self.inv_inertia_world() @ np.cross(r, impulse)

    # ––––– Integration –––––
    def inv_inertia_world(self):
        """Return 3×3 inverse inertia tensor in world coords."""
        if self.body_type == Body.STATIC:
            # Static bodies have infinite inertia (zero inverse)
            return np.zeros((3, 3))
        
        R = q_to_mat3(self.q)
        self._inv_inertia_world = R @ np.diag(self.J_local_inv) @ R.T
        return self._inv_inertia_world
    
    @property
    def inv_inertia(self):
        """Property to access world-space inverse inertia tensor."""
        return self.inv_inertia_world()

    def integrate_velocities(self, dt):
        """First integration step: forces → velocities."""
        if self.is_static or self.is_sleeping:
            return
            
        # linear
        a = self.force_acc * self.inv_mass
        self.vel += a * dt
        
        # angular
        alpha = self.inv_inertia_world() @ self.torque_acc
        self.ang_vel += alpha * dt
        
        # Apply damping
        self.vel *= self.linear_damping
        self.ang_vel *= self.angular_damping
        
        # Clear force accumulators
        self.force_acc[:] = 0
        self.torque_acc[:] = 0
    
    def integrate_positions(self, dt):
        """Second integration step: velocities → positions."""
        if self.is_static or self.is_sleeping:
            return
            
        # Update position
        self.pos += self.vel * dt
        
        # Update orientation (quaternion)
        if np.linalg.norm(self.ang_vel) > 1e-8:
            angle = np.linalg.norm(self.ang_vel) * dt
            axis = self.ang_vel / np.linalg.norm(self.ang_vel)
            dq = q_from_axis_angle(axis, angle)
            self.q = q_normalize(q_mul(dq, self.q))
    
    def integrate_explicit_euler(self, dt):
        """Combined single-step integration."""
        if self.is_static:
            return
            
        self.integrate_velocities(dt)
        self.integrate_positions(dt)
    
    # ––––– Transformations –––––
    def transform_point(self, local_point):
        """
        Transform a point from local to world coordinates.
        
        Parameters
        ----------
        local_point : array_like, shape (3,)
            Point in local coordinates
            
        Returns
        -------
        world_point : ndarray, shape (3,)
            Point in world coordinates
        """
        # Get rotation matrix from quaternion
        R = q_to_mat3(self.q)
        
        # Transform: rotate then translate
        return R @ local_point + self.pos
        
    def transform_direction(self, local_dir):
        """
        Transform a direction vector from local to world coordinates.
        
        Parameters
        ----------
        local_dir : array_like, shape (3,)
            Direction in local coordinates
            
        Returns
        -------
        world_dir : ndarray, shape (3,)
            Direction in world coordinates
        """
        # Get rotation matrix from quaternion
        R = q_to_mat3(self.q)
        
        # Only rotate, no translation
        return R @ local_dir
        
    def transform_direction_inverse(self, world_dir):
        """
        Transform a direction vector from world to local coordinates.
        
        Parameters
        ----------
        world_dir : array_like, shape (3,)
            Direction in world coordinates
            
        Returns
        -------
        local_dir : ndarray, shape (3,)
            Direction in local coordinates
        """
        # Get rotation matrix from quaternion
        R = q_to_mat3(self.q)
        
        # For a rotation matrix, the inverse is the transpose.
        # This rotates the world direction into the body's local frame.
        return R.T @ world_dir
    
    # NEW: Sleep state tracking
    def update_sleep_state(self, dt):
        """Update the sleeping state of the body."""
        if self.body_type != Body.DYNAMIC:
            return  # Only dynamic bodies can sleep
            
        # Check if motion is below thresholds
        lin_vel_sq = np.sum(self.vel**2)
        ang_vel_sq = np.sum(self.ang_vel**2)
        
        if lin_vel_sq < self.sleep_linear_threshold**2 and \
           ang_vel_sq < self.sleep_angular_threshold**2:
            self.sleep_time += dt
            if self.sleep_time > 1.0:  # Sleep after 1 second of low movement
                self.is_sleeping = True
                self.vel = np.zeros(3)  # Zero out velocity
                self.ang_vel = np.zeros(3)
        else:
            # Reset sleep counter if moving
            self.is_sleeping = False
            self.sleep_time = 0.0
    
    def wake_up(self):
        """Wake up a sleeping body."""
        if self.is_sleeping:
            self.is_sleeping = False
            self.sleep_time = 0.0