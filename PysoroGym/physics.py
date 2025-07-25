import numpy as np
from .math_utils import q_to_mat3

"""
Physics-related classes and utilities.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Physics materials
# ─────────────────────────────────────────────────────────────────────────────
class Material:
    """
    Material properties for physics simulation.
    """
    def __init__(self, friction=0.5, elasticity=0.3, density=1.0):
        """
        Initialize material properties.
        
        Parameters
        ----------
        friction : float
            Friction coefficient (0 = no friction, 1 = high friction)
        elasticity : float
            Elasticity/restitution coefficient (0 = no bounce, 1 = perfect bounce)
        density : float
            Material density (kg/m³)
        """
        self.friction = friction
        self.elasticity = elasticity
        self.density = density
    
    @classmethod
    def rubber(cls):
        """Predefined rubber material."""
        return cls(friction=0.9, elasticity=0.8, density=1500)
    
    @classmethod
    def metal(cls):
        """Predefined metal material."""
        return cls(friction=0.4, elasticity=0.2, density=7800)
    
    @classmethod
    def wood(cls):
        """Predefined wood material."""
        return cls(friction=0.6, elasticity=0.4, density=700)
    
    @classmethod
    def ice(cls):
        """Predefined ice material."""
        return cls(friction=0.1, elasticity=0.1, density=900)

# ─────────────────────────────────────────────────────────────────────────────
# Physics shapes - collision geometry
# ─────────────────────────────────────────────────────────────────────────────
class ShapeCollider:
    """
    Collision component for Body that wraps a Shape.
    Follows pymunk-like structure where shapes handle collision properties.
    """
    def __init__(self, shape, body=None, offset=(0,0,0), material=None):
        self.shape = shape            # geometric shape
        self.body = body              # parent body reference (can be None)
        self.offset = np.array(offset, dtype=float)  # local offset
        self.material = material or Material()
        self.filter_group = 0         # for collision filtering (0=all collide)
        self.sensor = False           # if True, detects but doesn't resolve collisions
    
    def set_body(self, body):
        """Attach this shape to a body."""
        self.body = body
        return self
    
    def get_world_transform(self):
        """Get world transform matrix for this shape."""
        if not self.body:
            return np.eye(4)
            
        # Create transform from body pose
        R = q_to_mat3(self.body.q)
        
        # Compose with local offset
        offset_world = R @ self.offset
        
        # Build 4x4 transform
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = self.body.pos + offset_world
        
        return transform
        
    def world_support(self, direction):
        """Get support point in world space."""
        if not self.body:
            return self.shape.support(direction)
            
        # Transform direction to local space
        R = q_to_mat3(self.body.q)
        p_local = self.shape.support(R.T @ direction)
        
        # Transform point back to world space
        return R @ (p_local + self.offset) + self.body.pos