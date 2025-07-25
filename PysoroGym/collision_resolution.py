import numpy as np

class Contact:
    """Contact information from collision detection."""
    def __init__(self, shape_a, shape_b, normal, depth, contact_a, contact_b):
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.body_a = shape_a.body
        self.body_b = shape_b.body
        self.normal = normal  # Normal from A to B
        self.depth = depth
        self.contact_a = contact_a  # Contact point on shape A
        self.contact_b = contact_b  # Contact point on shape B
        
        # Will be computed during resolution
        self.impulse = 0.0
        self.friction_impulse = np.zeros(3)


def resolve_contact(contact, dt):
    """
    Resolve a collision contact using impulse-based dynamics.
    
    Parameters
    ----------
    contact : Contact
        Contact information from collision detection
    dt : float
        Time step
    """
    body_a = contact.body_a
    body_b = contact.body_b
    
    # Skip if both bodies are static
    if getattr(body_a, 'is_static', False) and getattr(body_b, 'is_static', False):
        return
    
    # Get contact points relative to body centers
    r_a = contact.contact_a - body_a.pos
    r_b = contact.contact_b - body_b.pos
    
    # Calculate relative velocity at contact point
    v_a = body_a.vel + np.cross(body_a.ang_vel, r_a)
    v_b = body_b.vel + np.cross(body_b.ang_vel, r_b)
    v_rel = v_a - v_b
    
    # Relative velocity along normal
    v_n = np.dot(v_rel, contact.normal)
    
    # Don't resolve if velocities are separating
    if v_n > 0:
        return
    
    # Get material properties (use default if not specified)
    mat_a = getattr(contact.shape_a, 'material', None)
    mat_b = getattr(contact.shape_b, 'material', None)
    
    # Default material properties
    elasticity_a = getattr(mat_a, 'elasticity', 0.3) if mat_a else 0.3
    elasticity_b = getattr(mat_b, 'elasticity', 0.3) if mat_b else 0.3
    friction_a = getattr(mat_a, 'friction', 0.5) if mat_a else 0.5
    friction_b = getattr(mat_b, 'friction', 0.5) if mat_b else 0.5
    
    # Combined restitution (elasticity)
    restitution = min(elasticity_a, elasticity_b)
    
    # Combined friction
    friction = np.sqrt(friction_a * friction_b)
    
    # Calculate impulse scalar
    
    # Mass terms
    inv_mass_a = 0.0 if getattr(body_a, 'is_static', False) else getattr(body_a, 'inv_mass', 0.0)
    inv_mass_b = 0.0 if getattr(body_b, 'is_static', False) else getattr(body_b, 'inv_mass', 0.0)
    
    # Inertia terms
    if not getattr(body_a, 'is_static', False):
        r_a_cross_n = np.cross(r_a, contact.normal)
        # Try to get inv_inertia, if not available create identity matrix
        if hasattr(body_a, 'inv_inertia'):
            inv_inertia_a = body_a.inv_inertia
        else:
            print("Warning: Body A has no inv_inertia, using default value")
            inv_inertia_a = np.eye(3) * getattr(body_a, 'inv_mass', 1.0)
        
        angular_factor_a = np.dot(contact.normal, np.cross(inv_inertia_a @ r_a_cross_n, r_a))
    else:
        angular_factor_a = 0.0
    
    if not getattr(body_b, 'is_static', False):
        r_b_cross_n = np.cross(r_b, contact.normal)
        # Try to get inv_inertia, if not available create identity matrix
        if hasattr(body_b, 'inv_inertia'):
            inv_inertia_b = body_b.inv_inertia
        else:
            print("Warning: Body B has no inv_inertia, using default value")
            inv_inertia_b = np.eye(3) * getattr(body_b, 'inv_mass', 1.0)
        
        angular_factor_b = np.dot(contact.normal, np.cross(inv_inertia_b @ r_b_cross_n, r_b))
    else:
        angular_factor_b = 0.0
    
    # Denominator of impulse equation
    denominator = inv_mass_a + inv_mass_b + angular_factor_a + angular_factor_b
    
    if denominator < 1e-10:
        return
    
    # Normal impulse magnitude
    j_n = -(1 + restitution) * v_n / denominator
    
    # Apply normal impulse
    impulse_n = j_n * contact.normal
    
    if not getattr(body_a, 'is_static', False):
        body_a.vel += inv_mass_a * impulse_n
        if hasattr(body_a, 'inv_inertia'):
            body_a.ang_vel += inv_inertia_a @ np.cross(r_a, impulse_n)
    
    if not getattr(body_b, 'is_static', False):
        body_b.vel -= inv_mass_b * impulse_n
        if hasattr(body_b, 'inv_inertia'):
            body_b.ang_vel -= inv_inertia_b @ np.cross(r_b, impulse_n)
    
    # Friction impulse
    # Calculate tangent velocity (velocity perpendicular to normal)
    v_t = v_rel - v_n * contact.normal
    v_t_mag = np.linalg.norm(v_t)
    
    if v_t_mag > 1e-6:
        # Normalize tangent
        t = v_t / v_t_mag
        
        # Apply friction impulse (simplified)
        j_t = np.clip(-v_t_mag / denominator, -friction * abs(j_n), friction * abs(j_n))
        impulse_t = j_t * t
        
        if not getattr(body_a, 'is_static', False):
            body_a.vel += inv_mass_a * impulse_t
            if hasattr(body_a, 'inv_inertia'):
                body_a.ang_vel += inv_inertia_a @ np.cross(r_a, impulse_t)
        
        if not getattr(body_b, 'is_static', False):
            body_b.vel -= inv_mass_b * impulse_t
            if hasattr(body_b, 'inv_inertia'):
                body_b.ang_vel -= inv_inertia_b @ np.cross(r_b, impulse_t)
    
    # Position correction (to resolve penetration)
    beta = 0.2  # Baumgarte factor
    slop = 0.01  # Penetration slop
    correction_depth = max(contact.depth - slop, 0.0)
    
    if correction_depth > 0:
        correction = (beta / dt) * correction_depth * contact.normal
        
        if not getattr(body_a, 'is_static', False):
            body_a.pos += inv_mass_a / (inv_mass_a + inv_mass_b) * correction
        
        if not getattr(body_b, 'is_static', False):
            body_b.pos -= inv_mass_b / (inv_mass_a + inv_mass_b) * correction
    
    # Store impulse for debugging/visualization
    contact.impulse = j_n
    contact.friction_impulse = impulse_t if v_t_mag > 1e-6 else np.zeros(3)