import numpy as np
from .Body import Body

# Constants from ReactPhysics3D
BETA = 0.2  # Baumgarte stabilization
BETA_SPLIT_IMPULSE = 0.2
SLOP = 0.01  # Penetration slop
RESTITUTION_VELOCITY_THRESHOLD = 1.0  # m/s

class ContactManifold:
    """Groups multiple contact points between two shapes."""
    def __init__(self, shape_a, shape_b):
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.body_a = shape_a.body
        self.body_b = shape_b.body
        self.contacts = []
        
        # Friction vectors and impulses (stored for warm starting)
        self.friction_vector1 = np.zeros(3)
        self.friction_vector2 = np.zeros(3)
        self.friction_impulse1 = 0.0
        self.friction_impulse2 = 0.0
        self.friction_twist_impulse = 0.0
        
    def add_contact(self, normal, depth, contact_a, contact_b):
        contact = ContactPoint(normal, depth, contact_a, contact_b)
        self.contacts.append(contact)
        return contact

class ContactPoint:
    """Individual contact point data."""
    def __init__(self, normal, depth, contact_a, contact_b):
        self.normal = normal
        self.depth = depth
        self.contact_a = contact_a
        self.contact_b = contact_b
        
        # Impulses (stored for warm starting)
        self.penetration_impulse = 0.0
        self.penetration_split_impulse = 0.0
        self.is_resting_contact = False

class ContactSolver:
    """Iterative contact solver following ReactPhysics3D approach."""
    
    def __init__(self, use_split_impulse=True):
        self.use_split_impulse = use_split_impulse
        self.contact_constraints = []
        self.contact_points = []
        
    def solve(self, manifolds, dt, num_iterations=10):
        """Main solving routine."""
        if not manifolds:
            return
            
        # Initialize constraints
        self.init_constraints(manifolds, dt)
        
        # Warm start using previous frame impulses
        self.warm_start()
        
        # Solve velocity constraints iteratively
        for _ in range(num_iterations):
            self.solve_velocity_constraints()
            
        # Store impulses for next frame
        self.store_impulses(manifolds)
        
    def init_constraints(self, manifolds, dt):
        """Initialize constraint data for all contact manifolds."""
        self.contact_constraints = []
        self.contact_points = []
        
        for manifold in manifolds:
            if not manifold.contacts:
                continue
                
            # Create constraint for this manifold
            constraint = ContactManifoldConstraint(manifold, dt)
            self.contact_constraints.append(constraint)
            
            # Initialize contact points
            for contact in manifold.contacts:
                point = ContactPointConstraint(contact, constraint, dt)
                self.contact_points.append(point)
                
    def warm_start(self):
        """Apply impulses from previous frame."""
        for constraint in self.contact_constraints:
            constraint.warm_start()
            
    def solve_velocity_constraints(self):
        """One iteration of velocity constraint solving."""
        # Solve normal constraints (penetration)
        for point in self.contact_points:
            point.solve_penetration()
            
        # Solve friction constraints at manifold level
        for constraint in self.contact_constraints:
            constraint.solve_friction()

class ContactManifoldConstraint:
    """Constraint data for a contact manifold."""
    
    def __init__(self, manifold, dt):
        self.manifold = manifold
        self.dt = dt
        self.body_a = manifold.body_a
        self.body_b = manifold.body_b
        
        # Get masses and inertias
        self.inv_mass_a = self.body_a.inv_mass if self.body_a.body_type == Body.DYNAMIC else 0.0
        self.inv_mass_b = self.body_b.inv_mass if self.body_b.body_type == Body.DYNAMIC else 0.0
        
        if self.body_a.body_type == Body.DYNAMIC:
            self.inv_inertia_a = self.body_a.inv_inertia_world()
        else:
            self.inv_inertia_a = np.zeros((3, 3))
            
        if self.body_b.body_type == Body.DYNAMIC:
            self.inv_inertia_b = self.body_b.inv_inertia_world()
        else:
            self.inv_inertia_b = np.zeros((3, 3))
            
        # Compute average normal and friction point
        self.normal = np.zeros(3)
        self.friction_point_a = np.zeros(3)
        self.friction_point_b = np.zeros(3)
        
        for contact in manifold.contacts:
            self.normal += contact.normal
            self.friction_point_a += contact.contact_a
            self.friction_point_b += contact.contact_b
            
        num_contacts = len(manifold.contacts)
        self.normal /= num_contacts
        self.normal /= np.linalg.norm(self.normal)
        self.friction_point_a /= num_contacts
        self.friction_point_b /= num_contacts
        
        # Friction vectors and mass
        self.r1_friction = self.friction_point_a - self.body_a.pos
        self.r2_friction = self.friction_point_b - self.body_b.pos
        self.compute_friction_vectors()
        self.compute_friction_mass()
        
        # Material properties
        mat_a = getattr(manifold.shape_a, 'material', None)
        mat_b = getattr(manifold.shape_b, 'material', None)
        friction_a = getattr(mat_a, 'friction', 0.5) if mat_a else 0.5
        friction_b = getattr(mat_b, 'friction', 0.5) if mat_b else 0.5
        self.friction_coeff = np.sqrt(friction_a * friction_b)
        
    def compute_friction_vectors(self):
        """Compute orthogonal friction vectors."""
        # Get relative velocity at friction point
        v_a = self.body_a.vel + np.cross(self.body_a.ang_vel, self.r1_friction)
        v_b = self.body_b.vel + np.cross(self.body_b.ang_vel, self.r2_friction)
        v_rel = v_b - v_a
        
        # Project to tangent plane
        v_tangent = v_rel - np.dot(v_rel, self.normal) * self.normal
        
        if np.linalg.norm(v_tangent) > 1e-6:
            self.friction_vector1 = v_tangent / np.linalg.norm(v_tangent)
        else:
            # Get any orthogonal vector
            if abs(self.normal[0]) < 0.9:
                self.friction_vector1 = np.cross(self.normal, np.array([1, 0, 0]))
            else:
                self.friction_vector1 = np.cross(self.normal, np.array([0, 1, 0]))
            self.friction_vector1 /= np.linalg.norm(self.friction_vector1)
            
        self.friction_vector2 = np.cross(self.normal, self.friction_vector1)
        
    def compute_friction_mass(self):
        """Compute inverse mass matrices for friction constraints."""
        # First friction direction
        r1_cross_t1 = np.cross(self.r1_friction, self.friction_vector1)
        r2_cross_t1 = np.cross(self.r2_friction, self.friction_vector1)
        
        k1 = self.inv_mass_a + self.inv_mass_b
        k1 += np.dot(self.friction_vector1, np.cross(self.inv_inertia_a @ r1_cross_t1, self.r1_friction))
        k1 += np.dot(self.friction_vector1, np.cross(self.inv_inertia_b @ r2_cross_t1, self.r2_friction))
        
        self.inv_friction_mass1 = 1.0 / k1 if k1 > 1e-10 else 0.0
        
        # Second friction direction
        r1_cross_t2 = np.cross(self.r1_friction, self.friction_vector2)
        r2_cross_t2 = np.cross(self.r2_friction, self.friction_vector2)
        
        k2 = self.inv_mass_a + self.inv_mass_b
        k2 += np.dot(self.friction_vector2, np.cross(self.inv_inertia_a @ r1_cross_t2, self.r1_friction))
        k2 += np.dot(self.friction_vector2, np.cross(self.inv_inertia_b @ r2_cross_t2, self.r2_friction))
        
        self.inv_friction_mass2 = 1.0 / k2 if k2 > 1e-10 else 0.0
        
        # Twist friction
        k_twist = np.dot(self.normal, self.inv_inertia_a @ self.normal)
        k_twist += np.dot(self.normal, self.inv_inertia_b @ self.normal)
        
        self.inv_twist_mass = 1.0 / k_twist if k_twist > 1e-10 else 0.0
        
    def warm_start(self):
        """Apply friction impulses from previous frame."""
        # Implementation similar to ReactPhysics3D's warm start
        pass
        
    def solve_friction(self):
        """Solve friction constraints."""
        # Get total normal impulse
        total_normal_impulse = sum(p.penetration_impulse for p in self.contact_points 
                                  if p.constraint == self)
        
        friction_limit = self.friction_coeff * abs(total_normal_impulse)
        
        # Solve first friction direction
        v_rel = self.compute_relative_velocity()
        jv = np.dot(v_rel, self.friction_vector1)
        
        delta_lambda = -jv * self.inv_friction_mass1
        old_impulse = self.manifold.friction_impulse1
        self.manifold.friction_impulse1 = np.clip(old_impulse + delta_lambda, 
                                                  -friction_limit, friction_limit)
        delta_lambda = self.manifold.friction_impulse1 - old_impulse
        
        self.apply_friction_impulse(delta_lambda * self.friction_vector1)
        
        # Similar for second friction direction and twist...

class ContactPointConstraint:
    """Constraint data for a single contact point."""
    
    def __init__(self, contact, constraint, dt):
        self.contact = contact
        self.constraint = constraint
        self.dt = dt
        
        # Compute constraint data
        self.r1 = contact.contact_a - constraint.body_a.pos
        self.r2 = contact.contact_b - constraint.body_b.pos
        
        # Compute inverse mass matrix
        self.compute_inverse_mass()
        
        # Compute restitution bias
        self.compute_restitution_bias()
        
    def compute_inverse_mass(self):
        """Compute inverse effective mass for this contact."""
        r1_cross_n = np.cross(self.r1, self.contact.normal)
        r2_cross_n = np.cross(self.r2, self.contact.normal)
        
        k = self.constraint.inv_mass_a + self.constraint.inv_mass_b
        k += np.dot(self.contact.normal, 
                   np.cross(self.constraint.inv_inertia_a @ r1_cross_n, self.r1))
        k += np.dot(self.contact.normal,
                   np.cross(self.constraint.inv_inertia_b @ r2_cross_n, self.r2))
        
        self.inv_mass = 1.0 / k if k > 1e-10 else 0.0
        
    def solve_penetration(self):
        """Solve penetration constraint."""
        # Compute relative velocity
        v_rel = self.compute_relative_velocity()
        jv = np.dot(v_rel, self.contact.normal)
        
        # Compute bias
        beta = BETA_SPLIT_IMPULSE if self.constraint.use_split_impulse else BETA
        bias = 0.0
        if self.contact.depth > SLOP:
            bias = -(beta / self.dt) * max(0.0, self.contact.depth - SLOP)
        bias += self.restitution_bias
        
        # Compute impulse
        delta_lambda = -(jv + bias) * self.inv_mass
        old_impulse = self.contact.penetration_impulse
        self.contact.penetration_impulse = max(0.0, old_impulse + delta_lambda)
        delta_lambda = self.contact.penetration_impulse - old_impulse
        
        # Apply impulse
        self.apply_impulse(delta_lambda * self.contact.normal)