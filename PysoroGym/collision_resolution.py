import numpy as np
from .Body import Body
from .Shape import Plane

# Constants from ReactPhysics3D
BETA = 0.2  # Baumgarte stabilization
BETA_SPLIT_IMPULSE = 0.2
SLOP = 0.01  # Penetration slop
RESTITUTION_VELOCITY_THRESHOLD = 1.0  # m/s

class Contact:
    """
    Stores all information about a collision between two colliders.
    This class enforces that it is initialized with ShapeCollider objects.
    """
    def __init__(self, collider_a, collider_b, normal, depth, contact_a, contact_b):
        # --- Enforce consistent initialization ---
        # The arguments collider_a and collider_b MUST be ShapeCollider instances.
        self.collider_a = collider_a
        self.collider_b = collider_b
        
        # --- Derive all other references from the colliders ---
        # The raw geometry (e.g., Sphere, Box)
        self.shape_a = collider_a.shape
        self.shape_b = collider_b.shape
        
        # The parent physics body
        self.body_a = collider_a.body
        self.body_b = collider_b.body
        
        # --- Contact data ---
        self.normal = np.array(normal, dtype=float)
        self.depth = depth
        
        # World space contact points
        self.contact_a = np.array(contact_a, dtype=float)
        self.contact_b = np.array(contact_b, dtype=float)
        
        # --- Solver data ---
        self.shape_idx_a = -1
        self.shape_idx_b = -1
        self.restitution = 0.0
        self.friction = 0.0
        
        # --- Sanity Check ---
        # Warn if, for some reason, a body reference is still missing.
        if self.body_a is None or self.body_b is None:
            print(f"[Contact] CRITICAL: Could not resolve body reference from colliders.")

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
                constraint.contact_points.append(point)  # Link to constraint
                
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

    def store_impulses(self, manifolds):
        """Store impulses back to manifolds for warm starting next frame."""
        # Impulses are already stored directly in the manifold objects
        pass

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
        
    def compute_restitution_bias(self):
        """Compute restitution velocity bias."""
        mat_a = getattr(self.constraint.manifold.shape_a, 'material', None)
        mat_b = getattr(self.constraint.manifold.shape_b, 'material', None)
        
        restitution_a = getattr(mat_a, 'restitution', 0.0) if mat_a else 0.0
        restitution_b = getattr(mat_b, 'restitution', 0.0) if mat_b else 0.0
        restitution = max(restitution_a, restitution_b)
        
        if restitution > 0.0:
            v_rel = self.compute_relative_velocity()
            v_n = np.dot(v_rel, self.contact.normal)
            
            if v_n < -RESTITUTION_VELOCITY_THRESHOLD:
                self.restitution_bias = -restitution * v_n
            else:
                self.restitution_bias = 0.0
        else:
            self.restitution_bias = 0.0

    def compute_relative_velocity(self):
        """Compute relative velocity at contact point."""
        v_a = self.constraint.body_a.vel + np.cross(self.constraint.body_a.ang_vel, self.r1)
        v_b = self.constraint.body_b.vel + np.cross(self.constraint.body_b.ang_vel, self.r2)
        return v_b - v_a

    def apply_impulse(self, impulse):
        """Apply impulse to both bodies."""
        if self.constraint.body_a.body_type == Body.DYNAMIC:
            self.constraint.body_a.vel -= self.constraint.inv_mass_a * impulse
            self.constraint.body_a.ang_vel -= self.constraint.inv_inertia_a @ np.cross(self.r1, impulse)
    
        if self.constraint.body_b.body_type == Body.DYNAMIC:
            self.constraint.body_b.vel += self.constraint.inv_mass_b * impulse
            self.constraint.body_b.ang_vel += self.constraint.inv_inertia_b @ np.cross(self.r2, impulse)

    def warm_start(self):
        """Apply friction impulses from previous frame."""
        # Apply friction impulse 1
        if abs(self.manifold.friction_impulse1) > 1e-10:
            self.apply_friction_impulse(self.manifold.friction_impulse1 * self.friction_vector1)
    
        # Apply friction impulse 2
        if abs(self.manifold.friction_impulse2) > 1e-10:
            self.apply_friction_impulse(self.manifold.friction_impulse2 * self.friction_vector2)
    
        # Apply twist friction
        if abs(self.manifold.friction_twist_impulse) > 1e-10:
            self.apply_twist_impulse(self.manifold.friction_twist_impulse)

    def apply_twist_impulse(self, impulse):
        """Apply twist friction impulse."""
        if self.body_a.body_type == Body.DYNAMIC:
            self.body_a.ang_vel -= impulse * self.inv_inertia_a @ self.normal
    
        if self.body_b.body_type == Body.DYNAMIC:
            self.body_b.ang_vel += impulse * self.inv_inertia_b @ self.normal

    def solve_friction(self):
        """Solve friction constraints."""
        # Get total normal impulse from associated contact points
        total_normal_impulse = 0.0
        for point in self.contact_points:
            if point.constraint == self:
                total_normal_impulse += point.contact.penetration_impulse
    
        friction_limit = self.friction_coeff * abs(total_normal_impulse)
        
        # Solve first friction direction
        v_rel = self.compute_relative_velocity()
        jv1 = np.dot(v_rel, self.friction_vector1)
        
        delta_lambda1 = -jv1 * self.inv_friction_mass1
        old_impulse1 = self.manifold.friction_impulse1
        self.manifold.friction_impulse1 = np.clip(old_impulse1 + delta_lambda1, 
                                                  -friction_limit, friction_limit)
        delta_lambda1 = self.manifold.friction_impulse1 - old_impulse1
        
        if abs(delta_lambda1) > 1e-10:
            self.apply_friction_impulse(delta_lambda1 * self.friction_vector1)
        
        # Solve second friction direction
        v_rel = self.compute_relative_velocity()
        jv2 = np.dot(v_rel, self.friction_vector2)
        
        delta_lambda2 = -jv2 * self.inv_friction_mass2
        old_impulse2 = self.manifold.friction_impulse2
        self.manifold.friction_impulse2 = np.clip(old_impulse2 + delta_lambda2, 
                                                  -friction_limit, friction_limit)
        delta_lambda2 = self.manifold.friction_impulse2 - old_impulse2
        
        if abs(delta_lambda2) > 1e-10:
            self.apply_friction_impulse(delta_lambda2 * self.friction_vector2)
        
        # Solve twist friction
        ang_vel_rel = self.body_b.ang_vel - self.body_a.ang_vel
        jv_twist = np.dot(ang_vel_rel, self.normal)
        
        delta_lambda_twist = -jv_twist * self.inv_twist_mass
        old_twist = self.manifold.friction_twist_impulse
        self.manifold.friction_twist_impulse = np.clip(old_twist + delta_lambda_twist,
                                                       -friction_limit, friction_limit)
        delta_lambda_twist = self.manifold.friction_twist_impulse - old_twist
        
        if abs(delta_lambda_twist) > 1e-10:
            self.apply_twist_impulse(delta_lambda_twist)

# ---------------------------------------------------------------------
# Debug flags
DEBUG_GENERIC_CONTACT = True        # ← set to False to silence logs
# ---------------------------------------------------------------------

def clamp(value, min_value, max_value):
    """Clamp a value between min_value and max_value."""
    return max(min_value, min(value, max_value))

def resolve_plane_contact(plane_contact, dynamic_body, plane_body, dt):
    """
    Specialized contact resolution for a dynamic body against a static plane.
    """
    # print("\n=== PLANE CONTACT RESOLUTION DEBUG ===")
    # print(f"Dynamic body position: {dynamic_body.pos}")
    # print(f"Plane body position: {plane_body.pos}")
    # print(f"Initial contact normal: {plane_contact.normal}")
    # print(f"Contact point on plane: {plane_contact.contact_a}")
    # print(f"Contact point on object: {plane_contact.contact_b}")
    # print(f"Initial penetration depth: {plane_contact.depth}")
    
    # Ensure the normal points from the plane towards the dynamic body
    normal_check = np.dot(plane_contact.normal, (dynamic_body.pos - plane_body.pos))
    # print(f"Normal direction check: {normal_check}")
    
    if normal_check < 0:
        # print("WARNING: Flipping normal direction!")
        plane_contact.normal *= -1.0
        # print(f"Corrected normal: {plane_contact.normal}")

    # Relative velocity calculation
    r_b = plane_contact.contact_b - dynamic_body.pos
    # print(f"r_b vector: {r_b}")
    
    v_b = dynamic_body.vel + np.cross(dynamic_body.ang_vel, r_b)
    # print(f"Dynamic body velocity: {dynamic_body.vel}")
    # print(f"Dynamic body angular velocity: {dynamic_body.ang_vel}")
    # print(f"Total point velocity: {v_b}")
    
    normal_vel = np.dot(v_b, plane_contact.normal)
    # print(f"Normal velocity: {normal_vel}")

    # Resolve only if approaching
    if normal_vel >= 0:
        # print("Objects separating, skipping resolution")
        return

    # Simplified effective mass for one dynamic body
    inv_mass_b = dynamic_body.inv_mass
    # print(f"Inverse mass: {inv_mass_b}")
    
    inv_inertia_b = dynamic_body.inv_inertia_world()
    # print(f"Inverse inertia tensor: {inv_inertia_b}")
    
    r_b_cross_n = np.cross(r_b, plane_contact.normal)
    # print(f"r_b × normal: {r_b_cross_n}")
    
    angular_effect_b = np.dot(np.cross(inv_inertia_b @ r_b_cross_n, r_b), plane_contact.normal)
    # print(f"Angular effect term: {angular_effect_b}")
    
    inv_effective_mass = inv_mass_b + angular_effect_b
    # print(f"Inverse effective mass: {inv_effective_mass}")

    if inv_effective_mass < 1e-6:
        # print("Effective mass too small, skipping resolution")
        return

    # Impulse calculation
    restitution = 0.3
    
    # --- CHANGE: Improve impulse calculation to reduce jittering ---
    # 1. Eliminate restitution at low velocities to prevent micro-bounces
    if abs(normal_vel) < 0.2:
        restitution = 0.0
    
    j = -(1.0 + restitution) * normal_vel / inv_effective_mass
    impulse = j * plane_contact.normal

    # Apply impulse to the dynamic body
    dynamic_body.vel += inv_mass_b * impulse
    dynamic_body.ang_vel += inv_inertia_b @ np.cross(r_b, impulse)
    
    # 2. Apply damping for near-resting objects
    if abs(normal_vel) < 0.5:
        # Apply stronger damping the slower the object is moving
        damping_factor = 1.0 - min(abs(normal_vel), 0.2) / 0.5
        
        # Linear damping (more aggressive as velocity approaches zero)
        vel_magnitude = np.linalg.norm(dynamic_body.vel)
        if vel_magnitude < 0.5:
            damping_strength = 0.95 - (damping_factor * 0.1)  # 0.85-0.95 range
            dynamic_body.vel *= damping_strength
            
        # Angular damping (more aggressive)
        ang_vel_magnitude = np.linalg.norm(dynamic_body.ang_vel)
        if ang_vel_magnitude < 1.0:
            ang_damping = 0.9 - (damping_factor * 0.15)  # 0.75-0.9 range
            dynamic_body.ang_vel *= ang_damping
    
    # --- CHANGE: Improve positional correction to be more stable ---
    # Positional correction (Baumgarte stabilization)
    penetration_slop = 0.005  # Small slop to prevent jitter
    
    # 1. Calculate velocity along normal direction
    normal_velocity_magnitude = abs(normal_vel)
    
    # 2. Use an adaptive correction percent based on velocity
    # Higher velocity -> stronger correction to prevent tunneling
    base_percent = 0.8  # Base correction strength 
    velocity_scale = min(1.0, normal_velocity_magnitude / 5.0)  # Scale up to 1.0
    percent = base_percent + (velocity_scale * 0.4)  # Can go up to 1.2 for very fast objects
    
    # 3. Apply position correction with safety margin
    if plane_contact.depth > penetration_slop:
        # Basic correction based on penetration
        correction_depth = plane_contact.depth - penetration_slop
        
        # Add velocity-based safety margin to prevent tunneling
        # For fast-moving objects, push them further from the plane
        safety_margin = min(0.05, normal_velocity_magnitude * dt * 0.5)
        total_correction = correction_depth + safety_margin
        
        # Apply the correction
        correction_magnitude = (total_correction / inv_effective_mass) * percent
        correction = correction_magnitude * plane_contact.normal
        
        # Apply position correction
        dynamic_body.pos += inv_mass_b * correction
        
    # --- HIGH-VELOCITY STABILIZATION ---
    # For very fast-moving objects, add extra constraints
    if normal_velocity_magnitude > 10.0:
        # Cap maximum velocity to prevent extreme tunneling
        max_speed = 20.0
        if np.linalg.norm(dynamic_body.vel) > max_speed:
            dynamic_body.vel = dynamic_body.vel * (max_speed / np.linalg.norm(dynamic_body.vel))
            
        # For extremely fast objects, use smaller timesteps internally
        # by subdividing the impulse across multiple sub-steps
        sub_steps = min(5, int(normal_velocity_magnitude / 5.0))
        if sub_steps > 1:
            sub_dt = dt / sub_steps
            sub_impulse = impulse / sub_steps
            
            # Re-apply fractions of the impulse
            for _ in range(1, sub_steps):
                # Apply fraction of impulse
                dynamic_body.vel += (inv_mass_b * sub_impulse) 
                dynamic_body.ang_vel += inv_inertia_b @ np.cross(r_b, sub_impulse)
                
                # Move position slightly each sub-step
                dynamic_body.pos += dynamic_body.vel * sub_dt

def resolve_contact(contact, dt):
    """Dispatches contact resolution to the appropriate function."""
    # Since the Contact object is now standardized, we can access its
    # attributes with confidence. No more guessing or hasattr checks.
    
    # --- Dispatcher Logic ---
    # Case 1: Body A's shape is a plane, Body B is dynamic
    if isinstance(contact.shape_a, Plane) and contact.body_b.body_type == Body.DYNAMIC:
        resolve_plane_contact(contact, contact.body_b, contact.body_a, dt)
        return

    # Case 2: Body B's shape is a plane, Body A is dynamic
    if isinstance(contact.shape_b, Plane) and contact.body_a.body_type == Body.DYNAMIC:
        # Invert the contact normal to point from plane (B) to object (A)
        contact.normal *= -1.0
        resolve_plane_contact(contact, contact.body_a, contact.body_b, dt)
        return
        
    # --- Fallback to Generic Resolution ---
    resolve_generic_contact(contact, dt)


def resolve_generic_contact(contact, dt):
    """Generic contact resolution for two dynamic or one dynamic/one static body."""
    # --- Baumgarte Stabilization Constants ---
    BETA = 0.2   # How aggressively to correct position errors
    SLOP = 0.01  # Allowed penetration depth to prevent jittering

    # Get bodies and their properties
    body_a, body_b = contact.body_a, contact.body_b
    inv_mass_a = body_a.inv_mass if body_a.body_type == Body.DYNAMIC else 0.0
    inv_mass_b = body_b.inv_mass if body_b.body_type == Body.DYNAMIC else 0.0

    if inv_mass_a == 0.0 and inv_mass_b == 0.0:
        return  # Both bodies are static

    # 1. -------------------------------- Relative velocity & effective mass
    r_a = contact.contact_a - body_a.pos
    r_b = contact.contact_b - body_b.pos

    inv_inertia_a = body_a.inv_inertia_world() if body_a.body_type == Body.DYNAMIC else np.zeros((3, 3))
    inv_inertia_b = body_b.inv_inertia_world() if body_b.body_type == Body.DYNAMIC else np.zeros((3, 3))

    angular_effect_a = np.dot(np.cross(inv_inertia_a @ np.cross(r_a, contact.normal), r_a), contact.normal)
    angular_effect_b = np.dot(np.cross(inv_inertia_b @ np.cross(r_b, contact.normal), r_b), contact.normal)
    inv_effective_mass = inv_mass_a + inv_mass_b + angular_effect_a + angular_effect_b
    if inv_effective_mass < 1e-6:
        return

    v_a = body_a.vel + np.cross(body_a.ang_vel, r_a)
    v_b = body_b.vel + np.cross(body_b.ang_vel, r_b)
    rel_vel   = v_b - v_a
    normal_vel = np.dot(rel_vel, contact.normal)

    # 2. -------------------------------- Bias terms (restitution + Baumgarte)
    restitution        = 0.2
    restitution_bias   = 0.0 if normal_vel > -1.0 else -restitution * normal_vel
    positional_bias    = (BETA / dt) * max(0.0, contact.depth - SLOP)
    total_bias         = restitution_bias + positional_bias

    # 3. -------------------------------- Impulse
    j = -(normal_vel + total_bias) / inv_effective_mass
    impulse = j * contact.normal

    # 4. -------------------------------- Apply impulse
    if body_a.body_type == Body.DYNAMIC:
        body_a.vel     -= inv_mass_a * impulse
        body_a.ang_vel -= inv_inertia_a @ np.cross(r_a, impulse)

    if body_b.body_type == Body.DYNAMIC:
        body_b.vel     += inv_mass_b * impulse
        body_b.ang_vel += inv_inertia_b @ np.cross(r_b, impulse)

    # 5. -------------------------------- Debug output
    if DEBUG_GENERIC_CONTACT:
        print("\n--- resolve_generic_contact() ---")
        print(f"Contact normal            : {contact.normal}")
        print(f"Penetration depth         : {contact.depth}")
        print(f"Inv mass A / B            : {inv_mass_a} / {inv_mass_b}")
        print(f"r_a / r_b                 : {r_a} / {r_b}")
        print(f"Angular effect A / B      : {angular_effect_a} / {angular_effect_b}")
        print(f"Inverse effective mass    : {inv_effective_mass}")
        print(f"v_a / v_b                 : {v_a} / {v_b}")
        print(f"Relative velocity         : {rel_vel}")
        print(f"Normal relative velocity  : {normal_vel}")
        print(f"Restitution bias          : {restitution_bias}")
        print(f"Positional bias           : {positional_bias}")
        print(f"Impulse magnitude (j)     : {j}")
        print(f"Post-impulse vel/ang A    : {body_a.vel} / {body_a.ang_vel}")
        print(f"Post-impulse vel/ang B    : {body_b.vel} / {body_b.ang_vel}")