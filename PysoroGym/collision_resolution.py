import numpy as np
from .Body import Body
from .Shape import Plane

# ------------------------------------------------------------------
# Constants (tune as needed)
BETA = 0.2                      # Baumgarte position stabilization (velocity bias factor)
SLOP = 0.01                     # Penetration slop
RESTITUTION_VELOCITY_THRESHOLD = 1.0
FRICTION_EPSILON = 1e-8
NORMAL_EPSILON = 1e-8
MAX_CONTACTS_PER_MANIFOLD = 4
# ------------------------------------------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# -----------------------------------------------------------
# Basic contact data structures
# -----------------------------------------------------------
class ContactPoint:
    def __init__(self, normal, depth, local_point_a, local_point_b):
        self.normal = np.asarray(normal, dtype=float)
        self.depth = float(depth)
        self.local_point_a = np.asarray(local_point_a, dtype=float)
        self.local_point_b = np.asarray(local_point_b, dtype=float)
        # Cached world points (filled when building constraints)
        self.world_point_a = None
        self.world_point_b = None
        # Warm starting
        self.normal_impulse = 0.0
        self.tangent_impulse1 = 0.0
        self.tangent_impulse2 = 0.0
        self.twist_impulse = 0.0

class ContactManifold:
    def __init__(self, collider_a, collider_b):
        self.collider_a = collider_a
        self.collider_b = collider_b
        self.body_a = collider_a.body if collider_a is not None else None
        self.body_b = collider_b.body if collider_b is not None else None
        self.points: list[ContactPoint] = []
        # Aggregate cached impulses (optional)
        self.friction_impulse1 = 0.0
        self.friction_impulse2 = 0.0
        self.friction_twist_impulse = 0.0

    def add_contact(self, normal, depth, world_point_a, world_point_b):
        if depth < 0:
            return
        # Convert to local (simple if body has transform helpers; else store world)
        lp_a = world_point_a - self.body_a.position if self.body_a else world_point_a
        lp_b = world_point_b - self.body_b.position if self.body_b else world_point_b
        if len(self.points) < MAX_CONTACTS_PER_MANIFOLD:
            self.points.append(ContactPoint(normal, depth, lp_a, lp_b))

# Legacy single contact wrapper (kept for backward compatibility)
class Contact:
    def __init__(self, collider_a, collider_b, normal, depth,
                 point_a=None, point_b=None, contact_a=None, contact_b=None):
        # Accept old or new parameter names
        if point_a is None and contact_a is not None:
            point_a = contact_a
        if point_b is None and contact_b is not None:
            point_b = contact_b

        self.collider_a = collider_a
        self.collider_b = collider_b
        self.body_a = collider_a.body if collider_a else None
        self.body_b = collider_b.body if collider_b else None
        self.normal = np.asarray(normal, dtype=float)
        self.depth = float(depth)
        self.contact_a = np.asarray(point_a, dtype=float) if point_a is not None else np.zeros(3)
        self.contact_b = np.asarray(point_b, dtype=float) if point_b is not None else np.zeros(3)

        # Warm start / solver fields (added lazily elsewhere but define here for clarity)
        self.penetration_impulse = 0.0
        self.penetration_split_impulse = 0.0
        self.is_resting_contact = False

# -----------------------------------------------------------
# Constraint helpers
# -----------------------------------------------------------
class ContactPointConstraint:
    def __init__(self, manifold_constraint, cp: ContactPoint):
        self.manifold = manifold_constraint
        self.cp = cp

        self.body_a = manifold_constraint.body_a
        self.body_b = manifold_constraint.body_b

        # World space contact points
        if self.body_a:
            cp.world_point_a = self.body_a.position + cp.local_point_a
        else:
            cp.world_point_a = cp.local_point_a
        if self.body_b:
            cp.world_point_b = self.body_b.position + cp.local_point_b
        else:
            cp.world_point_b = cp.local_point_b

        self.world_point = 0.5 * (cp.world_point_a + cp.world_point_b)

        self.normal = manifold_constraint.normal
        self.t1 = manifold_constraint.t1
        self.t2 = manifold_constraint.t2

        # Vectors from centers to contact
        self.r_a = self.world_point - self.body_a.position if self.body_a else np.zeros(3)
        self.r_b = self.world_point - self.body_b.position if self.body_b else np.zeros(3)

        # Effective mass denominators (normal and tangents)
        self.eff_mass_n = self._compute_effective_mass(self.normal)
        self.eff_mass_t1 = self._compute_effective_mass(self.t1)
        self.eff_mass_t2 = self._compute_effective_mass(self.t2)

        # Velocity bias for restitution / Baumgarte
        self.velocity_bias = 0.0
        self._compute_velocity_bias()

    def _compute_effective_mass(self, dir_vec):
        denom = 0.0
        if self.body_a and self.body_a.body_type == Body.DYNAMIC:
            r_cross = np.cross(self.r_a, dir_vec)
            denom += self.body_a.inv_mass + r_cross @ (self.body_a.inv_inertia_world() @ r_cross)
        if self.body_b and self.body_b.body_type == Body.DYNAMIC:
            r_cross = np.cross(self.r_b, dir_vec)
            denom += self.body_b.inv_mass + r_cross @ (self.body_b.inv_inertia_world() @ r_cross)
        if denom < 1e-10:
            return 0.0
        return 1.0 / denom

    def _relative_velocity(self):
        v_a = np.zeros(3)
        v_b = np.zeros(3)
        if self.body_a:
            v_a = self.body_a.vel + np.cross(self.body_a.ang_vel, self.r_a)
        if self.body_b:
            v_b = self.body_b.vel + np.cross(self.body_b.ang_vel, self.r_b)
        return v_b - v_a

    def _compute_velocity_bias(self):
        # Baumgarte + restitution
        depth = self.cp.depth
        if depth > SLOP:
            baumgarte = BETA * (depth - SLOP)
        else:
            baumgarte = 0.0

        rel_v = self._relative_velocity()
        v_n = rel_v @ self.normal

        restitution = self.manifold.restitution
        rest_bias = 0.0
        if restitution > 0.0 and v_n < -RESTITUTION_VELOCITY_THRESHOLD:
            rest_bias = -restitution * v_n

        self.velocity_bias = baumgarte + rest_bias

    def warm_start(self):
        # Apply stored impulses
        if self.cp.normal_impulse != 0.0:
            self._apply_impulse(self.normal * self.cp.normal_impulse)
        if self.cp.tangent_impulse1 != 0.0:
            self._apply_impulse(self.t1 * self.cp.tangent_impulse1)
        if self.cp.tangent_impulse2 != 0.0:
            self._apply_impulse(self.t2 * self.cp.tangent_impulse2)

    def solve_normal(self):
        if self.eff_mass_n == 0.0:
            return
        rel_v = self._relative_velocity()
        v_rel_n = rel_v @ self.normal
        # Compute impulse scalar
        lambda_n = -(v_rel_n + self.velocity_bias) * self.eff_mass_n
        # Accumulate (clamp to keep non-negative)
        old_impulse = self.cp.normal_impulse
        self.cp.normal_impulse = max(0.0, old_impulse + lambda_n)
        delta = self.cp.normal_impulse - old_impulse
        if abs(delta) > 0.0:
            self._apply_impulse(self.normal * delta)

    def solve_friction(self):
        # Only apply friction if there is normal impulse
        normal_impulse = self.cp.normal_impulse
        if normal_impulse <= 0.0:
            return

        max_friction = self.manifold.friction * normal_impulse

        # Tangent 1
        if self.eff_mass_t1 > 0.0:
            rel_v = self._relative_velocity()
            vt1 = rel_v @ self.t1
            lambda_t1 = -vt1 * self.eff_mass_t1
            old = self.cp.tangent_impulse1
            self.cp.tangent_impulse1 = clamp(old + lambda_t1, -max_friction, max_friction)
            delta = self.cp.tangent_impulse1 - old
            if abs(delta) > 0.0:
                self._apply_impulse(self.t1 * delta)

        # Tangent 2
        if self.eff_mass_t2 > 0.0:
            rel_v = self._relative_velocity()
            vt2 = rel_v @ self.t2
            lambda_t2 = -vt2 * self.eff_mass_t2
            old2 = self.cp.tangent_impulse2
            self.cp.tangent_impulse2 = clamp(old2 + lambda_t2, -max_friction, max_friction)
            delta2 = self.cp.tangent_impulse2 - old2
            if abs(delta2) > 0.0:
                self._apply_impulse(self.t2 * delta2)

    def _apply_impulse(self, impulse):
        if self.body_a and self.body_a.body_type == Body.DYNAMIC:
            self.body_a.vel -= impulse * self.body_a.inv_mass
            self.body_a.ang_vel -= self.body_a.inv_inertia_world() @ np.cross(self.r_a, impulse)
        if self.body_b and self.body_b.body_type == Body.DYNAMIC:
            self.body_b.vel += impulse * self.body_b.inv_mass
            self.body_b.ang_vel += self.body_b.inv_inertia_world() @ np.cross(self.r_b, impulse)

class ContactManifoldConstraint:
    def __init__(self, manifold: ContactManifold):
        self.manifold = manifold
        self.body_a = manifold.body_a
        self.body_b = manifold.body_b
        # Take normal from first contact (all share)
        if manifold.points:
            self.normal = normalize(manifold.points[0].normal)
        else:
            self.normal = np.array([0.0, 1.0, 0.0])
        # Build tangent basis
        self.t1, self.t2 = build_tangent_basis(self.normal)

        # Friction & restitution
        fr_a = getattr(manifold.collider_a.material, 'friction', 0.5) if manifold.collider_a else 0.5
        fr_b = getattr(manifold.collider_b.material, 'friction', 0.5) if manifold.collider_b else 0.5
        self.friction = np.sqrt(fr_a * fr_b)

        e_a = getattr(manifold.collider_a.material, 'elasticity', 0.0) if manifold.collider_a else 0.0
        e_b = getattr(manifold.collider_b.material, 'elasticity', 0.0) if manifold.collider_b else 0.0
        self.restitution = max(e_a, e_b)

        # Per-point constraints
        self.point_constraints: list[ContactPointConstraint] = []
        for cp in manifold.points:
            self.point_constraints.append(ContactPointConstraint(self, cp))

    def warm_start(self):
        for pc in self.point_constraints:
            pc.warm_start()

    def solve(self):
        # Solve normal first for all points (block solver optional)
        for pc in self.point_constraints:
            pc.solve_normal()
        # Then friction for all
        for pc in self.point_constraints:
            pc.solve_friction()

# -----------------------------------------------------------
# Solver
# -----------------------------------------------------------
class ContactSolver:
    def __init__(self, use_split_impulse=False):
        self.use_split_impulse = use_split_impulse
        self.manifold_constraints: list[ContactManifoldConstraint] = []

    def init_constraints(self, manifolds):
        self.manifold_constraints.clear()
        for m in manifolds:
            if not m.points:
                continue
            self.manifold_constraints.append(ContactManifoldConstraint(m))

    def warm_start(self):
        for mc in self.manifold_constraints:
            mc.warm_start()

    def solve(self, manifolds, dt, iterations=10):
        if not manifolds:
            return
        self.init_constraints(manifolds)
        self.warm_start()
        for _ in range(iterations):
            for mc in self.manifold_constraints:
                mc.solve()

        if self.use_split_impulse:
            for mc in self.manifold_constraints:
                for pc in mc.point_constraints:
                    self.solve_split_impulse(pc, dt)

    def solve_split_impulse(self, pc: ContactPointConstraint, dt: float):
        """Applies a direct position correction impulse."""
        if pc.cp.depth <= SLOP:
            return

        # Simplified position correction logic
        baumgarte_bias = (BETA / dt) * max(0.0, pc.cp.depth - SLOP)
        
        # We need to re-calculate relative velocity at the contact point
        # for the split impulse, but since we apply it to position, we can simplify.
        # Here we calculate an impulse-like value to directly modify position.
        # This is a simplified model.
        
        lambda_p = -baumgarte_bias * pc.eff_mass_n

        # To avoid overcorrection, we can clamp the correction.
        # For simplicity, we apply a fraction of the impulse directly.
        # A more robust solution would use split velocities.
        
        correction = pc.normal * lambda_p * (dt * dt) # Heuristic scaling

        # Apply positional correction
        if pc.body_a and pc.body_a.body_type == Body.DYNAMIC:
            pc.body_a.pos -= correction * pc.body_a.inv_mass
        if pc.body_b and pc.body_b.body_type == Body.DYNAMIC:
            pc.body_b.pos += correction * pc.body_b.inv_mass


# -----------------------------------------------------------
# Utility math
# -----------------------------------------------------------
def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    return v / n

def build_tangent_basis(n):
    n = normalize(n)
    if abs(n[0]) < 0.9:
        other = np.array([1.0, 0.0, 0.0])
    else:
        other = np.array([0.0, 1.0, 0.0])
    t1 = normalize(np.cross(n, other))
    t2 = normalize(np.cross(n, t1))
    return t1, t2

# -----------------------------------------------------------
# Legacy single contact resolution (still callable)
# -----------------------------------------------------------
def resolve_plane_contact(plane_contact, dynamic_body, plane_body, dt):
    # Fallback simple impulse resolution with positional correction
    normal = plane_contact.normal
    rel_vel_n = dynamic_body.vel @ normal
    if rel_vel_n < 0.0:
        inv_mass = dynamic_body.inv_mass
        j = -(1.0 + 0.0) * rel_vel_n / (inv_mass)
        impulse = j * normal
        dynamic_body.vel += impulse * inv_mass
    # Positional correction
    if plane_contact.depth > SLOP:
        correction = (plane_contact.depth - SLOP) * 0.8
        dynamic_body.position += normal * correction

def resolve_generic_contact(contact: Contact, dt):
    a = contact.body_a
    b = contact.body_b
    if a is None or b is None:
        return
    if a.body_type != Body.DYNAMIC and b.body_type != Body.DYNAMIC:
        return

    n = normalize(contact.normal)
    ra = contact.contact_a - a.position
    rb = contact.contact_b - b.position

    va = a.vel + np.cross(a.ang_vel, ra)
    vb = b.vel + np.cross(b.ang_vel, rb)
    rel_v = vb - va
    rel_n = rel_v @ n
    if rel_n > 0.0:
        return

    inv_mass = 0.0
    if a.body_type == Body.DYNAMIC:
        inv_mass += a.inv_mass + np.cross(ra, n) @ (a.inv_inertia_world() @ np.cross(ra, n))
    if b.body_type == Body.DYNAMIC:
        inv_mass += b.inv_mass + np.cross(rb, n) @ (b.inv_inertia_world() @ np.cross(rb, n))
    if inv_mass < 1e-12:
        return
    j = -(1.0 + 0.0) * rel_n / inv_mass

    impulse = j * n
    if a.body_type == Body.DYNAMIC:
        a.vel -= impulse * a.inv_mass
        a.ang_vel -= a.inv_inertia_world() @ np.cross(ra, impulse)
    if b.body_type == Body.DYNAMIC:
        b.vel += impulse * b.inv_mass
        b.ang_vel += b.inv_inertia_world() @ np.cross(rb, impulse)

def resolve_contact(contact, dt):
    resolve_generic_contact(contact, dt)