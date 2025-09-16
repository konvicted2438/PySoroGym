"""
Physics constraints and solvers.
"""
import numpy as np
from typing import List, Optional
from PysoroGym.collision import CollisionPair, Contact
from PysoroGym.physics import Material


class ConstraintSolver:
    """Solves physics constraints including collisions"""
    
    def __init__(self, dt: float = 1.0/60.0):
        self.dt = dt
        self.restitution_threshold = 1.0
        self.friction_enabled = True
        self.warm_starting_enabled = False
        self.contact_cache = {}
        self.position_damping = 0.8  # Damping factor for position correction
        self.linear_velocity_threshold = 0.01
        self.angular_velocity_threshold = 0.01
        self.linear_velocity_damping = 0.005
        self.angular_velocity_damping = 0.005
    def solve_velocity_constraints(self, collision_pairs: List[CollisionPair], iterations: int = 8):
        """Solve velocity constraints for all collision pairs"""
        for _ in range(iterations):
            for collision in collision_pairs:
                for contact in collision.contacts:
                    self._solve_contact_velocity(
                        collision.body_a, collision.body_b, 
                        contact, collision.shape_a, collision.shape_b
                    )
                    
    def solve_position_constraints(self, collision_pairs: List[CollisionPair], 
                                 iterations: int = 3, max_penetration: float = 0.01,
                                 baumgarte_factor: float = 0.2):
        """Solve position constraints (penetration correction)"""
        for _ in range(iterations):
            for collision in collision_pairs:
                for contact in collision.contacts:
                    if contact.penetration > max_penetration:
                        self._solve_contact_position(
                            collision.body_a, collision.body_b,
                            contact, max_penetration, baumgarte_factor
                        )
                        
    def _solve_contact_velocity(self, body_a, body_b, contact, shape_a, shape_b):
        """Solve velocity constraint for a single contact"""
        # Get contact points and normal
        point_a = contact.point_a
        point_b = contact.point_b
        normal = np.ravel(contact.normal)  # Ensure normal is flattened to 1D
        
        # Calculate relative velocity - ensure proper reshaping
        vel_a = body_a.get_velocity_at_point(point_a)
        vel_b = body_b.get_velocity_at_point(point_b)
        
        # Ensure velocities are 3-element vectors
        vel_a = np.ravel(vel_a)[:3]  # Take only first 3 elements and flatten
        vel_b = np.ravel(vel_b)[:3]  # Take only first 3 elements and flatten
        
        relative_velocity = vel_b - vel_a
        
        # Check if separating - ensure result is scalar
        velocity_along_normal = np.dot(relative_velocity, normal)
        if isinstance(velocity_along_normal, np.ndarray):
            velocity_along_normal = float(velocity_along_normal.item())  # Convert to scalar
        
        # For kinematic bodies pushing dynamic bodies, we need special handling
        if body_a.body_type == 'kinematic' and body_b.body_type == 'dynamic':
            # Kinematic body A is pushing dynamic body B
            # We want to set B's velocity to match A's at the contact point
            if velocity_along_normal < 0:  # Objects are colliding
                # Calculate required velocity change for body B
                target_velocity = np.dot(vel_a, normal)
                current_velocity = np.dot(vel_b, normal)
                velocity_change = target_velocity - current_velocity
                
                # Apply impulse to body B to match kinematic body's velocity
                impulse_magnitude = body_b.mass * velocity_change
                impulse = normal * impulse_magnitude
                body_b.linear_velocity += impulse / body_b.mass
                
                # Also apply angular impulse if needed
                if body_b.inertia_tensor is not None:
                    r_b = point_b - body_b.get_world_center_of_mass()
                    body_b.angular_velocity += np.linalg.inv(body_b.inertia_tensor) @ np.cross(r_b, impulse)
            # Don't return here! Let position correction happen below
            
        elif body_b.body_type == 'kinematic' and body_a.body_type == 'dynamic':
            # Kinematic body B is pushing dynamic body A
            if velocity_along_normal > 0:  # Objects are colliding (note: sign is flipped)
                # Calculate required velocity change for body A
                target_velocity = np.dot(vel_b, normal)
                current_velocity = np.dot(vel_a, normal)
                velocity_change = current_velocity - target_velocity
                
                # Apply impulse to body A to match kinematic body's velocity
                impulse_magnitude = body_a.mass * velocity_change
                impulse = normal * impulse_magnitude
                body_a.linear_velocity -= impulse / body_a.mass
                
                # Also apply angular impulse if needed
                if body_a.inertia_tensor is not None:
                    r_a = point_a - body_a.get_world_center_of_mass()
                    body_a.angular_velocity -= np.linalg.inv(body_a.inertia_tensor) @ np.cross(r_a, impulse)
            # Don't return here! Let position correction happen below
        
        # For kinematic-dynamic pairs, skip the normal dynamic collision resolution
        # but DO NOT skip position correction
        if (body_a.body_type == 'kinematic' and body_b.body_type == 'dynamic') or \
           (body_b.body_type == 'kinematic' and body_a.body_type == 'dynamic'):
            return  # Skip the rest of velocity resolution, but position correction will still run
        
        # Only resolve if objects are approaching
        if velocity_along_normal > 0.001:  # Add small tolerance for numerical stability
            return
        
        # Get materials for restitution and friction
        restitution = 0.0
        friction = 0.0
        if hasattr(body_a, 'material') and hasattr(body_b, 'material'):
            if body_a.material and body_b.material:
                restitution = min(body_a.material.elasticity, body_b.material.elasticity)
                friction = (body_a.material.friction + body_b.material.friction) * 0.5
        
        # For resting contacts, apply stronger damping
        is_resting = abs(velocity_along_normal) < self.restitution_threshold
        if is_resting:
            restitution = 0.0  # No bounce
            
            # Apply stronger damping for bodies that are nearly at rest
            # This helps stabilize objects on the ground
            if body_a.body_type == 'dynamic' and not body_a.is_sleeping:
                linear_speed = np.linalg.norm(body_a.linear_velocity)
                angular_speed = np.linalg.norm(body_a.angular_velocity)
                
                # Apply progressive damping based on speed
                if linear_speed < 0.5:  # Increased threshold
                    damping_factor = 1.0 - (linear_speed / 0.5) * 0.1  # 0.9 to 1.0
                    body_a.linear_velocity *= (1.0 - damping_factor * 0.1)
                    
                if angular_speed < 0.5:  # Apply to angular too
                    damping_factor = 1.0 - (angular_speed / 0.5) * 0.1
                    body_a.angular_velocity *= (1.0 - damping_factor * 0.2)  # Stronger angular damping
                    
            if body_b.body_type == 'dynamic' and not body_b.is_sleeping:
                linear_speed = np.linalg.norm(body_b.linear_velocity)
                angular_speed = np.linalg.norm(body_b.angular_velocity)
                
                if linear_speed < 0.5:
                    damping_factor = 1.0 - (linear_speed / 0.5) * 0.1
                    body_b.linear_velocity *= (1.0 - damping_factor * 0.1)
                    
                if angular_speed < 0.5:
                    damping_factor = 1.0 - (angular_speed / 0.5) * 0.1
                    body_b.angular_velocity *= (1.0 - damping_factor * 0.2)
        
        # Use the proper impulse calculation that includes angular effects
        normal_impulse_magnitude = self._calculate_normal_impulse(
            body_a, body_b, point_a, point_b, normal, velocity_along_normal, restitution
        )
        
        # Apply the normal impulse (this handles both linear and angular velocity)
        normal_impulse = normal * normal_impulse_magnitude
        self._apply_impulse(body_a, body_b, point_a, point_b, normal_impulse)
        
        # Apply friction if enabled
        if self.friction_enabled and friction > 0:
            # Recalculate relative velocity after normal impulse
            vel_a_new = body_a.get_velocity_at_point(point_a)
            vel_b_new = body_b.get_velocity_at_point(point_b)
            relative_velocity_new = vel_b_new - vel_a_new
            
            self._apply_friction_impulse(
                body_a, body_b, point_a, point_b, normal, 
                normal_impulse_magnitude, friction, relative_velocity_new
            )
                
    def solve_position_constraints(self, collision_pairs: List[CollisionPair], 
                                 iterations: int = 3, max_penetration: float = 0.01,
                                 baumgarte_factor: float = 0.2):
        """Solve position constraints (penetration correction)"""
        for _ in range(iterations):
            for collision in collision_pairs:
                for contact in collision.contacts:
                    if contact.penetration > max_penetration:
                        self._solve_contact_position(
                            collision.body_a, collision.body_b,
                            contact, max_penetration, baumgarte_factor
                        )
                        
    def _solve_contact_velocity(self, body_a, body_b, contact, shape_a, shape_b):
        """Solve velocity constraint for a single contact"""
        # Get contact points and normal
        point_a = contact.point_a
        point_b = contact.point_b
        normal = np.ravel(contact.normal)  # Ensure normal is flattened to 1D
        #print(f"Solving contact velocity: {point_a}, {point_b}, normal: {normal}")
        
        # Calculate relative velocity - ensure proper reshaping
        vel_a = body_a.get_velocity_at_point(point_a)
        vel_b = body_b.get_velocity_at_point(point_b)
        
        # Ensure velocities are 3-element vectors
        vel_a = np.ravel(vel_a)[:3]  # Take only first 3 elements and flatten
        vel_b = np.ravel(vel_b)[:3]  # Take only first 3 elements and flatten
        
        relative_velocity = vel_b - vel_a
        
        # Check if separating - ensure result is scalar
        velocity_along_normal = np.dot(relative_velocity, normal)
        if isinstance(velocity_along_normal, np.ndarray):
            velocity_along_normal = float(velocity_along_normal.item())  # Convert to scalar
        
        # For kinematic bodies pushing dynamic bodies, we need special handling
        if body_a.body_type == 'kinematic' and body_b.body_type == 'dynamic':
            # Kinematic body A is pushing dynamic body B
            # We want to set B's velocity to match A's at the contact point
            if velocity_along_normal < 0:  # Objects are colliding
                # Calculate required velocity change for body B
                target_velocity = np.dot(vel_a, normal)
                current_velocity = np.dot(vel_b, normal)
                velocity_change = target_velocity - current_velocity
                
                # Apply impulse to body B to match kinematic body's velocity
                impulse_magnitude = body_b.mass * velocity_change
                impulse = normal * impulse_magnitude
                body_b.linear_velocity += impulse / body_b.mass
                
                # Also apply angular impulse if needed
                if body_b.inertia_tensor is not None:
                    r_b = point_b - body_b.get_world_center_of_mass()
                    body_b.angular_velocity += np.linalg.inv(body_b.inertia_tensor) @ np.cross(r_b, impulse)
            # Don't return here! Let position correction happen below
            
        elif body_b.body_type == 'kinematic' and body_a.body_type == 'dynamic':
            # Kinematic body B is pushing dynamic body A
            if velocity_along_normal > 0:  # Objects are colliding (note: sign is flipped)
                # Calculate required velocity change for body A
                target_velocity = np.dot(vel_b, normal)
                current_velocity = np.dot(vel_a, normal)
                velocity_change = current_velocity - target_velocity
                
                # Apply impulse to body A to match kinematic body's velocity
                impulse_magnitude = body_a.mass * velocity_change
                impulse = normal * impulse_magnitude
                body_a.linear_velocity -= impulse / body_a.mass
                
                # Also apply angular impulse if needed
                if body_a.inertia_tensor is not None:
                    r_a = point_a - body_a.get_world_center_of_mass()
                    body_a.angular_velocity -= np.linalg.inv(body_a.inertia_tensor) @ np.cross(r_a, impulse)
            # Don't return here! Let position correction happen below
        
        # For kinematic-dynamic pairs, skip the normal dynamic collision resolution
        # but DO NOT skip position correction
        if (body_a.body_type == 'kinematic' and body_b.body_type == 'dynamic') or \
           (body_b.body_type == 'kinematic' and body_a.body_type == 'dynamic'):
            return  # Skip the rest of velocity resolution, but position correction will still run
        
        # Only resolve if objects are approaching
        if velocity_along_normal > 0.001:  # Add small tolerance for numerical stability
            return
        
        # Get materials for restitution and friction
        restitution = 0.0
        friction = 0.0
        if hasattr(body_a, 'material') and hasattr(body_b, 'material'):
            if body_a.material and body_b.material:
                restitution = min(body_a.material.elasticity, body_b.material.elasticity)
                friction = (body_a.material.friction + body_b.material.friction) * 0.5
        
        # For resting contacts, apply stronger damping
        is_resting = abs(velocity_along_normal) < self.restitution_threshold
        if is_resting:
            restitution = 0.0  # No bounce
            
            # Apply stronger damping for bodies that are nearly at rest
            # This helps stabilize objects on the ground
            if body_a.body_type == 'dynamic' and not body_a.is_sleeping:
                linear_speed = np.linalg.norm(body_a.linear_velocity)
                angular_speed = np.linalg.norm(body_a.angular_velocity)
                
                # Apply progressive damping based on speed
                if linear_speed < 0.5:  # Increased threshold
                    damping_factor = 1.0 - (linear_speed / 0.5) * 0.1  # 0.9 to 1.0
                    body_a.linear_velocity *= (1.0 - damping_factor * 0.1)
                    
                if angular_speed < 0.5:  # Apply to angular too
                    damping_factor = 1.0 - (angular_speed / 0.5) * 0.1
                    body_a.angular_velocity *= (1.0 - damping_factor * 0.2)  # Stronger angular damping
                    
            if body_b.body_type == 'dynamic' and not body_b.is_sleeping:
                linear_speed = np.linalg.norm(body_b.linear_velocity)
                angular_speed = np.linalg.norm(body_b.angular_velocity)
                
                if linear_speed < 0.5:
                    damping_factor = 1.0 - (linear_speed / 0.5) * 0.1
                    body_b.linear_velocity *= (1.0 - damping_factor * 0.1)
                    
                if angular_speed < 0.5:
                    damping_factor = 1.0 - (angular_speed / 0.5) * 0.1
                    body_b.angular_velocity *= (1.0 - damping_factor * 0.2)
        
        # Use the proper impulse calculation that includes angular effects
        normal_impulse_magnitude = self._calculate_normal_impulse(
            body_a, body_b, point_a, point_b, normal, velocity_along_normal, restitution
        )
        
        # Apply the normal impulse (this handles both linear and angular velocity)
        normal_impulse = normal * normal_impulse_magnitude
        self._apply_impulse(body_a, body_b, point_a, point_b, normal_impulse)
        
        # Apply friction if enabled
        if self.friction_enabled and friction > 0:
            # Recalculate relative velocity after normal impulse
            vel_a_new = body_a.get_velocity_at_point(point_a)
            vel_b_new = body_b.get_velocity_at_point(point_b)
            relative_velocity_new = vel_b_new - vel_a_new
            
            self._apply_friction_impulse(
                body_a, body_b, point_a, point_b, normal, 
                normal_impulse_magnitude, friction, relative_velocity_new
            )
                
    def _calculate_normal_impulse(self, body_a, body_b, point_a, point_b, 
                                normal, velocity_along_normal, restitution):
        """Calculate normal impulse magnitude (using center of mass)"""
        # Relative positions from center of mass
        r_a = point_a - body_a.get_world_center_of_mass()
        r_b = point_b - body_b.get_world_center_of_mass()
        
        # Inverse masses
        inv_mass_a = 1.0 / body_a.mass if body_a.body_type == 'dynamic' else 0.0
        inv_mass_b = 1.0 / body_b.mass if body_b.body_type == 'dynamic' else 0.0
        
        # Angular contributions
        angular_a = np.zeros(3)
        angular_b = np.zeros(3)
        
        if body_a.body_type == 'dynamic' and body_a.inertia_tensor is not None:
            angular_a = np.cross(np.linalg.inv(body_a.inertia_tensor) @ np.cross(r_a, normal), r_a)
            
        if body_b.body_type == 'dynamic' and body_b.inertia_tensor is not None:
            angular_b = np.cross(np.linalg.inv(body_b.inertia_tensor) @ np.cross(r_b, normal), r_b)
            
        # Effective mass
        denominator = inv_mass_a + inv_mass_b + np.dot(angular_a + angular_b, normal)
        
        if denominator <= 0:
            return 0.0
            
        # Target velocity after collision
        target_velocity = 0
        if abs(velocity_along_normal) > self.restitution_threshold:
            target_velocity = -restitution * velocity_along_normal
            
        # Impulse magnitude
        return (target_velocity - velocity_along_normal) / denominator
        
    def _apply_impulse(self, body_a, body_b, point_a, point_b, impulse):
        """Apply impulse to bodies (using center of mass for calculations)"""
        if body_a.body_type == 'dynamic':
            body_a.linear_velocity -= impulse / body_a.mass
            if body_a.inertia_tensor is not None:
                # Vector from center of mass to contact point
                r_a = point_a - body_a.get_world_center_of_mass()
                body_a.angular_velocity -= np.linalg.inv(body_a.inertia_tensor) @ np.cross(r_a, impulse)
                
        if body_b.body_type == 'dynamic':
            body_b.linear_velocity += impulse / body_b.mass
            if body_b.inertia_tensor is not None:
                # Vector from center of mass to contact point
                r_b = point_b - body_b.get_world_center_of_mass()
                body_b.angular_velocity += np.linalg.inv(body_b.inertia_tensor) @ np.cross(r_b, impulse)

    def _apply_friction_impulse(self, body_a, body_b, point_a, point_b, 
                              normal, normal_impulse, friction, relative_velocity):
        """Apply friction impulse"""
        # Calculate tangent direction
        tangent_velocity = relative_velocity - normal * np.dot(relative_velocity, normal)
        tangent_speed = np.linalg.norm(tangent_velocity)
        
        # Apply friction even for very small tangent velocities to prevent sliding
        if tangent_speed < 0.0001:
            return
            
        tangent = tangent_velocity / tangent_speed
        
        # Calculate friction impulse magnitude
        max_static_friction = abs(normal_impulse) * friction * 1.2  # Higher static friction
        max_dynamic_friction = abs(normal_impulse) * friction
        
        # Calculate impulse to stop tangential motion
        r_a = point_a - body_a.position
        r_b = point_b - body_b.position
        
        inv_mass_a = 1.0 / body_a.mass if body_a.body_type == 'dynamic' else 0.0
        inv_mass_b = 1.0 / body_b.mass if body_b.body_type == 'dynamic' else 0.0
        
        # Similar calculation as normal impulse but for tangent
        angular_a = np.zeros(3)
        angular_b = np.zeros(3)
        
        if body_a.body_type == 'dynamic' and body_a.inertia_tensor is not None:
            angular_a = np.cross(np.linalg.inv(body_a.inertia_tensor) @ np.cross(r_a, tangent), r_a)
            
        if body_b.body_type == 'dynamic' and body_b.inertia_tensor is not None:
            angular_b = np.cross(np.linalg.inv(body_b.inertia_tensor) @ np.cross(r_b, tangent), r_b)
            
        denominator = inv_mass_a + inv_mass_b + np.dot(angular_a + angular_b, tangent)
        
        if denominator > 0:
            # Calculate impulse needed to stop all tangential motion
            friction_impulse_magnitude = -tangent_speed / denominator
            
            # Determine if we should use static or dynamic friction
            if abs(friction_impulse_magnitude) <= max_static_friction:
                # Static friction: completely stop tangential motion
                friction_impulse = tangent * friction_impulse_magnitude
            else:
                # Dynamic friction: apply maximum friction force
                friction_impulse = tangent * np.sign(friction_impulse_magnitude) * max_dynamic_friction
            
            # Apply friction impulse
            self._apply_impulse(body_a, body_b, point_a, point_b, friction_impulse)
            
    def _solve_contact_position(self, body_a, body_b, contact: Contact, 
                          max_penetration: float, baumgarte_factor: float):
        """Solve position constraint for penetration correction"""
        penetration = contact.penetration
        normal = np.ravel(contact.normal)  # Ensure normal is 1D
        
        # Only correct if penetration exceeds threshold
        if penetration <= max_penetration:
            return
            
        # We want to completely eliminate penetration, not just reduce to max_penetration
        # So we use the full penetration amount minus a small slop value
        slop = 0.0005  # Reduced slop even further
        correction = max(penetration - slop, 0.0) * baumgarte_factor * self.position_damping
        
        # Get inverse masses
        inv_mass_a = 1.0 / body_a.mass if body_a.body_type == 'dynamic' else 0.0
        inv_mass_b = 1.0 / body_b.mass if body_b.body_type == 'dynamic' else 0.0
        
        # For kinematic-dynamic collisions, only move the dynamic body
        if body_a.body_type == 'kinematic' and body_b.body_type == 'dynamic':
            # Move dynamic body B away from kinematic body A
            body_b.position += normal * correction
        elif body_b.body_type == 'kinematic' and body_a.body_type == 'dynamic':
            # Move dynamic body A away from kinematic body B
            body_a.position -= normal * correction
        else:
            # Normal case: distribute correction based on mass
            total_inv_mass = inv_mass_a + inv_mass_b
            if total_inv_mass > 0:
                # Apply position correction
                correction_factor = correction / total_inv_mass
                if body_a.body_type == 'dynamic':
                    body_a.position -= normal * correction_factor * inv_mass_a
                if body_b.body_type == 'dynamic':
                    body_b.position += normal * correction_factor * inv_mass_b