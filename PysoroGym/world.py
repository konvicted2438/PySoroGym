import numpy as np
import pygame
from .visulisation import OpenGLRenderer
from .math_utils import q_to_euler, q_to_mat3  # Add q_to_mat3 here
from .collision import gjk  # Only import gjk from collision
# --- CHANGE: Import the new solver components ---
from .collision_resolution import Contact, resolve_contact, ContactSolver, ContactManifold
from .aabb import AABBTree
from .Body import Body
from .Shape import Plane  # Also need to import Plane for isinstance checks


class World:
    """
    Physics world that manages simulation, rendering and collision detection.
    Provides a complete physics pipeline similar to Bullet/PhysX/pymunk.
    """
    def __init__(self, gravity=(0, -9.81, 0), renderer=None, dt=1.0/60.0):
        self.gravity = np.array(gravity, dtype=float)
        self.dt = dt
        self.bodies = []
        self.time = 0.0
        self.paused = False
        
        # Optional renderer
        self.renderer = renderer
        
        # Collision detection settings
        self.collision_iterations = 10  # Increased for better stability
        self.collision_pairs = []
        self.contacts = []  # Store contacts for visualization
        
        # Broad phase AABB tree
        self.aabb_tree = AABBTree(margin=0.1)
        
        # Store contacts between frames for warm starting
        self.persistent_contacts = {}  # Key: (body_id_a, body_id_b, shape_idx_a, shape_idx_b)
        
        # --- NEW: Instantiate the contact solver ---
        self.contact_solver = ContactSolver(use_split_impulse=True)
    
    # ––––– Body Management –––––
    def add(self, body):
        """Add a physics body to the world."""
        self.bodies.append(body)
        # Add to AABB tree
        self.aabb_tree.insert(body)
        return body
        
    def remove(self, body):
        """Remove a body from the world."""
        if body in self.bodies:
            self.bodies.remove(body)
            # Remove from AABB tree
            self.aabb_tree.remove(body)
            # Remove persistent contacts involving this body
            keys_to_remove = []
            for key in self.persistent_contacts:
                if id(body) in key[:2]:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.persistent_contacts[key]
    
    # ––––– Physics Pipeline –––––
    def step(self, dt=None):
        """Advance simulation by dt seconds"""
        self.contacts.clear()  # Use clear() to ensure complete removal
        if self.paused:
            return
        
        # Use provided dt or default
        if dt is None:
            dt = self.dt
        
        # 1. Apply gravity and forces
        for body in self.bodies:
            if not body.is_static and not body.is_sleeping:
                body.apply_force(self.gravity * body.mass)
        
        # 2. Integrate velocities (forces → velocities)
        for body in self.bodies:
            body.integrate_velocities(dt)
            
        # 3. Update AABB tree for moving bodies
        for body in self.bodies:
            if not body.is_static and not body.is_sleeping:
                self.aabb_tree.update(body)
            
        # 4. Broad phase collision detection using AABB tree
        self.collision_pairs = self._broad_phase_collision_detection()
        #print(f"Collision pairs found: {len(self.collision_pairs)}")
        
        # 5. Narrow phase collision detection
        new_contacts = self._narrow_phase_collision_detection()
        
        # 6. Update persistent contacts
        self._update_persistent_contacts(new_contacts)
        
        # 7. Collision response (multiple iterations for stability)
        # --- CHANGE: Use the new ContactSolver ---
        # Group contacts into manifolds
        manifolds = self._group_contacts_into_manifolds()
        
        # Wake up bodies involved in any contact
        for manifold in manifolds:
            if manifold.body_a and manifold.body_a.is_sleeping:
                manifold.body_a.wake_up()
            if manifold.body_b and manifold.body_b.is_sleeping:
                manifold.body_b.wake_up()
        
        # Solve all contact constraints iteratively
        self.contact_solver.solve(manifolds, dt, self.collision_iterations)
            
        # 8. NEW: Add tunneling prevention specifically for planes
        self._prevent_plane_tunneling(dt)
        
        # 9. Integrate positions (velocities → positions)
        for body in self.bodies:
            body.integrate_positions(dt)
            
        # 10. Update sleep states
        self._update_sleeping_bodies()
        
        self.time += dt
    
    def _broad_phase_collision_detection(self):
        """Use AABB tree for efficient broad phase collision detection."""
        # Get all potentially colliding pairs from AABB tree
        pairs = self.aabb_tree.query_pairs()
        
        # Filter out some invalid pairs
        valid_pairs = []
        for body_a, body_b in pairs:
            # Don't check collisions between static bodies
            if not (body_a.is_static and body_b.is_static):
                # Don't check sleeping bodies against each other
                if not (body_a.is_sleeping and body_b.is_sleeping):
                    valid_pairs.append((body_a, body_b))
        
        return valid_pairs
    
    def _narrow_phase_collision_detection(self):
        """Narrow phase collision detection using GJK/EPA."""
        contacts = []
        
        # Get potential collision pairs from broad phase
        for (body_a, body_b) in self.collision_pairs:
            # Skip if both are static
            if body_a.body_type == Body.STATIC and body_b.body_type == Body.STATIC:
                continue
            
            # Check collision between all shape pairs
            for shape_idx_a, shape_a in enumerate(body_a.shapes):
                for shape_idx_b, shape_b in enumerate(body_b.shapes):
                    # Use the new GJK-based collision detection
                    contact = gjk(shape_a, shape_b)
                    if contact:
                        # Store shape indices for persistent contact tracking
                        contact.shape_idx_a = shape_idx_a
                        contact.shape_idx_b = shape_idx_b
                        contacts.append(contact)
        
        return contacts
    
    def _update_persistent_contacts(self, new_contacts):
        """Update persistent contacts for warm starting."""
        # Clear current contacts
        self.contacts = []
        
        # Group new contacts by body pair
        for contact in new_contacts:
            key = (
                id(contact.body_a), 
                id(contact.body_b),
                getattr(contact, 'shape_idx_a', 0),
                getattr(contact, 'shape_idx_b', 0)
            )
            
            # Check if this is a persistent contact
            if key in self.persistent_contacts:
                old_contact = self.persistent_contacts[key]
                # Transfer accumulated impulses for warm starting
                contact.penetration_impulse = old_contact.penetration_impulse
                contact.penetration_split_impulse = old_contact.penetration_split_impulse
                contact.is_resting_contact = True
            else:
                # New contact
                contact.penetration_impulse = 0.0
                contact.penetration_split_impulse = 0.0
                contact.is_resting_contact = False
            
            # Store for next frame
            self.persistent_contacts[key] = contact
            self.contacts.append(contact)
        
        # Remove old contacts that no longer exist
        current_keys = set()
        for contact in new_contacts:
            key = (
                id(contact.body_a), 
                id(contact.body_b),
                getattr(contact, 'shape_idx_a', 0),
                getattr(contact, 'shape_idx_b', 0)
            )
            current_keys.add(key)
        
        keys_to_remove = []
        for key in self.persistent_contacts:
            if key not in current_keys:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.persistent_contacts[key]
    
    def _group_contacts_into_manifolds(self):
        """Group individual contacts into manifolds for the solver."""
        manifolds = {}  # Use a dict to group by body pair
        
        for contact in self.contacts:
            # Create a unique, order-independent key for the body pair
            key = tuple(sorted((id(contact.body_a), id(contact.body_b))))
            
            if key not in manifolds:
                # Create a new manifold. The ContactManifold will handle body references.
                manifolds[key] = ContactManifold(contact.collider_a, contact.collider_b)
            
            # Add the contact point to the existing manifold
            manifolds[key].add_contact(
                contact.normal, contact.depth, contact.contact_a, contact.contact_b
            )
            
        return list(manifolds.values())

    def _update_sleeping_bodies(self):
        """Check and update sleep state of bodies for performance."""
        for body in self.bodies:
            if not body.is_static and not body.is_kinematic:
                # Check if body is moving slowly enough to sleep
                vel_mag = np.linalg.norm(body.vel)
                ang_vel_mag = np.linalg.norm(body.ang_vel)
                
                if vel_mag < body.sleep_threshold and ang_vel_mag < body.sleep_threshold:
                    body.sleeping_timer += self.dt
                    if body.sleeping_timer > 1.0:  # Sleep after 1 second of low movement
                        body.is_sleeping = True
                else:
                    body.sleeping_timer = 0
                    body.is_sleeping = False
    
    # ––––– Spatial Queries –––––
    def query_region(self, min_point, max_point):
        """Query all bodies in a region."""
        from .aabb import AABB
        query_aabb = AABB(min_point, max_point)
        return self.aabb_tree.query_aabb(query_aabb)
    
    def ray_cast(self, origin, direction, max_distance=float('inf')):
        """Cast a ray and find first hit body."""
        # Simple implementation - can be optimized with AABB tree
        closest_hit = None
        closest_distance = max_distance
        
        for body in self.bodies:
            # Skip sleeping bodies optionally
            for shape in body.shapes:
                # Simplified ray casting - would need shape-specific implementation
                # This is a placeholder
                pass
                
        return closest_hit
    
    # ––––– Rendering Support –––––
    def get_render_shapes(self):
        """Get shapes for rendering from bodies."""
        render_shapes = []
        
        for body in self.bodies:
            # Process each shape attached to the body
            for shape_collider in body.shapes:
                # Create a renderable shape
                class RenderShape:
                    pass
                
                rs = RenderShape()
                rs.position = body.position
                rs.rotation = body.rotation
                rs.shape = shape_collider.shape  # Fixed: shape_collider is already the shape
                
                # Set color based on body type
                if body.is_static:
                    rs.color = (0.5, 0.5, 0.5)  # Grey for static
                elif body.is_kinematic:
                    rs.color = (0.2, 0.6, 0.8)  # Blue for kinematic
                elif body.is_sleeping:
                    rs.color = (0.8, 0.6, 0.2)  # Orange for sleeping
                else:
                    rs.color = (0.8, 0.2, 0.2)  # Red for dynamic
                
                render_shapes.append(rs)
        
        return render_shapes
    
    # ––––– Simulation Control –––––
    def toggle_pause(self):
        """Toggle simulation pause state."""
        self.paused = not self.paused
        return self.paused
    
    def reset_simulation(self):
        """Reset all dynamic bodies."""
        # Clear persistent contacts
        self.persistent_contacts.clear()
        self.contacts.clear()
        
        # Reset bodies to initial state (if stored)
        # To be implemented - could store initial states
        pass
    
    # ––––– Run Loop –––––
    def run(self):
        """Run the main simulation loop with rendering."""
        if self.renderer is None:
            self.renderer = OpenGLRenderer()
            
        clock = pygame.time.Clock()
        
        while self.renderer.running:
            # Handle renderer events
            self.renderer.handle_events()
            
            # Advance physics
            self.step()
            
            # Get shapes for rendering
            shapes = self.get_render_shapes()
            
            # Build simulation info
            sim_info = {
                "time": self.time, 
                "paused": self.paused,
                "bodies": len(self.bodies),
                "contacts": len(self.contacts),
                "collision_pairs": len(self.collision_pairs)
            }
            
            # Render
            self.renderer.render(shapes, sim_info)
            
            # Cap framerate
            clock.tick(60)
    
    def _prevent_plane_tunneling(self, dt):
        """Prevent fast-moving objects from tunneling through planes."""
        # Find all static plane bodies in the world
        from .Shape import Plane
        
        # Loop through all bodies to find static planes
        static_plane_bodies = []
        for body in self.bodies:
            if body.is_static:
                # Check if any of the shapes are planes
                for shape_collider in body.shapes:
                    if isinstance(shape_collider.shape, Plane):
                        static_plane_bodies.append(body)
                        break
        
        # Loop through all found static planes
        for plane_body in static_plane_bodies:
            # Check all shapes in the plane body (typically just one)
            for plane_collider in plane_body.shapes:
                plane_shape = plane_collider.shape
                if not isinstance(plane_shape, Plane):
                    continue
                
                # Get plane parameters in world space
                rotation_matrix = q_to_mat3(plane_body.q)
                plane_normal = rotation_matrix @ plane_shape.normal
                plane_normal = plane_normal / np.linalg.norm(plane_normal)
                plane_point = plane_body.pos
                
                # Check all dynamic bodies that might tunnel through
                for body in self.bodies:
                    if body.is_static or body.is_sleeping:
                        continue
                    
                    # Skip bodies that are already in contact (they're being handled)
                    already_in_contact = False
                    for contact in self.contacts:
                        if (contact.body_a is body and contact.body_b is plane_body) or \
                           (contact.body_b is body and contact.body_a is plane_body):
                            already_in_contact = True
                            break
                    
                    if already_in_contact:
                        continue
                    
                    # For each shape in the body
                    for shape_collider in body.shapes:
                        # Calculate current and predicted positions
                        current_pos = body.pos
                        
                        # Predict where the body will be after integration
                        predicted_pos = current_pos + body.vel * dt
                        
                        # Check current and predicted side of the plane
                        current_side = np.dot(current_pos - plane_point, plane_normal)
                        predicted_side = np.dot(predicted_pos - plane_point, plane_normal)
                        
                        # If crossing from above to below the plane in one step
                        if current_side >= 0 and predicted_side < 0:
                            # Calculate where it would hit the plane
                            t = current_side / (current_side - predicted_side)
                            hit_point = current_pos + t * (predicted_pos - current_pos)
                            
                            # 1. Limit velocity to prevent tunneling
                            # Calculate how much to scale back velocity
                            scale_factor = 0.8 * t  # Keep 80% of allowed velocity
                            
                            # Apply the velocity reduction
                            body.vel *= scale_factor
                            
                            # 2. Add a small upward correction to ensure it stays above
                            safety_margin = 0.01  # Small buffer above the plane
                            min_dist_from_plane = 0.02  # Minimum allowed distance
                            
                            # Calculate required position adjustment
                            required_height = min_dist_from_plane - predicted_side
                            if required_height > 0:
                                body.pos += required_height * plane_normal