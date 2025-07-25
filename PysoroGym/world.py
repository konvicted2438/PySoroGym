import numpy as np
import pygame
from .visulisation import OpenGLRenderer
from .math_utils import q_to_euler
from .collision import gjk  # Only import gjk from collision
from .collision_resolution import resolve_contact  # Import from collision_resolution
from .aabb import AABBTree
from .Body import Body


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
        self.collision_iterations = 3
        self.collision_pairs = []
        self.contacts = []  # Store contacts for visualization
        
        # Broad phase AABB tree
        self.aabb_tree = AABBTree(margin=0.1)
        
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
    
    # ––––– Physics Pipeline –––––
    def step(self):
        """Advance simulation by dt seconds"""
        if self.paused:
            return
        
        # Update timestep
        
        # 1. Apply gravity and forces
        for body in self.bodies:
            if not body.is_static and not body.is_sleeping:
                body.apply_force(self.gravity * body.mass)
        
        # 2. Integrate velocities (forces → velocities)
        for body in self.bodies:
            body.integrate_velocities(self.dt)
            
        # 3. Update AABB tree for moving bodies
        for body in self.bodies:
            if not body.is_static and not body.is_sleeping:
                self.aabb_tree.update(body)
            
        # 4. Broad phase collision detection using AABB tree
        self.collision_pairs = self._broad_phase_collision_detection()
        
        # 5. Narrow phase collision detection
        self.contacts = self._narrow_phase_collision_detection()
        
        # 6. Collision response (multiple iterations for stability)
        for _ in range(self.collision_iterations):
            self._solve_collisions()
            
        # 7. Integrate positions (velocities → positions)
        for body in self.bodies:
            body.integrate_positions(self.dt)
            
        # 8. Update sleep states
        self._update_sleeping_bodies()
        
        self.time += self.dt
    
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
            for shape_a in body_a.shapes:
                for shape_b in body_b.shapes:
                    # Use the new GJK-based collision detection
                    contact = gjk(shape_a, shape_b)
                    if contact:
                        contacts.append(contact)
        
        return contacts  # Fixed: return contacts instead of setting self.contacts
    
    def _solve_collisions(self):
        """Apply impulses to resolve collisions."""
        for contact in self.contacts:
            resolve_contact(contact, self.dt)
    
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
                rs.shape = shape_collider  # Fixed: shape_collider is already the shape
                
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