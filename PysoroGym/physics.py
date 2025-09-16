"""
Core physics simulation module.
"""
import numpy as np
from typing import List, Tuple, Optional
from PysoroGym.materials import Material
from PysoroGym.collision import CollisionDetector
from PysoroGym.constraints import ConstraintSolver


class PhysicsEngine:
    """Main physics engine that handles simulation"""
    
    def __init__(self, gravity=(0, -9.81, 0), dt=1.0/60.0):
        self.gravity = np.array(gravity)
        self.dt = dt
        self.bodies = []
        self.constraints = []
        
        # Components
        self.collision_detector = CollisionDetector()
        self.constraint_solver = ConstraintSolver(dt)
        
        # Collision pairs from last step
        self.collision_pairs = []
        
        # Simulation parameters
        self.velocity_iterations = 8
        self.position_iterations = 3
        self.max_penetration = 0.01
        self.baumgarte_factor = 0.2
        
    def add_body(self, body):
        """Add a body to the simulation"""
        if body not in self.bodies:
            self.bodies.append(body)
            body.physics_engine = self
            
    def remove_body(self, body):
        """Remove a body from the simulation"""
        if body in self.bodies:
            self.bodies.remove(body)
            body.physics_engine = None
            
    def step(self):
        """Perform one physics step"""
    
        # Update sleep states
        for body in self.bodies:
            body.update_sleep_state(self.dt)
            
        # Apply forces and integrate
        for body in self.bodies:
            if body.body_type == 'dynamic' and not body.is_sleeping:
                # Apply gravity to center of mass
                body.apply_force(body.mass * self.gravity)
                
                # Integrate velocities
                body.linear_velocity += (body.force / body.mass) * self.dt
                
                # Integrate angular velocity about center of mass
                if body.inertia_tensor is not None:
                    body.angular_velocity += np.linalg.inv(body.inertia_tensor) @ body.torque * self.dt
                
                # Apply damping
                body.linear_velocity *= (1.0 - body.linear_damping * self.dt)
                body.angular_velocity *= (1.0 - body.angular_damping * self.dt)
                
                # Clear forces
                body.force = np.zeros(3)
                body.torque = np.zeros(3)
        
        # Detect collisions - wake up bodies if needed
        collision_pairs = self.collision_detector.detect_collisions(self.bodies)
        
        # Wake up bodies involved in collisions
        for collision in collision_pairs:
            collision.body_a.wake_up()
            collision.body_b.wake_up()
        
        # Store collision pairs
        self.collision_pairs = collision_pairs
        
        # Solve velocity constraints
        self.constraint_solver.solve_velocity_constraints(
            self.collision_pairs, self.velocity_iterations
        )
        
        # Integrate positions (center of mass)
        for body in self.bodies:
            if body.body_type == 'dynamic' and not body.is_sleeping:
                # Update center of mass position
                body.position += body.linear_velocity * self.dt
                
                # Update orientation using quaternion
                if np.linalg.norm(body.angular_velocity) > 0:
                    angle = np.linalg.norm(body.angular_velocity) * self.dt
                    axis = body.angular_velocity / np.linalg.norm(body.angular_velocity)
                    rotation_quat = quaternion_from_axis_angle(axis, angle)
                    body.orientation = quaternion_multiply(rotation_quat, body.orientation)
                    body.orientation = body.orientation / np.linalg.norm(body.orientation)
                
                # Update shape poses
                body.update_shapes()
        
        # Solve position constraints
        self.constraint_solver.solve_position_constraints(
            self.collision_pairs, self.position_iterations,
            self.max_penetration, self.baumgarte_factor
        )
        
        # Post-stabilization: Zero out very small velocities
        for body in self.bodies:
            if body.body_type == 'dynamic' and not body.is_sleeping:
                # Check if body is nearly at rest on a surface
                is_resting = False
                for collision in collision_pairs:
                    if (collision.body_a == body or collision.body_b == body):
                        for contact in collision.contacts:
                            # Check if contact normal is mostly vertical (resting on ground)
                            if abs(contact.normal[1]) > 0.7:  # Y is up
                                is_resting = True
                                break
                
                if is_resting:
                    # Apply strong damping for resting objects
                    if np.linalg.norm(body.linear_velocity) < 0.1:
                        body.linear_velocity *= 0.9
                    if np.linalg.norm(body.angular_velocity) < 0.1:
                        body.angular_velocity *= 0.8
                        
                    # Zero out if extremely small
                    if np.linalg.norm(body.linear_velocity) < 0.01:
                        body.linear_velocity = np.zeros(3)
                    if np.linalg.norm(body.angular_velocity) < 0.01:
                        body.angular_velocity = np.zeros(3)
        
    def quaternion_to_body(self, orientation, vector):
        """Convert vector from world to body space using quaternion"""
        # Rotate by conjugate of orientation quaternion
        conj = quaternion_conjugate(orientation)
        rotation_matrix = quaternion_to_matrix(conj)
        return rotation_matrix @ vector
        
    def quaternion_to_world(self, orientation, vector):
        """Convert vector from body to world space using quaternion"""
        rotation_matrix = quaternion_to_matrix(orientation)
        return rotation_matrix @ vector
            
    def apply_gravity(self, body):
        """Apply gravity force to a body"""
        if body.body_type == 'dynamic' and body.mass > 0:
            body.force += self.gravity * body.mass
            
    def set_gravity(self, gravity):
        """Set gravity vector"""
        self.gravity = np.array(gravity)
        
    def get_collision_info(self):
        """Get information about current collisions"""
        info = {
            'num_pairs': len(self.collision_pairs),
            'total_contacts': sum(len(pair.contacts) for pair in self.collision_pairs),
            'pairs': []
        }
        
        for pair in self.collision_pairs:
            pair_info = {
                'body_a': id(pair.body_a),
                'body_b': id(pair.body_b),
                'num_contacts': len(pair.contacts),
                'max_penetration': max(c.penetration for c in pair.contacts) if pair.contacts else 0
            }
            info['pairs'].append(pair_info)
            
        return info
    
    def clear_bodies(self):
        """Clear all bodies and reset the physics engine state"""
        # Clear the list of bodies
        self.bodies = []
        
        # Reset collision-related data structures
        self.collision_pairs = []
        
        # Reset any cached collision data
        if hasattr(self, 'collision_cache'):
            self.collision_cache.clear()
        
        # Reset any constraint or joint data
        if hasattr(self, 'constraints'):
            self.constraints.clear()
        
        # Reset any spatial acceleration structures
        if hasattr(self, 'spatial_hash') or hasattr(self, 'bvh_tree'):
            # Re-initialize spatial partitioning if used
            self._init_spatial_partitioning()
        
        # Reset contact points
        if hasattr(self, 'contacts'):
            self.contacts.clear()
            
        # Reset any solvers
        if hasattr(self, 'solver'):
            self.solver.reset()


# Quaternion utilities
def quaternion_from_axis_angle(axis, angle):
    """Convert axis-angle to quaternion"""
    half_angle = angle / 2
    s = np.sin(half_angle)
    return np.array([np.cos(half_angle), s * axis[0], s * axis[1], s * axis[2]])
    
def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_conjugate(q):
    """Get quaternion conjugate"""
    return np.array([q[0], -q[1], -q[2], -q[3]])
    
def quaternion_to_matrix(q):
    """Convert quaternion to 3x3 rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])