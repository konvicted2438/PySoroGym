# filepath: c:\Users\shi178\OneDrive - CSIRO\Desktop\Python_scripts\PySoroGym\PysoroGym\world.py
"""
World class that manages the physics simulation and rendering.
"""
import pygame
import numpy as np
from PysoroGym.physics import PhysicsEngine
from PysoroGym.Body import Body


class World:
    """World manages physics simulation and rendering"""
    
    def __init__(self, gravity=(0, -9.81, 0), renderer=None, dt=1.0/60.0):
        # Physics engine
        self.physics_engine = PhysicsEngine(gravity=gravity, dt=dt)
        
        # Renderer
        self.renderer = renderer
        
        # Bodies in the world
        self.bodies = []
        
        # Simulation state
        self.running = True
        self.paused = False
        self.simulation_time = 0.0
        
        # Performance tracking
        self.physics_time = 0.0
        self.render_time = 0.0
        
        # Add callback support
        self.physics_update_callback = None
        
    def add(self, body):
        """Add a body to the world"""
        if body not in self.bodies:
            self.bodies.append(body)
            self.physics_engine.add_body(body)
            
    def remove(self, body):
        """Remove a body from the world"""
        if body in self.bodies:
            self.bodies.remove(body)
            self.physics_engine.remove_body(body)
            
    def step(self):
        """Perform one simulation step"""
        if not self.paused:
            # Time physics
            import time
            start_time = time.time()
            
            # Call physics update callback if provided
            if self.physics_update_callback is not None:
                self.physics_update_callback(self.physics_engine.dt)
            
            self.physics_engine.step()
            self.simulation_time += self.physics_engine.dt
            
            self.physics_time = time.time() - start_time
            
    def render(self):
        """Render the world"""
        if self.renderer is None:
            return
            
        import time
        start_time = time.time()
        
        # Prepare shape data for renderer
        render_shapes = []
        
        for body in self.bodies:
            # Update shapes to current body pose
            body.update_shapes()
            
            for shape, material in body.shapes.items():
                # Create render data
                render_data = RenderData(
                    shape=shape,
                    transform=body.get_world_transform(),
                    color=self._get_shape_color(body, shape)
                )
                render_shapes.append(render_data)
                
        # We're no longer adding contact normal lines here
        # (the previous code that added contact normals has been removed)
                    
        # Simulation info
        sim_info = {
            'time': self.simulation_time,
            'paused': self.paused,
            'bodies': len(self.bodies),
            'contacts': sum(len(c.contacts) for c in self.physics_engine.collision_pairs),
            'physics_ms': self.physics_time * 1000,
            'render_ms': self.render_time * 1000
        }
        
        # Render
        self.renderer.render(shapes=render_shapes, sim_info=sim_info)
        
        self.render_time = time.time() - start_time
        
    def _get_shape_color(self, body, shape):
        """Get color for a shape based on body type"""
        # Check if shape has custom color
        if hasattr(shape, 'color') and shape.color is not None:
            return shape.color
            
        # Otherwise use default coloring based on body type
        if body.body_type == Body.STATIC:
            return (0.5, 0.5, 0.5)  # Gray for static
        elif body.body_type == Body.KINEMATIC:
            return (0.5, 0.5, 1.0)  # Blue for kinematic
        else:
            # Dynamic bodies - different colors
            colors = [
                (1.0, 0.5, 0.5),  # Red
                (0.5, 1.0, 0.5),  # Green
                (0.5, 0.5, 1.0),  # Blue
                (1.0, 1.0, 0.5),  # Yellow
                (1.0, 0.5, 1.0),  # Magenta
                (0.5, 1.0, 1.0),  # Cyan
            ]
            # Use body index for consistent coloring
            index = self.bodies.index(body) % len(colors)
            return colors[index]
            
    def run(self):
        """Run the simulation loop"""
        import time
        target_fps = 60
        frame_time = 1.0 / target_fps
        
        # Use pygame clock only if renderer is available
        if self.renderer is not None:
            clock = pygame.time.Clock()
    
        while self.running:
            # Check if renderer exists and is still running
            if self.renderer is not None:
                if not self.renderer.running:
                    break
                # Handle events through renderer
                self.renderer.handle_events(physics_simulator=self)
            
            # Physics step
            self.step()
            
            # Render only if renderer exists
            if self.renderer is not None:
                self.render()
                # Control frame rate with pygame clock
                clock.tick(target_fps)
            else:
                # Simple time-based frame rate control when running headless
                time.sleep(max(0, frame_time - self.physics_time))
    
        # Only quit pygame if it was initialized (renderer exists)
        if self.renderer is not None:
            pygame.quit()
        
    def reset(self):
        """Reset the simulation"""
        # Remove all dynamic bodies
        bodies_to_remove = [b for b in self.bodies if b.body_type == Body.DYNAMIC]
        for body in bodies_to_remove:
            self.remove(body)
            
        self.simulation_time = 0.0
        
    def spawn_random_object(self):
        """Spawn a random object at a random position"""
        import random
        from PysoroGym import shapes
        
        # Random position above ground
        x = random.uniform(-3, 3)
        y = random.uniform(5, 10)
        z = random.uniform(-3, 3)
        
        # Create random shape
        shape_type = random.choice(['sphere', 'box', 'cylinder', 'capsule'])
        
        body = Body(body_type=Body.DYNAMIC, position=(x, y, z))
        
        if shape_type == 'sphere':
            radius = random.uniform(0.3, 0.8)
            shape = shapes.Sphere(radius)
        elif shape_type == 'box':
            size = [random.uniform(0.5, 1.5) for _ in range(3)]
            shape = shapes.Box(size)
        elif shape_type == 'cylinder':
            radius = random.uniform(0.3, 0.6)
            height = random.uniform(0.5, 1.5)
            shape = shapes.Cylinder(radius, height)
        else:  # capsule
            radius = random.uniform(0.2, 0.5)
            height = random.uniform(0.5, 1.0)
            shape = shapes.Capsule(radius, height)
            
        body.add_shape(shape)
        body.add_shape(shape)
        
        # Random initial velocity
        body.linear_velocity = np.array([
            random.uniform(-2, 2),
            random.uniform(-1, 1),
            random.uniform(-2, 2)
        ])
        
        body.angular_velocity = np.array([
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(-5, 5)
        ])
        
        self.add(body)

    def close(self):
        """Clean up resources when world is no longer needed"""
        self.running = False
        
        # Clean up renderer if it exists
        if self.renderer is not None:
            # Ensure pygame is properly quit if it was used
            pygame.quit()
            
        # Clear references to bodies
        self.bodies.clear()
        self.physics_engine.clear_bodies()


class RenderData:
    """Data for rendering a shape"""
    def __init__(self, shape, transform, color):
        self.shape = shape
        self.transform = transform
        self.color = color
        

class ContactNormalLine:
    """Line for visualizing contact normals"""
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.color = (1.0, 1.0, 0.0)  # Yellow for contact normals