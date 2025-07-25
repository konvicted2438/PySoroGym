import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import math

class OpenGLRenderer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.running = True
        
        # Camera parameters (FPS style)
        self.camera_pos = [0.0, 2.0, 5.0]
        self.camera_yaw = -90.0
        self.camera_pitch = 0.0
        
        # Camera vectors
        self.camera_front = [0.0, 0.0, -1.0]
        self.camera_up = [0.0, 1.0, 0.0]
        self.camera_right = [1.0, 0.0, 0.0]
        
        # Movement speeds
        self.movement_speed = 0.1
        self.mouse_sensitivity = 0.1
        self.last_mouse_pos = None
        
        # Mouse control
        self.mouse_captured = False
        
        # Create GLU quadric for soft robot segments
        self.quadric = None
        
        # Initialize pygame and OpenGL
        self.initialize()
        
        # Debugging options
        self.show_contact_normals = True  # Toggle for showing contact normals
        self.normal_length = 1.0  # Length of normal arrows

    def initialize(self):
        """Initialize pygame and OpenGL"""
        pygame.init()
        pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("CFM Physics Simulator - WASD to move, Mouse to look, C to toggle mouse capture, ESC to exit")
        
        # Set up OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set background color
        glClearColor(0.596, 0.651, 0.667, 1.0)
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        
        # Set up initial camera
        self.update_camera()
        
        # Set up lighting
        light_pos = [10.0, 10.0, 10.0, 1.0]
        light_ambient = [0.3, 0.3, 0.3, 1.0]
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        light_specular = [1.0, 1.0, 1.0, 1.0]
        
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
        
        # Create quadric for soft robot segments
        self.quadric = gluNewQuadric()
        gluQuadricNormals(self.quadric, GLU_SMOOTH)
    
    def handle_events(self, physics_simulator=None):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_c:
                    self.mouse_captured = not self.mouse_captured
                    if self.mouse_captured:
                        pygame.mouse.set_visible(False)
                        pygame.event.set_grab(True)
                        pygame.mouse.set_pos(self.width // 2, self.height // 2)
                        self.last_mouse_pos = None
                    else:
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)
                elif event.key == pygame.K_p and physics_simulator:  # Pause/unpause simulation
                    physics_simulator.pause_simulation()
                elif event.key == pygame.K_r and physics_simulator:  # Reset simulation
                    physics_simulator.reset_simulation()
                elif event.key == pygame.K_n:
                    self.show_contact_normals = not self.show_contact_normals
                    print(f"Contact normal visualization: {'ON' if self.show_contact_normals else 'OFF'}")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.normal_length *= 1.5
                    print(f"Normal arrow length: {self.normal_length}")
                elif event.key == pygame.K_MINUS:
                    self.normal_length /= 1.5
                    print(f"Normal arrow length: {self.normal_length}")

        # Handle mouse movement
        if self.mouse_captured:
            mouse_pos = pygame.mouse.get_pos()
            center_x, center_y = self.width // 2, self.height // 2
            
            mouse_dx = mouse_pos[0] - center_x
            mouse_dy = center_y - mouse_pos[1]
            
            if mouse_dx != 0 or mouse_dy != 0:
                self.camera_yaw += mouse_dx * self.mouse_sensitivity
                self.camera_pitch += mouse_dy * self.mouse_sensitivity
                self.camera_pitch = max(-89.0, min(89.0, self.camera_pitch))
                pygame.mouse.set_pos(center_x, center_y)
                    
        # Handle keyboard movement
        keys = pygame.key.get_pressed()
        actual_speed = self.movement_speed
        if keys[pygame.K_LCTRL]:
            actual_speed *= 3.0
        
        if keys[pygame.K_w]:
            for i in range(3):
                self.camera_pos[i] += self.camera_front[i] * actual_speed
        if keys[pygame.K_s]:
            for i in range(3):
                self.camera_pos[i] -= self.camera_front[i] * actual_speed
        if keys[pygame.K_a]:
            for i in range(3):
                self.camera_pos[i] -= self.camera_right[i] * actual_speed
        if keys[pygame.K_d]:
            for i in range(3):
                self.camera_pos[i] += self.camera_right[i] * actual_speed
        if keys[pygame.K_SPACE]:
            self.camera_pos[1] += actual_speed
        if keys[pygame.K_LSHIFT]:
            self.camera_pos[1] -= actual_speed
            
    def update_camera_vectors(self):
        """Update camera direction vectors based on yaw and pitch"""
        yaw_rad = math.radians(self.camera_yaw)
        pitch_rad = math.radians(self.camera_pitch)
        
        # Calculate new front vector
        front_x = math.cos(yaw_rad) * math.cos(pitch_rad)
        front_y = math.sin(pitch_rad)
        front_z = math.sin(yaw_rad) * math.cos(pitch_rad)
        
        # Normalize front vector
        length = math.sqrt(front_x**2 + front_y**2 + front_z**2)
        self.camera_front = [front_x/length, front_y/length, front_z/length]
        
        # Calculate right vector (cross product of front and world up)
        world_up = [0.0, 1.0, 0.0]
        self.camera_right = [
            self.camera_front[1] * world_up[2] - self.camera_front[2] * world_up[1],
            self.camera_front[2] * world_up[0] - self.camera_front[0] * world_up[2],
            self.camera_front[0] * world_up[1] - self.camera_front[1] * world_up[0]
        ]
        
        # Normalize right vector
        length = math.sqrt(sum(x**2 for x in self.camera_right))
        self.camera_right = [x/length for x in self.camera_right]
        
        # Calculate up vector (cross product of right and front)
        self.camera_up = [
            self.camera_right[1] * self.camera_front[2] - self.camera_right[2] * self.camera_front[1],
            self.camera_right[2] * self.camera_front[0] - self.camera_right[0] * self.camera_front[2],
            self.camera_right[0] * self.camera_front[1] - self.camera_right[1] * self.camera_front[0]
        ]
        
    def update_camera(self):
        """Update camera view matrix"""
        self.update_camera_vectors()
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Calculate look-at point
        look_at = [
            self.camera_pos[0] + self.camera_front[0],
            self.camera_pos[1] + self.camera_front[1],
            self.camera_pos[2] + self.camera_front[2]
        ]
        
        gluLookAt(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
                  look_at[0], look_at[1], look_at[2],
                  self.camera_up[0], self.camera_up[1], self.camera_up[2])

    def render(self, shapes=None, sim_info=None):
        """Main render function - only handles drawing"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Update camera
        self.update_camera()
        
        # Draw reference grid
        self.draw_grid(size=10, divisions=20)
        
        # Render physics shapes
        if shapes:
            self.render_shapes(shapes)
        
        # Update UI info with simulation data
        self._render_ui_info(sim_info)
        
        pygame.display.flip()
    
    def _render_ui_info(self, sim_info=None):
        """Render UI information overlay"""
        if sim_info:
            sim_time = sim_info.get("time", 0)
            paused = sim_info.get("paused", False)
            status = "PAUSED" if paused else "RUNNING"
            title = f"CFM Physics Simulator - {status} - Time: {sim_time:.2f}s - P:Pause R:Reset C:Mouse ESC:Exit"
            pygame.display.set_caption(title)
    
    def render_shapes(self, shapes):
        """Render a list of shapes"""
        for shape in shapes:
            self.render_shape(shape)
            
    def render_shape(self, shape):
        """Render a single shape based on mesh data"""
        glPushMatrix()
        
        # Apply transformations
        if hasattr(shape, 'position'):
            glTranslatef(*shape.position)
        if hasattr(shape, 'rotation'):
            glRotatef(shape.rotation[0], 1, 0, 0)
            glRotatef(shape.rotation[1], 0, 1, 0)
            glRotatef(shape.rotation[2], 0, 0, 1)
        
        # Set color
        if hasattr(shape, 'color'):
            glColor3f(*shape.color)
        
        # Special case for lines (no mesh)
        if hasattr(shape, 'start_point') and hasattr(shape, 'end_point'):
            self._render_line(shape)
        # Special case for SoftRobot
        elif hasattr(shape, 'backbone_points'):
            self._render_soft_robot(shape)
        # Handle physics body with mesh-based shape
        elif hasattr(shape, 'shape') and hasattr(shape.shape, 'vertices') and hasattr(shape.shape, 'indices'):
            # Render mesh from shape attribute
            self._render_mesh(shape.shape.vertices, shape.shape.indices, 
                             shape.shape.normals if hasattr(shape.shape, 'normals') else None)
        # Direct mesh data
        elif hasattr(shape, 'vertices') and hasattr(shape, 'indices'):
            # Render mesh directly
            self._render_mesh(shape.vertices, shape.indices,
                             shape.normals if hasattr(shape, 'normals') else None)
            
        glPopMatrix()
    
    def _render_mesh(self, vertices, indices, normals=None):
        """Render mesh from vertices and indices"""
        glBegin(GL_TRIANGLES)
        for face in indices:
            for i in face:
                if normals is not None and i < len(normals):
                    glNormal3fv(normals[i])
                glVertex3fv(vertices[i])
        glEnd()
        
    def _render_line(self, line):
        """Render a line"""
        glDisable(GL_LIGHTING)
        glLineWidth(line.width if hasattr(line, 'width') else 1.0)
        glBegin(GL_LINES)
        glVertex3f(*line.start_point)
        glVertex3f(*line.end_point)
        glEnd()
        glEnable(GL_LIGHTING)
    
    def _render_soft_robot(self, robot):
        """Render a soft robot as a swept cylinder along the backbone"""
        if len(robot.backbone_points) < 2:
            return
            
        # Render as connected cylinders between backbone points
        for i in range(len(robot.backbone_points) - 1):
            self._render_robot_segment(robot.backbone_points[i], 
                                     robot.backbone_points[i + 1], 
                                     robot.radius)
        
        # Also render backbone as a line for visualization
        glDisable(GL_LIGHTING)
        glColor3f(0.9, 0.3, 0.3)  # Slightly different color for backbone
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        for point in robot.backbone_points:
            glVertex3f(*point)
        glEnd()
        glEnable(GL_LIGHTING)
    
    def _render_robot_segment(self, start_point, end_point, radius):
        """Render a single segment of the robot as a cylinder"""
        # Calculate vector from start to end
        direction = np.array(end_point) - np.array(start_point)
        length = np.linalg.norm(direction)
        
        if length < 1e-6:  # Avoid division by zero
            return
            
        direction = direction / length  # Normalize
        
        # Calculate rotation to align cylinder with the segment
        # Default cylinder is along Z-axis, we need to rotate to align with direction
        up = np.array([0, 0, 1])
        
        # Calculate rotation axis and angle
        if np.allclose(direction, up):
            # Already aligned
            angle = 0
            axis = [1, 0, 0]
        elif np.allclose(direction, -up):
            # Opposite direction
            angle = 180
            axis = [1, 0, 0]
        else:
            # General case
            axis = np.cross(up, direction)
            axis = axis / np.linalg.norm(axis)
            angle = np.degrees(np.arccos(np.clip(np.dot(up, direction), -1, 1)))
        
        glPushMatrix()
        
        # Move to start point
        glTranslatef(*start_point)
        
        # Rotate to align with segment direction
        if angle != 0:
            glRotatef(angle, *axis)
        
        # Render cylinder
        gluCylinder(self.quadric, radius, radius, length, 16, 1)
        
        glPopMatrix()
        
    def draw_grid(self, size=10, divisions=20):
        """Draw reference grid"""
        glDisable(GL_LIGHTING)
        glColor3f(0.4, 0.4, 0.4)
        glLineWidth(1.0)
        
        step = size * 2 / divisions
        glBegin(GL_LINES)
        for i in range(divisions + 1):
            x = -size + i * step
            z = -size + i * step
            glVertex3f(x, 0.01, -size)
            glVertex3f(x, 0.01, size)
            glVertex3f(-size, 0.01, z)
            glVertex3f(size, 0.01, z)
        glEnd()
        glEnable(GL_LIGHTING)
        
    def run(self, physics_simulator=None):
        """Main application loop"""
        self.initialize()
        clock = pygame.time.Clock()
        
        while self.running:
            self.handle_events(physics_simulator)
            
            # Get shapes from physics simulator
            shapes = []
            if physics_simulator:
                shapes = physics_simulator.get_shapes()
                physics_simulator.update()
            
            self.render(shapes)
            clock.tick(60)
            
        pygame.quit()


# Validation function
def validate_mesh_rendering():
    """Test function to demonstrate mesh-based rendering"""
    import sys
    import os
    
    # Add parent directory to path to import Shape
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from PysoroGym.Shape import Sphere, Box, Cylinder, Plane, Polyhedron
    
    class SimpleBody:
        """Simple wrapper to hold a shape with position and color"""
        def __init__(self, shape, position=(0,0,0), color=(0.8,0.2,0.2)):
            self.shape = shape
            self.position = position
            self.color = color
    
    # Create renderer
    renderer = OpenGLRenderer(width=1024, height=768)
    
    # Create sample shapes using mesh-based representations
    shapes = [
        # Create a sphere at origin
        SimpleBody(Sphere(0.5), position=(0, 1, 0), color=(0.8, 0.2, 0.2)),
        
        # Create a box to the right
        SimpleBody(Box([0.4, 0.4, 0.4]), position=(1.5, 1, 0), color=(0.2, 0.8, 0.2)),
        
        # Create a cylinder to the left
        SimpleBody(Cylinder(0.4, 1.0), position=(-1.5, 1, 0), color=(0.2, 0.2, 0.8)),
        
        # Create a ground plane
        SimpleBody(Plane([10, 10], 10), position=(0, 0, 0), color=(0.5, 0.5, 0.5)),
        
        # Create a custom polyhedron (tetrahedron)
        SimpleBody(Polyhedron(
            vertices=np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0.5, 0, 0.866],
                [0.5, 0.816, 0.289]
            ]),
            indices=np.array([
                [0, 1, 2],
                [0, 1, 3],
                [1, 2, 3],
                [0, 2, 3]
            ])
        ), position=(-1.5, 2, 2), color=(0.8, 0.8, 0.2))
    ]
    
    # Main loop
    clock = pygame.time.Clock()
    sim_info = {"time": 0, "paused": False}
    
    while renderer.running:
        # Update time
        sim_info["time"] += 1/60
        
        # Handle events
        renderer.handle_events()
        
        # Render shapes
        renderer.render(shapes, sim_info)
        
        # Cap at 60fps
        clock.tick(60)

# Run validation if this file is executed directly
if __name__ == "__main__":
    validate_mesh_rendering()


