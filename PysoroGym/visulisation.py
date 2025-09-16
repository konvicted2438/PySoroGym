import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import math
from distance3d import colliders

class OpenGLRenderer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.running = True
        
        # Camera parameters (FPS style)
        self.camera_pos = [0.0, 2.0, 6.5]
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
        """Initialize pygame and OpenGL with Anti-Aliasing."""
        pygame.init()
        
        # --- Enable Multisample Anti-Aliasing (MSAA) ---
        # This must be done BEFORE setting the display mode.
        # 4x MSAA is a good balance of quality and performance.
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        
        pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("PysoroGym Physics Simulator - WASD to move, Mouse to look, C to toggle mouse capture, ESC to exit")
        
        # --- Enable the GL features for AA and high quality rendering ---
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Set up OpenGL
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        
        # --- Improved Lighting Setup ---
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)  # Main key light
        glEnable(GL_LIGHT1)  # Secondary fill light
        # glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE) # We will now control materials directly
        # glDisable(GL_COLOR_MATERIAL) # Ensure this is disabled
        
        # Enable smooth shading for more realistic surfaces
        glShadeModel(GL_SMOOTH)
        
        # Enable normalization to keep lighting correct even with scaled objects
        glEnable(GL_NORMALIZE)
        
        # Set up better global material properties for specular highlights
        mat_specular = [1.0, 1.0, 1.0, 1.0]
        mat_shininess = [50.0]
        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
        glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)
        
        # Set background color to a darker tone for better contrast
        glClearColor(0.518, 0.573, 0.576, 1.0)
        
        # Set up perspective (increased far plane for larger scenes)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.width / self.height), 0.1, 100.0)
        
        # Set up initial camera
        self.update_camera()
        
        # Configure the light sources
        self._setup_lighting()
        
        # Create quadric for soft robot segments
        self.quadric = gluNewQuadric()
        gluQuadricNormals(self.quadric, GLU_SMOOTH)

    def _setup_lighting(self):
        """Set up improved lighting with multiple, brighter light sources."""
        # --- Main Light (Key Light) ---
        # A bright, white light coming from the top-right.
        light0_pos = [5.0, 10.0, 5.0, 1.0]  # Positional light
        light0_ambient = [0.4, 0.4, 0.4, 1.0]  # Increased ambient contribution
        light0_diffuse = [1.0, 1.0, 1.0, 1.0]  # Already at max
        light0_specular = [1.0, 1.0, 1.0, 1.0]
        
        glLightfv(GL_LIGHT0, GL_POSITION, light0_pos)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular)
        
        # --- Secondary Light (Fill Light) ---
        # A dimmer, slightly blueish light to soften shadows.
        light1_pos = [-5.0, -5.0, -5.0, 0.0]  # Directional light
        light1_ambient = [0.0, 0.0, 0.0, 1.0]
        light1_diffuse = [0.6, 0.6, 0.7, 1.0]  # Significantly increased fill light brightness
        light1_specular = [0.5, 0.5, 0.5, 1.0]
        
        glLightfv(GL_LIGHT1, GL_POSITION, light1_pos)
        glLightfv(GL_LIGHT1, GL_AMBIENT, light1_ambient)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse)
        glLightfv(GL_LIGHT1, GL_SPECULAR, light1_specular)
        
        # Set a brighter global ambient light so shadows aren't pure black
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
    
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
                    pygame.mouse.set_visible(not self.mouse_captured)
                    if self.mouse_captured:
                        pygame.mouse.set_pos(self.width // 2, self.height // 2)
                        self.last_mouse_pos = (self.width // 2, self.height // 2)
                elif event.key == pygame.K_p and physics_simulator:
                    physics_simulator.paused = not physics_simulator.paused
                elif event.key == pygame.K_r and physics_simulator:
                    physics_simulator.reset()
                elif event.key == pygame.K_n:
                    self.show_contact_normals = not self.show_contact_normals
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.normal_length *= 1.5
                elif event.key == pygame.K_MINUS:
                    self.normal_length /= 1.5

        # Handle mouse movement
        if self.mouse_captured:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if self.last_mouse_pos:
                dx = mouse_x - self.last_mouse_pos[0]
                dy = mouse_y - self.last_mouse_pos[1]
                
                self.camera_yaw += dx * self.mouse_sensitivity
                self.camera_pitch -= dy * self.mouse_sensitivity
                self.camera_pitch = max(-89, min(89, self.camera_pitch))
                
                pygame.mouse.set_pos(self.width // 2, self.height // 2)
                self.last_mouse_pos = (self.width // 2, self.height // 2)
            else:
                self.last_mouse_pos = (mouse_x, mouse_y)
                    
        # Handle keyboard movement
        keys = pygame.key.get_pressed()
        actual_speed = self.movement_speed
        if keys[pygame.K_LCTRL]:
            actual_speed *= 3.0  # Speed boost
        
        if keys[pygame.K_w]:
            self.camera_pos[0] += actual_speed * self.camera_front[0]
            self.camera_pos[1] += actual_speed * self.camera_front[1]
            self.camera_pos[2] += actual_speed * self.camera_front[2]
        if keys[pygame.K_s]:
            self.camera_pos[0] -= actual_speed * self.camera_front[0]
            self.camera_pos[1] -= actual_speed * self.camera_front[1]
            self.camera_pos[2] -= actual_speed * self.camera_front[2]
        if keys[pygame.K_a]:
            self.camera_pos[0] -= actual_speed * self.camera_right[0]
            self.camera_pos[1] -= actual_speed * self.camera_right[1]
            self.camera_pos[2] -= actual_speed * self.camera_right[2]
        if keys[pygame.K_d]:
            self.camera_pos[0] += actual_speed * self.camera_right[0]
            self.camera_pos[1] += actual_speed * self.camera_right[1]
            self.camera_pos[2] += actual_speed * self.camera_right[2]
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
        
        # Draw the world coordinate system axes
        self.draw_axes()
        
        # # Draw reference grid
        # self.draw_grid(size=10, divisions=20)
        
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
            title = f"PysoroGym Physics Simulator - {status} - Time: {sim_time:.2f}s - P:Pause R:Reset C:Mouse ESC:Exit"
            pygame.display.set_caption(title)
    
    def render_shapes(self, shapes):
        """Render a list of shapes with their transformations and colors"""
        for shape_data in shapes:
            # Regular shape (ignore any contact normal lines)
            if hasattr(shape_data, 'shape') and hasattr(shape_data, 'transform'):
                self.render_shape(shape_data.shape, shape_data.transform, shape_data.color)
    
    def _render_contact_normal_line(self, normal_line):
        """Render a contact normal line"""
        glDisable(GL_LIGHTING)
        glColor3f(*normal_line.color)
        
        glBegin(GL_LINES)
        glVertex3f(*normal_line.start)
        glVertex3f(*normal_line.end)
        glEnd()
        
        # Draw a small sphere at the contact point
        glPushMatrix()
        glTranslatef(*normal_line.start)
        gluSphere(self.quadric, 0.05, 8, 8)
        glPopMatrix()
        
        glEnable(GL_LIGHTING)
    
    def render_shape(self, shape, transform, color=(0.5, 0.5, 0.5)):
        """Render a single shape using a full material definition for better shading."""
        glPushMatrix()
        
        # Apply transformation
        transform_gl = np.zeros((4, 4), dtype=np.float32)
        transform_gl[:] = transform.T
        glMultMatrixf(transform_gl)
        
        # --- Define the full material properties for this object ---
        
        # 1. Diffuse color (the main color of the object under direct light)
        diffuse_color = list(color) + [1.0] # Add alpha channel
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_color)
        
        # 2. Ambient color (the color of the object in shadow, now brighter)
        ambient_color = [c * 0.7 for c in color] + [1.0] # Was 0.4, now 70% of the main color
        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient_color)
        
        # 3. Specular color (the color of the shiny highlight)
        # For most non-metallic objects, this is white.
        specular_color = [1.0, 1.0, 1.0, 1.0]
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular_color)
        
        # 4. Shininess (how sharp and small the highlight is)
        # Higher values = smaller, sharper highlight (more like plastic/metal)
        # Lower values = larger, softer highlight (more matte)
        shininess = 60.0
        glMaterialf(GL_FRONT, GL_SHININESS, shininess)
        
        # We no longer need glColor3f as materials are now fully defined.
        # glColor3f(*color)
        
        # Render based on shape type (the shape IS the collider)
        if isinstance(shape, colliders.Box):
            self._render_box(shape.size)
        elif isinstance(shape, colliders.Sphere):
            self._render_sphere(shape.radius)
        elif isinstance(shape, colliders.Cylinder):
            self._render_cylinder(shape.radius, shape.length)
        elif isinstance(shape, colliders.Capsule):
            self._render_capsule(shape.radius, shape.height)
        elif isinstance(shape, colliders.Cone):
            self._render_cone(shape.radius, shape.height)
        elif isinstance(shape, colliders.Disk):
            self._render_disk(shape.radius)
        elif isinstance(shape, colliders.Ellipsoid):
            self._render_ellipsoid(shape.radii)
        elif isinstance(shape, colliders.Ellipse):
            self._render_ellipse(shape.radii)
        elif isinstance(shape, colliders.MeshGraph):
            self._render_mesh(shape.vertices, shape.triangles)
        elif isinstance(shape, colliders.ConvexHullVertices):
            self._render_convex_hull(shape.vertices)
        
        glPopMatrix()
    
    def _render_box(self, size):
        """Render a box with given size"""
        s = size / 2.0
        
        # Draw box faces
        glBegin(GL_QUADS)
        # Front face
        glNormal3f(0, 0, 1)
        glVertex3f(-s[0], -s[1], s[2])
        glVertex3f(s[0], -s[1], s[2])
        glVertex3f(s[0], s[1], s[2])
        glVertex3f(-s[0], s[1], s[2])
        # Back face
        glNormal3f(0, 0, -1)
        glVertex3f(-s[0], -s[1], -s[2])
        glVertex3f(-s[0], s[1], -s[2])
        glVertex3f(s[0], s[1], -s[2])
        glVertex3f(s[0], -s[1], -s[2])
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(-s[0], -s[1], -s[2])
        glVertex3f(-s[0], -s[1], s[2])
        glVertex3f(-s[0], s[1], s[2])
        glVertex3f(-s[0], s[1], -s[2])
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(s[0], -s[1], -s[2])
        glVertex3f(s[0], s[1], -s[2])
        glVertex3f(s[0], s[1], s[2])
        glVertex3f(s[0], -s[1], s[2])
        # Top face
        glNormal3f(0, 1, 0)
        glVertex3f(-s[0], s[1], -s[2])
        glVertex3f(-s[0], s[1], s[2])
        glVertex3f(s[0], s[1], s[2])
        glVertex3f(s[0], s[1], -s[2])
        # Bottom face
        glNormal3f(0, -1, 0)
        glVertex3f(-s[0], -s[1], -s[2])
        glVertex3f(s[0], -s[1], -s[2])
        glVertex3f(s[0], -s[1], s[2])
        glVertex3f(-s[0], -s[1], s[2])
        glEnd()
    
    def _render_sphere(self, radius):
        """Render a sphere with given radius"""
        gluSphere(self.quadric, radius, 32, 32)
    
    def _render_cylinder(self, radius, length):
        """Render a cylinder with given radius and length"""
        glPushMatrix()
        glTranslatef(0, 0, -length/2)
        gluCylinder(self.quadric, radius, radius, length, 32, 1)
        
        # Render cylinder caps
        gluDisk(self.quadric, 0, radius, 32, 1)
        glTranslatef(0, 0, length)
        gluDisk(self.quadric, 0, radius, 32, 1)
        glPopMatrix()
    
    def _render_capsule(self, radius, height):
        """Render a capsule as cylinder with spherical caps"""
        glPushMatrix()
        glTranslatef(0, 0, -height/2)
        
        # Render cylinder part
        gluCylinder(self.quadric, radius, radius, height, 32, 1)
        
        # Render bottom cap
        gluSphere(self.quadric, radius, 32, 32)
        
        # Render top cap
        glTranslatef(0, 0, height)
        gluSphere(self.quadric, radius, 32, 32)
        glPopMatrix()
    
    def _render_cone(self, radius, height):
        """Render a cone with given base radius and height"""
        glPushMatrix()
        glTranslatef(0, 0, -height/2)
        gluCylinder(self.quadric, radius, 0, height, 32, 1)
        # Base cap
        gluDisk(self.quadric, 0, radius, 32, 1)
        glPopMatrix()
        
    def _render_disk(self, radius):
        """Render a disk (flat circle) with given radius"""
        gluDisk(self.quadric, 0, radius, 32, 1)
        
    def _render_ellipsoid(self, radii):
        """Render an ellipsoid with given radii"""
        glPushMatrix()
        glScalef(radii[0], radii[1], radii[2])
        gluSphere(self.quadric, 1.0, 32, 32)
        glPopMatrix()
        
    def _render_ellipse(self, radii):
        """Render an ellipse (flat) with given radii"""
        # Render as a scaled disk
        glPushMatrix()
        glScalef(radii[0], radii[1], 1.0)
        gluDisk(self.quadric, 0, 1.0, 32, 1)
        glPopMatrix()
        
    def _render_mesh(self, vertices, triangles):
        """Render a mesh given vertices and triangle indices"""
        glBegin(GL_TRIANGLES)
        for triangle in triangles:
            # Calculate normal for the triangle
            v1 = vertices[triangle[0]]
            v2 = vertices[triangle[1]]
            v3 = vertices[triangle[2]]
            
            # Compute normal
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-6: # Check for non-zero length
                normal = normal / norm_len
            
            glNormal3f(*normal)
            glVertex3f(*v1)
            glVertex3f(*v2)
            glVertex3f(*v3)
        glEnd()
        
    def _render_convex_hull(self, vertices):
        """Render a convex hull as a point cloud (simplified)"""
        glPointSize(5.0)
        glBegin(GL_POINTS)
        for vertex in vertices:
            glVertex3f(*vertex)
        glEnd()
    
    def draw_axes(self, length=1.0):
        """Draw X, Y, Z coordinate axes at the origin."""
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        
        # X-axis in Red
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(length, 0.0, 0.0)
        
        # Y-axis in Green
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, length, 0.0)
        
        # Z-axis in Blue
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, length)
        
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def draw_grid(self, size=10, divisions=20):
        """Draw a reference grid on the XZ plane"""
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.3)
        
        glBegin(GL_LINES)
        step = size / divisions
        for i in range(-divisions, divisions + 1):
            # Lines parallel to X axis
            glVertex3f(-size, 0, i * step)
            glVertex3f(size, 0, i * step)
            # Lines parallel to Z axis
            glVertex3f(i * step, 0, -size)
            glVertex3f(i * step, 0, size)
        glEnd()
        
        glEnable(GL_LIGHTING)


