import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import queue
import time

class GJKDebugVisualizer:
    """A visualization tool for debugging GJK/EPA algorithms"""
    
    def __init__(self):
        # Data storage
        self.gjk_states = []
        self.epa_states = []
        self.current_state = None
        self.state_index = 0
        
        # Setup figure
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Add controls
        self.fig.text(0.02, 0.98, 'Controls: [Space] Play/Pause | [←/→] Previous/Next | [R] Reset | [Q] Quit', 
                     transform=self.fig.transFigure, fontsize=10, verticalalignment='top')
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Animation state
        self.playing = False
        self.animation_timer = None
        
        # Set up the plot
        self.setup_axes()
        
    def setup_axes(self):
        """Set up the 3D axes"""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('GJK/EPA Debug Visualization')
        
    def add_gjk_state(self, simplex_points, direction, iteration, shape_a_points=None, shape_b_points=None):
        """Add a GJK state to visualize"""
        state = {
            'type': 'gjk',
            'simplex': simplex_points.copy() if simplex_points is not None else [],
            'direction': direction.copy() if direction is not None else np.zeros(3),
            'iteration': iteration,
            'shape_a': shape_a_points,
            'shape_b': shape_b_points
        }
        self.gjk_states.append(state)
        
        # Auto-update if this is the first state
        if len(self.gjk_states) == 1:
            self.current_state = 'gjk'
            self.state_index = 0
            self.update_visualization()
    
    def add_epa_state(self, polytope_points, faces, min_normal, min_distance, iteration):
        """Add an EPA state to visualize"""
        state = {
            'type': 'epa',
            'polytope': polytope_points.copy() if polytope_points is not None else [],
            'faces': faces.copy() if faces is not None else [],
            'normal': min_normal.copy() if min_normal is not None else np.zeros(3),
            'distance': min_distance,
            'iteration': iteration
        }
        self.epa_states.append(state)
        
        # Switch to EPA visualization when it starts
        if len(self.epa_states) == 1:
            self.current_state = 'epa'
            self.state_index = 0
            self.update_visualization()
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == ' ':
            self.toggle_playback()
        elif event.key == 'left':
            self.previous_state()
        elif event.key == 'right':
            self.next_state()
        elif event.key == 'r':
            self.reset()
        elif event.key == 'q':
            plt.close(self.fig)
    
    def toggle_playback(self):
        """Toggle animation playback"""
        self.playing = not self.playing
        if self.playing:
            self.animate()
    
    def animate(self):
        """Animate through states"""
        if self.playing:
            self.next_state()
            if self.playing:  # Check if still playing after state change
                self.animation_timer = self.fig.canvas.new_timer(interval=500)
                self.animation_timer.single_shot = True
                self.animation_timer.add_callback(self.animate)
                self.animation_timer.start()
    
    def previous_state(self):
        """Go to previous state"""
        if self.current_state == 'gjk' and self.state_index > 0:
            self.state_index -= 1
            self.update_visualization()
        elif self.current_state == 'epa' and self.state_index > 0:
            self.state_index -= 1
            self.update_visualization()
        elif self.current_state == 'epa' and self.state_index == 0 and len(self.gjk_states) > 0:
            # Switch back to GJK
            self.current_state = 'gjk'
            self.state_index = len(self.gjk_states) - 1
            self.update_visualization()
    
    def next_state(self):
        """Go to next state"""
        if self.current_state == 'gjk':
            if self.state_index < len(self.gjk_states) - 1:
                self.state_index += 1
                self.update_visualization()
            elif len(self.epa_states) > 0:
                # Switch to EPA
                self.current_state = 'epa'
                self.state_index = 0
                self.update_visualization()
            else:
                self.playing = False
        elif self.current_state == 'epa':
            if self.state_index < len(self.epa_states) - 1:
                self.state_index += 1
                self.update_visualization()
            else:
                self.playing = False
    
    def reset(self):
        """Reset to first state"""
        self.playing = False
        if len(self.gjk_states) > 0:
            self.current_state = 'gjk'
            self.state_index = 0
        elif len(self.epa_states) > 0:
            self.current_state = 'epa'
            self.state_index = 0
        self.update_visualization()
    
    def update_visualization(self):
        """Update the current visualization"""
        self.ax.clear()
        self.setup_axes()
        
        if self.current_state == 'gjk' and self.state_index < len(self.gjk_states):
            self._visualize_gjk(self.gjk_states[self.state_index])
        elif self.current_state == 'epa' and self.state_index < len(self.epa_states):
            self._visualize_epa(self.epa_states[self.state_index])
        
        # Draw the updated figure
        plt.draw()
        plt.pause(0.001)
    
    def _visualize_gjk(self, data):
        """Visualize GJK state"""
        self.ax.set_title(f'GJK Iteration {data["iteration"]}')
        
        # Draw origin
        self.ax.scatter([0], [0], [0], c='black', s=100, marker='o', label='Origin')
        
        # Draw simplex points
        if len(data['simplex']) > 0:
            simplex = np.array(data['simplex'])
            self.ax.scatter(simplex[:, 0], simplex[:, 1], simplex[:, 2], 
                          c='red', s=50, marker='o', label='Simplex')
            
            # Draw simplex edges
            if len(simplex) >= 2:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        self.ax.plot([simplex[i,0], simplex[j,0]], 
                                   [simplex[i,1], simplex[j,1]], 
                                   [simplex[i,2], simplex[j,2]], 'r-', alpha=0.6)
            
            # Draw simplex faces (for triangles)
            if len(simplex) == 3:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                face = Poly3DCollection([simplex], alpha=0.3, facecolor='red', edgecolor='red')
                self.ax.add_collection3d(face)
            
            # Draw tetrahedron faces
            elif len(simplex) == 4:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                faces = [
                    [simplex[0], simplex[1], simplex[2]],
                    [simplex[0], simplex[1], simplex[3]],
                    [simplex[0], simplex[2], simplex[3]],
                    [simplex[1], simplex[2], simplex[3]]
                ]
                face_collection = Poly3DCollection(faces, alpha=0.2, facecolor='red', edgecolor='red')
                self.ax.add_collection3d(face_collection)
        
        # Draw search direction
        direction = data['direction']
        if np.linalg.norm(direction) > 0:
            self.ax.quiver(0, 0, 0, direction[0], direction[1], direction[2], 
                         color='blue', arrow_length_ratio=0.1, label='Search Direction')
        
        # Draw shapes if provided
        if data['shape_a'] is not None:
            self._draw_shape(data['shape_a'], 'green', 'Shape A')
        if data['shape_b'] is not None:
            self._draw_shape(data['shape_b'], 'orange', 'Shape B')
        
        self.ax.legend()
        self._set_equal_aspect()
    
    def _visualize_epa(self, data):
        """Visualize EPA state"""
        self.ax.set_title(f'EPA Iteration {data["iteration"]} - Min Distance: {data["distance"]:.4f}')
        
        # Draw origin
        self.ax.scatter([0], [0], [0], c='black', s=100, marker='o', label='Origin')
        
        # Draw polytope
        if len(data['polytope']) > 0 and len(data['faces']) > 0:
            polytope = np.array(data['polytope'])
            
            # Draw vertices
            self.ax.scatter(polytope[:, 0], polytope[:, 1], polytope[:, 2], 
                          c='blue', s=30, marker='o', label='Polytope Vertices')
            
            # Draw faces
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            face_verts = []
            for i in range(0, len(data['faces']), 3):
                if i+2 < len(data['faces']):
                    face = [polytope[data['faces'][i]], 
                           polytope[data['faces'][i+1]], 
                           polytope[data['faces'][i+2]]]
                    face_verts.append(face)
            
            if face_verts:
                face_collection = Poly3DCollection(face_verts, alpha=0.3, 
                                                 facecolor='cyan', edgecolor='blue')
                self.ax.add_collection3d(face_collection)
        
        # Draw minimum normal
        normal = data['normal']
        if np.linalg.norm(normal) > 0:
            self.ax.quiver(0, 0, 0, normal[0], normal[1], normal[2], 
                         color='red', arrow_length_ratio=0.1, label='Min Normal')
        
        self.ax.legend()
        self._set_equal_aspect()
    
    def _draw_shape(self, points, color, label):
        """Draw a shape from its vertices"""
        if points is not None and len(points) > 0:
            points = np.array(points)
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=color, s=20, marker='o', alpha=0.5, label=label)
    
    def _set_equal_aspect(self):
        """Set equal aspect ratio for 3D plot"""
        # Get current limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        
        # Find the largest range
        max_range = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) / 2.0
        
        # Set new limits
        mid_x = (xlim[1] + xlim[0]) / 2.0
        mid_y = (ylim[1] + ylim[0]) / 2.0
        mid_z = (zlim[1] + zlim[0]) / 2.0
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    def close(self):
        """Close the visualizer"""
        plt.close(self.fig)

# Global visualizer instance
_debug_visualizer = None

def get_debug_visualizer():
    """Get or create the global debug visualizer"""
    global _debug_visualizer
    if _debug_visualizer is None:
        _debug_visualizer = GJKDebugVisualizer()
    return _debug_visualizer