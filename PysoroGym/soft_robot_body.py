import numpy as np
from PysoroGym.Body import Body
from PysoroGym.shapes import MeshGraph
from PysoroGym.kinematic.soft_robot_kinematics import SoftRobotKinematics
from PysoroGym.materials import Material


class SoftRobotBody(Body):
    """A soft robot body that can contain multiple segments"""
    
    def __init__(self, position=(0, 0, 0), orientation=None, n_segments_per_robot=10, 
                 n_sides=8, material=None, scale=1.0, fixed_base=True, 
                 segment_configs=None, **kwargs):
        """
        Initialize a soft robot body with potentially multiple segments
        
        Parameters
        ----------
        position : tuple
            Initial position of the robot base
        orientation : array_like or None
            Initial orientation as rotation matrix (3x3)
        n_segments_per_robot : int
            Number of mesh segments per robot segment (for visualization resolution)
        n_sides : int
            Number of sides for the cylindrical mesh
        material : Material
            Material properties for the robot
        scale : float
            Scale factor to make the robot more visible
        fixed_base : bool
            If True, the base is fixed (kinematic)
        segment_configs : list of dict or None
            Configuration for each segment. If None, creates a single segment.
            Each dict can contain: {'L': length, 'rad_robot': radius, 'pressures': [p1,p2,p3]}
        """
        body_type = 'kinematic' if fixed_base else 'dynamic'
        super().__init__(body_type=body_type, position=position, **kwargs)
        
        # Initialize list of kinematics solvers (one per segment)
        self.segments = []
        
        # Default single segment if no configs provided
        if segment_configs is None:
            segment_configs = [{}]  # Single segment with default parameters
            
        # Create kinematics solver for each segment
        for config in segment_configs:
            kinematic = SoftRobotKinematics()
            # Override parameters if provided
            if 'L' in config:
                kinematic.L = config['L']
            if 'rad_robot' in config:
                kinematic.rad_robot = config['rad_robot']
            if 'rad_chamber' in config:
                kinematic.rad_chamber = config['rad_chamber']
            # Add more parameter overrides as needed
            
            self.segments.append({
                'kinematics': kinematic,
                'pressures': config.get('pressures', np.zeros(3)),
                'external_tip_force': config.get('tip_force', np.zeros(3))
            })
        
        # Visualization parameters
        self.scale = scale
        self.n_segments_per_robot = n_segments_per_robot
        self.n_sides = n_sides
        
        # Store material
        if material is None:
            material = Material(friction=0.8, elasticity=0.2)
        self.material = material
        
        # Base configuration
        self.fixed_base = fixed_base
        
        # Set initial orientation
        if orientation is not None:
            # Convert rotation matrix to quaternion if needed
            if isinstance(orientation, np.ndarray) and orientation.shape == (3, 3):
                # Convert rotation matrix to quaternion
                self.orientation = self.matrix_to_quaternion(orientation)
            else:
                # Assume it's already a quaternion
                self.orientation = orientation
        else:
            # Default to identity quaternion [w, x, y, z]
            self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Initialize mesh
        self._initialize_mesh()
        
        # Store backbone points for each segment
        self.segment_backbones = []
        
        # Store the last valid kinematic solution for stability checks
        self.last_valid_segment_solutions = None
        # Calculate a reasonable threshold for tip movement per frame
        total_length = sum(seg['kinematics'].L for seg in self.segments) * self.scale
        self.fly_away_threshold = total_length *0.5   # e.g., half the robot's length

        print(f"Soft robot initialized with {len(self.segments)} segments at position {position}")
        
    def _initialize_mesh(self):
        """Create initial mesh geometry for all segments"""
        # Solve the serial chain
        segment_solutions = self._solve_serial_chain()

        # If the solution failed (e.g., due to a warning), we cannot initialize the mesh.
        if not segment_solutions:
            # This is a critical failure during initialization.
            # The parameters likely cause numerical instability from the start.
            raise RuntimeError(
                "Kinematic chain failed to solve during initialization. "
                "This is often due to a numerical instability (e.g., overflow) "
                "in the forward kinematics calculation. Check initial segment parameters "
                "like length, radius, or initial pressures."
            )

        # Store the initial solution as the first valid one
        self.last_valid_segment_solutions = segment_solutions

        # Generate combined mesh
        vertices, triangles = self._create_multi_segment_mesh(segment_solutions)

        # Create the mesh shape
        self.mesh_shape = MeshGraph(vertices, triangles)
        self.add_shape(self.mesh_shape, material=self.material)
        
    def _solve_serial_chain(self):
        """
        Solve all segments in the chain from tip to base.
        Returns an empty list if any segment's kinematics fail.
        """
        n_segments = len(self.segments)
        if n_segments == 0:
            return []

        # This list will hold the successful kinematic results in reverse order (tip-to-base)
        tip_to_base_results = []

        # Solve backwards from tip to base
        for i in range(n_segments - 1, -1, -1):
            segment = self.segments[i]
            
            # Determine forces and moments acting on the tip of the current segment
            if i == n_segments - 1:
                # Tip segment: only has its own external force
                segment['kinematics'].F = segment['external_tip_force']
                segment['kinematics'].M = np.zeros(3)
            else:
                # Non-tip segment: forces are reactions from the previously solved child segment
                # The 'child' is the result from the last iteration, which is the last item in our list
                child_segment_dict, child_y_all, _, _ = tip_to_base_results[-1]
                
                # Reaction forces at the base of the child segment
                n_base_child = child_y_all[0, 12:15]
                m_base_child = child_y_all[0, 15:18]
                
                # Calculate child weight
                child_kin = child_segment_dict['kinematics']
                child_weight = child_kin.rho * child_kin.A_cross_section * child_kin.L * child_kin.g
                
                # Set forces for this segment's tip
                #segment['kinematics'].F = segment['external_tip_force'] - n_base_child + child_weight
                segment['kinematics'].F =  child_weight
                #segment['kinematics'].M = -m_base_child
            
            # Solve this segment's kinematics, catching any warnings
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                kin_results = segment['kinematics'].forward_kinematics(segment['pressures'])
                
                # If there was ANY warning or the result is None, the calculation is invalid.
                if w or kin_results is None:
                    # We cannot continue the chain, so return an empty list to signal failure.
                    return []
            
            _, _, y_all = kin_results
            tip_to_base_results.append((segment, y_all, None, None))

        # If we successfully solved all segments, reverse the list to get base-to-tip order
        base_to_tip_results = tip_to_base_results[::-1]
        
        # Now propagate the absolute positions and orientations from the base to the tip
        local_base_pos = np.array([0.0, 0.0, 0.0])
        local_base_rot = np.eye(3)
        
        final_results = []
        for i, (segment, y_all, _, _) in enumerate(base_to_tip_results):
            final_results.append((segment, y_all, local_base_pos, local_base_rot))
            
            # The tip of the current segment becomes the base for the next one
            tip_pos_local = y_all[-1, :3] * self.scale
            tip_rot_local = y_all[-1, 3:12].reshape(3, 3)
            
            local_base_pos = local_base_pos + local_base_rot @ tip_pos_local
            local_base_rot = local_base_rot @ tip_rot_local

        return final_results
    
    def _create_multi_segment_mesh(self, segment_solutions):
        """Create a combined mesh for all segments"""
        all_vertices = []
        all_triangles = []
        vertex_offset = 0
        
        # Store backbone points for visualization
        self.segment_backbones = []
        
        for segment, y_all, base_pos, base_rot in segment_solutions:
            # Get segment parameters
            kinematic = segment['kinematics']
            robot_radius = kinematic.rad_robot
            if robot_radius > 1:  # Assume mm
                robot_radius = robot_radius / 1000.0
            robot_radius *= self.scale
            
            n_points = len(y_all)
            
            # Create vertices for this segment
            segment_vertices = []
            backbone_points = []
            
            for i in range(n_points):
                # Get local position and orientation
                local_pos = y_all[i, :3] * self.scale
                local_R = y_all[i, 3:12].reshape(3, 3)
                
                # Transform to global
                global_pos = base_pos + base_rot @ local_pos
                global_R = base_rot @ local_R
                
                # Store backbone point
                backbone_points.append(global_pos)
                
                # if i == n_points - 1:
                #     print(f"  Tip position (global): {global_pos}")

                # Create ring of vertices
                for j in range(self.n_sides):
                    angle = 2 * np.pi * j / self.n_sides
                    
                    local_point = np.array([
                        robot_radius * np.cos(angle),
                        robot_radius * np.sin(angle),
                        0
                    ])
                    
                    global_point = global_pos + global_R @ local_point
                    segment_vertices.append(global_point)
            
            # Add end cap centers
            segment_vertices.append(backbone_points[0])   # Base center
            segment_vertices.append(backbone_points[-1])  # Tip center
            
            # Store backbone for this segment
            self.segment_backbones.append(np.array(backbone_points))
            
            # Create triangles for this segment
            segment_triangles = []
            
            # Side triangles
            for i in range(n_points - 1):
                for j in range(self.n_sides):
                    j_next = (j + 1) % self.n_sides
                    
                    idx1 = vertex_offset + i * self.n_sides + j
                    idx2 = vertex_offset + i * self.n_sides + j_next
                    idx3 = vertex_offset + (i + 1) * self.n_sides + j
                    idx4 = vertex_offset + (i + 1) * self.n_sides + j_next
                    
                    segment_triangles.append([idx1, idx3, idx2])
                    segment_triangles.append([idx2, idx3, idx4])
            
            # End cap triangles
            bottom_center_idx = vertex_offset + n_points * self.n_sides
            top_center_idx = vertex_offset + n_points * self.n_sides + 1
            
            for j in range(self.n_sides):
                j_next = (j + 1) % self.n_sides
                segment_triangles.append([bottom_center_idx, 
                                        vertex_offset + j, 
                                        vertex_offset + j_next])
            
            top_ring_start = vertex_offset + (n_points - 1) * self.n_sides
            for j in range(self.n_sides):
                j_next = (j + 1) % self.n_sides
                segment_triangles.append([top_center_idx, 
                                        top_ring_start + j_next, 
                                        top_ring_start + j])
            
            # Add to combined mesh
            all_vertices.extend(segment_vertices)
            all_triangles.extend(segment_triangles)
            
            # Update offset for next segment
            vertex_offset += len(segment_vertices)
        
        return np.array(all_vertices, dtype=np.float64), np.array(all_triangles, dtype=np.int64)
    
    def set_segment_pressures(self, segment_index, pressures):
        """Set pressures for a specific segment"""
        if 0 <= segment_index < len(self.segments):
            self.segments[segment_index]['pressures'] = np.array(pressures)
            print(f"Set pressures for segment {segment_index}: {pressures}")
            self.update_mesh()
        else:
            raise ValueError(f"Invalid segment index {segment_index}")
    
    def set_all_pressures(self, pressure_list):
        """Set pressures for all segments at once"""
        if len(pressure_list) != len(self.segments):
            raise ValueError(f"Expected {len(self.segments)} pressure arrays, got {len(pressure_list)}")
        
        for i, pressures in enumerate(pressure_list):
            self.segments[i]['pressures'] = np.array(pressures)
            #print(f"Set pressures for segment {i}: {pressures}")
        
        self.update_mesh()
    
    def get_all_pressures(self):
        """Get current pressures for all segments"""
        return [segment['pressures'].copy() for segment in self.segments]
    
    def get_segment_pressures(self, segment_index):
        """Get pressures for a specific segment"""
        if 0 <= segment_index < len(self.segments):
            return self.segments[segment_index]['pressures'].copy()
        else:
            raise ValueError(f"Invalid segment index {segment_index}")
    
    def get_segment_tip_position(self, segment_index):
        """Get the world position of the tip of a specific segment"""
        if not (0 <= segment_index < len(self.segments)):
            raise IndexError("Segment index out of range")
        
        # For intermediate segments, return the position where it connects to the next segment
        # This is stored in the segment's transform matrix
        segment = self.segments[segment_index]
        
        # The transform matrix stores the end position of this segment
        # Extract position from the 4x4 transform matrix
        if 'transform' in segment and hasattr(segment['transform'], 'shape'):
            # Position is in the last column of the 4x4 matrix
            position = segment['transform'][:3, 3]
            return np.array(position).flatten()
        else:
            # Fallback: estimate based on segment length
            base_pos = self.position if segment_index == 0 else self.segments[segment_index-1]['transform'][:3, 3]
            # Approximate position assuming vertical stacking
            offset = np.array([0, 0.1 * (segment_index + 1), 0])
            return np.array(base_pos + offset).flatten()
    
    def set_segment_tip_force(self, segment_index, force):
        """Set external tip force for a specific segment"""
        if 0 <= segment_index < len(self.segments):
            self.segments[segment_index]['external_tip_force'] = np.array(force)
        else:
            raise ValueError(f"Invalid segment index {segment_index}")
    
    def update_mesh(self):
        """Update mesh based on current kinematics solution"""
        # Store original position
        original_position = self.position.copy()
        
        # Solve the serial chain
        new_segment_solutions = self._solve_serial_chain()
        
        # If empty (due to a warning), skip update
        if not new_segment_solutions:
            return

        # --- STABILITY CHECK ---
        # Compare the new solution with the last known good one
        if self.last_valid_segment_solutions:
            # Get the tip position from the last valid solution
            last_tip_pos = self._get_tip_from_solution(self.last_valid_segment_solutions)
            
            # Get the tip position from the new solution
            new_tip_pos = self._get_tip_from_solution(new_segment_solutions)
            
            # If the tip moved too far in one step, it's unstable. Ignore it.
            if np.linalg.norm(new_tip_pos - last_tip_pos) > self.fly_away_threshold:
                # print("Fly-away detected, skipping frame.") # Optional: for debugging
                return

        # If the check passes, this is now the last valid solution
        self.last_valid_segment_solutions = new_segment_solutions
        
        # Generate new vertices (keep same triangles)
        vertices, triangles = self._create_multi_segment_mesh(new_segment_solutions)
        
        # Update the mesh shape's vertices directly
        self.mesh_shape.vertices = vertices - self.mesh_shape._center_of_mass
        self.mesh_shape._support_function.vertices = self.mesh_shape.vertices
        
        # IMPORTANT: Reset position to ensure the body stays fixed
        self.position = original_position
        
        # Update the shape in the body so collision detection gets fresh bounds
        self.update_shapes()
    
    def get_tip_position(self):
        """Get the tip position of the last segment"""
        if self.segment_backbones and len(self.segment_backbones) > 0:
            return self.segment_backbones[-1][-1]  # Last point of last segment
        return self.position

    def _get_tip_from_solution(self, segment_solution):
        """Helper to calculate the final tip position from a solution list."""
        # The last segment in the solution list is the final tip segment
        _, y_all, base_pos, base_rot = segment_solution[-1]
        
        # The tip of this segment is the last point in its y_all array
        tip_pos_local = y_all[-1, :3] * self.scale
        
        # Transform to world coordinates
        tip_pos_global = base_pos + base_rot @ tip_pos_local
        return tip_pos_global
    
    def get_segment_tip_position(self, segment_index):
        """Get tip position of a specific segment"""
        if 0 <= segment_index < len(self.segment_backbones):
            return self.segment_backbones[segment_index][-1]
        return None
    
    def quaternion_to_matrix(self, quaternion=None):
        """Convert a quaternion [w, x, y, z] to a rotation matrix (3x3)"""
        if quaternion is None:
            quaternion = self.orientation
        
        # Normalize the quaternion
        q = quaternion / np.linalg.norm(quaternion)
        
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Convert to rotation matrix
        matrix = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        
        return matrix
    
    def matrix_to_quaternion(self, matrix):
        """Convert rotation matrix to quaternion [w, x, y, z]"""
        # Based on method from Shepperd (1978)
        trace = np.trace(matrix)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (matrix[2, 1] - matrix[1, 2]) * s
            y = (matrix[0, 2] - matrix[2, 0]) * s
            z = (matrix[1, 0] - matrix[0, 1]) * s
        else:
            if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
                w = (matrix[2, 1] - matrix[1, 2]) / s
                x = 0.25 * s
                y = (matrix[0, 1] + matrix[1, 0]) / s
                z = (matrix[0, 2] + matrix[2, 0]) / s
            elif matrix[1, 1] > matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
                w = (matrix[0, 2] - matrix[2, 0]) / s
                x = (matrix[0, 1] + matrix[1, 0]) / s
                y = 0.25 * s
                z = (matrix[1, 2] + matrix[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
                w = (matrix[1, 0] - matrix[0, 1]) / s
                x = (matrix[0, 2] + matrix[2, 0]) / s
                y = (matrix[1, 2] + matrix[2, 1]) / s
                z = 0.25 * s
                
        return np.array([w, x, y, z])