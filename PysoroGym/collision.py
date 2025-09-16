"""
Collision detection module using distance3d library with AABB tree broad phase.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from distance3d import gjk, epa, colliders, containment
from distance3d.aabb_tree import AabbTree


@dataclass
class Contact:
    """Contact information between two shapes"""
    point_a: np.ndarray  # Contact point on shape A
    point_b: np.ndarray  # Contact point on shape B
    normal: np.ndarray   # Contact normal (from A to B)
    penetration: float   # Penetration depth
    

@dataclass
class CollisionPair:
    """Collision information between two bodies"""
    body_a: 'Body'
    body_b: 'Body'
    shape_a: 'Shape'  # Direct shape reference
    shape_b: 'Shape'  # Direct shape reference
    contacts: List[Contact]


class CollisionDetector:
    """Handles collision detection between shapes using AABB tree for broad phase"""
    
    def __init__(self):
        self.debug = False  # Enable for debugging
        
    def detect_collisions(self, bodies: List['Body']) -> List[CollisionPair]:
        """Detect all collisions between bodies"""
        # Collect all shapes with their bodies
        shape_body_pairs = []
        for body in bodies:
            # Update shape poses before collision detection
            body.update_shapes()
            for shape in body.shapes:
                shape_body_pairs.append((shape, body))

        if self.debug:
            print(f"Detecting collisions for {len(shape_body_pairs)} shapes")

        # Broad phase: Get potential collision pairs using AABB tree
        potential_pairs = self._broad_phase_aabb_tree(shape_body_pairs)
            
        if self.debug:
            print(f"Broad phase found {len(potential_pairs)} potential pairs")
            
        # Narrow phase: Check each potential collision pair
        collision_pairs = []
        for ((shape_a, body_a), (shape_b, body_b)) in potential_pairs:
            # Skip if both bodies are static
            if body_a.body_type == 'static' and body_b.body_type == 'static':
                continue
                
            # Narrow phase collision detection
            contacts = self._check_shape_collision(shape_a, shape_b)
            if contacts:
                collision_pairs.append(CollisionPair(
                    body_a=body_a,
                    body_b=body_b,
                    shape_a=shape_a,
                    shape_b=shape_b,
                    contacts=contacts
                ))
                
        return collision_pairs
        
    def _broad_phase_aabb_tree(self, shape_body_pairs: List[Tuple['Shape', 'Body']]) -> Set[Tuple[Tuple['Shape', 'Body'], Tuple['Shape', 'Body']]]:
        """
        Use AABB tree for efficient broad phase collision detection by building
        the tree and querying it for overlaps.
        """
        if len(shape_body_pairs) < 2:
            return set()

        if self.debug:
            print(f"Building AABB tree with {len(shape_body_pairs)} shapes")
            
        # Use simple N^2 comparison for now - we'll fix the AABB tree integration later
        # This simpler approach will help us verify that collision detection works
        potential_pairs = set()
        
        for i in range(len(shape_body_pairs)):
            shape_i, body_i = shape_body_pairs[i]
            aabb_i = self._compute_shape_aabb(shape_i)
            
            if aabb_i is None:
                continue
                
            for j in range(i + 1, len(shape_body_pairs)):
                shape_j, body_j = shape_body_pairs[j]
                
                # Skip if both bodies are static
                if body_i.body_type == 'static' and body_j.body_type == 'static':
                    continue
                    
                aabb_j = self._compute_shape_aabb(shape_j)
                
                if aabb_j is None:
                    continue
                    
                # Check for AABB overlap directly
                if self._aabbs_overlap(aabb_i, aabb_j):
                    if self.debug:
                        print(f"Found overlap between shape {i} and {j}")
                    potential_pairs.add(((shape_i, body_i), (shape_j, body_j)))
                    
        return potential_pairs

    def _aabbs_overlap(self, aabb1, aabb2):
        """Check if two AABBs overlap"""
        # AABB format is now (3, 2) - rows are dimensions, columns are min/max
        for i in range(3):  # For each dimension (x, y, z)
            if aabb1[i, 1] < aabb2[i, 0] or aabb2[i, 1] < aabb1[i, 0]:
                return False
        return True
        
    def _compute_shape_aabb(self, shape) -> Optional[np.ndarray]:
        """Compute AABB for a shape (which already has its pose)"""
        try:
            # Use shape's aabb() method since it inherits from distance3d colliders
            aabb = shape.aabb()
            
            # Add some margin to the AABB to account for movement and numerical issues
            margin = 0.05  # 5cm margin - adjust as needed
            
            # The aabb is already in the correct format (3, 2)
            # Just add margin to min/max values
            aabb[:, 0] -= margin  # mins (first column)
            aabb[:, 1] += margin  # maxs (second column)
            
            return aabb
        
        except Exception as e:
            if self.debug:
                print(f"Failed to compute AABB for shape: {e}")
            return None
            
    def _check_shape_collision(self, shape_a: 'Shape', shape_b: 'Shape') -> List[Contact]:
        """Check collision between two shapes"""
        contacts = []
        
        # Use GJK for narrow phase
        dist, closest_a_gjk, closest_b_gjk, simplex = gjk.gjk(shape_a, shape_b)
        
        if dist == 0.0:  # Collision detected
            try:
                # Use EPA to get penetration info
                mtv, _, _ = epa.epa(simplex, shape_a, shape_b)
                
                # Extract normal and penetration depth from MTV
                penetration = float(np.linalg.norm(mtv))
                normal = mtv / penetration if penetration > 1e-10 else np.array([0, 1, 0])

                #print(f"Collision detected: {shape_a} vs {shape_b}, depth={penetration}, normal={normal}")
            
                # Create contact with EPA results
                contact = Contact(
                    point_a=closest_a_gjk,
                    point_b=closest_b_gjk,
                    normal=normal,
                    penetration=penetration
                )
                contacts.append(contact)
            
            except Exception as e:
                # Fallback if EPA fails
                # Create a simple contact based on GJK results
                direction = closest_b_gjk - closest_a_gjk
                length = np.linalg.norm(direction)
                if length > 1e-10:
                    normal = direction / length
                else:
                    normal = np.array([0, 1, 0])
                
                contact = Contact(
                    point_a=closest_a_gjk,
                    point_b=closest_b_gjk,
                    normal=normal,
                    penetration=0.01  # Small default penetration
                )
                contacts.append(contact)
                
                if self.debug:
                    print(f"EPA failed: {e}, using fallback contact")
        
        return contacts