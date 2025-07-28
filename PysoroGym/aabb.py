import numpy as np

class AABB:
    """Axis-Aligned Bounding Box"""
    def __init__(self, min_point=None, max_point=None):
        if min_point is not None and max_point is not None:
            self.min_point = np.array(min_point, dtype=float)
            self.max_point = np.array(max_point, dtype=float)
        else:
            self.min_point = np.array([float('inf')] * 3)
            self.max_point = np.array([float('-inf')] * 3)
    
    def expand(self, point):
        """Expand AABB to include a point"""
        self.min_point = np.minimum(self.min_point, point)
        self.max_point = np.maximum(self.max_point, point)
    
    def expand_by_aabb(self, other):
        """Expand this AABB to include another AABB"""
        self.min_point = np.minimum(self.min_point, other.min_point)
        self.max_point = np.maximum(self.max_point, other.max_point)
    
    def overlaps(self, other):
        """Check if this AABB overlaps with another"""
        # Fixed: ensure we check all dimensions properly
        return (self.min_point[0] <= other.max_point[0] and self.max_point[0] >= other.min_point[0] and
                self.min_point[1] <= other.max_point[1] and self.max_point[1] >= other.min_point[1] and
                self.min_point[2] <= other.max_point[2] and self.max_point[2] >= other.min_point[2])
    
    def contains(self, point):
        """Check if point is inside AABB"""
        return np.all(point >= self.min_point) and np.all(point <= self.max_point)
    
    def center(self):
        """Get center of AABB"""
        return (self.min_point + self.max_point) * 0.5
    
    def half_extents(self):
        """Get half extents of AABB"""
        return (self.max_point - self.min_point) * 0.5
    
    def surface_area(self):
        """Calculate surface area of AABB"""
        dims = self.max_point - self.min_point
        return 2.0 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[2] * dims[0])
    
    def volume(self):
        """Calculate volume of AABB"""
        dims = self.max_point - self.min_point
        return dims[0] * dims[1] * dims[2]
    
    def merge(self, other):
        """Create a new AABB that contains both AABBs"""
        new_aabb = AABB()
        new_aabb.min_point = np.minimum(self.min_point, other.min_point)
        new_aabb.max_point = np.maximum(self.max_point, other.max_point)
        return new_aabb
    
    def __str__(self):
        return f"AABB(min={self.min_point}, max={self.max_point})"


class AABBNode:
    """Node in the AABB tree"""
    def __init__(self):
        self.aabb = AABB()
        self.parent = None
        self.left = None
        self.right = None
        self.height = 0
        self.body = None  # Leaf nodes store body reference
        
    def is_leaf(self):
        return self.left is None


class AABBTree:
    """Dynamic AABB tree for broad phase collision detection"""
    def __init__(self, margin=0.1):
        self.root = None
        self.margin = margin  # Margin to add to AABBs for stability
        self.node_map = {}  # Map from body to node for fast updates
        
    def insert(self, body):
        """Insert a body into the tree"""
        # Create AABB for body with margin
        aabb = self._compute_body_aabb(body)
        
        # Create new leaf node
        node = AABBNode()
        node.aabb = aabb
        node.body = body
        
        # Store mapping
        self.node_map[body] = node
        
        # Insert into tree
        if self.root is None:
            self.root = node
        else:
            # Find best sibling using volume heuristic (similar to distance3d)
            sibling = self._find_best_sibling(node)
            
            # Create new parent
            old_parent = sibling.parent
            new_parent = AABBNode()
            new_parent.parent = old_parent
            new_parent.aabb = sibling.aabb.merge(node.aabb)
            
            if old_parent is not None:
                # The sibling was not the root
                if old_parent.left == sibling:
                    old_parent.left = new_parent
                else:
                    old_parent.right = new_parent
                    
                new_parent.left = sibling
                new_parent.right = node
                sibling.parent = new_parent
                node.parent = new_parent
            else:
                # The sibling was the root
                new_parent.left = sibling
                new_parent.right = node
                sibling.parent = new_parent
                node.parent = new_parent
                self.root = new_parent
            
            # Walk back up and refit AABBs
            self._refit(new_parent.parent)
        
        return node
    
    def remove(self, body):
        """Remove a body from the tree"""
        if body not in self.node_map:
            return
            
        node = self.node_map[body]
        del self.node_map[body]
        
        if node == self.root:
            self.root = None
            return
            
        parent = node.parent
        sibling = parent.left if parent.right == node else parent.right
        
        if parent.parent is not None:
            # Connect sibling to grandparent
            if parent.parent.left == parent:
                parent.parent.left = sibling
            else:
                parent.parent.right = sibling
            sibling.parent = parent.parent
            
            # Refit from sibling's parent up
            self._refit(sibling.parent)
        else:
            # Parent was root
            self.root = sibling
            sibling.parent = None
    
    def update(self, body):
        """Update a body's position in the tree"""
        if body not in self.node_map:
            return
            
        node = self.node_map[body]
        new_aabb = self._compute_body_aabb(body)
        
        # If the new AABB is still within the fat AABB, no need to update tree
        if node.aabb.contains(new_aabb.min_point) and node.aabb.contains(new_aabb.max_point):
            return
            
        # Remove and reinsert
        self.remove(body)
        self.insert(body)
    
    def query_pairs(self):
        """Get all potentially colliding pairs"""
        pairs = []
        if self.root is not None:
            self._query_pairs_recursive(self.root, pairs)
        return pairs
    
    def query_aabb(self, aabb):
        """Query all bodies whose AABBs overlap with given AABB"""
        bodies = []
        if self.root is not None:
            self._query_aabb_recursive(self.root, aabb, bodies)
        return bodies
    
    def _compute_body_aabb(self, body):
        """Compute AABB for a body including all its shapes"""
        aabb = AABB()
        
        # Get body transform
        pos = body.position
        rot = body.rotation
        
        # Compute AABB for each shape
        for shape_collider in body.shapes:
            shape = shape_collider.shape
            
            if hasattr(shape, 'get_aabb'):
                # Shape can compute its own AABB
                shape_aabb = shape.get_aabb(pos, rot)
                aabb.expand_by_aabb(shape_aabb)
            elif hasattr(shape, 'aabb_min') and hasattr(shape, 'aabb_max'):
                # Shape has precomputed local AABB
                # Transform local AABB to world space
                local_min = shape.aabb_min
                local_max = shape.aabb_max
                
                # Get all 8 corners of local AABB
                corners = []
                for x in [local_min[0], local_max[0]]:
                    for y in [local_min[1], local_max[1]]:
                        for z in [local_min[2], local_max[2]]:
                            corner = np.array([x, y, z])
                            # Transform to world space
                            world_corner = body.transform_point(corner)
                            corners.append(world_corner)
                
                # Expand AABB to include all corners
                for corner in corners:
                    aabb.expand(corner)
            else:
                # Fallback - use bounding sphere
                if hasattr(shape, 'radius'):
                    radius = shape.radius
                elif hasattr(shape, 'size'):
                    radius = np.linalg.norm(shape.size) * 0.5
                else:
                    radius = 1.0  # Default
                    
                # Expand by sphere
                aabb.expand(pos - radius)
                aabb.expand(pos + radius)
        
        # Add margin for stability
        aabb.min_point -= self.margin
        aabb.max_point += self.margin
        
        return aabb
    
    def _find_best_sibling(self, node):
        """Find the best sibling for a new node using volume heuristic"""
        # Start with root
        best = self.root
        
        # Use a simple traversal (can be optimized with SAH later)
        stack = [self.root]
        best_cost = float('inf')
        
        while stack:
            current = stack.pop()
            
            # Cost of creating a new parent for current and node
            merged_aabb = current.aabb.merge(node.aabb)
            cost = merged_aabb.volume()
            
            if cost < best_cost:
                best_cost = cost
                best = current
            
            # Don't traverse into leaves
            if not current.is_leaf():
                stack.append(current.left)
                stack.append(current.right)
        
        return best
    
    def _refit(self, node):
        """Refit AABBs from node up to root"""
        while node is not None:
            if not node.is_leaf():
                # Recompute AABB from children
                node.aabb = node.left.aabb.merge(node.right.aabb)
                node.height = 1 + max(node.left.height, node.right.height)
            node = node.parent
    
    def _query_pairs_recursive(self, node, pairs):
        """Recursively find all overlapping pairs"""
        if node.is_leaf():
            return
        
        # Recurse into children
        if node.left is not None:
            self._query_pairs_recursive(node.left, pairs)
        if node.right is not None:
            self._query_pairs_recursive(node.right, pairs)
        
        # Check pairs between left and right subtrees
        if node.left is not None and node.right is not None:
            self._check_subtree_pairs(node.left, node.right, pairs)
    
    def _check_subtree_pairs(self, node1, node2, pairs):
        """Check for overlapping pairs between two subtrees"""
        # Check if AABBs overlap
        if not node1.aabb.overlaps(node2.aabb):
            return
        
        # If both are leaves, add pair
        if node1.is_leaf() and node2.is_leaf():
            if node1.body != node2.body:  # Don't self-collide
                pairs.append((node1.body, node2.body))
        elif node1.is_leaf():
            # node1 is leaf, recurse into node2
            if node2.left is not None:
                self._check_subtree_pairs(node1, node2.left, pairs)
            if node2.right is not None:
                self._check_subtree_pairs(node1, node2.right, pairs)
        elif node2.is_leaf():
            # node2 is leaf, recurse into node1
            if node1.left is not None:
                self._check_subtree_pairs(node1.left, node2, pairs)
            if node1.right is not None:
                self._check_subtree_pairs(node1.right, node2, pairs)
        else:
            # Both are branches, recurse into all combinations
            if node1.left is not None and node2.left is not None:
                self._check_subtree_pairs(node1.left, node2.left, pairs)
            if node1.left is not None and node2.right is not None:
                self._check_subtree_pairs(node1.left, node2.right, pairs)
            if node1.right is not None and node2.left is not None:
                self._check_subtree_pairs(node1.right, node2.left, pairs)
            if node1.right is not None and node2.right is not None:
                self._check_subtree_pairs(node1.right, node2.right, pairs)
    
    def _query_aabb_recursive(self, node, aabb, bodies):
        """Recursively find all bodies whose AABBs overlap with given AABB"""
        if not node.aabb.overlaps(aabb):
            return
            
        if node.is_leaf():
            bodies.append(node.body)
        else:
            if node.left is not None:
                self._query_aabb_recursive(node.left, aabb, bodies)
            if node.right is not None:
                self._query_aabb_recursive(node.right, aabb, bodies)