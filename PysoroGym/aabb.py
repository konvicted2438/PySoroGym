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
        return np.all(self.min_point <= other.max_point) and np.all(other.min_point <= self.max_point)
    
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
            # Find best sibling
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
            self._refit(new_parent)
        
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
            
            # Refit from sibling up
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
            else:
                # Simple approach - use bounding sphere
                if hasattr(shape, 'radius'):
                    radius = shape.radius
                elif hasattr(shape, 'size'):
                    radius = np.linalg.norm(shape.size) * 0.5
                else:
                    radius = 1.0  # Default
                    
                # Expand by sphere
                aabb.expand(pos - radius)
                aabb.expand(pos + radius)
        
        # Add margin
        aabb.min_point -= self.margin
        aabb.max_point += self.margin
        
        return aabb
    
    def _find_best_sibling(self, node):
        """Find the best sibling for a new node using SAH (Surface Area Heuristic)"""
        best = self.root
        best_cost = self._compute_cost(self.root, node)
        
        # Queue for traversal
        queue = [self.root]
        
        while queue:
            current = queue.pop(0)
            
            cost = self._compute_cost(current, node)
            if cost < best_cost:
                best_cost = cost
                best = current
            
            # Don't traverse deeper if cost can't improve
            if not current.is_leaf():
                inherited_cost = self._inherited_cost(current, node)
                
                if inherited_cost + self._lower_bound_cost(current.left) < best_cost:
                    queue.append(current.left)
                    
                if inherited_cost + self._lower_bound_cost(current.right) < best_cost:
                    queue.append(current.right)
        
        return best
    
    def _compute_cost(self, node, new_node):
        """Compute cost of inserting new_node as sibling of node"""
        merged = node.aabb.merge(new_node.aabb)
        return merged.surface_area()
    
    def _inherited_cost(self, node, new_node):
        """Cost inherited from ancestors when inserting"""
        cost = 0.0
        current = node
        
        while current.parent is not None:
            parent_aabb = current.parent.aabb
            new_parent_aabb = parent_aabb.merge(new_node.aabb)
            cost += new_parent_aabb.surface_area() - parent_aabb.surface_area()
            current = current.parent
            
        return cost
    
    def _lower_bound_cost(self, node):
        """Lower bound on cost for a subtree"""
        return node.aabb.surface_area()
    
    def _refit(self, node):
        """Refit AABBs from node up to root"""
        while node is not None:
            if not node.is_leaf():
                node.aabb = node.left.aabb.merge(node.right.aabb)
                node.height = 1 + max(node.left.height, node.right.height)
            node = node.parent
    
    def _query_pairs_recursive(self, node, pairs):
        """Recursively find all overlapping pairs"""
        if node.is_leaf():
            return
            
        # Check if children overlap
        if node.left.aabb.overlaps(node.right.aabb):
            if node.left.is_leaf() and node.right.is_leaf():
                # Both are leaves, add pair
                pairs.append((node.left.body, node.right.body))
            elif node.left.is_leaf():
                # Left is leaf, traverse right
                self._check_leaf_against_tree(node.left, node.right, pairs)
            elif node.right.is_leaf():
                # Right is leaf, traverse left
                self._check_leaf_against_tree(node.right, node.left, pairs)
            else:
                # Both are internal nodes
                self._query_pairs_recursive(node.left, pairs)
                self._query_pairs_recursive(node.right, pairs)
                self._check_tree_against_tree(node.left, node.right, pairs)
        
        # Continue traversal
        if not node.left.is_leaf():
            self._query_pairs_recursive(node.left, pairs)
        if not node.right.is_leaf():
            self._query_pairs_recursive(node.right, pairs)
    
    def _check_leaf_against_tree(self, leaf, tree, pairs):
        """Check a leaf node against all nodes in a subtree"""
        if tree.is_leaf():
            if leaf.aabb.overlaps(tree.aabb) and leaf.body != tree.body:
                pairs.append((leaf.body, tree.body))
        else:
            if leaf.aabb.overlaps(tree.left.aabb):
                self._check_leaf_against_tree(leaf, tree.left, pairs)
            if leaf.aabb.overlaps(tree.right.aabb):
                self._check_leaf_against_tree(leaf, tree.right, pairs)
    
    def _check_tree_against_tree(self, tree1, tree2, pairs):
        """Check all nodes in tree1 against all nodes in tree2"""
        if tree1.aabb.overlaps(tree2.aabb):
            if tree1.is_leaf():
                self._check_leaf_against_tree(tree1, tree2, pairs)
            elif tree2.is_leaf():
                self._check_leaf_against_tree(tree2, tree1, pairs)
            else:
                self._check_tree_against_tree(tree1.left, tree2, pairs)
                self._check_tree_against_tree(tree1.right, tree2, pairs)
    
    def _query_aabb_recursive(self, node, aabb, bodies):
        """Recursively find all bodies whose AABBs overlap with given AABB"""
        if not node.aabb.overlaps(aabb):
            return
            
        if node.is_leaf():
            bodies.append(node.body)
        else:
            self._query_aabb_recursive(node.left, aabb, bodies)
            self._query_aabb_recursive(node.right, aabb, bodies)