import numpy as np


class Simplex:
    """Stores simplex of Minkowski differences and support points from both shapes."""
    def __init__(self):
        self.v = np.empty((4, 3))   # Vertices of the Minkowski difference
        self.v1 = np.empty((4, 3))  # Support points from shape 1
        self.v2 = np.empty((4, 3))  # Support points from shape 2
        self.n_points = 0

    def __len__(self):
        return self.n_points

    def add_point(self, v, v1, v2):
        """Add a new point to the simplex."""
        self.v[self.n_points] = v
        self.v1[self.n_points] = v1
        self.v2[self.n_points] = v2
        self.n_points += 1

    def set_points(self, indices):
        """Keep only the points at the given indices."""
        new_v = np.empty((4, 3))
        new_v1 = np.empty((4, 3))
        new_v2 = np.empty((4, 3))
        
        for i, idx in enumerate(indices):
            new_v[i] = self.v[idx]
            new_v1[i] = self.v1[idx]
            new_v2[i] = self.v2[idx]
        
        self.v = new_v
        self.v1 = new_v1
        self.v2 = new_v2
        self.n_points = len(indices)

    def get_point(self, index):
        """Get the Minkowski difference point at index."""
        return self.v[index]

    def get_support_points(self, index):
        """Get the support points from both shapes at index."""
        return self.v1[index], self.v2[index]


def support_function(shape_a, shape_b, direction):
    """Get support point of Minkowski difference A-B in given direction.
    
    Parameters
    ----------
    shape_a : Shape
        First shape
    shape_b : Shape
        Second shape
    direction : array, shape (3,)
        Search direction
        
    Returns
    -------
    v : array, shape (3,)
        Support point in Minkowski difference
    v1 : array, shape (3,)
        Support point on shape A
    v2 : array, shape (3,)
        Support point on shape B
    """
    # Get support points from each shape
    if hasattr(shape_a, 'support'):
        v1 = shape_a.support(direction)
        # Apply body transformation if available
        if hasattr(shape_a, 'body') and shape_a.body is not None:
            v1 = shape_a.body.pos + v1
    elif hasattr(shape_a, 'world_support'):
        v1 = shape_a.world_support(direction)
    else:
        raise AttributeError("Shape A has no support or world_support method")
    
    # IMPORTANT: Use -direction for shape B
    if hasattr(shape_b, 'support'):
        v2 = shape_b.support(-direction)
        # Apply body transformation if available
        if hasattr(shape_b, 'body') and shape_b.body is not None:
            v2 = shape_b.body.pos + v2
    elif hasattr(shape_b, 'world_support'):
        v2 = shape_b.world_support(-direction)
    else:
        raise AttributeError("Shape B has no support or world_support method")
    
    # Minkowski difference
    v = v1 - v2
    
    return v, v1, v2