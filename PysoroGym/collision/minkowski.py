import numpy as np
from ..math_utils import q_to_mat3


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


def support_function(shape_a, shape_b, d_world):
    """
    World-space support of the Minkowski difference + the two original points.
    """
    # Body rotations (local → world)
    Ra = q_to_mat3(shape_a.body.q)
    Rb = q_to_mat3(shape_b.body.q)

    # --- A -------------------------------------------------------------
    d_a_local = Ra.T @ d_world                 # world → local
    p1_local  = shape_a.shape.support(d_a_local)
    p1_world  = Ra @ p1_local + shape_a.body.pos

    # --- B -------------------------------------------------------------
    d_b_local = Rb.T @ -d_world
    p2_local  = shape_b.shape.support(d_b_local)
    p2_world  = Rb @ p2_local + shape_b.body.pos

    v = p1_world - p2_world                    # Minkowski point
    return v, p1_world, p2_world