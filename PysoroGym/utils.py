import numpy as np

def rotation_matrix_from_axis_angle(axis, angle):
    """
    Create a 3x3 rotation matrix from an axis and angle using Rodrigues' formula.
    
    Parameters
    ----------
    axis : array_like, shape (3,)
        The rotation axis (will be normalized)
    angle : float
        The rotation angle in radians
        
    Returns
    -------
    R : ndarray, shape (3, 3)
        The rotation matrix
    """
    # Normalize the axis
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    
    # Components of the axis
    x, y, z = axis
    
    # Precompute some values
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    
    # Rodrigues' rotation formula
    R = np.array([
        [c + x*x*t,     x*y*t - z*s,  x*z*t + y*s],
        [y*x*t + z*s,   c + y*y*t,    y*z*t - x*s],
        [z*x*t - y*s,   z*y*t + x*s,  c + z*z*t]
    ])
    
    return R


def rotation_matrix_from_quaternion(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    
    Parameters
    ----------
    q : array_like, shape (4,)
        Quaternion in the form [w, x, y, z]
        
    Returns
    -------
    R : ndarray, shape (3, 3)
        The rotation matrix
    """
    w, x, y, z = q
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)]
    ])
    
    return R


def quaternion_from_axis_angle(axis, angle):
    """
    Create a quaternion from an axis and angle.
    
    Parameters
    ----------
    axis : array_like, shape (3,)
        The rotation axis (will be normalized)
    angle : float
        The rotation angle in radians
        
    Returns
    -------
    q : ndarray, shape (4,)
        Quaternion in the form [w, x, y, z]
    """
    # Normalize the axis
    axis = np.array(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 0:
        axis = axis / axis_norm
    
    # Create quaternion
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def transform_point(point, position, rotation_matrix):
    """
    Transform a point from local to world coordinates.
    
    Parameters
    ----------
    point : array_like, shape (3,)
        Point in local coordinates
    position : array_like, shape (3,)
        Position of the body in world coordinates
    rotation_matrix : array_like, shape (3, 3)
        Rotation matrix of the body
        
    Returns
    -------
    world_point : ndarray, shape (3,)
        Point in world coordinates
    """
    return rotation_matrix @ point + position


def transform_direction(direction, rotation_matrix):
    """
    Transform a direction vector by a rotation matrix.
    
    Parameters
    ----------
    direction : array_like, shape (3,)
        Direction in local coordinates
    rotation_matrix : array_like, shape (3, 3)
        Rotation matrix
        
    Returns
    -------
    world_direction : ndarray, shape (3,)
        Direction in world coordinates
    """
    return rotation_matrix @ direction