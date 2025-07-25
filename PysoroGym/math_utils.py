import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Quaternion utilities (w,x,y,z)
# ─────────────────────────────────────────────────────────────────────────────
def q_identity():
    """Return identity quaternion [1,0,0,0]"""
    return np.array([1.0, 0.0, 0.0, 0.0])

def q_normalize(q):
    """Normalize quaternion to unit length"""
    return q / np.linalg.norm(q)

def q_mul(a, b):
    """Hamilton product a⨂b."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def q_from_axis_angle(axis, angle):
    """Create quaternion from rotation axis and angle (radians)"""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    s = np.sin(angle * 0.5)
    return np.array([np.cos(angle*0.5), *(axis*s)])

def q_to_mat3(q):
    """Quaternion → 3×3 rotation matrix."""
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ])

def q_to_euler(q):
    """Convert quaternion to Euler angles (degrees) in XYZ order."""
    w, x, y, z = q
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.degrees(np.pi / 2 * np.sign(sinp))  # Use 90 degrees if out of range
    else:
        pitch = np.degrees(np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    
    return np.array([roll, pitch, yaw])