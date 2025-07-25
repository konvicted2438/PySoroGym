"""
Main collision module that imports from the new collision package.
This file maintains backward compatibility with the existing codebase.
"""

# Import the new collision detection functions from the collision package
from .collision.collision import detect_collision, detect_plane_collision, gjk
from .collision.minkowski import Simplex, support_function
from .collision.gjk_distance import gjk_distance
from .collision.epa import epa

# Import collision resolution
from .collision_resolution import resolve_contact, Contact

# Export all functions
__all__ = [
    'detect_collision',
    'detect_plane_collision', 
    'gjk',
    'Simplex',
    'support_function',
    'gjk_distance',
    'epa',
    'resolve_contact',
    'Contact'
]