"""
PysoroGym - A physics simulation library for soft robotics.
"""

# Import core components
from .materials import Material
from .physics import PhysicsEngine
from .Body import Body, ShapeInstance
from .World import World
from .visulisation import OpenGLRenderer
from .collision import CollisionDetector, Contact, CollisionPair
from .constraints import ConstraintSolver

# Import shapes
from . import shapes

# Add environments
from .envs import PushBoxEnv

# Version
__version__ = "0.1.0"

# Define what gets imported with "from PysoroGym import *"
__all__ = [
    'Material',
    'PhysicsEngine',
    'Body',
    'ShapeInstance',
    'World',
    'OpenGLRenderer',
    'CollisionDetector',
    'Contact',
    'CollisionPair',
    'ConstraintSolver',
    'shapes',
    'PushBoxEnv'
]

