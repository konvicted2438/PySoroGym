"""
Material properties for physics simulation.
"""
from dataclasses import dataclass


@dataclass
class Material:
    """Material properties for physics simulation"""
    friction: float = 0.1
    elasticity: float = 0.1  # Coefficient of restitution
    density: float = 1000.0  # kg/m^3