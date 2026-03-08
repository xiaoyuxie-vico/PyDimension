"""
Scaffold for future rotational symmetry data generation.
"""

from .base import BaseSymmetryGenerator


class RotationalSymmetryGenerator(BaseSymmetryGenerator):
    """Placeholder interface for rotationally equivariant synthetic data."""

    symmetry_type = "rotational"

    def __init__(self, config):
        self.config = config

    def generate(self, verbose: bool = False):
        raise NotImplementedError(
            "Rotational symmetry data generation is planned for a later phase."
        )
