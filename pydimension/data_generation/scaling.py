"""
Scaffold for future scaling symmetry data generation.
"""

from .base import BaseSymmetryGenerator


class ScalingSymmetryGenerator(BaseSymmetryGenerator):
    """Placeholder interface for scaling-aware synthetic data."""

    symmetry_type = "scaling"

    def __init__(self, config):
        self.config = config

    def generate(self, verbose: bool = False):
        raise NotImplementedError(
            "Scaling symmetry data generation is planned for a later phase."
        )
