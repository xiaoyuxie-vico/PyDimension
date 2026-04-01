"""
Translational symmetry generator backed by the original PyDimension 2.0 implementation.
"""

from .base import BaseSymmetryGenerator
from .generator import DataGenerator
from .config import DataGenerationConfig


class TranslationalSymmetryGenerator(BaseSymmetryGenerator):
    """Compatibility wrapper around the original synthetic data generator."""

    symmetry_type = "translational"

    def __init__(self, config: DataGenerationConfig):
        self.config = config
        self._generator = DataGenerator(config)

    def generate(self, verbose: bool = False):
        return self._generator.generate(verbose=verbose)

    def save_datasets(self):
        return self._generator.save_datasets()

    def create_visualization(self, filename: str = "data_generation_plots.png"):
        return self._generator.create_visualization(filename=filename)
