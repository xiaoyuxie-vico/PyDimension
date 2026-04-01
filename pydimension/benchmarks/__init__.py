"""
Benchmark descriptors and registry for reproducible symmetry-discovery tasks.
"""

from .registry import list_benchmarks, get_benchmark, register_benchmark
from .synthetic_translation import SYNTHETIC_TRANSLATIONAL_BENCHMARK
from .synthetic_rotation import SYNTHETIC_ROTATIONAL_BENCHMARK
from .synthetic_scaling import SYNTHETIC_SCALING_BENCHMARK
from .keyhole import KEYHOLE_BENCHMARK

__all__ = [
    "list_benchmarks",
    "get_benchmark",
    "register_benchmark",
    "SYNTHETIC_TRANSLATIONAL_BENCHMARK",
    "SYNTHETIC_ROTATIONAL_BENCHMARK",
    "SYNTHETIC_SCALING_BENCHMARK",
    "KEYHOLE_BENCHMARK",
]
