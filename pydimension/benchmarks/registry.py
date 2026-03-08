"""
Central registry of reproducible benchmark tasks.

Each benchmark is a dictionary with standard keys so that any benchmark can be
run through the same pipeline dispatcher without special-casing.
"""

from typing import Any, Dict, List, Optional

from .synthetic_translation import SYNTHETIC_TRANSLATIONAL_BENCHMARK
from .synthetic_rotation import SYNTHETIC_ROTATIONAL_BENCHMARK
from .synthetic_scaling import SYNTHETIC_SCALING_BENCHMARK
from .keyhole import KEYHOLE_BENCHMARK


_REGISTRY: Dict[str, Dict[str, Any]] = {
    SYNTHETIC_TRANSLATIONAL_BENCHMARK["name"]: SYNTHETIC_TRANSLATIONAL_BENCHMARK,
    SYNTHETIC_ROTATIONAL_BENCHMARK["name"]: SYNTHETIC_ROTATIONAL_BENCHMARK,
    SYNTHETIC_SCALING_BENCHMARK["name"]: SYNTHETIC_SCALING_BENCHMARK,
    KEYHOLE_BENCHMARK["name"]: KEYHOLE_BENCHMARK,
}


def list_benchmarks() -> List[str]:
    """Return names of all registered benchmarks."""
    return list(_REGISTRY.keys())


def get_benchmark(name: str) -> Dict[str, Any]:
    """Retrieve a benchmark descriptor by name."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(f"Unknown benchmark '{name}'. Available: {available}")
    return _REGISTRY[name]


def register_benchmark(descriptor: Dict[str, Any]) -> None:
    """Add a custom benchmark descriptor to the registry at runtime."""
    name = descriptor.get("name")
    if name is None:
        raise ValueError("Benchmark descriptor must have a 'name' key.")
    _REGISTRY[name] = descriptor
