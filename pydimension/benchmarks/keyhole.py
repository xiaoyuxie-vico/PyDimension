"""
Reference benchmark metadata for the real-world keyhole welding dataset.
"""

KEYHOLE_BENCHMARK = {
    "name": "keyhole",
    "description": (
        "Laser welding keyhole benchmark from the Nature Communications paper. "
        "Uses physical variables with real dimensional structure."
    ),
    "legacy_config": "pydimension/configs/config_keyhole.json",
    "v3_config": "pydimension/configs/config_keyhole.json",
    "symmetry": "translational",
    "status": "active",
}
