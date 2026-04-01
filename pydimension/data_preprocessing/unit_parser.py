"""
Unit string parsing: convert human-readable unit strings into fundamental
dimension vectors [Mass, Length, Time, Temperature, Current, Amount, Luminous].
"""

from typing import Dict, List


UNIT_RECOMMENDATIONS: Dict[str, str] = {
    "etaP": "W",
    "Vs": "m/s",
    "r0": "m",
    "alpha": "m²/s",
    "rho": "kg/m³",
    "cp": "J/(kg·K)",
    "Tv-T0": "K",
    "Lv": "J/kg",
    "Tl-T0": "K",
    "Lm": "J/kg",
    "e": "dimensionless",
    "Ke": "dimensionless",
    "e*": "dimensionless",
    "p*": "dimensionless",
}


def infer_units(variables: List[str]) -> Dict[str, str]:
    """Best-effort unit inference from variable names using known patterns."""
    units: Dict[str, str] = {}
    for var in variables:
        if var in UNIT_RECOMMENDATIONS:
            units[var] = UNIT_RECOMMENDATIONS[var]
        elif var.startswith("p") and var[1:].isdigit():
            units[var] = "dimensionless"
        elif var.endswith("*"):
            units[var] = "dimensionless"
        else:
            units[var] = "dimensionless"
    return units


def parse_dimensions(unit: str) -> List[int]:
    """Parse a unit string into a 7-element dimension vector.

    Order: [Mass, Length, Time, Temperature, Current, Amount, Luminous].
    """
    dims = [0, 0, 0, 0, 0, 0, 0]
    unit_lower = unit.lower().replace(" ", "").replace("·", "").replace("⋅", "").replace("*", "")

    if "dimensionless" in unit_lower or unit == "1":
        return dims

    cp_patterns = ["j/(kgk)", "j/kg/k", "jkg^-1k^-1", "jkg-1k-1", "j/(kg·k)", "j/(kg*k)"]
    if any(p in unit_lower for p in cp_patterns):
        return [0, 2, -2, -1, 0, 0, 0]

    if "kg" in unit_lower:
        dims[0] = -1 if "/kg" in unit_lower else 1

    if "kg/m³" in unit_lower or "kg/m^3" in unit_lower:
        dims[1] = -3
    elif "m²/s" in unit_lower or "m^2/s" in unit_lower:
        dims[1] = 2
    elif "m³" in unit_lower or "m^3" in unit_lower:
        dims[1] = 3
    elif "m²" in unit_lower or "m^2" in unit_lower:
        dims[1] = 2
    elif "m/s" in unit_lower:
        dims[1] = 1
    elif unit_lower == "m":
        dims[1] = 1

    if "/s²" in unit_lower or "/s^2" in unit_lower:
        dims[2] = -2
    elif "/s³" in unit_lower or "/s^3" in unit_lower:
        dims[2] = -3
    elif "/s" in unit_lower:
        dims[2] = -1

    if "(kg·k)" in unit_lower or "/(kg·k)" in unit_lower:
        dims[3] = -1
    elif unit_lower.endswith("k") or "k)" in unit_lower or unit_lower == "k":
        dims[3] = 1

    if "w" in unit_lower and "j" not in unit_lower:
        dims[0], dims[1], dims[2] = 1, 2, -3
    elif any(p in unit_lower for p in ["j/(kg·k)", "j/(kg*k)", "j/kg/k", "j/(kgk)"]):
        dims[0], dims[1], dims[2], dims[3] = 0, 2, -2, -1
    elif "j/kg" in unit_lower:
        dims[0], dims[1], dims[2] = 0, 2, -2
    elif "j" in unit_lower:
        dims[0], dims[1], dims[2] = 1, 2, -2

    return dims
