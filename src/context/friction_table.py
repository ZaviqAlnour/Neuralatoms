"""
NeuralAtoms Context Engine — Friction Lookup Table

Provides static and kinetic friction coefficients (μ) for human-environment 
interactions across various material pairs.

Based on standard engineering tables and biomechanical literature for 
footwear-surface interfaces.
"""

from typing import Dict, Tuple, Optional

# Material Pair -> (mu_static, mu_kinetic)
# Key format: "material1|material2" (alphabetical order for consistency)
FRICTION_REGISTRY: Dict[str, Tuple[float, float]] = {
    # Footwear/Skin on Surfaces
    "rubber|concrete":      (0.85, 0.65),
    "rubber|wood":          (0.75, 0.55),
    "rubber|steel":         (0.65, 0.45),
    "rubber|tile":          (0.60, 0.40),
    "rubber|grass":         (0.50, 0.35),
    "rubber|ice":           (0.15, 0.05),
    "rubber|linoleum":      (0.55, 0.45),
    "rubber|carpet":        (0.70, 0.60),
    
    # Skin (barefoot)
    "skin|concrete":        (0.70, 0.50),
    "skin|wood":            (0.60, 0.45),
    "skin|tile":            (0.50, 0.30),
    "skin|carpet":          (0.65, 0.55),
    "skin|ice":             (0.10, 0.03),
    
    # Defaults
    "default|surface":      (0.60, 0.45),
}

def get_friction_coefficients(
    material_a: str, 
    material_b: str = "rubber"
) -> Tuple[float, float]:
    """
    Retrieve static and kinetic friction coefficients for a material pair.
    
    Args:
        material_a: Detected surface material (e.g., 'concrete').
        material_b: Interaction material (defaults to 'rubber' for shoes).
        
    Returns:
        (mu_static, mu_kinetic)
    """
    # Standardize names and order
    pair = "|".join(sorted([material_a.lower(), material_b.lower()]))
    
    coeffs = FRICTION_REGISTRY.get(pair)
    if coeffs:
        return coeffs
        
    # Fallback to single material default if pair not found
    # Assume rubber interaction if not specified
    fallback_pair = f"{material_a.lower()}|rubber"
    coeffs = FRICTION_REGISTRY.get(fallback_pair)
    if coeffs:
        return coeffs
        
    return FRICTION_REGISTRY["default|surface"]

def get_material_list():
    """Returns list of supported surface materials."""
    materials = set()
    for key in FRICTION_REGISTRY.keys():
        for m in key.split("|"):
            if m not in ["rubber", "skin", "default", "surface"]:
                materials.add(m)
    return sorted(list(materials))

if __name__ == "__main__":
    # Test cases
    print(f"Rubber on Concrete: {get_friction_coefficients('concrete', 'rubber')}")
    print(f"Barefoot on Ice:    {get_friction_coefficients('ice', 'skin')}")
    print(f"Unknown Surface:     {get_friction_coefficients('obsidian')}")
