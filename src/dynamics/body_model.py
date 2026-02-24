"""
NeuralAtoms Dynamics Engine — Biomechanical Body Model

Maps SMPL-X 23+1 joints to a standard 24-segment body model using
Dempster (1955) anthropometric mass fractions scaled to a given total
body mass. Segment inertia is approximated as a solid cylinder.

Copyright © 2026 NeuralAtoms — Zaviq Alnour
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dempster (1955) Anthropometric Fractions
# Segment mass as fraction of total body mass (male/neutral approximation)
# ---------------------------------------------------------------------------
# fmt: off
# Dempster (1955) + Winter (2009) hybrid table.
# Trunk split: pelvis 0.142, abdomen/lower-trunk 0.139, thorax/upper-trunk 0.203.
# All values verified to sum to 1.000.
DEMPSTER_MASS_FRACTIONS: Dict[str, float] = {
    "pelvis":          0.142,   # sacrum + pelvis
    "abdomen":         0.139,   # lower trunk / lumbar
    "thorax":          0.203,   # thorax + upper trunk
    "head_neck":       0.081,   # head + neck
    "left_thigh":      0.100,
    "right_thigh":     0.100,
    "left_shank":      0.0465,
    "right_shank":     0.0465,
    "left_foot":       0.0145,
    "right_foot":      0.0145,
    "left_upper_arm":  0.028,
    "right_upper_arm": 0.028,
    "left_forearm":    0.016,
    "right_forearm":   0.016,
    "left_hand":       0.006,
    "right_hand":      0.006,
}
# fmt: on

# Auto-normalize so floating-point drift never breaks the model
_FRACTION_SUM = sum(DEMPSTER_MASS_FRACTIONS.values())
if abs(_FRACTION_SUM - 1.0) > 1e-9:
    DEMPSTER_MASS_FRACTIONS = {
        k: v / _FRACTION_SUM for k, v in DEMPSTER_MASS_FRACTIONS.items()
    }
    _FRACTION_SUM = 1.0


# ---------------------------------------------------------------------------
# SMPL Joint Index → Segment Name Mapping
# SMPL has 24 joints (0=pelvis/root, 1–23 body joints)
# ---------------------------------------------------------------------------
SMPL_TO_SEGMENT: Dict[int, str] = {
    0:  "pelvis",          # root
    1:  "left_thigh",      # l_hip
    2:  "right_thigh",     # r_hip
    3:  "abdomen",         # spine1
    4:  "left_shank",      # l_knee
    5:  "right_shank",     # r_knee
    6:  "abdomen",         # spine2 (shared with spine1 segment)
    7:  "left_foot",       # l_ankle
    8:  "right_foot",      # r_ankle
    9:  "thorax",          # spine3
    10: "left_foot",       # l_foot  (toes, merged with foot)
    11: "right_foot",      # r_foot
    12: "head_neck",       # neck
    13: "left_upper_arm",  # l_collar
    14: "right_upper_arm", # r_collar
    15: "head_neck",       # head
    16: "left_upper_arm",  # l_shoulder
    17: "right_upper_arm", # r_shoulder
    18: "left_forearm",    # l_elbow
    19: "right_forearm",   # r_elbow
    20: "left_hand",       # l_wrist
    21: "right_hand",      # r_wrist
    22: "left_hand",       # l_hand
    23: "right_hand",      # r_hand
}

# Canonical segment lengths (m) for a 1.75m tall person.
# Scaled proportionally for other heights.
_CANONICAL_HEIGHT = 1.75  # metres
_CANONICAL_LENGTHS: Dict[str, float] = {
    "pelvis":           0.19,
    "abdomen":          0.12,
    "thorax":           0.22,
    "head_neck":        0.25,
    "left_thigh":       0.40,
    "right_thigh":      0.40,
    "left_shank":       0.38,
    "right_shank":      0.38,
    "left_foot":        0.10,
    "right_foot":       0.10,
    "left_upper_arm":   0.28,
    "right_upper_arm":  0.28,
    "left_forearm":     0.27,
    "right_forearm":    0.27,
    "left_hand":        0.08,
    "right_hand":       0.08,
}

# Tracked joints for force_profile.csv export
TRACKED_JOINTS = [
    "l_ankle", "r_ankle",
    "l_knee",  "r_knee",
    "l_hip",   "r_hip",
    "spine1",  "spine2",
]

# SMPL joint index for each tracked joint
TRACKED_JOINT_INDICES: Dict[str, int] = {
    "l_ankle": 7,
    "r_ankle": 8,
    "l_knee":  4,
    "r_knee":  5,
    "l_hip":   1,
    "r_hip":   2,
    "spine1":  3,
    "spine2":  6,
}


# ---------------------------------------------------------------------------
# Segment Dataclass
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """A single biomechanical body segment."""
    name: str
    mass_kg: float
    length_m: float
    # Centre of mass as fraction along segment length from proximal end
    com_fraction: float = 0.5
    # Radius for cylinder approximation (m)
    radius_m: float = 0.05

    @property
    def com_offset_m(self) -> float:
        """Distance from proximal attachment to centre of mass (m)."""
        return self.length_m * self.com_fraction

    def inertia_cylinder(self) -> Tuple[float, float, float]:
        """
        Moment of inertia (Ixx, Iyy, Izz) for a solid cylinder about CoM.
        Longitudinal axis = Z by convention.
        Returns (Ixx, Iyy, Izz) in kg·m².
        """
        m = self.mass_kg
        r = self.radius_m
        L = self.length_m

        I_longitudinal = 0.5 * m * r ** 2                  # Izz (spin)
        I_transverse   = (1.0 / 12.0) * m * (3 * r**2 + L**2)  # Ixx, Iyy

        return (I_transverse, I_transverse, I_longitudinal)

    def inertia_diag(self) -> np.ndarray:
        """Returns diagonal inertia tensor as (3,) numpy array."""
        return np.array(self.inertia_cylinder(), dtype=np.float64)


# ---------------------------------------------------------------------------
# Body Model
# ---------------------------------------------------------------------------
@dataclass
class BodyModel:
    """
    Complete 24-segment biomechanical body model.
    Scaled from Dempster anthropometric fractions.
    """
    total_mass_kg: float
    height_m: float
    segments: Dict[str, Segment] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        total_mass_kg: float = 75.0,
        height_m: float = 1.75,
    ) -> "BodyModel":
        """
        Build a BodyModel scaled to given total mass and height.

        Args:
            total_mass_kg: Total body mass in kg (default: 75 kg).
            height_m: Standing height in metres (default: 1.75 m).

        Returns:
            Populated BodyModel instance.
        """
        height_scale = height_m / _CANONICAL_HEIGHT
        segments: Dict[str, Segment] = {}

        for seg_name, mass_frac in DEMPSTER_MASS_FRACTIONS.items():
            mass_kg = total_mass_kg * mass_frac
            base_length = _CANONICAL_LENGTHS.get(seg_name, 0.15)
            length_m = base_length * height_scale

            # Estimate radius from mass and length (cylinder volume = πr²L)
            # density ~1000 kg/m³ (soft tissue approximation)
            density = 1000.0
            volume = mass_kg / density
            radius_m = math.sqrt(max(volume / (math.pi * length_m), 1e-6))
            radius_m = min(max(radius_m, 0.02), 0.12)  # clamp 2–12 cm

            segments[seg_name] = Segment(
                name=seg_name,
                mass_kg=round(mass_kg, 4),
                length_m=round(length_m, 4),
                radius_m=round(radius_m, 4),
            )

        model = cls(
            total_mass_kg=total_mass_kg,
            height_m=height_m,
            segments=segments,
        )
        return model

    def total_computed_mass(self) -> float:
        """Sum of all segment masses (should equal total_mass_kg)."""
        return sum(s.mass_kg for s in self.segments.values())

    def get(self, segment_name: str) -> Segment:
        """Fetch segment by name."""
        if segment_name not in self.segments:
            raise KeyError(f"Segment '{segment_name}' not in body model.")
        return self.segments[segment_name]

    def segment_for_smpl_joint(self, joint_idx: int) -> Segment:
        """Return the segment associated with a given SMPL joint index."""
        seg_name = SMPL_TO_SEGMENT.get(joint_idx)
        if seg_name is None:
            raise KeyError(f"SMPL joint {joint_idx} has no segment mapping.")
        return self.get(seg_name)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"BodyModel ({self.total_mass_kg:.1f} kg, {self.height_m:.2f} m)",
            f"{'Segment':<22} {'Mass(kg)':>9} {'Length(m)':>10} {'Radius(m)':>10}",
            "-" * 54,
        ]
        for seg in self.segments.values():
            lines.append(
                f"{seg.name:<22} {seg.mass_kg:>9.3f} {seg.length_m:>10.3f} "
                f"{seg.radius_m:>10.3f}"
            )
        lines.append("-" * 54)
        lines.append(f"{'TOTAL':<22} {self.total_computed_mass():>9.3f}")
        return "\n".join(lines)


if __name__ == "__main__":
    bm = BodyModel.build(total_mass_kg=75.0, height_m=1.75)
    print(bm.summary())
