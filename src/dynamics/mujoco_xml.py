"""
NeuralAtoms Dynamics Engine — MuJoCo MJCF XML Builder

Procedurally generates a valid MuJoCo XML model of a human skeleton
from a BodyModel. The resulting XML can be compiled by MuJoCo in-memory
or written to disk for inspection.

Joint convention (matching SMPL):
  - Ball joints (3-DOF): hips, shoulders, spine, ankle
  - Hinge joints (1-DOF): knees, elbows
  - Y-axis is UP (matches WHAM world coordinate output)

Copyright © 2026 NeuralAtoms — Zaviq Alnour
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .body_model import BodyModel, Segment
except (ImportError, ValueError):
    from body_model import BodyModel, Segment

# ---------------------------------------------------------------------------
# Kinematic tree: (child_segment, parent_segment, joint_type, joint_axis)
# ---------------------------------------------------------------------------
# fmt: off
KINEMATIC_TREE = [
    # (body_name,       parent,           joint_type, joint_axis, limited)
    ("abdomen",         "pelvis",          "ball",   None,         False),
    ("thorax",          "abdomen",         "ball",   None,         False),
    ("head_neck",       "thorax",          "ball",   None,         False),
    ("left_thigh",      "pelvis",          "ball",   None,         False),
    ("left_shank",      "left_thigh",      "hinge",  "0 0 1",      True ),
    ("left_foot",       "left_shank",      "ball",   None,         False),
    ("right_thigh",     "pelvis",          "ball",   None,         False),
    ("right_shank",     "right_thigh",     "hinge",  "0 0 1",      True ),
    ("right_foot",      "right_shank",     "ball",   None,         False),
    ("left_upper_arm",  "thorax",          "ball",   None,         False),
    ("left_forearm",    "left_upper_arm",  "hinge",  "0 0 1",      True ),
    ("left_hand",       "left_forearm",    "ball",   None,         False),
    ("right_upper_arm", "thorax",          "ball",   None,         False),
    ("right_forearm",   "right_upper_arm", "hinge",  "0 0 1",      True ),
    ("right_hand",      "right_forearm",   "ball",   None,         False),
]
# fmt: on

# Joint DOF counts  (for nv calculation reference)
JOINT_DOF = {"ball": 3, "hinge": 1, "free": 6}

# Knee / elbow hinge limits in degrees (anatomical range)
HINGE_LIMITS_DEG = {
    "left_shank":      (-130, 0),
    "right_shank":     (-130, 0),
    "left_forearm":    (0, 145),
    "right_forearm":   (0, 145),
}


def _seg_pos(seg: Segment, parent: Optional[Segment]) -> str:
    """
    Compute segment attachment position relative to parent's frame.
    Placed at the distal end of the parent segment along Y-axis.
    """
    if parent is None:
        return "0 0 0"
    dist = parent.length_m
    return f"0 {dist:.4f} 0"


def _capsule_fromto(seg: Segment) -> str:
    """
    Capsule geometry: proximal–distal endpoint pair along Y-axis.
    """
    return f"0 0 0 0 {seg.length_m:.4f} 0"


def _inertia_str(seg: Segment) -> str:
    """Format diagonal inertia for MuJoCo <inertial> element."""
    Ixx, Iyy, Izz = seg.inertia_cylinder()
    return f"{Ixx:.6f} {Iyy:.6f} {Izz:.6f}"


def build_mjcf(
    body_model: BodyModel,
    timestep: float = 0.003333,   # 300 Hz internal sim (≥10× video FPS)
    gravity: float = -9.81,
) -> str:
    """
    Build a complete MuJoCo MJCF XML string from a BodyModel.

    Args:
        body_model: Populated BodyModel instance.
        timestep: MuJoCo simulation timestep in seconds.
        gravity: Gravitational acceleration (negative = downward Y).

    Returns:
        Valid MJCF XML string ready for mujoco.MjModel.from_xml_string().
    """
    root = ET.Element("mujoco", model="neuralatoms_human")

    # ---- Compiler ----
    ET.SubElement(root, "compiler",
                  coordinate="local",
                  angle="radian",
                  eulerseq="xyz")

    # ---- Options ----
    ET.SubElement(root, "option",
                  timestep=str(timestep),
                  gravity=f"0 {gravity:.4f} 0",
                  integrator="RK4")

    # ---- Defaults ----
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint",
                  damping="0.5",
                  armature="0.01",
                  frictionloss="0.0")
    ET.SubElement(default, "geom",
                  type="capsule",
                  rgba="0.7 0.7 0.8 1",
                  condim="3",
                  friction="0.8 0.02 0.001")

    # ---- Assets ----
    ET.SubElement(root, "asset")

    # ---- World body ----
    worldbody = ET.SubElement(root, "worldbody")

    # Floor plane
    ET.SubElement(worldbody, "geom",
                  name="floor",
                  type="plane",
                  size="10 10 0.1",
                  pos="0 0 0",
                  rgba="0.3 0.3 0.3 1")

    # Root free body — pelvis floating in world
    pelvis_seg = body_model.get("pelvis")
    pelvis_body = ET.SubElement(worldbody, "body",
                                name="pelvis",
                                pos="0 0.9 0")  # start 0.9 m above floor

    ET.SubElement(pelvis_body, "freejoint", name="root")

    ET.SubElement(pelvis_body, "inertial",
                  pos=f"0 {pelvis_seg.com_offset_m:.4f} 0",
                  mass=str(pelvis_seg.mass_kg),
                  diaginertia=_inertia_str(pelvis_seg))

    ET.SubElement(pelvis_body, "geom",
                  name="pelvis_geom",
                  type="capsule",
                  fromto=_capsule_fromto(pelvis_seg),
                  size=str(pelvis_seg.radius_m))

    # Build segment map for lookup during tree construction
    body_elements = {"pelvis": pelvis_body}

    for (seg_name, parent_name, joint_type, joint_axis, limited) in KINEMATIC_TREE:
        if seg_name not in body_model.segments:
            continue

        seg = body_model.get(seg_name)
        parent_seg = body_model.segments.get(parent_name)
        parent_elem = body_elements.get(parent_name, pelvis_body)

        pos_str = _seg_pos(seg, parent_seg)
        body_elem = ET.SubElement(parent_elem, "body",
                                  name=seg_name,
                                  pos=pos_str)

        # Joint
        joint_attrs = {
            "name": f"j_{seg_name}",
            "type": joint_type,
        }
        if joint_type == "hinge":
            joint_attrs["axis"] = joint_axis
            if limited and seg_name in HINGE_LIMITS_DEG:
                lo, hi = HINGE_LIMITS_DEG[seg_name]
                lo_r = round(lo * np.pi / 180, 4)
                hi_r = round(hi * np.pi / 180, 4)
                joint_attrs["range"] = f"{lo_r} {hi_r}"
                joint_attrs["limited"] = "true"
        elif joint_type == "ball":
            joint_attrs["limited"] = "false"

        ET.SubElement(body_elem, "joint", **joint_attrs)

        # Inertial
        ET.SubElement(body_elem, "inertial",
                      pos=f"0 {seg.com_offset_m:.4f} 0",
                      mass=str(seg.mass_kg),
                      diaginertia=_inertia_str(seg))

        # Geometry
        ET.SubElement(body_elem, "geom",
                      name=f"{seg_name}_geom",
                      type="capsule",
                      fromto=_capsule_fromto(seg),
                      size=str(seg.radius_m))

        body_elements[seg_name] = body_elem

    # ---- Actuators (passive — only needed for forward dynamics, not inverse) ----
    # We register actuators so MuJoCo's nv is fully defined, but they aren't
    # used during mj_inverse (which only reads qfrc_inverse).
    actuator = ET.SubElement(root, "actuator")
    for (seg_name, _, joint_type, _, _) in KINEMATIC_TREE:
        if seg_name not in body_model.segments:
            continue
        joint_name = f"j_{seg_name}"
        if joint_type == "hinge":
            ET.SubElement(actuator, "motor",
                          name=f"motor_{seg_name}",
                          joint=joint_name,
                          gear="1",
                          ctrllimited="false")
        elif joint_type == "ball":
            for ax, axis_label in enumerate(["x", "y", "z"]):
                ET.SubElement(actuator, "general",
                              name=f"motor_{seg_name}_{axis_label}",
                              joint=joint_name,
                              gear="1",
                              ctrllimited="false",
                              biastype="none",
                              dyntype="none")

    # ---- Serialize ----
    ET.indent(root, space="  ")
    xml_str = ET.tostring(root, encoding="unicode", xml_declaration=False)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + xml_str


def save_mjcf(
    body_model: BodyModel,
    output_path: str,
    timestep: float = 0.003333,
) -> str:
    """
    Build MJCF XML and write to disk.

    Args:
        body_model: BodyModel instance.
        output_path: Full path to write the .xml file.
        timestep: Simulation timestep.

    Returns:
        The XML string written.
    """
    xml_str = build_mjcf(body_model, timestep=timestep)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return xml_str


if __name__ == "__main__":
    from body_model import BodyModel
    bm = BodyModel.build(75.0, 1.75)
    xml = build_mjcf(bm)
    print(xml[:2000])
    print(f"\n... ({len(xml)} chars total)")
