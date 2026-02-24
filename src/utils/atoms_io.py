"""
NeuralAtoms Utils — Unified Atoms I/O

Provides unified read/write access to .atoms and .atoms.npz files.
Handles the multi-modal schema (kinematics + dynamics + context).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("neuralatoms.utils.io")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_KINEMATIC_KEYS = {"pose_body", "pose_root", "trans", "fps", "n_frames"}
REQUIRED_DYNAMICS_KEYS  = {"joint_torques"}
REQUIRED_CONTEXT_KEYS   = {"surface_label", "mu_static", "mu_kinetic"}


def save_atoms_npz(
    filepath: str | Path,
    data: Dict[str, Any],
) -> Path:
    """
    Save atoms data as a compressed .npz file.
    """
    filepath = Path(filepath)
    # Ensure standard extension
    if not str(filepath).endswith(".atoms.npz"):
        if str(filepath).endswith(".atoms"):
            filepath = filepath.with_suffix(".atoms.npz")
        else:
            filepath = filepath.parent / (filepath.name + ".atoms.npz")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(filepath), **data)
    
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info(f"Saved: {filepath.name} ({file_size_mb:.2f} MB)")
    return filepath


def load_atoms_npz(filepath: str | Path) -> Dict[str, Any]:
    """
    Load an atoms .npz file as a dictionary.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Atoms file not found: {filepath}")

    data = dict(np.load(str(filepath), allow_pickle=True))
    return data


def validate_atoms_data(data: Dict[str, Any], module: str = "kinematics") -> bool:
    """
    Validate that the data contains required keys for a specific module.
    """
    if module == "kinematics":
        missing = REQUIRED_KINEMATIC_KEYS - data.keys()
    elif module == "dynamics":
        missing = REQUIRED_DYNAMICS_KEYS - data.keys()
    elif module == "context":
        missing = REQUIRED_CONTEXT_KEYS - data.keys()
    else:
        logger.error(f"Unknown module for validation: {module}")
        return False

    if missing:
        logger.error(f"Atoms data missing required {module} keys: {missing}")
        return False
    return True


def merge_experience_sets(
    primary_data: Dict[str, Any],
    *other_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge multiple atoms data dictionaries into a single unified set.
    """
    merged = primary_data.copy()
    for other in other_data:
        merged.update(other)
    return merged
