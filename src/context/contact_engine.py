"""
NeuralAtoms Context Engine — Contact Detection

Identifies contact points between the human skeleton and the environment or 
objects by analyzing the spatial proximity of joints to SAM2-segmented masks.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("neuralatoms.context.contact")


class ContactEngine:
    """
    Detects physical contacts between the subjects and the environment.
    """
    def __init__(self, proximity_threshold_px: int = 15):
        self.proximity_threshold_px = proximity_threshold_px
        # SAM2 Predictor placeholder
        self._predictor = None
        self.is_ready = False # Segment-Anything-2 required
        
    def detect_contacts(
        self, 
        joints_2d: np.ndarray, 
        masks: Dict[int, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Check proximity of 2D joints to object masks.
        
        Args:
            joints_2d: (N_joints, 2) pixel coordinates.
            masks: Dictionary mapping object_id -> binary mask (H, W).
            
        Returns:
            List of contact events: [{"joint_idx": i, "object_id": obj_id, "prob": p}]
        """
        contacts = []
        
        for obj_id, mask in masks.items():
            # Compute distance transform of the mask
            # This gives distance to the nearest non-zero pixel for every pixel
            # We want distance inside the mask to be 0
            binary_mask = (mask > 0).astype(np.uint8)
            dist_map = cv2.distanceTransform(1 - binary_mask, cv2.DIST_L2, 3)
            
            for j_idx, joint in enumerate(joints_2d):
                x, y = int(joint[0]), int(joint[1])
                
                # Bounds check
                if 0 <= y < dist_map.shape[0] and 0 <= x < dist_map.shape[1]:
                    dist = dist_map[y, x]
                    if dist < self.proximity_threshold_px:
                        contacts.append({
                            "joint_idx": j_idx,
                            "object_id": obj_id,
                            "distance_px": float(dist),
                            "is_touching": dist == 0
                        })
                        
        return contacts

    def estimate_contact_forces(self, contacts: List[Dict[str, Any]], torques: np.ndarray):
        """
        Phase 4: Map torques to contact points to estimate external force vectors.
        """
        pass


if __name__ == "__main__":
    # Internal test
    engine = ContactEngine()
    dummy_joints = np.array([[100, 100], [200, 200]])
    dummy_mask = np.zeros((480, 640), dtype=np.uint8)
    dummy_mask[95:105, 95:105] = 1 # Overlaps with joint 0
    
    contacts = engine.detect_contacts(dummy_joints, {1: dummy_mask})
    print(f"Detected contacts: {contacts}")
