"""
NeuralAtoms Utils — Geometry & Coordinate Systems

Implements the Gravity-View (GV) Coordinate System for aligning monocular 
reconstructions to a physically consistent ground plane.
"""

import numpy as np
from typing import Tuple, Optional

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm

class GravityView:
    """
    Solves for the GV coordinate system relative to a camera view.
    
    The GV frame is defined such that:
    - Y-axis aligns with negative gravity (World Up).
    - Z-axis aligns with the lateral camera view, projected onto the horizontal plane.
    - X-axis is the cross product of Y and Z.
    """
    def __init__(self, gravity_dir: np.ndarray = np.array([0, -1, 0])):
        # In many monocular systems, gravity is assumed along -Y by default
        self.g = normalize(gravity_dir)
        
    @staticmethod
    def solve_gv_alignment(
        camera_forward: np.ndarray, 
        gravity_est: np.ndarray = np.array([0, -1, 0])
    ) -> np.ndarray:
        """
        Compute the 3x3 rotation matrix R that transforms Camera-space coordinates 
        into Gravity-View coordinates.
        
        Args:
            camera_forward: The forward vector of the camera in world units (or identity).
            gravity_est: The estimated gravity vector in camera space.
            
        Returns:
            R_gv: (3, 3) rotation matrix.
        """
        # World Up (Y_gv) is opposite of gravity
        y_gv = -normalize(gravity_est)
        
        # World Forward (Z_gv) is the camera forward projected onto the horizontal plane
        # horiz_forward = forward - (forward . y_gv) * y_gv
        f = normalize(camera_forward)
        z_gv = normalize(f - np.dot(f, y_gv) * y_gv)
        
        # World Right (X_gv)
        x_gv = np.cross(y_gv, z_gv)
        
        # Build rotation matrix (columns are unit vectors of the new basis)
        R_gv = np.stack([x_gv, y_gv, z_gv], axis=1)
        return R_gv

def apply_gv_transform(points: np.ndarray, R_gv: np.ndarray) -> np.ndarray:
    """
    Apply GV rotation to a set of points.
    points: (N, 3) or (N, J, 3)
    """
    return points @ R_gv.T

def estimate_gravity_from_contacts(
    foot_points: np.ndarray, 
    is_contact: np.ndarray
) -> np.ndarray:
    """
    Heuristic: Use points where the foot is in contact to estimate the floor normal.
    Assumes gravity is perpendicular to the floor.
    """
    contact_pts = foot_points[is_contact > 0.5]
    if len(contact_pts) < 3:
        return np.array([0, -1, 0]) # Default
        
    # Fit a plane using SVD
    centroid = np.mean(contact_pts, axis=0)
    centered = contact_pts - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :] # Smallest singular value vector
    
    # Ensure normal points 'up' (against gravity)
    # If the camera usually looks slightly down, normal.y should be positive
    if normal[1] < 0:
        normal = -normal
        
    # Gravity is the opposite of the floor normal
    return -normal
