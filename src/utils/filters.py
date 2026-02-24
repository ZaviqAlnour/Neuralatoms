"""
NeuralAtoms Utils — Filtering and Signal Processing

Common filtering utilities used across kinematics and dynamics modules.
"""

import numpy as np
from typing import Tuple, Optional

try:
    from scipy.signal import savgol_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("neuralatoms.utils.filters")


def savgol_smooth(
    data: np.ndarray,
    window: int = 7,
    polyorder: int = 3,
    axis: int = 0
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to a multi-dimensional signal.
    """
    if not _SCIPY_AVAILABLE:
        logger.warning("scipy not installed, skipping Savitzky-Golay smoothing.")
        return data

    n_frames = data.shape[axis]
    win = min(window, n_frames)
    if win % 2 == 0:
        win -= 1
    if win < polyorder + 1:
        win = polyorder + 2
        if win % 2 == 0:
            win += 1
            
    if n_frames < win:
        return data

    return savgol_filter(data, win, polyorder, axis=axis)


def compute_finite_derivatives(
    data: np.ndarray,
    fps: float,
    smooth: bool = False,
    smooth_window: int = 7,
    smooth_polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity and acceleration using finite differences.
    """
    dt = 1.0 / fps
    velocity = np.gradient(data, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)

    if smooth:
        velocity = savgol_smooth(velocity, smooth_window, smooth_polyorder)
        acceleration = savgol_smooth(acceleration, smooth_window, smooth_polyorder)

    return velocity, acceleration
