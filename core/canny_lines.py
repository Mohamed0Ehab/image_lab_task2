"""Stub implementations for Canny edge detection and Hough line transform.

Member 1 must replace the bodies.  Signatures must not change.
"""
from __future__ import annotations

import numpy as np

from core.types import Line


def canny(
    gray: np.ndarray,
    sigma: float,
    low_t: float,
    high_t: float,
) -> np.ndarray:
    """Canny edge detection on a grayscale image.

    Parameters
    ----------
    gray:   H×W uint8 grayscale array.
    sigma:  Gaussian blur standard deviation.
    low_t:  Lower hysteresis threshold.
    high_t: Upper hysteresis threshold.

    Returns
    -------
    H×W uint8 binary edge map (0 / 255).
    """
    raise NotImplementedError("canny: not implemented yet (Member 1)")


def hough_lines(
    edges: np.ndarray,
    threshold: int,
) -> list[Line]:
    """Detect straight lines in an edge map via the Hough transform.

    Parameters
    ----------
    edges:     H×W uint8 binary edge map.
    threshold: Minimum number of votes for a line to be accepted.

    Returns
    -------
    List of Line(rho, theta) in image coordinates.
    """
    raise NotImplementedError("hough_lines: not implemented yet (Member 1)")
