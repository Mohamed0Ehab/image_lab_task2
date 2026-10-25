"""Stub implementations for Hough circle detection and ellipse detection.

Member 2 must replace the bodies.  Signatures must not change.
"""
from __future__ import annotations

import numpy as np

from core.types import Circle, Ellipse


def hough_circles(
    edges: np.ndarray,
    r_min: int,
    r_max: int,
) -> list[Circle]:
    """Detect circles in an edge map via the Hough transform.

    Parameters
    ----------
    edges: H×W uint8 binary edge map.
    r_min: Minimum circle radius to search for (pixels).
    r_max: Maximum circle radius to search for (pixels).

    Returns
    -------
    List of Circle(cx, cy, r) in image coordinates.
    """
    raise NotImplementedError("hough_circles: not implemented yet (Member 2)")


def detect_ellipses(
    edges: np.ndarray,
    **params,
) -> list[Ellipse]:
    """Detect ellipses in an edge map.

    Parameters
    ----------
    edges:  H×W uint8 binary edge map.
    params: Keyword arguments (min_axis, max_axis, vote_threshold, …).

    Returns
    -------
    List of Ellipse(cx, cy, a, b, angle) in image coordinates.
    """
    raise NotImplementedError("detect_ellipses: not implemented yet (Member 2)")
