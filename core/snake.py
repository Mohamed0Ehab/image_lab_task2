"""Stub implementation for the active-contour (snake) algorithm.

Member 3 must replace the body.  Signature must not change.
"""
from __future__ import annotations

import numpy as np

from core.types import SnakeParams, SnakeResult


def run_snake(
    gray: np.ndarray,
    init_pts: list,
    params: SnakeParams,
) -> SnakeResult:
    """Run the active-contour (snake) algorithm on a grayscale image.

    Parameters
    ----------
    gray:      H×W uint8 grayscale image.
    init_pts:  List of (x, y) tuples forming the initial contour.
    params:    SnakeParams with alpha, beta, gamma, max_iter.

    Returns
    -------
    SnakeResult with pts, chain, perimeter, area and history.
    """
    raise NotImplementedError("run_snake: not implemented yet (Member 3)")
