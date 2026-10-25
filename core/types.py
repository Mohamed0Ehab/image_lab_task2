"""Plain dataclasses shared between core algorithms and the GUI.

No logic lives here – only data definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Line:
    """A line in Hough space."""
    rho: float
    theta: float  # radians


@dataclass
class Circle:
    """A detected circle."""
    cx: int
    cy: int
    r: float


@dataclass
class Ellipse:
    """A detected ellipse."""
    cx: float
    cy: float
    a: float      # semi-major axis length (pixels)
    b: float      # semi-minor axis length (pixels)
    angle: float  # rotation angle in radians


@dataclass
class SnakeParams:
    """Hyper-parameters for the active-contour (snake) algorithm."""
    alpha: float = 1.0   # continuity weight
    beta: float = 1.0    # curvature weight
    gamma: float = 1.0   # step size / external energy weight
    max_iter: int = 200


@dataclass
class SnakeResult:
    """Output of the active-contour algorithm."""
    pts: list              # [(x, y), …] – final contour points
    chain: str             # Freeman 8-direction chain-code string
    perimeter: float
    area: float
    history: list          # list[list[(x, y)]] – one entry per iteration
