"""Operation catalog and feature runner.

``CATALOG`` is the single source of truth for every operation the app offers.
``run_feature()`` is the ONLY place that calls ``core`` functions.
No algorithm logic lives in this file; it only dispatches to stubs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from core.canny_lines      import canny, hough_lines
from core.circles_ellipses import hough_circles, detect_ellipses
from core.snake            import run_snake
from core.types            import SnakeParams, SnakeResult

logger = logging.getLogger(__name__)


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class FeatureEntry:
    """Metadata for one operation in the catalog."""
    id:               str
    name:             str
    group:            str
    params_schema:    dict
    kind:             str               # "image"|"lines"|"circles"|"ellipses"|"snake"
    needs_second_image: bool = False
    func:             Callable = None   # the stub in core


@dataclass
class FeatureOutput:
    """Result returned by run_feature()."""
    result: Optional[np.ndarray]        # new image, or None for shape results
    shapes: list = field(default_factory=list)   # Line | Circle | Ellipse list
    snake:  Optional[SnakeResult] = None


# ============================================================================
# Shared parameter schemas
# ============================================================================

CANNY_PARAMS = {
    "sigma": {
        "type": "float",
        "min": 0.5,
        "max": 5.0,
        "step": 0.1,
        "default": 1.4,
        "label": "Gaussian Sigma",
        "decimals": 1,
    },
    "low_t": {
        "type": "int",
        "min": 0,
        "max": 255,          # was 1000
        "step": 1,
        "default": 50,
        "label": "Canny Low Threshold",
    },
    "high_t": {
        "type": "int",
        "min": 1,
        "max": 255,          # was 1500
        "step": 1,
        "default": 150,
        "label": "Canny High Threshold",
    },
}


def _prefixed_schema(
    schema: dict,
    prefix: str,
) -> dict:
    """Create an independent schema with prefixed parameter keys."""

    return {
        f"{prefix}{key}": {
            **spec,
            "label": (
                f"{prefix.replace('_', ' ').title()} "
                f"{spec.get('label', key)}"
            ),
        }
        for key, spec in schema.items()
    }


HOUGH_CANNY_PARAMS = _prefixed_schema(
    CANNY_PARAMS,
    "canny_",
)


HOUGH_PARAMS = {
    **HOUGH_CANNY_PARAMS,

    "threshold": {
        "type": "int",
        "min": 1,
        "max": 1000,
        "step": 1,
        "default": 100,
        "label": "Hough Vote Threshold",
    },

    "theta_step": {
        "type": "int",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 1,
        "label": "Angular Resolution (degrees)",
    },

    "min_rho_distance": {
        "type": "int",
        "min": 0,
        "max": 30,
        "step": 1,
        "default": 4,
        "label": "Minimum Rho Separation (px)",
    },

    "min_theta_distance": {
        "type": "int",
        "min": 0,
        "max": 30,
        "step": 1,
        "default": 3,
        "label": "Minimum Angle Separation (degrees)",
    },
}

# ===========================================================================
# CATALOG  (built from FeatureEntry objects – never hardcode names/keys elsewhere)
# ===========================================================================

CATALOG: dict[str, list[FeatureEntry]] = {
    "Edges": [
        FeatureEntry(
            id="canny",
            name="Canny",
            group="Edges",
            params_schema={
                "sigma": {
                    "type": "float", "min": 0.5, "max": 5.0,
                    "step": 0.1, "default": 1.4,
                    "label": "Sigma", "decimals": 1,
                },
                "low_t": {
                    "type": "int", "min": 0, "max": 255,
                    "step": 1, "default": 50,
                    "label": "Low Threshold",
                },
                "high_t": {
                    "type": "int", "min": 0, "max": 255,
                    "step": 1, "default": 150,
                    "label": "High Threshold",
                },
            },
            kind="image",
            func=canny,
        )
    ],
    "Lines": [
        FeatureEntry(
            id="hough_lines",
            name="Hough Lines",
            group="Lines",
            params_schema={
                key: dict(spec)
                for key, spec in HOUGH_PARAMS.items()
            },
            kind="lines",
            func=hough_lines,
        ),
    ],  
    "Circles": [
        FeatureEntry(
            id="hough_circles",
            name="Hough Circles",
            group="Circles",
            params_schema={
                "r_min": {
                    "type": "int", "min": 1, "max": 300,
                    "step": 1, "default": 10,
                    "label": "Min Radius (px)",
                },
                "r_max": {
                    "type": "int", "min": 1, "max": 500,
                    "step": 1, "default": 100,
                    "label": "Max Radius (px)",
                },
            },
            kind="circles",
            func=hough_circles,
        )
    ],
    "Ellipses": [
        FeatureEntry(
            id="detect_ellipses",
            name="Ellipse Detection",
            group="Ellipses",
            params_schema={
                "min_axis": {
                    "type": "int", "min": 1, "max": 200,
                    "step": 1, "default": 10,
                    "label": "Min Axis (px)",
                },
                "max_axis": {
                    "type": "int", "min": 1, "max": 500,
                    "step": 1, "default": 100,
                    "label": "Max Axis (px)",
                },
                "vote_threshold": {
                    "type": "int", "min": 1, "max": 200,
                    "step": 1, "default": 50,
                    "label": "Vote Threshold",
                },
            },
            kind="ellipses",
            func=detect_ellipses,
        )
    ],
    "Snake": [
        FeatureEntry(
            id="run_snake",
            name="Active Contour",
            group="Snake",
            params_schema={
                "alpha": {
                    "type": "float", "min": 0.0, "max": 5.0,
                    "step": 0.1, "default": 1.0,
                    "label": "Alpha (continuity)", "decimals": 1,
                },
                "beta": {
                    "type": "float", "min": 0.0, "max": 5.0,
                    "step": 0.1, "default": 1.0,
                    "label": "Beta (curvature)", "decimals": 1,
                },
                "gamma": {
                    "type": "float", "min": 0.0, "max": 5.0,
                    "step": 0.1, "default": 2.0,
                    "label": "Gamma (step size)", "decimals": 1,
                },
                "max_iter": {
                    "type": "int", "min": 1, "max": 2000,
                    "step": 1, "default": 200,
                    "label": "Max Iterations",
                },
                "init_radius": {
                    "type": "int", "min": 5, "max": 500,
                    "step": 1, "default": 50,
                    "label": "Init Radius (px)",
                },
            },
            kind="snake",
            func=run_snake,
        )
    ],
}


# Flat lookup by id
_BY_ID: dict[str, FeatureEntry] = {
    e.id: e for entries in CATALOG.values() for e in entries
}


def get_entry(op_id: str) -> Optional[FeatureEntry]:
    """Return the FeatureEntry for *op_id*, or None."""
    return _BY_ID.get(op_id)


# ===========================================================================
# run_feature  –  the ONLY caller of core functions
# ===========================================================================

def run_feature(
    entry:       FeatureEntry,
    input_array: np.ndarray,
    params:      dict,
    second:      Optional[np.ndarray] = None,
    init_pts:    Optional[list]       = None,
) -> FeatureOutput:
    """Dispatch *entry* with prepared inputs.  Calls the core stub.

    If the core stub raises ``NotImplementedError``, this exception propagates
    upward and is caught by ``run_safely`` in the caller.

    The GUI layer converts colour images to grayscale here when an operation
    requires it (using ``gui.image_io``, not any core function).
    """
    from gui.image_io import ensure_gray

    op = entry.id
    logger.debug("run_feature: %s  params=%s", op, params)

    if op == "canny":
        gray   = ensure_gray(input_array)
        result = entry.func(gray, params["sigma"], params["low_t"], params["high_t"])
        return FeatureOutput(result=result)

    if op == "hough_lines":
        gray = ensure_gray(input_array)

        # First stage: Canny edge detection.
        # Every parameter comes from the sidebar.
        edges = canny(
            gray,
            sigma=params["canny_sigma"],
            low_t=params["canny_low_t"],
            high_t=params["canny_high_t"],
        )

        # Second stage: Hough line detection.
        shapes = entry.func(
            edges,
            threshold=params["threshold"],
            theta_step=params["theta_step"],
            min_rho_distance=params["min_rho_distance"],
            min_theta_distance=params["min_theta_distance"],
        )

        logger.info(
            "Hough Lines: %d edge pixels, %d detected lines",
            np.count_nonzero(edges),
            len(shapes),
        )

        return FeatureOutput(
            result=None,
            shapes=shapes,
        )

    
    if op == "hough_circles":
        gray   = ensure_gray(input_array)
        shapes = entry.func(gray, params["r_min"], params["r_max"])
        return FeatureOutput(result=None, shapes=shapes)

    if op == "detect_ellipses":
        gray   = ensure_gray(input_array)
        ep     = {
            k: params[k]
            for k in ("min_axis", "max_axis", "vote_threshold")
            if k in params
        }
        shapes = entry.func(gray, **ep)
        return FeatureOutput(result=None, shapes=shapes)

    if op == "run_snake":
        gray = ensure_gray(input_array)
        pts  = init_pts or []
        sp   = SnakeParams(
            alpha    = float(params.get("alpha",    1.0)),
            beta     = float(params.get("beta",     1.0)),
            gamma    = float(params.get("gamma",    1.0)),
            max_iter = int(  params.get("max_iter", 200)),
        )
        snake_res = entry.func(gray, pts, sp)
        return FeatureOutput(result=None, snake=snake_res)

    raise ValueError(f"Unknown feature id: {op!r}")
