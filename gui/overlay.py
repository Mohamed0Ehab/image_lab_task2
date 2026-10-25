"""Shape overlay renderer for ImageViewer windows.

``OverlayRenderer`` stores detected shapes and draws them on top of the
image pixmap inside ``_ImageDisplay.paintEvent``.  Drawing happens in
*image-coordinate space* (the painter has already been translated and
scaled), so shapes stay pixel-locked at any zoom level.

Pen widths are divided by zoom to keep a constant *screen* stroke width.
"""
from __future__ import annotations

import math
import logging
from typing import Optional

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF

from core.types import Circle, Ellipse, Line

logger = logging.getLogger(__name__)

# Screen-space sizes (constant regardless of zoom)
_LINE_W_PX   = 1.5
_DOT_R_PX    = 3.0
_CROSS_LEN   = 6.0   # centre cross half-length for circles


# ---------------------------------------------------------------------------
# Line clipping helper (Liang-Barsky algorithm)
# ---------------------------------------------------------------------------

def _clip_to_rect(
    x1: float, y1: float, x2: float, y2: float,
    rx0: float, ry0: float, rx1: float, ry1: float,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """Clip a line segment to an axis-aligned rectangle.

    Returns ``((cx1, cy1), (cx2, cy2))`` or ``None`` if fully outside.
    """
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - rx0, rx1 - x1, y1 - ry0, ry1 - y1)
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        elif pi < 0:
            t0 = max(t0, qi / pi)
        else:
            t1 = min(t1, qi / pi)
    if t0 > t1:
        return None
    return (x1 + t0 * dx, y1 + t0 * dy), (x1 + t1 * dx, y1 + t1 * dy)


def _line_endpoints(
    rho: float, theta: float, img_w: int, img_h: int,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """Compute two image-space endpoints for a Hough (rho, theta) line."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x0 = cos_t * rho
    y0 = sin_t * rho
    diag = math.hypot(img_w, img_h)
    # Direction along the line (perpendicular to the normal)
    dx, dy = -sin_t, cos_t
    x1, y1 = x0 + diag * dx, y0 + diag * dy
    x2, y2 = x0 - diag * dx, y0 - diag * dy
    return _clip_to_rect(x1, y1, x2, y2, 0, 0, float(img_w), float(img_h))


# ---------------------------------------------------------------------------
# OverlayRenderer
# ---------------------------------------------------------------------------

class OverlayRenderer:
    """Stores shape overlays and draws them in image-coordinate space."""

    def __init__(self) -> None:
        self._lines: list[Line] = []
        self._circles: list[Circle] = []
        self._ellipses: list[Ellipse] = []
        self._snake_pts: list[tuple[float, float]] = []
        self._img_w: int = 0
        self._img_h: int = 0

        # Toggle and colour per overlay type
        self.show_lines:     bool   = True
        self.lines_color:    QColor = QColor("#f38ba8")   # Catppuccin red

        self.show_circles:   bool   = True
        self.circles_color:  QColor = QColor("#a6e3a1")   # Catppuccin green

        self.show_ellipses:  bool   = True
        self.ellipses_color: QColor = QColor("#f9e2af")   # Catppuccin yellow

        self.show_snake:     bool   = True
        self.snake_color:    QColor = QColor("#89b4fa")   # Catppuccin blue

    # --- Setters -----------------------------------------------------------

    def set_image_size(self, w: int, h: int) -> None:
        """Update image dimensions (needed for line clipping)."""
        self._img_w, self._img_h = w, h

    def set_lines(self, lines: list[Line]) -> None:
        """Replace the stored line overlay."""
        self._lines = list(lines)

    def set_circles(self, circles: list[Circle]) -> None:
        """Replace the stored circle overlay."""
        self._circles = list(circles)

    def set_ellipses(self, ellipses: list[Ellipse]) -> None:
        """Replace the stored ellipse overlay."""
        self._ellipses = list(ellipses)

    def set_snake(self, pts: list[tuple[float, float]]) -> None:
        """Replace the snake contour overlay."""
        self._snake_pts = list(pts)

    def clear(self) -> None:
        """Remove all stored shapes."""
        self._lines = []
        self._circles = []
        self._ellipses = []
        self._snake_pts = []

    def has_content(self) -> bool:
        """Return True if at least one overlay shape is stored."""
        return bool(
            self._lines or self._circles or self._ellipses or self._snake_pts
        )

    # --- Drawing -----------------------------------------------------------

    def draw(self, painter: QPainter, zoom: float) -> None:
        """Draw all enabled overlays.

        The painter must already be in *image-coordinate space* (translate +
        scale applied).  *zoom* is used to keep pen widths constant in screen
        pixels.
        """
        pw = max(0.3, _LINE_W_PX / zoom)   # pen width in image coords

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self.show_lines and self._lines:
            self._draw_lines(painter, pw)

        if self.show_circles and self._circles:
            self._draw_circles(painter, pw, zoom)

        if self.show_ellipses and self._ellipses:
            self._draw_ellipses(painter, pw)

        if self.show_snake and self._snake_pts:
            self._draw_snake(painter, pw, zoom)

    # --- Private drawing methods -------------------------------------------

    def _draw_lines(self, painter: QPainter, pw: float) -> None:
        painter.setPen(QPen(self.lines_color, pw))
        for line in self._lines:
            pts = _line_endpoints(line.rho, line.theta, self._img_w, self._img_h)
            if pts is None:
                continue
            (x1, y1), (x2, y2) = pts
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _draw_circles(self, painter: QPainter, pw: float, zoom: float) -> None:
        painter.setPen(QPen(self.circles_color, pw))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        cross = _CROSS_LEN / zoom
        for c in self._circles:
            painter.drawEllipse(QPointF(c.cx, c.cy), float(c.r), float(c.r))
            # Centre cross
            painter.drawLine(
                QPointF(c.cx - cross, c.cy), QPointF(c.cx + cross, c.cy)
            )
            painter.drawLine(
                QPointF(c.cx, c.cy - cross), QPointF(c.cx, c.cy + cross)
            )

    def _draw_ellipses(self, painter: QPainter, pw: float) -> None:
        painter.setPen(QPen(self.ellipses_color, pw))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for e in self._ellipses:
            painter.save()
            painter.translate(e.cx, e.cy)
            painter.rotate(math.degrees(e.angle))
            painter.drawEllipse(QPointF(0.0, 0.0), e.a, e.b)
            painter.restore()

    def _draw_snake(self, painter: QPainter, pw: float, zoom: float) -> None:
        pts = self._snake_pts
        if not pts:
            return
        dot_r = _DOT_R_PX / zoom

        # Polyline (closed)
        painter.setPen(QPen(self.snake_color, pw))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        poly = QPolygonF([QPointF(x, y) for x, y in pts])
        poly.append(QPointF(pts[0][0], pts[0][1]))   # close the loop
        painter.drawPolyline(poly)

        # Point dots
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.snake_color))
        for x, y in pts:
            painter.drawEllipse(QPointF(x, y), dot_r, dot_r)
