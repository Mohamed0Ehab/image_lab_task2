"""Reusable widgets: JumpSlider, CollapsibleBox, ParamPanel, ImageViewer.

These are the building blocks used everywhere in the application.
No algorithm logic lives here.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from PySide6.QtCore import (
    Qt, Signal, QTimer, QPoint, QPointF, QRectF, QSize, QSizeF,
)
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QKeyEvent, QMouseEvent, QPainter,
    QPen, QPixmap, QPolygonF, QResizeEvent, QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QMenu, QPushButton,
    QSizePolicy, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from gui.image_io import MODES, ImageData, array_to_qpixmap, convert_mode
from gui.overlay import OverlayRenderer

logger = logging.getLogger(__name__)

# Screen-space constants for snake contour input
_CONTOUR_LINE_W_PX = 1.5
_CONTOUR_DOT_R_PX  = 5.0
_HIT_RADIUS_PX     = 8.0


# ===========================================================================
# JumpSlider
# ===========================================================================

class JumpSlider(QSlider):
    """A QSlider whose handle jumps instantly to the clicked position."""

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._jump(event.pos())
        super().mousePressEvent(event)

    def _jump(self, pos: QPoint) -> None:
        if self.orientation() == Qt.Orientation.Horizontal:
            ratio = pos.x() / max(1, self.width())
        else:
            ratio = 1.0 - pos.y() / max(1, self.height())
        val = self.minimum() + ratio * (self.maximum() - self.minimum())
        self.setValue(round(val))


# ===========================================================================
# CollapsibleBox
# ===========================================================================

class CollapsibleBox(QWidget):
    """A widget section with a toggle-button header that shows/hides content."""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self._btn = QToolButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(True)
        self._btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._btn.clicked.connect(self._toggle)
        self._refresh_label()

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(4, 2, 4, 4)
        self._body_layout.setSpacing(4)

        root.addWidget(self._btn)
        root.addWidget(self._body)

    # --- Private ---

    def _toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._refresh_label()

    def _refresh_label(self) -> None:
        arrow = "▼" if self._btn.isChecked() else "▶"
        self._btn.setText(f"  {arrow}  {self._title}")

    # --- Public API ---

    def set_title(self, title: str) -> None:
        """Change the header title text."""
        self._title = title
        self._refresh_label()

    def set_expanded(self, expanded: bool) -> None:
        """Programmatically expand or collapse the box."""
        self._btn.setChecked(expanded)
        self._body.setVisible(expanded)
        self._refresh_label()

    def content_layout(self) -> QVBoxLayout:
        """Return the layout to which child widgets should be added."""
        return self._body_layout


# ===========================================================================
# ParamPanel
# ===========================================================================

class ParamPanel(QGroupBox):
    """Builds parameter controls from a schema dict; emits ``changed``."""

    changed = Signal()

    def __init__(
        self, title: str = "Parameters", parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(title, parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 20, 8, 8)
        self._layout.setSpacing(6)
        self._entries: dict[str, dict] = {}

    # --- Public API ---

    def set_schema(self, schema: dict) -> None:
        """Rebuild all controls from *schema*.

        Schema format::

            {
                "key": {
                    "type":    "int" | "float" | "combo" | "bool",
                    "label":   "Human label",
                    "min":     ..., "max": ..., "step": ..., "default": ...,
                    "decimals": ...,      # float only
                    "options": [...],     # combo only
                }
            }
        """
        # Clear existing widgets
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._entries.clear()

        if not schema:
            lbl = QLabel("No parameters.")
            lbl.setStyleSheet("color:#6c7086;font-size:11px;")
            self._layout.addWidget(lbl)
            self._layout.addStretch()
            return

        for key, spec in schema.items():
            ptype      = spec.get("type", "int")
            label_text = spec.get("label", key)
            wrapper    = QWidget()
            col        = QVBoxLayout(wrapper)
            col.setContentsMargins(0, 0, 0, 2)
            col.setSpacing(2)
            entry: dict = {"type": ptype}

            if ptype in ("int", "float"):
                header_lbl = QLabel(label_text)
                header_lbl.setStyleSheet("font-size:11px;color:#a6adc8;")
                col.addWidget(header_lbl)

                row_w = QWidget()
                row   = QHBoxLayout(row_w)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(6)
                slider = JumpSlider(Qt.Orientation.Horizontal)

                if ptype == "int":
                    mn  = int(spec.get("min", 0))
                    mx  = int(spec.get("max", 100))
                    stp = int(spec.get("step", 1))
                    dflt = int(spec.get("default", mn))
                    slider.setRange(mn, mx)
                    slider.setSingleStep(stp)
                    slider.setValue(dflt)

                    spin = QSpinBox()
                    spin.setRange(mn, mx)
                    spin.setSingleStep(stp)
                    spin.setValue(dflt)
                    spin.setFixedWidth(70)

                    # Bidirectional sync without infinite loops
                    slider.valueChanged.connect(spin.setValue)
                    spin.valueChanged.connect(slider.setValue)
                    slider.valueChanged.connect(lambda _: self.changed.emit())

                    entry.update({"slider": slider, "spin": spin})

                else:   # float
                    mn   = float(spec.get("min", 0.0))
                    mx   = float(spec.get("max", 1.0))
                    stp  = float(spec.get("step", 0.1))
                    dflt = float(spec.get("default", mn))
                    decs = int(spec.get("decimals", 2))
                    scale = max(1, round(1.0 / stp)) if stp > 0 else 100

                    slider.setRange(round(mn * scale), round(mx * scale))
                    slider.setValue(round(dflt * scale))

                    spin = QDoubleSpinBox()
                    spin.setRange(mn, mx)
                    spin.setSingleStep(stp)
                    spin.setDecimals(decs)
                    spin.setValue(dflt)
                    spin.setFixedWidth(75)

                    def _s2sp(v: int, sp: QDoubleSpinBox = spin, sc: int = scale) -> None:
                        sp.blockSignals(True)
                        sp.setValue(v / sc)
                        sp.blockSignals(False)

                    def _sp2s(v: float, sl: JumpSlider = slider, sc: int = scale) -> None:
                        sl.blockSignals(True)
                        sl.setValue(round(v * sc))
                        sl.blockSignals(False)

                    slider.valueChanged.connect(_s2sp)
                    slider.valueChanged.connect(lambda _: self.changed.emit())
                    spin.valueChanged.connect(_sp2s)
                    spin.valueChanged.connect(lambda _: self.changed.emit())

                    entry.update({"slider": slider, "spin": spin, "scale": scale})

                row.addWidget(slider, 1)
                row.addWidget(spin)
                col.addWidget(row_w)

            elif ptype == "combo":
                header_lbl = QLabel(label_text)
                header_lbl.setStyleSheet("font-size:11px;color:#a6adc8;")
                col.addWidget(header_lbl)

                combo = QComboBox()
                options = spec.get("options", [])
                combo.addItems([str(o) for o in options])
                dflt = spec.get("default", options[0] if options else "")
                idx  = combo.findText(str(dflt))
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                combo.currentIndexChanged.connect(lambda _: self.changed.emit())
                col.addWidget(combo)
                entry["combo"] = combo

            elif ptype == "bool":
                chk = QCheckBox(label_text)
                chk.setChecked(bool(spec.get("default", False)))
                chk.stateChanged.connect(lambda _: self.changed.emit())
                col.addWidget(chk)
                entry["checkbox"] = chk

            self._entries[key] = entry
            self._layout.addWidget(wrapper)

        self._layout.addStretch()

    def values(self) -> dict:
        """Return current value for every parameter key in the schema."""
        out: dict = {}
        for key, entry in self._entries.items():
            t = entry["type"]
            if t in ("int", "float"):
                out[key] = entry["spin"].value()
            elif t == "combo":
                out[key] = entry["combo"].currentText()
            elif t == "bool":
                out[key] = entry["checkbox"].isChecked()
        return out


# ===========================================================================
# _ImageDisplay  (internal rendering widget)
# ===========================================================================

class _ImageDisplay(QWidget):
    """Low-level widget: renders image + overlays + snake contour input.

    The painter pipeline in ``paintEvent`` uses *image-coordinate space*
    (translate by pan_x/pan_y, scale by zoom) so that overlays stay locked
    to the image pixels at any zoom level.
    """

    view_changed    = Signal(float, float, float)   # zoom, h_ratio, v_ratio
    mouse_info      = Signal(int, int, object)       # img_x, img_y, pixel
    activated_click = Signal()                      # bare left-click (no drag)
    contour_changed = Signal(list)                  # [(x,y),...] image coords

    MIN_ZOOM = 0.02
    MAX_ZOOM = 64.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap:  Optional[QPixmap]     = None
        self._array:   Optional[np.ndarray]  = None
        self._zoom     = 1.0
        self._pan_x    = 0.0
        self._pan_y    = 0.0

        # Pan interaction
        self._drag_start: Optional[QPoint]   = None
        self._drag_pan0:  tuple[float, float] = (0.0, 0.0)
        self._is_panning = False

        # Overlay renderer (set by ImageViewer)
        self._overlay: Optional[OverlayRenderer] = None

        # Snake contour input mode
        self._snake_mode:     bool                         = False
        self._polygon_mode:   bool                         = True    # vs. circle
        self._contour_pts:    list[tuple[float, float]]    = []
        self._drag_pt_idx:    Optional[int]                = None
        self._circle_center:  Optional[tuple[float, float]] = None
        self._circle_radius:  float                         = 0.0
        self._circle_dragging: bool                         = False

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setMinimumSize(80, 60)

        # Debounce auto-fit on resize
        self._fit_timer = QTimer(self, singleShot=True)
        self._fit_timer.timeout.connect(self._fit)

    # --- Pixmap / clear ---

    def set_pixmap(self, pixmap: QPixmap, array: np.ndarray) -> None:
        self._pixmap = pixmap
        self._array  = array
        if self._overlay is not None and pixmap is not None:
            self._overlay.set_image_size(pixmap.width(), pixmap.height())
        self._fit()

    def clear(self) -> None:
        """Remove the displayed image."""
        self._pixmap = None
        self._array  = None
        self._zoom   = 1.0
        self._pan_x  = 0.0
        self._pan_y  = 0.0
        self.update()

    def set_overlay(self, renderer: OverlayRenderer) -> None:
        self._overlay = renderer
        self.update()

    def refresh_overlay(self) -> None:
        """Trigger a repaint (call after updating the OverlayRenderer)."""
        self.update()

    # --- Snake contour ---

    def set_snake_mode(self, enabled: bool, polygon: bool = True) -> None:
        """Enter (or leave) snake contour input mode."""
        prev_polygon = self._polygon_mode
        self._snake_mode   = enabled
        self._polygon_mode = polygon
        # Clear the opposite contour type when switching modes
        if enabled:
            if polygon and not prev_polygon:
                # Switched from circle → polygon: clear circle
                self._circle_center = None
                self._circle_radius = 0.0
            elif not polygon and prev_polygon:
                # Switched from polygon → circle: clear polygon points
                self._contour_pts = []
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        self.update()

    def set_contour_pts(self, pts: list[tuple[float, float]]) -> None:
        self._contour_pts = list(pts)
        self.update()

    def get_contour_pts(self) -> list[tuple[float, float]]:
        return list(self._contour_pts)

    def get_circle_as_polygon(self, n: int = 64) -> list[tuple[float, float]]:
        """Sample *n* points around the current circle contour."""
        if self._circle_center is None or self._circle_radius < 1:
            return []
        cx, cy = self._circle_center
        return [
            (
                cx + self._circle_radius * math.cos(2 * math.pi * i / n),
                cy + self._circle_radius * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]

    def clear_contour(self) -> None:
        self._contour_pts    = []
        self._circle_center  = None
        self._circle_radius  = 0.0
        self.update()

    # --- Zoom / pan helpers ---

    def _fit(self) -> None:
        """Auto-fit image to widget, preserving aspect ratio."""
        if self._pixmap is None or self._pixmap.isNull():
            return
        w, h   = self.width(), self.height()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if w < 1 or h < 1 or pw < 1 or ph < 1:
            return
        self._zoom  = min(w / pw, h / ph)
        self._pan_x = (w - pw * self._zoom) / 2
        self._pan_y = (h - ph * self._zoom) / 2
        self.update()

    def zoom(self)    -> float: return self._zoom
    def h_ratio(self) -> float:
        if not self._pixmap or self._pixmap.width() == 0: return 0.0
        return self._pan_x / (self._pixmap.width() * self._zoom)
    def v_ratio(self) -> float:
        if not self._pixmap or self._pixmap.height() == 0: return 0.0
        return self._pan_y / (self._pixmap.height() * self._zoom)

    def sync_view(self, zoom: float, h_ratio: float, v_ratio: float) -> None:
        """Sync view to another viewer's zoom and pan ratios."""
        self._zoom = zoom
        if self._pixmap:
            self._pan_x = h_ratio * self._pixmap.width()  * zoom
            self._pan_y = v_ratio * self._pixmap.height() * zoom
        self.update()

    def _w2i(self, wx: float, wy: float) -> tuple[float, float]:
        """Widget coordinates → image coordinates."""
        return (wx - self._pan_x) / self._zoom, (wy - self._pan_y) / self._zoom

    # --- Paint ---

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#181825"))

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor("#6c7086"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "Drop an image here\nor click  Open…"
            )
            painter.end()
            return

        # Enter image-coordinate space
        painter.save()
        painter.translate(self._pan_x, self._pan_y)
        painter.scale(self._zoom, self._zoom)

        # 1. Base image
        painter.drawPixmap(
            QRectF(0, 0, self._pixmap.width(), self._pixmap.height()),
            self._pixmap,
            QRectF(self._pixmap.rect()),
        )

        # 2. Shape overlays
        if self._overlay is not None and self._overlay.has_content():
            self._overlay.draw(painter, self._zoom)

        # 3. Snake contour input
        if self._snake_mode:
            self._paint_contour_input(painter)

        painter.restore()
        painter.end()

    def _paint_contour_input(self, painter: QPainter) -> None:
        """Draw in-progress contour (polygon points or circle preview)."""
        pw    = max(0.3, _CONTOUR_LINE_W_PX / self._zoom)
        dot_r = max(0.3, _CONTOUR_DOT_R_PX  / self._zoom)
        color = QColor("#cba6f7")   # Catppuccin mauve
        circ_c = QColor("#fab387")  # Catppuccin peach

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Circle preview
        if self._circle_center is not None and self._circle_radius > 0:
            pen = QPen(circ_c, pw, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            cx, cy = self._circle_center
            painter.drawEllipse(
                QPointF(cx, cy), self._circle_radius, self._circle_radius
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(circ_c))
            painter.drawEllipse(QPointF(cx, cy), dot_r, dot_r)

        # Polygon lines
        pts = self._contour_pts
        if len(pts) >= 2:
            painter.setPen(QPen(color, pw))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            poly = QPolygonF([QPointF(x, y) for x, y in pts])
            painter.drawPolyline(poly)

        # Point dots
        if pts:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            for i, (x, y) in enumerate(pts):
                r = dot_r * 1.4 if i == 0 else dot_r   # larger first point
                painter.drawEllipse(QPointF(x, y), r, r)

    # --- Events ---

    def resizeEvent(self, _event: QResizeEvent) -> None:  # noqa: N802
        self._fit_timer.start(80)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        dy = event.angleDelta().y()
        if dy == 0:
            return
        factor  = 1.15 if dy > 0 else 1.0 / 1.15
        mx, my  = event.position().x(), event.position().y()
        new_z   = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if new_z == self._zoom:
            return
        ratio       = new_z / self._zoom
        self._pan_x = mx - ratio * (mx - self._pan_x)
        self._pan_y = my - ratio * (my - self._pan_y)
        self._zoom  = new_z
        self.update()
        self.view_changed.emit(self._zoom, self.h_ratio(), self.v_ratio())

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self.activated_click.emit()

        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = event.pos()

        if self._snake_mode:
            if self._polygon_mode:
                hit = self._nearest_pt(pos.x(), pos.y())
                if hit is not None:
                    self._drag_pt_idx = hit
                else:
                    ix, iy = self._w2i(pos.x(), pos.y())
                    self._contour_pts.append((ix, iy))
                    self.update()
                    self.contour_changed.emit(list(self._contour_pts))
            else:
                ix, iy = self._w2i(pos.x(), pos.y())
                self._circle_center   = (ix, iy)
                self._circle_radius   = 0.0
                self._circle_dragging = True
                self.update()
        else:
            self._drag_start  = pos
            self._drag_pan0   = (self._pan_x, self._pan_y)
            self._is_panning  = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.pos()

        # Status bar pixel info
        if self._array is not None and self._pixmap:
            ix, iy = self._w2i(pos.x(), pos.y())
            ix_i, iy_i = int(ix), int(iy)
            arr = self._array
            if 0 <= iy_i < arr.shape[0] and 0 <= ix_i < arr.shape[1]:
                self.mouse_info.emit(ix_i, iy_i, arr[iy_i, ix_i])

        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return

        if self._snake_mode:
            if self._polygon_mode and self._drag_pt_idx is not None:
                ix, iy = self._w2i(pos.x(), pos.y())
                self._contour_pts[self._drag_pt_idx] = (ix, iy)
                self.update()
                self.contour_changed.emit(list(self._contour_pts))
            elif not self._polygon_mode and self._circle_dragging:
                ix, iy = self._w2i(pos.x(), pos.y())
                cx, cy = self._circle_center  # type: ignore[misc]
                self._circle_radius = math.hypot(ix - cx, iy - cy)
                self.update()
        else:
            if self._drag_start is not None:
                delta = pos - self._drag_start
                if not self._is_panning and (
                    abs(delta.x()) > 3 or abs(delta.y()) > 3
                ):
                    self._is_panning = True
                if self._is_panning:
                    self._pan_x = self._drag_pan0[0] + delta.x()
                    self._pan_y = self._drag_pan0[1] + delta.y()
                    self.update()
                    self.view_changed.emit(
                        self._zoom, self.h_ratio(), self.v_ratio()
                    )

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._snake_mode:
            if self._drag_pt_idx is not None:
                self._drag_pt_idx = None
            elif self._circle_dragging:
                self._circle_dragging = False
                pts = self.get_circle_as_polygon()
                if pts:
                    self.contour_changed.emit(pts)
        else:
            self._drag_start  = None
            self._is_panning  = False

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._snake_mode:
            self._fit()
            self.view_changed.emit(self._zoom, self.h_ratio(), self.v_ratio())

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        """Right-click in polygon mode closes the contour."""
        if self._snake_mode and self._polygon_mode and len(self._contour_pts) >= 3:
            self.contour_changed.emit(list(self._contour_pts))
        else:
            super().contextMenuEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if self._snake_mode:
            if event.key() == Qt.Key.Key_Return:
                self.contour_changed.emit(list(self._contour_pts))
            elif event.key() == Qt.Key.Key_Escape:
                self.clear_contour()
            elif (
                event.key() == Qt.Key.Key_Z
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier
                and self._polygon_mode
                and self._contour_pts
            ):
                self._contour_pts.pop()
                self.update()
                self.contour_changed.emit(list(self._contour_pts))

    # --- Helpers ---

    def _nearest_pt(self, wx: float, wy: float) -> Optional[int]:
        """Return index of contour point within _HIT_RADIUS_PX of (wx, wy)."""
        for i, (px, py) in enumerate(self._contour_pts):
            spx = px * self._zoom + self._pan_x
            spy = py * self._zoom + self._pan_y
            if math.hypot(spx - wx, spy - wy) < _HIT_RADIUS_PX:
                return i
        return None


# ===========================================================================
# ImageViewer
# ===========================================================================

class ImageViewer(QGroupBox):
    """One image window: header controls + _ImageDisplay + overlay support.

    Signals
    -------
    view_changed(zoom, h_ratio, v_ratio)
    image_loaded(viewer_id)
    activated(viewer_id)
    request_open(viewer_id)
    request_save(viewer_id)
    request_clear(viewer_id)
    mode_changed_sig(viewer_id, mode_name)
    mouse_info(viewer_id, x, y, pixel)
    file_dropped(viewer_id, path)
    contour_changed(viewer_id, pts)
    copy_to(src_id, dst_id)
    swap_with(id_a, id_b)
    """

    view_changed     = Signal(float, float, float)
    image_loaded     = Signal(int)
    activated        = Signal(int)
    request_open     = Signal(int)
    request_save     = Signal(int)
    request_clear    = Signal(int)
    mode_changed_sig = Signal(int, str)
    mouse_info       = Signal(int, int, int, object)
    file_dropped     = Signal(int, str)
    contour_changed  = Signal(int, list)
    copy_to            = Signal(int, int)
    swap_with          = Signal(int, int)
    send_result_here   = Signal(int)
    maximize_requested = Signal(int)

    def __init__(
        self,
        title: str,
        viewer_id: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._id         = viewer_id
        self._title_base = title
        self._source:          Optional[np.ndarray] = None
        self._current:         Optional[np.ndarray] = None
        self._image_data:      Optional[ImageData]  = None
        self._undo_source:     Optional[np.ndarray] = None
        self._undo_current:    Optional[np.ndarray] = None
        self._undo_image_data: Optional[ImageData]  = None
        self._history:    list[str]             = []
        self._is_active   = False
        self._role        = ""
        self._overlay     = OverlayRenderer()
        self._setup_ui()
        self._update_frame()

    # --- UI construction ---

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # ── Header row ──────────────────────────────────────────────────────
        hdr  = QWidget()
        hl   = QHBoxLayout(hdr)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)

        self._lbl_title = QLabel(self._title_base)
        self._lbl_title.setStyleSheet(
            "font-weight:bold;font-size:11px;color:#cdd6f4;"
        )
        hdr.mouseDoubleClickEvent = lambda ev: self.maximize_requested.emit(self._id) if ev.button() == Qt.MouseButton.LeftButton else None
        self._lbl_title.mouseDoubleClickEvent = lambda ev: self.maximize_requested.emit(self._id) if ev.button() == Qt.MouseButton.LeftButton else None
        hl.addWidget(self._lbl_title, 1)

        self._badge = QLabel()
        self._badge.setStyleSheet(
            "font-size:10px;padding:1px 6px;border-radius:3px;font-weight:bold;"
        )
        self._badge.hide()
        hl.addWidget(self._badge)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(MODES)
        self._mode_combo.setFixedWidth(96)
        self._mode_combo.setToolTip("Display mode for this window")
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        hl.addWidget(self._mode_combo)

        self._btn_open  = QPushButton("Open…")
        self._btn_open.setToolTip("Open image in this window")
        self._btn_open.clicked.connect(lambda: self.request_open.emit(self._id))
        hl.addWidget(self._btn_open)

        self._btn_save  = QPushButton("Save…")
        self._btn_save.setToolTip("Save this window's image")
        self._btn_save.clicked.connect(lambda: self.request_save.emit(self._id))
        hl.addWidget(self._btn_save)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setToolTip("Clear this window")
        self._btn_clear.clicked.connect(lambda: self.request_clear.emit(self._id))
        hl.addWidget(self._btn_clear)

        # Apply dedicated non-clipping styling & generous minimum widths
        for btn, min_w in [(self._btn_open, 62), (self._btn_save, 58), (self._btn_clear, 52)]:
            btn.setMinimumWidth(min_w)
            btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            btn.setStyleSheet(
                "QPushButton {"
                "  background-color: #313244;"
                "  color: #cdd6f4;"
                "  border: 1px solid #45475a;"
                "  padding: 2px 6px;"
                "  font-size: 11px;"
                "  border-radius: 4px;"
                "  text-align: center;"
                "}"
                "QPushButton:hover {"
                "  background-color: #45475a;"
                "  border-color: #585b70;"
                "}"
                "QPushButton:pressed {"
                "  background-color: #585b70;"
                "}"
            )

        outer.addWidget(hdr)

        # ── Display widget ────────────────────────────────────────────────────
        self._display = _ImageDisplay()
        self._display.set_overlay(self._overlay)
        self._display.view_changed.connect(self._on_view_changed)
        self._display.mouse_info.connect(
            lambda x, y, px: self.mouse_info.emit(self._id, x, y, px)
        )
        self._display.activated_click.connect(
            lambda: self.activated.emit(self._id)
        )
        self._display.contour_changed.connect(
            lambda pts: self.contour_changed.emit(self._id, pts)
        )
        outer.addWidget(self._display, 1)

        self.setAcceptDrops(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._ctx_menu)

    # --- Style ---

    def _update_frame(self) -> None:
        """Redraw border based on active state."""
        colour = "#89b4fa" if self._is_active else "#45475a"
        width  = 2        if self._is_active else 1
        self.setStyleSheet(
            f"ImageViewer{{border:{width}px solid {colour};border-radius:8px;}}"
        )

    # --- Mode events ---

    def _on_mode_changed(self, mode: str) -> None:
        self.apply_mode()
        self.mode_changed_sig.emit(self._id, mode)

    # --- View events ---

    def _on_view_changed(self, zoom: float, h: float, v: float) -> None:
        self._update_title()
        self.view_changed.emit(zoom, h, v)

    def _update_title(self) -> None:
        arr = self._current
        if arr is not None:
            mode_str = "Grayscale" if arr.ndim == 2 else "Color RGB"
            h, w = arr.shape[:2]
            zoom_pct = int(self._display.zoom() * 100)
            self._lbl_title.setText(
                f"{self._title_base} · {mode_str} · {w}×{h} · {zoom_pct}%"
            )
        else:
            self._lbl_title.setText(self._title_base)

    # =========================================================================
    # Public API
    # =========================================================================

    @property
    def viewer_id(self) -> int:
        return self._id

    def set_title_base(self, title: str) -> None:
        self._title_base = title
        self._update_title()

    def set_image(self, array_or_data: np.ndarray | ImageData) -> None:
        """Display *array_or_data* without modifying stored source."""
        if isinstance(array_or_data, ImageData):
            self._image_data = array_or_data
            arr = array_or_data.as_mode(self._mode_combo.currentText())
        else:
            arr = array_or_data
            if self._image_data is None or self._image_data.array is not arr:
                self._image_data = ImageData.from_array(arr)

        self._current = arr
        px = array_to_qpixmap(arr)
        h  = arr.shape[0] if arr.ndim >= 1 else 0
        w  = arr.shape[1] if arr.ndim >= 2 else 0
        self._overlay.set_image_size(w, h)
        self._display.set_pixmap(px, arr)
        self._update_title()
        self.image_loaded.emit(self._id)

    def set_source_and_display(
        self, array_or_data: np.ndarray | ImageData, op_name: str = "Load"
    ) -> None:
        """Set a new source and display it in the current mode."""
        self._undo_source     = self._source
        self._undo_current    = self._current
        self._undo_image_data = self._image_data
        if isinstance(array_or_data, ImageData):
            self._image_data = array_or_data
            self._source     = array_or_data.array
        else:
            self._image_data = ImageData.from_array(array_or_data)
            self._source     = array_or_data

        self._history.append(op_name)
        self.apply_mode()

    def apply_mode(self) -> None:
        """Re-convert the stored source with the current mode combo settings."""
        if self._source is None:
            return
        mode = self._mode_combo.currentText()
        if self._image_data is not None:
            arr = self._image_data.as_mode(mode)
        else:
            arr = convert_mode(self._source, mode)
        self._current = arr
        
        # --- FIX: Ensure overlay knows the image width & height for line clipping ---
        h = arr.shape[0] if arr.ndim >= 1 else 0
        w = arr.shape[1] if arr.ndim >= 2 else 0
        self._overlay.set_image_size(w, h)
        # ----------------------------------------------------------------------------
        
        px = array_to_qpixmap(arr)
        self._display.set_pixmap(px, arr)
        self._update_title()
        self.image_loaded.emit(self._id)

    def set_mode(self, mode: str) -> None:
        """Set mode combo text and update display without re-emitting signal."""
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentText(mode)
        self._mode_combo.blockSignals(False)
        self.apply_mode()

    def clear(self) -> None:
        """Clear all image data and overlays."""
        self._source          = None
        self._current         = None
        self._image_data      = None
        self._undo_source     = None
        self._undo_current    = None
        self._undo_image_data = None
        self._history         = []
        self._overlay.clear()
        self._display.clear()
        self._update_title()
        self.image_loaded.emit(self._id)

    def undo(self) -> bool:
        """Restore the previous source/current.  Returns True if undone."""
        if self._undo_source is None and self._undo_current is None:
            return False
        self._source          = self._undo_source
        self._current         = self._undo_current
        self._image_data      = self._undo_image_data
        self._undo_source     = None
        self._undo_current    = None
        self._undo_image_data = None
        if self._history:
            self._history.pop()
        if self._current is not None:
            px = array_to_qpixmap(self._current)
            self._display.set_pixmap(px, self._current)
        else:
            self._display.clear()
        self._update_title()
        self.image_loaded.emit(self._id)
        return True

    # --- Accessors ---

    def source(self)      -> Optional[np.ndarray]: return self._source
    def current(self)     -> Optional[np.ndarray]: return self._current
    def image_data(self)  -> Optional[ImageData]:  return self._image_data
    def has_image(self)   -> bool: return self._current is not None
    def history(self)     -> list[str]: return list(self._history)
    def overlay(self)     -> OverlayRenderer: return self._overlay
    def display_widget(self) -> _ImageDisplay: return self._display
    def mode_combo(self)  -> QComboBox: return self._mode_combo
    def current_mode(self) -> str: return self._mode_combo.currentText()

    def set_active(self, active: bool) -> None:
        self._is_active = active
        self._update_frame()

    def set_role(self, role: str) -> None:
        """Set the INPUT / OUTPUT / BOTH badge (or empty string to hide it)."""
        self._role = role
        _ROLE = {
            "INPUT":  ("#a6e3a1", "#1e1e2e"),
            "OUTPUT": ("#89b4fa", "#11111b"),
            "BOTH":   ("#f9e2af", "#1e1e2e"),
        }
        if role in _ROLE:
            bg, fg = _ROLE[role]
            self._badge.setText(role)
            self._badge.setStyleSheet(
                f"font-size:10px;padding:1px 6px;border-radius:3px;"
                f"font-weight:bold;background:{bg};color:{fg};"
            )
            self._badge.show()
        else:
            self._badge.hide()

    def sync_view(self, zoom: float, h_ratio: float, v_ratio: float) -> None:
        self._display.sync_view(zoom, h_ratio, v_ratio)

    # --- Snake contour ---

    def set_snake_mode(self, enabled: bool, polygon: bool = True) -> None:
        self._display.set_snake_mode(enabled, polygon)

    def get_contour_pts(self) -> list[tuple[float, float]]:
        return self._display.get_contour_pts()

    def get_circle_as_polygon(self) -> list[tuple[float, float]]:
        return self._display.get_circle_as_polygon()

    def clear_contour(self) -> None:
        self._display.clear_contour()

    # --- Drag-and-drop ---

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            self.activated.emit(self._id)
            self.file_dropped.emit(self._id, urls[0].toLocalFile())

    # --- Context menu ---

    def _ctx_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.addAction("Open…").triggered.connect(
            lambda: self.request_open.emit(self._id)
        )
        menu.addAction("Save…").triggered.connect(
            lambda: self.request_save.emit(self._id)
        )
        menu.addAction("Clear").triggered.connect(
            lambda: self.request_clear.emit(self._id)
        )
        menu.addSeparator()
        for n in range(1, 5):
            if n == self._id:
                continue
            a = menu.addAction(f"Copy to Window {n}")
            a.triggered.connect(
                lambda _=False, dst=n: self.copy_to.emit(self._id, dst)
            )
        menu.addSeparator()
        for n in range(1, 5):
            if n == self._id:
                continue
            a = menu.addAction(f"Swap with Window {n}")
            a.triggered.connect(
                lambda _=False, dst=n: self.swap_with.emit(self._id, dst)
            )
        menu.addSeparator()
        menu.addAction("Use as Input").triggered.connect(
            lambda: self.activated.emit(self._id)
        )
        menu.addAction("Send Result Here").triggered.connect(
            lambda: self.send_result_here.emit(self._id)
        )
        menu.exec(self.mapToGlobal(pos))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self.activated.emit(self._id)
        super().mousePressEvent(event)

