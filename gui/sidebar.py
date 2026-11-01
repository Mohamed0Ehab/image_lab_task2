"""Sidebar panel: operation picker, parameters, routing, actions, snake controls.

The sidebar is built entirely from CATALOG.  It never touches core directly
and holds no image data.  Communication with MainWindow is via Qt signals only.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui  import QColor
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QColorDialog, QComboBox, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QToolButton,
    QVBoxLayout, QWidget,
)

from gui.features import CATALOG, FeatureEntry, get_entry
from gui.image_io import MODES
from gui.widgets  import CollapsibleBox, JumpSlider, ParamPanel

logger = logging.getLogger(__name__)

# Tool to overlay/accent color mapping
TOOL_ACCENT_COLORS: dict[str, str] = {
    "canny": "#cba6f7",          # Mauve (Edges)
    "hough_lines": "#f38ba8",    # Pink (Lines overlay color)
    "hough_circles": "#a6e3a1",  # Green (Circles overlay color)
    "detect_ellipses": "#f9e2af",# Yellow (Ellipses overlay color)
    "run_snake": "#89b4fa",      # Blue (Snake overlay color)
    "snake": "#89b4fa",          # Blue
}


# ===========================================================================
# Helper: small two-column info row
# ===========================================================================

def _info_row(lbl_text: str, value_text: str = "—") -> tuple[QWidget, QLabel]:
    """Return (row_widget, value_label) for a key/value display pair."""
    row   = QWidget()
    hl    = QHBoxLayout(row)
    hl.setContentsMargins(0, 0, 0, 0)
    key_l = QLabel(lbl_text)
    key_l.setStyleSheet("font-size:11px;color:#a6adc8;")
    key_l.setFixedWidth(80)
    val_l = QLabel(value_text)
    val_l.setStyleSheet("font-size:11px;color:#cdd6f4;")
    val_l.setWordWrap(True)
    hl.addWidget(key_l)
    hl.addWidget(val_l, 1)
    return row, val_l


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        "font-weight:bold;font-size:11px;color:#89b4fa;padding-top:4px;"
    )
    return lbl


def _make_btn(text: str, accent: bool = False) -> QPushButton:
    btn = QPushButton(text)
    if accent:
        btn.setStyleSheet(
            "QPushButton{"
            "background:#89b4fa;color:#11111b;border:none;"
            "border-radius:6px;padding:8px 14px;font-weight:bold;font-size:12px;"
            "}"
            "QPushButton:hover{background:#b4d0fb;}"
            "QPushButton:pressed{background:#74a0e8;}"
            "QPushButton:disabled{background:#45475a;color:#6c7086;}"
        )
    return btn


# ===========================================================================
# Sidebar
# ===========================================================================

class Sidebar(QScrollArea):
    """Left sidebar panel.  Emits Qt signals; holds no image data."""

    # ── Signals ─────────────────────────────────────────────────────────────
    apply_requested          = Signal(str, dict, int, int, int)
    # (op_id, params, input_a_id, input_b_id, output_id)
    # output_id == 0  means "same as input"

    routing_changed          = Signal()
    mode_changed             = Signal(str)        # "Color RGB" | "Grayscale"
    open_image_requested     = Signal(int, str)   # window_id, "Read as" mode
    use_output_as_input      = Signal()
    reset_to_source          = Signal()
    clear_all_requested      = Signal()
    undo_requested           = Signal()
    save_output_requested    = Signal()
    swap_windows_requested   = Signal(int, int)   # win_a, win_b

    op_selected              = Signal(str)         # op_id, "" = none
    snake_mode_requested     = Signal(bool, bool)  # enabled, polygon_mode
    clear_contour_requested  = Signal()
    undo_contour_point_requested = Signal()        # undo last polygon point

    animation_play           = Signal()
    animation_pause          = Signal()
    animation_step           = Signal()
    animation_step_back      = Signal()
    animation_speed_changed  = Signal(int)         # ms per frame

    overlay_lines_toggled    = Signal(bool)
    overlay_circles_toggled  = Signal(bool)
    overlay_ellipses_toggled = Signal(bool)
    overlay_snake_toggled    = Signal(bool)
    overlay_lines_color      = Signal(object)      # QColor
    overlay_circles_color    = Signal(object)
    overlay_ellipses_color   = Signal(object)
    overlay_snake_color      = Signal(object)

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_entry: Optional[FeatureEntry] = None
        self._active_window: int = 1
        self._window_has_image: list[bool] = [False, False, False, False]
        self._anim_playing = False

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMinimumWidth(340)
        self.setMaximumWidth(340)
        self.setStyleSheet("QScrollArea{border:none;}")

        body = QWidget()
        self._body_layout = QVBoxLayout(body)
        self._body_layout.setContentsMargins(8, 8, 8, 8)
        self._body_layout.setSpacing(8)
        self.setWidget(body)

        self._build_upload()
        self._build_operation_picker()
        self._build_params()
        self._build_routing()
        self._build_actions()
        self._build_snake_section()
        self._build_overlay_section()
        self._build_history()

        self._body_layout.addStretch()
        self._update_controls_enabled()

    # =========================================================================
    # Section builders
    # =========================================================================

    def _build_upload(self) -> None:
        box = CollapsibleBox("Upload")
        cl  = box.content_layout()
        row = QWidget()
        hl  = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)

        self._upload_mode = QComboBox()
        self._upload_mode.addItems(MODES)
        self._upload_mode.setToolTip("Read as (default mode for the Open dialog)")
        self._upload_mode.currentTextChanged.connect(self.mode_changed.emit)

        self._btn_upload = QPushButton("Open…")
        self._btn_upload.setToolTip("Open image into active/selected window")
        self._btn_upload.clicked.connect(self._on_open_image_clicked)

        lbl = QLabel("Read as:")
        lbl.setStyleSheet("font-size:11px;color:#a6adc8;")
        hl.addWidget(self._btn_upload, 1)
        hl.addWidget(lbl)
        hl.addWidget(self._upload_mode)
        cl.addWidget(row)
        self._body_layout.addWidget(box)

    def _build_operation_picker(self) -> None:
        box = CollapsibleBox("Operation")
        cl  = box.content_layout()

        # Category filter
        filter_row = QWidget()
        frl = QHBoxLayout(filter_row)
        frl.setContentsMargins(0, 0, 0, 0)
        frl.addWidget(QLabel("Category:"))
        self._cat_filter = QComboBox()
        self._cat_filter.addItem("All")
        for g in CATALOG:
            self._cat_filter.addItem(g)
        self._cat_filter.currentTextChanged.connect(self._filter_ops)
        frl.addWidget(self._cat_filter, 1)
        cl.addWidget(filter_row)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#45475a;")
        cl.addWidget(sep)

        # Radio buttons per group
        self._op_group    = QButtonGroup(self)
        self._op_group.setExclusive(True)
        self._op_widgets:  dict[str, list[QWidget]] = {}
        self._op_buttons:  dict[str, QRadioButton]  = {}

        for group_name, entries in CATALOG.items():
            heading = _section_label(group_name)
            group_widgets: list[QWidget] = [heading]
            cl.addWidget(heading)

            for entry in entries:
                btn = QRadioButton(entry.name)
                color = TOOL_ACCENT_COLORS.get(entry.id, "#89b4fa")
                btn.setStyleSheet(f"""
                    QRadioButton {{
                        color: #cdd6f4;
                        font-size: 12px;
                        padding: 3px 0;
                        spacing: 8px;
                    }}
                    QRadioButton::indicator {{
                        width: 14px;
                        height: 14px;
                        border-radius: 8px;
                        border: 2px solid #6c7086;
                        background-color: #1e1e2e;
                    }}
                    QRadioButton::indicator:hover {{
                        border-color: {color};
                    }}
                    QRadioButton::indicator:checked {{
                        border: 2px solid {color};
                        background-color: {color};
                    }}
                """)
                btn.toggled.connect(
                    lambda checked, e=entry: self._on_op_toggled(e, checked)
                )
                self._op_group.addButton(btn)
                self._op_buttons[entry.id] = btn
                group_widgets.append(btn)
                cl.addWidget(btn)

            self._op_widgets[group_name] = group_widgets

        self._body_layout.addWidget(box)

    def _build_params(self) -> None:
        self._param_panel = ParamPanel("Parameters")
        self._param_panel.set_schema({})
        self._live_chk = QCheckBox("Live preview (re-apply on change)")
        self._live_chk.setStyleSheet("font-size:11px;")
        self._live_chk.stateChanged.connect(self._on_live_changed)
        self._param_panel.changed.connect(self._on_param_changed)

        box = CollapsibleBox("Parameters")
        box.content_layout().addWidget(self._param_panel)
        box.content_layout().addWidget(self._live_chk)
        self._body_layout.addWidget(box)

    def _build_routing(self) -> None:
        grp = QGroupBox("Routing")
        gl  = QVBoxLayout(grp)
        gl.setContentsMargins(8, 20, 8, 8)
        gl.setSpacing(6)

        def _win_combo() -> QComboBox:
            c = QComboBox()
            for n in range(1, 5):
                c.addItem(f"Window {n}")
            return c

        # Input A
        row_a = QWidget(); hl_a = QHBoxLayout(row_a)
        hl_a.setContentsMargins(0, 0, 0, 0)
        self._lbl_input_a = QLabel("Input:")
        self._lbl_input_a.setFixedWidth(60)
        self._input_a_combo = _win_combo()
        self._input_a_combo.setCurrentIndex(0)
        hl_a.addWidget(self._lbl_input_a)
        hl_a.addWidget(self._input_a_combo, 1)
        gl.addWidget(row_a)

        # Input B (shown only when needs_second_image)
        self._input_b_row = QWidget()
        hl_b = QHBoxLayout(self._input_b_row)
        hl_b.setContentsMargins(0, 0, 0, 0)
        lbl_b = QLabel("Input B:")
        lbl_b.setFixedWidth(60)
        self._input_b_combo = _win_combo()
        self._input_b_combo.setCurrentIndex(1)
        hl_b.addWidget(lbl_b)
        hl_b.addWidget(self._input_b_combo, 1)
        self._input_b_row.hide()
        gl.addWidget(self._input_b_row)

        # Output
        row_o = QWidget(); hl_o = QHBoxLayout(row_o)
        hl_o.setContentsMargins(0, 0, 0, 0)
        lbl_o = QLabel("Output:")
        lbl_o.setFixedWidth(60)
        self._output_combo = _win_combo()
        self._output_combo.insertItem(0, "Same as input")
        self._output_combo.setCurrentIndex(2)   # Window 2 default
        hl_o.addWidget(lbl_o)
        hl_o.addWidget(self._output_combo, 1)
        gl.addWidget(row_o)

        # Connect routing combos → update role badges in MainWindow
        self._input_a_combo.currentIndexChanged.connect(self._on_routing_changed)
        self._input_b_combo.currentIndexChanged.connect(self._on_routing_changed)
        self._output_combo.currentIndexChanged.connect(self._on_routing_changed)

        self._body_layout.addWidget(grp)

    def _build_actions(self) -> None:
        box = CollapsibleBox("Actions")
        cl  = box.content_layout()

        self._btn_apply = _make_btn("▶  Apply", accent=True)
        self._btn_apply.clicked.connect(self._on_apply)
        cl.addWidget(self._btn_apply)

        # Row: use-output
        self._btn_use_output = QPushButton("Use Output as Input")
        self._btn_use_output.setToolTip("Ctrl+U")
        self._btn_use_output.clicked.connect(self.use_output_as_input.emit)
        cl.addWidget(self._btn_use_output)

        # Row: reset / clear all
        row1 = QWidget(); hl1 = QHBoxLayout(row1)
        hl1.setContentsMargins(0, 0, 0, 0); hl1.setSpacing(4)
        self._btn_reset = QPushButton("Reset to Source")
        self._btn_reset.setToolTip("Ctrl+R")
        self._btn_reset.clicked.connect(self.reset_to_source.emit)
        self._btn_clear_all = QPushButton("Clear All")
        self._btn_clear_all.setToolTip("Clear all windows")
        self._btn_clear_all.clicked.connect(self.clear_all_requested.emit)
        hl1.addWidget(self._btn_reset, 1)
        hl1.addWidget(self._btn_clear_all, 1)
        cl.addWidget(row1)

        # Row: undo / save
        row2 = QWidget(); hl2 = QHBoxLayout(row2)
        hl2.setContentsMargins(0, 0, 0, 0); hl2.setSpacing(4)
        self._btn_undo = QPushButton("Undo")
        self._btn_undo.setToolTip("Ctrl+Z")
        self._btn_undo.clicked.connect(self.undo_requested.emit)
        self._btn_save_out = QPushButton("Save…")
        self._btn_save_out.setToolTip("Save output image (Ctrl+S)")
        self._btn_save_out.clicked.connect(self.save_output_requested.emit)
        hl2.addWidget(self._btn_undo, 1)
        hl2.addWidget(self._btn_save_out, 1)
        cl.addWidget(row2)

        # Swap
        swap_row = QWidget(); swl = QHBoxLayout(swap_row)
        swl.setContentsMargins(0, 0, 0, 0); swl.setSpacing(4)
        lbl_sw = QLabel("Swap:")
        lbl_sw.setFixedWidth(38)
        self._swap_a = QComboBox()
        self._swap_b = QComboBox()
        for c in (self._swap_a, self._swap_b):
            for n in range(1, 5):
                c.addItem(f"Win {n}")
        self._swap_b.setCurrentIndex(1)
        self._btn_swap = QPushButton("Swap")
        self._btn_swap.setFixedWidth(48)
        self._btn_swap.clicked.connect(self._on_swap)
        swl.addWidget(lbl_sw)
        swl.addWidget(self._swap_a, 1)
        swl.addWidget(QLabel("↔"))
        swl.addWidget(self._swap_b, 1)
        swl.addWidget(self._btn_swap)
        cl.addWidget(swap_row)

        self._body_layout.addWidget(box)

    def _build_snake_section(self) -> None:
        self._snake_box = CollapsibleBox("Active Contour (Snake)")

        cl = self._snake_box.content_layout()

        # ── Contour input ───────────────────────────────────────────────────
        cl.addWidget(_section_label("Initial contour"))

        mode_row = QWidget(); mrl = QHBoxLayout(mode_row)
        mrl.setContentsMargins(0, 0, 0, 0); mrl.setSpacing(6)
        self._snake_poly_btn   = QRadioButton("Click polygon")
        self._snake_circle_btn = QRadioButton("Drag circle")
        self._snake_poly_btn.setChecked(True)

        snake_radio_style = """
            QRadioButton {
                color: #cdd6f4;
                font-size: 11px;
                padding: 2px 0;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 13px;
                height: 13px;
                border-radius: 7px;
                border: 2px solid #6c7086;
                background-color: #1e1e2e;
            }
            QRadioButton::indicator:hover {
                border-color: #89b4fa;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #89b4fa;
                background-color: #89b4fa;
            }
        """
        self._snake_poly_btn.setStyleSheet(snake_radio_style)
        self._snake_circle_btn.setStyleSheet(snake_radio_style)

        self._snake_mode_group = QButtonGroup(self)
        self._snake_mode_group.addButton(self._snake_poly_btn)
        self._snake_mode_group.addButton(self._snake_circle_btn)
        self._snake_poly_btn.toggled.connect(self._on_snake_contour_mode)
        self._snake_circle_btn.toggled.connect(self._on_snake_contour_mode)
        mrl.addWidget(self._snake_poly_btn)
        mrl.addWidget(self._snake_circle_btn)
        cl.addWidget(mode_row)

        self._btn_clear_contour = QPushButton("Clear Contour")
        self._btn_clear_contour.clicked.connect(self.clear_contour_requested.emit)
        cl.addWidget(self._btn_clear_contour)

        self._btn_undo_point = QPushButton("↩ Undo Point  (Ctrl+Z)")
        self._btn_undo_point.setToolTip(
            "Remove the last placed polygon point.\n"
            "You can also press Ctrl+Z while the image is focused."
        )
        self._btn_undo_point.clicked.connect(self._on_undo_contour_point)
        cl.addWidget(self._btn_undo_point)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("color:#45475a;"); cl.addWidget(sep1)

        # ── Results ─────────────────────────────────────────────────────────
        cl.addWidget(_section_label("Results"))

        row_chain, self._snake_chain = _info_row("Chain code:")
        row_perim, self._snake_perim = _info_row("Perimeter:")
        row_area,  self._snake_area  = _info_row("Area:")
        row_npts,  self._snake_npts  = _info_row("Points:")

        cl.addWidget(row_chain)
        cl.addWidget(row_perim)
        cl.addWidget(row_area)
        cl.addWidget(row_npts)

        self._btn_copy_result = QPushButton("Copy to clipboard")
        self._btn_copy_result.clicked.connect(self._copy_snake_result)
        cl.addWidget(self._btn_copy_result)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#45475a;"); cl.addWidget(sep2)

        # ── Animation ───────────────────────────────────────────────────────
        cl.addWidget(_section_label("Animation"))

        anim_row = QWidget(); arl = QHBoxLayout(anim_row)
        arl.setContentsMargins(0, 0, 0, 0); arl.setSpacing(4)
        self._btn_play_pause = QPushButton("▶ Play")
        self._btn_play_pause.setCheckable(True)
        self._btn_step_back = QPushButton("⏮ Back")
        self._btn_step_back.setToolTip("Step back one frame")
        self._btn_step  = QPushButton("⏭ Fwd")
        self._btn_step.setToolTip("Step forward one frame")
        for b in (self._btn_play_pause, self._btn_step_back, self._btn_step):
            b.setFixedHeight(28)
        self._btn_play_pause.clicked.connect(self._on_play_pause_toggle)
        self._btn_step_back.clicked.connect(self.animation_step_back.emit)
        self._btn_step.clicked.connect(self.animation_step.emit)
        arl.addWidget(self._btn_play_pause)
        arl.addWidget(self._btn_step_back)
        arl.addWidget(self._btn_step)
        cl.addWidget(anim_row)

        speed_row = QWidget(); srl = QHBoxLayout(speed_row)
        srl.setContentsMargins(0, 0, 0, 0); srl.setSpacing(6)
        srl.addWidget(QLabel("Speed:"))
        self._speed_slider = JumpSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(10, 500)
        self._speed_slider.setValue(100)
        self._speed_slider.setToolTip("ms per animation frame")
        self._speed_slider.valueChanged.connect(
            lambda v: self.animation_speed_changed.emit(v)
        )
        srl.addWidget(self._speed_slider, 1)
        self._speed_val = QLabel("100 ms")
        self._speed_val.setFixedWidth(48)
        self._speed_slider.valueChanged.connect(
            lambda v: self._speed_val.setText(f"{v} ms")
        )
        srl.addWidget(self._speed_val)
        cl.addWidget(speed_row)

        self._btn_cancel_snake = QPushButton("Cancel")
        self._btn_cancel_snake.setStyleSheet(
            "QPushButton{background:#f38ba8;color:#1e1e2e;}"
            "QPushButton:hover{background:#f5a9b8;}"
        )
        self._btn_cancel_snake.hide()
        cl.addWidget(self._btn_cancel_snake)

        # Disabled until a snake result arrives
        for w in (
            self._btn_play_pause, self._btn_step_back, self._btn_step,
            self._speed_slider, self._btn_copy_result,
        ):
            w.setEnabled(False)

        self._snake_box.hide()   # shown only when snake op is selected
        self._body_layout.addWidget(self._snake_box)

    def _build_overlay_section(self) -> None:
        box = CollapsibleBox("Overlay Visibility")
        cl  = box.content_layout()

        def _ov_row(label, checked, toggle_sig, color_sig, default_color):
            row = QWidget(); hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(4)
            chk = QCheckBox(label)
            chk.setChecked(checked)
            chk.stateChanged.connect(lambda s: toggle_sig.emit(bool(s)))
            color_btn = QToolButton()
            color_btn.setFixedSize(22, 22)
            color_btn.setStyleSheet(
                f"background:{default_color};border-radius:4px;border:1px solid #45475a;"
            )
            color_btn.clicked.connect(
                lambda _=False, sig=color_sig, btn=color_btn, dc=default_color:
                    self._pick_color(sig, btn, dc)
            )
            hl.addWidget(chk, 1)
            hl.addWidget(color_btn)
            cl.addWidget(row)
            return chk, color_btn

        self._ov_lines_chk,    self._ov_lines_btn    = _ov_row(
            "Lines",    True, self.overlay_lines_toggled,
            self.overlay_lines_color,    "#f38ba8"
        )
        self._ov_circles_chk,  self._ov_circles_btn  = _ov_row(
            "Circles",  True, self.overlay_circles_toggled,
            self.overlay_circles_color,  "#a6e3a1"
        )
        self._ov_ellipses_chk, self._ov_ellipses_btn = _ov_row(
            "Ellipses", True, self.overlay_ellipses_toggled,
            self.overlay_ellipses_color, "#f9e2af"
        )
        self._ov_snake_chk,    self._ov_snake_btn    = _ov_row(
            "Snake",    True, self.overlay_snake_toggled,
            self.overlay_snake_color,    "#89b4fa"
        )

        self._body_layout.addWidget(box)

    def _build_history(self) -> None:
        grp = QGroupBox("History (active window)")
        gl  = QVBoxLayout(grp)
        gl.setContentsMargins(8, 20, 8, 8)
        self._history_label = QLabel("—")
        self._history_label.setStyleSheet(
            "font-size:11px;color:#a6adc8;padding:2px;"
        )
        self._history_label.setWordWrap(True)
        gl.addWidget(self._history_label)
        self._body_layout.addWidget(grp)

    # =========================================================================
    # Event handlers
    # =========================================================================

    def _on_op_toggled(self, entry: FeatureEntry, checked: bool) -> None:
        if not checked:
            return
        self._current_entry = entry
        self._param_panel.set_schema(entry.params_schema)
        is_snake = (entry.kind == "snake")
        self._snake_box.setVisible(is_snake)
        self._input_b_row.setVisible(entry.needs_second_image)
        self.op_selected.emit(entry.id)
        if is_snake:
            polygon = self._snake_poly_btn.isChecked()
            self.snake_mode_requested.emit(True, polygon)
        else:
            self.snake_mode_requested.emit(False, True)

    def _on_snake_contour_mode(self, _checked: bool) -> None:
        polygon = self._snake_poly_btn.isChecked()
        self.snake_mode_requested.emit(True, polygon)

    def _on_open_image_clicked(self) -> None:
        mode = self._upload_mode.currentText()
        self.open_image_requested.emit(self._active_window, mode)

    def _on_apply(self) -> None:
        if self._current_entry is None:
            return
        params   = self._param_panel.values()
        input_a  = self._input_a_combo.currentIndex() + 1       # 1-based
        input_b  = self._input_b_combo.currentIndex() + 1
        out_idx  = self._output_combo.currentIndex()
        output   = 0 if out_idx == 0 else out_idx               # 0 = same as input
        self.apply_requested.emit(
            self._current_entry.id, params, input_a, input_b, output
        )

    def _on_live_changed(self) -> None:
        if self._live_chk.isChecked() and self._current_entry is not None:
            self._on_apply()

    def _on_param_changed(self) -> None:
        if self._live_chk.isChecked() and self._current_entry is not None:
            self._on_apply()

    def _on_routing_changed(self) -> None:
        self.routing_changed.emit()

    def _on_swap(self) -> None:
        a = self._swap_a.currentIndex() + 1
        b = self._swap_b.currentIndex() + 1
        if a != b:
            self.swap_windows_requested.emit(a, b)

    def _on_play_pause_toggle(self, checked: bool) -> None:
        if checked:
            self._anim_playing = True
            self._btn_play_pause.setText("⏸ Pause")
            self.animation_play.emit()
        else:
            self._anim_playing = False
            self._btn_play_pause.setText("▶ Play")
            self.animation_pause.emit()

    def _on_undo_contour_point(self) -> None:
        """Signal the display to remove the last polygon point."""
        self.undo_contour_point_requested.emit()

    def _copy_snake_result(self) -> None:
        from PySide6.QtWidgets import QApplication
        text = (
            f"Chain:     {self._snake_chain.text()}\n"
            f"Perimeter: {self._snake_perim.text()}\n"
            f"Area:      {self._snake_area.text()}\n"
            f"Points:    {self._snake_npts.text()}"
        )
        QApplication.clipboard().setText(text)

    def _filter_ops(self, cat: str) -> None:
        for group_name, widgets in self._op_widgets.items():
            visible = (cat == "All" or cat == group_name)
            for w in widgets:
                w.setVisible(visible)

    def _pick_color(self, signal, btn: QToolButton, default_hex: str) -> None:
        color = QColorDialog.getColor(QColor(default_hex), self, "Choose colour")
        if color.isValid():
            btn.setStyleSheet(
                f"background:{color.name()};border-radius:4px;border:1px solid #45475a;"
            )
            signal.emit(color)

    # =========================================================================
    # Public API called by MainWindow
    # =========================================================================

    def set_active_window(self, window_id: int) -> None:
        """Update default routing to use *window_id* as input."""
        self._active_window = window_id
        idx = window_id - 1
        self._input_a_combo.setCurrentIndex(idx)
        # Output defaults to next window
        out = (window_id % 4) + 1
        # find in output combo: index 0 = "Same as input", 1-4 = windows
        self._output_combo.setCurrentIndex(out)

    def set_input_window(self, window_id: int) -> None:
        """Set input routing to *window_id* (1..4)."""
        if 1 <= window_id <= 4:
            self._input_a_combo.setCurrentIndex(window_id - 1)

    def set_output_window(self, window_id: int) -> None:
        """Set output routing to *window_id* (0 = same as input, 1..4 = Window 1..4)."""
        if 0 <= window_id <= 4:
            self._output_combo.setCurrentIndex(window_id)

    def set_upload_mode(self, mode: str) -> None:
        """Update upload mode combo text without triggering signal loop."""
        self._upload_mode.blockSignals(True)
        self._upload_mode.setCurrentText(mode)
        self._upload_mode.blockSignals(False)

    def set_window_has_image(self, window_id: int, has: bool) -> None:
        """Track which windows have images; enable controls when any do."""
        self._window_has_image[window_id - 1] = has
        self._update_controls_enabled()

    def update_snake_result(self, result) -> None:
        """Populate the snake results box from a SnakeResult."""
        self._snake_chain.setText(str(result.chain)[:60] or "—")
        perim_cm, perim_px = result.perimeter
        area_cm, area_px = result.area
        self._snake_perim.setText(f"{perim_px:.1f} px  /  {perim_cm:.2f} cm")
        self._snake_area.setText(f"{area_px:.1f} px²  /  {area_cm:.2f} cm²")
        self._snake_npts.setText(str(len(result.pts)))
        has_hist = bool(result.history)
        self._btn_play_pause.setEnabled(has_hist)
        self._btn_step_back.setEnabled(has_hist)
        self._btn_step.setEnabled(has_hist)
        self._speed_slider.setEnabled(has_hist)
        self._btn_copy_result.setEnabled(True)

    def clear_snake_result(self) -> None:
        """Reset the snake results box to placeholder dashes."""
        for lbl in (
            self._snake_chain, self._snake_perim,
            self._snake_area, self._snake_npts,
        ):
            lbl.setText("—")
        # Reset play/pause button to its initial play state
        self._btn_play_pause.setChecked(False)
        self._btn_play_pause.setText("▶ Play")
        self._anim_playing = False
        for w in (
            self._btn_play_pause, self._btn_step_back, self._btn_step,
            self._speed_slider, self._btn_copy_result,
        ):
            w.setEnabled(False)

    def show_cancel_snake(self, visible: bool) -> None:
        """Show or hide the Cancel button during a snake run."""
        self._btn_cancel_snake.setVisible(visible)

    def update_history(self, history: list[str]) -> None:
        """Show the active window's operation history as a breadcrumb."""
        if not history:
            self._history_label.setText("—")
        else:
            self._history_label.setText(" → ".join(history))

    def routing_ids(self) -> tuple[int, int, int]:
        """Return (input_a, input_b, output) window IDs (0 = same as input)."""
        input_a = self._input_a_combo.currentIndex() + 1
        input_b = self._input_b_combo.currentIndex() + 1
        out_idx = self._output_combo.currentIndex()
        output  = 0 if out_idx == 0 else out_idx
        return input_a, input_b, output

    def cancel_snake_button(self) -> QPushButton:
        return self._btn_cancel_snake

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _update_controls_enabled(self) -> None:
        any_image = any(self._window_has_image)
        for w in (
            self._btn_apply, self._btn_use_output, self._btn_reset,
            self._btn_clear_all, self._btn_undo, self._btn_save_out,
            self._btn_swap,
        ):
            w.setEnabled(any_image)
