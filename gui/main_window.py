"""Main window for ImageLab – Shapes.

Features
--------
- 2x2 resizable grid of ImageViewer windows with splitter resizing.
- Layout switcher: 2x2 grid, 1x4 horizontal, 4x1 vertical.
- Window double-click maximize/restore.
- Catalog-driven sidebar panel with free I/O routing.
- Live role badges (INPUT / OUTPUT / BOTH / none).
- Centralised error handling via run_safely.
- Background QThread worker for Snake algorithm with live animation player.
- Linked zoom & pan across windows.
- Full status bar (active window, mouse coordinates, pixel value, zoom level).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QSplitter,
    QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from core.snake import run_snake
from core.types import SnakeParams, SnakeResult
from gui.error_handler import AppError, run_safely, show_error
from gui.features import FeatureEntry, FeatureOutput, get_entry, run_feature
from gui.image_io import MODES, ImageData, ensure_gray, load_image, save_image
from gui.sidebar import Sidebar
from gui.widgets import ImageViewer

logger = logging.getLogger(__name__)


# ===========================================================================
# Background worker for active contour (Snake)
# ===========================================================================

class SnakeWorker(QThread):
    """Background worker for run_snake so the UI never freezes."""

    result_ready = Signal(object)      # SnakeResult
    error_occurred = Signal(Exception)

    def __init__(
        self,
        gray: np.ndarray,
        pts: list[tuple[float, float]],
        params: SnakeParams,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._gray = gray
        self._pts = pts
        self._params = params

    def run(self) -> None:
        try:
            # Calls core stub (which raises NotImplementedError)
            res = run_snake(self._gray, self._pts, self._params)
            self.result_ready.emit(res)
        except Exception as exc:
            self.error_occurred.emit(exc)


# ===========================================================================
# MainWindow
# ===========================================================================

class MainWindow(QMainWindow):
    """Primary application window for ImageLab – Shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ImageLab – Shapes")
        self.resize(1340, 860)
        self.setMinimumSize(960, 600)

        self._active_window_id: int = 1
        self._maximized_window_id: Optional[int] = None
        self._layout_mode: str = "2x2"
        self._syncing_view: bool = False
        self._current_mode: str = "Color RGB"
        self._syncing_mode: bool = False

        # Snake execution & animation state
        self._snake_worker: Optional[SnakeWorker] = None
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_history: list[list[tuple[float, float]]] = []
        self._anim_frame_idx: int = 0
        self._anim_target_viewer: Optional[ImageViewer] = None

        self._setup_ui()
        self._setup_menus()
        self._setup_statusbar()
        self._connect_signals()

        # Initial UI state
        self._set_active_window(1)
        self._update_role_badges()
        self._apply_layout(self._layout_mode)

    # =========================================================================
    # UI Setup
    # =========================================================================

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # 1. Left Sidebar
        self._sidebar = Sidebar()
        main_layout.addWidget(self._sidebar)

        # 2. Right Workspace Area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # 2a. Top Workspace Control Bar
        ctrl_bar = QWidget()
        cbl = QHBoxLayout(ctrl_bar)
        cbl.setContentsMargins(4, 2, 4, 2)
        cbl.setSpacing(8)

        lbl_layout = QLabel("Layout:")
        lbl_layout.setStyleSheet("font-weight:bold;color:#a6adc8;font-size:11px;")
        cbl.addWidget(lbl_layout)

        self._layout_combo = QComboBox()
        self._layout_combo.addItems(["2x2", "1x4", "4x1"])
        self._layout_combo.currentTextChanged.connect(self._on_layout_changed)
        cbl.addWidget(self._layout_combo)

        self._chk_sync = QCheckBox("Link Zoom && Pan")
        self._chk_sync.setStyleSheet("font-size:11px;color:#cdd6f4;")
        cbl.addWidget(self._chk_sync)

        btn_fit_all = QPushButton("Fit All")
        btn_fit_all.setFixedHeight(26)
        btn_fit_all.clicked.connect(self._fit_all_windows)
        cbl.addWidget(btn_fit_all)

        cbl.addStretch(1)

        hint_lbl = QLabel("Tip: Double-click header to maximize/restore")
        hint_lbl.setStyleSheet("color:#6c7086;font-size:11px;font-style:italic;")
        cbl.addWidget(hint_lbl)

        right_layout.addWidget(ctrl_bar)

        # 2b. Splitter container for the 4 image windows
        self._workspace_area = QWidget()
        self._workspace_layout = QVBoxLayout(self._workspace_area)
        self._workspace_layout.setContentsMargins(0, 0, 0, 0)
        self._workspace_layout.setSpacing(0)
        right_layout.addWidget(self._workspace_area, 1)

        main_layout.addWidget(right_panel, 1)

        # Create the 4 ImageViewers
        self._viewers: list[ImageViewer] = [
            ImageViewer(f"Window {i}", viewer_id=i)
            for i in range(1, 5)
        ]

    # =========================================================================
    # Menus & Actions
    # =========================================================================

    def _setup_menus(self) -> None:
        mb = self.menuBar()

        # ── File Menu ───────────────────────────────────────────────────────
        file_menu = mb.addMenu("&File")

        act_open = QAction("&Open…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(lambda: self._open_image_dialog(self._active_window_id))
        file_menu.addAction(act_open)

        act_save = QAction("&Save…", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(lambda: self._save_image_dialog(self._active_window_id))
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_clear_cur = QAction("&Clear", self)
        act_clear_cur.setShortcut(QKeySequence("Ctrl+W"))
        act_clear_cur.triggered.connect(lambda: self._clear_window(self._active_window_id))
        file_menu.addAction(act_clear_cur)

        act_clear_all = QAction("Clear &All", self)
        act_clear_all.triggered.connect(self._clear_all_windows)
        file_menu.addAction(act_clear_all)

        file_menu.addSeparator()

        act_exit = QAction("E&xit", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # ── Edit Menu ───────────────────────────────────────────────────────
        edit_menu = mb.addMenu("&Edit")

        act_undo = QAction("&Undo", self)
        act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        act_undo.triggered.connect(self._on_undo_requested)
        edit_menu.addAction(act_undo)

        act_reset = QAction("&Reset Window to Source", self)
        act_reset.setShortcut(QKeySequence("Ctrl+R"))
        act_reset.triggered.connect(self._on_reset_to_source)
        edit_menu.addAction(act_reset)

        act_use_out = QAction("Use Output as &New Input", self)
        act_use_out.setShortcut(QKeySequence("Ctrl+U"))
        act_use_out.triggered.connect(self._on_use_output_as_input)
        edit_menu.addAction(act_use_out)

        # ── View Menu ───────────────────────────────────────────────────────
        view_menu = mb.addMenu("&View")

        layout_menu = view_menu.addMenu("&Layout")
        for mode in ("2x2", "1x4", "4x1"):
            act_l = QAction(f"Layout {mode}", self)
            act_l.triggered.connect(lambda _=False, m=mode: self._layout_combo.setCurrentText(m))
            layout_menu.addAction(act_l)

        view_menu.addSeparator()

        act_link = QAction("Link Zoom && Pan", self)
        act_link.setCheckable(True)
        act_link.setChecked(False)
        act_link.toggled.connect(self._chk_sync.setChecked)
        self._chk_sync.toggled.connect(act_link.setChecked)
        view_menu.addAction(act_link)

        act_fit = QAction("&Fit Active Window", self)
        act_fit.setShortcut(QKeySequence("Ctrl+0"))
        act_fit.triggered.connect(lambda: self._fit_window(self._active_window_id))
        view_menu.addAction(act_fit)

        act_fit_all_menu = QAction("Fit All Windows", self)
        act_fit_all_menu.triggered.connect(self._fit_all_windows)
        view_menu.addAction(act_fit_all_menu)

        # ── Process Menu ────────────────────────────────────────────────────
        proc_menu = mb.addMenu("&Process")

        act_apply = QAction("&Apply Selected Feature", self)
        act_apply.setShortcut(QKeySequence("Ctrl+Return"))
        act_apply.triggered.connect(self._sidebar._on_apply)
        proc_menu.addAction(act_apply)

        act_clear_cnt = QAction("Clear Snake &Contour", self)
        act_clear_cnt.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        act_clear_cnt.triggered.connect(self._on_clear_contour)
        proc_menu.addAction(act_clear_cnt)

        # ── Help Menu ───────────────────────────────────────────────────────
        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About ImageLab", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # =========================================================================
    # Status Bar
    # =========================================================================

    def _setup_statusbar(self) -> None:
        sb = self.statusBar()
        sb.setStyleSheet("QStatusBar{background:#11111b;color:#a6adc8;border-top:1px solid #313244;}")

        self._sb_active = QLabel("Active: Window 1")
        self._sb_active.setStyleSheet("color:#89b4fa;font-weight:bold;padding-right:12px;")
        sb.addWidget(self._sb_active)

        self._sb_pos = QLabel("Pos: —")
        self._sb_pos.setStyleSheet("color:#cdd6f4;padding-right:12px;")
        sb.addWidget(self._sb_pos)

        self._sb_val = QLabel("Val: —")
        self._sb_val.setStyleSheet("color:#cdd6f4;padding-right:12px;")
        sb.addWidget(self._sb_val)

        self._sb_zoom = QLabel("Zoom: 100%")
        self._sb_zoom.setStyleSheet("color:#a6adc8;padding-right:12px;")
        sb.addPermanentWidget(self._sb_zoom)

    # =========================================================================
    # Signal Wiring
    # =========================================================================

    def _connect_signals(self) -> None:
        # Sidebar signals
        self._sidebar.apply_requested.connect(self._on_apply_requested)
        self._sidebar.routing_changed.connect(self._update_role_badges)
        self._sidebar.mode_changed.connect(self._sync_all_modes)
        self._sidebar.open_image_requested.connect(self._open_image_dialog)
        self._sidebar.use_output_as_input.connect(self._on_use_output_as_input)
        self._sidebar.reset_to_source.connect(self._on_reset_to_source)
        self._sidebar.clear_all_requested.connect(self._clear_all_windows)
        self._sidebar.undo_requested.connect(self._on_undo_requested)
        self._sidebar.save_output_requested.connect(self._on_save_output_requested)
        self._sidebar.swap_windows_requested.connect(self._on_swap_windows)
        self._sidebar.op_selected.connect(self._on_op_selected)
        self._sidebar.snake_mode_requested.connect(self._on_snake_mode_requested)
        self._sidebar.clear_contour_requested.connect(self._on_clear_contour)

        self._sidebar.animation_play.connect(self._on_anim_play)
        self._sidebar.animation_pause.connect(self._on_anim_pause)
        self._sidebar.animation_step.connect(self._on_anim_step)
        self._sidebar.animation_step_back.connect(self._on_anim_step_back)
        self._sidebar.animation_speed_changed.connect(self._on_anim_speed_changed)
        self._sidebar.cancel_snake_button().clicked.connect(self._on_cancel_snake)
        self._sidebar.undo_contour_point_requested.connect(self._on_undo_contour_point)

        # Overlay style toggles & colors from sidebar
        self._sidebar.overlay_lines_toggled.connect(self._on_overlay_lines_toggled)
        self._sidebar.overlay_circles_toggled.connect(self._on_overlay_circles_toggled)
        self._sidebar.overlay_ellipses_toggled.connect(self._on_overlay_ellipses_toggled)
        self._sidebar.overlay_snake_toggled.connect(self._on_overlay_snake_toggled)
        self._sidebar.overlay_lines_color.connect(self._on_overlay_lines_color)
        self._sidebar.overlay_circles_color.connect(self._on_overlay_circles_color)
        self._sidebar.overlay_ellipses_color.connect(self._on_overlay_ellipses_color)
        self._sidebar.overlay_snake_color.connect(self._on_overlay_snake_color)

        # ImageViewer signals
        for viewer in self._viewers:
            viewer.activated.connect(self._set_active_window)
            viewer.request_open.connect(self._open_image_dialog)
            viewer.request_save.connect(self._save_image_dialog)
            viewer.request_clear.connect(self._clear_window)
            viewer.mode_changed_sig.connect(self._on_window_mode_changed)
            viewer.image_loaded.connect(self._on_image_loaded)
            viewer.file_dropped.connect(self._on_file_dropped)
            viewer.mouse_info.connect(self._on_mouse_info)
            viewer.view_changed.connect(lambda z, h, v, vid=viewer.viewer_id: self._on_view_changed(vid, z, h, v))
            viewer.copy_to.connect(self._on_copy_window)
            viewer.swap_with.connect(self._on_swap_windows)
            viewer.send_result_here.connect(self._on_send_result_here)
            viewer.maximize_requested.connect(self._on_toggle_maximize)

    # =========================================================================
    # Window Layout Management (Splitters)
    # =========================================================================

    def _on_layout_changed(self, mode: str) -> None:
        self._layout_mode = mode
        self._apply_layout(mode)

    def _apply_layout(self, mode: str) -> None:
        """Arrange the 4 ImageViewers inside splitters according to *mode*."""
        # Detach viewers from existing layout/splitters
        for v in self._viewers:
            v.setParent(None)

        # Clear existing splitter widgets from workspace layout
        while self._workspace_layout.count():
            item = self._workspace_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if mode == "2x2":
            v_split = QSplitter(Qt.Orientation.Vertical)
            h_split_top = QSplitter(Qt.Orientation.Horizontal)
            h_split_bottom = QSplitter(Qt.Orientation.Horizontal)

            h_split_top.addWidget(self._viewers[0])
            h_split_top.addWidget(self._viewers[1])
            h_split_bottom.addWidget(self._viewers[2])
            h_split_bottom.addWidget(self._viewers[3])

            v_split.addWidget(h_split_top)
            v_split.addWidget(h_split_bottom)

            h_split_top.setSizes([500, 500])
            h_split_bottom.setSizes([500, 500])
            v_split.setSizes([400, 400])
            self._workspace_layout.addWidget(v_split)

        elif mode == "1x4":
            h_split = QSplitter(Qt.Orientation.Horizontal)
            for v in self._viewers:
                h_split.addWidget(v)
            h_split.setSizes([250, 250, 250, 250])
            self._workspace_layout.addWidget(h_split)

        elif mode == "4x1":
            v_split = QSplitter(Qt.Orientation.Vertical)
            for v in self._viewers:
                v_split.addWidget(v)
            v_split.setSizes([200, 200, 200, 200])
            self._workspace_layout.addWidget(v_split)

        # If a window was maximized, re-apply the maximized state
        if self._maximized_window_id is not None:
            self._set_maximized(self._maximized_window_id)
        else:
            for v in self._viewers:
                v.show()

    def _on_toggle_maximize(self, viewer_id: int) -> None:
        """Double-clicking a window header toggles maximize/restore."""
        if self._maximized_window_id == viewer_id:
            # Restore all windows
            self._maximized_window_id = None
            for v in self._viewers:
                v.show()
        else:
            # Maximize viewer_id
            self._set_maximized(viewer_id)

    def _set_maximized(self, viewer_id: int) -> None:
        self._maximized_window_id = viewer_id
        for v in self._viewers:
            if v.viewer_id == viewer_id:
                v.show()
            else:
                v.hide()

    # =========================================================================
    # Active Window & Role Badges
    # =========================================================================

    def _get_viewer(self, viewer_id: int) -> ImageViewer:
        return self._viewers[viewer_id - 1]

    def _set_active_window(self, viewer_id: int) -> None:
        self._active_window_id = viewer_id
        for v in self._viewers:
            v.set_active(v.viewer_id == viewer_id)

        self._sidebar.set_active_window(viewer_id)
        self._sb_active.setText(f"Active: Window {viewer_id}")

        cur_v = self._get_viewer(viewer_id)
        self._sidebar.update_history(cur_v.history())
        zoom_pct = int(cur_v.display_widget().zoom() * 100)
        self._sb_zoom.setText(f"Zoom: {zoom_pct}%")

        self._update_role_badges()

    def _update_role_badges(self) -> None:
        """Update the INPUT / OUTPUT / BOTH badges across all windows."""
        in_a, in_b, out = self._sidebar.routing_ids()
        # If output == 0, it means "Same as input" (in_a)
        target_out = in_a if out == 0 else out

        entry = self._sidebar._current_entry
        needs_second = (entry is not None and entry.needs_second_image)

        for i in range(1, 5):
            is_in = (i == in_a) or (needs_second and i == in_b)
            is_out = (i == target_out)
            viewer = self._get_viewer(i)

            if is_in and is_out:
                viewer.set_role("BOTH")
            elif is_in:
                viewer.set_role("INPUT")
            elif is_out:
                viewer.set_role("OUTPUT")
            else:
                viewer.set_role("")

    # =========================================================================
    # File Operations (Load, Save, Clear, Mode Change)
    # =========================================================================

    def _open_image_dialog(self, viewer_id: int, mode: Optional[str] = None) -> None:
        """Open a file dialog and load an image into Window *viewer_id*."""
        viewer = self._get_viewer(viewer_id)
        target_mode = mode or self._current_mode

        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Open Image for Window {viewer_id}",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*.*)",
        )
        if not path:
            return

        def _do_load() -> None:
            img_data = ImageData.load(path, as_gray=(target_mode == "Grayscale"))
            viewer.set_source_and_display(img_data, op_name=f"Load ({Path(path).name})")
            self._set_active_window(viewer_id)
            self._sync_all_modes(target_mode)

        run_safely(_do_load, self)

    def _on_file_dropped(self, viewer_id: int, file_path: str) -> None:
        """Handle drag-and-drop of an image file onto a viewer window."""
        viewer = self._get_viewer(viewer_id)

        def _do_drop() -> None:
            img_data = ImageData.load(file_path, as_gray=(self._current_mode == "Grayscale"))
            viewer.set_source_and_display(img_data, op_name=f"Load ({Path(file_path).name})")
            self._set_active_window(viewer_id)
            self._sync_all_modes(self._current_mode)

        run_safely(_do_drop, self)

    def _save_image_dialog(self, viewer_id: int) -> None:
        """Save the displayed image of Window *viewer_id*."""
        viewer = self._get_viewer(viewer_id)
        arr = viewer.current()
        if arr is None:
            show_error(AppError(f"Window {viewer_id} has no image to save."), self)
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save Window {viewer_id} Image",
            f"window_{viewer_id}.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp);;All Files (*.*)",
        )
        if not path:
            return

        def _do_save() -> None:
            ImageData.save(path, arr)

        run_safely(_do_save, self)

    def _clear_window(self, viewer_id: int) -> None:
        self._get_viewer(viewer_id).clear()
        self._sidebar.set_window_has_image(viewer_id, False)
        self._update_role_badges()

    def _clear_all_windows(self) -> None:
        for i in range(1, 5):
            self._clear_window(i)
        self._sidebar.clear_snake_result()
        self._sidebar.update_history([])

    def _on_window_mode_changed(self, viewer_id: int, mode_name: str) -> None:
        self._sync_all_modes(mode_name)

    def _sync_all_modes(self, mode_name: str) -> None:
        """Synchronize the display mode across all windows and the sidebar."""
        if self._syncing_mode:
            return
        self._syncing_mode = True
        try:
            self._current_mode = mode_name
            self._sidebar.set_upload_mode(mode_name)
            for v in self._viewers:
                v.set_mode(mode_name)
        finally:
            self._syncing_mode = False

    def _on_image_loaded(self, viewer_id: int) -> None:
        has = self._get_viewer(viewer_id).has_image()
        self._sidebar.set_window_has_image(viewer_id, has)
        self._sidebar.update_history(self._get_viewer(self._active_window_id).history())

    # =========================================================================
    # Window Actions (Copy, Swap, Send Result Here, Fit)
    # =========================================================================

    def _on_copy_window(self, src_id: int, dst_id: int) -> None:
        src = self._get_viewer(src_id)
        dst = self._get_viewer(dst_id)
        if not src.has_image():
            return
        if src.image_data() is not None:
            dst.set_source_and_display(src.image_data(), op_name=f"Copy from W{src_id}")
        else:
            dst.set_source_and_display(src.current().copy(), op_name=f"Copy from W{src_id}")
        self._set_active_window(dst_id)

    def _on_swap_windows(self, win_a: int, win_b: int) -> None:
        if win_a == win_b:
            return
        va = self._get_viewer(win_a)
        vb = self._get_viewer(win_b)

        # Swap source, current, image_data, history, and undo buffers
        va._source, vb._source = vb._source, va._source
        va._current, vb._current = vb._current, va._current
        va._image_data, vb._image_data = vb._image_data, va._image_data
        va._undo_source, vb._undo_source = vb._undo_source, va._undo_source
        va._undo_current, vb._undo_current = vb._undo_current, va._undo_current
        va._undo_image_data, vb._undo_image_data = vb._undo_image_data, va._undo_image_data
        va._history, vb._history = list(vb._history), list(va._history)

        # Refresh both displays
        if va._current is not None:
            va.apply_mode()
        else:
            va.clear()

        if vb._current is not None:
            vb.apply_mode()
        else:
            vb.clear()

        self._set_active_window(win_b)

    def _on_send_result_here(self, viewer_id: int) -> None:
        self._sidebar.set_output_window(viewer_id)
        self._update_role_badges()

    def _fit_window(self, viewer_id: int) -> None:
        self._get_viewer(viewer_id).display_widget()._fit()

    def _fit_all_windows(self) -> None:
        for v in self._viewers:
            v.display_widget()._fit()

    # =========================================================================
    # Sidebar Action Handlers (Apply, Undo, Reset, Use Output)
    # =========================================================================

    def _on_use_output_as_input(self) -> None:
        _, _, out = self._sidebar.routing_ids()
        in_a, _, _ = self._sidebar.routing_ids()
        target_out = in_a if out == 0 else out
        self._sidebar.set_input_window(target_out)
        self._set_active_window(target_out)

    def _on_reset_to_source(self) -> None:
        viewer = self._get_viewer(self._active_window_id)
        if viewer.source() is None:
            return
        viewer.overlay().clear()
        viewer.set_image(viewer.source().copy())
        viewer._history.append("Reset")
        viewer._update_title()
        self._sidebar.update_history(viewer.history())

    def _on_undo_requested(self) -> None:
        viewer = self._get_viewer(self._active_window_id)
        if viewer.undo():
            self._sidebar.update_history(viewer.history())

    def _on_save_output_requested(self) -> None:
        in_a, _, out = self._sidebar.routing_ids()
        target_out = in_a if out == 0 else out
        self._save_image_dialog(target_out)

    # =========================================================================
    # Operation Dispatch & Execution
    # =========================================================================

    def _on_op_selected(self, op_id: str) -> None:
        self._update_role_badges()

    def _on_snake_mode_requested(self, enabled: bool, polygon_mode: bool) -> None:
        in_a, _, _ = self._sidebar.routing_ids()
        in_viewer = self._get_viewer(in_a)
        # Enable contour input mode on the input window
        in_viewer.set_snake_mode(enabled, polygon=polygon_mode)

    def _on_clear_contour(self) -> None:
        in_a, _, _ = self._sidebar.routing_ids()
        self._get_viewer(in_a).clear_contour()

    def _on_undo_contour_point(self) -> None:
        """Remove the last polygon point from the active snake input viewer."""
        in_a, _, _ = self._sidebar.routing_ids()
        display = self._get_viewer(in_a).display_widget()
        pts = display.get_contour_pts()
        if pts:
            pts.pop()
            display.set_contour_pts(pts)
            display.update()

    def _on_apply_requested(
        self,
        op_id: str,
        params: dict,
        input_a_id: int,
        input_b_id: int,
        output_id: int,
    ) -> None:
        """Route inputs, call run_feature(), and direct outputs."""
        entry = get_entry(op_id)
        if entry is None:
            return

        in_win = self._get_viewer(input_a_id)
        in_arr = in_win.current()
        if in_arr is None:
            show_error(
                AppError(f"Input Window {input_a_id} is empty. Please load an image first."),
                self,
            )
            return

        second_arr: Optional[np.ndarray] = None
        if entry.needs_second_image:
            b_win = self._get_viewer(input_b_id)
            second_arr = b_win.current()
            if second_arr is None:
                show_error(
                    AppError(f"Input B (Window {input_b_id}) is empty."),
                    self,
                )
                return

        target_out_id = input_a_id if output_id == 0 else output_id
        out_win = self._get_viewer(target_out_id)

        # ── Snake algorithm special handling (QThread worker) ───────────────
        if entry.kind == "snake":
            self._start_snake_worker(entry, in_win, in_arr, params, out_win)
            return

        # ── Standard features (Canny, Lines, Circles, Ellipses) ──────────────
        def _do_run() -> None:
            output = run_feature(
                entry,
                in_arr,
                params,
                second=second_arr,
            )
            self._apply_feature_output(entry, in_arr, output, out_win)

        run_safely(_do_run, self)

    def _apply_feature_output(
        self,
        entry: FeatureEntry,
        in_arr: np.ndarray,
        output: FeatureOutput,
        out_win: ImageViewer,
    ) -> None:
        """Handle the FeatureOutput from run_feature."""
        if entry.kind == "image":
            if output.result is not None:
                out_win.overlay().clear()  # <-- CLEAR PREVIOUS OVERLAYS HERE
                out_win.set_source_and_display(output.result, op_name=entry.name)
        elif entry.kind in ("lines", "circles", "ellipses"):
            # Shape detection: copy input image if output is a different window
            if out_win.viewer_id != self._active_window_id or not out_win.has_image():
                out_win.set_source_and_display(in_arr.copy(), op_name=entry.name)
            else:
                out_win._history.append(entry.name)
                out_win._update_title()

            out_win.overlay().clear()  # <-- Also clear previous shapes before drawing new ones

            # Set overlays
            if entry.kind == "lines":
                out_win.overlay().set_lines(output.shapes)
            elif entry.kind == "circles":
                out_win.overlay().set_circles(output.shapes)
            elif entry.kind == "ellipses":
                out_win.overlay().set_ellipses(output.shapes)

            out_win.display_widget().update()
    # =========================================================================
    # Snake QThread Worker & Live Animation
    # =========================================================================

    def _start_snake_worker(
        self,
        entry: FeatureEntry,
        in_win: ImageViewer,
        in_arr: np.ndarray,
        params: dict,
        out_win: ImageViewer,
    ) -> None:
        """Launch the snake worker thread so UI never freezes."""
        # Grab initial contour points from input window
        pts = in_win.get_contour_pts()
        circle_pts = in_win.get_circle_as_polygon()
        is_circle_contour = False

        if not pts and circle_pts:
            # Circle mode: use the user-drawn circle as the initial contour
            pts = circle_pts
            is_circle_contour = True
        elif not pts:
            # No contour drawn at all — show an error
            show_error(
                AppError(
                    "No contour drawn.\n\n"
                    "Please draw an initial contour before running Active Contour:\n"
                    "• Polygon mode: click to place at least 3 points on the image.\n"
                    "• Drag circle mode: click and drag to draw a circle around the object."
                ),
                self,
            )
            return
        else:
            # Polygon contour validation
            if len(pts) < 3:
                show_error(
                    AppError(
                        "Please place at least 3 points on the polygon contour "
                        "before running Active Contour."
                    ),
                    self,
                )
                return
            # Auto-close visually: update display so contour appears closed in UI
            if pts[0] != pts[-1]:
                in_win.display_widget().set_contour_pts(list(pts) + [pts[0]])
                in_win.display_widget().update()
            # Strip duplicate tail before passing to algorithm (pts[0] != pts[-1])

        gray = ensure_gray(in_arr)
        h_img, w_img = gray.shape[:2]
        # Clamp all contour points to valid image pixel coordinates
        pts = [
            (max(0.0, min(float(x), w_img - 1)), max(0.0, min(float(y), h_img - 1)))
            for x, y in pts
        ]
        sp = SnakeParams(
            alpha=float(params.get("alpha", 1.0)),
            beta=float(params.get("beta", 1.0)),
            gamma=float(params.get("gamma", 1.0)),
            max_iter=int(params.get("max_iter", 200)),
        )

        self._anim_target_viewer = out_win
        self._sidebar.show_cancel_snake(True)

        self._snake_worker = SnakeWorker(gray, pts, sp, self)
        self._snake_worker.result_ready.connect(
            lambda res: self._on_snake_worker_done(res, in_arr, out_win)
        )
        self._snake_worker.error_occurred.connect(self._on_snake_worker_error)
        self._snake_worker.start()

    def _on_snake_worker_error(self, exc: Exception) -> None:
        self._sidebar.show_cancel_snake(False)
        self._snake_worker = None
        show_error(exc, self)

    def _on_cancel_snake(self) -> None:
        if self._snake_worker and self._snake_worker.isRunning():
            self._snake_worker.terminate()
            self._snake_worker.wait()
            self._snake_worker = None
        self._sidebar.show_cancel_snake(False)

    def _on_snake_worker_done(
        self,
        result: SnakeResult,
        in_arr: np.ndarray,
        out_win: ImageViewer,
    ) -> None:
        self._sidebar.show_cancel_snake(False)
        self._snake_worker = None

        if out_win.viewer_id != self._active_window_id or not out_win.has_image():
            out_win.set_source_and_display(in_arr.copy(), op_name="Active Contour")
        else:
            out_win._history.append("Active Contour")
            out_win._update_title()

        out_win.overlay().set_snake(result.pts)
        out_win.display_widget().update()

        # Update sidebar results box
        self._sidebar.update_snake_result(result)

        # Prepare animation history
        self._anim_history = list(result.history) if result.history else []
        self._anim_frame_idx = 0

    def _on_anim_play(self) -> None:
        if self._anim_history:
            ms = self._sidebar._speed_slider.value()
            self._anim_timer.start(ms)

    def _on_anim_pause(self) -> None:
        self._anim_timer.stop()

    def _on_anim_step(self) -> None:
        if not self._anim_history or self._anim_target_viewer is None:
            return
        self._anim_frame_idx = (self._anim_frame_idx + 1) % len(self._anim_history)
        pts = self._anim_history[self._anim_frame_idx]
        self._anim_target_viewer.overlay().set_snake(pts)
        self._anim_target_viewer.display_widget().update()

    def _on_anim_step_back(self) -> None:
        if not self._anim_history or self._anim_target_viewer is None:
            return
        self._anim_frame_idx = (self._anim_frame_idx - 1) % len(self._anim_history)
        pts = self._anim_history[self._anim_frame_idx]
        self._anim_target_viewer.overlay().set_snake(pts)
        self._anim_target_viewer.display_widget().update()

    def _on_anim_speed_changed(self, ms: int) -> None:
        if self._anim_timer.isActive():
            self._anim_timer.setInterval(ms)

    def _on_anim_tick(self) -> None:
        self._on_anim_step()

    # =========================================================================
    # Overlay Styling & Toggles
    # =========================================================================

    def _on_overlay_lines_toggled(self, visible: bool) -> None:
        for v in self._viewers:
            v.overlay().show_lines = visible
            v.display_widget().update()

    def _on_overlay_circles_toggled(self, visible: bool) -> None:
        for v in self._viewers:
            v.overlay().show_circles = visible
            v.display_widget().update()

    def _on_overlay_ellipses_toggled(self, visible: bool) -> None:
        for v in self._viewers:
            v.overlay().show_ellipses = visible
            v.display_widget().update()

    def _on_overlay_snake_toggled(self, visible: bool) -> None:
        for v in self._viewers:
            v.overlay().show_snake = visible
            v.display_widget().update()

    def _on_overlay_lines_color(self, color: QColor) -> None:
        for v in self._viewers:
            v.overlay().lines_color = color
            v.display_widget().update()

    def _on_overlay_circles_color(self, color: QColor) -> None:
        for v in self._viewers:
            v.overlay().circles_color = color
            v.display_widget().update()

    def _on_overlay_ellipses_color(self, color: QColor) -> None:
        for v in self._viewers:
            v.overlay().ellipses_color = color
            v.display_widget().update()

    def _on_overlay_snake_color(self, color: QColor) -> None:
        for v in self._viewers:
            v.overlay().snake_color = color
            v.display_widget().update()

    # =========================================================================
    # Synchronised Zoom & Pan
    # =========================================================================

    def _on_view_changed(self, viewer_id: int, zoom: float, h_ratio: float, v_ratio: float) -> None:
        if viewer_id == self._active_window_id:
            zoom_pct = int(zoom * 100)
            self._sb_zoom.setText(f"Zoom: {zoom_pct}%")

        if not self._chk_sync.isChecked() or self._syncing_view:
            return

        self._syncing_view = True
        try:
            for v in self._viewers:
                if v.viewer_id != viewer_id:
                    v.sync_view(zoom, h_ratio, v_ratio)
        finally:
            self._syncing_view = False

    # =========================================================================
    # Mouse Tracking (Status Bar)
    # =========================================================================

    def _on_mouse_info(self, viewer_id: int, x: int, y: int, px_val: object) -> None:
        if x < 0 or y < 0:
            self._sb_pos.setText("Pos: —")
            self._sb_val.setText("Val: —")
            return

        self._sb_pos.setText(f"Pos: ({x}, {y})")
        if px_val is not None:
            if isinstance(px_val, (list, tuple, np.ndarray)):
                val_str = ", ".join(str(int(c)) for c in px_val)
                self._sb_val.setText(f"Val: [{val_str}]")
            else:
                self._sb_val.setText(f"Val: {px_val}")
        else:
            self._sb_val.setText("Val: —")

    # =========================================================================
    # About Dialog
    # =========================================================================

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About ImageLab – Shapes",
            "<h3>ImageLab – Shapes</h3>"
            "<p>A modern GUI shell for edge detection, Hough transforms, "
            "and active contour models.</p>"
            "<p>Built with Python and PySide6.</p>",
        )
