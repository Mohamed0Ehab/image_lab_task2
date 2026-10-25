"""Application entry point for ImageLab – Shapes.

Configures:
- Logging
- Catppuccin Mocha dark Fusion theme & stylesheet
- Global sys.excepthook routing unhandled exceptions to error_handler.show_error
- MainWindow launch
"""
from __future__ import annotations

import logging
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from gui.error_handler import show_error
from gui.main_window import MainWindow

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("imagelab")


# ---------------------------------------------------------------------------
# Global unhandled exception hook
# ---------------------------------------------------------------------------

def _global_excepthook(exc_type, exc_value, exc_traceback) -> None:
    """Intercept uncaught exceptions and present a non-fatal error dialog."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "Uncaught exception: %s: %s",
        exc_type.__name__,
        exc_value,
        exc_info=(exc_type, exc_value, exc_traceback),
    )
    show_error(exc_value)


# ---------------------------------------------------------------------------
# Catppuccin Mocha Dark Theme & Styling
# ---------------------------------------------------------------------------

def apply_catppuccin_theme(app: QApplication) -> None:
    """Apply the Catppuccin Mocha palette and styles using Fusion style."""
    app.setStyle("Fusion")

    # Base Catppuccin Mocha colors
    c_base     = QColor("#1e1e2e")
    c_mantle   = QColor("#181825")
    c_crust    = QColor("#11111b")
    c_surface0 = QColor("#313244")
    c_surface1 = QColor("#45475a")
    c_surface2 = QColor("#585b70")
    c_text     = QColor("#cdd6f4")
    c_subtext  = QColor("#a6adc8")
    c_accent   = QColor("#89b4fa")
    c_red      = QColor("#f38ba8")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          c_mantle)
    palette.setColor(QPalette.ColorRole.WindowText,      c_text)
    palette.setColor(QPalette.ColorRole.Base,            c_crust)
    palette.setColor(QPalette.ColorRole.AlternateBase,   c_base)
    palette.setColor(QPalette.ColorRole.ToolTipBase,     c_base)
    palette.setColor(QPalette.ColorRole.ToolTipText,     c_text)
    palette.setColor(QPalette.ColorRole.Text,            c_text)
    palette.setColor(QPalette.ColorRole.Button,          c_surface0)
    palette.setColor(QPalette.ColorRole.ButtonText,      c_text)
    palette.setColor(QPalette.ColorRole.BrightText,      c_red)
    palette.setColor(QPalette.ColorRole.Highlight,       c_accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, c_crust)

    # Disabled colors
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, c_surface2)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       c_surface2)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, c_surface2)

    app.setPalette(palette)

    # Global stylesheet for polished controls
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #181825;
            color: #cdd6f4;
            font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif;
            font-size: 12px;
        }

        /* ── Splitters ── */
        QSplitter::handle {
            background-color: #313244;
        }
        QSplitter::handle:horizontal {
            width: 4px;
        }
        QSplitter::handle:vertical {
            height: 4px;
        }
        QSplitter::handle:hover {
            background-color: #89b4fa;
        }

        /* ── Push Buttons ── */
        QPushButton {
            background-color: #313244;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 5px;
            padding: 5px 10px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #45475a;
            border-color: #585b70;
        }
        QPushButton:pressed {
            background-color: #585b70;
        }
        QPushButton:disabled {
            background-color: #1e1e2e;
            color: #585b70;
            border-color: #313244;
        }

        /* ── Combo Box ── */
        QComboBox {
            background-color: #313244;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 5px;
            padding: 3px 8px;
        }
        QComboBox:hover {
            border-color: #89b4fa;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 18px;
            border-left: none;
        }
        QComboBox QAbstractItemView {
            background-color: #1e1e2e;
            color: #cdd6f4;
            selection-background-color: #89b4fa;
            selection-color: #11111b;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px;
        }

        /* ── Spin Boxes ── */
        QSpinBox, QDoubleSpinBox {
            background-color: #313244;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 2px 4px;
        }
        QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: #89b4fa;
        }

        /* ── Sliders ── */
        QSlider::groove:horizontal {
            border: none;
            height: 4px;
            background: #313244;
            border-radius: 2px;
        }
        QSlider::sub-page:horizontal {
            background: #89b4fa;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #cdd6f4;
            border: 1px solid #89b4fa;
            width: 12px;
            margin-top: -4px;
            margin-bottom: -4px;
            border-radius: 6px;
        }
        QSlider::handle:horizontal:hover {
            background: #89b4fa;
        }

        /* ── Radio Buttons ── */
        QRadioButton {
            color: #cdd6f4;
            spacing: 8px;
            font-size: 12px;
        }
        QRadioButton::indicator {
            width: 14px;
            height: 14px;
            border-radius: 8px;
            border: 2px solid #6c7086;
            background-color: #1e1e2e;
        }
        QRadioButton::indicator:hover {
            border-color: #cdd6f4;
        }
        QRadioButton::indicator:checked {
            border: 2px solid #89b4fa;
            background-color: #89b4fa;
        }
        QRadioButton::indicator:disabled {
            border-color: #45475a;
            background-color: #181825;
        }

        /* ── Check Boxes ── */
        QCheckBox {
            color: #cdd6f4;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            border: 2px solid #6c7086;
            background-color: #1e1e2e;
        }
        QCheckBox::indicator:hover {
            border-color: #cdd6f4;
        }
        QCheckBox::indicator:checked {
            border: 2px solid #89b4fa;
            background-color: #89b4fa;
        }
        QCheckBox::indicator:disabled {
            border-color: #45475a;
            background-color: #181825;
        }

        /* ── Group Boxes ── */
        QGroupBox {
            border: 1px solid #313244;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 8px;
            font-weight: bold;
            color: #89b4fa;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 4px;
        }

        /* ── Scrollbars ── */
        QScrollBar:vertical {
            border: none;
            background: #181825;
            width: 8px;
            margin: 0px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #313244;
            min-height: 20px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical:hover {
            background: #45475a;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }

        /* ── Menus & Tooltips ── */
        QMenuBar {
            background-color: #11111b;
            color: #cdd6f4;
            border-bottom: 1px solid #313244;
        }
        QMenuBar::item:selected {
            background-color: #313244;
            border-radius: 4px;
        }
        QMenu {
            background-color: #1e1e2e;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 6px;
            padding: 4px;
        }
        QMenu::item {
            padding: 5px 20px 5px 12px;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background-color: #89b4fa;
            color: #11111b;
        }
        QMenu::separator {
            height: 1px;
            background-color: #313244;
            margin: 4px 6px;
        }
        QToolTip {
            background-color: #1e1e2e;
            color: #cdd6f4;
            border: 1px solid #45475a;
            padding: 4px;
            border-radius: 4px;
        }
    """)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    """Run ImageLab – Shapes."""
    sys.excepthook = _global_excepthook

    app = QApplication(sys.argv)
    app.setApplicationName("ImageLab – Shapes")
    apply_catppuccin_theme(app)

    window = MainWindow()
    window.show()

    logger.info("ImageLab – Shapes GUI initialized successfully.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
